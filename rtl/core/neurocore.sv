// SpikeMesh neurocore (SG2: inference). Contract: docs/SPEC.md §2-§5, §8.
//
// See rtl/core/README.md for the microarchitecture. Summary:
//   * Input engine: one flit at a time. SPIKE = read one weight row
//     (LANES weights per cycle) + the matching accumulator group, add, write
//     back (2-stage pipeline, G = N_NEURONS/LANES cycles per spike).
//     CFG_WRITE = 1 cycle. READ_REQ = 3 cycles + output-FIFO space.
//   * Update engine: on update_start walks every neuron (1 neuron/cycle,
//     2-stage pipeline), writes v/refc, clears bank `parity`, and pushes each
//     firing neuron with a non-empty fan-out into a small spike FIFO.
//   * Emitter: pops the spike FIFO, reads fan-out entries (1/cycle), forms
//     SPIKE flits tagged parity^1 into a 4-entry output FIFO (backpressure safe).
//   * Spikes for the *other* bank are accepted while UPDATE runs (needed so a
//     mesh where cores feed each other cannot deadlock), and so are CTRL
//     flits. CFG/READ flits and spikes for the bank being updated wait until
//     UPDATE ends (head-of-line: the host must not send CFG/READ during STEP).
`default_nettype none

module neurocore #(
    parameter integer X              = 0,
    parameter integer Y              = 0,
    parameter integer N_NEURONS      = 256,
    parameter integer N_AXONS        = 256,
    parameter integer W_BITS         = 8,
    parameter integer FANOUT_DEPTH   = 1024,
    parameter integer LANES          = 8,
    // SPEC §2: W_BITS + log2(N_AXONS) + 1. Exact as long as no axon gets more
    // than two spikes in one timestep; override for heavier fan-in.
    parameter integer ACC_W          = W_BITS + $clog2(N_AXONS) + 1
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    /* verilator lint_off UNUSEDSIGNAL */  // dest_x/dest_y were consumed by the router
    input  wire [47:0] in_flit,
    /* verilator lint_on UNUSEDSIGNAL */
    output wire        out_valid,
    input  wire        out_ready,
    output wire [47:0] out_flit,
    input  wire        parity,
    input  wire        learn_start,
    input  wire        update_start,
    output wire        busy
);
    // ------------------------------------------------------------------
    // Derived sizes
    // ------------------------------------------------------------------
    localparam integer NB     = (N_NEURONS > 1) ? $clog2(N_NEURONS) : 1;
    localparam integer AB     = (N_AXONS > 1) ? $clog2(N_AXONS) : 1;
    localparam integer LB0    = $clog2(LANES);            // may be 0
    localparam integer LBW    = (LB0 > 0) ? LB0 : 1;
    localparam integer G      = N_NEURONS / LANES;        // groups per row
    localparam integer GB0    = $clog2(G);                // may be 0
    localparam integer GAW    = (G > 1) ? GB0 : 1;
    localparam integer WDEPTH = N_AXONS * G;
    localparam integer WAW    = (WDEPTH > 1) ? $clog2(WDEPTH) : 1;
    localparam integer FAW    = (FANOUT_DEPTH > 1) ? $clog2(FANOUT_DEPTH) : 1;
    localparam integer WROW   = LANES * W_BITS;
    localparam integer AROW   = LANES * ACC_W;
    localparam integer N_WTS  = N_AXONS * N_NEURONS;

    localparam [GAW-1:0] G_LAST = GAW'(G - 1);
    localparam [NB:0]    N_LAST = (NB + 1)'(N_NEURONS - 1);
    localparam [3:0]     XC     = 4'(X);
    localparam [3:0]     YC     = 4'(Y);

    // Flit types (SPEC §4)
    localparam [1:0] T_SPIKE = 2'd0, T_CFG = 2'd1, T_READ = 2'd2, T_CTRL = 2'd3;

    // Input FSM states
    localparam [1:0] S_IDLE = 2'd0, S_SPK = 2'd1, S_RDW = 2'd2, S_RDP = 2'd3;

    localparam integer OF_DEPTH = 4;   // output FIFO
    localparam integer SF_DEPTH = 4;   // firing-neuron FIFO

    // ------------------------------------------------------------------
    // Reset-time accumulator clear (SPEC §5: acc banks are control state)
    // ------------------------------------------------------------------
    reg           init_busy;
    reg [GAW-1:0] clr_g;
    always @(posedge clk) begin
        if (!rst_n) begin
            init_busy <= 1'b1;
            clr_g     <= '0;
        end else if (init_busy) begin
            if (clr_g == G_LAST) init_busy <= 1'b0;
            clr_g <= clr_g + 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // Core registers (SPEC §5). Spec widths for §2 registers; learning
    // registers (widths not specified) are stored as 16 bits.
    // ------------------------------------------------------------------
    reg signed [15:0] r_theta;
    reg        [3:0]  r_leak;
    reg               r_reset_mode;
    reg        [3:0]  r_refrac;
    reg signed [15:0] r_vmin;
    reg        [15:0] r_lreg [0:13];   // registers 8..21
    wire              learn_en = (r_lreg[0] != 16'd0);

    // 32-bit index temporaries below are sliced; their upper bits are unused.
    /* verilator lint_off UNUSEDSIGNAL */
    // ------------------------------------------------------------------
    // Input flit decode
    // ------------------------------------------------------------------
    wire [1:0]  in_type  = in_flit[47:46];
    wire        in_par   = in_flit[37];
    wire [AB-1:0] in_axon = in_flit[AB-1:0];
    wire        is_spike = (in_type == T_SPIKE);
    wire        is_cfg   = (in_type == T_CFG);
    wire        is_rdreq = (in_type == T_READ) && !in_flit[37];
    wire        is_ctrl  = (in_type == T_CTRL);

    wire [21:0] dec_addr = is_cfg ? in_flit[37:16] : in_flit[36:15];
    wire [15:0] cfg_data = in_flit[15:0];
    wire [1:0]  d_region = dec_addr[21:20];
    wire [31:0] d_low    = {12'd0, dec_addr[19:0]};

    // weights: [19:0] = axon*N + neuron = (row address) * LANES + lane
    wire          d_w_ok   = (d_low < 32'(N_WTS));
    wire [31:0]   d_w_a32  = d_low >> LB0;
    wire [WAW-1:0] d_w_addr = d_w_a32[WAW-1:0];
    wire [31:0]   d_w_l32  = d_low & 32'(LANES - 1);
    wire [LBW-1:0] d_w_lane = d_w_l32[LBW-1:0];
    // neuron fields
    wire [31:0]   d_n32    = d_low >> 2;
    wire          d_n_ok   = (d_n32 < 32'(N_NEURONS));
    wire [NB-1:0] d_n      = d_n32[NB-1:0];
    wire [1:0]    d_field  = dec_addr[1:0];
    // fan-out entries
    wire [31:0]   d_f32    = d_low >> 1;
    wire          d_f_ok   = (d_f32 < 32'(FANOUT_DEPTH));
    wire [FAW-1:0] d_f     = d_f32[FAW-1:0];
    wire          d_word   = dec_addr[0];
    wire [7:0]    d_reg    = dec_addr[7:0];

    /* verilator lint_on UNUSEDSIGNAL */
    // core register read mux (combinational, indexed by the incoming flit)
    reg [15:0] d_regval;
    always @* begin
        d_regval = 16'd0;
        case (d_reg)
            8'd0: d_regval = r_theta;
            8'd1: d_regval = {12'd0, r_leak};
            8'd2: d_regval = {15'd0, r_reset_mode};
            8'd3: d_regval = {12'd0, r_refrac};
            8'd4: d_regval = r_vmin;
            default:
                if (d_reg >= 8'd8 && d_reg <= 8'd21) d_regval = r_lreg[4'(d_reg - 8'd8)];
        endcase
    end

    // ------------------------------------------------------------------
    // Memory port signals
    // ------------------------------------------------------------------
    // weights
    reg              w_we;  reg [LANES-1:0] w_wmask; reg [WAW-1:0] w_waddr; reg [WROW-1:0] w_wdata;
    reg              w_re;  reg [WAW-1:0] w_raddr;   wire [WROW-1:0] w_rdata;
    // accumulator banks
    reg              a_we [0:1]; reg [GAW-1:0] a_waddr [0:1]; reg [AROW-1:0] a_wdata [0:1];
    reg              a_re [0:1]; reg [GAW-1:0] a_raddr [0:1]; wire [AROW-1:0] a_rdata [0:1];
    // neuron state: v, bias, refc, fan-out pointer
    reg              n_re;  reg [NB-1:0] n_raddr;
    reg              v_we, b_we, rc_we, p_we;
    reg [NB-1:0]     v_waddr, b_waddr, rc_waddr, p_waddr;
    reg [15:0]       v_wdata, b_wdata, p_wdata; reg [3:0] rc_wdata;
    wire [15:0]      v_rdata, b_rdata, p_rdata; wire [3:0] rc_rdata;
    // fan-out entries: dest {x,y} and axon
    reg              f_re;  reg [FAW-1:0] f_raddr;
    reg              fd_we, fa_we; reg [FAW-1:0] f_waddr;
    reg [7:0]        fd_wdata; reg [9:0] fa_wdata;
    wire [7:0]       fd_rdata; wire [9:0] fa_rdata;

    sram_1r1w #(.DEPTH(WDEPTH), .WIDTH(WROW), .NLANE(LANES)) u_wmem (
        .clk(clk), .we(w_we), .wmask(w_wmask), .waddr(w_waddr), .wdata(w_wdata),
        .re(w_re), .raddr(w_raddr), .rdata(w_rdata));

    genvar gb;
    generate
        for (gb = 0; gb < 2; gb = gb + 1) begin : g_acc
            sram_1r1w #(.DEPTH(G), .WIDTH(AROW), .NLANE(1)) u_acc (
                .clk(clk), .we(a_we[gb]), .wmask(1'b1), .waddr(a_waddr[gb]), .wdata(a_wdata[gb]),
                .re(a_re[gb]), .raddr(a_raddr[gb]), .rdata(a_rdata[gb]));
        end
    endgenerate

    sram_1r1w #(.DEPTH(N_NEURONS), .WIDTH(16), .NLANE(1)) u_v (
        .clk(clk), .we(v_we), .wmask(1'b1), .waddr(v_waddr), .wdata(v_wdata),
        .re(n_re), .raddr(n_raddr), .rdata(v_rdata));
    sram_1r1w #(.DEPTH(N_NEURONS), .WIDTH(16), .NLANE(1)) u_bias (
        .clk(clk), .we(b_we), .wmask(1'b1), .waddr(b_waddr), .wdata(b_wdata),
        .re(n_re), .raddr(n_raddr), .rdata(b_rdata));
    sram_1r1w #(.DEPTH(N_NEURONS), .WIDTH(4), .NLANE(1)) u_refc (
        .clk(clk), .we(rc_we), .wmask(1'b1), .waddr(rc_waddr), .wdata(rc_wdata),
        .re(n_re), .raddr(n_raddr), .rdata(rc_rdata));
    sram_1r1w #(.DEPTH(N_NEURONS), .WIDTH(16), .NLANE(1)) u_ptr (
        .clk(clk), .we(p_we), .wmask(1'b1), .waddr(p_waddr), .wdata(p_wdata),
        .re(n_re), .raddr(n_raddr), .rdata(p_rdata));
    sram_1r1w #(.DEPTH(FANOUT_DEPTH), .WIDTH(8), .NLANE(1)) u_fodest (
        .clk(clk), .we(fd_we), .wmask(1'b1), .waddr(f_waddr), .wdata(fd_wdata),
        .re(f_re), .raddr(f_raddr), .rdata(fd_rdata));
    sram_1r1w #(.DEPTH(FANOUT_DEPTH), .WIDTH(10), .NLANE(1)) u_foaxon (
        .clk(clk), .we(fa_we), .wmask(1'b1), .waddr(f_waddr), .wdata(fa_wdata),
        .re(f_re), .raddr(f_raddr), .rdata(fa_rdata));

    // ------------------------------------------------------------------
    // Control state
    // ------------------------------------------------------------------
    reg [1:0]     st;
    // spike accumulate, issue side
    reg [AB-1:0]  spk_axon;
    reg           spk_bank;
    reg [GAW-1:0] spk_g;
    // spike accumulate, write-back stage
    reg           s2_valid, s2_bank, s2_fwd;
    reg [GAW-1:0] s2_g;
    reg [AROW-1:0] s2_fwd_data;
    // read request
    reg [1:0]     rd_region, rd_field;
    reg           rd_ok, rd_word;
    reg [LBW-1:0] rd_lane;
    reg [15:0]    rd_addr16, rd_regval;
    reg [3:0]     rd_rx, rd_ry;
    reg [47:0]    rd_resp;
    // learning placeholder / reward
    reg           learn_busy;
    reg signed [31:0] reward_acc;  // summed CTRL REWARD values (SG6 consumes)
    // update
    reg           upd_pending, upd_active, upd_par;
    reg [NB:0]    r_n;
    reg           r_done;
    reg           c_valid;
    reg [NB-1:0]  c_n;
    // spike FIFO {neuron, base, count}
    localparam integer SFW = NB + 16;
    reg [SFW-1:0] sf_mem [0:SF_DEPTH-1];
    reg [1:0]     sf_wp, sf_rp;
    reg [2:0]     sf_cnt;
    // emitter
    reg           e_active;
    reg [NB-1:0]  e_n;
    reg [9:0]     e_cur;
    reg [5:0]     e_rem;
    reg           e_d_valid;
    reg [NB-1:0]  e_d_n;
    // output FIFO
    reg [47:0]    of_mem [0:OF_DEPTH-1];
    reg [1:0]     of_wp, of_rp;
    reg [2:0]     of_cnt;

    // ------------------------------------------------------------------
    // Input handshake
    // ------------------------------------------------------------------
    wire upd_hold = upd_pending | upd_active;
    assign in_ready = !init_busy && (st == S_IDLE) && !learn_busy &&
                      (!upd_hold || is_ctrl || (is_spike && (in_par != upd_par)));
    wire accept  = in_valid && in_ready;
    wire acc_spk = accept && is_spike;
    wire acc_cfg = accept && is_cfg;
    wire acc_rd  = accept && is_rdreq;

    // spike row issue
    wire           spk_issue      = acc_spk || (st == S_SPK);
    wire           spk_issue_bank = (st == S_SPK) ? spk_bank : in_par;
    wire [AB-1:0]  spk_issue_axon = (st == S_SPK) ? spk_axon : in_axon;
    wire [GAW-1:0] spk_issue_g    = (st == S_SPK) ? spk_g : '0;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]    spk_wa32       = ({{(32-AB){1'b0}}, spk_issue_axon} << GB0) |
                                    {{(32-GAW){1'b0}}, spk_issue_g};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [WAW-1:0] spk_waddr      = spk_wa32[WAW-1:0];

    // write-back stage: acc + sign-extended weights, lane-parallel
    wire [AROW-1:0] s2_old = s2_fwd ? s2_fwd_data : a_rdata[s2_bank];
    reg  [AROW-1:0] s2_sum;
    integer li;
    always @* begin
        for (li = 0; li < LANES; li = li + 1)
            s2_sum[li*ACC_W +: ACC_W] = s2_old[li*ACC_W +: ACC_W] +
                {{(ACC_W-W_BITS){w_rdata[li*W_BITS + W_BITS - 1]}}, w_rdata[li*W_BITS +: W_BITS]};
    end

    // ------------------------------------------------------------------
    // Update engine: stage R (issue) / stage C (compute + write back)
    // ------------------------------------------------------------------
    wire upd_can_start = upd_pending && !upd_active && !learn_busy &&
                         ((st == S_IDLE) || ((st == S_SPK) && (spk_bank != upd_par))) &&
                         !(s2_valid && (s2_bank == upd_par));
    wire u_issue = upd_active && !r_done && (sf_cnt <= 3'(SF_DEPTH - 2));
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]    r_n32   = {{(31-NB){1'b0}}, r_n};
    wire [31:0]    r_g32   = r_n32 >> LB0;
    wire           r_lane0 = ((r_n32 & 32'(LANES - 1)) == 32'd0);
    wire [31:0]    c_n32   = {{(32-NB){1'b0}}, c_n};
    wire [31:0]    c_g32   = c_n32 >> LB0;
    wire [31:0]    c_l32   = c_n32 & 32'(LANES - 1);
    wire [LBW-1:0] c_lane  = c_l32[LBW-1:0];
    wire           c_lane0 = (c_l32 == 32'd0);
    /* verilator lint_on UNUSEDSIGNAL */

    // neuron arithmetic (SPEC §2)
    wire [AROW-1:0]         c_accrow = a_rdata[upd_par];
    wire signed [ACC_W-1:0] c_acc    = c_accrow[c_lane*ACC_W +: ACC_W];
    wire signed [31:0]      c_acc32  = $signed({{(32-ACC_W){c_acc[ACC_W-1]}}, c_acc});
    wire [15:0]             c_acc_lo = c_acc32[15:0];
    wire signed [15:0]      c_acc16  = (c_acc32 > 32'sd32767)  ? 16'sh7fff :
                                       (c_acc32 < -32'sd32768) ? 16'sh8000 : c_acc_lo;
    wire signed [15:0]      cv       = v_rdata;
    wire signed [15:0]      cb       = b_rdata;
    wire signed [15:0]      c_leak   = (r_leak == 4'd0) ? 16'sd0 : (cv >>> r_leak);
    wire signed [19:0]      c_sum    = {{4{cv[15]}}, cv} - {{4{c_leak[15]}}, c_leak} +
                                       {{4{c_acc16[15]}}, c_acc16} + {{4{cb[15]}}, cb};
    wire signed [15:0]      c_v2     = (c_sum > 20'sd32767)  ? 16'sh7fff :
                                       (c_sum < -20'sd32768) ? 16'sh8000 : c_sum[15:0];
    wire                    c_refr   = (rc_rdata != 4'd0);
    wire                    c_fire   = !c_refr && (c_v2 >= r_theta);
    wire signed [15:0]      c_vreset = r_reset_mode ? 16'sd0 : (c_v2 - r_theta);
    wire signed [15:0]      c_vkeep  = (c_v2 > r_vmin) ? c_v2 : r_vmin;
    wire [15:0]             c_vnew   = c_refr ? cv : (c_fire ? c_vreset : c_vkeep);
    wire [3:0]              c_rcnew  = c_refr ? (rc_rdata - 4'd1) : (c_fire ? r_refrac : 4'd0);
    wire [5:0]              c_cnt    = p_rdata[15:10];
    wire                    sf_push  = c_valid && c_fire && (c_cnt != 6'd0);

    // Learning hooks for SG6 (pre/post spike events). Not consumed yet.
    /* verilator lint_off UNUSEDSIGNAL */
    wire           hook_pre_valid   = acc_spk && learn_en;
    wire [AB-1:0]  hook_pre_axon    = in_axon;
    wire           hook_post_valid  = c_valid && c_fire && learn_en;
    wire [NB-1:0]  hook_post_neuron = c_n;
    /* verilator lint_on UNUSEDSIGNAL */

    // emitter
    wire       sf_empty = (sf_cnt == 3'd0);
    wire       sf_pop   = !e_active && !sf_empty;
    wire [SFW-1:0] sf_head = sf_mem[sf_rp];
    wire       e_credit = ({1'b0, of_cnt} + {3'd0, e_d_valid} + 4'd1) <= 4'(OF_DEPTH);
    // A popped entry issues its first fan-out read in the pop cycle.
    wire       e_issue  = (e_active || sf_pop) && e_credit;
    wire [9:0] e_addr   = e_active ? e_cur : sf_head[15:6];
    wire [5:0] e_left   = e_active ? e_rem : sf_head[5:0];
    wire [3:0] e_dx     = fd_rdata[7:4];
    wire [3:0] e_dy     = fd_rdata[3:0];
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] e_n32   = {{(32-NB){1'b0}}, e_d_n};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [9:0] e_nid    = e_n32[9:0];
    wire [47:0] e_flit  = (e_dx == 4'hF) ?
        {T_SPIKE, 4'hF, 4'h0, ~upd_par, 7'd0, XC, YC, 12'd0, e_nid} :
        {T_SPIKE, e_dx, e_dy, ~upd_par, 27'd0, fa_rdata};

    // output FIFO
    wire of_full  = (of_cnt == 3'(OF_DEPTH));
    wire of_push  = e_d_valid || ((st == S_RDP) && !of_full);
    wire [47:0] of_wdata = e_d_valid ? e_flit : rd_resp;
    assign out_valid = (of_cnt != 3'd0);
    assign out_flit  = of_mem[of_rp];
    wire of_pop = out_valid && out_ready;

    wire upd_finish = upd_active && r_done && !c_valid && sf_empty && !e_active && !e_d_valid;

    assign busy = init_busy | (st != S_IDLE) | s2_valid | learn_busy | learn_start |
                  update_start | upd_pending | upd_active | out_valid;

    // ------------------------------------------------------------------
    // Memory port muxes
    // ------------------------------------------------------------------
    integer bi;
    always @* begin
        // weights: CFG write; read for spike rows or READ_REQ
        w_we    = acc_cfg && (d_region == 2'd0) && d_w_ok;
        w_wmask = {{(LANES-1){1'b0}}, 1'b1} << d_w_lane;
        w_waddr = d_w_addr;
        w_wdata = {LANES{cfg_data[W_BITS-1:0]}};
        w_re    = spk_issue || (acc_rd && (d_region == 2'd0));
        w_raddr = spk_issue ? spk_waddr : d_w_addr;

        // accumulator banks
        for (bi = 0; bi < 2; bi = bi + 1) begin
            a_we[bi]    = 1'b0;
            a_waddr[bi] = s2_g;
            a_wdata[bi] = s2_sum;
            if (init_busy) begin
                a_we[bi] = 1'b1; a_waddr[bi] = clr_g; a_wdata[bi] = '0;
            end else if (s2_valid && (s2_bank == bi[0])) begin
                a_we[bi] = 1'b1;
            end else if (c_valid && c_lane0 && (upd_par == bi[0])) begin
                a_we[bi] = 1'b1; a_waddr[bi] = c_g32[GAW-1:0]; a_wdata[bi] = '0;
            end
            a_re[bi]    = 1'b0;
            a_raddr[bi] = spk_issue_g;
            if (spk_issue && (spk_issue_bank == bi[0])) begin
                a_re[bi] = 1'b1;
            end else if (u_issue && r_lane0 && (upd_par == bi[0])) begin
                a_re[bi] = 1'b1; a_raddr[bi] = r_g32[GAW-1:0];
            end
        end

        // neuron state
        n_re    = u_issue || (acc_rd && (d_region == 2'd1));
        n_raddr = upd_active ? r_n[NB-1:0] : d_n;
        v_we  = c_valid || (acc_cfg && (d_region == 2'd1) && d_n_ok && (d_field == 2'd0));
        b_we  = (acc_cfg && (d_region == 2'd1) && d_n_ok && (d_field == 2'd1));
        rc_we = c_valid || (acc_cfg && (d_region == 2'd1) && d_n_ok && (d_field == 2'd2));
        p_we  = (acc_cfg && (d_region == 2'd1) && d_n_ok && (d_field == 2'd3));
        v_waddr  = c_valid ? c_n : d_n;
        rc_waddr = c_valid ? c_n : d_n;
        b_waddr  = d_n;
        p_waddr  = d_n;
        v_wdata  = c_valid ? c_vnew : cfg_data;
        rc_wdata = c_valid ? c_rcnew : cfg_data[3:0];
        b_wdata  = cfg_data;
        p_wdata  = cfg_data;

        // fan-out entries
        f_re     = e_issue || (acc_rd && (d_region == 2'd2));
        f_raddr  = e_issue ? e_addr[FAW-1:0] : d_f;
        fd_we    = acc_cfg && (d_region == 2'd2) && d_f_ok && !d_word;
        fa_we    = acc_cfg && (d_region == 2'd2) && d_f_ok && d_word;
        f_waddr  = d_f;
        fd_wdata = cfg_data[15:8];
        fa_wdata = cfg_data[9:0];
    end

    // ------------------------------------------------------------------
    // Read-response data (S_RDW)
    // ------------------------------------------------------------------
    reg [15:0] rd_data;
    wire [W_BITS-1:0] rd_w = w_rdata[rd_lane*W_BITS +: W_BITS];
    always @* begin
        rd_data = 16'd0;
        case (rd_region)
            2'd0: if (rd_ok) rd_data = {{(16-W_BITS){rd_w[W_BITS-1]}}, rd_w};
            2'd1: if (rd_ok) case (rd_field)
                      2'd0: rd_data = v_rdata;
                      2'd1: rd_data = b_rdata;
                      2'd2: rd_data = {12'd0, rc_rdata};
                      default: rd_data = p_rdata;
                  endcase
            2'd2: if (rd_ok) rd_data = rd_word ? {6'd0, fa_rdata} : {fd_rdata, 8'd0};
            default: rd_data = rd_regval;
        endcase
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            st          <= S_IDLE;
            spk_axon    <= '0;
            spk_bank    <= 1'b0;
            spk_g       <= '0;
            s2_valid    <= 1'b0;
            s2_bank     <= 1'b0;
            s2_g        <= '0;
            s2_fwd      <= 1'b0;
            learn_busy  <= 1'b0;
            reward_acc  <= 32'sd0;
            upd_pending <= 1'b0;
            upd_active  <= 1'b0;
            upd_par     <= 1'b0;
            r_n         <= '0;
            r_done      <= 1'b0;
            c_valid     <= 1'b0;
            c_n         <= '0;
            sf_wp <= 2'd0; sf_rp <= 2'd0; sf_cnt <= 3'd0;
            e_active    <= 1'b0;
            e_n <= '0; e_cur <= 10'd0; e_rem <= 6'd0;
            e_d_valid   <= 1'b0;
            e_d_n       <= '0;
            of_wp <= 2'd0; of_rp <= 2'd0; of_cnt <= 3'd0;
            r_theta      <= 16'sd64;
            r_leak       <= 4'd0;
            r_reset_mode <= 1'b0;
            r_refrac     <= 4'd0;
            r_vmin       <= -16'sd32768;
            for (k = 0; k < 14; k = k + 1) r_lreg[k] <= 16'd0;
            r_lreg[16-8] <= 16'd1;                                        // lfsr_seed
            r_lreg[17-8] <= 16'(-(1 << (W_BITS - 1)));                    // w_min
            r_lreg[18-8] <= 16'((1 << (W_BITS - 1)) - 1);                 // w_max
        end else begin
            // ---------------- input engine ----------------
            case (st)
                S_IDLE: begin
                    if (acc_spk) begin
                        spk_axon <= in_axon;
                        spk_bank <= in_par;
                        spk_g    <= (G > 1) ? GAW'(1) : '0;
                        if (G > 1) st <= S_SPK;
                    end
                    if (acc_rd) begin
                        rd_region <= d_region;
                        rd_field  <= d_field;
                        rd_word   <= d_word;
                        rd_lane   <= d_w_lane;
                        rd_ok     <= (d_region == 2'd0) ? d_w_ok :
                                     (d_region == 2'd1) ? d_n_ok :
                                     (d_region == 2'd2) ? d_f_ok : 1'b1;
                        rd_addr16 <= dec_addr[15:0];
                        rd_regval <= d_regval;
                        rd_rx     <= in_flit[14:11];
                        rd_ry     <= in_flit[10:7];
                        st        <= S_RDW;
                    end
                    if (acc_cfg && (d_region == 2'd3)) begin
                        case (d_reg)
                            8'd0: r_theta      <= cfg_data;
                            8'd1: r_leak       <= cfg_data[3:0];
                            8'd2: r_reset_mode <= cfg_data[0];
                            8'd3: r_refrac     <= cfg_data[3:0];
                            8'd4: r_vmin       <= cfg_data;
                            default:
                                if (d_reg >= 8'd8 && d_reg <= 8'd21)
                                    r_lreg[4'(d_reg - 8'd8)] <= cfg_data;
                        endcase
                    end
                    if (accept && is_ctrl && (in_flit[7:0] == 8'h01))
                        reward_acc <= reward_acc + {{16{in_flit[23]}}, in_flit[23:8]};
                end
                S_SPK: begin
                    spk_g <= spk_g + 1'b1;
                    if (spk_g == G_LAST) st <= S_IDLE;
                end
                S_RDW: begin
                    rd_resp <= {T_READ, rd_rx, rd_ry, 1'b1, 5'd0, rd_addr16, rd_data};
                    st      <= S_RDP;
                end
                default: begin   // S_RDP
                    if (!of_full) st <= S_IDLE;
                end
            endcase

            // spike write-back stage (+ forwarding when the next row reads
            // the group being written this cycle; only possible when G == 1)
            s2_valid    <= spk_issue;
            s2_bank     <= spk_issue_bank;
            s2_g        <= spk_issue_g;
            s2_fwd      <= spk_issue && s2_valid && (s2_bank == spk_issue_bank) && (s2_g == spk_issue_g);
            s2_fwd_data <= s2_sum;

            // ---------------- learn (placeholder, SG6) ----------------
            // learn_en = 0: nothing to do. learn_en = 1: the SG6 engine hooks
            // in here; for now LEARN is a 1-cycle no-op.
            learn_busy <= learn_start;

            // ---------------- update engine ----------------
            if (update_start && !upd_hold) begin
                upd_pending <= 1'b1;
                upd_par     <= parity;
            end
            if (upd_can_start) begin
                upd_pending <= 1'b0;
                upd_active  <= 1'b1;
                r_n         <= '0;
                r_done      <= 1'b0;
            end
            if (u_issue) begin
                r_n <= r_n + 1'b1;
                if (r_n == N_LAST) r_done <= 1'b1;
            end
            c_valid <= u_issue;
            c_n     <= r_n[NB-1:0];
            if (upd_finish) upd_active <= 1'b0;

            // spike FIFO
            if (sf_push) begin
                sf_mem[sf_wp] <= {c_n, p_rdata[9:0], c_cnt};
                sf_wp <= sf_wp + 2'd1;
            end
            if (sf_pop) sf_rp <= sf_rp + 2'd1;
            sf_cnt <= sf_cnt + {2'd0, sf_push} - {2'd0, sf_pop};

            // emitter
            if (sf_pop) e_n <= sf_head[SFW-1:16];
            if (e_issue) begin
                e_cur    <= e_addr + 10'd1;
                e_rem    <= e_left - 6'd1;
                e_active <= (e_left != 6'd1);
            end else if (sf_pop) begin
                e_cur    <= e_addr;
                e_rem    <= e_left;
                e_active <= 1'b1;
            end
            e_d_valid <= e_issue;
            e_d_n     <= e_active ? e_n : sf_head[SFW-1:16];

            // output FIFO
            if (of_push) begin
                of_mem[of_wp] <= of_wdata;
                of_wp <= of_wp + 2'd1;
            end
            if (of_pop) of_rp <= of_rp + 2'd1;
            of_cnt <= of_cnt + {2'd0, of_push} - {2'd0, of_pop};
        end
    end
endmodule

`default_nettype wire
