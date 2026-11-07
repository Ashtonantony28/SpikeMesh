// SpikeMesh on-chip learning engine (SG6). Contract: docs/SPEC.md §6, with
// the v0.2 changes of §9: trace decay floor max(1, x >> k), rounded reward,
// non-plastic axons (an arrival on a >= plastic_axon_max sets only pre_seen,
// which drives the x_pre bump; pre_flag, which drives depression, stays clear;
// POT draws sr() for them but does not write).
//
// Instantiated by neurocore only when LEARN > 0. Owns the learning state:
// x_pre[A] / x_post[N] traces (uint8, sram_1r1w), pre_flag[A] / post_flag[N]
// (flip-flops), the 16-bit Galois LFSR, and (RSTDP=1 only) the int16
// eligibility memory e[a][n], laid out like the weight memory (row
// a*G + group, LANES lanes). It borrows the weight memory ports while busy.
//
// LEARN(t) phases, in SPEC order (each phase drains its 1-cycle pipeline
// before the next starts, so there are no read-after-write hazards):
//   DECAY  : x_pre/x_post decay (skipped when both shifts are 0)
//   ELIG   : R-STDP only: e decay and reward application, one row per cycle
//   DEP    : axons ascending; flagged rows processed LANES weights per cycle,
//            LFSR stepped LANES times per cycle (lane 0 first = ascending n)
//   POT    : neurons ascending; flagged columns processed one weight per
//            cycle (axons ascending), LFSR stepped once per weight
//   BUMP   : trace bumps for flagged axons/neurons; flags cleared at the end
`default_nettype none

module learn_engine #(
    parameter integer N_NEURONS = 256,
    parameter integer N_AXONS   = 256,
    parameter integer W_BITS    = 8,
    parameter integer LANES     = 8,
    parameter integer RSTDP     = 1
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,          // run LEARN (learn_en already checked)
    output wire               busy,
    // configuration registers (SPEC §5)
    input  wire [15:0]        a_plus,
    input  wire [15:0]        a_minus,
    input  wire [15:0]        stdp_shift,
    input  wire [15:0]        tau_pre,
    input  wire [15:0]        tau_post,
    input  wire [15:0]        inc_pre,
    input  wire [15:0]        inc_post,
    input  wire signed [15:0] w_min,
    input  wire signed [15:0] w_max,
    input  wire               rstdp_en,       // already gated with RSTDP
    input  wire [15:0]        tau_elig,
    input  wire [15:0]        reward_shift,
    input  wire [15:0]        plastic_max,    // reg 23: 0 = all axons plastic
    input  wire               lfsr_load,      // CFG write of lfsr_seed
    input  wire [15:0]        lfsr_seed,
    output reg  [15:0]        lfsr,
    input  wire signed [31:0] reward,
    output reg                reward_clr,
    // spike events
    input  wire               pre_set,        // spike arrival (learn_en)
    input  wire               pre_plastic,    // ... on a plastic axon
    input  wire [$clog2(N_AXONS)-1:0]   pre_axon,
    input  wire               post_wr,
    input  wire [$clog2(N_NEURONS)-1:0] post_n,
    input  wire               post_fire,
    // weight memory (borrowed while busy)
    output reg                w_re,
    output reg  [WAW-1:0]     w_raddr,
    input  wire [LANES*W_BITS-1:0] w_rdata,
    output reg                w_we,
    output reg  [LANES-1:0]   w_wmask,
    output reg  [WAW-1:0]     w_waddr,
    output reg  [LANES*W_BITS-1:0] w_wdata,
    // debug access (only while !busy): sel 0 elig[idx], 1 x_pre[idx],
    // 2 x_post[idx], 3 pre_flag[idx], 4 post_flag[idx]; read data valid the
    // cycle after dbg_re.
    input  wire               dbg_we,
    input  wire               dbg_re,
    input  wire [2:0]         dbg_sel,
    input  wire [19:0]        dbg_idx,
    input  wire [15:0]        dbg_wdata,
    output reg  [15:0]        dbg_rdata
);
    localparam integer NB    = $clog2(N_NEURONS);
    localparam integer AB    = $clog2(N_AXONS);
    localparam integer LB0   = $clog2(LANES);
    localparam integer LBW   = (LB0 > 0) ? LB0 : 1;
    localparam integer G     = N_NEURONS / LANES;
    localparam integer GB0   = $clog2(G);
    localparam integer GAW   = (G > 1) ? GB0 : 1;
    localparam integer WDEPTH = N_AXONS * G;
    localparam integer WAW   = (WDEPTH > 1) ? $clog2(WDEPTH) : 1;
    localparam integer XROW  = LANES * 8;
    localparam integer EROW  = LANES * 16;
    localparam integer WROW  = LANES * W_BITS;
    localparam integer TMAX  = (N_AXONS > G) ? N_AXONS : G;       // trace pass length
    localparam integer CW    = $clog2(WDEPTH + TMAX) + 1;           // pass counter width
    localparam integer N_WTS = N_AXONS * N_NEURONS;

    localparam [2:0] P_IDLE = 3'd0, P_DECAY = 3'd1, P_ELIG = 3'd2, P_DEP = 3'd3,
                     P_POT = 3'd4, P_BUMP = 3'd5;

    // ------------------------------------------------------------------
    // Galois LFSR, taps 0xB400 (SPEC §6)
    // ------------------------------------------------------------------
    function automatic [15:0] lfsr_next(input [15:0] s);
        lfsr_next = s[0] ? ((s >> 1) ^ 16'hB400) : (s >> 1);
    endfunction

    // stochastic rounding: sr(m) with s = stdp_shift. m < 2^24, so s >= 24
    // behaves exactly like s = 24 (m >> s = 0, mask covers all of m).
    function automatic [24:0] sr(input [23:0] m, input [4:0] s, input [15:0] rnd);
        reg [23:0] mask, q;
        begin
            mask = (24'd1 << s) - 24'd1;
            if (s == 5'd24) mask = 24'hFFFFFF;
            q    = (s == 5'd24) ? 24'd0 : (m >> s);
            sr   = {1'b0, q} + {24'd0, (({8'd0, rnd} & mask) < (m & mask))};
        end
    endfunction

    // clamp(v, lo, hi) = min(max(v, lo), hi), as numpy/golden do
    function automatic [15:0] clamp27(input signed [26:0] v, input signed [15:0] lo, input signed [15:0] hi);
        reg signed [26:0] t;
        begin
            t = (v < 27'(lo)) ? 27'(lo) : v;
            t = (t > 27'(hi)) ? 27'(hi) : t;
            clamp27 = t[15:0];
        end
    endfunction

    function automatic [15:0] clamp50(input signed [49:0] v, input signed [15:0] lo, input signed [15:0] hi);
        reg signed [49:0] t;
        begin
            t = (v < 50'(lo)) ? 50'(lo) : v;
            t = (t > 50'(hi)) ? 50'(hi) : t;
            clamp50 = t[15:0];
        end
    endfunction

    // SPEC §9.1 trace decay: x -= max(1, x >> k) when x > 0 and k > 0
    function automatic [7:0] decay8(input [7:0] x, input [3:0] k);
        reg [7:0] d;
        begin
            d = x >> k;
            decay8 = (k == 4'd0 || x == 8'd0) ? x : (x - ((d == 8'd0) ? 8'd1 : d));
        end
    endfunction

    wire [4:0] s_eff     = (stdp_shift > 16'd24) ? 5'd24 : stdp_shift[4:0];
    wire [3:0] tpre_eff  = (tau_pre  > 16'd8) ? 4'd8 : tau_pre[3:0];
    wire [3:0] tpost_eff = (tau_post > 16'd8) ? 4'd8 : tau_post[3:0];
    wire [3:0] telig_eff = (tau_elig > 16'd15) ? 4'd15 : tau_elig[3:0];
    // |r * e| < 2^47, so any shift >= 48 gives 0 even with the rounding term
    wire [5:0] rs_eff    = (reward_shift > 16'd48) ? 6'd48 : reward_shift[5:0];
    wire signed [49:0] rs_half = (rs_eff == 6'd0) ? 50'sd0 : (50'sd1 <<< (rs_eff - 6'd1));

    // ------------------------------------------------------------------
    // Flags
    // ------------------------------------------------------------------
    reg [N_AXONS-1:0]   pre_flag;
    reg [N_AXONS-1:0]   pre_seen;   // any arrival (plastic or not): x_pre bump
    reg [N_NEURONS-1:0] post_flag;

    // ------------------------------------------------------------------
    // Memories
    // ------------------------------------------------------------------
    reg              xp_we, xp_re;   reg [AB-1:0] xp_waddr, xp_raddr; reg [7:0] xp_wdata; wire [7:0] xp_rdata;
    reg              xq_we, xq_re;   reg [GAW-1:0] xq_waddr, xq_raddr; reg [LANES-1:0] xq_wmask;
    reg [XROW-1:0]   xq_wdata;       wire [XROW-1:0] xq_rdata;
    reg              e_we, e_re;     reg [WAW-1:0] e_waddr, e_raddr; reg [LANES-1:0] e_wmask;
    reg [EROW-1:0]   e_wdata;        wire [EROW-1:0] e_rdata;

    sram_1r1w #(.DEPTH(N_AXONS), .WIDTH(8), .NLANE(1)) u_xpre (
        .clk(clk), .we(xp_we), .wmask(1'b1), .waddr(xp_waddr), .wdata(xp_wdata),
        .re(xp_re), .raddr(xp_raddr), .rdata(xp_rdata));
    sram_1r1w #(.DEPTH(G), .WIDTH(XROW), .NLANE(LANES)) u_xpost (
        .clk(clk), .we(xq_we), .wmask(xq_wmask), .waddr(xq_waddr), .wdata(xq_wdata),
        .re(xq_re), .raddr(xq_raddr), .rdata(xq_rdata));
    generate
        if (RSTDP != 0) begin : g_elig
            sram_1r1w #(.DEPTH(WDEPTH), .WIDTH(EROW), .NLANE(LANES)) u_elig (
                .clk(clk), .we(e_we), .wmask(e_wmask), .waddr(e_waddr), .wdata(e_wdata),
                .re(e_re), .raddr(e_raddr), .rdata(e_rdata));
        end else begin : g_noelig
            assign e_rdata = '0;
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_e = e_we | e_re | (|e_waddr) | (|e_raddr) | (|e_wmask) | (|e_wdata);
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    reg [2:0]     ph;
    reg           issue_done;
    reg [CW-1:0]  cnt;          // DECAY/BUMP index, ELIG row
    reg [AB:0]    d_a;          // DEP axon pointer (== N_AXONS: done)
    reg [GAW-1:0] d_g;
    reg [NB:0]    p_n;          // POT neuron pointer
    reg [AB-1:0]  p_a;
    reg           in_run;
    reg           apply_rw;     // reward applied in this LEARN
    reg signed [31:0] r_snap;
    // compute stage
    reg           k_valid;
    reg [2:0]     k_ph;
    reg [CW-1:0]  k_i;
    reg [WAW-1:0] k_row;
    reg [LBW-1:0] k_lane;
    reg           k_fixed;      // POT: axon k is non-plastic (sr drawn, no write)

    wire use_e = rstdp_en;
    assign busy = (ph != P_IDLE) || start;

    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] d_a32  = {{(31-AB){1'b0}}, d_a};
    wire [31:0] p_n32  = {{(31-NB){1'b0}}, p_n};
    wire [31:0] cnt32  = {{(32-CW){1'b0}}, cnt};
    wire [31:0] dep_row32 = (d_a32 << GB0) | {{(32-GAW){1'b0}}, d_g};
    wire [31:0] pot_g32   = p_n32 >> LB0;
    wire [31:0] pot_row32 = ({{(32-AB){1'b0}}, p_a} << GB0) | pot_g32;
    wire [31:0] pot_l32   = p_n32 & 32'(LANES - 1);
    wire [31:0] k_i32     = {{(32-CW){1'b0}}, k_i};
    /* verilator lint_on UNUSEDSIGNAL */
    wire        d_flag  = (d_a32 < 32'(N_AXONS)) ? pre_flag[d_a[AB-1:0] ] : 1'b0;
    wire        p_flag  = (p_n32 < 32'(N_NEURONS)) ? post_flag[p_n[NB-1:0]] : 1'b0;

    // issue-side decisions (combinational)
    reg issue;
    always @* begin
        issue = 1'b0;
        case (ph)
            P_DECAY, P_BUMP: issue = !issue_done;
            P_ELIG:          issue = !issue_done;
            P_DEP:           issue = !issue_done && in_run;
            P_POT:           issue = !issue_done && in_run;
            default:         issue = 1'b0;
        endcase
    end

    // ------------------------------------------------------------------
    // Compute stage (lane-parallel)
    // ------------------------------------------------------------------
    reg [15:0] chain [0:LANES];
    integer ci;
    always @* begin
        chain[0] = lfsr;
        for (ci = 0; ci < LANES; ci = ci + 1) chain[ci+1] = lfsr_next(chain[ci]);
    end

    wire is_dep = (k_ph == P_DEP);
    wire [15:0] coef = is_dep ? a_minus : a_plus;
    wire signed [15:0] lo = use_e ? -16'sd32768 : w_min;
    wire signed [15:0] hi = use_e ?  16'sd32767 : w_max;

    reg [EROW-1:0] stdp_res;       // per lane 16-bit result (weight or elig)
    reg [EROW-1:0] elig_res;       // ELIG pass: decayed e
    reg [EROW-1:0] rw_res;         // ELIG pass: rewarded w (16-bit per lane)
    reg [XROW-1:0] xq_res;
    reg [7:0]      xp_res;
    integer l;
    reg [7:0]  xl;
    reg [23:0] m;
    reg [24:0] dlt;
    reg signed [15:0] tgt, ev, ed;
    reg signed [47:0] prod;
    reg signed [49:0] dw;
    reg [8:0] xb;
    always @* begin
        xl = 8'd0; m = 24'd0; dlt = 25'd0; tgt = 16'sd0; ev = 16'sd0; ed = 16'sd0;
        prod = 48'sd0; dw = 50'sd0; xb = 9'd0; xp_res = 8'd0;
        stdp_res = '0; elig_res = '0; rw_res = '0; xq_res = '0;
        for (l = 0; l < LANES; l = l + 1) begin
            // STDP (DEP: all lanes; POT: lane k_lane is the one written)
            xl  = is_dep ? xq_rdata[l*8 +: 8] : xp_rdata;
            m   = coef * xl;
            dlt = sr(m, s_eff, is_dep ? chain[l+1] : chain[1]);
            tgt = use_e ? $signed(e_rdata[l*16 +: 16])
                        : $signed({{(16-W_BITS){w_rdata[l*W_BITS + W_BITS - 1]}}, w_rdata[l*W_BITS +: W_BITS]});
            stdp_res[l*16 +: 16] = is_dep ? clamp27(27'(tgt) - $signed({2'b00, dlt}), lo, hi)
                                          : clamp27(27'(tgt) + $signed({2'b00, dlt}), lo, hi);
            // ELIG pass: e -= e >>> tau_elig; w = clamp(w + ((r*e + half) >>> reward_shift))
            ev = $signed(e_rdata[l*16 +: 16]);
            ed = (tau_elig == 16'd0) ? ev : (ev - (ev >>> telig_eff));
            elig_res[l*16 +: 16] = ed;
            prod = r_snap * ed;
            dw   = ($signed({{2{prod[47]}}, prod}) + rs_half) >>> rs_eff;
            rw_res[l*16 +: 16] = clamp50(dw + 50'($signed({{(16-W_BITS){w_rdata[l*W_BITS + W_BITS - 1]}},
                                                                w_rdata[l*W_BITS +: W_BITS]})), w_min, w_max);
            // x_post decay / bump
            xl = xq_rdata[l*8 +: 8];
            if (k_ph == P_DECAY)
                xq_res[l*8 +: 8] = decay8(xl, (tau_post == 16'd0) ? 4'd0 : tpost_eff);
            else begin
                xb = {1'b0, xl} + ((inc_post > 16'd255) ? 9'd255 : inc_post[8:0]);
                xq_res[l*8 +: 8] = (post_flag[(k_i32 * LANES + l) % N_NEURONS] && (k_i32 < 32'(G)))
                                   ? ((xb > 9'd255) ? 8'd255 : xb[7:0]) : xl;
            end
        end
        if (k_ph == P_DECAY)
            xp_res = decay8(xp_rdata, (tau_pre == 16'd0) ? 4'd0 : tpre_eff);
        else begin
            xb = {1'b0, xp_rdata} + ((inc_pre > 16'd255) ? 9'd255 : inc_pre[8:0]);
            xp_res = (pre_seen[k_i32 % N_AXONS] && (k_i32 < 32'(N_AXONS)))
                     ? ((xb > 9'd255) ? 8'd255 : xb[7:0]) : xp_rdata;
        end
    end

    // pack a 16-bit-per-lane row into W_BITS-per-lane weights
    reg [WROW-1:0] stdp_w, rw_w;
    integer pl;
    always @* begin
        for (pl = 0; pl < LANES; pl = pl + 1) begin
            stdp_w[pl*W_BITS +: W_BITS] = stdp_res[pl*16 +: W_BITS];
            rw_w[pl*W_BITS +: W_BITS]   = rw_res[pl*16 +: W_BITS];
        end
    end

    // ------------------------------------------------------------------
    // Debug decode
    // ------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0]    di32    = {12'd0, dbg_idx};
    wire [31:0]    di_row  = di32 >> LB0;
    wire [31:0]    di_l32  = di32 & 32'(LANES - 1);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [LBW-1:0] di_lane = di_l32[LBW-1:0];
    wire [LANES-1:0] di_mask = {{(LANES-1){1'b0}}, 1'b1} << di_lane;
    wire di_e_ok  = (di32 < 32'(N_WTS));
    wire di_a_ok  = (di32 < 32'(N_AXONS));
    wire di_n_ok  = (di32 < 32'(N_NEURONS));
    reg [2:0]     dq_sel;
    reg           dq_ok;
    reg [LBW-1:0] dq_lane;
    reg           dq_flag;

    always @* begin
        dbg_rdata = 16'd0;
        if (dq_ok) case (dq_sel)
            3'd0: dbg_rdata = e_rdata[dq_lane*16 +: 16];
            3'd1: dbg_rdata = {8'd0, xp_rdata};
            3'd2: dbg_rdata = {8'd0, xq_rdata[dq_lane*8 +: 8]};
            3'd3, 3'd4: dbg_rdata = {15'd0, dq_flag};
            default: dbg_rdata = 16'd0;
        endcase
    end

    // ------------------------------------------------------------------
    // Memory port muxes
    // ------------------------------------------------------------------
    always @* begin
        // weight memory
        w_re    = issue && ((ph == P_ELIG) || (((ph == P_DEP) || (ph == P_POT)) && !use_e));
        w_raddr = (ph == P_ELIG) ? cnt[WAW-1:0] : (ph == P_DEP) ? dep_row32[WAW-1:0] : pot_row32[WAW-1:0];
        w_we    = k_valid && (((k_ph == P_ELIG) && apply_rw) || (k_ph == P_DEP && !use_e) ||
                              ((k_ph == P_POT) && !use_e && !k_fixed));
        w_wmask = (k_ph == P_POT) ? ({{(LANES-1){1'b0}}, 1'b1} << k_lane) : {LANES{1'b1}};
        w_waddr = k_row;
        w_wdata = (k_ph == P_ELIG) ? rw_w : stdp_w;
        // eligibility memory
        e_re    = (issue && ((ph == P_ELIG) || (((ph == P_DEP) || (ph == P_POT)) && use_e))) ||
                  (dbg_re && dbg_sel == 3'd0);
        e_raddr = issue ? w_raddr : di_row[WAW-1:0];
        e_we    = (k_valid && ((k_ph == P_ELIG) || ((k_ph == P_DEP) && use_e) ||
                               ((k_ph == P_POT) && use_e && !k_fixed))) ||
                  (dbg_we && dbg_sel == 3'd0 && di_e_ok);
        e_wmask = k_valid ? w_wmask : di_mask;
        e_waddr = k_valid ? k_row : di_row[WAW-1:0];
        e_wdata = k_valid ? ((k_ph == P_ELIG) ? elig_res : stdp_res) : {LANES{dbg_wdata}};
        // x_pre
        xp_re    = (issue && ((ph == P_DECAY) || (ph == P_BUMP) || (ph == P_POT))) || (dbg_re && dbg_sel == 3'd1);
        xp_raddr = issue ? ((ph == P_POT) ? p_a : cnt[AB-1:0]) : dbg_idx[AB-1:0];
        xp_we    = (k_valid && ((k_ph == P_DECAY) || (k_ph == P_BUMP)) && (k_i32 < 32'(N_AXONS))) ||
                   (dbg_we && dbg_sel == 3'd1 && di_a_ok);
        xp_waddr = k_valid ? k_i[AB-1:0] : dbg_idx[AB-1:0];
        xp_wdata = k_valid ? xp_res : dbg_wdata[7:0];
        // x_post
        xq_re    = (issue && ((ph == P_DECAY) || (ph == P_BUMP) || (ph == P_DEP))) || (dbg_re && dbg_sel == 3'd2);
        xq_raddr = issue ? ((ph == P_DEP) ? d_g : cnt[GAW-1:0]) : di_row[GAW-1:0];
        xq_we    = (k_valid && ((k_ph == P_DECAY) || (k_ph == P_BUMP)) && (k_i32 < 32'(G))) ||
                   (dbg_we && dbg_sel == 3'd2 && di_n_ok);
        xq_wmask = k_valid ? {LANES{1'b1}} : di_mask;
        xq_waddr = k_valid ? k_i[GAW-1:0] : di_row[GAW-1:0];
        xq_wdata = k_valid ? xq_res : {LANES{dbg_wdata[7:0]}};
    end

    // ------------------------------------------------------------------
    // Sequencer
    // ------------------------------------------------------------------
    wire trace_skip = (tau_pre == 16'd0) && (tau_post == 16'd0);
    wire elig_needed = rstdp_en && ((tau_elig != 16'd0) || (reward != 32'sd0));
    wire drained = issue_done && !k_valid && !issue;

    always @(posedge clk) begin
        if (!rst_n) begin
            ph <= P_IDLE; issue_done <= 1'b0; cnt <= '0; d_a <= '0; d_g <= '0;
            p_n <= '0; p_a <= '0; in_run <= 1'b0; apply_rw <= 1'b0; r_snap <= 32'sd0;
            k_valid <= 1'b0; k_ph <= P_IDLE; k_i <= '0; k_row <= '0; k_lane <= '0; k_fixed <= 1'b0;
            lfsr <= 16'd1; reward_clr <= 1'b0;
            pre_flag <= '0; pre_seen <= '0; post_flag <= '0;
            dq_sel <= 3'd0; dq_ok <= 1'b0; dq_lane <= '0; dq_flag <= 1'b0;
        end else begin
            reward_clr <= 1'b0;
            // flags from the spike / update datapaths
            if (pre_set) begin
                pre_seen[pre_axon] <= 1'b1;
                if (pre_plastic) pre_flag[pre_axon] <= 1'b1;
            end
            if (post_wr) post_flag[post_n] <= post_fire;
            // debug read capture
            if (dbg_re) begin
                dq_sel  <= dbg_sel;
                dq_lane <= di_lane;
                dq_ok   <= (dbg_sel == 3'd0) ? di_e_ok : (dbg_sel == 3'd1 || dbg_sel == 3'd3) ? di_a_ok : di_n_ok;
                dq_flag <= (dbg_sel == 3'd3) ? pre_flag[dbg_idx[AB-1:0]] : post_flag[dbg_idx[NB-1:0]];
            end
            // LFSR
            if (lfsr_load) lfsr <= (lfsr_seed == 16'd0) ? 16'd1 : lfsr_seed;
            else if (k_valid && (k_ph == P_DEP)) lfsr <= chain[LANES];
            else if (k_valid && (k_ph == P_POT)) lfsr <= chain[1];

            // compute-stage registers
            k_valid <= issue;
            k_ph    <= ph;
            k_i     <= cnt;
            k_row   <= (ph == P_ELIG) ? cnt[WAW-1:0] : (ph == P_DEP) ? dep_row32[WAW-1:0] : pot_row32[WAW-1:0];
            k_lane  <= pot_l32[LBW-1:0];
            k_fixed <= (plastic_max != 16'd0) && ({{(16-AB){1'b0}}, p_a} >= plastic_max);

            case (ph)
                P_IDLE: if (start) begin
                    issue_done <= 1'b0; cnt <= '0;
                    r_snap   <= reward;
                    apply_rw <= rstdp_en && (reward != 32'sd0);
                    if (!trace_skip) ph <= P_DECAY;
                    else if (elig_needed) ph <= P_ELIG;
                    else begin ph <= P_DEP; d_a <= '0; in_run <= 1'b0; end
                end
                P_DECAY, P_BUMP, P_ELIG: begin
                    if (issue) begin
                        cnt <= cnt + 1'b1;
                        if (cnt32 == ((ph == P_ELIG) ? 32'(WDEPTH - 1) : 32'(TMAX - 1))) issue_done <= 1'b1;
                    end
                    if (drained) begin
                        issue_done <= 1'b0; cnt <= '0;
                        if (ph == P_DECAY && elig_needed) ph <= P_ELIG;
                        else if (ph == P_BUMP) begin
                            ph <= P_IDLE; pre_flag <= '0; pre_seen <= '0; post_flag <= '0;
                        end else begin
                            if (ph == P_ELIG && apply_rw) reward_clr <= 1'b1;
                            ph <= P_DEP; d_a <= '0; in_run <= 1'b0;
                        end
                    end
                end
                P_DEP: begin
                    if (!issue_done) begin
                        if (!in_run) begin
                            if (d_a32 == 32'(N_AXONS)) issue_done <= 1'b1;
                            else if (d_flag) begin in_run <= 1'b1; d_g <= '0; end
                            else d_a <= d_a + 1'b1;
                        end else begin
                            d_g <= d_g + 1'b1;
                            if ({{(32-GAW){1'b0}}, d_g} == 32'(G - 1)) begin
                                in_run <= 1'b0; d_a <= d_a + 1'b1;
                            end
                        end
                    end else if (drained) begin
                        issue_done <= 1'b0; ph <= P_POT; p_n <= '0; in_run <= 1'b0;
                    end
                end
                P_POT: begin
                    if (!issue_done) begin
                        if (!in_run) begin
                            if (p_n32 == 32'(N_NEURONS)) issue_done <= 1'b1;
                            else if (p_flag) begin in_run <= 1'b1; p_a <= '0; end
                            else p_n <= p_n + 1'b1;
                        end else begin
                            p_a <= p_a + 1'b1;
                            if ({{(32-AB){1'b0}}, p_a} == 32'(N_AXONS - 1)) begin
                                in_run <= 1'b0; p_n <= p_n + 1'b1;
                            end
                        end
                    end else if (drained) begin
                        issue_done <= 1'b0; cnt <= '0; ph <= P_BUMP;
                    end
                end
                default: ph <= P_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
