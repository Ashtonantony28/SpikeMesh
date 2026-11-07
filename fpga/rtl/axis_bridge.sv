// axis_bridge: Zynq PS path to the SpikeMesh host port.
//
//  * s_axis (64-bit TDATA, from AXI DMA MM2S): one 48-bit word per beat in
//    TDATA[47:0]. Words are flits or link commands (flit_link.sv). TLAST is
//    ignored.
//  * m_axis (64-bit TDATA, to AXI DMA S2MM): chip flits and link events,
//    zero-extended. TLAST is set on event words (STEP_DONE / FLUSH_DONE /
//    STATUS / RESET_DONE), so a simple-mode S2MM transfer ends at the event.
//  * s_axi (AXI4-Lite, 32-bit). Register map:
//      0x00 CTRL    W: [0] STEP, [1] FLUSH, [3] SOFT_RESET (self-clearing pulses)
//                   R/W: [8] HOST_SEL (0 = AXI path, 1 = UART path; kv260_top)
//      0x04 STATUS  R: [0] done (sticky, set when STEP_DONE is emitted), [1] link busy,
//                      [2] t_parity, [3] err (sticky: chip err or bad command),
//                      [4] chip running, [5] uart rx overflow, [6] uart frame error
//                   W: write 1 to [0] / [3] to clear them
//      0x08 STEP_COUNT       0x0C LAST_STEP_CYCLES (step -> step_done)
//      0x10 LAST_DRAIN_CYCLES  0x14 BUSY_CYCLES (cumulative, chip running)
//      0x18 CYCLES (free running)  0x1C IN_FLITS  0x20 OUT_FLITS
//      0x24 DROP_CNT [7:0]
//      0x28 ID = 0x534D_0107 ("SM", SG7 v1)
//      0x2C CONFIG = {8'd0, log2(N_AXONS)[3:0], log2(N_NEURONS)[3:0], W_BITS[3:0],
//                     LANES log2 [3:0], MESH_Y[3:0], MESH_X[3:0]}
`default_nettype none

module axis_bridge #(
    parameter integer MESH_X       = 2,
    parameter integer MESH_Y       = 2,
    parameter integer N_NEURONS    = 256,
    parameter integer N_AXONS      = 256,
    parameter integer W_BITS       = 8,
    parameter integer LANES        = 8,
    parameter integer DRAIN_CYCLES = 0,
    parameter integer HOST_SEL_RST = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    // AXI-Stream in
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tlast,
    /* verilator lint_on UNUSEDSIGNAL */
    // AXI-Stream out
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [63:0] m_axis_tdata,
    output wire [7:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
    // AXI4-Lite slave
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [7:0]  s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [3:0]  s_axi_wstrb,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [7:0]  s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    // chip host port
    output wire        chip_rst_n,
    output wire        chip_in_valid,
    input  wire        chip_in_ready,
    output wire [47:0] chip_in_flit,
    input  wire        chip_out_valid,
    output wire        chip_out_ready,
    input  wire [47:0] chip_out_flit,
    output wire        chip_step,
    input  wire        chip_step_done,
    input  wire        chip_t_parity,
    input  wire        chip_running,
    input  wire        chip_err,
    input  wire [7:0]  chip_drop_cnt,
    input  wire        chip_drained,     // spikemesh_top drained (DRAIN_CYCLES = 0)
    // misc
    input  wire        uart_rx_overflow,
    input  wire        uart_frame_err,
    output reg         host_sel
);
    reg  reg_step, reg_flush, reg_reset;
    reg  done_sticky, err_sticky;
    wire link_busy, done_pulse, cmd_err;
    wire [31:0] step_count, last_step_cycles, last_drain_cycles, busy_cycles, cycle_count, in_flits, out_flits;
    wire [47:0] o_data;

    flit_link #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .DRAIN_CYCLES(DRAIN_CYCLES)) u_link (
        .clk(clk), .rst_n(rst_n),
        .i_valid(s_axis_tvalid), .i_ready(s_axis_tready), .i_data(s_axis_tdata[47:0]),
        .reg_step(reg_step), .reg_flush(reg_flush), .reg_reset(reg_reset),
        .o_valid(m_axis_tvalid), .o_ready(m_axis_tready), .o_data(o_data), .o_last(m_axis_tlast),
        .chip_rst_n(chip_rst_n),
        .chip_in_valid(chip_in_valid), .chip_in_ready(chip_in_ready), .chip_in_flit(chip_in_flit),
        .chip_out_valid(chip_out_valid), .chip_out_ready(chip_out_ready), .chip_out_flit(chip_out_flit),
        .chip_step(chip_step), .chip_step_done(chip_step_done), .chip_t_parity(chip_t_parity),
        .chip_running(chip_running), .chip_err(chip_err), .chip_drop_cnt(chip_drop_cnt), .chip_drained(chip_drained),
        .link_busy(link_busy), .done_pulse(done_pulse), .cmd_err(cmd_err),
        .step_count(step_count), .last_step_cycles(last_step_cycles),
        .last_drain_cycles(last_drain_cycles), .busy_cycles(busy_cycles),
        .cycle_count(cycle_count), .in_flits(in_flits), .out_flits(out_flits));

    assign m_axis_tdata = {16'd0, o_data};
    assign m_axis_tkeep = 8'hFF;
    assign s_axi_bresp  = 2'b00;
    assign s_axi_rresp  = 2'b00;

    localparam [31:0] CONFIG = {8'd0, 4'($clog2(N_AXONS)), 4'($clog2(N_NEURONS)), 4'(W_BITS),
                                4'($clog2(LANES)), 4'(MESH_Y), 4'(MESH_X)};

    // ---------------- AXI-Lite write ----------------
    /* verilator lint_off UNUSEDSIGNAL */   // only word-aligned, defined bits are decoded
    reg [7:0]  aw_addr;
    reg [31:0] w_data;
    /* verilator lint_on UNUSEDSIGNAL */
    reg        aw_have, w_have;
    always @(posedge clk) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
            aw_have <= 1'b0; w_have <= 1'b0; aw_addr <= 8'd0; w_data <= 32'd0;
            reg_step <= 1'b0; reg_flush <= 1'b0; reg_reset <= 1'b0;
            host_sel <= HOST_SEL_RST[0]; done_sticky <= 1'b0; err_sticky <= 1'b0;
        end else begin
            reg_step <= 1'b0; reg_flush <= 1'b0; reg_reset <= 1'b0;
            if (done_pulse) done_sticky <= 1'b1;
            if (cmd_err || chip_err) err_sticky <= 1'b1;
            s_axi_awready <= !aw_have && !s_axi_awready && s_axi_awvalid && !s_axi_bvalid;
            s_axi_wready  <= !w_have && !s_axi_wready && s_axi_wvalid && !s_axi_bvalid;
            if (s_axi_awvalid && s_axi_awready) begin aw_have <= 1'b1; aw_addr <= s_axi_awaddr; end
            if (s_axi_wvalid && s_axi_wready) begin w_have <= 1'b1; w_data <= s_axi_wdata; end
            if (aw_have && w_have && !s_axi_bvalid) begin
                aw_have <= 1'b0; w_have <= 1'b0; s_axi_bvalid <= 1'b1;
                case (aw_addr[7:2])
                    6'h00: begin
                        reg_step  <= w_data[0];
                        reg_flush <= w_data[1];
                        reg_reset <= w_data[3];
                        host_sel  <= w_data[8];
                    end
                    6'h01: begin
                        if (w_data[0]) done_sticky <= 1'b0;
                        if (w_data[3]) err_sticky <= 1'b0;
                    end
                    default: ;
                endcase
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end

    // ---------------- AXI-Lite read ----------------
    always @(posedge clk) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; s_axi_rdata <= 32'd0;
        end else begin
            s_axi_arready <= !s_axi_arready && s_axi_arvalid && !s_axi_rvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[7:2])
                    6'h00: s_axi_rdata <= {23'd0, host_sel, 8'd0};
                    6'h01: s_axi_rdata <= {25'd0, uart_frame_err, uart_rx_overflow, chip_running,
                                           err_sticky, chip_t_parity, link_busy, done_sticky};
                    6'h02: s_axi_rdata <= step_count;
                    6'h03: s_axi_rdata <= last_step_cycles;
                    6'h04: s_axi_rdata <= last_drain_cycles;
                    6'h05: s_axi_rdata <= busy_cycles;
                    6'h06: s_axi_rdata <= cycle_count;
                    6'h07: s_axi_rdata <= in_flits;
                    6'h08: s_axi_rdata <= out_flits;
                    6'h09: s_axi_rdata <= {24'd0, chip_drop_cnt};
                    6'h0A: s_axi_rdata <= 32'h534D_0107;
                    6'h0B: s_axi_rdata <= CONFIG;
                    default: s_axi_rdata <= 32'd0;
                endcase
            end
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        end
    end
endmodule

`default_nettype wire
