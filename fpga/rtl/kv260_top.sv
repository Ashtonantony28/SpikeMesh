// kv260_top: SpikeMesh chip + host bridges for the Kria KV260.
//
// spikemesh_top (SPEC §8) gets its host port from one of two bridges:
//   * axis_bridge: AXI DMA streams + AXI-Lite registers (Zynq PS / PYNQ path)
//   * uart_bridge: 8N1 UART on two PL pins (PMOD + USB-UART adapter)
// The AXI-Lite register CTRL[8] (HOST_SEL) selects the bridge. Its reset value
// is HOST_SEL_RST. Both bridges use the same word protocol (fpga/rtl/flit_link.sv).
`default_nettype none

module kv260_top #(
    parameter integer MESH_X       = 2,
    parameter integer MESH_Y       = 2,
    parameter integer N_NEURONS    = 256,
    parameter integer N_AXONS      = 256,
    parameter integer W_BITS       = 8,
    parameter integer FANOUT_DEPTH = 1024,
    parameter integer LANES        = 8,
    parameter integer CLK_HZ       = 100_000_000,
    parameter integer BAUD         = 921_600,
    parameter integer DRAIN_CYCLES = 64,
    parameter integer HOST_SEL_RST = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    // AXI-Stream from DMA MM2S
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tlast,
    // AXI-Stream to DMA S2MM
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [63:0] m_axis_tdata,
    output wire [7:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
    // AXI4-Lite control
    input  wire [7:0]  s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [7:0]  s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,
    // UART
    input  wire        uart_rx,
    output wire        uart_tx,
    // status (LED / ILA friendly)
    output wire        t_parity,
    output wire        chip_err
);
    // chip
    wire        c_rst_n;
    wire        c_in_valid, c_in_ready, c_out_valid, c_out_ready, c_step, c_step_done, c_running;
    wire [47:0] c_in_flit, c_out_flit;
    wire [7:0]  c_drop;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        c_drained;   // spikemesh_top quiet flag; the link uses its own drain timer
    /* verilator lint_on UNUSEDSIGNAL */

    spikemesh_top #(
        .MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS),
        .W_BITS(W_BITS), .FANOUT_DEPTH(FANOUT_DEPTH), .LANES(LANES)
    ) u_chip (
        .clk(clk), .rst_n(c_rst_n),
        .host_in_valid(c_in_valid), .host_in_ready(c_in_ready), .host_in_flit(c_in_flit),
        .host_out_valid(c_out_valid), .host_out_ready(c_out_ready), .host_out_flit(c_out_flit),
        .step(c_step), .step_done(c_step_done), .t_parity(t_parity),
        .err(chip_err), .drop_cnt(c_drop), .busy(c_running), .drained(c_drained)
    );

    // bridges
    wire        host_sel;
    wire        a_rst_n, a_in_valid, a_out_ready, a_step;
    wire        u_rst_n, u_in_valid, u_out_ready, u_step;
    wire [47:0] a_in_flit, u_in_flit;
    wire        u_ovf, u_ferr;
    /* verilator lint_off UNUSEDSIGNAL */
    wire        u_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    axis_bridge #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS),
                  .W_BITS(W_BITS), .LANES(LANES), .DRAIN_CYCLES(DRAIN_CYCLES),
                  .HOST_SEL_RST(HOST_SEL_RST)) u_axis (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready), .s_axis_tdata(s_axis_tdata),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready), .m_axis_tdata(m_axis_tdata),
        .m_axis_tkeep(m_axis_tkeep), .m_axis_tlast(m_axis_tlast),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot), .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready), .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr),
        .s_axi_arprot(s_axi_arprot), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .chip_rst_n(a_rst_n), .chip_in_valid(a_in_valid), .chip_in_ready(c_in_ready && !host_sel),
        .chip_in_flit(a_in_flit), .chip_out_valid(c_out_valid && !host_sel), .chip_out_ready(a_out_ready),
        .chip_out_flit(c_out_flit), .chip_step(a_step), .chip_step_done(c_step_done && !host_sel),
        .chip_t_parity(t_parity), .chip_running(c_running), .chip_err(chip_err), .chip_drop_cnt(c_drop),
        .uart_rx_overflow(u_ovf), .uart_frame_err(u_ferr), .host_sel(host_sel));

    uart_bridge #(.CLK_HZ(CLK_HZ), .BAUD(BAUD), .MESH_X(MESH_X), .MESH_Y(MESH_Y),
                  .N_NEURONS(N_NEURONS), .DRAIN_CYCLES(DRAIN_CYCLES)) u_uart (
        .clk(clk), .rst_n(rst_n), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .chip_rst_n(u_rst_n), .chip_in_valid(u_in_valid), .chip_in_ready(c_in_ready && host_sel),
        .chip_in_flit(u_in_flit), .chip_out_valid(c_out_valid && host_sel), .chip_out_ready(u_out_ready),
        .chip_out_flit(c_out_flit), .chip_step(u_step), .chip_step_done(c_step_done && host_sel),
        .chip_t_parity(t_parity), .chip_running(c_running), .chip_err(chip_err), .chip_drop_cnt(c_drop),
        .rx_overflow(u_ovf), .rx_frame_err(u_ferr), .link_busy(u_busy));

    assign c_rst_n     = host_sel ? u_rst_n : a_rst_n;
    assign c_in_valid  = host_sel ? u_in_valid : a_in_valid;
    assign c_in_flit   = host_sel ? u_in_flit : a_in_flit;
    assign c_out_ready = host_sel ? u_out_ready : a_out_ready;
    assign c_step      = host_sel ? u_step : a_step;
endmodule

`default_nettype wire
