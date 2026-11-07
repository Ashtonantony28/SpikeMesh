// Verilog-2001 wrapper around kv260_top (SystemVerilog) so that Vivado IP
// Integrator can use it as a module reference: Vivado does not accept a
// SystemVerilog file as the top of a module reference. The port names follow
// the AXI naming convention (s_axi_*, s_axis_*, m_axis_*), so Vivado infers the
// interfaces. The attributes below bind them to clk / rst_n.
module kv260_top_wrap #(
    parameter MESH_X       = 2,
    parameter MESH_Y       = 2,
    parameter N_NEURONS    = 256,
    parameter N_AXONS      = 256,
    parameter W_BITS       = 8,
    parameter FANOUT_DEPTH = 1024,
    parameter LANES        = 8,
    parameter CLK_HZ       = 100000000,
    parameter BAUD         = 921600,
    parameter DRAIN_CYCLES = 0,   // 0: use spikemesh_top drained; > 0: idle-timer fallback
    parameter HOST_SEL_RST = 0,
    parameter LEARN_MASK   = 0,   // bit y*MESH_X+x: that core gets learning hardware
    parameter LEARN_MODE   = 1    // 1 STDP, 2 STDP + R-STDP
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:s_axis:m_axis, ASSOCIATED_RESET rst_n, FREQ_HZ 100000000" *)
    input  wire        clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        rst_n,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tlast,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [63:0] m_axis_tdata,
    output wire [7:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
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
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire        t_parity,
    output wire        chip_err
);
    kv260_top #(
        .MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS),
        .W_BITS(W_BITS), .FANOUT_DEPTH(FANOUT_DEPTH), .LANES(LANES), .CLK_HZ(CLK_HZ),
        .BAUD(BAUD), .DRAIN_CYCLES(DRAIN_CYCLES), .HOST_SEL_RST(HOST_SEL_RST),
        .LEARN_MASK(LEARN_MASK), .LEARN_MODE(LEARN_MODE)
    ) u_top (
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
        .uart_rx(uart_rx), .uart_tx(uart_tx), .t_parity(t_parity), .chip_err(chip_err)
    );
endmodule
