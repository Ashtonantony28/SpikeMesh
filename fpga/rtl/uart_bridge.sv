// uart_bridge: 8N1 UART <-> SpikeMesh host port.
//
// Every word in both directions is 6 bytes, little-endian (byte 0 = bits
// [7:0]), SPEC §7. Host -> chip words are flits or link commands, and
// chip -> host words are flits or link events. flit_link.sv describes the
// command/event words (header 0xFFC: type CTRL, dest 15/15).
//
// Resynchronisation: if a word is incomplete and no byte arrives for
// RESYNC_CYCLES cycles, the partial word is discarded. Default is 16 byte
// times. A host that loses sync pauses for ~1 ms and resends.
// Received bytes go through a BYTE_FIFO-deep FIFO. On overflow the byte is
// dropped and the sticky `rx_overflow` is set. The host avoids overflow by
// waiting for STEP_DONE / FLUSH_DONE before it sends again (host/fpga_backend.py).
`default_nettype none

module uart_bridge #(
    parameter integer CLK_HZ        = 100_000_000,
    parameter integer BAUD          = 921_600,
    parameter integer MESH_X        = 2,
    parameter integer MESH_Y        = 2,
    parameter integer N_NEURONS     = 256,
    parameter integer DRAIN_CYCLES  = 64,
    parameter integer BYTE_FIFO     = 64,
    parameter integer CLKS_PER_BIT  = (CLK_HZ + BAUD / 2) / BAUD,
    parameter integer RESYNC_CYCLES = 16 * 10 * CLKS_PER_BIT
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        uart_rx,
    output wire        uart_tx,
    // chip host port (spikemesh_top)
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
    // status
    output reg         rx_overflow,
    output reg         rx_frame_err,
    output wire        link_busy
);
    localparam integer FAW = $clog2(BYTE_FIFO);
    localparam integer RW  = $clog2(RESYNC_CYCLES + 1);

    // ---------------- receive: bytes -> FIFO -> 48-bit words ----------------
    wire       rxb_valid, rxb_ferr;
    wire [7:0] rxb;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst_n(rst_n), .rx(uart_rx), .valid(rxb_valid), .data(rxb), .frame_err(rxb_ferr));

    reg [7:0]   fifo [0:BYTE_FIFO-1];
    reg [FAW:0] f_cnt;
    reg [FAW-1:0] f_wp, f_rp;
    wire f_empty = (f_cnt == '0);
    wire f_full  = (f_cnt == (FAW+1)'(BYTE_FIFO));

    reg [47:0]  asm_word;
    reg [2:0]   asm_n;        // bytes collected
    reg         w_valid;      // assembled word waiting for flit_link
    wire        w_ready;
    reg [RW-1:0] idle_cnt;
    wire        f_pop = !f_empty && !w_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            f_cnt <= '0; f_wp <= '0; f_rp <= '0;
            asm_word <= 48'd0; asm_n <= 3'd0; w_valid <= 1'b0; idle_cnt <= '0;
            rx_overflow <= 1'b0; rx_frame_err <= 1'b0;
        end else begin
            if (rxb_ferr) rx_frame_err <= 1'b1;
            if (rxb_valid && f_full) rx_overflow <= 1'b1;
            if (rxb_valid && !f_full) begin
                fifo[f_wp] <= rxb;
                f_wp <= f_wp + 1'b1;
            end
            if (f_pop) f_rp <= f_rp + 1'b1;
            f_cnt <= f_cnt + {{FAW{1'b0}}, (rxb_valid && !f_full)} - {{FAW{1'b0}}, f_pop};

            if (w_valid && w_ready) w_valid <= 1'b0;
            if (f_pop) begin
                asm_word <= {fifo[f_rp], asm_word[47:8]};
                idle_cnt <= '0;
                if (asm_n == 3'd5) begin
                    asm_n <= 3'd0;
                    w_valid <= 1'b1;
                end else asm_n <= asm_n + 3'd1;
            end else if (asm_n != 3'd0 && f_empty) begin
                if (idle_cnt == RW'(RESYNC_CYCLES)) begin
                    asm_n <= 3'd0;           // drop the partial word
                    idle_cnt <= '0;
                end else idle_cnt <= idle_cnt + 1'b1;
            end
        end
    end

    // ---------------- link controller ----------------
    wire        o_valid, o_last;
    wire [47:0] o_data;
    reg         o_ready;
    /* verilator lint_off UNUSEDSIGNAL */
    wire done_pulse, cmd_err;
    wire [31:0] step_count, last_step_cycles, last_drain_cycles, busy_cycles, cycle_count, in_flits, out_flits;
    /* verilator lint_on UNUSEDSIGNAL */

    flit_link #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .DRAIN_CYCLES(DRAIN_CYCLES)) u_link (
        .clk(clk), .rst_n(rst_n),
        .i_valid(w_valid), .i_ready(w_ready), .i_data(asm_word),
        .reg_step(1'b0), .reg_flush(1'b0), .reg_reset(1'b0),
        .o_valid(o_valid), .o_ready(o_ready), .o_data(o_data), .o_last(o_last),
        .chip_rst_n(chip_rst_n),
        .chip_in_valid(chip_in_valid), .chip_in_ready(chip_in_ready), .chip_in_flit(chip_in_flit),
        .chip_out_valid(chip_out_valid), .chip_out_ready(chip_out_ready), .chip_out_flit(chip_out_flit),
        .chip_step(chip_step), .chip_step_done(chip_step_done), .chip_t_parity(chip_t_parity),
        .chip_running(chip_running), .chip_err(chip_err), .chip_drop_cnt(chip_drop_cnt),
        .link_busy(link_busy), .done_pulse(done_pulse), .cmd_err(cmd_err),
        .step_count(step_count), .last_step_cycles(last_step_cycles),
        .last_drain_cycles(last_drain_cycles), .busy_cycles(busy_cycles),
        .cycle_count(cycle_count), .in_flits(in_flits), .out_flits(out_flits));

    // ---------------- transmit: words -> 6 bytes ----------------
    reg  [47:0] tx_word;
    reg  [2:0]  tx_left;      // bytes left in tx_word
    wire        txb_ready;
    always @* o_ready = (tx_left == 3'd0);

    always @(posedge clk) begin
        if (!rst_n) begin
            tx_word <= 48'd0; tx_left <= 3'd0;
        end else if (tx_left == 3'd0) begin
            if (o_valid) begin tx_word <= o_data; tx_left <= 3'd6; end
        end else if (txb_ready) begin
            tx_word <= {8'd0, tx_word[47:8]};
            tx_left <= tx_left - 3'd1;
        end
    end

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst_n(rst_n), .valid(tx_left != 3'd0), .ready(txb_ready),
        .data(tx_word[7:0]), .tx(uart_tx));

    /* verilator lint_off UNUSEDSIGNAL */
    wire unused = o_last;
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
