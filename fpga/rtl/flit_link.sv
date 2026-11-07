// flit_link: host-link controller shared by uart_bridge and axis_bridge.
//
// Host -> chip: an in-order stream of 48-bit words. A word whose header
// [47:38] is 10'b11_1111_1111 (type CTRL, dest_x = 15, dest_y = 15) is a link
// COMMAND (a host never sends a host-bound flit, so this is unambiguous);
// every other word is a flit forwarded to the chip host port.
//
// Commands (opcode = word[7:0]):
//   0x01 STEP        pulse `step` (waits while a STEP is still running), wait
//                    for step_done, then drain, then emit STEP_DONE
//   0x02 FLUSH       drain, then emit FLUSH_DONE
//   0x03 STATUS      emit STATUS
//   0x04 SOFT_RESET  hold chip_rst_n low for 16 cycles, emit RESET_DONE
//   other            ignored, sets the sticky `cmd_err`
// The same commands can be raised by pulses on reg_step / reg_flush /
// reg_reset (AXI-Lite path); they are taken between words.
//
// Chip -> host: host_out flits pass through unchanged; event words use the
// same 0xFFC header (no chip flit can have it: host-bound spikes have
// dest_y = 0, READ_RESP has type 10):
//   0x81 STEP_DONE   [8] t_parity after the step, [9] err, [37:16] STEP cycles
//                    (step pulse -> step_done, 22-bit saturating)
//   0x82 FLUSH_DONE
//   0x83 STATUS      [8] t_parity, [9] err, [10] chip running, [23:16] drop_cnt,
//                    [27:24] MESH_X, [31:28] MESH_Y, [37:32] log2(N_NEURONS)
//   0x84 RESET_DONE
// Event words carry o_last = 1 (AXI-Stream TLAST: one DMA transfer per event).
//
// Drain: after step_done the chip may still have host-bound spikes in the
// routers (SPEC §8). The event word (STEP_DONE / FLUSH_DONE) is emitted only
// when the chip is drained, so every spike of UPDATE(t) precedes STEP_DONE(t)
// in the output stream:
//   DRAIN_CYCLES = 0 (default): spikemesh_top's `drained` (no STEP running,
//     mesh and cores quiet, every UPDATE spike delivered to a core or to the
//     host-out FIFO) AND host_out_valid low (that FIFO is empty). Exact.
//   DRAIN_CYCLES > 0 (fallback, e.g. a chip without `drained`): host_out_valid
//     low for DRAIN_CYCLES consecutive cycles. `chip_drained` is ignored.
`default_nettype none

module flit_link #(
    parameter integer MESH_X       = 2,
    parameter integer MESH_Y       = 2,
    parameter integer N_NEURONS    = 256,
    parameter integer DRAIN_CYCLES = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    // host -> link
    input  wire        i_valid,
    output wire        i_ready,
    input  wire [47:0] i_data,
    input  wire        reg_step,
    input  wire        reg_flush,
    input  wire        reg_reset,
    // link -> host
    output wire        o_valid,
    input  wire        o_ready,
    output wire [47:0] o_data,
    output wire        o_last,
    // chip
    output wire        chip_rst_n,
    output wire        chip_in_valid,
    input  wire        chip_in_ready,
    output wire [47:0] chip_in_flit,
    input  wire        chip_out_valid,
    output wire        chip_out_ready,
    input  wire [47:0] chip_out_flit,
    output reg         chip_step,
    input  wire        chip_step_done,
    input  wire        chip_t_parity,
    input  wire        chip_running,
    input  wire        chip_err,
    input  wire [7:0]  chip_drop_cnt,
    input  wire        chip_drained,
    // status / counters
    output wire        link_busy,
    output reg         done_pulse,      // STEP_DONE emitted
    output reg         cmd_err,
    output reg  [31:0] step_count,
    output reg  [31:0] last_step_cycles,
    output reg  [31:0] last_drain_cycles,
    output reg  [31:0] busy_cycles,
    output reg  [31:0] cycle_count,
    output reg  [31:0] in_flits,
    output reg  [31:0] out_flits
);
    localparam [9:0] HDR = 10'b11_1111_1111;
    localparam [7:0] C_STEP = 8'h01, C_FLUSH = 8'h02, C_STATUS = 8'h03, C_RESET = 8'h04;
    localparam [7:0] E_STEP = 8'h81, E_FLUSH = 8'h82, E_STATUS = 8'h83, E_RESET = 8'h84;
    localparam integer DW = $clog2(((DRAIN_CYCLES > 0) ? DRAIN_CYCLES : 1) + 1);
    localparam [5:0] LOGN = 6'($clog2(N_NEURONS));

    localparam [2:0] L_IDLE = 3'd0, L_STEPW = 3'd1, L_RUN = 3'd2, L_DRAIN = 3'd3,
                     L_EMIT = 3'd4, L_RST = 3'd5;

    reg [2:0]    st;
    reg [7:0]    ev_op;        // event to emit after drain / reset
    reg [DW-1:0] quiet;
    reg [4:0]    rst_cnt;
    reg [47:0]   ev_word;
    reg [31:0]   step_cyc, drain_cyc;

    wire is_cmd   = (i_data[47:38] == HDR);
    wire in_idle  = (st == L_IDLE);
    wire reg_any  = reg_step | reg_flush | reg_reset;
    // in IDLE: register commands first, then the word stream
    wire take_word = in_idle && !reg_any && i_valid;
    wire fwd       = take_word && !is_cmd;
    wire take_cmd  = take_word && is_cmd;

    assign chip_in_valid = fwd;
    assign chip_in_flit  = i_data;
    assign i_ready       = in_idle && !reg_any && (is_cmd || chip_in_ready);
    assign chip_rst_n    = rst_n && (st != L_RST);

    // output: chip flits pass through except while an event word is pending
    wire emitting = (st == L_EMIT);
    assign o_valid        = emitting ? 1'b1 : chip_out_valid;
    assign o_data         = emitting ? ev_word : chip_out_flit;
    assign o_last         = emitting;
    assign chip_out_ready = emitting ? 1'b0 : o_ready;

    assign link_busy = !in_idle;

    wire [7:0] op = reg_step ? C_STEP : reg_flush ? C_FLUSH : reg_reset ? C_RESET : i_data[7:0];
    wire       start_cmd = in_idle && (reg_any || take_cmd);

    function automatic [47:0] event_word(input [7:0] e, input [37:8] payload);
        event_word = {HDR, payload, e};
    endfunction

    wire drain_ok = (DRAIN_CYCLES == 0) ? (chip_drained && !chip_out_valid)
                                        : (quiet == DW'(DRAIN_CYCLES));

    wire [21:0] step_sat = (step_cyc > 32'h3FFFFF) ? 22'h3FFFFF : step_cyc[21:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= L_IDLE; ev_op <= 8'd0; quiet <= '0; rst_cnt <= 5'd0; ev_word <= 48'd0;
            chip_step <= 1'b0; done_pulse <= 1'b0; cmd_err <= 1'b0;
            step_count <= 32'd0; last_step_cycles <= 32'd0; last_drain_cycles <= 32'd0;
            busy_cycles <= 32'd0; cycle_count <= 32'd0; in_flits <= 32'd0; out_flits <= 32'd0;
            step_cyc <= 32'd0; drain_cyc <= 32'd0;
        end else begin
            chip_step  <= 1'b0;
            done_pulse <= 1'b0;
            cycle_count <= cycle_count + 32'd1;
            if (chip_running) busy_cycles <= busy_cycles + 32'd1;
            if (fwd && chip_in_ready) in_flits <= in_flits + 32'd1;
            if (chip_out_valid && chip_out_ready) out_flits <= out_flits + 32'd1;

            case (st)
                L_IDLE: if (start_cmd) begin
                    case (op)
                        C_STEP:   st <= L_STEPW;
                        C_FLUSH:  begin ev_op <= E_FLUSH; quiet <= '0; drain_cyc <= 32'd0; st <= L_DRAIN; end
                        C_STATUS: begin
                            ev_word <= event_word(E_STATUS, {LOGN, 4'(MESH_Y), 4'(MESH_X),
                                                             chip_drop_cnt, 5'd0, chip_running, chip_err, chip_t_parity});
                            st <= L_EMIT;
                        end
                        C_RESET:  begin rst_cnt <= 5'd0; st <= L_RST; end
                        default:  cmd_err <= 1'b1;
                    endcase
                end
                L_STEPW: if (!chip_running) begin       // never overlap two STEPs
                    chip_step <= 1'b1;
                    step_cyc  <= 32'd0;
                    st        <= L_RUN;
                end
                L_RUN: begin
                    step_cyc <= step_cyc + 32'd1;
                    if (chip_step_done) begin
                        last_step_cycles <= step_cyc + 32'd1;
                        step_count <= step_count + 32'd1;
                        ev_op <= E_STEP; quiet <= '0; drain_cyc <= 32'd0;
                        st <= L_DRAIN;
                    end
                end
                L_DRAIN: begin
                    drain_cyc <= drain_cyc + 32'd1;
                    if (chip_out_valid) quiet <= '0;
                    else if (quiet != DW'(DRAIN_CYCLES)) quiet <= quiet + 1'b1;
                    if (drain_ok) begin
                        if (ev_op == E_STEP) last_drain_cycles <= drain_cyc;
                        ev_word <= event_word(ev_op, (ev_op == E_STEP) ?
                                              {step_sat, 6'd0, chip_err, chip_t_parity} : 30'd0);
                        st <= L_EMIT;
                    end
                end
                L_EMIT: if (o_ready) begin
                    if (ev_word[7:0] == E_STEP) done_pulse <= 1'b1;
                    st <= L_IDLE;
                end
                L_RST: begin
                    rst_cnt <= rst_cnt + 5'd1;
                    if (rst_cnt == 5'd15) begin
                        ev_word <= event_word(E_RESET, 30'd0);
                        st <= L_EMIT;
                    end
                end
                default: st <= L_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
