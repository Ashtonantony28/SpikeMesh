// SpikeMesh tick controller (SPEC §3, §8).
//
// On `step` (1-cycle pulse, accepted only in IDLE):
//   QUIESCE : wait until the registered global busy OR has been low for
//             QUIESCE_CYCLES consecutive cycles (see below)
//   LEARN   : pulse learn_start, then wait until no core is busy
//   UPDATE  : pulse update_start, then wait until no core is busy
//   DONE    : step_done (1-cycle pulse, combinational in the last wait cycle)
//             and parity flips on the same clock edge.
//
// Quiescence: busy_r <= |{net_busy, core_busy} (registered OR); quiet_cnt
// counts consecutive cycles with busy_r low (saturating). Quiescence is
// declared in cycle c iff quiet_cnt >= QUIESCE_CYCLES and busy_r is low, i.e.
// the raw OR was low in every one of cycles c-1 ... c-QUIESCE_CYCLES-1.
//
// Start guard: after learn_start/update_start the controller ignores
// core_busy for START_GUARD cycles. Core contract: busy must be high no later
// than START_GUARD cycles after the cycle in which the start pulse is high
// (a core that registers the pulse and raises busy the next cycle is fine),
// or the core must have finished the work by then.
//
// drained: level, high while no STEP is running AND the quiescence condition
// holds (raw busy OR low for the last QUIESCE_CYCLES+1 cycles). After
// step_done it rises once every spike emitted by UPDATE(t) has reached its
// core (and been accumulated) or the host-out FIFO; host-bound flits may
// still be waiting in that FIFO (host_out_valid). Host injection makes it
// fall again. It is the exact replacement for an "output idle N cycles" rule.
//
// A `step` that arrives while a STEP is running is ignored and sets the
// sticky `step_err` output.
module tick_ctrl #(
    parameter int NC             = 4,
    parameter int QUIESCE_CYCLES = 4,
    parameter int START_GUARD    = 2
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          step,
    input  logic          net_busy,      // router FIFOs / link registers / host input pending
    input  logic [NC-1:0] core_busy,
    output logic          learn_start,
    output logic          update_start,
    output logic          step_done,
    output logic          parity,
    output logic          quiescent,     // 1 in the cycle quiescence is declared
    output logic          running,       // a STEP is in progress
    output logic          step_err,
    output logic          drained        // idle and quiet (see header)
);
    localparam int QW = $clog2(QUIESCE_CYCLES + 1);
    localparam int GW = $clog2(START_GUARD + 1);

    typedef enum logic [1:0] { S_IDLE, S_QUIESCE, S_WAIT_LEARN, S_WAIT_UPD } state_t;
    state_t        state;
    logic          busy_r;
    logic [QW-1:0] quiet_cnt;
    logic [GW-1:0] guard;
    logic          core_any;

    assign core_any  = |core_busy;
    assign quiescent = (state == S_QUIESCE) && (quiet_cnt >= QW'(QUIESCE_CYCLES)) && !busy_r;
    assign step_done = (state == S_WAIT_UPD) && (guard == '0) && !core_any;
    assign running   = (state != S_IDLE);
    assign drained   = (state == S_IDLE) && (quiet_cnt >= QW'(QUIESCE_CYCLES)) && !busy_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy_r    <= 1'b1;
            quiet_cnt <= '0;
        end else begin
            busy_r <= net_busy | core_any;
            if (busy_r)                                quiet_cnt <= '0;
            else if (quiet_cnt != QW'(QUIESCE_CYCLES)) quiet_cnt <= quiet_cnt + QW'(1);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            guard        <= '0;
            learn_start  <= 1'b0;
            update_start <= 1'b0;
            parity       <= 1'b0;
            step_err     <= 1'b0;
        end else begin
            learn_start  <= 1'b0;
            update_start <= 1'b0;
            if (guard != '0) guard <= guard - GW'(1);
            if (step && state != S_IDLE) step_err <= 1'b1;
            case (state)
                S_IDLE: if (step) state <= S_QUIESCE;
                S_QUIESCE: if (quiescent) begin
                    learn_start <= 1'b1;
                    guard       <= GW'(START_GUARD);
                    state       <= S_WAIT_LEARN;
                end
                S_WAIT_LEARN: if (guard == '0 && !core_any) begin
                    update_start <= 1'b1;
                    guard        <= GW'(START_GUARD);
                    state        <= S_WAIT_UPD;
                end
                S_WAIT_UPD: if (step_done) begin
                    parity <= ~parity;
                    state  <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
`ifdef FORMAL
    // ==================================================================
    // Formal properties (formal/run.sh). Inputs are unconstrained except
    // that reset is applied in cycle 0 only.
    // ==================================================================
    logic f_past_valid = 1'b0;
    always @(posedge clk) f_past_valid <= 1'b1;
    always @(*) assume (rst_n == f_past_valid);

    // raw busy history: f_hist[k] = raw OR (k+1) cycles ago; ones after reset
    logic [QUIESCE_CYCLES:0] f_hist;
    always @(posedge clk) begin
        if (!rst_n) f_hist <= '1;
        else        f_hist <= {f_hist[QUIESCE_CYCLES-1:0], net_busy | core_any};
    end
    // phase tracker: 0 idle, 1 quiesce, 2 learn issued, 3 update issued
    logic [1:0] f_phase;
    always @(posedge clk) begin
        if (!rst_n) f_phase <= 2'd0;
        else if (state == S_IDLE && step)  f_phase <= 2'd1;
        else if (quiescent)                f_phase <= 2'd2;
        else if (state == S_WAIT_LEARN && guard == '0 && !core_any) f_phase <= 2'd3;
        else if (step_done)                f_phase <= 2'd0;
    end
    always @(*) begin
        if (rst_n) begin
            // T1 step_done never while any core is busy
            if (step_done) assert (!core_any);
            // T2 quiescence never declared if any busy input was high in the
            //    last QUIESCE_CYCLES+1 cycles
            if (quiescent) assert (f_hist == '0);
            if (drained)   assert (f_hist == '0 && state == S_IDLE);
            // T3 the start pulses follow the SPEC section 8 order
            if (learn_start)  assert (f_phase == 2'd2 && state == S_WAIT_LEARN);
            if (update_start) assert (f_phase == 2'd3 && state == S_WAIT_UPD);
            if (step_done)    assert (f_phase == 2'd3);
            assert (!(learn_start && update_start));
            // helper invariants (make the induction go through)
            assert (state == S_IDLE || state == S_QUIESCE || state == S_WAIT_LEARN || state == S_WAIT_UPD);
            assert (guard <= GW'(START_GUARD));
            assert (quiet_cnt <= QW'(QUIESCE_CYCLES));
            assert (busy_r == f_hist[0]);
            for (int k = 0; k < QUIESCE_CYCLES; k++)
                if (quiet_cnt > QW'(k)) assert (!f_hist[k+1]);
            assert ((f_phase == 2'd0) == (state == S_IDLE));
            assert ((f_phase == 2'd1) == (state == S_QUIESCE));
            assert ((f_phase == 2'd2) == (state == S_WAIT_LEARN));
            assert ((f_phase == 2'd3) == (state == S_WAIT_UPD));
            if (learn_start || update_start) assert (guard == GW'(START_GUARD));
            if (state == S_IDLE || state == S_QUIESCE) assert (!learn_start && !update_start);
        end
    end
`endif
endmodule
