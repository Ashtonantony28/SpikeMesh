// Behavioural stand-in for `neurocore` (SPEC §8 ports) used only by the NoC /
// top-level tests in tests/noc. NOT synthesisable intent; not the real core.
//
// Behaviour (all host-bound flits are SPIKE-to-host, src = (X, Y)):
//  * Every accepted input flit holds `busy` for 0..7 random cycles.
//  * SPIKE with parity q: rx_cnt[q]++. If axon[9] = 1 the stub also forwards
//    a host spike {neuron = {2'b00, axon[7:0]}, parity q}.
//  * learn_start: emits a REPORT host spike {neuron = {2'b10, rx_cnt[p][7:0]},
//    parity p}, clears rx_cnt[p], then stays busy (LEARN) for 4..19 cycles,
//    refusing input (allowed by SPEC §8).
//  * update_start: busy for 2..9 cycles, then emits UPD_HOST host spikes
//    {neuron = {2'b01, k}, parity p^1} and UPD_FWD spikes to core (0,0) with
//    axon = {1'b1, X[3:0], Y[3:0], k[0]} and parity p^1. Input is still
//    accepted during UPDATE (see rtl/noc/README.md, message-dependent deadlock).
module neurocore #(
    parameter int X            = 0,
    parameter int Y            = 0,
    parameter int N_NEURONS    = 256,
    parameter int N_AXONS      = 256,
    parameter int W_BITS       = 8,
    parameter int FANOUT_DEPTH = 1024,
    parameter int LANES        = 1,
    parameter int UPD_HOST     = 2,
    parameter int UPD_FWD      = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    output logic        in_ready,
    input  logic [47:0] in_flit,
    output logic        out_valid,
    input  logic        out_ready,
    output logic [47:0] out_flit,
    input  logic        parity,
    input  logic        learn_start,
    input  logic        update_start,
    output logic        busy
);
    localparam int QD = 16;
    logic [47:0] q [QD];
    int          q_rd, q_cnt;
    logic [15:0] lfsr;
    int          proc_cnt, learn_cnt, upd_wait, upd_emit;
    logic        upd_active;
    int          rx_cnt [2];
    logic        acc_in, deq;
    logic [47:0] push_a, push_b;
    logic        do_a, do_b;

    function automatic logic [47:0] host_spk(input int neuron, input logic par);
        host_spk = {2'b00, 4'd15, 4'd0, par, 7'd0, 4'(X), 4'(Y), 12'd0, 10'(neuron)};
    endfunction
    function automatic logic [47:0] spk(input int dx, input int dy, input int axon, input logic par);
        spk = {2'b00, 4'(dx), 4'(dy), par, 27'd0, 10'(axon)};
    endfunction

    assign out_valid = (q_cnt != 0);
    assign out_flit  = q[q_rd];
    assign deq       = out_valid && out_ready;
    assign in_ready  = (proc_cnt == 0) && (learn_cnt == 0) && (q_cnt <= QD - 4);
    assign acc_in    = in_valid && in_ready;
    assign busy      = (proc_cnt != 0) || (learn_cnt != 0) || upd_active || (q_cnt != 0) || acc_in;

    always @(posedge clk) begin
        if (!rst_n) begin
            q_rd <= 0; q_cnt <= 0; proc_cnt <= 0; learn_cnt <= 0;
            upd_wait <= 0; upd_emit <= 0; upd_active <= 1'b0;
            rx_cnt[0] <= 0; rx_cnt[1] <= 0;
            lfsr <= 16'hACE1 ^ 16'(X * 77 + Y * 131 + 1);
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            do_a = 1'b0; do_b = 1'b0; push_a = '0; push_b = '0;
            if (proc_cnt != 0)  proc_cnt  <= proc_cnt - 1;
            if (learn_cnt != 0) learn_cnt <= learn_cnt - 1;
            // ---- input ----
            if (acc_in) begin
                proc_cnt <= int'(lfsr[2:0]);
                if (in_flit[47:46] == 2'b00) begin
                    rx_cnt[in_flit[37]] <= rx_cnt[in_flit[37]] + 1;
                    if (in_flit[9]) begin
                        do_a = 1'b1;
                        push_a = host_spk(int'(in_flit[7:0]), in_flit[37]);
                    end
                end
            end
            // ---- learn ----
            if (learn_start) begin
                do_b = 1'b1;
                push_b = host_spk(512 + (rx_cnt[parity] & 255), parity);
                rx_cnt[parity] <= (acc_in && in_flit[47:46] == 2'b00 && in_flit[37] == parity) ? 1 : 0;
                learn_cnt <= 4 + int'(lfsr[3:0]);
            end
            // ---- update ----
            if (update_start) begin
                upd_active <= 1'b1;
                upd_wait   <= 2 + int'(lfsr[6:4]);
                upd_emit   <= 0;
            end else if (upd_active) begin
                if (upd_wait != 0) upd_wait <= upd_wait - 1;
                else if (upd_emit == UPD_HOST + UPD_FWD) upd_active <= 1'b0;
                else if (!do_b && q_cnt < QD - 2) begin
                    do_b = 1'b1;
                    if (upd_emit < UPD_HOST) push_b = host_spk(256 + upd_emit, ~parity);
                    else push_b = spk(0, 0, 512 + X * 32 + Y * 2 + ((upd_emit - UPD_HOST) & 1), ~parity);
                    upd_emit <= upd_emit + 1;
                end
            end
            // ---- queue ----
            begin
                int wr, n;
                n  = q_cnt - (deq ? 1 : 0);
                wr = (q_rd + q_cnt) % QD;
                if (do_a) begin q[wr] <= push_a; wr = (wr + 1) % QD; n = n + 1; end
                if (do_b) begin q[wr] <= push_b; n = n + 1; end
                if (n > QD) $error("core_stub(%0d,%0d): queue overflow", X, Y);
                q_cnt <= n;
                if (deq) q_rd <= (q_rd + 1) % QD;
            end
        end
    end
endmodule
