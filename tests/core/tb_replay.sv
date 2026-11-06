// Trace-replay testbench for neurocore (used for the Verilator run; cocotb 2.1
// needs Verilator >= 5.036 and 5.020 is installed). tests/core/vl_replay.py
// writes the command trace from the golden-model stimulus generator and
// checks the output log.
//
// Trace lines "<op> <hex>":  0 flit -> input queue | 1 wait quiet (queue empty,
// busy low 2 cycles) | 2 learn pulse, wait !busy | 3 update pulse with
// parity=arg (one cycle, no wait) | 6 wait !busy | 4 set output tag | 9 end.
// Output lines "<tag> <flit hex>" for every accepted output flit.
// Self-addressed SPIKE outputs are looped back into the input queue.
`timescale 1ns/1ps
`default_nettype none

module tb_replay;
    parameter integer X = 2, Y = 1, N_NEURONS = 32, N_AXONS = 32, W_BITS = 8;
    parameter integer FANOUT_DEPTH = 64, LANES = 4, LEARN = 0;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg         rst_n = 1'b0;
    reg         in_valid = 1'b0;
    reg  [47:0] in_flit = 48'd0;
    wire        in_ready;
    wire        out_valid;
    reg         out_ready = 1'b0;
    wire [47:0] out_flit;
    reg         parity = 1'b0, learn_start = 1'b0, update_start = 1'b0;
    wire        busy;

    neurocore #(.X(X), .Y(Y), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS), .W_BITS(W_BITS),
                .FANOUT_DEPTH(FANOUT_DEPTH), .LANES(LANES), .LEARN(LEARN)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready), .in_flit(in_flit),
        .out_valid(out_valid), .out_ready(out_ready), .out_flit(out_flit), .parity(parity),
        .learn_start(learn_start), .update_start(update_start), .busy(busy));

    logic [47:0] inq [$];
    integer fo, fi, tag, bp_pct, gap_pct, r, op;
    reg [63:0] arg;
    reg [31:0] rs;
    reg pulse_l, pulse_u;
    reg busy_s;
    longint unsigned ncycles;

    function automatic [31:0] xs(input [31:0] x);
        reg [31:0] y;
        y = x ^ (x << 13); y = y ^ (y >> 17); y = y ^ (y << 5);
        return y;
    endfunction

    // one clock cycle: drive at negedge, sample 1 ns later (settled); the
    // handshakes sampled here complete at the following posedge.
    task automatic cycle();
        reg drive, ordy;
        @(negedge clk);
        learn_start  = pulse_l; pulse_l = 1'b0;
        update_start = pulse_u; pulse_u = 1'b0;
        rs = xs(rs);
        drive = (inq.size() != 0) && ((rs % 100) >= gap_pct);
        in_valid = drive;
        if (drive) in_flit = inq[0];
        rs = xs(rs);
        ordy = (rs % 100) >= bp_pct;
        out_ready = ordy;
        #1;
        ncycles++;
        if (drive && in_ready) void'(inq.pop_front());
        if (ordy && out_valid) begin
            $fdisplay(fo, "%0d %012h", tag, out_flit);
            if (out_flit[47:46] == 2'b00 && out_flit[45:42] == 4'(X) && out_flit[41:38] == 4'(Y))
                inq.push_back(out_flit);
        end
        busy_s = busy;
    endtask

    task automatic wait_quiet();
        integer cnt;
        cnt = 0;
        while (cnt < 2) begin
            cycle();
            cnt = (inq.size() == 0 && !busy_s) ? cnt + 1 : 0;
        end
    endtask

    task automatic wait_not_busy();
        cycle();
        while (busy_s) cycle();
    endtask

    initial begin
        string tf, of;
        integer seed;
        if (!$value$plusargs("trace=%s", tf)) $fatal(1, "need +trace=");
        if (!$value$plusargs("out=%s", of)) $fatal(1, "need +out=");
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("bp=%d", bp_pct)) bp_pct = 30;
        if (!$value$plusargs("gap=%d", gap_pct)) gap_pct = 10;
        rs = 32'h9E3779B9 ^ seed;
        tag = -1; pulse_l = 0; pulse_u = 0; ncycles = 0;
        fi = $fopen(tf, "r");
        fo = $fopen(of, "w");
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        wait_not_busy();
        while (1) begin
            r = $fscanf(fi, "%d %h\n", op, arg);
            if (r != 2) break;
            case (op)
                0: inq.push_back(arg[47:0]);
                1: wait_quiet();
                2: begin pulse_l = 1'b1; wait_not_busy(); end
                3: begin parity = arg[0]; pulse_u = 1'b1; cycle(); end
                6: wait_not_busy();
                4: tag = int'(arg[31:0]);
                default: break;
            endcase
        end
        wait_quiet();
        $display("tb_replay: done, %0d cycles", ncycles);
        $fclose(fo);
        $finish;
    end
endmodule

`default_nettype wire
