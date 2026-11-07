// Command-driven system testbench for spikemesh_top (SG5). Used by
// host/rtl_backend.py with both Icarus (-g2012) and Verilator (--binary --timing).
//
// Commands are read from +cmd=<path> (a named pipe for interactive use, or a
// file), one per line "<op> <hex>":
//   0 <flit>  queue a host->chip flit (driven on host_in with valid/ready)
//   1 0       STEP: wait until every queued flit is accepted, pulse `step`,
//             wait `step_done`, then wait until `drained` and host_out empty
//   2 0       SYNC: wait until every queued flit is accepted, `drained` and
//             host_out empty (used after CFG loads and READ_REQs)
//   9 0       finish
// Responses go to +out=<path>, flushed after each 1/2 command:
//   F <flit>                 every chip->host flit, in arrival order
//   D <cycles> <err> <drop>  command done: cycles it took, sticky err, drop_cnt
// +bp=<pct> applies random host_out backpressure (default 0). A command that
// takes more than +maxwait=<cycles> (default 20M) stops the simulation ($fatal).
`timescale 1ns/1ps
`default_nettype none

module tb_sys;
    parameter integer MESH_X = 2, MESH_Y = 2, N_NEURONS = 32, N_AXONS = 32, W_BITS = 8;
    parameter integer FANOUT_DEPTH = 64, LANES = 4, FIFO_DEPTH = 4, LEARN_MODE = 1;
    parameter logic [224:0] LEARN_MASK = '0;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg         rst_n = 1'b0;
    reg         host_in_valid = 1'b0;
    reg  [47:0] host_in_flit = 48'd0;
    wire        host_in_ready;
    wire        host_out_valid;
    reg         host_out_ready = 1'b0;
    wire [47:0] host_out_flit;
    reg         step = 1'b0;
    wire        step_done, t_parity, err, busy, drained;
    wire [7:0]  drop_cnt;

    spikemesh_top #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS),
                    .W_BITS(W_BITS), .FANOUT_DEPTH(FANOUT_DEPTH), .LANES(LANES), .FIFO_DEPTH(FIFO_DEPTH),
                    .LEARN_MASK(LEARN_MASK), .LEARN_MODE(LEARN_MODE)) dut (
        .clk(clk), .rst_n(rst_n),
        .host_in_valid(host_in_valid), .host_in_ready(host_in_ready), .host_in_flit(host_in_flit),
        .host_out_valid(host_out_valid), .host_out_ready(host_out_ready), .host_out_flit(host_out_flit),
        .step(step), .step_done(step_done), .t_parity(t_parity),
        .err(err), .drop_cnt(drop_cnt), .busy(busy), .drained(drained));

    logic [47:0] inq [$];
    integer fi, fo, r, op, bp_pct;
    longint unsigned max_wait;
    reg [63:0] arg;
    reg [31:0] rs;
    reg pulse_step, running;
    reg s_done, s_drained, s_hov;
    longint unsigned ncycles, c0;

    function automatic [31:0] xs(input [31:0] x);
        reg [31:0] y;
        y = x ^ (x << 13); y = y ^ (y >> 17); y = y ^ (y << 5);
        return y;
    endfunction

    // One clock: drive at negedge, sample 1 ns later; sampled handshakes
    // complete at the next posedge.
    task automatic cycle();
        reg drive, ordy;
        @(negedge clk);
        step = pulse_step; pulse_step = 1'b0;
        drive = (inq.size() != 0);
        host_in_valid = drive;
        if (drive) host_in_flit = inq[0];
        rs = xs(rs);
        ordy = (rs % 100) >= bp_pct;
        host_out_ready = ordy;
        #1;
        ncycles++;
        if (drive && host_in_ready) void'(inq.pop_front());
        if (ordy && host_out_valid) $fdisplay(fo, "F %012h", host_out_flit);
        s_done = step_done; s_drained = drained; s_hov = host_out_valid;
    endtask

    task automatic wait_drained();
        integer cnt;
        cnt = 0;
        while (cnt < 2) begin
            cycle();
            cnt = (inq.size() == 0 && s_drained && !s_hov) ? cnt + 1 : 0;
            if (ncycles - c0 > max_wait) $fatal(1, "tb_sys: drain timeout");
        end
    endtask

    task automatic done();
        $fdisplay(fo, "D %0d %0d %0d", ncycles - c0, err, drop_cnt);
        $fflush(fo);
    endtask

    initial begin
        string cf, of;
        integer seed;
        if (!$value$plusargs("cmd=%s", cf)) $fatal(1, "need +cmd=");
        if (!$value$plusargs("out=%s", of)) $fatal(1, "need +out=");
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        if (!$value$plusargs("bp=%d", bp_pct)) bp_pct = 0;
        if (!$value$plusargs("maxwait=%d", max_wait)) max_wait = 20000000;
        rs = 32'h9E3779B9 ^ seed;
        pulse_step = 1'b0; ncycles = 0;
        fi = $fopen(cf, "r");
        fo = $fopen(of, "w");
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        c0 = ncycles;
        wait_drained();
        done();                       // handshake: reset finished
        running = 1'b1;
        while (running) begin
            r = $fscanf(fi, "%d %h", op, arg);   // no trailing \n: it would block on a pipe
            if (r != 2) running = 1'b0;
            else if (op == 0) inq.push_back(arg[47:0]);
            else if (op == 1) begin
                c0 = ncycles;
                while (inq.size() != 0) cycle();
                pulse_step = 1'b1;
                cycle();
                while (!s_done) begin
                    cycle();
                    if (ncycles - c0 > max_wait) $fatal(1, "tb_sys: step_done timeout");
                end
                wait_drained();
                done();
            end else if (op == 2) begin
                c0 = ncycles;
                wait_drained();
                done();
            end else running = 1'b0;
        end
        $fclose(fo);
        $finish;
    end
endmodule

`default_nettype wire
