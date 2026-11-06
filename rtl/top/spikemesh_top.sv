// SpikeMesh chip top (SPEC §8): MESH_X x MESH_Y neurocores on a router mesh,
// tick controller, host port on the W side of core (0,0).
// Core index n = y*MESH_X + x, y grows to the north (rtl/noc/router.sv).
module spikemesh_top #(
    parameter int MESH_X         = 2,
    parameter int MESH_Y         = 2,
    parameter int N_NEURONS      = 256,
    parameter int N_AXONS        = 256,
    parameter int W_BITS         = 8,
    parameter int FANOUT_DEPTH   = 1024,
    parameter int LANES          = 1,
    parameter int FIFO_DEPTH     = 4,
    parameter int QUIESCE_CYCLES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        host_in_valid,
    output logic        host_in_ready,
    input  logic [47:0] host_in_flit,
    output logic        host_out_valid,
    input  logic        host_out_ready,
    output logic [47:0] host_out_flit,
    input  logic        step,
    output logic        step_done,
    output logic        t_parity,
    // debug / status (not in SPEC §8 table; safe to leave unconnected)
    output logic        err,         // sticky: NoC drop/overflow or step while running
    output logic [7:0]  drop_cnt,
    output logic        busy         // a STEP is running
);
    localparam int NC = MESH_X * MESH_Y;

    logic [NC-1:0]    c_out_valid, c_out_ready, c_in_valid, c_in_ready, c_busy;
    logic [NC*48-1:0] c_out_flit, c_in_flit;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [NC-1:0]    router_busy;
    logic             quiescent;
    /* verilator lint_on UNUSEDSIGNAL */
    logic             mesh_busy, mesh_err, step_err;
    logic             learn_start, update_start;

    mesh #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .FIFO_DEPTH(FIFO_DEPTH)) u_mesh (
        .clk(clk), .rst_n(rst_n),
        .host_in_valid(host_in_valid), .host_in_ready(host_in_ready), .host_in_flit(host_in_flit),
        .host_out_valid(host_out_valid), .host_out_ready(host_out_ready), .host_out_flit(host_out_flit),
        .core_out_valid(c_out_valid), .core_out_ready(c_out_ready), .core_out_flit(c_out_flit),
        .core_in_valid(c_in_valid), .core_in_ready(c_in_ready), .core_in_flit(c_in_flit),
        .router_busy(router_busy), .busy(mesh_busy), .err(mesh_err), .drop_cnt(drop_cnt)
    );

    genvar gx, gy;
    generate
        for (gy = 0; gy < MESH_Y; gy++) begin : g_y
            for (gx = 0; gx < MESH_X; gx++) begin : g_x
                localparam int N = gy * MESH_X + gx;
                neurocore #(
                    .X(gx), .Y(gy), .N_NEURONS(N_NEURONS), .N_AXONS(N_AXONS),
                    .W_BITS(W_BITS), .FANOUT_DEPTH(FANOUT_DEPTH), .LANES(LANES)
                ) u_core (
                    .clk(clk), .rst_n(rst_n),
                    .in_valid(c_in_valid[N]), .in_ready(c_in_ready[N]), .in_flit(c_in_flit[N*48 +: 48]),
                    .out_valid(c_out_valid[N]), .out_ready(c_out_ready[N]), .out_flit(c_out_flit[N*48 +: 48]),
                    .parity(t_parity), .learn_start(learn_start), .update_start(update_start),
                    .busy(c_busy[N])
                );
            end
        end
    endgenerate

    tick_ctrl #(.NC(NC), .QUIESCE_CYCLES(QUIESCE_CYCLES)) u_tick (
        .clk(clk), .rst_n(rst_n), .step(step),
        // a host flit waiting at the (credit-limited) host input counts as in flight
        .net_busy(mesh_busy | host_in_valid),
        .core_busy(c_busy),
        .learn_start(learn_start), .update_start(update_start), .step_done(step_done),
        .parity(t_parity), .quiescent(quiescent), .running(busy), .step_err(step_err)
    );

    assign err = mesh_err | step_err;
endmodule
