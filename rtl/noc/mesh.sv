// SpikeMesh MESH_X x MESH_Y router mesh (SPEC §4, §8).
// Router/core index n = y*MESH_X + x; y grows to the north (see router.sv).
// The host attaches to the W port of router (0,0) through valid/ready
// adapters (noc_host_in / noc_host_out). All other mesh-edge inputs are tied
// off; XY routing never targets an edge output that has no neighbour
// (proven per router in formal/), and an out-of-mesh destination is dropped
// with a sticky error at the first router that sees it.
module mesh #(
    parameter int MESH_X     = 2,
    parameter int MESH_Y     = 2,
    parameter int FIFO_DEPTH = 4,
    localparam int NC        = MESH_X * MESH_Y
) (
    input  logic             clk,
    input  logic             rst_n,
    // host port
    input  logic             host_in_valid,
    output logic             host_in_ready,
    input  logic [47:0]      host_in_flit,
    output logic             host_out_valid,
    input  logic             host_out_ready,
    output logic [47:0]      host_out_flit,
    // core -> router (core out_*)
    input  logic [NC-1:0]    core_out_valid,
    output logic [NC-1:0]    core_out_ready,
    input  logic [NC*48-1:0] core_out_flit,
    // router -> core (core in_*)
    output logic [NC-1:0]    core_in_valid,
    input  logic [NC-1:0]    core_in_ready,
    output logic [NC*48-1:0] core_in_flit,
    // status
    output logic [NC-1:0]    router_busy,
    output logic             busy,      // OR of all router busy
    output logic             err,       // sticky: any drop or overflow
    output logic [7:0]       drop_cnt   // sum of per-router drop counts (saturating)
);
    localparam int PN = 0, PE = 1, PS = 2, PW = 3, PL = 4;

    logic [NC*5-1:0]    ov;     // router out_valid, port p of router n at n*5+p
    /* verilator lint_off UNUSEDSIGNAL */
    logic [NC*5*48-1:0] of;     // router out_flit (mesh-edge outputs unused)
    /* verilator lint_on UNUSEDSIGNAL */
    logic [NC*4-1:0]    co;     // router credit_out
    logic [NC-1:0]      rerr;
    logic [NC*8-1:0]    rdrop;

    logic        hin_v, hout_credit, hout_err;
    logic [47:0] hin_f;
    logic        hin_credit;

    genvar gx, gy;
    generate
        for (gy = 0; gy < MESH_Y; gy++) begin : g_y
            for (gx = 0; gx < MESH_X; gx++) begin : g_x
                localparam int N  = gy * MESH_X + gx;
                localparam int NN = (gy + 1) * MESH_X + gx;
                localparam int NS = (gy - 1) * MESH_X + gx;
                localparam int NE = gy * MESH_X + gx + 1;
                localparam int NW = gy * MESH_X + gx - 1;
                logic [4:0]      iv;
                logic [5*48-1:0] ifl;
                logic [3:0]      ci;

                // ---- north neighbour (x, y+1): its S output / S credit ----
                if (gy < MESH_Y - 1) begin : g_n
                    assign iv[PN]           = ov[NN*5 + PS];
                    assign ifl[PN*48 +: 48] = of[(NN*5 + PS)*48 +: 48];
                    assign ci[PN]           = co[NN*4 + PS];
                end else begin : g_n_edge
                    assign iv[PN]           = 1'b0;
                    assign ifl[PN*48 +: 48] = '0;
                    assign ci[PN]           = 1'b0;
                end
                // ---- south neighbour (x, y-1) ----
                if (gy > 0) begin : g_s
                    assign iv[PS]           = ov[NS*5 + PN];
                    assign ifl[PS*48 +: 48] = of[(NS*5 + PN)*48 +: 48];
                    assign ci[PS]           = co[NS*4 + PN];
                end else begin : g_s_edge
                    assign iv[PS]           = 1'b0;
                    assign ifl[PS*48 +: 48] = '0;
                    assign ci[PS]           = 1'b0;
                end
                // ---- east neighbour (x+1, y) ----
                if (gx < MESH_X - 1) begin : g_e
                    assign iv[PE]           = ov[NE*5 + PW];
                    assign ifl[PE*48 +: 48] = of[(NE*5 + PW)*48 +: 48];
                    assign ci[PE]           = co[NE*4 + PW];
                end else begin : g_e_edge
                    assign iv[PE]           = 1'b0;
                    assign ifl[PE*48 +: 48] = '0;
                    assign ci[PE]           = 1'b0;
                end
                // ---- west neighbour (x-1, y), or the host at (0,0) ----
                if (gx > 0) begin : g_w
                    assign iv[PW]           = ov[NW*5 + PE];
                    assign ifl[PW*48 +: 48] = of[(NW*5 + PE)*48 +: 48];
                    assign ci[PW]           = co[NW*4 + PE];
                end else if (gy == 0) begin : g_w_host
                    assign iv[PW]           = hin_v;
                    assign ifl[PW*48 +: 48] = hin_f;
                    assign ci[PW]           = hout_credit;
                end else begin : g_w_edge
                    assign iv[PW]           = 1'b0;
                    assign ifl[PW*48 +: 48] = '0;
                    assign ci[PW]           = 1'b0;
                end
                // ---- local core ----
                assign iv[PL]           = core_out_valid[N];
                assign ifl[PL*48 +: 48] = core_out_flit[N*48 +: 48];

                router #(
                    .X(gx), .Y(gy), .MESH_X(MESH_X), .MESH_Y(MESH_Y), .FIFO_DEPTH(FIFO_DEPTH)
                ) u_router (
                    .clk(clk), .rst_n(rst_n),
                    .in_valid(iv), .in_flit(ifl),
                    .l_in_ready(core_out_ready[N]),
                    .credit_out(co[N*4 +: 4]),
                    .out_valid(ov[N*5 +: 5]), .out_flit(of[N*5*48 +: 5*48]),
                    .l_out_ready(core_in_ready[N]),
                    .credit_in(ci),
                    .busy(router_busy[N]), .err(rerr[N]), .drop_cnt(rdrop[N*8 +: 8])
                );
                assign core_in_valid[N]          = ov[N*5 + PL];
                assign core_in_flit[N*48 +: 48]  = of[(N*5 + PL)*48 +: 48];
            end
        end
    endgenerate

    assign hin_credit = co[0*4 + PW];

    noc_host_in #(.FIFO_DEPTH(FIFO_DEPTH)) u_host_in (
        .clk(clk), .rst_n(rst_n),
        .host_valid(host_in_valid), .host_ready(host_in_ready), .host_flit(host_in_flit),
        .link_valid(hin_v), .link_flit(hin_f), .credit_in(hin_credit)
    );
    noc_host_out #(.FIFO_DEPTH(FIFO_DEPTH)) u_host_out (
        .clk(clk), .rst_n(rst_n),
        .link_valid(ov[0*5 + PW]), .link_flit(of[(0*5 + PW)*48 +: 48]),
        .credit_out(hout_credit),
        .host_valid(host_out_valid), .host_ready(host_out_ready), .host_flit(host_out_flit),
        .err(hout_err)
    );

    always @(*) begin
        logic [11:0] sum;
        sum = '0;
        for (int n = 0; n < NC; n++) sum = sum + {4'd0, rdrop[n*8 +: 8]};
        drop_cnt = (sum > 12'd255) ? 8'hFF : sum[7:0];
    end
    assign busy = |router_busy;
    assign err  = (|rerr) | hout_err;
endmodule
