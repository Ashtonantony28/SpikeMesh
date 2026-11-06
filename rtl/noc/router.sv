// SpikeMesh 5-port mesh router (SPEC §4, §8).
//
// Coordinates: core (x, y); x grows to the EAST, y grows to the NORTH.
//   N port <-> (x, y+1)   E port <-> (x+1, y)
//   S port <-> (x, y-1)   W port <-> (x-1, y)
// Port index: 0 = N, 1 = E, 2 = S, 3 = W, 4 = L (local core).
// "Input port N" carries flits arriving FROM the north neighbour.
//
// Flow control:
//   * N/E/S/W: credit based. out_valid[p] is a one-cycle flit pulse; the
//     sender holds a credit counter (initialised to FIFO_DEPTH = the
//     receiver's FIFO depth) and the receiver pulses credit_out[p] for one
//     cycle per flit it pops. Mesh-edge ports used for the host are adapted to
//     valid/ready in mesh.sv (noc_host_in / noc_host_out).
//   * L: plain valid/ready in both directions.
//
// Routing (dimension-order XY, computed on the FIFO head):
//   dest_x == 15 (host): west while x > 0, then south while y > 0, then out
//     of the W port of (0,0).
//   dest_x/dest_y outside the mesh (dest_x in [MESH_X, 14] or
//     dest_y >= MESH_Y, dest_x != 15): DROPPED at the first router that
//     sees it, the sticky `err` flag is set and `drop_cnt` increments
//     (saturating). The mesh never hangs on such a flit.
//   otherwise: E/W until x matches, then N/S until y matches, then L.
//
// Pipeline: a flit written into an input FIFO in cycle c is arbitrated in
// c+1 and sits in that output's link register in c+2 (i.e. 2 cycles/hop).
// Output arbitration: per output port, round robin over the 5 inputs; the
// pointer moves to one past the input that was granted.
module router #(
    parameter int X          = 0,
    parameter int Y          = 0,
    parameter int MESH_X     = 2,
    parameter int MESH_Y     = 2,
    parameter int FIFO_DEPTH = 4
) (
    input  logic            clk,
    input  logic            rst_n,
    // ---- input side (flits entering this router) ----
    input  logic [4:0]      in_valid,
    input  logic [5*48-1:0] in_flit,      // port p at [p*48 +: 48]
    output logic            l_in_ready,   // local input (valid/ready)
    output logic [3:0]      credit_out,   // credit return to N/E/S/W upstream
    // ---- output side (flits leaving this router) ----
    output logic [4:0]      out_valid,
    output logic [5*48-1:0] out_flit,
    input  logic            l_out_ready,  // local output (valid/ready)
    input  logic [3:0]      credit_in,    // credits from N/E/S/W downstream
    // ---- status ----
    output logic            busy,         // any FIFO non-empty or link register full
    output logic            err,          // sticky: dropped flit or protocol violation
    output logic [7:0]      drop_cnt      // saturating count of dropped flits
);
    localparam int PN = 0, PE = 1, PS = 2, PW = 3, PL = 4, PDROP = 5;
    localparam int CW = $clog2(FIFO_DEPTH + 1);
    localparam logic [3:0] HOST_X = 4'd15;

    // ------------------------------------------------------------------
    // Input FIFOs
    // ------------------------------------------------------------------
`ifdef FORMAL
    (* anyconst *) logic [47:0] fv_tag;     // the one flit value tracked by the tag proof
    logic [CW-1:0] fv_tcnt [5];
`endif
    logic [47:0]   head  [5];
    /* verilator lint_off UNUSEDSIGNAL */
    logic [CW-1:0] fcount[5];   // used by formal properties only
    logic [4:0]    fovf;        // [4] unused: local port is valid/ready
    /* verilator lint_on UNUSEDSIGNAL */
    logic [4:0]    fempty, ffull, fwr, pop;

    genvar gi;
    generate
        for (gi = 0; gi < 5; gi++) begin : g_in
            assign fwr[gi] = (gi == PL) ? (in_valid[gi] && !ffull[gi]) : in_valid[gi];
            noc_fifo #(.W(48), .DEPTH(FIFO_DEPTH)) u_fifo (
                .clk(clk), .rst_n(rst_n),
                .wr(fwr[gi]), .wdata(in_flit[gi*48 +: 48]),
                .rd(pop[gi]), .head(head[gi]), .count(fcount[gi]),
                .full(ffull[gi]), .empty(fempty[gi]), .ovf(fovf[gi])
`ifdef FORMAL
                , .fv_tag(fv_tag), .fv_tag_cnt(fv_tcnt[gi])
`endif
            );
        end
    endgenerate
    assign l_in_ready = !ffull[PL];

    // ------------------------------------------------------------------
    // Route computation (pure function of the head flit)
    // ------------------------------------------------------------------
    /* verilator lint_off UNSIGNED */
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic [2:0] route(input logic [47:0] f);
        logic [3:0] dx, dy;
        dx = f[45:42];
        dy = f[41:38];
        if (dx == HOST_X) begin
            if (X > 0)      route = 3'(PW);
            else if (Y > 0) route = 3'(PS);
            else            route = 3'(PW);
        end else if ({28'd0, dx} >= MESH_X || {28'd0, dy} >= MESH_Y) begin
            route = 3'(PDROP);
        end else if ({28'd0, dx} > X) route = 3'(PE);
        else if ({28'd0, dx} < X)     route = 3'(PW);
        else if ({28'd0, dy} > Y)     route = 3'(PN);
        else if ({28'd0, dy} < Y)     route = 3'(PS);
        else                          route = 3'(PL);
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */
    /* verilator lint_on UNSIGNED */

    logic [2:0] rt [5];
    logic [4:0] req [5];   // req[o][i]: input i's head wants output o
    logic [4:0] drop;
    always @(*) begin
        for (int o = 0; o < 5; o++) req[o] = '0;
        for (int i = 0; i < 5; i++) begin
            rt[i]   = route(head[i]);
            drop[i] = !fempty[i] && (rt[i] == 3'(PDROP));
            for (int o = 0; o < 5; o++)
                if (!fempty[i] && rt[i] == 3'(o)) req[o][i] = 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // Output side: credit counters, link registers, round-robin arbiters
    // ------------------------------------------------------------------
    logic [CW-1:0] credits [4];
    logic [4:0]    oreg_v;
    logic [47:0]   oreg_f [5];
    logic [2:0]    rr     [5];
    logic [4:0]    can_send;
    logic [4:0]    send;
    logic [2:0]    gsel   [5];      // granted input per output
    logic [4:0]    gnt    [5];      // gnt[o][i]

    always @(*) begin
        for (int o = 0; o < 4; o++) can_send[o] = (credits[o] != '0);
        can_send[PL] = !oreg_v[PL] || l_out_ready;
    end

    always @(*) begin
        for (int o = 0; o < 5; o++) begin
            logic       found;
            logic [3:0] sum;   // 4 bits: rr (<=4) + k (<=4) can reach 8
            logic [2:0] idx;
            found   = 1'b0;
            gsel[o] = '0;
            gnt[o]  = '0;
            for (int k = 0; k < 5; k++) begin
                sum = {1'b0, rr[o]} + 4'(k);
                idx = (sum >= 4'd5) ? 3'(sum - 4'd5) : sum[2:0];
                if (!found && req[o][idx]) begin
                    found   = 1'b1;
                    gsel[o] = idx;
                end
            end
            send[o] = found && can_send[o];
            if (send[o]) gnt[o][gsel[o]] = 1'b1;
        end
        for (int i = 0; i < 5; i++)
            pop[i] = drop[i] | gnt[0][i] | gnt[1][i] | gnt[2][i] | gnt[3][i] | gnt[4][i];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int o = 0; o < 4; o++) credits[o] <= CW'(FIFO_DEPTH);
            for (int o = 0; o < 5; o++) rr[o] <= '0;
            oreg_v     <= '0;
            credit_out <= '0;
            err        <= 1'b0;
            drop_cnt   <= '0;
        end else begin
            for (int o = 0; o < 4; o++) begin
                credits[o] <= credits[o] - CW'(send[o]) + CW'(credit_in[o]);
                oreg_v[o]  <= send[o];
            end
            if (send[PL])         oreg_v[PL] <= 1'b1;
            else if (l_out_ready) oreg_v[PL] <= 1'b0;
            for (int o = 0; o < 5; o++)
                if (send[o]) rr[o] <= (gsel[o] == 3'd4) ? 3'd0 : gsel[o] + 3'd1;
            credit_out <= pop[3:0];
            if (|drop || |fovf[3:0]) err <= 1'b1;
            if (|drop)
                drop_cnt <= (drop_cnt > 8'hFF - 8'($countones(drop))) ? 8'hFF
                                                                    : drop_cnt + 8'($countones(drop));
        end
    end

    always_ff @(posedge clk) begin
        for (int o = 0; o < 5; o++)
            if (send[o]) oreg_f[o] <= head[gsel[o]];
    end

    generate
        for (gi = 0; gi < 5; gi++) begin : g_out
            assign out_valid[gi]          = oreg_v[gi];
            assign out_flit[gi*48 +: 48] = oreg_f[gi];
        end
    endgenerate

    assign busy = !(&fempty) || (|oreg_v);

`ifdef FORMAL
    // ==================================================================
    // Formal properties (formal/run.sh). Environment assumptions are made
    // only at this module's ports, so the proofs hold for any neighbour that
    // obeys the credit / valid-ready protocol.
    // ==================================================================
    logic f_past_valid = 1'b0;
    always @(posedge clk) f_past_valid <= 1'b1;
    always @(*) assume (rst_n == f_past_valid);   // reset in cycle 0 only

    // ---- independent reference XY routing (written differently on purpose)
    function automatic logic [2:0] ref_route(input logic [47:0] f);
        integer dx, dy;
        dx = {28'd0, f[45:42]};
        dy = {28'd0, f[41:38]};
        if (dx == 15)                          ref_route = (X == 0 && Y != 0) ? 3'(PS) : 3'(PW);
        else if (dx >= MESH_X || dy >= MESH_Y) ref_route = 3'(PDROP);
        else if (dx != X)                      ref_route = (dx < X) ? 3'(PW) : 3'(PE);
        else if (dy != Y)                      ref_route = (dy < Y) ? 3'(PS) : 3'(PN);
        else                                   ref_route = 3'(PL);
    endfunction

    // ---- environment: credit protocol on N/E/S/W ----
    // up_owed[i]: flits the upstream neighbour sent into input i minus credits
    // it got back. It may only send while up_owed < FIFO_DEPTH.
    // dn_owed[o]: flits sent on output o minus credits received. The
    // downstream may only return a credit it owes (any delay, any order).
    logic [CW-1:0] up_owed [4];
    logic [CW-1:0] dn_owed [4];
    always @(*) begin
        for (int i = 0; i < 4; i++) begin
            assume (!(in_valid[i] && up_owed[i] == CW'(FIFO_DEPTH)));
            assume (!(credit_in[i] && dn_owed[i] == '0));
        end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin up_owed[i] <= '0; dn_owed[i] <= '0; end
        end else begin
            for (int i = 0; i < 4; i++) begin
                up_owed[i] <= up_owed[i] + CW'(in_valid[i]) - CW'(credit_out[i]);
                dn_owed[i] <= dn_owed[i] + CW'(out_valid[i]) - CW'(credit_in[i]);
            end
        end
    end

    // ---- conservation (flit counting) ----
    localparam int FW = 8;
    logic [FW-1:0] f_inflight, f_held;
    logic [2:0]    f_acc, f_dep, f_drp;
    always @(*) begin
        f_acc = '0; f_dep = '0; f_drp = '0; f_held = '0;
        for (int i = 0; i < 5; i++) begin
            f_acc  = f_acc + 3'(fwr[i] && (!ffull[i] || pop[i]));
            f_drp  = f_drp + 3'(drop[i]);
            f_held = f_held + FW'(fcount[i]) + FW'(oreg_v[i]);
        end
        for (int o = 0; o < 4; o++) f_dep = f_dep + 3'(oreg_v[o]);
        f_dep = f_dep + 3'(oreg_v[PL] && l_out_ready);
    end
    always @(posedge clk) begin
        if (!rst_n) f_inflight <= '0;
        else        f_inflight <= f_inflight + FW'(f_acc) - FW'(f_dep) - FW'(f_drp);
    end

    // ---- tagged flit: one arbitrary flit value fv_tag, injected at most once
    logic [1:0] t_in, t_out, t_drop;
    logic [2:0] t_in_now, t_out_now, t_drop_now;
    logic [3:0] t_held;
    always @(*) begin
        t_in_now = '0; t_out_now = '0; t_drop_now = '0; t_held = '0;
        for (int i = 0; i < 5; i++) begin
            if (fwr[i] && (!ffull[i] || pop[i]) && in_flit[i*48 +: 48] == fv_tag) t_in_now = t_in_now + 3'd1;
            if (drop[i] && head[i] == fv_tag) t_drop_now = t_drop_now + 3'd1;
            t_held = t_held + 4'(fv_tcnt[i]) + 4'(oreg_v[i] && oreg_f[i] == fv_tag);
        end
        for (int o = 0; o < 4; o++) if (oreg_v[o] && oreg_f[o] == fv_tag) t_out_now = t_out_now + 3'd1;
        if (oreg_v[PL] && l_out_ready && oreg_f[PL] == fv_tag) t_out_now = t_out_now + 3'd1;
        assume (2'(t_in_now) + t_in <= 2'd1 && t_in_now <= 3'd1);
    end
    always @(posedge clk) begin
        if (!rst_n) begin t_in <= '0; t_out <= '0; t_drop <= '0; end
        else begin
            t_in   <= t_in + 2'(t_in_now);
            t_out  <= t_out + 2'(t_out_now);
            t_drop <= t_drop + 2'(t_drop_now);
        end
    end

    always @(*) begin
        if (rst_n) begin
            // P1 credit counters never exceed FIFO_DEPTH and exactly mirror the
            //    downstream FIFO space (=> downstream FIFO never overflows)
            for (int o = 0; o < 4; o++) begin
                assert (credits[o] <= CW'(FIFO_DEPTH));
                assert ({1'b0, credits[o]} + {1'b0, dn_owed[o]} == (CW+1)'(FIFO_DEPTH));
            end
            // P2 input FIFO occupancy + returned-but-unseen credit == upstream debt,
            //    so a credit-respecting upstream can never overflow an input FIFO
            for (int i = 0; i < 4; i++) begin
                assert (fcount[i] <= CW'(FIFO_DEPTH));
                assert ({1'b0, fcount[i]} + (CW+1)'(credit_out[i]) == {1'b0, up_owed[i]});
                assert (!fovf[i]);
            end
            // P3 conservation: accepted - departed - dropped == flits held
            assert (f_inflight == f_held);
            // P4 tagged flit: never duplicated, never lost or corrupted
            assert ({2'b0, t_in} == {2'b0, t_out} + {2'b0, t_drop} + t_held);
            assert (t_out + t_drop <= 2'd1);
            // P5 a departing tagged flit uses the reference XY port; drops only off-mesh
            for (int o = 0; o < 4; o++)
                if (oreg_v[o] && oreg_f[o] == fv_tag) assert (ref_route(fv_tag) == 3'(o));
            if (oreg_v[PL] && oreg_f[PL] == fv_tag) assert (ref_route(fv_tag) == 3'(PL));
            if (t_drop_now != '0) assert (ref_route(fv_tag) == 3'(PDROP));
            // P6 each input is granted by at most one output; arbitration is
            //    work conserving (a requested output that can send, sends)
            for (int i = 0; i < 5; i++)
                assert ($countones({gnt[0][i], gnt[1][i], gnt[2][i], gnt[3][i], gnt[4][i], drop[i]}) <= 1);
            for (int o = 0; o < 5; o++)
                if (req[o] != '0 && can_send[o]) assert (send[o]);
            // P7 XY turn rule: a flit that arrived travelling in Y (input N or S)
            //    never turns into X, except host-bound flits leaving (0,0) west
            for (int o = PE; o <= PW; o += 2)
                if (gnt[o][PN] || gnt[o][PS])
                    assert (o == PW && X == 0 && Y == 0 && head[gnt[o][PN] ? PN : PS][45:42] == HOST_X);
            // P8 no U-turns / X reversal: a flit never leaves on the port it
            //    arrived on (except host->host loopback on (0,0)'s W port)
            for (int o = 0; o < 4; o++)
                if (gnt[o][o]) assert (o == PW && X == 0 && Y == 0);
            // P9 edge ports without a neighbour never fire (host port excepted)
            if (Y == MESH_Y - 1) assert (!oreg_v[PN]);
            if (Y == 0)          assert (!oreg_v[PS]);
            if (X == MESH_X - 1) assert (!oreg_v[PE]);
            if (X == 0 && Y != 0) assert (!oreg_v[PW]);
            // P10 busy is exactly "some flit held"
            assert (busy == (f_held != '0));
            // round-robin pointers stay in range
            for (int o = 0; o < 5; o++) assert (rr[o] <= 3'd4);
        end
    end
`endif
endmodule
