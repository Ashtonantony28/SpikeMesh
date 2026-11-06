// Host-side adapters between the valid/ready host stream (SPEC §7/§8) and
// the credit-based W port of router (0,0).

// Mesh -> host. A FIFO_DEPTH-deep receive FIFO that returns credits to the
// router's W output; the host drains it with valid/ready.
module noc_host_out #(
    parameter int FIFO_DEPTH = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        link_valid,   // from router W output
    input  logic [47:0] link_flit,
    output logic        credit_out,   // to router credit_in[W]
    output logic        host_valid,
    input  logic        host_ready,
    output logic [47:0] host_flit,
    output logic        err           // sticky: FIFO overflow (never, if credits honoured)
);
    localparam int CW = $clog2(FIFO_DEPTH + 1);
    logic empty, ovf, pop;
    /* verilator lint_off UNUSEDSIGNAL */
    logic full;
    logic [CW-1:0] count;
    /* verilator lint_on UNUSEDSIGNAL */
    assign host_valid = !empty;
    assign pop        = host_valid && host_ready;
    noc_fifo #(.W(48), .DEPTH(FIFO_DEPTH)) u_fifo (
        .clk(clk), .rst_n(rst_n), .wr(link_valid), .wdata(link_flit),
        .rd(pop), .head(host_flit), .count(count), .full(full), .empty(empty), .ovf(ovf)
    );
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            credit_out <= 1'b0;
            err        <= 1'b0;
        end else begin
            credit_out <= pop;
            if (ovf) err <= 1'b1;
        end
    end
endmodule
