// Host-side adapters between the valid/ready host stream (SPEC §7/§8) and
// the credit-based W port of router (0,0).

// Host -> mesh. Holds a credit counter for router (0,0)'s W input FIFO.
// Combinational pass-through (no storage): host_in_ready = credits != 0.
module noc_host_in #(
    parameter int FIFO_DEPTH = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        host_valid,
    output logic        host_ready,
    input  logic [47:0] host_flit,
    output logic        link_valid,   // to router W input
    output logic [47:0] link_flit,
    input  logic        credit_in     // from router credit_out[W]
);
    localparam int CW = $clog2(FIFO_DEPTH + 1);
    logic [CW-1:0] credits;
    assign host_ready = (credits != '0);
    assign link_valid = host_valid && host_ready;
    assign link_flit  = host_flit;
    always_ff @(posedge clk) begin
        if (!rst_n) credits <= CW'(FIFO_DEPTH);
        else        credits <= credits - CW'(link_valid) + CW'(credit_in);
    end
`ifdef FORMAL
    always @(*) if (rst_n) assert (credits <= CW'(FIFO_DEPTH));
`endif
endmodule

