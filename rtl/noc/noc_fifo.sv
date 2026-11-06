// SpikeMesh NoC input FIFO (SPEC §8). Synchronous, first-word-fall-through:
// `head` is valid combinationally whenever `count != 0`.
// State is (rd, count) only; the write pointer is derived, which keeps the
// state space free of unreachable pointer combinations (helps k-induction).
// Writing a full FIFO is a protocol violation: the write is ignored and
// `ovf` pulses (the router turns it into a sticky error; formal proves it
// never happens under the credit protocol).
module noc_fifo #(
    parameter int W     = 48,
    parameter int DEPTH = 4
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       wr,
    input  logic [W-1:0]               wdata,
    input  logic                       rd,
    output logic [W-1:0]               head,
    output logic [$clog2(DEPTH+1)-1:0] count,
    output logic                       full,
    output logic                       empty,
    output logic                       ovf
`ifdef FORMAL
    ,
    input  logic [W-1:0]               fv_tag,      // formal: flit value to track
    output logic [$clog2(DEPTH+1)-1:0] fv_tag_cnt   // formal: copies of fv_tag held
`endif
);
    localparam int PW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam int CW = $clog2(DEPTH + 1);

    logic [W-1:0]  mem [DEPTH];
    logic [PW-1:0] rptr;
    logic [PW-1:0] wptr;
    logic          do_rd, do_wr;

    assign empty = (count == '0);
    assign full  = (count == CW'(DEPTH));
    assign head  = mem[rptr];
    assign do_rd = rd && !empty;
    assign do_wr = wr && (!full || do_rd);
    assign ovf   = wr && full && !do_rd;

    function automatic logic [PW-1:0] wrap(input logic [PW:0] v);
        wrap = (v >= (PW+1)'(DEPTH)) ? PW'(v - (PW+1)'(DEPTH)) : v[PW-1:0];
    endfunction

    always @(*) wptr = wrap({1'b0, rptr} + (PW+1)'(count));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rptr  <= '0;
            count <= '0;
        end else begin
            if (do_rd) rptr <= wrap({1'b0, rptr} + (PW+1)'(1));
            count <= count + CW'(do_wr) - CW'(do_rd);
        end
    end

    always_ff @(posedge clk) begin
        if (do_wr) mem[wptr] <= wdata;
    end
`ifdef FORMAL
    // copies of the tracked flit among the valid entries
    always @(*) begin
        logic [PW-1:0] p;
        fv_tag_cnt = '0;
        for (int k = 0; k < DEPTH; k++) begin
            p = wrap({1'b0, rptr} + (PW+1)'(k));
            if (CW'(k) < count && mem[p] == fv_tag) fv_tag_cnt = fv_tag_cnt + CW'(1);
        end
    end
    always @(*) begin
        assert (count <= CW'(DEPTH));
        assert (rptr < PW'(DEPTH) || DEPTH == 1);
    end
`endif
endmodule
