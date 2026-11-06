// sram_1r1w: simple dual-port synchronous RAM (one write port, one read port,
// single clock). Registered read data, held while `re` is low. Optional
// per-lane write mask (NLANE equal lanes of WIDTH/NLANE bits).
//
// Read-during-write to the same address returns the OLD data (the core never
// relies on this: it forwards internally where the hazard can occur).
//
// This is the only memory primitive used by the neurocore. It infers block RAM
// on FPGA (Yosys synth_xilinx / Vivado: simple-dual-port BRAM with byte-write
// enables) and can be swapped for a 1R1W SRAM macro wrapper for ASIC.
// Contents are NOT reset (SPEC §5).
`default_nettype none

module sram_1r1w #(
    parameter integer DEPTH = 256,
    parameter integer WIDTH = 16,
    parameter integer NLANE = 1,
    parameter integer AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  wire              clk,
    input  wire              we,
    input  wire [NLANE-1:0]  wmask,
    input  wire [AW-1:0]     waddr,
    input  wire [WIDTH-1:0]  wdata,
    input  wire              re,
    input  wire [AW-1:0]     raddr,
    output reg  [WIDTH-1:0]  rdata
);
    localparam integer LW = WIDTH / NLANE;

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    integer i;
    always @(posedge clk) begin
        if (we) begin
            for (i = 0; i < NLANE; i = i + 1)
                if (wmask[i]) mem[waddr][i*LW +: LW] <= wdata[i*LW +: LW];
        end
        if (re) rdata <= mem[raddr];
    end
endmodule

`default_nettype wire
