// FPGA drop-in for rtl/core/sram_1r1w.sv (same module name, ports and
// behaviour: one write port with per-lane mask, one registered read port held
// while re is low, read-during-write returns old data). Use this file INSTEAD
// of rtl/core/sram_1r1w.sv in FPGA builds. The only difference is a RAM-style
// attribute: large memories go to UltraRAM, which keeps a 4x4 mesh of
// N=A=256 cores inside the KV260's 144 BRAM36. Vivado honours ram_style;
// Yosys honours it for synth_xilinx -family xcup.
`default_nettype none

/* verilator lint_off DECLFILENAME */  // same module name as rtl/core/sram_1r1w.sv on purpose
module sram_1r1w #(
    parameter integer DEPTH = 256,
    parameter integer WIDTH = 16,
    parameter integer NLANE = 1,
    parameter integer AW    = (DEPTH > 1) ? $clog2(DEPTH) : 1,
    // memories of at least URAM_MIN_BITS bits and at least 4096 deep go to URAM
    parameter integer URAM_MIN_BITS = 262144
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
    integer i;
    generate
        if ((DEPTH * WIDTH >= URAM_MIN_BITS) && (DEPTH >= 4096)) begin : g_uram
            (* ram_style = "ultra" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
            always @(posedge clk) begin
                if (we) begin
                    for (i = 0; i < NLANE; i = i + 1)
                        if (wmask[i]) mem[waddr][i*LW +: LW] <= wdata[i*LW +: LW];
                end
                if (re) rdata <= mem[raddr];
            end
        end else begin : g_auto
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            always @(posedge clk) begin
                if (we) begin
                    for (i = 0; i < NLANE; i = i + 1)
                        if (wmask[i]) mem[waddr][i*LW +: LW] <= wdata[i*LW +: LW];
                end
                if (re) rdata <= mem[raddr];
            end
        end
    endgenerate
endmodule

`default_nettype wire
