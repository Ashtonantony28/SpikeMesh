// 8N1 UART receiver. CLKS_PER_BIT clock cycles per bit; samples mid-bit
// after a 2-flop synchronizer. A start bit that is not low at mid-bit is
// treated as a glitch; a missing stop bit sets frame_err for one cycle and
// the byte is dropped.
`default_nettype none

module uart_rx #(
    parameter integer CLKS_PER_BIT = 108
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg        valid,
    output reg  [7:0] data,
    output reg        frame_err
);
    localparam integer CW = $clog2(CLKS_PER_BIT + 1);
    reg [1:0]    sync;
    reg [1:0]    st;          // 0 idle, 1 start, 2 data, 3 stop
    reg [CW-1:0] cnt;
    reg [2:0]    bitn;
    reg [7:0]    sh;
    wire         rxs = sync[1];

    always @(posedge clk) begin
        if (!rst_n) begin
            sync <= 2'b11; st <= 2'd0; cnt <= '0; bitn <= 3'd0; sh <= 8'd0;
            valid <= 1'b0; data <= 8'd0; frame_err <= 1'b0;
        end else begin
            sync      <= {sync[0], rx};
            valid     <= 1'b0;
            frame_err <= 1'b0;
            case (st)
                2'd0: if (!rxs) begin st <= 2'd1; cnt <= '0; end
                2'd1: if (cnt == CW'(CLKS_PER_BIT / 2 - 1)) begin
                          cnt <= '0;
                          if (!rxs) begin st <= 2'd2; bitn <= 3'd0; end
                          else st <= 2'd0;
                      end else cnt <= cnt + 1'b1;
                2'd2: if (cnt == CW'(CLKS_PER_BIT - 1)) begin
                          cnt <= '0;
                          sh  <= {rxs, sh[7:1]};
                          if (bitn == 3'd7) st <= 2'd3;
                          bitn <= bitn + 3'd1;
                      end else cnt <= cnt + 1'b1;
                default: if (cnt == CW'(CLKS_PER_BIT - 1)) begin
                          cnt <= '0;
                          st  <= 2'd0;
                          if (rxs) begin valid <= 1'b1; data <= sh; end
                          else frame_err <= 1'b1;
                      end else cnt <= cnt + 1'b1;
            endcase
        end
    end
endmodule

`default_nettype wire
