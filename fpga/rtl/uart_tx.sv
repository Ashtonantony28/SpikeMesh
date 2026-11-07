// 8N1 UART transmitter with valid/ready byte input.
`default_nettype none

module uart_tx #(
    parameter integer CLKS_PER_BIT = 108
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid,
    output wire       ready,
    input  wire [7:0] data,
    output reg        tx
);
    localparam integer CW = $clog2(CLKS_PER_BIT + 1);
    /* verilator lint_off UNUSEDSIGNAL */
    reg [9:0]    sh;          // {stop, data, start}, LSB first; sh[0] is already on the line
    /* verilator lint_on UNUSEDSIGNAL */
    reg [3:0]    left;        // bits left to send
    reg [CW-1:0] cnt;

    assign ready = (left == 4'd0);

    always @(posedge clk) begin
        if (!rst_n) begin
            sh <= 10'h3FF; left <= 4'd0; cnt <= '0; tx <= 1'b1;
        end else if (left == 4'd0) begin
            tx <= 1'b1;
            if (valid) begin
                sh   <= {1'b1, data, 1'b0};
                left <= 4'd10;
                cnt  <= '0;
                tx   <= 1'b0;          // start bit
            end
        end else if (cnt == CW'(CLKS_PER_BIT - 1)) begin
            cnt  <= '0;
            left <= left - 4'd1;
            sh   <= {1'b1, sh[9:1]};
            tx   <= (left == 4'd1) ? 1'b1 : sh[1];
        end else begin
            cnt <= cnt + 1'b1;
        end
    end
endmodule

`default_nettype wire
