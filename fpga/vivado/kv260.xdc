# SpikeMesh KV260 constraints.
# The clock is pl_clk0 from the PS (100 MHz). The zynq_ultra_ps_e IP constrains it.
#
# UART on PMOD connector J2 (3.3 V): pin 1 = uart_rx (FPGA input), pin 2 = uart_tx.
# Connect a 3.3 V USB-UART (e.g. Digilent PmodUSBUART or an FTDI cable) with its
# TXD to J2 pin 1 and RXD to J2 pin 2, plus GND.
# The pin numbers come from the KV260 carrier XDC (PMOD pins H12, E10, D10, C11,
# B10, E12, D11, B11 for J2 pins 1-4 and 7-10). CHECK them against the
# kv260 carrier XDC for your board revision before you build.
set_property PACKAGE_PIN H12 [get_ports uart_rx]
set_property PACKAGE_PIN E10 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports {uart_rx uart_tx}]
set_property PULLUP true [get_ports uart_rx]

# Asynchronous UART lines: uart_rx goes through a 2-flop synchronizer,
# and uart_tx is a slow registered output.
set_false_path -from [get_ports uart_rx]
set_false_path -to   [get_ports uart_tx]
