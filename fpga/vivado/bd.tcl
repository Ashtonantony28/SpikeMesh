# Block design: Zynq UltraScale+ PS + AXI DMA (simple mode) + SpikeMesh.
#
#   PS M_AXI_HPM0_FPD -> SmartConnect -> { AXI DMA S_AXI_LITE, spikemesh_0 s_axi }
#   AXI DMA M_AXIS_MM2S -> spikemesh_0 s_axis     (host -> chip words)
#   spikemesh_0 m_axis  -> AXI DMA S_AXIS_S2MM     (chip -> host words, TLAST on events)
#   AXI DMA M_AXI_MM2S/S2MM -> SmartConnect -> PS S_AXI_HP0_FPD
#   pl_clk0 (100 MHz) drives everything; proc_sys_reset makes the resets.
#   uart_rx / uart_tx are BD ports (PMOD pins, see kv260.xdc).
# PYNQ names: overlay.axi_dma_0, overlay.spikemesh_0 (host/fpga_backend.py).

proc create_spikemesh_bd {name mesh_x mesh_y host_sel baud have_board} {
    create_bd_design $name
    current_bd_design $name

    # ---- Zynq UltraScale+ PS
    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
    if {$have_board} {
        apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
            -config {apply_board_preset "1"} $ps
    }
    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {64} \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
        CONFIG.PSU__USE__IRQ0 {1} \
    ] $ps

    # ---- reset
    set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]

    # ---- AXI DMA, simple (register) mode, 64-bit streams
    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0]
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_sg_length_width {26} \
        CONFIG.c_addr_width {64} \
        CONFIG.c_m_axi_mm2s_data_width {64} \
        CONFIG.c_m_axis_mm2s_tdata_width {64} \
        CONFIG.c_m_axi_s2mm_data_width {64} \
        CONFIG.c_s_axis_s2mm_tdata_width {64} \
        CONFIG.c_mm2s_burst_size {16} \
        CONFIG.c_s2mm_burst_size {16} \
    ] $dma

    # ---- SpikeMesh (module reference to the Verilog wrapper)
    set sm [create_bd_cell -type module -reference kv260_top_wrap spikemesh_0]
    set_property -dict [list \
        CONFIG.MESH_X $mesh_x CONFIG.MESH_Y $mesh_y \
        CONFIG.HOST_SEL_RST $host_sel CONFIG.BAUD $baud CONFIG.CLK_HZ {100000000} \
    ] $sm

    # ---- interconnect
    set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_ctrl]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc_ctrl
    set smc_data [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_data]
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $smc_data
    set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat irq_concat]
    set_property -dict [list CONFIG.NUM_PORTS {2}] $irq

    connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $smc_ctrl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $smc_ctrl/M00_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
    connect_bd_intf_net [get_bd_intf_pins $smc_ctrl/M01_AXI] [get_bd_intf_pins $sm/s_axi]
    connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $smc_data/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $smc_data/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins $smc_data/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $sm/s_axis]
    connect_bd_intf_net [get_bd_intf_pins $sm/m_axis] [get_bd_intf_pins $dma/S_AXIS_S2MM]

    # ---- clocks and resets
    set clk [get_bd_pins $ps/pl_clk0]
    connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk] [get_bd_pins $ps/saxihp0_fpd_aclk] \
        [get_bd_pins $rst/slowest_sync_clk] [get_bd_pins $dma/s_axi_lite_aclk] \
        [get_bd_pins $dma/m_axi_mm2s_aclk] [get_bd_pins $dma/m_axi_s2mm_aclk] \
        [get_bd_pins $smc_ctrl/aclk] [get_bd_pins $smc_data/aclk] [get_bd_pins $sm/clk]
    connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
    connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $dma/axi_resetn] \
        [get_bd_pins $smc_ctrl/aresetn] [get_bd_pins $smc_data/aresetn] [get_bd_pins $sm/rst_n]

    # ---- interrupts
    connect_bd_net [get_bd_pins $dma/mm2s_introut] [get_bd_pins $irq/In0]
    connect_bd_net [get_bd_pins $dma/s2mm_introut] [get_bd_pins $irq/In1]
    connect_bd_net [get_bd_pins $irq/dout] [get_bd_pins $ps/pl_ps_irq0]

    # ---- external UART pins (PMOD), status pins left internal
    create_bd_port -dir I uart_rx
    create_bd_port -dir O uart_tx
    connect_bd_net [get_bd_ports uart_rx] [get_bd_pins $sm/uart_rx]
    connect_bd_net [get_bd_ports uart_tx] [get_bd_pins $sm/uart_tx]

    assign_bd_address
    validate_bd_design
    save_bd_design
}
