# SpikeMesh on the Kria KV260: non-project-mode Vivado build.
#
#   cd fpga/vivado
#   vivado -mode batch -source build.tcl -tclargs [MESH_X] [MESH_Y] [HOST_SEL_RST] [BAUD] [LEARN_MASK] [LEARN_MODE]
#
# Defaults: 2 2 0 921600 0 1. LEARN_MASK is a decimal core mask (bit y*MESH_X+x),
# LEARN_MODE 1 = STDP, 2 = STDP + R-STDP. HOST_SEL_RST: 0 = AXI DMA path (PYNQ), 1 = UART on PMOD.
# Outputs go to build_<X>x<Y>/: spikemesh.bit, spikemesh.hwh (PYNQ),
# spikemesh.xsa, and utilization/timing reports.
# Written for Vivado 2022.2 or newer. This script has not been run in the
# cloud session that wrote it; see README.md.

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir .. ..]]
set mesh_x   2
set mesh_y   2
set host_sel 0
set baud     921600
if {$argc >= 1} { set mesh_x   [lindex $argv 0] }
if {$argc >= 2} { set mesh_y   [lindex $argv 1] }
if {$argc >= 3} { set host_sel [lindex $argv 2] }
if {$argc >= 4} { set baud     [lindex $argv 3] }
set learn_mask 0
set learn_mode 1
if {$argc >= 5} { set learn_mask [lindex $argv 4] }
if {$argc >= 6} { set learn_mode [lindex $argv 5] }

set part xck26-sfvc784-2LV-c
set suffix ""
if {$learn_mask != 0} { set suffix "_learn${learn_mask}m${learn_mode}" }
set out  [file join $script_dir "build_${mesh_x}x${mesh_y}${suffix}"]
file mkdir $out
puts "SpikeMesh build: mesh ${mesh_x}x${mesh_y}, HOST_SEL_RST=$host_sel, BAUD=$baud, LEARN_MASK=$learn_mask, LEARN_MODE=$learn_mode, part $part -> $out"

create_project -in_memory -part $part
set board [lindex [get_board_parts -quiet -latest_file_version {*kv260_som*}] 0]
if {$board ne ""} {
    puts "Using board part $board"
    set_property board_part $board [current_project]
} else {
    puts "WARNING: KV260 board files not installed; the PS is configured from the part only (see README)."
}
set_property target_language Verilog [current_project]

# ---- RTL (the FPGA build uses fpga/rtl/sram_1r1w_fpga.sv instead of rtl/core/sram_1r1w.sv)
set sv_files [list \
    $root/fpga/rtl/sram_1r1w_fpga.sv \
    $root/rtl/core/learn_engine.sv \
    $root/rtl/core/neurocore.sv \
    $root/rtl/noc/noc_fifo.sv \
    $root/rtl/noc/router.sv \
    $root/rtl/noc/noc_host_in.sv \
    $root/rtl/noc/noc_host_out.sv \
    $root/rtl/noc/mesh.sv \
    $root/rtl/top/tick_ctrl.sv \
    $root/rtl/top/spikemesh_top.sv \
    $root/fpga/rtl/flit_link.sv \
    $root/fpga/rtl/uart_rx.sv \
    $root/fpga/rtl/uart_tx.sv \
    $root/fpga/rtl/uart_bridge.sv \
    $root/fpga/rtl/axis_bridge.sv \
    $root/fpga/rtl/kv260_top.sv]
read_verilog -sv $sv_files
read_verilog $root/fpga/rtl/kv260_top_wrap.v
read_xdc $script_dir/kv260.xdc
update_compile_order -fileset sources_1

# ---- block design
source [file join $script_dir bd.tcl]
create_spikemesh_bd spikemesh_bd $mesh_x $mesh_y $host_sel $baud [expr {$board ne ""}] $learn_mask $learn_mode
set bd_file [get_files spikemesh_bd.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
read_verilog $wrapper

# ---- implementation
synth_design -top spikemesh_bd_wrapper -part $part -flatten_hierarchy rebuilt
write_checkpoint -force $out/post_synth.dcp
report_utilization -file $out/utilization_synth.rpt
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force $out/post_route.dcp
report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -file $out/utilization_hier.rpt
report_timing_summary -max_paths 20 -file $out/timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "Worst setup slack: $wns ns"
if {$wns < 0} { puts "WARNING: timing not met (WNS $wns ns). See README 'If timing fails'." }
write_bitstream -force $out/spikemesh.bit

# ---- PYNQ needs the .hwh next to the .bit, with the same base name
set hwh [lindex [get_files -all -of_objects $bd_file *.hwh] 0]
if {$hwh ne ""} { file copy -force $hwh $out/spikemesh.hwh } else { puts "WARNING: no .hwh found" }
if {[catch {write_hw_platform -fixed -include_bit -force $out/spikemesh.xsa} msg]} {
    puts "NOTE: write_hw_platform skipped: $msg"
}
puts "Done: $out/spikemesh.bit"
