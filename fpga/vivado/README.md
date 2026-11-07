# SpikeMesh on the Kria KV260

> **Not run in Vivado.** These scripts were written in a cloud session that
> has no Vivado. The RTL is simulated (Icarus and cocotb, `tests/fpga`),
> linted (Verilator `-Wall`) and synthesized with Yosys `synth_xilinx
> -family xcup`. The Tcl, the block design and the pin constraints have
> **not** run through Vivado. Treat the first build as a bring-up. The steps
> most likely to need a fix are listed under *Known risks*.

## What gets built

```
 Zynq US+ PS ── M_AXI_HPM0_FPD ── SmartConnect ──┬── axi_dma_0 S_AXI_LITE (DMA control)
     ▲                                           └── spikemesh_0 s_axi (registers)
     │ S_AXI_HP0 ◄── SmartConnect ◄── axi_dma_0 M_AXI_MM2S / M_AXI_S2MM
     │
 axi_dma_0 MM2S ──AXIS 64b──► spikemesh_0 (kv260_top) ──AXIS 64b──► axi_dma_0 S2MM
                                   │  uart_rx / uart_tx ──► PMOD J2
```

`spikemesh_0` is `fpga/rtl/kv260_top_wrap.v`, a Verilog-2001 wrapper around
`kv260_top.sv`. The wrapper exists because Vivado module references can't
use a SystemVerilog top. `kv260_top.sv` contains:

* `spikemesh_top`: the chip (`rtl/top`), N = A = 256, W_BITS = 8, LANES = 8,
  FANOUT_DEPTH = 1024, and LEARN = 0 (`spikemesh_top` has no LEARN parameter).
* `axis_bridge`: the Zynq path (AXI DMA plus an AXI-Lite register block).
* `uart_bridge`: an 8N1 UART, 921600 baud by default.

Both bridges speak the same 48-bit word protocol through `flit_link.sv`.
CTRL[8] (HOST_SEL) chooses which bridge drives the chip host port.

The FPGA build uses `fpga/rtl/sram_1r1w_fpga.sv` in place of
`rtl/core/sram_1r1w.sv`. The ports are identical. The only change is that
the 64 K x 8 weight array of each core is mapped to UltraRAM. Without this,
a 4x4 mesh does not fit (see the table below).

## Requirements

* Vivado 2022.2 or newer, with the Zynq UltraScale+ device support installed.
* Optional: KV260 board files. In Vivado, open
  *Tools → Vivado Store → Boards*, install *Kria KV260 Vision AI Starter
  Kit*, and restart Vivado. Without them, the script configures the PS from
  the part alone and prints a warning. The PS DDR and MIO settings then fall
  back to defaults, which is fine for a PL-only bitstream that PYNQ loads on
  top of a running Linux.
* On the board: Kria-PYNQ (v3.0 or newer) on the Ubuntu image that Kria
  supports.

## Build

```sh
cd fpga/vivado
vivado -mode batch -source build.tcl -tclargs 2 2 0          # 2x2 mesh, AXI DMA host path
vivado -mode batch -source build.tcl -tclargs 4 4 0          # 4x4 mesh
vivado -mode batch -source build.tcl -tclargs 2 2 1 921600   # host path = PMOD UART at reset
```

The arguments are `MESH_X MESH_Y HOST_SEL_RST BAUD`, and the defaults are
`2 2 0 921600`. A 2x2 build should take tens of minutes. The outputs go to
`build_<X>x<Y>/`:

| File | Use |
|---|---|
| `spikemesh.bit`, `spikemesh.hwh` | Copy both to the board for PYNQ. They must keep the same base name. |
| `spikemesh.xsa` | For Vitis or PetaLinux, if you want them. |
| `utilization.rpt`, `utilization_hier.rpt`, `timing.rpt` | Post-route reports |
| `post_synth.dcp`, `post_route.dcp` | Checkpoints |

The script prints `Worst setup slack: … ns` after routing. A negative
value means timing failed (see below).

## Run with PYNQ (AXI DMA path)

```sh
scp build_2x2/spikemesh.{bit,hwh} ubuntu@kria:~/spikemesh/
scp -r host compiler golden ubuntu@kria:~/spikemesh/      # the Python package code
```

Then, on the board (as root, or as a user with PYNQ rights):

```python
from host.fpga_backend import PynqDmaBackend
be = PynqDmaBackend("spikemesh.bit")      # loads the overlay and checks ID 0x534D0107 and CONFIG
be.send_flits(cfg_flits)                  # flits produced by the compiler
for t in range(T):
    be.send_flits(inputs[t])
    be.step()                             # STEP -> waits for STEP_DONE, raises LinkError on chip err
out = be.recv_flits()                     # FLUSH, then returns the host-bound flits
print(be.counters())                      # STEP_COUNT, LAST_STEP_CYCLES, BUSY_CYCLES, ...
```

`PynqDmaBackend` arms S2MM **before** it triggers STEP or FLUSH. If S2MM is
not armed, the chip's output backs up, and the chip stalls until a receive
is posted.

## Run over UART (any PC)

The KV260's own micro-USB UART connects to the PS (Linux console), not to
the PL. Use a 3.3 V USB-UART on **PMOD J2**:

| USB-UART | KV260 J2 |
|---|---|
| TXD | pin 1 (`uart_rx`) |
| RXD | pin 2 (`uart_tx`) |
| GND | pin 5 or 11 |

There are two ways to select the UART path. You can build with
`HOST_SEL_RST=1`, or you can write `0x100` to CTRL (offset 0x00) from the
PS, for example with `pynq.MMIO`. Any CTRL write also sets HOST_SEL from bit
8, so writing 0 switches back to AXI. The PL still needs a loaded bitstream,
so the board has to be booted and the overlay loaded first.

```python
from host.fpga_backend import UartBackend
be = UartBackend("/dev/ttyUSB0", baud=921600)
```

Wire format: every word is 6 bytes, little-endian, in both directions. The
link command and event words are documented in `fpga/rtl/flit_link.sv`.

## Register map (AXI-Lite, `spikemesh_0`)

The authoritative map is the header comment of `fpga/rtl/axis_bridge.sv`.

| Offset | Register | Contents |
|---|---|---|
| 0x00 | CTRL | Write-only pulses: STEP [0], FLUSH [1], SOFT_RESET [3]. Read/write: HOST_SEL [8]. |
| 0x04 | STATUS | done [0] (sticky, W1C), busy [1], t_parity [2], err [3] (sticky, W1C), running [4], UART overflow [5], UART frame error [6] |
| 0x08 | STEP_COUNT | |
| 0x0C | LAST_STEP_CYCLES | |
| 0x10 | LAST_DRAIN_CYCLES | |
| 0x14 | BUSY_CYCLES | |
| 0x18 | CYCLES | |
| 0x1C | IN_FLITS | |
| 0x20 | OUT_FLITS | |
| 0x24 | DROP_CNT | |
| 0x28 | ID | 0x534D0107 |
| 0x2C | CONFIG | |

## Resources (Yosys estimate, not Vivado)

These are Yosys 0.33 `synth_xilinx -family xcup` results for `kv260_top`
(the PS, DMA and SmartConnect are not included), at N = A = 256, LANES = 8,
W_BITS = 8 and FANOUT_DEPTH = 1024. The KV260 (XCK26) has 117 120 LUTs,
234 240 FFs, 144 BRAM36 (288 BRAM18), 64 URAM288 and 1 248 DSPs. Vivado
results will differ: expect fewer LUTs, and possibly different BRAM and
LUTRAM choices.

| Build | Logic LUTs | FFs | LUTRAM (RAM32M16 / RAM64M8) | BRAM36 equivalent (RAMB36 + RAMB18/2), of 144 | URAM288, of 64 | DSP | Fits? |
|---|---|---|---|---|---|---|---|
| 2x2, plain `sram_1r1w` (BRAM only) | 9 558 | 5 061 | 160 / 18 | **74** (64 + 20/2) | 0 | 0 | yes |
| 2x2, `sram_1r1w_fpga` (URAM) | 9 588 | 5 101 | 160 / 18 | **10** | **8** | 0 | yes |
| 4x4, plain `sram_1r1w` (BRAM only) | 42 490 | 18 821 | 680 / 66 | **296** (256 + 80/2) | 0 | 0 | **no: needs 206 % of the BRAM** |
| 4x4, `sram_1r1w_fpga` (URAM), **the default** | 43 390 | 19 009 | 680 / 66 | **40** (28 %) | **32** (50 %) | 0 | yes, at 37 % of the LUTs |
| 2x2 URAM, every core LEARN=1 (a local edit of `neurocore.sv`) | 31 171 | 8 065 | 192 / 50 | 10 | 8 | 34 | yes |

Each core's 64 K x 8 weight memory takes 2 URAM288 or 16 RAMB36. The
fan-out table and the neuron state take 5 RAMB18. The NoC FIFOs map to
LUTRAM.

LEARN=1 adds about 5 400 LUTs and 8 to 9 DSPs per core. The SG6 single-core
synth agrees: LEARN=1 is 7 653 LUTs and 10 DSPs, against 1 625 LUTs
for an inference core. At that rate, a 4x4 mesh with all cores learning
would come to roughly 130 k LUTs in Yosys, which is over the KV260's 117 k.
Vivado usually maps logic more tightly, but plan on learning in only some
of the cores, or on a 3x3 or smaller mesh, until there is a Vivado number.
LEARN=2 adds trace memory: 32 RAMB36 per core in the SG6 BRAM build.

## If timing fails

The target is 100 MHz. Nothing in this design has been timed in Vivado.
If timing fails, try these in order:

1. Lower `pl_clk0`. In `bd.tcl`, change
   `PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ` to 75 or 50. Also change
   `CLK_HZ` on `spikemesh_0` to the same value, so that the UART baud
   divider stays correct. The DMA path doesn't care about the clock rate.
2. Look at `timing.rpt`. Paths through the neurocore accumulator and lane
   adders are the most likely to fail. Paths that read a URAM straight into
   logic are another candidate. Vivado can pipeline a URAM output if a
   register stage is added after it, but that would change `rtl/core`
   timing, so it is a change for the core owner.
3. Add `-directive Explore` to `place_design` and `route_design`.

## Known risks (check these first)

* **PMOD pins.** H12 and E10 were taken from the KV260 carrier
  documentation but not checked against a board. Before you build, compare
  them with the official `kv260` carrier XDC for your board revision.
* **Block design property names.** The `zynq_ultra_ps_e` and `axi_dma`
  CONFIG keys were written conservatively, but Vivado versions differ. If
  `bd.tcl` stops on a property, open the BD in the GUI. Then run
  `report_property [get_bd_cells zynq_ps]` and fix the name.
* **Drain window.** STEP_DONE is sent after `host_out` has been idle for
  `DRAIN_CYCLES` (64) cycles following `step_done`. This assumes that no
  host-bound spike is in flight in the mesh for longer than that. That
  holds for meshes up to 4x4 without heavy congestion. Raise
  `DRAIN_CYCLES` if spikes ever show up a step late.
* **LEARN.** `spikemesh_top` doesn't expose `LEARN`, so FPGA builds have
  inference cores only.
