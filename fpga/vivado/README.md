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

* `spikemesh_top`: the chip (`rtl/top`) at SPEC v0.2, with N = A = 256, W_BITS = 8, LANES = 8,
  FANOUT_DEPTH = 1024, and the adaptive threshold on.
  * By default every core is inference-only.
  * `LEARN_MASK` (bit `y*MESH_X + x`) gives selected cores learning hardware.
  * `LEARN_MODE` picks the kind: 1 = STDP, 2 = STDP + R-STDP.
* `axis_bridge`: the Zynq path (AXI DMA plus an AXI-Lite register block).
* `uart_bridge`: an 8N1 UART, 921600 baud by default.

Both bridges speak the same 48-bit word protocol through `flit_link.sv`.
CTRL[8] (HOST_SEL) chooses which bridge drives the chip host port.

STEP_DONE and FLUSH_DONE are sent once `spikemesh_top`'s `drained` is high
and the host-out FIFO is empty. At that point every spike of UPDATE(t) has
already gone to the host, so it is exact and needs no timeout. Typically
this is 4 to 10 cycles after `step_done`. The old idle timer is still
available as a fallback: set `DRAIN_CYCLES` > 0 on `spikemesh_0`.

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
vivado -mode batch -source build.tcl -tclargs 4 4 0 921600 1 2   # 4x4, core (0,0) with R-STDP
```

The arguments are `MESH_X MESH_Y HOST_SEL_RST BAUD LEARN_MASK LEARN_MODE`,
and the defaults are `2 2 0 921600 0 1`. `LEARN_MASK` is a decimal number,
32 bits at most. A 2x2 build should take tens of minutes. The outputs go to
`build_<X>x<Y>/`, or to `build_<X>x<Y>_learn<mask>m<mode>/` for a learning
build:

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

SPEC v0.2 RTL, with the URAM memory (`sram_1r1w_fpga`), which is the default. Bold marks the budget that limits each build.

| Build | Logic LUTs, of 117 120 | FFs | LUTRAM (RAM32M16 / RAM64M8) | BRAM36 equivalent (RAMB18/2), of 144 | URAM288, of 64 | DSP | Fits? |
|---|---|---|---|---|---|---|---|
| 2x2, inference | 13 249 (11 %) | 5 804 | 188 / 18 | 12 | 8 | 0 | yes |
| 4x4, inference | **55 536 (47 %)** | 20 468 | 740 / 66 | 48 (33 %) | 32 (50 %) | 0 | yes |
| 4x4, core (0,0) STDP (`LEARN_MASK=1 LEARN_MODE=1`) | **62 021 (53 %)** | 21 462 | 748 / 74 | 48 | 32 | 10 | yes |
| 4x4, core (0,0) R-STDP (`LEARN_MASK=1 LEARN_MODE=2`) | **64 526 (55 %)** | 21 496 | 748 / 74 | 48 | 36 | 24 | yes |

Each core's memories map as follows:

| Memory | Maps to |
|---|---|
| 64 K x 8 weights | 2 URAM288 |
| Fan-out table, neuron state, and the v0.2 theta_a | 6 RAMB18 |
| R-STDP eligibility (1 Mbit) | 4 URAM288 |
| NoC FIFOs | LUTRAM |

v0.1 used 43 390 LUTs and 40 BRAM36 for the 4x4 inference build; the
growth since then is v0.2 (adaptive threshold, new registers) plus the
rtl/top and rtl/noc updates.

**How many learning cores fit:** each STDP core costs about 6.5 k LUTs and
each R-STDP core about 9 k. In Yosys numbers, a 4x4 mesh therefore has room
for about 9 STDP cores or 6 R-STDP cores. Vivado usually maps logic more
tightly than Yosys, so this is a lower bound.

Without the URAM memory (with plain `rtl/core/sram_1r1w.sv`), each core's
weights take 16 RAMB36. A 4x4 mesh then needs about 300 BRAM36, more than
the KV260's 144, so it does not fit (measured at v0.1: 296).

The memory budget is not what limits the URAM builds. BRAM sits at 33 %, and
URAM goes up by only 4 per R-STDP core.

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
* **Drain.** STEP_DONE waits for `spikemesh_top`'s `drained`, as described
  above. If a future chip drops that signal, build with `DRAIN_CYCLES` > 0
  to use the idle-timer fallback. The timer assumes no host-bound spike
  stays in the mesh for longer than that many cycles.
* **Learning cores.** The host programs learning registers through CFG
  flits, as in simulation. Size the mask using the LUT figures below.
