# SpikeMesh audit 1 (final)

Date: 2026-11-07. Scope: the whole repo against `docs/SPEC.md` v0.2: RTL, golden model, compiler, host and FPGA link, trainer, learning experiments, tests and README claims. This was the single closing audit round. The auditor ran no git commands that change state.

## Summary

The core claim holds. The RTL equals the golden model bit for bit on every configuration the suite covers, and the golden model equals the independent `reference_run` and the integer torch model. No critical findings.

Two high findings were fixed, each with a regression test that fails before the fix and passes after it:

* **H1.** The RTL never reset the STDP trace memories, although SPEC §5 says it must. The tests hid this because they either wrote every trace first or ran on Verilator, which zero-initialises memories.
* **H2.** On the KV260 AXI DMA path, a burst of READ_REQs (for example a weight readback after learning) would hang the host. The cocotb AXI model always drains the output stream, so the tests could not see it.

Six medium and six low findings stay open, listed below. The five items already on record were checked against the code and artifacts, not copied.

## Test results

| Command | Before the fixes | After the fixes |
|---|---|---|
| `make test` (golden, compiler, train, core, noc, system) | 238 passed, 1 skipped (26 + 84 + 32 + 65 + 13 + 18; the skip is the `-m slow` all-sample run) | 241 passed, 1 skipped (26 + 84 + 32 + 68 + 13 + 18) |
| `pytest tests/learn` (not in `make test`) | 31 passed | 31 passed |
| `pytest tests/fpga` (not in `make test`) | 8 passed | 9 passed |
| `make lint` (Verilator `-Wall`, `spikemesh_top`) | clean | clean |
| Verilator `-Wall` lint of `kv260_top` with every FPGA source | clean | clean |
| `formal/run.sh` | not re-run (see *Not verified*). The last summary in `formal/build/summary.txt` shows 11 proofs/BMCs passing and 5 mutants failing as expected. | not re-run |

New tests: `tests/core/test_core.py::test_learn_traces_reset` (3 configs) and `tests/fpga/test_fpga.py::test_axi_backend_bulk_readback_no_deadlock`.

## Findings

| ID | Sev | Status | Where | Finding |
|---|---|---|---|---|
| H1 | high | **fixed** | `rtl/core/learn_engine.sv` (trace SRAMs `u_xpre` / `u_xpost`) | Traces not reset (SPEC §5 violation) |
| H2 | high | **fixed** | `host/fpga_backend.py` `send_flits`; `_PynqIO.send` | AXI DMA host deadlocks on READ_REQ bursts |
| M1 | medium | open (known) | `rtl/core/sram_1r1w.sv`; `compiler/compile.py:415` | Sparse weight images rely on zeroed weight SRAM |
| M2 | medium | open | `rtl/core/learn_engine.sv` `u_elig`; SPEC §5 | R-STDP eligibility memory is never reset, and the SPEC does not say what it holds after reset |
| M3 | medium | open | `learn/stdp_unsupervised.py:14-17`; `rtl/core/neurocore.sv:35` | Direct inhibition breaks the SPEC §2 one-spike-per-axon rule; the RTL accumulator can wrap where golden saturates |
| M4 | medium | open (known) | `rtl/core/neurocore.sv:660-663` | Every spike sweeps all `N_NEURONS` |
| M5 | medium | fixed (lead) | `Makefile:3` | `make test` skips `tests/fpga` and `tests/learn` |
| M6 | medium | open (known) | `fpga/vivado/*`, `fpga/vivado/README.md` | Vivado never run; the Yosys resource table has no report in the repo |
| L1 | low | open (known) | `formal/run.sh` | Router FIFO_DEPTH 4 not proven; whole-system quiescence not formally composed |
| L2 | low | open (known) | `learn/README.md` | Unsupervised STDP reaches 60.4 %, below the 85 % target; some ceiling claims have no artifact |
| L3 | low | open | `host/fpga_backend.py` `soft_reset` | After SOFT_RESET the chip runs the old program with default registers |
| L4 | low | fixed (lead, reg 22 = 0 now written) | `compiler/compile.py:376-385` | Compiler does not write reg 22 (`dbg_elig`) |
| L5 | low | open | `fpga/rtl/uart_bridge.sv`, `flit_link.sv` | A UART-only host cannot see UART overflow or framing errors |
| L6 | low | open | `docs/adr/`, `README.md` | ADRs cited by the SPEC are missing; the top README says "under construction" |

### H1 (high, fixed): trace memories were not reset

SPEC §5 lists "learning flags and traces" as control state that `rst_n` resets, and golden starts `x_pre` and `x_post` at 0. In the RTL, the traces were plain `sram_1r1w` memories with no reset path. Only the accumulator banks and `theta_a` were cleared.

* **Failure scenario.** Run STDP on an ASIC. Or, on the KV260, run a learning session, `soft_reset()`, then learn again: BRAM keeps its contents, so the first LEARN uses stale traces. The host code in `learn/` never writes traces, so the weights silently diverge from golden.
* **Why the tests missed it.** The core tests write every trace through the debug space before learning. `test_stdp_learning_in_mesh` runs only on Verilator, which zero-initialises memories. On Icarus, the new test read X.
* **Fix.** `learn_engine` now clears both trace memories after reset: one entry or row per cycle, `max(N_AXONS, G)` cycles, with `busy` held high. This is 256 cycles at N = A = 256, within the core's existing N-cycle reset clear. The fix is documented in `rtl/core/README.md`.
* **Regression test.** `test_learn_traces_reset` (`small_l2`, `rect_l2`, `g1_l2`; Icarus; mode `stdp_notrace`) never writes the traces. It failed before the fix and passes after.

### H2 (high, fixed): AXI DMA host path deadlocks on READ_REQ bursts

On the board, `_PynqIO.send` blocks on `sendchannel.wait()`. S2MM is armed only inside `_command`, during STEP, FLUSH and STATUS.

* **Failure scenario.** `be.send_flits([read_req(...)] * 768)` followed by `be.recv_flits()`, which is the weight readback pattern `test_stdp_learning_in_mesh` uses. The READ_RESPs fill the core output FIFO, the router FIFOs and the host-out FIFO (about 12 flits). The core then stalls in `S_RDP`, host_in runs out of credits, `s_axis_tready` drops, and MM2S never completes. The host hangs before it ever arms S2MM. On UART the same burst pushes unbounded responses into the OS receive buffer, which has no flow control.
* **Why the tests missed it.** `tests/fpga/fpga_tb.py:axis_sink` drains `m_axis` all the time, which a real AXI DMA does not do, and the readback in the test is only 9 READ_REQs.
* **Fix.** `_LinkBackend.send_flits` leaves at most 4 READ_RESPs unread. Four always fit in the target core's own 4-entry output FIFO, whichever cores they come from. Before the 5th READ_REQ it sends the words so far and issues FLUSH; the responses go into `_rx` in order. The counter resets on STEP and FLUSH.
* **Regression test.** `test_axi_backend_bulk_readback_no_deadlock` uses an I/O model that drains only while a transfer is armed. It failed before the fix and passes after.

### M1 (medium, known, verified): sparse images need zeroed weight SRAM

Confirmed. `sram_1r1w` has no reset, and the core's reset clear covers only the accumulator banks and `theta_a`. `dense_weights=False` therefore works on Verilator, which zero-initialises memories, and on the FPGA right after the bitstream loads, because BRAM and URAM start at 0. It does not work after SOFT_RESET, or on an ASIC.

The Verilator sparse tests in `tests/system` pass because of the simulator's zero-init, so they do not show that sparse images are safe.

**Suggested fix:** make `compile_net(dense_weights=False)` require an explicit `assume_zeroed_weights=True`, or have the host clear the used rows once.

### M2 (medium): eligibility memory never reset

SPEC §5 names neither the eligibility memory `e[a][n]` as reset state nor as an undefined memory. Golden starts it at 0, the RTL never clears it, and no host code writes it (`learn/rstdp_adapt.py` runs on golden).

* **Failure scenario.** On the KV260, run an R-STDP session, `soft_reset()`, and start a new one. The first non-zero REWARD applies `r·e_stale` to every synapse of the core.
* **Suggested fix.** In the SPEC, list `e` as an undefined memory and have the host zero it through reg 22 before enabling R-STDP. An RTL clear would cost A·G cycles at reset, which breaks the core harness's reset timeout.

### M3 (medium): direct inhibition violates the accumulator precondition

With `inh="direct"`, every firing excitatory neuron sends a SPIKE to each of the k inhibitory axons of its own core. One inhibitory axon therefore receives m spikes in one timestep when m neurons fire.

SPEC §2 makes the accumulator exact only for one spike per axon per timestep, and the RTL's default `ACC_W` is 17 bits. The inhibitory sum is −128·m·k:

* With k = 9 and m ≥ 57 simultaneous spikes, the sum passes −65 536. The RTL accumulator wraps to a large positive value, so inhibition becomes excitation. Golden uses int64, saturates at −32 768, and cannot flag the case.
* The best run uses k = 4, where m must reach about 128 out of 200 neurons. That is unlikely once training has settled, but possible early with random weights.

The learning results are golden-only, so they are not affected, but this mapping is not RTL-safe as written.

**Suggested fix:** make golden assert that `|acc|` fits `ACC_W` (or model the wrap), and document that this mapping needs a larger `ACC_W`.

### M4 (medium, known, verified): every spike sweeps all neurons

`S_SPK` always runs G = N/LANES groups. This matches the measured 31.8 cycles per spike in `tests/system/perf.json`, about 256/8. The 10-neuron output core pays the full 32 cycles per hidden spike, which leaves roughly 2x throughput unused on the trained net.

**Suggested fix:** a "used neurons" core register that limits the sweep.

### M5 (medium): `make test` is not the whole suite

`Makefile:3` runs golden, compiler, train, core, noc and system only. The FPGA link tests (9, about 2 min) and the learning tests (31, about 20 s) run only when someone invokes them by hand. A green `make test` therefore says nothing about the host link, which is the area H2 was in.

**Suggested fix:** add `test-fpga` and `test-learn` targets.

### M6 (medium, known, verified): no Vivado run, and the resource numbers have no artifact

`fpga/vivado/README.md` says so itself. Still unverified:

* timing at 100 MHz (every "projected at 100 MHz" performance figure assumes it);
* URAM inference;
* the block-design property names and the PMOD pins.

The Yosys resource table in that README has no log or report in the repo.

**Suggested fix:** commit the Yosys `stat` output next to the table, and do the first Vivado build as a bring-up.

### Low

* **L1.** The router is proven at FIFO_DEPTH 2 (k-induction) plus BMC depth 4. Its FIFO_DEPTH 4 induction did not finish (`formal/run.sh:56-58`), so depth 4, the shipped depth, is covered by simulation only. The tick controller proof takes `net_busy` and `core_busy` as free inputs. The step "busy covers every in-flight flit", across `neurocore.busy`, router P10 and the link registers, is argued by hand, not proven end to end.
* **L2.** `runs/learn/v02_stdp_best_s0.json` confirms 0.604 (seed 1: 0.575), so the 85 % target is missed. Several claims have no script or artifact in the repo: the k-means ceiling (77.5 / 82.1 / 83.7 %), the MLP probe sweep (97.8 % and others) in `train/README.md`, and "0 raster mismatches on all 10 000 test samples". The numbers that do have artifacts match their artifacts: the f1/f2/f3 test accuracies (`runs/*/log.json`), the R-STDP seeds (`runs/learn/v02_rstdp_seed*.json`) and the system perf table (`tests/system/perf.json`).
* **L3.** `soft_reset()` resets the core registers to their defaults (theta 64, leak 0, v_min −32768 and so on) but keeps the weights and fan-out pointers. A host that calls `run()` afterwards gets spikes from a half-configured chip and no error. **Suggested fix:** clear `SpikeMeshHost.prog`, or make `soft_reset` reload the program.
* **L4.** The compiler writes `learn_en = 0` but never reg 22 (`dbg_elig`). On an R-STDP build left with reg 22 = 1 and no reset, loading a program writes every weight into the eligibility memory. **Suggested fix:** also write reg 22 = 0, and reg 19 = 0.
* **L5.** The `rx_overflow` and `rx_frame_err` flags are readable only through AXI-Lite STATUS. The UART STATUS event does not carry them, so a UART-only host sees a dropped byte only as a timeout.
* **L6.** `docs/adr/` is empty, although the SPEC cites ADR-001 to ADR-008 and requires an ADR for every change. `README.md` still says "under construction" (PLAN SG8 lists the final README).

## Items checked and found correct

* Flit encodings in the core emitter, the READ_RESP layout and the link event words.
* Read-during-write forwarding in the accumulator pipeline.
* Concurrent UPDATE and spike accumulation on opposite banks.
* No lost READ_RESP when emitter and read response collide (they cannot overlap).
* Deadlock freedom during STEP: the core accepts p^1 spikes without needing output space.
* `drained` and STEP_DONE ordering, including the host-out FIFO check.
* Compiler fan-out base and count limits.
* `reference_run` is an independent implementation, not a call into golden, so "RTL == golden == reference_run" is not a self-comparison.
* The README test counts (train 32, learn 31, system 18) match the suite.

## Not verified

* `formal/run.sh` (about 12 min). No reason was found to doubt its last recorded summary.
* Anything on hardware: Vivado synthesis, timing, place and route, the board, PYNQ, the real AXI DMA (H2 is fixed against a model of it, not the IP), the UART at 921 600 baud, and the PMOD pins.
* The `-m slow` all-256-sample system test (skipped in `make test`) and the 10 000-sample bit-exactness claim.
* Real N-MNIST and DVS Gesture (`train/data_real.py`): no network access to the dataset hosts.
* Re-running the learning experiments and training runs. Only their artifacts were compared with the READMEs.
* ACC_W overflow on the RTL for the M3 mapping; M3 was found by reasoning, not by a failing run.
