# SpikeMesh build plan

Full rationale: `/mnt/project-files/project-plans/neuromorphic-snn-chip-PLAN.md` (project folder). Contract: `docs/SPEC.md`.

## Decisions (2026-11-06, Ashton approved the plan; Claude's recommendations taken)
- FPGA target: AMD Kria KV260. Bitstream builds and board runs happen on Ashton's machine (Vivado); cloud sessions do RTL, simulation, Yosys synthesis checks.
- No event camera purchase: webcam-to-events emulator for the live demo.
- Silicon: stop at GDS (OpenLane/LibreLane, SKY130). Tiny Tapeout stays optional.
- Datasets: N-MNIST / DVS128 Gesture hosts are blocked from the cloud container. Cloud runs use MNIST (reachable) converted to saccade events ("emulated N-MNIST", always labelled as such). Real-dataset scripts ship ready for Ashton's machine.

## Scope of this build pass (cloud, simulation-verified)
| SG | Content | Owner dir | Exit test |
|---|---|---|---|
| SG0 | SPEC, golden model, NetIR, CI skeleton | docs/, golden/, compiler/netir.py | `pytest tests/golden` |
| SG1 | Bit-exact snnTorch QAT trainer, emulated N-MNIST | train/ | torch-int == golden rasters; accuracy target in train/README |
| SG2 | Neurocore RTL (inference) | rtl/core/ | cocotb vs golden, random + trained nets |
| SG3 | Router, mesh, tick controller, top | rtl/noc/, rtl/top/ | cocotb stress + SymbiYosys props |
| SG4 | Compiler: NetIR → placement → CFG flit image | compiler/ | golden chip from image == reference_run |
| SG5 | Integration: full chip RTL (Verilator) == golden | tests/system/ | trained net bit-exact end-to-end |
| SG6 | On-chip learning RTL (STDP, R-STDP) | rtl/core/ | weights after learning == golden |
| SG7 | FPGA top for KV260, Yosys resource report, host API | fpga/, host/ | synth clean; resource table |
| SG8 | Final audit + README | docs/AUDIT-1.md | done 2026-11-07: 2 high fixed, 0 critical; project closed at this audit |

Deferred to Ashton's hardware: M5/M6/M8 board runs, real DVS Gesture, live demo, M9 GDS run.
