# tests/system: full-chip RTL == golden (SG5)

These tests run the whole chip RTL through the host API and check it bit for bit against the golden model. The chip is `spikemesh_top` plus `rtl/noc` and the real `rtl/core`.

```
python3 -m pytest -q tests/system            # 18 tests, about 1.5 min (Verilator + Icarus)
python3 -m pytest -q tests/system -m slow    # all 256 trained-net samples, about 1 min
```

## How the RTL is driven

`host/rtl_backend.py` (`RtlBackend`) implements `host.api.Backend` (`send_flits`, `step`, `recv_flits`). It starts the simulator once and talks to `tests/system/tb_sys.sv` through two named pipes. The simulator is either Verilator (`--binary --timing`, the fast path) or Icarus 12. cocotb is not used, because cocotb 2.1 needs Verilator 5.036 or newer and 5.020 is installed.

* `step()` waits until the host has handed over every queued flit, then pulses `step` and waits for `step_done`.
* It then waits for the chip's `drained` output and for the host output FIFO to be empty. At that point every host-bound spike from UPDATE(t) has been collected (SPEC §10).
* `recv_flits()` first syncs if flits were sent since the last sync, for example READ_REQs.

Each binary is cached in `tests/system/sim_build/` by a hash of the sources and parameters. A Verilator build takes about 6 s.

## What is checked

| Test | Config | Check |
|---|---|---|
| `test_random_net_icarus` ×3, `test_random_net_verilator` ×12 | random NetIRs (2–3 layers, W_BITS 8 and 4, dense and sparse weight images) compiled with `compile_net` onto 2x2 / 3x3, N = A = 32, LANES = 4, FANOUT_DEPTH 128; host-out backpressure 0 / 20 / 30 / 70 % | 2–3 samples each with `reset_state()` in between. Output and every hidden raster (`probe_layers`) must be equal across RTL, GoldenBackend and `reference_run`. No NoC error or drop. |
| `test_backend_readback_and_cycles` | 2x2 | Every configured weight is read back with READ_REQ, and the RTL responses equal the golden ones flit for flit. |
| `test_stdp_learning_in_mesh` | 2x2, `LEARN_MASK` = the layer-0 core, STDP (`LEARN_MODE` 1) | 16 learning steps with probed rasters. The RTL rasters equal golden during learning. All 24x32 weights are read back by READ_REQ; they equal golden, and learning changed more than 50 of them. |
| `test_trained_net_fast_path` (64 samples) / `test_trained_net_all_samples` (`-m slow`, 256) | trained 256→256→10 net (`tests/train/fixtures`), 2x2, N = A = 256, LANES = 8, T = 24 | Hidden and output rasters equal `reference_run` and the integer torch model. The first 8 samples are also compared with GoldenBackend. Predictions equal torch, so accuracy is equal: **0.9766 on all 256 samples, RTL = integer torch**. Writes `perf.json`, `cycles_vs_spikes.csv` and `.png`. |

## Performance

Measured on the trained net, all 256 emulated N-MNIST test samples, Verilator. This is the deployment program, without probes. Clock projections assume 100 MHz.

| Metric | Value |
|---|---|
| Cycles per timestep (mean / min / max) | 2 772 / 272 / 4 927 |
| Cycles per sample (T = 24), inference | 66 521 |
| Cycles per sample including `reset_state()` | 68 407 (reset adds 1 886) |
| Input events / hidden spikes / output spikes per sample | 936 / 1 118 / 34 |
| Synaptic ops per sample (events × 256 + hidden spikes × 10) | 250 900 |
| SOPs per cycle | 3.77 |
| Projected samples/s at 100 MHz (with state reset) | 1 503 (1 462) |
| Projected latency per sample at 100 MHz | 665 µs |
| Projected throughput at 100 MHz | 0.38 GSOP/s, 36 k timesteps/s |
| One-time program load (70 736 CFG flits) | 70 747 cycles |

**Event-driven scaling.** Cycles per sample fit 36.9 × input events + 32.0 k (R² = 0.93; the plot is `cycles_vs_spikes.png`, the data `cycles_vs_spikes.csv`). The intercept is mostly hidden-layer spikes. Fitted against all spikes (input events + hidden spikes), the cost is **31.8 cycles per spike + 1 144 cycles per sample (R² = 0.99996)**. Time is almost entirely proportional to spike traffic; only about 48 cycles per timestep are fixed.

**Where the cycles go.** 31.8 cycles per spike ≈ N_NEURONS / LANES = 256 / 8. Each delivered spike sweeps all 256 physical neurons of its core in 8-lane groups, even on the output core, which holds only 10 neurons. A hidden spike therefore costs as much as an input event while doing only 10 synaptic ops. That is why the measured rate is 3.8 SOP/cycle against 8 SOP/cycle of lane capacity. A core register that limits the sweep to the used neurons would make the 1 118 hidden spikes per sample about 16× cheaper. That would save about 1 118 × 30 ≈ 33 k of the 66.5 k cycles per sample on this net, roughly halving them.

## Findings

* **RTL equals golden end to end** on every configuration tested: inference, sparse and dense images, W_BITS 4 and 8, backpressure, multi-core layers on 3x3, state reset between samples, STDP learning and READ_REQ readback.
* **Sparse images need zeroed weight SRAM.** `dense_weights=False` (the compiler documents that it assumes zeroed weight memory) only works when the weight SRAM starts at zero. `rtl/core/sram_1r1w.sv` is not reset. In 4-state Icarus, a sparse image leaves X in the unwritten synapses and the first STEP never completes. Verilator initialises memories to 0 by default, so it passes there. On silicon the result would be garbage weights. The compiler default (`dense_weights=True`) is safe. The Icarus tests use dense images; the Verilator tests use both.
