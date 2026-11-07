# SpikeMesh

A digital neuromorphic processor in SystemVerilog. Leaky integrate-and-fire neurocores talk only by sending spikes as 48-bit packets over an event-driven 2D mesh network-on-chip, and cores can learn on-chip with STDP or reward-modulated STDP. Networks are trained in PyTorch/snnTorch with quantisation-aware training whose integer forward pass is bit-exact with the hardware.

**The core claim:** the trained PyTorch model, the integer golden model (`golden/spikemesh.py`) and the full chip RTL produce identical spike rasters. This is checked on every hidden and output spike of a trained network.

## Results (all reproducible from this repo)

| What | Result | Evidence |
|---|---|---|
| Trained SNN, 256→256→10, emulated N-MNIST (MNIST + saccade events, T=24) | 97.46% integer test accuracy (float 97.43%) | `train/README.md` |
| Torch integer model vs layer reference | identical spikes on all 10,000 test samples (reported by the training run; not stored as an artifact) | `train/README.md` |
| Full chip RTL (2×2 mesh, Verilator) vs torch/golden, trained net | identical hidden + output spikes on 256 samples; 97.66% on those samples | `tests/system/perf.json`, `make test-system` (`-m slow` for all 256) |
| Throughput at 100 MHz (projected from cycle counts) | ~1,500 samples/s, 3.77 synaptic ops/cycle, 665 µs per sample | `tests/system/perf.json` |
| Event-driven cost | cycles/sample = 31.8 × spikes + 1,144 (R² = 0.99996) | `tests/system/cycles_vs_spikes.png` |
| On-chip STDP / R-STDP RTL | weights, traces, LFSR identical to golden | `tests/core` (learn tests), `tests/system` |
| Unsupervised STDP (Diehl & Cook style, 200 neurons) | 60.4% (plan target 85%; ideal k-means ceiling on this encoding 82.1%) | `learn/README.md` |
| R-STDP on-device learning of an unseen digit | new class 0% → 95–99%; old classes drop to 55–87% | `learn/README.md` |
| KV260 resources (Yosys estimate, 4×4 mesh, UltraRAM weights) | 47% LUTs, 33% BRAM, 50% URAM | `fpga/vivado/README.md` |
| Formal | router safety properties proven by induction (FIFO depth 2); tick controller proven | `formal/run.sh`, `rtl/noc/README.md` |

## Layout

| Dir | Contents |
|---|---|
| `docs/SPEC.md` | The contract (v0.2): neuron math, timestep schedule, packets, address map, learning rules |
| `golden/` | Integer golden model (the oracle) |
| `train/` | Bit-exact QAT trainer, emulated N-MNIST data, real N-MNIST/DVS Gesture loader (`data_real.py`) |
| `compiler/` | NetIR → placement → config flits → binary image |
| `host/` | Host API with golden, RTL-simulation and FPGA (UART / PYNQ DMA) backends |
| `rtl/core`, `rtl/noc`, `rtl/top` | Neurocore (+ learn engine), router/mesh, tick controller, chip top |
| `fpga/` | KV260 wrapper, UART and AXI-Stream bridges, Vivado Tcl |
| `learn/` | On-chip learning experiments |
| `formal/` | Yosys SAT induction proofs |
| `docs/AUDIT-1.md` | Final audit: findings and their status |

## Running

```
pip install -r requirements.txt     # plus apt: iverilog verilator yosys
make test                           # full suite (~10 min)
make lint
python -m train.train --help
```

## Not yet done (needs hardware or tools outside the cloud container)

- Vivado build and board runs on the KV260 (`fpga/vivado/README.md` has the steps). The Tcl has never been executed.
- Real N-MNIST and DVS128 Gesture (download hosts were blocked here; `train/data_real.py` is ready).
- Live event-camera / webcam-emulator demo.
- OpenLane / SKY130 GDS of one core.
- Open audit items (medium/low) are listed in `docs/AUDIT-1.md`, including: sparse weight images need zeroed SRAM, the R-STDP eligibility memory has no reset value, and every spike sweeps all neurons of a core (about 2× throughput left on the table).
