# SG1: bit-exact QAT trainer

Trains spiking networks whose eval forward pass is *exactly* the SpikeMesh
integer neuron (docs/SPEC.md §2) with the one-timestep-per-layer latency of
§3, and exports them as an integer `NetIR` for the compiler.

| File | What |
|---|---|
| `data.py` | MNIST idx download (ossci S3, md5-checked, cached in `data/mnist`) → **emulated N-MNIST** event streams, bool `(T, 256)` |
| `data_real.py` | Real N-MNIST / DVS128 Gesture via `tonic`, same 16×16 binning. **Untested in the cloud** (hosts blocked); only its binning function has a test |
| `model.py` | `SpikeMeshNet`: LIF layers, int8 fake-quant (per-layer scale, STE), fast-sigmoid surrogate |
| `export.py` | `export_netir(model)` → `NetIR` |
| `train.py` | CLI training loop; writes `runs/<out>/{best.pt, net.npz, log.json}` |
| `make_fixture.py` | Copies a trained `net.npz` + 256 test samples into `tests/train/fixtures` |

## Emulated N-MNIST (not real N-MNIST)

Real N-MNIST was recorded by an ATIS sensor making three saccades in front of
a screen. We emulate that from plain MNIST, deterministically:

1. 28×28 → 16×16 area average (16·16 = 256 = `N_AXONS`, one core's inputs).
2. Translate the image (bilinear, sub-pixel) along a closed triangle
   (0,0) → (+d/2,+d) → (−d/2,+d) → (0,0), d = 2 px, T/3 steps per saccade.
3. DVS pixel model: each pixel keeps a reference level, fires when
   |frame − ref| ≥ 0.1 and moves ref by whole threshold steps. ON/OFF merged.

Default **T = 24** (8 steps per saccade), ~41 events per timestep (~1000 per
sample). T, d and threshold were picked with a non-spiking MLP probe on the
flattened rasters (24/2.0/0.1: 97.8 %; 24/1.5/0.15: 96.9 %; 18/1.5/0.1:
97.4 %). Fewer timesteps make every simulation (golden, RTL) proportionally
cheaper; 24 is the smallest that kept the probe near its best.

```
python -m train.data          # build caches in data/ (~10 s)
```

## Model and exact integer semantics

Each layer has real weights `W`, bias `B`, `theta_real = 1`, and a scale
`s = max|W| / 127` (detached; `fixed_scale` overrides it when a NetIR is
loaded). The membrane is simulated in integer units:

* `w = round(clamp(W/s, -128, 127))`, `b = round(clamp(B/s, int16))`,
  `theta = clamp(round(1/s), 1, 32767)`, `v_min = round(v_min_real/s)`
  (QAT: straight-through rounding).
* Per timestep, exactly §2: refractory check, `leak = floor(v / 2^k)`
  (= `v >>> k`), `v2 = sat16(v - leak + sat16(acc) + b)`, spike if
  `v2 >= theta`, subtract/zero reset, else `max(v2, v_min)`.
* §3 latency: layer 0 integrates input t at t; layer i integrates layer
  i−1's spikes from t−1.
* Training-only differences: surrogate gradient (fast sigmoid of
  `(v2-theta)/theta`), reset path detached. `quant=False` uses unrounded
  values and an exact `v/2^k` leak: that is the "float" model below.

**Readout (hardware decision):** class = output neuron with the most spikes
over all T timesteps; ties, including no spikes at all, go to the lowest
index. Loss: cross-entropy on `0.3 × spike counts`, plus optionally 0.5 ×
cross-entropy on the mean output membrane (`--mem-weight`, training only).

### Bit-exactness guarantee

In `quant=True` mode every value in the forward pass is an integer held in
float32 with magnitude < 2^24 (|acc| ≤ 256·128, |v2| < 2^17 before
saturation), so sums, the power-of-two divide and `floor` are exact.
`export_netir` reads the same `int_params(quant=True)` tensors the forward
pass uses. Therefore, for any input, the per-layer spike rasters of
`model(x, quant=True)` equal `compiler.netir.reference_run(export_netir(model), x)`,
which the tests also check against the golden `Chip`. This is checked, not
assumed:

* `tests/train` (32 tests, ~25 s): random float models with random leak /
  reset / refractory / v_min, random integer NetIRs including saturating
  ones (bias ±32767, theta up to 32767, leak and refrac up to 15), the
  committed trained net on 256 test samples, a tiny end-to-end training run,
  the golden `Chip` (hand-placed on a 2×2 mesh) for small random nets and
  for the full 256→256→10 trained net.
* One-off check on the full test set: `runs/f2` (and `runs/f3`) model vs `reference_run` on
  all 10 000 emulated test samples: **0 raster mismatches**, reference_run
  accuracy 97.46 % (f3: 97.39 %), equal to the torch integer accuracy.

## Results (emulated N-MNIST, 16×16, T = 24, MNIST test set, 10 000 samples)

Common settings: leak_shift 2, subtract reset, refrac 0, v_min = −theta,
Adam + OneCycle lr 5e-3, batch 128, 15 epochs (3 float, then 12 QAT),
55 000 train / 5 000 validation (model selection on validation int accuracy).
CPU, 2 threads per run, two runs in parallel.

| Run | Architecture | Loss | Float test | **Int test** | Gap (float − int) | Train time |
|---|---|---|---|---|---|---|
| f1 | 256→256→10 | count CE | 97.36 % | **97.43 %** | −0.07 pp | 17.6 min |
| f2 | 256→256→10 | count CE + 0.5 membrane CE | 97.43 % | **97.46 %** | −0.03 pp | 17.7 min |
| f3 | 256→256→256→10 | count CE + 0.5 membrane CE | 97.47 % | **97.39 %** | +0.08 pp | 28.6 min wall, stopped by the time budget after 11 of 15 epochs (CPU shared with other jobs, 3 threads) |

`runs/f2/net.npz` is the committed fixture (`tests/train/fixtures/trained_net.npz`):
thetas 87 / 97, leak_shift 2, weights in [−127, 127], mean firing rate
0.19 (hidden) / 0.14 (output) spikes per neuron per timestep.
For reference, a non-spiking float MLP (256·24 → 256 → 10) on the same
rasters gets 97.8 %, so the SNN is within ~0.4 pp of what the input encoding allows.
The float/int gap is ~0, as expected after QAT.

## Commands

```
python -m train.train --epochs 15 --float-epochs 3 --lr 5e-3 --leak 2 \
    --count-gain 0.3 --mem-weight 0.5 --out runs/f2            # ~18 min, 2 threads
python -m train.train --hidden 256,256 ...                      # 3-layer variant
python -m train.make_fixture runs/f2/net.npz                    # refresh test fixture
python -m pytest -q tests/train
```

Real data (on a machine that can reach the dataset hosts):

```
python -m train.data_real --dataset nmnist --T 24
python -m train.train --real nmnist --out runs/real_nmnist ...
python -m train.data_real --dataset dvsgesture --T 24           # 11 classes
```

Every layer has fan-in ≤ 256 (asserted), so each layer fits one core.
