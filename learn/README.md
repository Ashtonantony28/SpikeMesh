# On-chip learning experiments (golden model, SPEC v0.2, bit-exact)

Every weight change in these experiments comes from the SPEC §6/§9 STDP and
R-STDP logic running inside `golden.spikemesh.Chip` (`SPEC_VERSION = "0.2"`).
After setup, the host only injects events, sends REWARD packets, writes core
registers and neuron `v`/`bias`, and reads output spikes. It never writes a weight.

| File | What |
|---|---|
| `stdp_unsupervised.py` | Diehl & Cook-style unsupervised STDP on emulated N-MNIST |
| `rstdp_adapt.py` | Learning a held-out digit class on the chip with R-STDP and host REWARD packets |
| `../tests/learn/` | 31 tests, ~20 s: golden LEARN speed-up equivalence, pinned small versions of both experiments |
| `../tests/golden/test_v02.py` | the five SPEC v0.2 rules, plus the new register and debug-space map |

Golden speed-up (pure refactor): `LFSR16.take` slices a cached copy of the 65535-state
cycle, and LEARN steps 2 and 3 handle all flagged rows or columns in one vectorised call. `tests/learn/test_golden_speedup.py`
compares this against a reference with one loop iteration per row/column and one `next()` per draw.
That reference now includes the v0.2 rules. It checks equality after every step in 16 random STDP and R-STDP
scenarios that also exercise `plastic_axon_max` and the adaptive threshold. On a 256 × 256 core with 40
presynaptic and 10 postsynaptic spikes per step, LEARN went from 2.1 ms to 0.59 ms.

## 1. Unsupervised STDP (`stdp_unsupervised.py`)

**Mapping (one core).** The input uses axons 0..P−1, the P most active pixels. The host
drops the 256−P least active border pixels, because a 16 × 16 input uses all
256 axons. The remaining k = 256−P axons are inhibitory, with weights fixed at −128 by
`plastic_axon_max = P` (§9 rule 3). Every excitatory neuron fans out to all k
of them in its own core, plus one entry to the host (n × (k+1) ≤ 1024 fan-out entries), which gives
lateral inhibition one step later. An option with two cores and a separate inhibitory core (`--inh core`)
also exists.

**Host schedule per sample.** In the pattern phase (24 steps), `a_minus = 0`. In the complement phase (1 step),
`theta = 32767` and `a_plus = 0`, and the host sends events on every pixel that stayed silent, so the
recent winners are depressed on pixels outside the pattern. Diehl & Cook's normalisation is thus done by
on-chip STDP. Then v = 0, one drain step, and v = 0 again. The adaptive threshold (§9 rule 4:
`th_inc`, `th_decay_shift`) is active while learning. For labelling and test the host writes both
registers to 0, which freezes `theta_a`, and optionally lowers the base `theta` (`--test-theta`).

**Results.** One pass over 3 000 training samples, neurons labelled on 1 000 other training
samples, scored on 1 000 test samples. Each run takes about 0.8 min.

| Golden | Variant | Neurons | Test acc |
|---|---|---|---|
| v0.1 *(history, commit adff334)* | SPEC v0.1 only | 100 | 12.6 % |
| v0.1 *(history)* | + trace-decay fix prototype | 100 | 31.5 % |
| v0.1 *(history)* | + trace fix + adaptive-threshold prototype | 100 | 40.9 % |
| **v0.2** | decay floor + fixed inhibition (`plastic_axon_max`), no adaptive threshold | 100 | 33.8 % |
| **v0.2** | + adaptive threshold (`th_inc` 200, decay shift 10) | 100 | 43.1 % |
| **v0.2** | + `th_inc` 400, decay shift 12, k = 4, `test_theta` 1500 | 200 | **60.4 %** (seed 0), 57.5 % (seed 1) |

Ceiling for prototype learning on this encoding: non-spiking cosine k-means on per-pixel
event counts gets 77.5 % with 100 prototypes, **82.1 % with 200** and 83.7 % with 400.
So the best chip result is about 22 percentage points below the ideal learner with the same number of
neurons, and the 85 % target is above even the ideal at ≤ 256 neurons.

What I tried within about 40 min of compute:
* **Stronger homeostasis helps most.** Raising `th_inc` from 200 to 400 (decay shift 12) took
  100 neurons from 43 % to 55 %.
* **More neurons help.** 200 neurons with k = 4 inhibitory axons gave 57.5 %.
* **Lowering the base threshold at test helps.** Spikes are sparse (7–15 per sample); with
  `test_theta` 1500 the readout improves to 60 %.
* **Did not help:**
  * longer training (10 000 samples: 48–53 %);
  * slower learning (`stdp_shift` 9 or 10: 54–56 %);
  * refractory period 2 (54 %);
  * stronger `a_plus` / `a_minus` (58 %);
  * a smaller initial weight spread (58 %);
  * two cores of 128 neurons with an inhibitory core (27 %): the extra hop delays inhibition to t+2,
    so many neurons fire per sample (220 spikes).

The remaining gap looks structural. Inhibition is delayed by one step, and the network has
no per-neuron weight normalisation, only the complement-phase approximation.

Reproduce (results in `runs/learn/v02_*.json`):
```
python -m learn.stdp_unsupervised --out runs/learn/v02_stdp_base.json
python -m learn.stdp_unsupervised --a-minus 16 --th-inc 200 --th-decay-shift 10 --out runs/learn/v02_stdp_adaptive.json
B="--a-minus 16 --th-inc 400 --th-decay-shift 12 --n-exc 200 --k-inh 4 --test-theta 1500"
python -m learn.stdp_unsupervised $B --out runs/learn/v02_stdp_best_s0.json
python -m learn.stdp_unsupervised $B --seed 1 --lfsr-seed 4660 --out runs/learn/v02_stdp_best_s1.json
```

## 2. On-device adaptation with R-STDP (`rstdp_adapt.py`)

1. The host trains a 256→256→10 network with QAT (3 epochs) on emulated N-MNIST **without class 9**
   (`runs/learn/holdout9.npz`).
2. On the chip: the hidden layer is core (0,0), the frozen output layer is core (1,0), and every hidden
   neuron also fans out to core (1,1). Neuron 0 of that core is the new class head: weights 0,
   R-STDP enabled, `a_minus = 0`, and a bias of 170 so that it fires a little on any input.
3. After each sample the host sends one REWARD packet, decided from the output spikes it has just read:
   * class-9 sample: +32, unless the head already wins by 4;
   * other sample: −24 if the head would win, otherwise 0.

   Twelve silent drain steps follow (`tau_elig_shift` 3). After adaptation, the host writes the head's
   bias to 0. For readout, the head stands in for the untrained class-9 neuron.

The adaptation stream is 1 000 training samples, half class 9; the test set is 1 000 balanced test
samples. Each run takes 0.5 min.

| Golden | Seed | New class before → after | Old classes before → after |
|---|---|---|---|
| v0.1 *(history)* | 0 | 0 % → 96.0 % | 97.4 % → 89.6 % |
| v0.1 *(history)* | 1 | 0 % → 98.6 % | 96.4 % → 72.0 % |
| **v0.2** | 0 | 0 % → **95.6 %** | 97.4 % → 87.4 % |
| **v0.2** | 1 | 0 % → **99.4 %** | 96.4 % → 55.4 % |
| **v0.2** | 2 | 0 % → **96.0 %** | 96.2 % → 85.0 % |

The new class is always learned. How much accuracy the old classes keep still depends on the
sample order: in seed 1 the head keeps firing on old-class samples.

Rounded reward (§9 rule 2) made learning more robust. In the small pinned test, stream seed 1 used to end with a dead head
under v0.1, and now all five stream seeds learn. It did not fix forgetting of the old classes.
Increasing the negative reward to −32 or −40 changed old-class accuracy by only a few points
(seed 1: 63–70 %). Reward-by-label alone (+ for class 9, − for everything else) still drives the head's weights negative.

```
python -m learn.rstdp_adapt --seed 0 --out runs/learn/v02_rstdp_seed0.json   # seeds 1, 2 likewise
```

## SPEC history

The v0.1 findings from these experiments became SPEC v0.2 §9 (ADR-007), implemented in
`golden/spikemesh.py` and pinned in `tests/golden/test_v02.py`:

1. decay floor `x -= max(1, x >> k)`;
2. rounded reward;
3. `plastic_axon_max` (reg 23);
4. per-neuron adaptive threshold `theta_a` (regs 24/25, debug space 6);
5. rewards dropped while learning is off.

The compiler writes regs 23–25 = 0 and `theta_a` = 0 for every used neuron, so compiled
inference networks behave exactly as before.

Open observations, not SPEC changes:
* Lateral inhibition costs fan-out (n × (k+1) ≤ 1024) and pixel axons, because the 16 × 16 input fills all 256 axons.
* Inhibition through a separate core arrives one step later still, which breaks winner-take-all.
