# On-chip learning experiments (golden model, bit-exact)

Every weight change in these experiments comes from the SPEC §6 STDP / R-STDP
logic running inside `golden.spikemesh.Chip`. After setup, the host only
injects events, sends REWARD packets, writes core registers or neuron `v`/`bias`
(CFG writes), and reads output spikes. It never writes a weight. Anything not
in SPEC v0.1 is a clearly flagged prototype in `learn/`, never a change to `golden/`.

| File | What |
|---|---|
| `stdp_unsupervised.py` | Diehl & Cook-style unsupervised STDP on emulated N-MNIST |
| `rstdp_adapt.py` | Learning a held-out digit class on the chip with R-STDP and host REWARD packets |
| `../tests/learn/` | 28 tests, ~14 s: golden speed-up equivalence, pinned small versions of both experiments, the SPEC findings |

Golden speed-up (`golden/spikemesh.py`, no semantic change): `LFSR16.take` now
slices a cached copy of the full 65535-state cycle, and LEARN steps 2 and 3
handle all flagged rows or columns in one vectorised call, drawing LFSR values in
the same order. `tests/learn/test_golden_speedup.py` keeps the original loops as
reference functions. It checks that the LFSR sequences match (several seeds, up to 70 000
draws), and that weights, eligibility, traces, v, refc, LFSR state and output spikes match after every
step for 10 random STDP and R-STDP scenarios. On a 256 × 256 core with 40
presynaptic and 10 postsynaptic spikes per step, LEARN went from 2.1 ms to 0.59 ms.

## 1. Unsupervised STDP (`stdp_unsupervised.py`)

**Mapping (one core).** The input uses axons 0..246: the 247 most active pixels
of the 16 × 16 input. The host drops the 9 least active border pixels, because a 16 × 16 input
uses all 256 axons. Axons 247..255 are 9 inhibitory axons. Each of the
100 excitatory neurons has 9 fan-out entries to those axons of its own core,
plus one to the host (100 × 10 = 1000 ≤ 1024 fan-out entries). Inhibitory
weights start at −128, so lateral inhibition lands one timestep after a spike.

**Host schedule.** A plain STDP run collapses: weights either run away to
`w_max` or all decay to `w_min`. So each sample uses a two-phase register
schedule. Phase 1 is the pattern: 24 timesteps with `a_minus = 0`, so only
Hebbian potentiation happens. Phase 2 is the complement: one timestep with `theta = 32767`
and `a_plus = 0`. In it, the host injects events on every pixel that stayed silent and on the
inhibitory axons. "Pre after post" then depresses exactly the recent winners'
weights on pixels outside the pattern. This is Diehl & Cook's (x_pre − x_tar)
normalisation, done by on-chip STDP. After the sample, the host writes v = 0, runs one drain step, and
writes v = 0 again.

**Results.** 100 excitatory neurons, one pass over 3 000 training samples,
neurons labelled on 1 000 separate training samples, scored on 1 000 test
samples. Each run takes about 1.4 min.

| Variant | SPEC | Test acc | Notes |
|---|---|---|---|
| SPEC v0.1 only | yes | **12.6 %** | 87 of 100 neurons never fire; the 13 that do respond to everything |
| + trace-decay fix (prototype) | **no** | 31.5 % | 94 neurons labelled, all 10 classes covered |
| + trace fix + adaptive threshold (prototype) | **no** | **40.9 %** | `theta_plus` 200, decay shift 10 |

**The 85 % target was not reached.** Two reasons:

1. *The encoding limits even ideal prototype learning.* These numbers are non-spiking float
   baselines on per-pixel event counts of the same data. Cosine k-means gets 77.5 % with 100
   prototypes and 83.7 % with 400. Cosine class means get 68.9 %. A plain dot product with
   class means gets 17.6 %. So per-neuron weight normalisation is essential, and
   ≥ 85 % would need more than 256 neurons even in the ideal case. Hand-placed class
   templates loaded onto the chip with learning off got only 20–35 %, because
   one-step-delayed inhibition and saturating counts give a poor winner-take-all.
2. *SPEC findings, below.* The trace-decay floor and the lack of homeostasis
   are what separate rows 1, 2 and 3 above.

Commands (results in `runs/learn/*.json`):
```
C="--n-train 3000 --n-label 1000 --n-test 1000 --tau-pre-shift 1 --stdp-shift 8 --refrac 4 --w-init-max 30 --theta 3000 --v-min -3000 --a-plus 2"
python -m learn.stdp_unsupervised $C --tau-post-shift 3 --inc-post 128 --a-minus 4 --out runs/learn/stdp_spec.json
python -m learn.stdp_unsupervised $C --tau-post-shift 1 --inc-post 255 --a-minus 48 --trace-fix 1 --out runs/learn/stdp_tracefix.json
python -m learn.stdp_unsupervised $C --tau-post-shift 1 --inc-post 255 --a-minus 16 --trace-fix 1 --adaptive 1 --theta-plus 200 --theta-decay-shift 10 --out runs/learn/stdp_tracefix_adaptive.json
```

## 2. On-device adaptation with R-STDP (`rstdp_adapt.py`)

1. The host trains a 256→256→10 network with QAT (3 epochs, `train/model.py`) on emulated N-MNIST
   **without class 9**. Training takes 3 min and the result is cached in `runs/learn/holdout9.npz`.
2. On the chip: the hidden layer is core (0,0), the frozen output layer is core (1,0), and every hidden
   neuron also fans out to the adaptation core (1,1). Neuron 0 of that core is the new
   class head: weights 0, R-STDP enabled, `a_minus = 0`, and a bias of 170 so that it fires
   a little on any input (exploration).
3. After each sample the host sends one REWARD packet. The reward depends on the
   output spikes the host has just read (decision mode):
   * class-9 sample: +32, unless the head already beats the best old neuron by 4;
   * other sample: −24 only if the head would win, otherwise 0.

   Twelve silent drain steps let eligibility decay (`tau_elig_shift = 3`). After
   adaptation, the host writes the head's bias to 0, a neuron-parameter write that turns exploration off.
4. Readout: the old neuron for class 9 is ignored and the head stands in for it.

The adaptation stream is 1 000 training samples, half class 9. The test set is 1 000
balanced test samples (500 class 9 / 500 other). Each run takes 0.5 min.

| Seed | New class before → after | Old classes before → after |
|---|---|---|
| 0 | 0 % → **96.0 %** | 97.4 % → 89.6 % |
| 1 | 0 % → **98.6 %** | 96.4 % → 72.0 % |

The new class is learned reliably. How much accuracy the old classes keep varies a lot between seeds, because
the head steals some old-class samples. The method is fragile:
* With reward by label alone (+ for class 9, − for all others), the head dies in
  every setting tried. Its weights go negative, it stops firing, and it never recovers.
* Changing the margin from 4 to 2 with a −32 reward also kills it.
* In the pinned small test, one of three stream seeds ends with a dead head.

The cause is the reward-rounding asymmetry below, plus the fact that no
eligibility can form without a postsynaptic spike.

```
python -m learn.rstdp_adapt --seed 0 --out runs/learn/rstdp_seed0.json
python -m learn.rstdp_adapt --seed 1 --out runs/learn/rstdp_seed1.json
```

## SPEC findings (for an ADR; nothing was changed)

1. **Traces never decay to zero (§6 step 1).** `x -= x >> k` has fixed points
   1..2^k−1. Every synapse whose neuron ever spiked keeps a permanent trace
   floor, so with stochastic rounding every later pre or post event causes background
   depression or potentiation. In unsupervised STDP this slowly kills every neuron that has
   ever fired. Fix: decay by `max(1, x >> k)` while `x > 0`
   (prototype `--trace-fix`, which gives +18.9 pp). Pinned by
   `test_spec_trace_decay_never_reaches_zero`.
2. **Reward rounding is biased (§6 R-STDP).** `(r * e) >>> reward_shift` rounds
   toward −∞. A small positive product gives 0, a small negative one gives −1, so weak
   eligibility only ever gets depressed. Suggested fix: round to nearest, or use the LFSR's
   stochastic rounding (`sr`). Pinned by `test_reward_rounding_is_floor`.
3. **No per-axon plasticity mask.** In a learning core, recurrent inhibitory
   synapses are plastic too. Here the inhibitory weights stayed within
   [−128, −123] because inhibition follows the spike, but potentiation-heavy settings
   turned them excitatory (up to +119), and the network ran away. Suggested fix: a
   `plastic_axon_max` core register.
4. **No homeostasis.** A per-neuron adaptive threshold (prototype `--adaptive`:
   +`theta_plus` per spike, `>> theta_decay_shift` decay) adds 9.4 pp on top of the trace fix.
   This is a proposed SPEC extension with a prototype only.
5. **Rewards sent while R-STDP is off.** The golden model keeps `pending_reward` when `rstdp_en = 0` or
   `learn_en = 0`, and applies it whenever R-STDP is next switched on. The SPEC says nothing about
   this case; it should say whether such rewards are discarded.
6. **Axon budget.** A 16 × 16 input uses every one of the 256 axons, which leaves none for
   inhibition, so the STDP mapping drops 9 pixels.
