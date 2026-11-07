"""Diehl & Cook-style unsupervised STDP on the SpikeMesh golden chip.

Everything that learns runs inside golden.spikemesh.Chip with SPEC §6 STDP,
so results transfer bit-exactly to RTL that matches the golden model.

Mapping (one core, (0, 0)):
* axons 0 .. P-1 (P = 256 - K): the P most active pixels of the emulated
  N-MNIST 16x16 input (the K least active border pixels are dropped by the
  host to free axons; they are almost always silent).
* axons P .. 255: K inhibitory axons. Every excitatory neuron's fan-out
  sends one SPIKE to each of these K axons of its *own* core (negative
  recurrent fan-out), plus one to the host. Their weights start at w_min
  (strongly negative): any spike inhibits every neuron at t+1, i.e.
  lateral inhibition (soft winner-take-all; the winner inhibits itself too,
  which is harmless right after its own reset).
  Because learning is per core, STDP also touches these weights. Inhibition
  always follows the winner's spike (post-before-pre), so it is depressed
  and stays pinned at w_min; we report its drift.
* N excitatory neurons, fan-out K + 1 entries each (N * (K + 1) <= 1024).

Host protocol per sample: inject the T event frames, run T STEPs (learning
on), then write v = 0 for every neuron (CFG writes), run one silent STEP so
in-flight inhibition drains, write v = 0 again. No weight is ever written by
the host after initialisation.

Labelling and test: learning off (learn_en = 0), each neuron is labelled with
the class it responds to most (mean spike count per class on a labelled
subset); a test sample is classified by the class whose neurons spike most
on average (Diehl & Cook's rule), ties -> lowest class.

Optional (--adaptive, --trace-fix): NOT in SPEC v0.1. Prototypes of proposed
SPEC changes (see learn/README.md), implemented
by subclassing the golden Core here, never by changing golden/.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from golden.spikemesh import Chip, Core, FanoutEntry, HOST_X, sat16
from train.data import emulated_nmnist


@dataclass
class Cfg:
    n_exc: int = 100
    k_inh: int = 9               # inhibitory axons; n_exc * (k_inh + 1) <= 1024
    w_inh: int = -128            # initial weight on every inhibitory axon
    w_init_max: int = 20         # initial excitatory weights ~ U[0, w_init_max]
    theta: int = 600
    leak_shift: int = 3
    reset_mode: int = 1
    refrac: int = 2
    v_min: int = -200
    bias: int = 0                # constant per-neuron bias (host-written once)
    phased: bool = True          # two-phase host schedule (see module docstring)
    comp_steps: int = 1          # complement-phase timesteps
    a_plus: int = 8
    a_minus: int = 3
    stdp_shift: int = 6
    tau_pre_shift: int = 2
    tau_post_shift: int = 2
    inc_pre: int = 64
    inc_post: int = 64
    w_min: int = -128
    w_max: int = 127
    lfsr_seed: int = 0xACE1
    adaptive: bool = False       # proposed extension, not SPEC
    trace_fix: bool = False      # proposed SPEC fix: traces decay all the way to 0
    theta_plus: int = 20         # adaptive: threshold increment per spike
    theta_decay_shift: int = 10  # adaptive: offset -= offset >> shift every step
    seed: int = 0


class ProtoCore(Core):
    """PROTOTYPES of proposed SPEC changes (each behind its own flag).

    adaptive: per-neuron adaptive threshold. theta_eff[n] = theta + th_off[n];
      th_off += theta_plus on a spike, th_off -= th_off >> theta_decay_shift
      every UPDATE. Not part of SPEC v0.1.
    trace_fix: SPEC v0.1 decays traces with x -= x >> k, which never reaches 0:
      every x in [1, 2^k - 1] is a fixed point, so a synapse whose neuron ever
      spiked keeps a permanent trace floor. The fix decays by max(1, x >> k)
      while x > 0. Everything else is the golden LEARN unchanged.
    """

    def __init__(self, *a, adaptive=False, trace_fix=False, theta_plus=20, theta_decay_shift=10, **k):
        super().__init__(*a, **k)
        self.th_off = np.zeros(self.N, dtype=np.int64)
        self.theta_plus, self.theta_decay_shift = theta_plus, theta_decay_shift
        self.adapt = adaptive
        self.trace_fix = trace_fix

    def learn(self):
        R = self.regs
        if not (self.trace_fix and R.learn_en):
            return super().learn()
        kp, kq = R.tau_pre_shift, R.tau_post_shift
        if kp:
            self.x_pre -= np.where(self.x_pre > 0, np.maximum(self.x_pre >> kp, 1), 0)
        if kq:
            self.x_post -= np.where(self.x_post > 0, np.maximum(self.x_post >> kq, 1), 0)
        R.tau_pre_shift = R.tau_post_shift = 0      # golden step 1 then skips decay
        try:
            super().learn()
        finally:
            R.tau_pre_shift, R.tau_post_shift = kp, kq

    def update(self, parity):
        R = self.regs
        acc = self.acc[parity]
        refr = self.refc > 0
        leak = (self.v >> R.leak_shift) if R.leak_shift else np.zeros_like(self.v)
        v2 = sat16(self.v - leak + sat16(acc) + self.bias)
        th = np.minimum(R.theta + self.th_off, 32767)
        spike = (~refr) & (v2 >= th)
        reset_v = (v2 - th) if R.reset_mode == 0 else np.zeros_like(v2)
        new_v = np.where(spike, reset_v, np.maximum(v2, R.v_min))
        self.v = np.where(refr, self.v, new_v)
        self.refc = np.where(refr, self.refc - 1, np.where(spike, R.refrac, 0))
        acc[:] = 0
        if R.learn_en:
            self.post_flag[:] = spike
        if self.adapt:
            self.th_off -= self.th_off >> self.theta_decay_shift
            self.th_off += np.where(spike, self.theta_plus, 0)
        return spike


def pixel_order(x_train: np.ndarray) -> np.ndarray:
    """Pixels sorted by event activity, most active first (host-side choice)."""
    act = x_train.sum((0, 1))
    return np.argsort(-act, kind="stable")


class STDPNet:
    def __init__(self, cfg: Cfg, pixels: np.ndarray):
        self.cfg = c = cfg
        assert c.n_exc * (c.k_inh + 1) <= 1024, "fan-out memory overflow"
        assert c.n_exc <= 256
        self.P = 256 - c.k_inh
        self.pixels = np.asarray(pixels[: self.P])
        self.chip = Chip(1, 1, 256, 256)
        if c.adaptive or c.trace_fix:
            self.chip.cores[(0, 0)] = ProtoCore(0, 0, 256, 256, adaptive=c.adaptive, trace_fix=c.trace_fix,
                                                theta_plus=c.theta_plus,
                                                theta_decay_shift=c.theta_decay_shift)
        k = self.core = self.chip.core(0, 0)
        rng = np.random.default_rng(c.seed)
        for name in ("theta", "leak_shift", "reset_mode", "refrac", "v_min", "lfsr_seed",
                     "w_min", "w_max", "tau_pre_shift", "tau_post_shift", "inc_pre", "inc_post",
                     "stdp_shift"):
            k.set_reg(name, getattr(c, name))
        k.set_reg("stdp_a_plus", c.a_plus)
        k.set_reg("stdp_a_minus", c.a_minus)
        k.w[:] = 0
        k.bias[: c.n_exc] = c.bias
        k.w[: self.P, : c.n_exc] = rng.integers(0, c.w_init_max + 1, (self.P, c.n_exc))
        k.w[self.P:, : c.n_exc] = c.w_inh
        for n in range(c.n_exc):
            ents = [FanoutEntry(0, 0, self.P + j) for j in range(c.k_inh)] + [FanoutEntry(HOST_X, 0, 0)]
            k.set_fanout(n, ents, n * (c.k_inh + 1))
        self.learning(True)

    def learning(self, on: bool):
        self.core.set_reg("learn_en", int(on))
        if isinstance(self.core, ProtoCore):
            self.core.adapt = on and self.cfg.adaptive  # thresholds frozen in labelling / test

    def present(self, frames: np.ndarray) -> np.ndarray:
        """frames: bool (T, 256) pixel events. Returns spike counts (n_exc,)."""
        ch, k, c = self.chip, self.core, self.cfg
        learn = bool(k.regs.learn_en)
        counts = np.zeros(c.n_exc, np.int64)
        ev = frames[:, self.pixels]
        if learn and c.phased:            # pattern phase: Hebbian potentiation only
            k.set_reg("stdp_a_plus", c.a_plus)
            k.set_reg("stdp_a_minus", 0)
        for t in range(len(ev)):
            for a in np.flatnonzero(ev[t]):
                ch.inject(0, 0, int(a))
            for (_, _, n) in ch.step():
                counts[n] += 1
        k.v[:] = 0          # host CFG writes: neuron field 0 (v) for every neuron
        if learn and c.phased:
            # complement phase: no firing (theta = max), depression only. Events on
            # every pixel axon that stayed silent in this sample and on the inhibitory
            # axons: "pre after post" depresses exactly the recent winners' weights.
            k.set_reg("theta", 32767)
            k.set_reg("stdp_a_plus", 0)
            k.set_reg("stdp_a_minus", c.a_minus)
            comp = np.concatenate([np.flatnonzero(~ev.any(0)), np.arange(self.P, 256)])
            for _ in range(c.comp_steps):
                for a in comp:
                    ch.inject(0, 0, int(a))
                ch.step()
                k.v[:] = 0
            k.set_reg("theta", c.theta)
        ch.step()           # drain in-flight inhibition (no input, v <= 0: silent)
        k.v[:] = 0
        return counts

    def run(self, xs: np.ndarray) -> np.ndarray:
        return np.stack([self.present(x) for x in xs])

    def exc_weights(self):
        return self.core.w[: self.P, : self.cfg.n_exc]

    def inh_weights(self):
        return self.core.w[self.P:, : self.cfg.n_exc]


def assign_labels(counts: np.ndarray, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    rates = np.stack([counts[y == c].mean(0) if (y == c).any() else np.zeros(counts.shape[1])
                      for c in range(n_classes)])
    lab = rates.argmax(0)
    lab[rates.max(0) == 0] = -1          # never-firing neurons get no label
    return lab


def classify(counts: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    score = np.full((len(counts), n_classes), -1.0)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            score[:, c] = counts[:, m].mean(1)
    return score.argmax(1)


def experiment(cfg: Cfg, n_train: int, n_label: int, n_test: int, epochs: int = 1, log=print):
    xtr, ytr = emulated_nmnist("train")
    xte, yte = emulated_nmnist("test")
    rng = np.random.default_rng(cfg.seed)
    net = STDPNet(cfg, pixel_order(xtr[:10000]))
    t0 = time.time()
    tr_idx = rng.permutation(len(xtr) - 10000)[:n_train]         # learning set
    lab_idx = len(xtr) - 10000 + np.arange(n_label)                # disjoint labelling set
    hist = []
    for ep in range(epochs):
        net.learning(True)
        tot = 0
        win = np.zeros(cfg.n_exc, np.int64)
        for i, j in enumerate(tr_idx):
            cnt = net.present(xtr[j])
            tot += cnt.sum()
            win += cnt
            if (i + 1) % max(1, n_train // 5) == 0:
                iw = net.inh_weights()
                log(f"  ep{ep} {i + 1}/{n_train}  spikes/sample {tot / (i + 1):.1f}  "
                    f"active neurons {(win > 0).sum()}  top-share {win.max() / max(1, win.sum()):.2f}  "
                    f"exc w mean {net.exc_weights().mean():.1f}  inh w mean {iw.mean():.1f} max {iw.max()}  "
                    f"{(time.time() - t0) / 60:.1f} min")
                win[:] = 0
        hist.append(tot / n_train)
    net.learning(False)
    c_lab = net.run(xtr[lab_idx])
    labels = assign_labels(c_lab, ytr[lab_idx])
    c_te = net.run(xte[:n_test])
    pred = classify(c_te, labels)
    acc = float((pred == yte[:n_test]).mean())
    res = dict(cfg=asdict(cfg), n_train=n_train, n_label=n_label, n_test=n_test, epochs=epochs,
               test_acc=acc, label_acc=float((classify(c_lab, labels) == ytr[lab_idx]).mean()),
               labels_hist=np.bincount(labels[labels >= 0], minlength=10).tolist(),
               unlabelled=int((labels < 0).sum()),
               test_spikes_per_sample=float(c_te.sum(1).mean()),
               silent_test_samples=int((c_te.sum(1) == 0).sum()),
               inh_w_range=[int(net.inh_weights().min()), int(net.inh_weights().max())],
               exc_w_mean=float(net.exc_weights().mean()),
               minutes=(time.time() - t0) / 60)
    return res, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f, v in asdict(Cfg()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v) if not isinstance(v, bool) else int,
                        default=v)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-label", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = vars(ap.parse_args(argv))
    cfg = Cfg(**{f: (bool(a[f]) if isinstance(v, bool) else a[f]) for f, v in asdict(Cfg()).items()})
    res, net = experiment(cfg, a["n_train"], a["n_label"], a["n_test"], a["epochs"])
    print(json.dumps(res))
    if a["out"]:
        from pathlib import Path
        Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(a["out"]).write_text(json.dumps(res, indent=1))
        np.save(Path(a["out"]).with_suffix(".w.npy"), net.core.w)
    return res


if __name__ == "__main__":
    main()
