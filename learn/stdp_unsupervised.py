"""Diehl & Cook-style unsupervised STDP on the SpikeMesh golden chip (SPEC v0.2).

Everything that learns runs inside golden.spikemesh.Chip with SPEC §6/§9
STDP, so results transfer bit-exactly to RTL that matches the golden model.
The host never writes a weight after initialisation.

Excitatory cores: `n_cores` cores (1 or 2) at (0,0), (1,0), each with
`n_exc` neurons. Axons 0..P-1 (P = 256 - k_inh) carry the P most active
pixels of the 16x16 emulated N-MNIST input (the host drops the least active
border pixels to free axons, and injects every event into every excitatory
core). Axons P..255 are k_inh inhibitory axons whose weights are fixed at
`w_inh` by `plastic_axon_max = P` (SPEC §9 rule 3), so STDP cannot touch them.

Lateral inhibition (`inh = "direct"`, 1 core): every excitatory neuron's
fan-out sends one SPIKE to each inhibitory axon of its own core (plus one to
the host), so any spike inhibits all neurons at t+1. Needs
n_exc * (k_inh + 1) <= 1024.

Lateral inhibition (`inh = "core"`, any number of cores): every excitatory
neuron sends one SPIKE to an inhibitory core at (0,1) (plus one to the host).
That core has `n_inh` neurons; neuron j has weight 127 from every excitatory
axon, theta = 127 and bias -127 * j, so it fires when at least j+1 excitatory
neurons spiked in the previous step (graded inhibition), and fans out to
every inhibitory axon of every excitatory core. Inhibition arrives at t+2.

Host schedule per sample (`phased`): pattern phase (T steps, a_minus = 0,
Hebbian potentiation only), then a complement phase (comp_steps steps with
theta = 32767 and a_plus = 0: events on every pixel that stayed silent, so
"pre after post" depresses the recent winners' weights on pixels outside the
pattern, Diehl & Cook's x_tar normalisation done by on-chip STDP). Then v = 0
(CFG writes), silent drain steps, v = 0 again.

Adaptive threshold (SPEC §9 rule 4): `th_inc` / `th_decay_shift` registers
during training; for labelling and test the host writes both to 0, which
freezes theta_a (Diehl & Cook also freeze theta at test time).

Labelling / test: learning off; each neuron labelled with the class of its
highest mean response on a labelled subset; a sample is classified by the
class whose neurons spike most on average (ties -> lowest class).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from golden.spikemesh import Chip, FanoutEntry, HOST_X
from train.data import emulated_nmnist

INH_CORE = (0, 1)


@dataclass
class Cfg:
    n_exc: int = 100             # excitatory neurons per core
    n_cores: int = 1
    inh: str = "direct"          # "direct" (recurrent fan-out) or "core" (inhibitory core)
    n_inh: int = 8               # inh = "core": inhibitory neurons
    k_inh: int = 9               # inhibitory axons per excitatory core
    w_inh: int = -128
    plastic_inh: bool = True     # plastic_axon_max = P (SPEC v0.2); False: inhibition is plastic
    w_init_max: int = 30
    theta: int = 3000
    leak_shift: int = 3
    reset_mode: int = 1
    refrac: int = 4
    v_min: int = -3000
    bias: int = 0
    phased: bool = True
    comp_steps: int = 1
    drain: int = 1
    a_plus: int = 2
    a_minus: int = 48
    stdp_shift: int = 8
    tau_pre_shift: int = 1
    tau_post_shift: int = 1
    inc_pre: int = 64
    inc_post: int = 255
    w_min: int = -128
    w_max: int = 127
    lfsr_seed: int = 0xACE1
    th_inc: int = 0              # SPEC §9 adaptive threshold (0 = off)
    th_decay_shift: int = 10
    test_theta: int = 0          # > 0: host writes this base theta for labelling / test
    seed: int = 0


def pixel_order(x_train: np.ndarray) -> np.ndarray:
    """Pixels sorted by event activity, most active first (host-side choice)."""
    return np.argsort(-x_train.sum((0, 1)), kind="stable")


class STDPNet:
    def __init__(self, cfg: Cfg, pixels: np.ndarray):
        self.cfg = c = cfg
        assert c.inh in ("direct", "core") and 1 <= c.n_cores <= 2
        assert c.n_exc <= 256
        if c.inh == "direct":
            assert c.n_cores == 1, "direct inhibition only within one core"
            assert c.n_exc * (c.k_inh + 1) <= 1024, "fan-out memory overflow"
        else:
            assert c.n_exc * 2 <= 1024 and c.n_exc * c.n_cores <= 256
            assert c.n_inh * c.k_inh * c.n_cores <= 1024 and c.k_inh * c.n_cores < 64
        self.P = 256 - c.k_inh
        self.pixels = np.asarray(pixels[: self.P])
        self.exc_xy = [(0, 0), (1, 0)][: c.n_cores]
        self.chip = Chip(2, 2, 256, 256)
        rng = np.random.default_rng(c.seed)
        self.cores = [self.chip.core(*xy) for xy in self.exc_xy]
        for ci, k in enumerate(self.cores):
            for name in ("theta", "leak_shift", "reset_mode", "refrac", "v_min", "w_min", "w_max",
                         "tau_pre_shift", "tau_post_shift", "inc_pre", "inc_post", "stdp_shift",
                         "th_decay_shift"):
                k.set_reg(name, getattr(c, name))
            k.set_reg("lfsr_seed", c.lfsr_seed + ci)
            k.set_reg("stdp_a_plus", c.a_plus)
            k.set_reg("stdp_a_minus", c.a_minus)
            k.set_reg("plastic_axon_max", self.P if c.plastic_inh else 0)
            k.w[:] = 0
            k.w[: self.P, : c.n_exc] = rng.integers(0, c.w_init_max + 1, (self.P, c.n_exc))
            k.w[self.P:, : c.n_exc] = c.w_inh
            k.bias[: c.n_exc] = c.bias
            k.theta_a[:] = 0
            for n in range(c.n_exc):
                if c.inh == "direct":
                    ents = [FanoutEntry(*self.exc_xy[ci], self.P + j) for j in range(c.k_inh)]
                else:
                    ents = [FanoutEntry(*INH_CORE, ci * c.n_exc + n)]
                k.set_fanout(n, ents + [FanoutEntry(HOST_X, 0, 0)], n * (len(ents) + 1))
        if c.inh == "core":
            ik = self.chip.core(*INH_CORE)
            ik.set_reg("theta", 127)
            ik.set_reg("reset_mode", 1)
            ik.w[: c.n_exc * c.n_cores, : c.n_inh] = 127
            ik.bias[: c.n_inh] = -127 * np.arange(c.n_inh)
            for j in range(c.n_inh):
                ents = [FanoutEntry(*xy, self.P + a) for xy in self.exc_xy for a in range(c.k_inh)]
                ik.set_fanout(j, ents, j * len(ents))
        self.learning(True)

    def learning(self, on: bool):
        for k in self.cores:
            k.set_reg("learn_en", int(on))
            # adaptive threshold active only while learning (host freezes it for test)
            k.set_reg("th_inc", self.cfg.th_inc if on else 0)
            k.set_reg("th_decay_shift", self.cfg.th_decay_shift if on else 0)
            if self.cfg.test_theta:
                k.set_reg("theta", self.cfg.theta if on else self.cfg.test_theta)

    def _zero_v(self):
        for k in self.chip.cores.values():
            k.v[:] = 0            # host CFG writes: neuron field 0 (v)

    def present(self, frames: np.ndarray) -> np.ndarray:
        """frames: bool (T, 256) pixel events. Returns spike counts (n_cores * n_exc,)."""
        ch, c = self.chip, self.cfg
        learn = bool(self.cores[0].regs.learn_en)
        counts = np.zeros(c.n_cores * c.n_exc, np.int64)
        ev = frames[:, self.pixels]
        phased = learn and c.phased
        if phased:
            for k in self.cores:
                k.set_reg("stdp_a_plus", c.a_plus)
                k.set_reg("stdp_a_minus", 0)
        idx = {xy: i for i, xy in enumerate(self.exc_xy)}

        def step():
            for (sx, sy, n) in ch.step():
                ci = idx.get((sx, sy))
                if ci is not None:
                    counts[ci * c.n_exc + n] += 1

        for t in range(len(ev)):
            for a in np.flatnonzero(ev[t]):
                for xy in self.exc_xy:
                    ch.inject(*xy, int(a))
            step()
        self._zero_v()
        if phased:
            for k in self.cores:
                k.set_reg("theta", 32767)
                k.set_reg("stdp_a_plus", 0)
                k.set_reg("stdp_a_minus", c.a_minus)
            comp = np.flatnonzero(~ev.any(0))
            if not c.plastic_inh:      # v0.1 style: also depress the (plastic) inhibitory weights
                comp = np.concatenate([comp, np.arange(self.P, 256)])
            for _ in range(c.comp_steps):
                for a in comp:
                    for xy in self.exc_xy:
                        ch.inject(*xy, int(a))
                ch.step()
                self._zero_v()
            for k in self.cores:
                k.set_reg("theta", c.theta)     # (phased only runs while learning)
        for _ in range(c.drain + (1 if c.inh == "core" else 0)):
            ch.step()             # silent: in-flight inhibition drains, v <= 0
            self._zero_v()
        return counts

    def run(self, xs: np.ndarray) -> np.ndarray:
        return np.stack([self.present(x) for x in xs])

    def exc_weights(self):
        return np.concatenate([k.w[: self.P, : self.cfg.n_exc] for k in self.cores], 1)

    def inh_weights(self):
        return np.concatenate([k.w[self.P:, : self.cfg.n_exc] for k in self.cores], 1)


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
    for ep in range(epochs):
        net.learning(True)
        tot, win = 0, 0
        for i, j in enumerate(tr_idx):
            cnt = net.present(xtr[j])
            tot += cnt.sum()
            win = win + cnt
            if (i + 1) % max(1, n_train // 5) == 0:
                log(f"  ep{ep} {i + 1}/{n_train}  spikes/sample {tot / (i + 1):.1f}  "
                    f"active neurons {(np.asarray(win) > 0).sum()}  "
                    f"exc w mean {net.exc_weights().mean():.1f}  "
                    f"theta_a mean {np.mean([k.theta_a[:cfg.n_exc].mean() for k in net.cores]):.0f}  "
                    f"{(time.time() - t0) / 60:.1f} min")
                win = 0
    net.learning(False)
    c_lab = net.run(xtr[lab_idx])
    labels = assign_labels(c_lab, ytr[lab_idx])
    c_te = net.run(xte[:n_test])
    pred = classify(c_te, labels)
    res = dict(cfg=asdict(cfg), n_train=n_train, n_label=n_label, n_test=n_test, epochs=epochs,
               test_acc=float((pred == yte[:n_test]).mean()),
               label_acc=float((classify(c_lab, labels) == ytr[lab_idx]).mean()),
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
        typ = int if isinstance(v, bool) else type(v)
        ap.add_argument("--" + f.replace("_", "-"), type=typ, default=v)
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
        Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(a["out"]).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
