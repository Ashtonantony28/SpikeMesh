"""On-device adaptation with R-STDP: learn a held-out digit class on the chip.

1. Train (host, QAT, train/model.py) a 256 -> 256 -> 10 network on emulated
   N-MNIST with one class (default 9) removed from the training set. Its
   output neuron for that class never learns to fire.
2. Map it onto the golden chip: hidden layer on core (0,0), output layer on
   core (1,0) (frozen, learn_en = 0). Every hidden neuron also fans out to a
   third core (1,1), the *adaptation core*, whose neuron 0 is the new class
   head. Its weights start at 0; it has a small positive bias so that it
   fires a little on every input (exploration).
3. Adaptation (only on-chip learning; the host never writes a weight):
   core (1,1) runs R-STDP (SPEC §6, rstdp_en = 1). After each sample the host
   sends one CTRL REWARD packet to core (1,1):
       r = +R_pos if the sample is the new class, -R_neg otherwise.
   STDP pairing (hidden spike -> head spike) is written to the eligibility
   memory; the reward converts it into weight changes at the next LEARN.
   Depression (a_minus) is 0, so eligibility is non-negative and the reward
   sign decides the direction. The host then runs `drain` silent steps so
   eligibility decays (tau_elig_shift) before the next sample, and zeroes v.
4. Readout (host): spike counts of the old output neurons, with the
   untrained neuron of the held-out class replaced by the head's count;
   argmax, ties -> lowest class index.

Reported: accuracy on the new class and on the old classes, before and after
adaptation, on emulated N-MNIST test samples.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from compiler.netir import NetIR
from golden.spikemesh import Chip, FanoutEntry, HOST_X
from train.data import emulated_nmnist
from train.export import export_netir
from train.model import SpikeMeshNet, count_ce_loss

REPO = Path(__file__).resolve().parent.parent
HID, OUT, NEW = (0, 0), (1, 0), (1, 1)


# ----------------------------------------------------------------- host training
def train_holdout(holdout: int = 9, epochs: int = 3, hidden: int = 256, limit: int | None = None,
                  threads: int = 2, seed: int = 0, log=print) -> NetIR:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    x, y = emulated_nmnist("train")
    keep = y != holdout
    x, y = x[keep], y[keep]
    if limit:
        x, y = x[:limit], y[:limit]
    m = SpikeMeshNet([256, hidden, 10], [2, 2], [0, 0], [0, 0], [-1.0, -1.0],
                     init_std=[0.15, 0.15])
    bs, lr = 128, 5e-3
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(len(x))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            r, v2 = m(torch.from_numpy(x[idx]).float(), quant=ep >= 1 or epochs == 1)
            loss = count_ce_loss(r[-1], v2, torch.from_numpy(y[idx]), 0.3, 0.5)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        log(f"  holdout-{holdout} training epoch {ep} done, {(time.time() - t0) / 60:.1f} min")
    return export_netir(m, meta={"holdout_class": holdout, "epochs": epochs, "limit": limit})


# ----------------------------------------------------------------- chip
@dataclass
class AdaptCfg:
    holdout: int = 9
    head_theta: int = 600
    head_bias: int = 170         # exploration: v -> 4*bias > theta, so it fires on any input
    head_leak: int = 2
    a_plus: int = 16
    stdp_shift: int = 4
    tau_pre_shift: int = 1
    tau_post_shift: int = 1
    inc_pre: int = 64
    inc_post: int = 64
    tau_elig_shift: int = 3
    reward_pos: int = 32
    reward_neg: int = 24
    reward_shift: int = 8
    reward_mode: str = "decision"  # "decision" or "label" (see adapt())
    margin: int = 4              # decision mode: stop rewarding once the head wins by this much
    eval_bias: int = 0           # >= 0: host rewrites the head bias after adaptation (exploration off)
    drain: int = 12              # silent steps after each sample (eligibility decays)
    lfsr_seed: int = 0x1D0F


class AdaptChip:
    def __init__(self, net: NetIR, cfg: AdaptCfg):
        assert len(net.layers) == 2
        self.net, self.cfg = net, cfg
        self.chip = ch = Chip(2, 2, 256, 256)
        L0, L1 = net.layers
        for L, xy in ((L0, HID), (L1, OUT)):
            k = ch.core(*xy)
            k.w[: L.n_in, : L.n_out] = L.weight
            k.bias[: L.n_out] = L.bias
            for name in ("theta", "leak_shift", "reset_mode", "refrac", "v_min"):
                k.set_reg(name, getattr(L, name))
        hk = ch.core(*HID)
        for n in range(L0.n_out):
            hk.set_fanout(n, [FanoutEntry(*OUT, n), FanoutEntry(*NEW, n)], 2 * n)
        ok = ch.core(*OUT)
        for n in range(L1.n_out):
            ok.set_fanout(n, [FanoutEntry(HOST_X, 0, 0)], n)
        self.nk = nk = ch.core(*NEW)
        nk.w[:] = 0
        nk.bias[0] = cfg.head_bias
        for name, v in dict(theta=cfg.head_theta, leak_shift=cfg.head_leak, reset_mode=0, refrac=0,
                            v_min=-cfg.head_theta, learn_en=1, rstdp_en=1, stdp_a_plus=cfg.a_plus,
                            stdp_a_minus=0, stdp_shift=cfg.stdp_shift, tau_pre_shift=cfg.tau_pre_shift,
                            tau_post_shift=cfg.tau_post_shift, inc_pre=cfg.inc_pre, inc_post=cfg.inc_post,
                            tau_elig_shift=cfg.tau_elig_shift, reward_shift=cfg.reward_shift,
                            lfsr_seed=cfg.lfsr_seed, w_min=-128, w_max=127).items():
            nk.set_reg(name, v)
        nk.set_fanout(0, [FanoutEntry(HOST_X, 0, 0)], 0)   # only neuron 0 is used
        self.n_old = L1.n_out

    def learning(self, on: bool):
        self.nk.set_reg("learn_en", int(on))

    def present(self, frames: np.ndarray, reward=None) -> np.ndarray:
        """Returns counts: [old output 0..9, head] (11,).

        reward: None, an int, or a function counts -> int (the host decides the
        reward from the output spikes it has just received)."""
        ch = self.chip
        counts = np.zeros(self.n_old + 1, np.int64)
        for t in range(len(frames)):
            for a in np.flatnonzero(frames[t]):
                ch.inject(*HID, int(a))
            for (sx, sy, n) in ch.step():
                counts[n if (sx, sy) == OUT else self.n_old] += 1
        # two more steps let hidden spikes of the last frames reach the outputs
        for _ in range(2):
            for (sx, sy, n) in ch.step():
                counts[n if (sx, sy) == OUT else self.n_old] += 1
        if callable(reward):
            reward = reward(counts)
        if reward:
            ch.reward(*NEW, int(reward))     # CTRL REWARD packet, applied at next LEARN
        for k in ch.cores.values():
            k.v[:] = 0                       # host CFG writes between samples
        for _ in range(self.cfg.drain):
            ch.step()                        # silent; eligibility decays; reward applied
            for k in ch.cores.values():
                k.v[:] = 0
        return counts

    def predict(self, counts: np.ndarray) -> np.ndarray:
        """Old neurons except the held-out one, plus the head standing in for it."""
        c = counts.copy()
        h = self.cfg.holdout
        c[..., h] = c[..., self.n_old]
        return c[..., : self.n_old].argmax(-1)


def evaluate(ac: AdaptChip, x, y):
    ac.learning(False)
    counts = np.stack([ac.present(xi) for xi in x])
    pred = ac.predict(counts)
    new = y == ac.cfg.holdout
    return dict(acc_new=float((pred[new] == y[new]).mean()), acc_old=float((pred[~new] == y[~new]).mean()),
                acc_all=float((pred == y).mean()),
                head_spikes_new=float(counts[new, -1].mean()), head_spikes_old=float(counts[~new, -1].mean()))


def adapt(ac: AdaptChip, x, y, log=print):
    """R-STDP adaptation. Reward per sample, computed by the host from the spikes
    it receives (it already sees every output spike for the readout):

    label mode:    +reward_pos on new-class samples, -reward_neg on old ones.
    decision mode: new-class sample: +reward_pos unless the head already beats
                   the best old neuron by `margin`; old-class sample:
                   -reward_neg only if the head would win (count >= best old),
                   else 0. Punishing only harmful firing keeps the head alive.
    """
    ac.learning(True)
    c = ac.cfg
    h, n_old = c.holdout, ac.n_old

    def reward_for(label):
        def f(counts):
            head = counts[n_old]
            old = np.delete(counts[:n_old], h).max()
            if c.reward_mode == "label":
                return c.reward_pos if label == h else -c.reward_neg
            if label == h:
                return c.reward_pos if head < old + c.margin else 0
            return -c.reward_neg if head >= old else 0
        return f

    stats = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        cnt = ac.present(xi, reward=reward_for(int(yi)))
        stats.append(ac.predict(cnt) == yi)
        if (i + 1) % max(1, len(x) // 4) == 0:
            w = ac.nk.w[: ac.net.layers[0].n_out, 0]
            log(f"  adapt {i + 1}/{len(x)}  running acc {np.mean(stats[-len(x) // 4:]):.3f}  "
                f"head w: mean {w.mean():.1f} min {w.min()} max {w.max()}")


def balanced_stream(x, y, holdout, n, rng):
    """n samples, half from the new class, half from old classes, shuffled."""
    new = rng.permutation(np.flatnonzero(y == holdout))[: n // 2]
    old = rng.permutation(np.flatnonzero(y != holdout))[: n - len(new)]
    idx = rng.permutation(np.concatenate([new, old]))
    return x[idx], y[idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f, v in asdict(AdaptCfg()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--net", default=None, help="cached holdout NetIR (.npz); trained if missing")
    ap.add_argument("--n-adapt", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000, help="test samples (balanced 50/50 new/old)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = vars(ap.parse_args(argv))
    cfg = AdaptCfg(**{f: a[f] for f in asdict(AdaptCfg())})
    t0 = time.time()
    netp = Path(a["net"] or REPO / "runs" / "learn" / f"holdout{cfg.holdout}.npz")
    if netp.exists():
        net = NetIR.load(netp)
    else:
        net = train_holdout(cfg.holdout, a["train_epochs"], limit=a["train_limit"])
        netp.parent.mkdir(parents=True, exist_ok=True)
        net.save(netp)
    rng = np.random.default_rng(a["seed"])
    xtr, ytr = emulated_nmnist("train")
    xte, yte = emulated_nmnist("test")
    xa, ya = balanced_stream(xtr, ytr, cfg.holdout, a["n_adapt"], rng)
    xt, yt = balanced_stream(xte, yte, cfg.holdout, a["n_test"], rng)
    ac = AdaptChip(net, cfg)
    before = evaluate(ac, xt, yt)
    print("before", json.dumps(before), flush=True)
    adapt(ac, xa, ya)
    if cfg.eval_bias >= 0:
        ac.nk.bias[0] = cfg.eval_bias        # one CFG write (neuron field 1), not a weight
    after = evaluate(ac, xt, yt)
    print("after ", json.dumps(after), flush=True)
    res = dict(cfg=asdict(cfg), n_adapt=a["n_adapt"], n_test=a["n_test"], before=before, after=after,
               minutes=(time.time() - t0) / 60, net=str(netp))
    print(json.dumps(res))
    if a["out"]:
        Path(a["out"]).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
