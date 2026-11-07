"""Integer golden model of the SpikeMesh chip (docs/SPEC.md v0.1).

Timestep-exact, not cycle-exact. This is the oracle that the RTL and the
PyTorch trainer are compared against. Every arithmetic step mirrors SPEC.md;
section numbers are cited inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOST_X = 15


def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def sat16(x):
    return sat(x, -32768, 32767)


class LFSR16:
    """16-bit Galois LFSR, taps 0xB400 (SPEC §6)."""

    def __init__(self, seed: int):
        self.state = (seed & 0xFFFF) or 1

    def next(self) -> int:
        s = self.state
        lsb = s & 1
        s >>= 1
        if lsb:
            s ^= 0xB400
        self.state = s
        return s

    # The maximal-length sequence is cached once: _SEQ[i + 1] = next(_SEQ[i]),
    # _SEQ[0] = 1, and _POS[state] = i. take(k) is then a slice of the cycle
    # (pure speed-up; identical to calling next() k times, see tests/learn).
    _SEQ: np.ndarray | None = None
    _POS: np.ndarray | None = None
    PERIOD = 65535

    @classmethod
    def _cycle(cls):
        if cls._SEQ is None:
            seq = np.empty(cls.PERIOD, dtype=np.int64)
            s = 1
            for i in range(cls.PERIOD):
                seq[i] = s
                s = (s >> 1) ^ (0xB400 if s & 1 else 0)
            pos = np.full(1 << 16, -1, dtype=np.int64)
            pos[seq] = np.arange(cls.PERIOD)
            cls._SEQ, cls._POS = seq, pos
        return cls._SEQ, cls._POS

    def take(self, k: int) -> np.ndarray:
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        seq, pos = self._cycle()
        i = int(pos[self.state])
        out = seq[(i + 1 + np.arange(k)) % self.PERIOD]
        self.state = int(out[-1])
        return out


@dataclass
class CoreRegs:
    theta: int = 64
    leak_shift: int = 0
    reset_mode: int = 0
    refrac: int = 0
    v_min: int = -32768
    learn_en: int = 0
    stdp_a_plus: int = 0
    stdp_a_minus: int = 0
    stdp_shift: int = 0
    tau_pre_shift: int = 0
    tau_post_shift: int = 0
    inc_pre: int = 0
    inc_post: int = 0
    lfsr_seed: int = 1
    w_min: int = -128
    w_max: int = 127
    rstdp_en: int = 0
    tau_elig_shift: int = 0
    reward_shift: int = 0


REG_INDEX = {
    0: "theta", 1: "leak_shift", 2: "reset_mode", 3: "refrac", 4: "v_min",
    8: "learn_en", 9: "stdp_a_plus", 10: "stdp_a_minus", 11: "stdp_shift",
    12: "tau_pre_shift", 13: "tau_post_shift", 14: "inc_pre", 15: "inc_post",
    16: "lfsr_seed", 17: "w_min", 18: "w_max", 19: "rstdp_en",
    20: "tau_elig_shift", 21: "reward_shift",
}
SIGNED_REGS = {"theta", "v_min", "w_min", "w_max"}


def _s16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


@dataclass
class FanoutEntry:
    dest_x: int
    dest_y: int
    dest_axon: int


class Core:
    def __init__(self, x: int, y: int, n_neurons: int = 256, n_axons: int = 256, w_bits: int = 8):
        self.x, self.y = x, y
        self.N, self.A, self.w_bits = n_neurons, n_axons, w_bits
        self.regs = CoreRegs()
        if w_bits != 8:
            self.regs.w_min = -(1 << (w_bits - 1))
            self.regs.w_max = (1 << (w_bits - 1)) - 1
        self.w = np.zeros((n_axons, n_neurons), dtype=np.int64)
        self.v = np.zeros(n_neurons, dtype=np.int64)
        self.bias = np.zeros(n_neurons, dtype=np.int64)
        self.refc = np.zeros(n_neurons, dtype=np.int64)
        self.acc = np.zeros((2, n_neurons), dtype=np.int64)
        # fan-out: per neuron (base, count) into a flat entry list
        self.fo_base = np.zeros(n_neurons, dtype=np.int64)
        self.fo_count = np.zeros(n_neurons, dtype=np.int64)
        self.fanout: dict[int, FanoutEntry] = {}
        # learning state (§6)
        self.x_pre = np.zeros(n_axons, dtype=np.int64)
        self.x_post = np.zeros(n_neurons, dtype=np.int64)
        self.pre_flag = np.zeros(n_axons, dtype=bool)
        self.post_flag = np.zeros(n_neurons, dtype=bool)
        self.elig = np.zeros((n_axons, n_neurons), dtype=np.int64)
        self.pending_reward = 0
        self.lfsr = LFSR16(self.regs.lfsr_seed)

    # ---------------- configuration (§5) ----------------
    def set_reg(self, name: str, value: int):
        setattr(self.regs, name, int(value))
        if name == "lfsr_seed":
            self.lfsr = LFSR16(int(value))

    def set_fanout(self, neuron: int, entries: list[FanoutEntry], base: int):
        assert len(entries) < 64
        self.fo_base[neuron] = base
        self.fo_count[neuron] = len(entries)
        for i, e in enumerate(entries):
            self.fanout[base + i] = e

    def cfg_write(self, addr: int, data: int):
        region = (addr >> 20) & 3
        low = addr & 0xFFFFF
        if region == 0:
            a, n = divmod(low, self.N)
            wmask = (1 << self.w_bits) - 1
            val = data & wmask
            if val & (1 << (self.w_bits - 1)):
                val -= 1 << self.w_bits
            self.w[a, n] = val
        elif region == 1:
            n, f = low >> 2, low & 3
            if f == 0:
                self.v[n] = _s16(data)
            elif f == 1:
                self.bias[n] = _s16(data)
            elif f == 2:
                self.refc[n] = data & 0xF
            else:
                self.fo_base[n] = data & 0x3FF
                self.fo_count[n] = (data >> 10) & 0x3F
        elif region == 2:
            idx, word = low >> 1, low & 1
            e = self.fanout.setdefault(idx, FanoutEntry(0, 0, 0))
            if word == 0:
                e.dest_x = (data >> 12) & 0xF
                e.dest_y = (data >> 8) & 0xF
            else:
                e.dest_axon = data & 0x3FF
        else:
            r = low & 0xFF
            if r in REG_INDEX:
                name = REG_INDEX[r]
                val = _s16(data) if name in SIGNED_REGS else data & 0xFFFF
                self.set_reg(name, val)

    def cfg_read(self, addr: int) -> int:
        region = (addr >> 20) & 3
        low = addr & 0xFFFFF
        if region == 0:
            a, n = divmod(low, self.N)
            return int(self.w[a, n]) & 0xFFFF
        if region == 1:
            n, f = low >> 2, low & 3
            vals = [self.v[n] & 0xFFFF, self.bias[n] & 0xFFFF, self.refc[n],
                    (self.fo_count[n] << 10) | self.fo_base[n]]
            return int(vals[f])
        if region == 2:
            e = self.fanout.get(low >> 1, FanoutEntry(0, 0, 0))
            return (e.dest_x << 12 | e.dest_y << 8) if (low & 1) == 0 else e.dest_axon
        r = low & 0xFF
        return int(getattr(self.regs, REG_INDEX[r])) & 0xFFFF if r in REG_INDEX else 0

    # ---------------- runtime ----------------
    def receive_spike(self, axon: int, parity: int):
        self.acc[parity] += self.w[axon]
        if self.regs.learn_en:
            self.pre_flag[axon] = True

    def receive_reward(self, r: int):
        self.pending_reward += r

    def _sr(self, m: np.ndarray) -> np.ndarray:
        s = self.regs.stdp_shift
        rnd = self.lfsr.take(m.size)
        if s == 0:
            return m.copy()
        mask = (1 << s) - 1
        return (m >> s) + ((rnd & mask) < (m & mask)).astype(np.int64)

    def learn(self):
        """LEARN(t), SPEC §6."""
        R = self.regs
        if not R.learn_en:
            return
        # 1. trace decay (and eligibility decay)
        if R.tau_pre_shift:
            self.x_pre -= self.x_pre >> R.tau_pre_shift
        if R.tau_post_shift:
            self.x_post -= self.x_post >> R.tau_post_shift
        if R.rstdp_en:
            if R.tau_elig_shift:
                self.elig -= self.elig >> R.tau_elig_shift
            if self.pending_reward:
                dw = (self.pending_reward * self.elig) >> R.reward_shift
                self.w = sat(self.w + dw, R.w_min, R.w_max)
                self.pending_reward = 0
        target = self.elig if R.rstdp_en else self.w
        lo, hi = (-32768, 32767) if R.rstdp_en else (R.w_min, R.w_max)
        # 2. depression, rows ascending (one LFSR draw per weight, row-major;
        #    every weight is visited once, so the rows are done together)
        rows = np.flatnonzero(self.pre_flag)
        if rows.size:
            d = self._sr(np.tile(R.stdp_a_minus * self.x_post, rows.size)).reshape(rows.size, self.N)
            target[rows] = sat(target[rows] - d, lo, hi)
        # 3. potentiation, columns ascending (draws column-major)
        cols = np.flatnonzero(self.post_flag)
        if cols.size:
            d = self._sr(np.tile(R.stdp_a_plus * self.x_pre, cols.size)).reshape(cols.size, self.A)
            target[:, cols] = sat(target[:, cols] + d.T, lo, hi)
        # 4. trace bumps
        self.x_pre[self.pre_flag] = np.minimum(self.x_pre[self.pre_flag] + R.inc_pre, 255)
        self.x_post[self.post_flag] = np.minimum(self.x_post[self.post_flag] + R.inc_post, 255)
        self.pre_flag[:] = False
        self.post_flag[:] = False

    def update(self, parity: int) -> np.ndarray:
        """UPDATE(t), SPEC §2. Returns boolean spike vector."""
        R = self.regs
        acc = self.acc[parity]
        refr = self.refc > 0
        leak = (self.v >> R.leak_shift) if R.leak_shift else np.zeros_like(self.v)
        v2 = sat16(self.v - leak + sat16(acc) + self.bias)
        spike = (~refr) & (v2 >= R.theta)
        reset_v = (v2 - R.theta) if R.reset_mode == 0 else np.zeros_like(v2)
        new_v = np.where(spike, reset_v, np.maximum(v2, R.v_min))
        self.v = np.where(refr, self.v, new_v)
        self.refc = np.where(refr, self.refc - 1, np.where(spike, R.refrac, 0))
        acc[:] = 0
        if R.learn_en:
            self.post_flag[:] = spike
        return spike


class Chip:
    def __init__(self, mesh_x=2, mesh_y=2, n_neurons=256, n_axons=256, w_bits=8):
        self.mx, self.my = mesh_x, mesh_y
        self.cores = {(x, y): Core(x, y, n_neurons, n_axons, w_bits)
                      for x in range(mesh_x) for y in range(mesh_y)}
        self.t = 0
        self.inflight: list[tuple[int, int, int, int]] = []  # (x, y, axon, parity)

    def core(self, x, y) -> Core:
        return self.cores[(x, y)]

    def inject(self, x: int, y: int, axon: int, t: int | None = None):
        """Host input event to be integrated at timestep t (default: current)."""
        t = self.t if t is None else t
        self.inflight.append((x, y, axon, t & 1))

    def reward(self, x: int, y: int, r: int):
        self.core(x, y).receive_reward(r)

    def step(self) -> list[tuple[int, int, int]]:
        """STEP(t), SPEC §3. Returns host-bound spikes (src_x, src_y, neuron)."""
        p = self.t & 1
        # 1. quiesce: deliver everything tagged for this timestep
        pending, self.inflight = self.inflight, []
        for (x, y, axon, par) in pending:
            assert par == p, "packet parity mismatch"
            self.core(x, y).receive_spike(axon, par)
        # 2. learn
        for c in self.cores.values():
            c.learn()
        # 3. update + emit
        out = []
        for (x, y), c in sorted(self.cores.items()):
            spikes = c.update(p)
            for n in np.flatnonzero(spikes):
                base, cnt = int(c.fo_base[n]), int(c.fo_count[n])
                for i in range(base, base + cnt):
                    e = c.fanout[i]
                    if e.dest_x == HOST_X:
                        out.append((x, y, int(n)))
                    else:
                        self.inflight.append((e.dest_x, e.dest_y, e.dest_axon, p ^ 1))
        self.t += 1
        return out

    def run(self, events_per_step: list[list[tuple[int, int, int]]]):
        """events_per_step[t] = list of (x, y, axon). Returns list of host outputs per step."""
        outs = []
        for evs in events_per_step:
            for (x, y, a) in evs:
                self.inject(x, y, a)
            outs.append(self.step())
        return outs
