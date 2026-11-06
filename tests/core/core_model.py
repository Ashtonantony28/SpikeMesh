"""Shared (simulator-independent) stimulus generation and golden reference for
the neurocore tests: used by the cocotb testbench (core_tb.py) and by the
Verilator trace-replay flow (vl_replay.py).

Cycle-agnostic and timestep-exact: every timestep the multiset of output flits
must equal what the golden model emits, and READ_REQ readback of v/refc/
weights must equal golden state. Configuration comes from env vars set by
tests/core/test_core.py (N, A, LANES, W, FD, X, Y, SEED, STEPS, MODE, BP).
"""
from __future__ import annotations

import os
import random
from collections import Counter


from compiler import flits
from golden.spikemesh import Core, REG_INDEX

N = int(os.environ.get("N", 32))
A = int(os.environ.get("A", 32))
LANES = int(os.environ.get("LANES", 4))
W = int(os.environ.get("W", 8))
FD = int(os.environ.get("FD", 64))
CX = int(os.environ.get("X", 2))
CY = int(os.environ.get("Y", 1))
SEED = int(os.environ.get("SEED", 1))
STEPS = int(os.environ.get("STEPS", 12))
MODE = os.environ.get("MODE", "mixed")
BP = float(os.environ.get("BP", 0.3))          # out_ready low probability
IN_GAP = float(os.environ.get("IN_GAP", 0.1))  # in_valid gap probability
G = N // LANES
WMIN, WMAX = -(1 << (W - 1)), (1 << (W - 1)) - 1


def s16(v):
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def cfg(g: Core, addr, data):
    g.cfg_write(addr, data)
    return flits.cfg_write(CX, CY, addr, data)


def all_weight_flits(g, rng, density=1.0, lo=WMIN, hi=WMAX):
    fl = []
    for a in range(A):
        for n in range(N):
            w = rng.randint(lo, hi) if rng.random() < density else 0
            fl.append(cfg(g, flits.addr_weight(a, n, N), w & ((1 << W) - 1)))
    return fl


def random_network(rng, mode):
    """Returns (golden core, cfg flits, host axons usable for input, params)."""
    g = Core(CX, CY, N, A, W)
    fl = []
    sat = mode == "sat"
    # --- core registers
    regs = {
        0: rng.randint(1, 300) if not sat else rng.choice([1, 100, 32767, rng.randint(1000, 32767)]),
        1: rng.choice([0, 0, 1, 2, 3, 4, rng.randint(0, 15)]),
        2: rng.randint(0, 1),
        3: rng.choice([0, 0, 1, 2, rng.randint(0, 15)]),
        4: rng.choice([-32768, rng.randint(-200, 0), rng.randint(-32768, 32767) if sat else -5]),
    }
    if mode == "refrac":
        regs[3] = rng.randint(2, 15)
    if mode == "leakneg":
        regs[1] = rng.randint(1, 6)
        regs[4] = rng.randint(-400, -50)
    if mode in ("reset0", "reset1"):
        regs[2] = int(mode[-1])
    for r, v in regs.items():
        fl.append(cfg(g, flits.addr_reg(r), v & 0xFFFF))
    # --- weights (all written; memories are undefined after reset)
    dens = 1.0 if mode != "sparse" else 0.15
    if mode == "accsat":   # even neurons all-max, odd neurons all-min: both acc saturations
        for a in range(A):
            for n in range(N):
                fl.append(cfg(g, flits.addr_weight(a, n, N), (WMAX if n % 2 == 0 else WMIN) & ((1 << W) - 1)))
    elif sat:
        fl += all_weight_flits(g, rng, 1.0, WMAX - 3, WMAX) if rng.random() < 0.5 else all_weight_flits(g, rng, 1.0, WMIN, WMIN + 3)
    else:
        fl += all_weight_flits(g, rng, dens)
    # --- fan-out: self-loop entries target distinct axons, so each axon gets
    # at most one loop spike and at most one host spike per timestep (keeps
    # acc within the SPEC §2 accumulator width).
    loop_axons = rng.sample(range(A), A // 3) if mode != "accsat" else []
    host_axons = [a for a in range(A) if a not in loop_axons]
    free_loop = list(loop_axons)
    nxt = rng.randint(0, 3)
    entries = {}
    for n in range(N):
        cnt = rng.choice([0, 1, 1, 2, 3, rng.randint(0, 6)])
        if nxt + cnt > FD:
            cnt = 0
        base = nxt if cnt else rng.randint(0, 1023)
        for i in range(cnt):
            r = rng.random()
            if r < 0.3 and free_loop:
                e = (CX, CY, free_loop.pop())
            elif r < 0.5:
                e = (15, rng.randint(0, 15), rng.randint(0, 1023))   # host-bound; dy/axon ignored
            else:
                while True:
                    dx, dy = rng.randint(0, 14), rng.randint(0, 15)
                    if (dx, dy) != (CX, CY):
                        break
                e = (dx, dy, rng.randint(0, 1023))
            entries[base + i] = e
        if cnt:
            nxt += cnt + rng.choice([0, 0, 1])
        fl.append(cfg(g, flits.addr_neuron(n, 3), (cnt << 10) | base))
    for idx, (dx, dy, ax) in entries.items():
        fl.append(cfg(g, flits.addr_fanout(idx, 0), dx << 12 | dy << 8 | rng.randint(0, 255)))
        fl.append(cfg(g, flits.addr_fanout(idx, 1), ax | rng.randint(0, 63) << 10))
    # --- neuron state
    for n in range(N):
        if sat:
            v, b = rng.choice([32767, -32768, rng.randint(-32768, 32767)]), rng.choice([32767, -32768, 0, rng.randint(-32768, 32767)])
        elif mode == "leakneg":
            v, b = rng.randint(-3000, 50), rng.randint(-10, 30)
        else:
            v, b = rng.randint(-100, 100), rng.randint(-20, 40)
        rc = rng.choice([0, 0, 0, rng.randint(0, 15)])
        fl.append(cfg(g, flits.addr_neuron(n, 0), v & 0xFFFF))
        fl.append(cfg(g, flits.addr_neuron(n, 1), b & 0xFFFF))
        fl.append(cfg(g, flits.addr_neuron(n, 2), rc))
    rng.shuffle(fl)   # order of independent writes must not matter
    return g, fl, host_axons


COV = Counter()


def coverage(g: Core, p):
    """Count which SPEC §2 corner cases this UPDATE exercises."""
    import numpy as np
    R = g.regs
    acc = g.acc[p]
    act = g.refc == 0
    leak = (g.v >> R.leak_shift) if R.leak_shift else np.zeros_like(g.v)
    raw = g.v - leak + np.clip(acc, -32768, 32767) + g.bias
    v2 = np.clip(raw, -32768, 32767)
    fire = act & (v2 >= R.theta)
    COV["refractory"] += int((~act).sum())
    COV["fire"] += int(fire.sum())
    COV["acc_sat_hi"] += int((act & (acc > 32767)).sum())
    COV["acc_sat_lo"] += int((act & (acc < -32768)).sum())
    COV["v_sat_hi"] += int((act & (raw > 32767)).sum())
    COV["v_sat_lo"] += int((act & (raw < -32768)).sum())
    COV["leak_neg_v"] += int((act & (g.v < 0) & (R.leak_shift > 0)).sum())
    COV["v_min_clamp"] += int((act & ~fire & (v2 < R.v_min)).sum())
    COV["fire_reset" + str(R.reset_mode)] += int(fire.sum())


def golden_step(g: Core, t, axons):
    p = t & 1
    for a in axons:
        g.receive_spike(a, p)
    g.learn()
    coverage(g, p)
    spikes = g.update(p)
    exp, loop = [], []
    for n in [int(i) for i in spikes.nonzero()[0]]:
        base, cnt = int(g.fo_base[n]), int(g.fo_count[n])
        for i in range(base, base + cnt):
            e = g.fanout[i % FD]
            if e.dest_x == flits.HOST_X:
                exp.append(flits.host_spike(CX, CY, n, p ^ 1))
            else:
                exp.append(flits.spike(e.dest_x, e.dest_y, e.dest_axon, p ^ 1))
                if (e.dest_x, e.dest_y) == (CX, CY):
                    loop.append(e.dest_axon)
    return exp, loop



def make_events(rng, host_axons, mode=MODE, steps=STEPS):
    """Host input axons per timestep (distinct axons, at most one per axon)."""
    events = [rng.sample(host_axons, rng.randint(0, len(host_axons))) for _ in range(steps + 1)]
    if mode == "accsat":   # every axon twice: |acc| = 2 * A * |w| > 32767, still within ACC_W
        events = [list(range(A)) * 2 if t % 2 == 0 else events[t] for t in range(steps + 1)]
    return events


def state_addrs(rng, n_weights=64):
    addrs = [flits.addr_neuron(n, f) for n in range(N) for f in (0, 2)]
    addrs += [flits.addr_neuron(rng.randrange(N), rng.randrange(4)) for _ in range(8)]
    addrs += [flits.addr_weight(rng.randrange(A), rng.randrange(N), N) for _ in range(n_weights)]
    return addrs


def check_coverage(mode=MODE):
    if mode == "sat":
        assert COV["v_sat_hi"] + COV["v_sat_lo"] > 0, COV
    if mode == "accsat":
        assert COV["acc_sat_hi"] > 0 and COV["acc_sat_lo"] > 0, COV
    if mode == "leakneg":
        assert COV["leak_neg_v"] > 0 and COV["v_min_clamp"] > 0, COV
    if mode == "refrac":
        assert COV["refractory"] > 0, COV
