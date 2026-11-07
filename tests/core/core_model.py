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
LEARN = int(os.environ.get("LEARN", 0))        # RTL build parameter
ADAPT = int(os.environ.get("ADAPT", 1))        # RTL build parameter ADAPT_TH
G = N // LANES
WMIN, WMAX = -(1 << (W - 1)), (1 << (W - 1)) - 1


def s16(v):
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


DBG_THETA_A = 6          # region 11 debug space 6: theta_a[n] (SPEC §9.4)
REG_PAM, REG_TH_INC, REG_TH_DSHIFT = 23, 24, 25


def is_adapt_addr(addr):
    """Addresses that an ADAPT_TH = 0 build ignores (regs 24/25, theta_a)."""
    return (addr >> 20 & 3) == 3 and ((addr >> 16 & 0xF) == DBG_THETA_A or
                                      ((addr >> 16 & 0xF) == 0 and (addr & 0xFF) in (REG_TH_INC, REG_TH_DSHIFT)))


def cfg(g: Core, addr, data):
    if ADAPT or not is_adapt_addr(addr):   # ADAPT_TH = 0: golden keeps theta_a = 0
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
    # adaptive threshold (SPEC §9.4); "adapt" stresses the saturations and floor
    if mode == "adapt":
        regs[REG_TH_INC] = rng.choice([0xFFFF, rng.randint(0x8000, 0xFFFF)])   # small values: other modes
        regs[REG_TH_DSHIFT] = rng.choice([1, 2, rng.randint(5, 15), 15])
        regs[0] = rng.randint(1, 60)
        regs[2], regs[3] = 0, rng.choice([0, 0, 1])   # saturated neurons keep firing
    else:
        regs[REG_TH_INC] = rng.choice([0, 0, rng.randint(0, 50), rng.randint(0, 0xFFFF)])
        regs[REG_TH_DSHIFT] = rng.choice([0, 0, rng.randint(0, 15)])
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
        elif mode == "adapt" and n % 4 == 0:   # v saturates: fires even at theta_eff = 32767
            v, b = 32767, 32767
        else:
            v, b = rng.randint(-100, 100), rng.randint(-20, 40)
        rc = rng.choice([0, 0, 0, rng.randint(0, 15)])
        fl.append(cfg(g, flits.addr_neuron(n, 0), v & 0xFFFF))
        fl.append(cfg(g, flits.addr_neuron(n, 1), b & 0xFFFF))
        fl.append(cfg(g, flits.addr_neuron(n, 2), rc))
        if mode == "adapt":
            ta = rng.choice([0, 1, 2, rng.randint(0, 30), 32767, 65535, rng.randint(0, 65535), rng.randint(32700, 32767)])
        else:
            ta = rng.choice([0, 0, 0, rng.randint(0, 40), rng.randint(0, 65535)])
        fl.append(cfg(g, dbg_addr(DBG_THETA_A, n), ta))
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
    th_raw = R.theta + g.theta_a
    fire = act & (v2 >= np.minimum(32767, th_raw))
    k = R.th_decay_shift
    ta_dec = g.theta_a - (np.where(g.theta_a > 0, np.maximum(g.theta_a >> k, 1), 0) if k else 0)
    COV["theta_sat"] += int((act & (th_raw > 32767)).sum())
    COV["theta_a_fire"] += int((fire & (g.theta_a > 0)).sum())
    COV["theta_a_floor"] += int(((g.theta_a > 0) & (k > 0) & ((g.theta_a >> k) == 0)).sum())
    COV["theta_a_inc_sat"] += int((fire & (ta_dec + R.th_inc > 65535)).sum())
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
    addrs += [dbg_addr(DBG_THETA_A, n) for n in range(N)]
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
    if mode == "adapt" and ADAPT:
        assert all(COV[k] > 0 for k in ("theta_sat", "theta_a_fire", "theta_a_floor", "theta_a_inc_sat")), COV


# ---------------------------------------------------------------------------
# Learning (SPEC §6) helpers. Debug address map (proposed; region 11):
#   [19:16] = 1 x_pre[a], 2 x_post[n] (R/W, uint8), 3 pre_flag[a], 4 post_flag[n] (R),
#   5: [1:0] = 0 LFSR state, 1 reward sum [15:0], 2 reward sum [31:16] (R).
#   reg 22 bit 0 (LEARN = 2): region 00 reads/writes address e[a][n] instead of w.
# ---------------------------------------------------------------------------
DBG_XPRE, DBG_XPOST, DBG_PREF, DBG_POSTF, DBG_MISC = 1, 2, 3, 4, 5
REG_DBG_ELIG = 22


def dbg_addr(sub, idx):
    return 3 << 20 | sub << 16 | idx


def learn_config(rng, g: Core, mode):
    """Learning registers + random initial traces / eligibility.

    Returns (shuffle-safe flits, ordered flits). Golden is updated to match
    (traces and eligibility directly, since golden has no address for them)."""
    rst = mode.startswith("rstdp")
    ext = mode.endswith("ext")
    s0 = mode.endswith("s0")
    bigshift = mode.endswith("bigshift")      # stdp_shift >= 24 (beyond the 24-bit product)
    lo = rng.choice([WMIN, rng.randint(WMIN, 0)])
    hi = rng.choice([WMAX, rng.randint(0, WMAX)])
    if rng.random() < 0.2:          # w_min > w_max: clamp gives w_max (numpy order)
        lo, hi = rng.randint(1, WMAX), rng.randint(WMIN, 0)
    big = lambda: rng.choice([rng.randint(0, 0xFFFF), 0xFFFF, rng.randint(0, 4000)])
    regs = {
        8: rng.choice([1, rng.randint(1, 0xFFFF)]),
        9: big() if ext else rng.randint(0, 24),
        10: big() if ext else rng.randint(0, 24),
        # stdp_shift <= 63: golden's mask overflows int64 beyond that
        11: rng.choice([24, 25, 32, 40, 63]) if bigshift else 0 if s0 else (rng.choice([20, 23, 24, 25, 31, 32, 40, 63, rng.randint(9, 30)]) if ext
                          else rng.randint(1, 8)),
        12: rng.choice([0, 0, 1, 2, 3, rng.randint(0, 9), 0xFFFF if ext else 4]),
        13: rng.choice([0, 0, 1, 2, 3, rng.randint(0, 9), 0xFFFF if ext else 4]),
        14: big() if ext else rng.randint(0, 80),
        15: big() if ext else rng.randint(0, 80),
        16: rng.choice([0, 1, rng.randint(0, 0xFFFF)]),
        17: lo & 0xFFFF,
        18: hi & 0xFFFF,
        19: rng.randint(1, 0xFFFF) if rst else 0,
        20: rng.choice([0, 1, 2, 3, rng.randint(0, 16), 0xFFFF if ext else 5]),
        # reward_shift <= 63: golden v0.2's rounding term 1 << (rs - 1) is int64
        21: (rng.choice([48, 49, 63]) if bigshift else
             rng.choice([rng.randint(0, 12), 1, 40, 47, 48, 49, 63]) if ext else rng.randint(0, 12)),
        # plastic_axon_max (SPEC §9.3)
        REG_PAM: rng.choice([0, 0, rng.randint(1, A - 1), rng.randint(1, A - 1), A,
                             rng.randint(A + 1, 0xFFFF)]),
    }
    fl = [cfg(g, flits.addr_reg(r), v) for r, v in regs.items()]
    ordered = []
    # mode "*_notrace" (AUDIT-1 regression): traces are never written, so the
    # RTL must start them at their SPEC §5 reset value 0, like golden does.
    for a in range(0 if mode.endswith("notrace") else A):
        x = rng.choice([0, rng.randint(0, 255), 255])
        g.x_pre[a] = x
        ordered.append(flits.cfg_write(CX, CY, dbg_addr(DBG_XPRE, a), x | rng.randint(0, 255) << 8))
    for n in range(0 if mode.endswith("notrace") else N):
        x = rng.choice([0, rng.randint(0, 255), 255])
        g.x_post[n] = x
        ordered.append(flits.cfg_write(CX, CY, dbg_addr(DBG_XPOST, n), x))
    if LEARN == 2:
        ordered.append(flits.cfg_write(CX, CY, flits.addr_reg(REG_DBG_ELIG), 1))
        for a in range(A):
            for n in range(N):
                e = rng.choice([0, rng.randint(-300, 300), rng.randint(-32768, 32767)]) if rst else 0
                g.elig[a, n] = e
                ordered.append(flits.cfg_write(CX, CY, flits.addr_weight(a, n, N), e & 0xFFFF))
        ordered.append(flits.cfg_write(CX, CY, flits.addr_reg(REG_DBG_ELIG), 0))
    return fl, ordered


def learn_light_addrs():
    addrs = [dbg_addr(DBG_XPRE, a) for a in range(A)] + [dbg_addr(DBG_XPOST, n) for n in range(N)]
    addrs += [dbg_addr(DBG_PREF, a) for a in range(A)] + [dbg_addr(DBG_POSTF, n) for n in range(N)]
    addrs += [dbg_addr(DBG_MISC, i) for i in range(3)]
    return addrs


def learn_expected(g: Core, addr, pending_pre=frozenset()):
    """Expected read data for a learning debug address (golden has none)."""
    region, sub, idx = addr >> 20 & 3, addr >> 16 & 0xF, addr & 0xFFFF
    if region != 3 or sub == 0:
        return g.cfg_read(addr)
    if sub == DBG_XPRE:
        return int(g.x_pre[idx])
    if sub == DBG_XPOST:
        return int(g.x_post[idx])
    if sub == DBG_PREF:
        pam = g.regs.plastic_axon_max
        return int(bool(g.pre_flag[idx]) or (idx in pending_pre and bool(g.regs.learn_en) and (pam == 0 or idx < pam)))
    if sub == DBG_POSTF:
        return int(g.post_flag[idx])
    if sub == DBG_THETA_A:
        return g.cfg_read(addr)
    if sub == DBG_MISC:
        return [g.lfsr.state, g.pending_reward & 0xFFFF, (g.pending_reward >> 16) & 0xFFFF][idx & 3]
    raise ValueError(hex(addr))
