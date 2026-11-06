"""cocotb unit tests for rtl/noc/router.sv (single router, environment models
the four credit-based neighbours and the valid/ready local core)."""
from __future__ import annotations

import random
from collections import defaultdict, deque

import cocotb
from cocotb.triggers import FallingEdge

from nocutil import (dval, reset, PE, PL, PN, PS, PW, PNAME, HOST_X, get_field, ival, mk, set_field, start,
                     tag_of, xy_route)


class RouterEnv:
    def __init__(self, dut, seed=1, in_rate=1.0, cred_rate=1.0, lready_rate=1.0):
        self.dut = dut
        self.X = int(dut.X.value)
        self.Y = int(dut.Y.value)
        self.MX = int(dut.MESH_X.value)
        self.MY = int(dut.MESH_Y.value)
        self.D = int(dut.FIFO_DEPTH.value)
        self.rng = random.Random(seed)
        self.in_rate, self.cred_rate, self.lready_rate = in_rate, cred_rate, lready_rate
        self.txq = [deque() for _ in range(5)]         # flits to inject per input port
        self.ucred = [self.D] * 4                       # upstream credit counters
        self.dq = [deque() for _ in range(4)]           # downstream FIFO models
        self.l_hold = None                              # flit held on the L input
        self.sent = []                                  # (cycle, in_port, flit)
        self.recv = defaultdict(list)                   # out_port -> [(cycle, flit)]
        self.cycle = 0
        self.errors = []

    def drained(self):
        return all(not q for q in self.txq) and self.l_hold is None and ival(self.dut.busy) == 0

    async def run(self, max_cycles):
        dut = self.dut
        rng = self.rng
        dut.in_valid.value = 0
        dut.in_flit.value = 0
        dut.credit_in.value = 0
        dut.l_out_ready.value = 0
        for _ in range(max_cycles):
            await FallingEdge(dut.clk)
            self.cycle += 1
            out_v = ival(dut.out_valid)
            out_f = dval(dut.out_flit)
            cr_out = ival(dut.credit_out)
            # ---- outputs: credit ports ----
            cin = 0
            for o in range(4):
                if out_v >> o & 1:
                    f = get_field(out_f, o, 48)
                    self.recv[o].append((self.cycle, f))
                    self.dq[o].append(f)
                    if len(self.dq[o]) > self.D:
                        self.errors.append(f"downstream FIFO overflow on {PNAME[o]}")
                if self.dq[o] and rng.random() < self.cred_rate:
                    self.dq[o].popleft()
                    cin |= 1 << o
            dut.credit_in.value = cin
            # ---- output: local valid/ready ----
            lr = 1 if rng.random() < self.lready_rate else 0
            dut.l_out_ready.value = lr
            if (out_v >> PL & 1) and lr:
                self.recv[PL].append((self.cycle, get_field(out_f, PL, 48)))
            # ---- inputs: credit ports ----
            iv, ifl = 0, 0
            for i in range(4):
                if cr_out >> i & 1:
                    self.ucred[i] += 1
                    if self.ucred[i] > self.D:
                        self.errors.append(f"credit counter overflow on input {PNAME[i]}")
                if self.txq[i] and self.ucred[i] > 0 and rng.random() < self.in_rate:
                    f = self.txq[i].popleft()
                    self.ucred[i] -= 1
                    iv |= 1 << i
                    ifl = set_field(ifl, i, 48, f)
                    self.sent.append((self.cycle, i, f))
            # ---- input: local valid/ready ----
            if self.l_hold is None and self.txq[PL] and rng.random() < self.in_rate:
                self.l_hold = self.txq[PL].popleft()
            if self.l_hold is not None:
                iv |= 1 << PL
                ifl = set_field(ifl, PL, 48, self.l_hold)
                if ival(dut.l_in_ready):
                    self.sent.append((self.cycle, PL, self.l_hold))
                    self.l_hold = None
            dut.in_valid.value = iv
            dut.in_flit.value = ifl
            if self.cycle > 5 and self.drained() and all(not q for q in self.dq) and not iv:
                return True
        return False

    def check(self, expect_drop=()):
        """Every non-dropped flit leaves exactly once, on the XY port, and
        flits from the same input to the same output stay in order."""
        got = {}
        for o, lst in self.recv.items():
            for (_, f) in lst:
                t = tag_of(f)
                if t in got:
                    self.errors.append(f"duplicate tag {t:#x}")
                got[t] = (o, f)
        order = defaultdict(list)
        for (_, i, f) in self.sent:
            exp = xy_route(self.X, self.Y, self.MX, self.MY, f)
            t = tag_of(f)
            if exp == 5:
                if t in got:
                    self.errors.append(f"tag {t:#x} should be dropped but left on {PNAME[got[t][0]]}")
                continue
            if t not in got:
                self.errors.append(f"tag {t:#x} from {PNAME[i]} lost (expected {PNAME[exp]})")
                continue
            if got[t][0] != exp or got[t][1] != f:
                self.errors.append(f"tag {t:#x} from {PNAME[i]}: left on {PNAME[got[t][0]]}, expected {PNAME[exp]}")
            order[(i, exp)].append(t)
        pos = {}
        for o, lst in self.recv.items():
            for k, (_, f) in enumerate(lst):
                pos[tag_of(f)] = (o, k)
        for key, tags in order.items():
            ks = [pos[t][1] for t in tags]
            if ks != sorted(ks):
                self.errors.append(f"reordering on {PNAME[key[0]]}->{PNAME[key[1]]}")
        assert not self.errors, "\n".join(self.errors[:20])


def all_dests(env):
    """One destination per reachable output plus host and out-of-mesh."""
    d = [(x, y) for x in range(env.MX) for y in range(env.MY)]
    d += [(HOST_X, 0), (HOST_X, 3)]
    d += [(env.MX, 0), (0, env.MY), (14, 14)]
    return d


@cocotb.test()
async def every_direction(dut):
    """Each input port sends to every mesh node, the host and off-mesh
    coordinates; each flit must leave on the XY port (or be dropped)."""
    await start(dut)
    env = RouterEnv(dut, seed=11)
    tag = 1
    n_drop = 0
    for i in range(5):
        for (dx, dy) in all_dests(env):
            f = mk(dx, dy, tag)
            env.txq[i].append(f)
            if xy_route(env.X, env.Y, env.MX, env.MY, f) == 5:
                n_drop += 1
            tag += 1
    assert await env.run(3000), "router did not drain"
    env.check()
    assert ival(dut.err) == (1 if n_drop else 0)
    assert ival(dut.drop_cnt) == min(n_drop, 255)
    used = {o for o, l in env.recv.items() if l}
    dut._log.info("router (%d,%d): %d flits, %d dropped, outputs used %s",
                  env.X, env.Y, len(env.sent), n_drop, "".join(PNAME[o] for o in sorted(used)))


@cocotb.test()
async def hop_latency(dut):
    """A flit presented on an input leaves the router 2 cycles later."""
    await start(dut)
    env = RouterEnv(dut, seed=3)
    env.txq[PL].append(mk(env.X, env.Y, 0x77))      # L -> L
    assert await env.run(100)
    env.check()
    (c_in, _, _), = env.sent
    (c_out, _), = env.recv[PL]
    dut._log.info("L->L latency %d cycles", c_out - c_in)
    assert c_out - c_in == 2


@cocotb.test()
async def arbitration_fairness(dut):
    """All five inputs flood the local output: grants must rotate round robin
    (every window of 5 consecutive grants covers all 5 inputs)."""
    await start(dut)
    env = RouterEnv(dut, seed=5)
    n = 60
    for i in range(5):
        for k in range(n):
            env.txq[i].append(mk(env.X, env.Y, i << 16 | k))
    assert await env.run(5000)
    env.check()
    seq = [tag_of(f) >> 16 for (_, f) in env.recv[PL]]
    counts = [seq.count(i) for i in range(5)]
    assert counts == [n] * 5
    # steady state: once all inputs have a head flit, strict rotation
    steady = seq[10:len(seq) - 20]
    for k in range(len(steady) - 4):
        assert len(set(steady[k:k + 5])) == 5, f"unfair window at {k}: {steady[k:k+5]}"
    # with the local output always ready it must sustain 1 flit/cycle
    cyc = [c for (c, _) in env.recv[PL]]
    span = cyc[-1] - cyc[0] + 1
    dut._log.info("L output: %d flits in %d cycles", len(cyc), span)
    assert span <= len(cyc) + 6


@cocotb.test()
async def backpressure_random(dut):
    """Random traffic on all inputs, random credit return and random local
    ready: no loss, duplication, misrouting, reordering or overflow."""
    await start(dut)
    for seed, (ir, cr, lr) in enumerate([(0.9, 0.2, 0.2), (1.0, 0.5, 0.9), (0.5, 0.05, 0.05), (1.0, 1.0, 1.0)]):
        await reset(dut)
        env = RouterEnv(dut, seed=100 + seed, in_rate=ir, cred_rate=cr, lready_rate=lr)
        dests = [(x, y) for x in range(env.MX) for y in range(env.MY)] + [(HOST_X, 0)]
        tag = seed << 24
        for i in range(5):
            for _ in range(80):
                env.txq[i].append(mk(*env.rng.choice(dests), tag))
                tag += 1
        ok = await env.run(40000)
        assert ok, (f"router did not drain (seed {seed}): txq={[len(q) for q in env.txq]} hold={env.l_hold} "
                    f"busy={ival(dut.busy)} dq={[len(q) for q in env.dq]} ucred={env.ucred} "
                    f"out_valid={ival(dut.out_valid):05b} sent={len(env.sent)} recv={sum(len(v) for v in env.recv.values())}")
        env.check()
        assert ival(dut.err) == 0
        dut._log.info("seed %d rates %s: %d flits OK in %d cycles", seed, (ir, cr, lr), len(env.sent), env.cycle)
