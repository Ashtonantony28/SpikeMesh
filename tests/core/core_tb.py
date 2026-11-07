"""cocotb testbench for rtl/core/neurocore.sv, checked against golden.spikemesh.Core.

Cycle-agnostic and timestep-exact: every timestep the multiset of output flits
must equal what the golden model emits, and READ_REQ readback of v/refc/
weights must equal golden state. Configuration comes from env vars set by
tests/core/test_core.py (N, A, LANES, W, FD, X, Y, SEED, STEPS, MODE, BP).
"""
from __future__ import annotations

import os
import random
from collections import Counter, deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly

from compiler import flits
from golden.spikemesh import Core, REG_INDEX
from core_model import (N, A, W, FD, CX, CY, G, SEED, STEPS, MODE, BP, IN_GAP, COV, LANES, LEARN,
                        cfg, all_weight_flits, random_network, golden_step, make_events,
                        state_addrs, check_coverage, learn_config, learn_light_addrs, learn_expected,
                        REG_DBG_ELIG, ADAPT, DBG_THETA_A, DBG_MISC, REG_TH_DSHIFT, dbg_addr)

class Harness:
    """Single-coroutine cycle driver: drive on the falling edge, sample after
    settling (ReadOnly); a handshake seen there completes at the next rising edge."""

    def __init__(self, dut, rng, bp=0.0, in_gap=0.0, loopback=True):
        self.dut, self.rng, self.bp, self.in_gap, self.loopback = dut, rng, bp, in_gap, loopback
        self.in_q: deque[int] = deque()
        self.out_log: list[tuple[int, int]] = []   # (tag, flit)
        self.tag = -1
        self.pulse = None
        self.busy = 1
        self.cycles = 0
        self.parity = 0

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        d.rst_n.value = 0
        d.in_valid.value = 0
        d.in_flit.value = 0
        d.out_ready.value = 0
        d.parity.value = 0
        d.learn_start.value = 0
        d.update_start.value = 0
        for _ in range(3):
            await FallingEdge(d.clk)
        d.rst_n.value = 1
        await self.run_until(lambda: not self.busy, 10 * G + 50)

    async def cycle(self):
        d = self.dut
        await FallingEdge(d.clk)
        d.learn_start.value = 1 if self.pulse == "learn" else 0
        d.update_start.value = 1 if self.pulse == "update" else 0
        d.parity.value = self.parity
        self.pulse = None
        drive = bool(self.in_q) and self.rng.random() >= self.in_gap
        d.in_valid.value = int(drive)
        if drive:
            d.in_flit.value = self.in_q[0]
        ordy = self.rng.random() >= self.bp
        d.out_ready.value = int(ordy)
        await ReadOnly()
        self.cycles += 1
        if drive and int(d.in_ready.value):
            self.in_q.popleft()
        if ordy and int(d.out_valid.value):
            f = int(d.out_flit.value)
            self.out_log.append((self.tag, f))
            if self.loopback and (f >> 46) == flits.SPIKE and ((f >> 42) & 0xF, (f >> 38) & 0xF) == (CX, CY):
                self.in_q.append(f)
        self.busy = int(d.busy.value)

    async def run_until(self, pred, max_cycles=200000):
        for _ in range(max_cycles):
            await self.cycle()
            if pred():
                return
        raise TimeoutError("run_until timed out")

    async def idle(self, n=3):
        """Input queue drained and busy low for n consecutive cycles."""
        cnt = 0
        while cnt < n:
            await self.cycle()
            cnt = cnt + 1 if (not self.in_q and not self.busy) else 0

    async def send(self, fl):
        self.in_q.extend(fl)
        await self.idle(2)

    async def read(self, addrs, rx=flits.HOST_X, ry=0):
        """READ_REQ each address; returns the data list (checks resp format)."""
        start = len(self.out_log)
        self.in_q.extend(flits.read_req(CX, CY, a, rx, ry) for a in addrs)
        await self.idle(2)
        resps = [f for _, f in self.out_log[start:]]
        assert len(resps) == len(addrs), (len(resps), len(addrs))
        out = []
        for a, f in zip(addrs, resps):
            d = flits.decode(f)
            assert d["type"] == flits.READ and d["resp"], hex(f)
            assert (d["dest_x"], d["dest_y"]) == (rx, ry)
            assert d["addr16"] == a & 0xFFFF, (hex(a), hex(d["addr16"]))
            assert (f >> 32) & 0x1F == 0
            out.append(d["data"])
        return out


async def check_state(h: Harness, g: Core, rng, n_weights=64):
    addrs = state_addrs(rng, n_weights)
    got = await h.read(addrs, rng.randint(0, 15), rng.randint(0, 15))
    exp = [g.cfg_read(a) for a in addrs]
    bad = [(hex(a), x, y) for a, x, y in zip(addrs, got, exp) if x != y]
    assert not bad, f"readback mismatch (addr, rtl, golden): {bad[:10]}"


@cocotb.test()
async def test_random(dut):
    rng = random.Random(SEED)
    h = Harness(dut, rng, bp=0.0, in_gap=0.0)
    await h.start()
    g, fl, host_axons = random_network(rng, MODE)
    await h.send(fl)
    h.bp, h.in_gap = BP, IN_GAP
    # host events per timestep (distinct axons, at most one per axon)
    events = make_events(rng, host_axons)
    loop_next: list[int] = []
    early = False
    total_out = 0
    for t in range(STEPS):
        p = t & 1
        h.parity = p
        if not early:
            h.in_q.extend(flits.spike(CX, CY, a, p) for a in events[t])
        if rng.random() < 0.3:
            h.in_q.append(flits.reward(CX, CY, rng.randint(-100, 100)))
        await h.idle(2)                              # quiesce
        exp, loop = golden_step(g, t, events[t] + loop_next)
        loop_next = loop
        h.pulse = "learn"
        await h.cycle()
        await h.run_until(lambda: not h.busy, 1000)
        h.tag = t
        h.pulse = "update"
        await h.cycle()
        # Flits the core must hold off until UPDATE ends (head-of-line): a
        # READ_REQ of a v field, a CFG write and a REWARD, sent mid-UPDATE.
        mid_reads = []
        if rng.random() < 0.3:
            n = rng.randrange(N)
            mid_reads.append(flits.addr_neuron(n, rng.choice([0, 2])))
            h.in_q.append(flits.read_req(CX, CY, mid_reads[0], 15, 0))
            h.in_q.append(cfg(g, flits.addr_reg(9), rng.randint(0, 0xFFFF)))
            h.in_q.append(flits.reward(CX, CY, rng.randint(-100, 100)))
        early = rng.random() < 0.5                   # inject t+1 inputs during UPDATE(t)
        if early:
            h.in_q.extend(flits.spike(CX, CY, a, p ^ 1) for a in events[t + 1])
        await h.run_until(lambda: not h.busy, 200000)
        if mid_reads:
            await h.idle(2)
        got = [f for tag, f in h.out_log if tag == t and (f >> 46) == flits.SPIKE]
        resp = [flits.decode(f) for tag, f in h.out_log if tag == t and (f >> 46) == flits.READ]
        assert [r["data"] for r in resp] == [g.cfg_read(a) for a in mid_reads], (resp, mid_reads)
        total_out += len(got)
        if Counter(got) != Counter(exp):
            extra = Counter(got) - Counter(exp)
            missing = Counter(exp) - Counter(got)
            raise AssertionError(f"t={t}: extra={[hex(f) for f in extra]} missing={[hex(f) for f in missing]}")
    await h.idle(2)
    h.tag = -2
    await check_state(h, g, rng)
    if LEARN:   # learn_en = 0 here: every REWARD must have been dropped (SPEC §9.5)
        got = await h.read([dbg_addr(DBG_MISC, 1), dbg_addr(DBG_MISC, 2)])
        assert got == [g.pending_reward & 0xFFFF, (g.pending_reward >> 16) & 0xFFFF] == [0, 0], got
    dut._log.info(f"MODE={MODE} SEED={SEED} N={N} A={A} LANES={LANES}: {STEPS} steps, "
                  f"{total_out} output flits, {h.cycles} cycles total; coverage {dict(COV)}")
    check_coverage()


@cocotb.test()
async def test_cfg_readback(dut):
    """Write every location of every region with random data; read it all back."""
    rng = random.Random(SEED)
    h = Harness(dut, rng, bp=BP, in_gap=IN_GAP, loopback=False)
    await h.start()
    g = Core(CX, CY, N, A, W)
    # reset values of core registers
    regs = list(range(32))
    got = await h.read([flits.addr_reg(r) for r in regs])
    assert got == [g.cfg_read(flits.addr_reg(r)) for r in regs], got
    fl = []
    for r in regs:
        name = REG_INDEX.get(r)
        if name in ("leak_shift", "refrac") or r == REG_TH_DSHIFT:
            v = rng.randint(0, 15)
        elif name == "reset_mode":
            v = rng.randint(0, 1)
        elif r == REG_DBG_ELIG:
            v = 0          # debug register: would redirect weight writes to e[a][n]
        else:
            v = rng.randint(0, 0xFFFF)
        fl.append(cfg(g, flits.addr_reg(r), v))
    for a in range(A):
        for n in range(N):
            # upper data bits must be ignored (SPEC §5)
            fl.append(cfg(g, flits.addr_weight(a, n, N), rng.randint(0, 0xFFFF)))
    for n in range(N):
        for f in range(4):
            fl.append(cfg(g, flits.addr_neuron(n, f), rng.randint(0, 0xFFFF)))
    for e in range(FD):
        for wd in range(2):
            fl.append(cfg(g, flits.addr_fanout(e, wd), rng.randint(0, 0xFFFF)))
    # theta_a (SPEC §9.4): every neuron, plus out-of-range indices (ignored)
    ta_idx = list(range(N)) + [N, N + 3, 0xFFFF]
    for n in ta_idx:
        fl.append(cfg(g, dbg_addr(DBG_THETA_A, n), rng.randint(0, 0xFFFF)))
    rng.shuffle(fl)
    await h.send(fl)
    addrs = [flits.addr_reg(r) for r in regs]
    addrs += [flits.addr_weight(a, n, N) for a in range(A) for n in range(N)]
    addrs += [flits.addr_neuron(n, f) for n in range(N) for f in range(4)]
    addrs += [flits.addr_fanout(e, wd) for e in range(FD) for wd in range(2)]
    addrs += [dbg_addr(DBG_THETA_A, n) for n in ta_idx]
    rng.shuffle(addrs)
    got = await h.read(addrs, 15, 0)
    exp = [g.cfg_read(a) for a in addrs]
    bad = [(hex(a), x, y) for a, x, y in zip(addrs, got, exp) if x != y]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:10]}"
    dut._log.info(f"cfg readback: {len(addrs)} addresses OK")


@cocotb.test()
async def test_cycles(dut):
    """Measure cycles per spike and per UPDATE (no backpressure)."""
    rng = random.Random(SEED)
    h = Harness(dut, rng, loopback=False)
    await h.start()
    g = Core(CX, CY, N, A, W)
    fl = all_weight_flits(g, rng, 1.0, 0, 0)
    for n in range(N):
        for f in range(4):
            fl.append(cfg(g, flits.addr_neuron(n, f), 0))
    await h.send(fl)
    # back-to-back spikes: K spikes
    K = 16
    c0 = h.cycles
    h.in_q.extend(flits.spike(CX, CY, a % A, 0) for a in range(K))
    await h.run_until(lambda: not h.in_q)
    await h.run_until(lambda: not h.busy)
    spk = h.cycles - c0
    # UPDATE with no firing neurons
    h.pulse = "update"
    c0 = h.cycles
    await h.cycle()
    await h.run_until(lambda: not h.busy)
    upd0 = h.cycles - c0
    # UPDATE with every neuron firing to one host entry (theta=1, bias=1)
    fl = [cfg(g, flits.addr_reg(0), 1), cfg(g, flits.addr_fanout(0, 0), 0xF000)]
    for n in range(N):
        fl += [cfg(g, flits.addr_neuron(n, 1), 1), cfg(g, flits.addr_neuron(n, 3), (1 << 10) | 0)]
    await h.send(fl)
    h.pulse = "update"
    c0 = h.cycles
    start = len(h.out_log)
    await h.cycle()
    await h.run_until(lambda: not h.busy)
    upd1 = h.cycles - c0
    assert len(h.out_log) - start == N
    dut._log.info(f"CYCLES N={N} A={A} LANES={LANES}: {K} back-to-back spikes in {spk} cycles "
                  f"({spk / K:.2f}/spike, G={G}); UPDATE no-fire {upd0} cycles; "
                  f"UPDATE all-fire (1 flit each) {upd1} cycles")


async def check_learn_state(h: Harness, g, rng, pending_pre, full, n_weights=None):
    """Read back traces/flags/LFSR/reward (+ weights and eligibility if full)."""
    import numpy as np
    addrs = learn_light_addrs()
    if full:
        if n_weights is None:
            waddrs = [flits.addr_weight(a, n, N) for a in range(A) for n in range(N)]
        else:
            waddrs = [flits.addr_weight(rng.randrange(A), rng.randrange(N), N) for _ in range(n_weights)]
        addrs += waddrs
    got = await h.read(addrs)
    exp = [learn_expected(g, a, pending_pre) for a in addrs]
    bad = [(hex(a), x, y) for a, x, y in zip(addrs, got, exp) if x != y]
    assert not bad, f"{len(bad)} learn-state mismatches (addr, rtl, golden): {bad[:12]}"
    if full and LEARN == 2:
        await h.send([flits.cfg_write(CX, CY, flits.addr_reg(REG_DBG_ELIG), 1)])
        got = await h.read(waddrs)
        await h.send([flits.cfg_write(CX, CY, flits.addr_reg(REG_DBG_ELIG), 0)])
        exp = [int(g.elig[(a & 0xFFFFF) // N, (a & 0xFFFFF) % N]) & 0xFFFF for a in waddrs]
        bad = [(hex(a), x, y) for a, x, y in zip(waddrs, got, exp) if x != y]
        assert not bad, f"{len(bad)} eligibility mismatches: {bad[:12]}"
    return np


@cocotb.test()
async def test_learn(dut):
    """Multi-step STDP / R-STDP run vs golden: spikes per step, traces, flags,
    LFSR and reward state every step, all weights and eligibility periodically."""
    rng = random.Random(SEED)
    h = Harness(dut, rng, bp=0.0, in_gap=0.0)
    await h.start()
    g, fl, host_axons = random_network(rng, "mixed")
    lfl, ordered = learn_config(rng, g, MODE)
    fl += lfl
    rng.shuffle(fl)
    await h.send(fl)
    await h.send(ordered)
    await check_learn_state(h, g, rng, frozenset(), True, None if A * N <= 4096 else 512)
    h.bp, h.in_gap = BP, IN_GAP
    events = make_events(rng, host_axons, "mixed")
    loop_next: list[int] = []
    early = False
    learn_log = []
    full_every = int(os.environ.get("FULL_EVERY", 3))
    orig_learn, orig_rstdp = g.regs.learn_en, g.regs.rstdp_en
    toggles = Counter()
    for t in range(STEPS):
        p = t & 1
        h.parity = p
        # SPEC §9.5: switch R-STDP / learning off and on between steps, so some
        # rewards arrive while off (dropped). learn_en only changes when no
        # spike for this step was delivered early (golden takes spikes at step time).
        if t > 0 and MODE.startswith("rstdp") and rng.random() < 0.3:
            v = rng.choice([0, orig_rstdp])
            toggles["rstdp_en=%d" % bool(v)] += 1
            h.in_q.append(cfg(g, flits.addr_reg(19), v))
        if t > 0 and not early and not loop_next and rng.random() < 0.2:
            v = rng.choice([0, orig_learn])
            toggles["learn_en=%d" % bool(v)] += 1
            h.in_q.append(cfg(g, flits.addr_reg(8), v))
        if not early:
            h.in_q.extend(flits.spike(CX, CY, a, p) for a in events[t])
        for _ in range(rng.choice([0, 0, 1, 2])):
            r = rng.randint(-200, 200)
            g.receive_reward(r)
            h.in_q.append(flits.reward(CX, CY, r))
        if rng.random() < 0.3:
            # pulse learn_start as soon as the last input flit is accepted (its
            # accumulation may still be running): LEARN must wait for it
            await h.run_until(lambda: not h.in_q, 100000)
        else:
            await h.idle(2)
        n_pre = len(set(events[t] + loop_next))
        n_post = int(g.post_flag.sum())
        exp, loop = golden_step(g, t, events[t] + loop_next)
        loop_next = loop
        h.pulse = "learn"
        c0 = h.cycles
        await h.cycle()
        await h.run_until(lambda: not h.busy, 2_000_000)
        learn_log.append((h.cycles - c0, n_pre, n_post))
        h.tag = t
        h.pulse = "update"
        await h.cycle()
        if rng.random() < 0.3:          # reward arriving during UPDATE(t) counts for LEARN(t+1)
            r = rng.randint(-200, 200)
            g.receive_reward(r)
            h.in_q.append(flits.reward(CX, CY, r))
        early = rng.random() < 0.5
        if early:
            h.in_q.extend(flits.spike(CX, CY, a, p ^ 1) for a in events[t + 1])
        await h.run_until(lambda: not h.busy, 2_000_000)
        await h.idle(2)
        got = [f for tag, f in h.out_log if tag == t]
        if Counter(got) != Counter(exp):
            extra = Counter(got) - Counter(exp)
            missing = Counter(exp) - Counter(got)
            raise AssertionError(f"t={t}: extra={[hex(f) for f in extra]} missing={[hex(f) for f in missing]}")
        h.tag = -2
        pending = frozenset(loop_next + (events[t + 1] if early else []))
        full = (t % full_every == full_every - 1) or t == STEPS - 1
        await check_learn_state(h, g, rng, pending, full, None if A * N <= 4096 else 1024)
    dut._log.info(f"LEARN MODE={MODE} SEED={SEED} N={N} A={A} LANES={LANES} W={W} LEARN={LEARN}: "
                  f"{STEPS} steps OK; LEARN cycles (cycles, #pre, #post) {learn_log}; "
                  f"lfsr={g.lfsr.state:#06x} reward={g.pending_reward}; toggles {dict(toggles)}")
