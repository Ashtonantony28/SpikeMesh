"""cocotb system tests for rtl/top/spikemesh_top.sv with the behavioural
core stub (tests/noc/core_stub.sv). Checks the STEP protocol end to end:
every parity-p spike reaches its core before learn_start of STEP t (the stub
reports its per-parity receive count at learn_start), every host-bound flit
arrives exactly once, parity toggles, no NoC error."""
from __future__ import annotations

import random
from collections import Counter, deque

import cocotb
from cocotb.triggers import FallingEdge

from nocutil import dval, flits, ival, start

UPD_HOST, UPD_FWD = 2, 1          # core_stub defaults


class TopEnv:
    def __init__(self, dut, seed, sink_rate):
        self.dut = dut
        self.MX = int(dut.MESH_X.value)
        self.MY = int(dut.MESH_Y.value)
        self.NC = self.MX * self.MY
        self.rng = random.Random(seed)
        self.sink_rate = sink_rate
        self.hin = deque()
        self.hold = None
        self.host_rx = []
        self.cycle = 0
        self.step_req = False
        self.ev = Counter()          # events seen this cycle: ls/us/sd

    async def driver(self):
        dut, rng = self.dut, self.rng
        dut.host_in_valid.value = 0
        dut.host_out_ready.value = 0
        dut.step.value = 0
        while True:
            await FallingEdge(dut.clk)
            self.cycle += 1
            for k, sig in (("ls", dut.u_tick.learn_start), ("us", dut.u_tick.update_start), ("sd", dut.step_done)):
                if ival(sig):
                    self.ev[k] += 1
            hr = 1 if rng.random() < self.sink_rate else 0
            if hr and ival(dut.host_out_valid):
                self.host_rx.append(dval(dut.host_out_flit))
            dut.host_out_ready.value = hr
            if self.hold is None and self.hin and rng.random() < 0.8:
                self.hold = self.hin.popleft()
            if self.hold is not None:
                dut.host_in_valid.value = 1
                dut.host_in_flit.value = self.hold
                if ival(dut.host_in_ready):
                    self.hold = None
            else:
                dut.host_in_valid.value = 0
            dut.step.value = 1 if self.step_req else 0
            self.step_req = False

    async def wait_event(self, key, limit=20000):
        n0 = self.ev[key]
        for _ in range(limit):
            await FallingEdge(self.dut.clk)
            if self.ev[key] > n0:
                return
        raise AssertionError(f"timeout waiting for {key}")


@cocotb.test()
async def step_protocol(dut):
    await start(dut, zero=("step", "host_in_valid", "host_out_ready"))
    nsteps = 12
    env = TopEnv(dut, 7, 0.6)
    cocotb.start_soon(env.driver())
    rng = env.rng
    MX, MY, NC = env.MX, env.MY, env.NC
    nodes = [(x, y) for y in range(MY) for x in range(MX)]
    exp_report = {}                   # (core, step) -> count
    exp_host = Counter()
    cnt = Counter()                   # (core, parity) -> spikes for the next report

    def host_events(t, n):
        p = t & 1
        for _ in range(n):
            (x, y) = rng.choice(nodes)
            fwd = rng.random() < 0.3
            ax = (512 if fwd else 0) | rng.randrange(256)
            env.hin.append(flits.spike(x, y, ax, p))
            cnt[((x, y), p)] += 1
            if fwd:
                exp_host[flits.host_spike(x, y, ax & 255, p)] += 1

    host_events(0, rng.randrange(5, 40))
    for t in range(nsteps):
        p = t & 1
        assert ival(dut.t_parity) == p
        env.step_req = True
        await env.wait_event("ls")
        for (x, y) in nodes:
            exp_report[((x, y), t)] = cnt[((x, y), p)]
            cnt[((x, y), p)] = 0
            exp_host[flits.host_spike(x, y, 512 + exp_report[((x, y), t)], p)] += 1
        await env.wait_event("us")
        # UPDATE(t) emissions, integrated at t+1 (parity p^1)
        for (x, y) in nodes:
            for k in range(UPD_HOST):
                exp_host[flits.host_spike(x, y, 256 + k, p ^ 1)] += 1
            for k in range(UPD_FWD):
                ax = (x * 32 + y * 2 + (k & 1)) & 255
                cnt[((0, 0), p ^ 1)] += 1
                exp_host[flits.host_spike(0, 0, ax, p ^ 1)] += 1
        # host events for t+1 may be injected any time after UPDATE(t) starts
        host_events(t + 1, rng.randrange(0, 20))
        await env.wait_event("sd")
        host_events(t + 1, rng.randrange(0, 20))
        for _ in range(rng.randrange(0, 30)):
            await FallingEdge(dut.clk)
    # drain: one more STEP absorbs everything, then wait for idle mesh
    for _ in range(3000):
        await FallingEdge(dut.clk)
        if not env.hin and env.hold is None and ival(dut.u_mesh.busy) == 0 and ival(dut.host_out_valid) == 0:
            break
    # pending host events for step nsteps were counted into cnt but not reported: drop those
    got = Counter(env.host_rx)
    # flits for spikes injected for step `nsteps` (forwarded ones) are expected too
    missing = exp_host - got
    extra = got - exp_host
    assert not missing and not extra, f"missing {len(missing)} extra {len(extra)}: " \
        f"{[flits.decode(f) for f in list(missing)[:3]]} / {[flits.decode(f) for f in list(extra)[:3]]}"
    assert ival(dut.err) == 0
    assert env.ev["sd"] == nsteps and env.ev["ls"] == nsteps and env.ev["us"] == nsteps
    dut._log.info("%dx%d: %d STEPs, %d host flits, all learn-time reports exact", MX, MY, nsteps, len(env.host_rx))
