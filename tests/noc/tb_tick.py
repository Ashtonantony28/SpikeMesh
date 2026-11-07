"""cocotb tests for rtl/top/tick_ctrl.sv. Inputs change at the falling edge;
outputs are sampled 1 ns before the rising edge (the values the edge sees)."""
from __future__ import annotations

import random

import cocotb
from cocotb.triggers import FallingEdge, Timer

from nocutil import ival, reset, start


class TickEnv:
    def __init__(self, dut, seed):
        self.dut = dut
        self.NC = int(dut.NC.value)
        self.Q = int(dut.QUIESCE_CYCLES.value)
        self.G = int(dut.START_GUARD.value)
        self.rng = random.Random(seed)
        self.cycle = 0
        self.hist = []        # per cycle: dict of inputs/outputs
        self.core_left = [0] * self.NC
        self.net_left = 0
        self.errors = []

    async def tick(self, step=0, net=None, core=None):
        """Advance one cycle. net/core override the internal busy models."""
        dut = self.dut
        await FallingEdge(dut.clk)
        nb = (1 if self.net_left > 0 else 0) if net is None else net
        cb = sum((1 << k) for k in range(self.NC) if self.core_left[k] > 0) if core is None else core
        dut.step.value = step
        dut.net_busy.value = nb
        dut.core_busy.value = cb
        await Timer(4, unit="ns")
        rec = dict(c=self.cycle, step=step, net=nb, core=cb,
                   ls=ival(dut.learn_start), us=ival(dut.update_start), sd=ival(dut.step_done),
                   q=ival(dut.quiescent), dr=ival(dut.drained), par=ival(dut.parity), run=ival(dut.running), err=ival(dut.step_err))
        self.hist.append(rec)
        self.cycle += 1
        self.net_left = max(0, self.net_left - 1)
        self.core_left = [max(0, v - 1) for v in self.core_left]
        return rec

    def check_invariants(self):
        """SPEC §3/§8 checks over the recorded trace."""
        H, Q = self.hist, self.Q
        phase = 0                      # 0 idle, 1 wait learn, 2 wait update, 3 wait done
        for i, r in enumerate(H):
            raw = lambda j: H[j]["net"] or H[j]["core"]  # noqa: E731
            if r["q"]:
                lo = [raw(j) for j in range(max(0, i - Q - 1), i)]   # inputs before the trace are 0
                if any(lo):
                    self.errors.append(f"cycle {i}: quiescent with busy in last {Q + 1} cycles")
            if r["dr"]:
                lo = [raw(j) for j in range(max(0, i - Q - 1), i)]
                if any(lo) or r["run"]:
                    self.errors.append(f"cycle {i}: drained while busy or running")
            if r["sd"] and r["core"]:
                self.errors.append(f"cycle {i}: step_done while core busy")
            if r["ls"]:
                if phase != 1:
                    self.errors.append(f"cycle {i}: learn_start out of order (phase {phase})")
                if not H[i - 1]["q"]:
                    self.errors.append(f"cycle {i}: learn_start without quiescence")
                phase = 2
            if r["us"]:
                if phase != 2 or H[i - 1]["core"]:
                    self.errors.append(f"cycle {i}: update_start out of order or with busy core")
                phase = 3
            if r["sd"]:
                if phase != 3:
                    self.errors.append(f"cycle {i}: step_done out of order")
                if i + 1 < len(H) and H[i + 1]["par"] == r["par"]:
                    self.errors.append(f"cycle {i}: parity did not flip")
                phase = 0
            if r["step"] and phase == 0 and not r["run"]:
                phase = 1
        assert not self.errors, "\n".join(self.errors[:20])


@cocotb.test()
async def ordering_idle(dut):
    """No traffic, no busy cores: step -> learn_start -> update_start ->
    step_done, each once, parity flips; report the minimum STEP latency."""
    await start(dut)
    env = TickEnv(dut, 1)
    for _ in range(8):
        await env.tick()
    p0 = ival(dut.parity)
    await env.tick(step=1)
    s = env.cycle - 1
    for _ in range(30):
        await env.tick()
    ev = {k: [r["c"] for r in env.hist if r[k]] for k in ("ls", "us", "sd")}
    assert len(ev["ls"]) == len(ev["us"]) == len(ev["sd"]) == 1, ev
    assert s < ev["ls"][0] < ev["us"][0] < ev["sd"][0]
    assert ival(dut.parity) == p0 ^ 1
    env.check_invariants()
    dut._log.info("idle STEP: learn +%d, update +%d, done +%d cycles after step",
                  ev["ls"][0] - s, ev["us"][0] - s, ev["sd"][0] - s)


@cocotb.test()
async def waits_for_quiescence(dut):
    """Traffic (net_busy) in flight after step: learn_start must wait until
    busy has been low QUIESCE_CYCLES cycles, and must come soon after."""
    await start(dut)
    env = TickEnv(dut, 2)
    for trial in range(30):
        # traffic is already in flight when step arrives (e.g. spikes from UPDATE(t-1))
        await env.tick(net=1)
        await env.tick(step=1, net=1)
        # bursts of activity with short gaps (< Q) that must NOT count as quiet
        for _ in range(env.rng.randrange(1, 5)):
            for _ in range(env.rng.randrange(1, 10)):
                await env.tick(net=1)
            for _ in range(env.rng.randrange(0, env.Q)):
                await env.tick(net=0)
        if env.rng.random() < 0.5:
            for _ in range(env.rng.randrange(1, 6)):
                await env.tick(net=0, core=1 << env.rng.randrange(env.NC))
        last_busy = max(r["c"] for r in env.hist if r["net"] or r["core"])
        seen = None
        for _ in range(60):
            r = await env.tick(net=0, core=0)
            if r["ls"]:
                seen = r["c"]
                break
        if seen is None:
            for r in env.hist[-80:]:
                dut._log.error("%s", r)
        assert seen is not None, "learn_start never came"
        assert seen - last_busy >= env.Q + 1, f"learn_start {seen - last_busy} cycles after last busy"
        assert seen - last_busy <= env.Q + 3, f"quiescence detection too slow: {seen - last_busy}"
        while not (await env.tick(net=0, core=0))["sd"]:
            pass
    env.check_invariants()
    dut._log.info("30 STEPs with in-flight traffic: quiescence respected")


@cocotb.test()
async def waits_for_busy_cores(dut):
    """Cores raise busy one cycle after learn_start / update_start for random
    lengths; update_start and step_done wait for every core."""
    await start(dut)
    env = TickEnv(dut, 3)
    for trial in range(40):
        await env.tick(step=1)
        lengths = []
        while True:
            r = await env.tick()
            if r["ls"] or r["us"]:
                ln = [env.rng.choice([0, 1, 2, env.rng.randrange(3, 40)]) for _ in range(env.NC)]
                # busy visible from the next cycle (registered response to the pulse)
                env.core_left = [v + 1 if v else 0 for v in ln]
                env.pending_core = ln
                lengths.append((r["c"], max(ln)))
            if r["sd"]:
                break
            assert env.cycle < 100000
        (cl, ml), (cu, mu) = lengths
        assert cu >= cl + ml + 1, (cl, ml, cu)
        assert env.hist[-1]["c"] >= cu + mu
    env.check_invariants()
    dut._log.info("40 STEPs with busy cores OK")


@cocotb.test()
async def step_while_running(dut):
    """A step pulse during a running STEP is ignored and sets step_err."""
    await start(dut)
    env = TickEnv(dut, 4)
    await env.tick(step=1)
    await env.tick(step=0, net=1)
    await env.tick(step=1, net=1)
    for _ in range(40):
        await env.tick(net=0)
    assert sum(r["sd"] for r in env.hist) == 1
    assert ival(dut.step_err) == 1
    await reset(dut)
    assert ival(dut.step_err) == 0


@cocotb.test()
async def random_soak(dut):
    """Random busy activity at all times and random step pulses (only when
    idle); invariants checked on every cycle."""
    await start(dut)
    env = TickEnv(dut, 5)
    steps = 0
    for _ in range(6000):
        if env.rng.random() < 0.05:
            env.net_left = env.rng.randrange(1, 12)
        if env.rng.random() < 0.05:
            env.core_left[env.rng.randrange(env.NC)] = env.rng.randrange(1, 15)
        running = (env.hist[-1]["run"] or env.hist[-1]["step"]) if env.hist else 0
        st = 1 if (not running and env.rng.random() < 0.2) else 0
        r = await env.tick(step=st)
        steps += r["sd"]
    env.check_invariants()
    assert steps > 20, steps
    assert ival(dut.step_err) == 0
    dut._log.info("soak: %d STEPs completed, invariants hold", steps)
