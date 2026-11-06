"""cocotb stress tests for rtl/noc/mesh.sv: random all-to-all traffic (cores
and host) with random sink backpressure. Checks exactly-once delivery to the
right node, in-order delivery per (src, dst) pair, and a deadlock bound."""
from __future__ import annotations

import random
from collections import defaultdict, deque

import cocotb
from cocotb.triggers import FallingEdge

from nocutil import HOST_X, dval, get_field, ival, mk, reset, set_field, start, tag_of


class MeshEnv:
    HOST = -1

    def __init__(self, dut, seed, inj_rate, sink_rate):
        self.dut = dut
        self.MX = int(dut.MESH_X.value)
        self.MY = int(dut.MESH_Y.value)
        self.NC = self.MX * self.MY
        self.rng = random.Random(seed)
        self.inj_rate, self.sink_rate = inj_rate, sink_rate
        self.txq = {s: deque() for s in [self.HOST] + list(range(self.NC))}
        self.hold = {s: None for s in self.txq}
        self.expect = {}                         # tag -> (src, dst_node)
        self.sent = defaultdict(list)            # (src, dst) -> [tag] in send order
        self.recv = defaultdict(list)            # dst node -> [tag]
        self.cycle = 0
        self.errors = []

    def node_xy(self, n):
        return n % self.MX, n // self.MX

    def add_packets(self, per_src, oob_frac=0.0):
        dsts = list(range(self.NC)) + [self.HOST]
        for s in self.txq:
            for k in range(per_src):
                tag = ((s + 1) << 24) | k
                if self.rng.random() < oob_frac:
                    f = mk(self.rng.choice([self.MX, 14]), self.rng.randrange(16), tag)
                    self.txq[s].append((f, None))
                    continue
                d = self.rng.choice(dsts)
                if d == self.HOST:
                    f = mk(HOST_X, self.rng.randrange(16), tag)   # dest_y ignored for host
                else:
                    f = mk(*self.node_xy(d), tag)
                self.txq[s].append((f, d))
                self.expect[tag] = (s, d)

    def pending(self):
        return sum(len(q) for q in self.txq.values()) + sum(h is not None for h in self.hold.values())

    async def run(self, max_cycles):
        dut, rng, NC = self.dut, self.rng, self.NC
        dut.host_in_valid.value = 0
        dut.core_out_valid.value = 0
        dut.core_in_ready.value = 0
        dut.host_out_ready.value = 0
        delivered_target = len(self.expect)
        n_recv = 0
        for _ in range(max_cycles):
            await FallingEdge(dut.clk)
            self.cycle += 1
            # ---- sinks ----
            civ = ival(dut.core_in_valid)
            cif = dval(dut.core_in_flit)
            rdy = 0
            for n in range(NC):
                if rng.random() < self.sink_rate:
                    rdy |= 1 << n
                    if civ >> n & 1:
                        self.recv[n].append(tag_of(get_field(cif, n, 48)))
                        n_recv += 1
            dut.core_in_ready.value = rdy
            hr = 1 if rng.random() < self.sink_rate else 0
            dut.host_out_ready.value = hr
            if hr and ival(dut.host_out_valid):
                self.recv[self.HOST].append(tag_of(dval(dut.host_out_flit)))
                n_recv += 1
            # ---- sources (valid/ready, valid held until accepted) ----
            cor = ival(dut.core_out_ready)
            hir = ival(dut.host_in_ready)
            cov, cof, hiv = 0, 0, 0
            for s in self.txq:
                if self.hold[s] is None and self.txq[s] and rng.random() < self.inj_rate:
                    self.hold[s] = self.txq[s].popleft()
                h = self.hold[s]
                if h is None:
                    continue
                f, d = h
                if s == self.HOST:
                    hiv = 1
                    dut.host_in_flit.value = f
                    ready = hir
                else:
                    cov |= 1 << s
                    cof = set_field(cof, s, 48, f)
                    ready = cor >> s & 1
                if ready:
                    if d is not None:
                        self.sent[(s, d)].append(tag_of(f))
                    self.hold[s] = None
            dut.host_in_valid.value = hiv
            dut.core_out_valid.value = cov
            dut.core_out_flit.value = cof
            if n_recv >= delivered_target and self.pending() == 0 and ival(dut.busy) == 0:
                return True
        return False

    def check(self):
        where = {}
        for d, tags in self.recv.items():
            for t in tags:
                if t in where:
                    self.errors.append(f"tag {t:#x} delivered twice")
                if t not in self.expect:
                    self.errors.append(f"unexpected tag {t:#x} at {d}")
                where[t] = d
        for t, (s, d) in self.expect.items():
            if t not in where:
                self.errors.append(f"tag {t:#x} {s}->{d} lost")
            elif where[t] != d:
                self.errors.append(f"tag {t:#x} {s}->{d} delivered to {where[t]}")
        pos = {}
        for d, tags in self.recv.items():
            for k, t in enumerate(tags):
                pos[t] = k
        for (s, d), tags in self.sent.items():
            ks = [pos[t] for t in tags if t in pos]
            if ks != sorted(ks):
                self.errors.append(f"out-of-order delivery {s}->{d}")
        assert not self.errors, "\n".join(self.errors[:20])


async def stress(dut, per_src, scenarios, max_cycles=60000):
    await start(dut)
    for k, (seed, inj, sink) in enumerate(scenarios):
        await reset(dut)
        env = MeshEnv(dut, seed, inj, sink)
        env.add_packets(per_src)
        ok = await env.run(max_cycles)
        assert ok, (f"no drain within {max_cycles} cycles (deadlock?) scenario {k}: pending={env.pending()} "
                    f"recv={sum(len(v) for v in env.recv.values())}/{len(env.expect)} busy={ival(dut.busy)}")
        env.check()
        assert ival(dut.err) == 0
        dut._log.info("%dx%d scenario %d (inj %.2f, sink %.2f): %d packets delivered exactly once, in order, in %d cycles",
                      env.MX, env.MY, k, inj, sink, len(env.expect), env.cycle)


@cocotb.test()
async def all_to_all_stress(dut):
    """Random all-to-all traffic incl. host, under several load/backpressure mixes."""
    nc = int(dut.MESH_X.value) * int(dut.MESH_Y.value)
    per_src = 150 if nc <= 4 else 90
    await stress(dut, per_src, [(1, 1.0, 1.0), (2, 1.0, 0.3), (3, 0.3, 0.1), (4, 1.0, 0.05), (5, 0.7, 0.7)])


@cocotb.test()
async def hotspot_host(dut):
    """Every core and the host send only to the host while the host drains slowly."""
    await start(dut)
    env = MeshEnv(dut, 9, 1.0, 0.25)
    for s in env.txq:
        for k in range(40):
            tag = ((s + 1) << 24) | k
            env.txq[s].append((mk(HOST_X, 0, tag), env.HOST))
            env.expect[tag] = (s, env.HOST)
    assert await env.run(60000), "hotspot did not drain"
    env.check()
    dut._log.info("hotspot: %d packets to host OK in %d cycles", len(env.expect), env.cycle)


@cocotb.test()
async def out_of_mesh_dropped(dut):
    """Packets to off-mesh coordinates are dropped (sticky err, drop_cnt)
    and do not disturb the rest of the traffic."""
    await start(dut)
    env = MeshEnv(dut, 21, 0.8, 0.8)
    env.add_packets(40, oob_frac=0.1)
    n_oob = sum(1 for q in env.txq.values() for (_, d) in q if d is None)
    assert await env.run(30000), "did not drain"
    env.check()
    assert n_oob > 0 and ival(dut.err) == 1 and ival(dut.drop_cnt) == min(n_oob, 255), (n_oob, ival(dut.drop_cnt))
    dut._log.info("out-of-mesh: %d dropped, err=1, other %d delivered", n_oob, len(env.expect))
