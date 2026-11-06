"""Verilator flow for the randomized neurocore test (trace replay).

Generates the same kind of stimulus as core_tb.test_random (core_model.py),
writes a command trace for tb_replay.sv, runs the Verilator binary, and checks
the per-timestep output multiset and final READ_REQ readback against golden.
Config via the same env vars as core_tb.py; argv[1] = Verilator binary,
argv[2] = work directory.
"""
from __future__ import annotations

import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from compiler import flits
from core_model import (CX, CY, SEED, STEPS, MODE, BP, IN_GAP, COV, random_network, golden_step,
                        make_events, state_addrs, check_coverage)


def main(binary: str, work: str):
    work_p = Path(work)
    rng = random.Random(SEED)
    g, fl, host_axons = random_network(rng, MODE)
    events = make_events(rng, host_axons)
    lines = [f"0 {f:012x}" for f in fl] + ["1 0"]
    expected: dict[int, list[int]] = {}
    loop_next: list[int] = []
    early = False
    for t in range(STEPS):
        p = t & 1
        if not early:
            lines += [f"0 {flits.spike(CX, CY, a, p):012x}" for a in events[t]]
        if rng.random() < 0.3:
            lines.append(f"0 {flits.reward(CX, CY, rng.randint(-100, 100)):012x}")
        lines.append("1 0")
        exp, loop_next_new = golden_step(g, t, events[t] + loop_next)
        loop_next = loop_next_new
        expected[t] = exp
        lines += ["2 0", f"4 {t:x}", f"3 {p:x}"]
        early = rng.random() < 0.5
        if early:
            lines += [f"0 {flits.spike(CX, CY, a, p ^ 1):012x}" for a in events[t + 1]]
        lines.append("6 0")
    lines += ["1 0", "4 fffffffe"]
    addrs = state_addrs(rng)
    rx, ry = rng.randint(0, 15), rng.randint(0, 15)
    lines += [f"0 {flits.read_req(CX, CY, a, rx, ry):012x}" for a in addrs] + ["1 0", "9 0"]
    (work_p / "trace.txt").write_text("\n".join(lines) + "\n")
    out = work_p / "out.txt"
    res = subprocess.run([binary, f"+trace={work_p / 'trace.txt'}", f"+out={out}", f"+seed={SEED}",
                          f"+bp={int(BP * 100)}", f"+gap={int(IN_GAP * 100)}"],
                         capture_output=True, text=True, timeout=600)
    print(res.stdout.strip())
    assert res.returncode == 0, res.stderr
    got: dict[int, list[int]] = {}
    for ln in out.read_text().splitlines():
        tag, f = ln.split()
        got.setdefault(int(tag), []).append(int(f, 16))
    for t in range(STEPS):
        assert Counter(got.get(t, [])) == Counter(expected[t]), f"t={t} mismatch"
    resps = got.get(-2, [])
    assert len(resps) == len(addrs)
    for a, f in zip(addrs, resps):
        d = flits.decode(f)
        assert d["resp"] and (d["dest_x"], d["dest_y"]) == (rx, ry) and d["addr16"] == a & 0xFFFF
        assert d["data"] == g.cfg_read(a), (hex(a), d["data"], g.cfg_read(a))
    check_coverage()
    n_out = sum(len(v) for v in expected.values())
    print(f"vl_replay OK: MODE={MODE} SEED={SEED} {STEPS} steps, {n_out} output flits, "
          f"{len(addrs)} readbacks; coverage {dict(COV)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
