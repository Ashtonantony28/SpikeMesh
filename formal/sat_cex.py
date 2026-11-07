#!/usr/bin/env python3
"""Summarise a yosys `sat -show-all` counterexample: list the assertions
(source file:line) that are enabled and false, per time step.
Usage: formal/sat_cex.py yosys.log"""
import re
import sys
from collections import defaultdict

rows = defaultdict(dict)
pat = re.compile(r"^\s+(\d+)\s+(\S*formal\$(\S+?):(\d+)\$(\d+)_(CHECK|EN)\S*)\s+(\S+)")
for line in open(sys.argv[1]):
    m = pat.match(line)
    if m:
        step, full, f, ln, uid, kind, val = m.groups()
        scope = full.split("$0$formal")[0]
        rows[(int(step), scope + f + ":" + ln, uid)][kind] = val
bad = sorted({(s, loc) for (s, loc, _), d in rows.items() if d.get("CHECK") == "0" and d.get("EN", "1") == "1"})
for s, loc in bad:
    print(f"step {s}: assertion failed at {loc}")
if not bad:
    print("no failing assertion found in log")
