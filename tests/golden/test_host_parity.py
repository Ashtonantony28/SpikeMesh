"""GoldenBackend tags host-bound spikes like the RTL core: parity p ^ 1 for UPDATE(t) (SPEC §3).

Found by the Confluence SoC chip env, which compares raw host flits; the system tests only
compared decoded rasters, which drop the parity field."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "system"))

from compiler import flits as F  # noqa: E402
from compiler.compile import compile_net  # noqa: E402
from host.api import GoldenBackend, SpikeMeshHost  # noqa: E402
from sysutil import rand_net  # noqa: E402


def test_host_spike_parity_is_next_step():
    rng = np.random.default_rng(7)
    net = rand_net(rng, [16, 12, 8])
    prog = compile_net(net, 2, 2, 32, 32, fanout_depth=128, probe_layers=(0,))
    be = GoldenBackend.for_program(prog)
    h = SpikeMeshHost(be)
    h.load(prog)
    seen = 0
    for t in range(12):
        p = be.chip.t & 1
        fl = [F.spike(x, y, a, p) for k in range(16) if rng.random() < 0.5 for (x, y, a) in prog.input_map[k]]
        be.send_flits(fl)
        be.step()
        for f in be.recv_flits():
            d = F.decode(f)
            if d["type"] == F.SPIKE and d["dest_x"] == F.HOST_X:
                assert d["parity"] == p ^ 1, f"t={t}: host spike tagged {d['parity']}, want {p ^ 1}"
                seen += 1
    assert seen > 0
