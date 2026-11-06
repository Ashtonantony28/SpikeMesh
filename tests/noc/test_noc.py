"""pytest entry points for the SpikeMesh NoC / top-level cocotb tests.
Run: python3 -m pytest -q tests/noc   (Icarus Verilog 12, cocotb 2.x)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cocotb_tools.check_results import get_results
from cocotb_tools.runner import get_runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NOC = [ROOT / "rtl/noc" / f for f in ("noc_fifo.sv", "router.sv", "noc_host_in.sv", "noc_host_out.sv", "mesh.sv")]
# cocotb's runner exports sys.path as the simulator's PYTHONPATH
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
TOP = NOC + [ROOT / "rtl/top/tick_ctrl.sv", ROOT / "rtl/top/spikemesh_top.sv", HERE / "core_stub.sv"]


def run(name, toplevel, sources, module, params, testcase=None):
    build = HERE / "sim_build" / name
    r = get_runner("icarus")
    r.build(sources=sources, hdl_toplevel=toplevel, parameters=params, build_dir=build, always=True,
            timescale=("1ns", "1ps"))
    xml = r.test(hdl_toplevel=toplevel, test_module=module, build_dir=build, test_dir=build,
                 testcase=testcase, seed=12345)
    total, failed = get_results(Path(xml))
    assert total > 0 and failed == 0, f"{failed}/{total} cocotb tests failed in {name}"


@pytest.mark.parametrize("x,y,mx,my", [(1, 1, 3, 3), (0, 0, 2, 2), (0, 1, 2, 2), (1, 0, 3, 3), (2, 2, 3, 3), (0, 0, 1, 1)])
def test_router(x, y, mx, my):
    run(f"router_{x}{y}_{mx}{my}", "router", NOC[:2], "tb_router",
        {"X": x, "Y": y, "MESH_X": mx, "MESH_Y": my, "FIFO_DEPTH": 4})


@pytest.mark.parametrize("mx,my", [(2, 2), (3, 3)])
def test_mesh(mx, my):
    run(f"mesh_{mx}x{my}", "mesh", NOC, "tb_mesh", {"MESH_X": mx, "MESH_Y": my, "FIFO_DEPTH": 4})


def test_mesh_3x2():
    run("mesh_3x2", "mesh", NOC, "tb_mesh", {"MESH_X": 3, "MESH_Y": 2, "FIFO_DEPTH": 2}, testcase="all_to_all_stress")


@pytest.mark.parametrize("q", [4, 7])
def test_tick_ctrl(q):
    run(f"tick_q{q}", "tick_ctrl", [ROOT / "rtl/top/tick_ctrl.sv"], "tb_tick",
        {"NC": 4, "QUIESCE_CYCLES": q, "START_GUARD": 2})


@pytest.mark.parametrize("mx,my", [(2, 2), (3, 3)])
def test_top(mx, my):
    run(f"top_{mx}x{my}", "spikemesh_top", TOP, "tb_top",
        {"MESH_X": mx, "MESH_Y": my, "N_NEURONS": 16, "N_AXONS": 16, "FANOUT_DEPTH": 16})
