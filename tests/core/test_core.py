"""pytest entry point for the neurocore cocotb tests (SG2).

    python -m pytest -q tests/core            # Icarus, all configs
    SPIKEMESH_VERILATOR=0 ...                 # skip the Verilator run
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
RTL = [ROOT / "rtl/core/sram_1r1w.sv", ROOT / "rtl/core/neurocore.sv"]
BUILD = ROOT / "tests/core/sim_build"

CONFIGS = {
    "small": dict(N=32, A=32, LANES=4, W=8, FD=64),
    "w4": dict(N=32, A=32, LANES=8, W=4, FD=64),
    "g1": dict(N=16, A=16, LANES=16, W=8, FD=32),     # one group per row: forwarding path
    "l1": dict(N=16, A=32, LANES=1, W=8, FD=32),      # one neuron per cycle
    "big": dict(N=256, A=256, LANES=8, W=8, FD=1024),
}
X, Y = 2, 1
_runners: dict = {}


def _run(sim, cfg_name, testcase, **env):
    c = CONFIGS[cfg_name]
    bdir = BUILD / f"{sim}_{cfg_name}"
    runner = _runners.get((sim, cfg_name))
    if runner is None:
        runner = get_runner(sim)
        params = dict(X=X, Y=Y, N_NEURONS=c["N"], N_AXONS=c["A"], W_BITS=c["W"],
                      FANOUT_DEPTH=c["FD"], LANES=c["LANES"])
        build_args = ["-Wno-fatal"] if sim == "verilator" else []
        runner.build(sources=RTL, hdl_toplevel="neurocore", parameters=params,
                     build_dir=bdir, always=True, build_args=build_args, timescale=("1ns", "1ps"))
        _runners[(sim, cfg_name)] = runner
    full_env = {"N": c["N"], "A": c["A"], "LANES": c["LANES"], "W": c["W"], "FD": c["FD"], "X": X, "Y": Y}
    full_env.update(env)
    full_env = {k: str(v) for k, v in full_env.items()}
    full_env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "tests/core"), os.environ.get("PYTHONPATH", "")])
    results = runner.test(hdl_toplevel="neurocore", test_module="core_tb", testcase=testcase,
                          build_dir=bdir, test_dir=bdir, extra_env=full_env,
                          results_xml=str(bdir / ("results_" + testcase + "_" + "_".join(f"{k}{v}" for k, v in sorted(env.items())) + ".xml")))
    from cocotb_tools.runner import get_results
    num_tests, num_failed = get_results(results)
    assert num_tests >= 1 and num_failed == 0


# ---------------- small config: broad randomized coverage ----------------
@pytest.mark.parametrize("seed,mode", [
    (1, "mixed"), (2, "mixed"), (3, "mixed"), (4, "mixed"), (5, "sparse"),
    (6, "sat"), (7, "sat"), (8, "refrac"), (9, "leakneg"), (10, "reset0"), (11, "reset1"),
])
def test_small_random(seed, mode):
    _run("icarus", "small", "test_random", SEED=seed, MODE=mode, STEPS=24, BP=0.3)


def test_small_random_heavy_backpressure():
    _run("icarus", "small", "test_random", SEED=21, MODE="mixed", STEPS=10, BP=0.8, IN_GAP=0.5)


def test_small_cfg_readback():
    _run("icarus", "small", "test_cfg_readback", SEED=3, BP=0.5)


def test_small_cycles():
    _run("icarus", "small", "test_cycles", SEED=1)


@pytest.mark.parametrize("seed", [31, 32])
def test_w4_random(seed):
    _run("icarus", "w4", "test_random", SEED=seed, MODE="mixed", STEPS=10)


def test_w4_cfg_readback():
    _run("icarus", "w4", "test_cfg_readback", SEED=4)


@pytest.mark.parametrize("cfg", ["g1", "l1"])
def test_edge_lanes(cfg):
    _run("icarus", cfg, "test_random", SEED=41, MODE="mixed", STEPS=10)
    _run("icarus", cfg, "test_cfg_readback", SEED=42)


# ---------------- full-size config ----------------
def test_big_random():
    _run("icarus", "big", "test_random", SEED=51, MODE="sparse", STEPS=4, BP=0.3)


def test_big_accsat():
    _run("icarus", "big", "test_random", SEED=52, MODE="accsat", STEPS=3, BP=0.2)


def test_big_cycles():
    _run("icarus", "big", "test_cycles", SEED=1)


# ---------------- Verilator ----------------
# cocotb 2.1's Verilator support needs Verilator >= 5.036 (5.020 installed), so
# the Verilator run replays a golden-generated command trace through
# tb_replay.sv (see vl_replay.py) instead of using cocotb.
_vl_bins: dict = {}


def _vl(cfg_name, **env):
    import subprocess
    import sys
    c = CONFIGS[cfg_name]
    work = BUILD / f"vl_{cfg_name}"
    work.mkdir(parents=True, exist_ok=True)
    if cfg_name not in _vl_bins:
        cmd = ["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
               "-j", "4", "--top-module", "tb_replay", "-Mdir", str(work / "obj"), "-o", "Vtb",
               f"-GX={X}", f"-GY={Y}", f"-GN_NEURONS={c['N']}", f"-GN_AXONS={c['A']}",
               f"-GW_BITS={c['W']}", f"-GFANOUT_DEPTH={c['FD']}", f"-GLANES={c['LANES']}",
               str(ROOT / "tests/core/tb_replay.sv"), *map(str, RTL)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        _vl_bins[cfg_name] = work / "obj" / "Vtb"
    full_env = dict(os.environ)
    full_env.update({"N": c["N"], "A": c["A"], "LANES": c["LANES"], "W": c["W"], "FD": c["FD"],
                     "X": X, "Y": Y, **env})
    full_env = {k: str(v) for k, v in full_env.items()}
    full_env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "tests/core")])
    r = subprocess.run([sys.executable, str(ROOT / "tests/core/vl_replay.py"), str(_vl_bins[cfg_name]), str(work)],
                       env=full_env, capture_output=True, text=True, timeout=900)
    print(r.stdout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


needs_vl = pytest.mark.skipif(shutil.which("verilator") is None or os.environ.get("SPIKEMESH_VERILATOR") == "0",
                              reason="verilator not available")


@needs_vl
@pytest.mark.parametrize("seed,mode", [(61, "mixed"), (62, "sat"), (63, "refrac"), (64, "leakneg")])
def test_verilator_small_random(seed, mode):
    _vl("small", SEED=seed, MODE=mode, STEPS=40, BP=0.3)


@needs_vl
@pytest.mark.parametrize("seed,mode", [(71, "mixed"), (72, "accsat")])
def test_verilator_big_random(seed, mode):
    _vl("big", SEED=seed, MODE=mode, STEPS=8, BP=0.3)
