"""pytest entry point for the FPGA host-link tests (SG7).

    python -m pytest -q tests/fpga

* cocotb + Icarus on kv260_top (2x2 mesh, N=A=32, LANES=4, UART at 8 clocks per
  bit): raw UART round trip, the host API over the UART backend and over the
  AXI-Stream/AXI-Lite backend, with compiled-net outputs compared to golden.
* fpga/rtl/sram_1r1w_fpga.sv (the URAM-hinted memory) is checked in place of
  rtl/core/sram_1r1w.sv: it is used in every simulation here, and the core's
  Verilator trace-replay test runs at N=A=256 (where the URAM branch applies).
* Codec unit tests for host/fpga_backend.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cocotb_tools.check_results import get_results
from cocotb_tools.runner import get_runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORE = [ROOT / "fpga/rtl/sram_1r1w_fpga.sv", ROOT / "rtl/core/learn_engine.sv", ROOT / "rtl/core/neurocore.sv"]
NOC = [ROOT / "rtl/noc" / f for f in ("noc_fifo.sv", "router.sv", "noc_host_in.sv", "noc_host_out.sv", "mesh.sv")]
TOP = [ROOT / "rtl/top/tick_ctrl.sv", ROOT / "rtl/top/spikemesh_top.sv"]
FPGA = [ROOT / "fpga/rtl" / f for f in ("flit_link.sv", "uart_rx.sv", "uart_tx.sv", "uart_bridge.sv",
                                         "axis_bridge.sv", "kv260_top.sv")]
SRC = CORE + NOC + TOP + FPGA
CFG = dict(MESH_X=2, MESH_Y=2, N_NEURONS=32, N_AXONS=32, W_BITS=8, FANOUT_DEPTH=64, LANES=4,
           CLK_HZ=8_000_000, BAUD=1_000_000)
ENV = {"MX": "2", "MY": "2", "N": "32", "A": "32", "FD": "64", "LANES": "4", "CPB": "8"}
_runners: dict = {}


def _run(host_sel: int, testcase: str, drain: int = 0, learn_mask: int = 0, **env):
    """drain = 0: STEP_DONE waits for spikemesh_top `drained`; > 0: idle-timer fallback.
    learn_mask: kv260_top LEARN_MASK (learning cores, LEARN_MODE 2)."""
    name = f"kv260_sel{host_sel}" + (f"_timer{drain}" if drain else "") + (f"_lm{learn_mask}" if learn_mask else "")
    bdir = HERE / "sim_build" / name
    r = _runners.get(name)
    if r is None:
        r = get_runner("icarus")
        r.build(sources=SRC, hdl_toplevel="kv260_top", parameters={**CFG, "HOST_SEL_RST": host_sel, "DRAIN_CYCLES": drain,
                                               "LEARN_MASK": learn_mask, "LEARN_MODE": 2},
                build_dir=bdir, always=True, timescale=("1ns", "1ps"))
        _runners[name] = r
    e = {**ENV, "DRAIN": str(drain), **{k: str(v) for k, v in env.items()}}
    e["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(HERE)])
    xml = r.test(hdl_toplevel="kv260_top", test_module="fpga_tb", testcase=testcase, build_dir=bdir,
                 test_dir=bdir, extra_env=e, results_xml=str(bdir / f"res_{testcase}.xml"))
    total, failed = get_results(Path(xml))
    assert total > 0 and failed == 0, f"{failed}/{total} cocotb tests failed"


def test_uart_roundtrip():
    _run(1, "test_uart_roundtrip")


def test_uart_backend_net():
    _run(1, "test_uart_backend_net", SEED=3)


@pytest.mark.parametrize("seed,bp", [(5, 0.3), (6, 0.7)])
def test_axi_backend_net(seed, bp):
    _run(0, "test_axi_backend_net", SEED=seed, BP=bp)


def test_axi_backend_net_learning_cores():
    """LEARN_MASK pass-through: cores (0,0) and (1,1) built with R-STDP hardware;
    with learn_en = 0 (compiled inference image) the outputs must still match golden."""
    _run(0, "test_axi_backend_net", learn_mask=0b1001, SEED=9, BP=0.3)


def test_axi_backend_net_timer_fallback():
    """DRAIN_CYCLES > 0: STEP_DONE after an idle timer instead of `drained`."""
    _run(0, "test_axi_backend_net", drain=32, SEED=8, BP=0.3)


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not available")
def test_fpga_sram_core_256():
    """neurocore at N=A=256/LANES=8 with the URAM-hinted sram (trace replay vs golden)."""
    work = HERE / "sim_build" / "vl_core_fpga_sram"
    work.mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-j", "4",
           "--top-module", "tb_replay", "-Mdir", str(work / "obj"), "-o", "Vtb",
           "-GN_NEURONS=256", "-GN_AXONS=256", "-GLANES=8", "-GFANOUT_DEPTH=1024", "-GX=2", "-GY=1",
           str(ROOT / "tests/core/tb_replay.sv"), *map(str, CORE)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    env = dict(os.environ, N="256", A="256", LANES="8", W="8", FD="1024", X="2", Y="1", SEED="7",
               MODE="mixed", STEPS="8", BP="0.3", LEARN="0",
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests/core")]))
    r = subprocess.run([sys.executable, str(ROOT / "tests/core/vl_replay.py"), str(work / "obj" / "Vtb"), str(work)],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "vl_replay OK" in r.stdout


def test_codec():
    from host import fpga_backend as FB
    from compiler import flits as F
    words = [F.spike(1, 2, 3, 1), F.cfg_write(0, 0, F.addr_reg(0), 77), FB.cmd_word(FB.CMD_STEP), (1 << 48) - 1]
    data = FB.pack_words(words)
    assert len(data) == 24 and data[:6] == words[0].to_bytes(6, "little")
    assert FB.unpack_words(data) == words
    assert FB.is_link_word(FB.cmd_word(FB.CMD_FLUSH))
    # no chip flit a host could send or receive uses the link header
    for f in (F.host_spike(14, 15, 1023, 1), F.read_resp(15, 15, 0xFFFF, 0xFFFF), F.reward(14, 15, -1),
              F.spike(14, 15, 1023, 1), F.read_req(14, 15, 0x3FFFFF, 15, 15)):
        assert not FB.is_link_word(f), hex(f)
    st = FB.decode_event(FB.HDR | 6 << 32 | 3 << 28 | 2 << 24 | 9 << 16 | 0b101 << 8 | FB.EV_STATUS)
    assert st == {"op": FB.EV_STATUS, "t_parity": 1, "err": 0, "running": 1, "drop_cnt": 9,
                  "mesh_x": 2, "mesh_y": 3, "n_neurons": 64}
    with pytest.raises(ValueError):
        FB.unpack_words(b"\x00" * 7)
