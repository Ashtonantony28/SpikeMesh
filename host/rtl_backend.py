"""RTL backend for the SpikeMesh host API (host/api.py ``Backend``).

Runs the full chip RTL (rtl/top/spikemesh_top.sv + rtl/noc + the real
rtl/core) in a simulator and talks to it interactively through two named
pipes and the command-driven testbench tests/system/tb_sys.sv:

    from host.api import SpikeMeshHost
    from host.rtl_backend import RtlBackend
    be = RtlBackend.for_program(prog, lanes=4, sim="verilator")   # or sim="icarus"
    host = SpikeMeshHost(be); host.load(prog); out = host.run(events)
    be.close()

Two simulators share the same harness:
  * ``sim="icarus"``: Icarus Verilog 12, no C++ build. Good for small configs.
  * ``sim="verilator"``: Verilator ``--binary --timing``. It is the fast path
    for full-size (N = A = 256) runs; the build takes about a minute and is
    cached. cocotb 2.1 needs Verilator >= 5.036 and 5.020 is installed, so
    this backend does not use cocotb.

A STEP ends when ``step_done`` has pulsed, the chip's ``drained`` output is
high (no router or core busy) and the host output FIFO is empty. Every
host-bound spike from UPDATE(t) is then in the list that the next
``recv_flits`` returns, as the Backend contract requires.

Measured cycle counts: ``step_cycles`` (one entry per step(): from the
``step`` pulse to drained) and ``sync_cycles`` (loads and reads).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from host.api import Backend

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "tests" / "system" / "tb_sys.sv"


def rtl_sources() -> list[Path]:
    srcs = sorted((ROOT / "rtl" / "noc").glob("*.sv")) + sorted((ROOT / "rtl" / "top").glob("*.sv"))
    srcs += sorted((ROOT / "rtl" / "core").glob("*.sv"))
    return srcs + [TB]


class RtlError(RuntimeError):
    pass


class RtlBackend(Backend):
    def __init__(self, mesh_x=2, mesh_y=2, n_neurons=32, n_axons=32, w_bits=8, fanout_depth=64,
                 lanes=4, *, sim="verilator", learn_mask=0, learn_mode=1, fifo_depth=4, bp=0,
                 seed=1, build_root=None):
        self.mesh_x, self.mesh_y = mesh_x, mesh_y
        self.params = dict(MESH_X=mesh_x, MESH_Y=mesh_y, N_NEURONS=n_neurons, N_AXONS=n_axons,
                           W_BITS=w_bits, FANOUT_DEPTH=fanout_depth, LANES=lanes, FIFO_DEPTH=fifo_depth,
                           LEARN_MODE=learn_mode, LEARN_MASK=int(learn_mask))
        self.sim = sim
        self.bp, self.seed = bp, seed
        self.build_root = Path(build_root or os.environ.get("SPIKEMESH_RTL_BUILD",
                                                              ROOT / "tests" / "system" / "sim_build"))
        self.binary = self._build()
        self._out: list[int] = []
        self._dirty = False
        self.step_cycles: list[int] = []
        self.sync_cycles: list[int] = []
        self.err = 0
        self.drop_cnt = 0
        self._start()

    @classmethod
    def for_program(cls, prog, **kw) -> "RtlBackend":
        return cls(prog.mesh_x, prog.mesh_y, prog.n_neurons, prog.n_axons, prog.w_bits,
                   prog.fanout_depth, **kw)

    # ------------------------------------------------------------------ build
    def _key(self) -> str:
        h = hashlib.sha1()
        for p in rtl_sources():
            h.update(p.name.encode())
            h.update(p.read_bytes())
        h.update(repr(sorted(self.params.items())).encode())
        return f"{self.sim}_{h.hexdigest()[:12]}"

    def _pval(self, k, v) -> str:
        return f"225'h{v:x}" if k == "LEARN_MASK" else str(v)

    def _build(self) -> Path:
        d = self.build_root / self._key()
        exe = d / ("Vtb" if self.sim == "verilator" else "tb.vvp")
        if exe.exists():
            return exe
        d.mkdir(parents=True, exist_ok=True)
        srcs = [str(p) for p in rtl_sources()]
        if self.sim == "verilator":
            cmd = ["verilator", "--binary", "--timing", "-O3", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                   "-j", str(min(8, os.cpu_count() or 2)), "--top-module", "tb_sys",
                   "-Mdir", str(d / "obj"), "-o", "../Vtb"]
            cmd += [f"-G{k}={self._pval(k, v)}" for k, v in self.params.items()]
        elif self.sim == "icarus":
            cmd = ["iverilog", "-g2012", "-s", "tb_sys", "-o", str(exe)]
            cmd += [f"-Ptb_sys.{k}={self._pval(k, v)}" for k, v in self.params.items()]
        else:
            raise ValueError(f"unknown simulator {self.sim!r}")
        r = subprocess.run(cmd + srcs, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not exe.exists():
            raise RtlError(f"{self.sim} build failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
        return exe

    # --------------------------------------------------------------- process
    def _start(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="smrtl_"))
        cmd_p, out_p = self._tmp / "cmd", self._tmp / "out"
        os.mkfifo(cmd_p)
        os.mkfifo(out_p)
        plus = [f"+cmd={cmd_p}", f"+out={out_p}", f"+bp={self.bp}", f"+seed={self.seed}"]
        if self.sim == "verilator":
            argv = [str(self.binary)] + plus
        else:
            argv = ["vvp", "-n", str(self.binary)] + plus
        self._log = open(self._tmp / "sim.log", "w")
        self.proc = subprocess.Popen(argv, stdout=self._log, stderr=subprocess.STDOUT)
        # the testbench opens cmd (read) first, then out (write): same order here
        self._cmd = open(cmd_p, "w")
        self._rsp = open(out_p, "r")
        self._wait_done()              # reset + initial drain

    def _sim_log(self) -> str:
        try:
            self._log.flush()
            return (self._tmp / "sim.log").read_text()[-3000:]
        except OSError:
            return ""

    def _wait_done(self) -> int:
        while True:
            line = self._rsp.readline()
            if not line:
                raise RtlError(f"simulator exited unexpectedly:\n{self._sim_log()}")
            tag, *rest = line.split()
            if tag == "F":
                self._out.append(int(rest[0], 16))
            elif tag == "D":
                self.err, self.drop_cnt = int(rest[1]), int(rest[2])
                return int(rest[0])
            else:
                raise RtlError(f"bad response line {line!r}")

    def _send(self, text: str):
        try:
            self._cmd.write(text)
            self._cmd.flush()
        except BrokenPipeError as e:
            raise RtlError(f"simulator gone:\n{self._sim_log()}") from e

    # ------------------------------------------------------------ Backend API
    def send_flits(self, flits):
        if not len(flits):
            return
        self._cmd.write("".join(f"0 {int(f):012x}\n" for f in flits))
        self._dirty = True

    def step(self):
        self._send("1 0\n")
        self.step_cycles.append(self._wait_done())
        self._dirty = False

    def sync(self) -> int:
        """Wait until all sent flits were consumed and the chip is drained."""
        self._send("2 0\n")
        c = self._wait_done()
        self.sync_cycles.append(c)
        self._dirty = False
        return c

    def recv_flits(self):
        if self._dirty:
            self.sync()
        out, self._out = self._out, []
        return out

    def close(self):
        if getattr(self, "proc", None) is None:
            return
        try:
            self._send("9 0\n")
            self._cmd.close()
        except (RtlError, OSError):
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._rsp.close()
        self._log.close()
        self.proc = None
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
