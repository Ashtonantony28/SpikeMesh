"""Backend-agnostic SpikeMesh host API (docs/SPEC.md §3, §4, §7).

    host = SpikeMeshHost(GoldenBackend(mesh_x, mesh_y, n_neurons, n_axons, w_bits))
    host.load(program)                       # program from compiler.compile_net / image.load
    out = host.run(events)                   # events: bool (T, n_inputs) -> bool (T, n_outputs)

A backend only moves 48-bit flits and steps time. Every backend (golden model,
cocotb/Verilator RTL, KV260 FPGA) implements the three-method interface
``Backend`` below; all compilation / encoding / decoding lives in the host.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from compiler import flits as F


class Backend(ABC):
    """Minimal chip transport.

    send_flits(flits): deliver host->chip flits (CFG_WRITE, SPIKE, READ_REQ, CTRL)
        into the host port in order. SPIKE flits carry their integration parity.
    step(): run one STEP(t) (SPEC §3) to completion (step_done).
    recv_flits(): return (and forget) every chip->host flit received so far, in
        arrival order. Host-bound spikes emitted in UPDATE(t) must be returned by
        the recv_flits() call that follows step() for t.
    """

    mesh_x: int
    mesh_y: int

    @abstractmethod
    def send_flits(self, flits: list[int]) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def recv_flits(self) -> list[int]: ...


class GoldenBackend(Backend):
    """Drives golden.spikemesh.Chip purely through flits."""

    def __init__(self, mesh_x=2, mesh_y=2, n_neurons=256, n_axons=256, w_bits=8):
        from golden.spikemesh import Chip
        self.mesh_x, self.mesh_y = mesh_x, mesh_y
        self.chip = Chip(mesh_x, mesh_y, n_neurons, n_axons, w_bits)
        self._out: list[int] = []

    @classmethod
    def for_program(cls, prog) -> "GoldenBackend":
        return cls(prog.mesh_x, prog.mesh_y, prog.n_neurons, prog.n_axons, prog.w_bits)

    def _core(self, d):
        key = (d["dest_x"], d["dest_y"])
        if key not in self.chip.cores:
            raise ValueError(f"flit addressed to non-existent core {key}")
        return self.chip.cores[key]

    def send_flits(self, flits):
        for f in flits:
            d = F.decode(int(f))
            if d["type"] == F.CFG_WRITE:
                self._core(d).cfg_write(d["addr"], d["data"])
            elif d["type"] == F.SPIKE:
                if d["dest_x"] == F.HOST_X:
                    raise ValueError("host cannot send a host-bound spike")
                self._core(d)
                self.chip.inflight.append((d["dest_x"], d["dest_y"], d["axon"], d["parity"]))
            elif d["type"] == F.READ:
                if d.get("resp"):
                    raise ValueError("host cannot send READ_RESP")
                c = self._core(d)
                val = c.cfg_read(d["addr"])
                self._out.append(F.read_resp(d["reply_x"], d["reply_y"], d["addr"], val))
            elif d["type"] == F.CTRL and d["opcode"] == F.CTRL_REWARD:
                self._core(d).receive_reward(d["reward"])

    def step(self):
        p = self.chip.t & 1
        for (sx, sy, n) in self.chip.step():
            self._out.append(F.host_spike(sx, sy, n, p))

    def recv_flits(self):
        out, self._out = self._out, []
        return out


class SpikeMeshHost:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.prog = None
        self.t = 0

    def load(self, prog) -> None:
        if (prog.mesh_x, prog.mesh_y) != (self.backend.mesh_x, self.backend.mesh_y):
            raise ValueError(f"program is for a {prog.mesh_x}x{prog.mesh_y} mesh, backend is "
                             f"{self.backend.mesh_x}x{self.backend.mesh_y}")
        self.backend.send_flits(prog.flits)
        self.prog = prog

    def reset_state(self) -> None:
        """Return every used neuron to v = 0, refc = 0 with empty accumulators:
        silence fan-out, step twice to drain both parity banks, rewrite state,
        restore fan-out pointers."""
        p = self.prog
        self.backend.send_flits(p.pointer_flits(silence=True))
        for _ in range(2):
            self.backend.step()
            self.t += 1
        self.backend.recv_flits()
        self.backend.send_flits(p.state_reset_flits() + p.pointer_flits())

    def run(self, events: np.ndarray, probes: bool = False):
        """Run T timesteps from the current state. events: bool (T, n_inputs).

        Returns bool (T, n_outputs) of the final layer; with probes=True returns
        (out, {layer: (T, width) raster}) for the final + probed layers.
        """
        p = self.prog
        if p is None:
            raise RuntimeError("load() a program first")
        events = np.asarray(events, bool)
        if events.ndim != 2 or events.shape[1] != p.n_inputs:
            raise ValueError(f"events must be (T, {p.n_inputs}), got {events.shape}")
        T = events.shape[0]
        out = np.zeros((T, p.n_outputs), bool)
        layers = sorted(set(p.probe_layers) | {len(p.layer_sizes) - 1})
        rasters = {L: np.zeros((T, p.layer_sizes[L]), bool) for L in layers}
        for t in range(T):
            par = self.t & 1
            fl = []
            for k in np.flatnonzero(events[t]):
                for (x, y, a) in p.input_map[k]:
                    fl.append(F.spike(x, y, a, par))
            self.backend.send_flits(fl)
            self.backend.step()
            for f in self.backend.recv_flits():
                d = F.decode(f)
                if d["type"] != F.SPIKE or d["dest_x"] != F.HOST_X:
                    continue
                key = (d["src_x"], d["src_y"], d["neuron"])
                if key not in p.probe_map:
                    raise RuntimeError(f"unexpected host spike from {key}")
                L, idx = p.probe_map[key]
                rasters[L][t, idx] = True
                if key in p.output_map:
                    out[t, p.output_map[key]] = True
            self.t += 1
        return (out, rasters) if probes else out
