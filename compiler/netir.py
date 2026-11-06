"""Network IR: the hand-off format from the trainer to the compiler.

A network is a list of fully-connected spiking layers. Layer 0 reads the
network input (host-injected events); layer i > 0 reads the spikes of layer
i - 1. All values are integers matching docs/SPEC.md §2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class LayerIR:
    weight: np.ndarray            # int, shape (n_in, n_out), within [-2^(W_BITS-1), 2^(W_BITS-1)-1]
    bias: np.ndarray              # int16, shape (n_out,)
    theta: int                    # int16 > 0
    leak_shift: int = 0           # 0..15
    reset_mode: int = 0           # 0 subtract, 1 zero
    refrac: int = 0               # 0..15
    v_min: int = -32768

    @property
    def n_in(self):
        return self.weight.shape[0]

    @property
    def n_out(self):
        return self.weight.shape[1]


@dataclass
class NetIR:
    n_inputs: int
    layers: list[LayerIR] = field(default_factory=list)
    w_bits: int = 8
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path):
        path = Path(path)
        arrays = {}
        hdr = {"n_inputs": self.n_inputs, "w_bits": self.w_bits, "meta": self.meta, "layers": []}
        for i, L in enumerate(self.layers):
            arrays[f"w{i}"] = L.weight.astype(np.int16)
            arrays[f"b{i}"] = L.bias.astype(np.int16)
            d = asdict(L)
            d.pop("weight"); d.pop("bias")
            hdr["layers"].append(d)
        np.savez(path, header=json.dumps(hdr), **arrays)

    @staticmethod
    def load(path: str | Path) -> "NetIR":
        z = np.load(path, allow_pickle=False)
        hdr = json.loads(str(z["header"]))
        layers = [LayerIR(weight=z[f"w{i}"].astype(np.int64), bias=z[f"b{i}"].astype(np.int64), **d)
                  for i, d in enumerate(hdr["layers"])]
        return NetIR(hdr["n_inputs"], layers, hdr["w_bits"], hdr["meta"])


def reference_run(net: NetIR, inputs: np.ndarray) -> list[np.ndarray]:
    """Layer-level integer reference. inputs: bool (T, n_inputs).

    Implements the SPEC §3 latency: layer 0 integrates input t at t; layer i
    integrates layer i-1's spikes from t-1. Returns per-layer bool spike
    rasters of shape (T, n_out). No learning.
    """
    T = inputs.shape[0]
    v = [np.zeros(L.n_out, np.int64) for L in net.layers]
    refc = [np.zeros(L.n_out, np.int64) for L in net.layers]
    prev = [np.zeros(L.n_out, bool) for L in net.layers]
    rasters = [np.zeros((T, L.n_out), bool) for L in net.layers]
    for t in range(T):
        cur = []
        for i, L in enumerate(net.layers):
            src = inputs[t] if i == 0 else prev[i - 1]
            acc = src.astype(np.int64) @ L.weight
            r = refc[i] > 0
            leak = (v[i] >> L.leak_shift) if L.leak_shift else 0
            v2 = np.clip(v[i] - leak + np.clip(acc, -32768, 32767) + L.bias, -32768, 32767)
            s = (~r) & (v2 >= L.theta)
            nv = np.where(s, (v2 - L.theta) if L.reset_mode == 0 else 0, np.maximum(v2, L.v_min))
            v[i] = np.where(r, v[i], nv)
            refc[i] = np.where(r, refc[i] - 1, np.where(s, L.refrac, 0))
            rasters[i][t] = s
            cur.append(s)
        prev = cur
    return rasters
