"""Shared helpers for the SG5 system tests (full RTL chip vs golden)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.netir import LayerIR, NetIR  # noqa: E402

FIXTURE = ROOT / "tests" / "train" / "fixtures" / "trained_net.npz"
FIXTURE_INPUTS = ROOT / "tests" / "train" / "fixtures" / "test_inputs.npz"
HAVE_VERILATOR = shutil.which("verilator") is not None
HAVE_ICARUS = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def fixture_inputs():
    """Emulated N-MNIST test samples: bool (S, T, 256), labels (S,), min_acc."""
    z = np.load(FIXTURE_INPUTS)
    shape = tuple(int(v) for v in z["shape"])
    x = np.unpackbits(z["x"], count=int(np.prod(shape))).reshape(shape).astype(bool)
    return x, z["y"], float(z["min_acc"])


def rand_net(rng, sizes, w_bits=8):
    """Random integer NetIR (same recipe as tests/compiler)."""
    lo, hi = (-20, 40) if w_bits == 8 else (-8, 7)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w = rng.integers(lo, hi + 1, (n_in, n_out))
        w[rng.random(w.shape) < 0.2] = 0
        layers.append(LayerIR(weight=w, bias=rng.integers(-4, 5, n_out),
                              theta=int(rng.integers(20, 80)) if w_bits == 8 else int(rng.integers(6, 20)),
                              leak_shift=int(rng.integers(0, 4)), reset_mode=int(rng.integers(0, 2)),
                              refrac=int(rng.integers(0, 3)), v_min=int(rng.choice([-32768, -50, 0]))))
    return NetIR(sizes[0], layers, w_bits=w_bits)


def assert_same(a: np.ndarray, b: np.ndarray, what: str):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    bad = np.argwhere(a != b)
    assert bad.size == 0, f"{what}: {len(bad)} mismatching (t, neuron) cells, first {bad[:5].tolist()}"
