"""Real N-MNIST / DVS128 Gesture via tonic, binned like train/data.py.

UNTESTED IN THE CLOUD: the dataset hosts (figshare / mendeley) are blocked
from the build container, so only `events_to_raster` is covered by tests
(on synthetic events). Run this on a machine with internet access:

    pip install tonic
    python -m train.data_real --dataset nmnist --T 24      # writes the same cache format
    python -m train.train --real nmnist ...                 # (see train/README.md)

Binning: x -> x * 16 // W, y -> y * 16 // H (W, H = sensor size), so the
input is 16 x 16 = 256 axons (one core). ON/OFF polarity merged. Time is
split into T equal bins over the sample's own [t_min, t_max]; an axon is
active in a bin if any event fell there. DVS Gesture has 11 classes, so
use an 11-neuron output layer for it.
"""
from __future__ import annotations

import argparse

import numpy as np

from train.data import DATA_DIR, DEFAULT_T, N_IN, SIDE

SENSORS = {"nmnist": (34, 34), "dvsgesture": (128, 128)}


def events_to_raster(ev: np.ndarray, sensor_wh: tuple[int, int], T: int = DEFAULT_T) -> np.ndarray:
    """tonic structured events (fields x, y, t[, p]) -> bool (T, 256)."""
    W, H = sensor_wh
    out = np.zeros((T, N_IN), bool)
    if len(ev) == 0:
        return out
    x = np.asarray(ev["x"], np.int64) * SIDE // W
    y = np.asarray(ev["y"], np.int64) * SIDE // H
    t = np.asarray(ev["t"], np.int64)
    t0, span = t.min(), t.max() - t.min() + 1
    tb = (t - t0) * T // span
    out[tb, y * SIDE + x] = True
    return out


def load_real(name: str, train: bool, T: int = DEFAULT_T, limit: int | None = None):
    import tonic  # only needed here

    root = str(DATA_DIR / "tonic")
    if name == "nmnist":
        ds = tonic.datasets.NMNIST(save_to=root, train=train)
    elif name == "dvsgesture":
        ds = tonic.datasets.DVSGesture(save_to=root, train=train)
    else:
        raise ValueError(name)
    n = len(ds) if limit is None else min(limit, len(ds))
    xs = np.zeros((n, T, N_IN), bool)
    ys = np.zeros(n, np.int64)
    for i in range(n):
        ev, label = ds[i]
        xs[i] = events_to_raster(ev, SENSORS[name], T)
        ys[i] = label
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SENSORS), default="nmnist")
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    a = ap.parse_args()
    for split, train in (("train", True), ("test", False)):
        x, y = load_real(a.dataset, train, a.T)
        p = DATA_DIR / f"real_{a.dataset}_{split}_T{a.T}.npz"
        np.savez(p, ev=np.packbits(x.reshape(-1)), y=y, n=len(y))
        print(p, x.shape, "events/sample", x.sum((1, 2)).mean())


def load_cached(name: str, split: str, T: int = DEFAULT_T):
    """Read the cache written by `main` (same layout as the emulated cache)."""
    z = np.load(DATA_DIR / f"real_{name}_{split}_T{T}.npz")
    n = int(z["n"])
    return np.unpackbits(z["ev"], count=n * T * N_IN).reshape(n, T, N_IN).astype(bool), z["y"]


if __name__ == "__main__":
    main()
