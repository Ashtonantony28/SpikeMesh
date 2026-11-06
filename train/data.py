"""MNIST -> *emulated* N-MNIST event streams (16x16, polarity merged).

This is NOT the real N-MNIST dataset. Real N-MNIST (Orchard et al. 2015)
was recorded by moving an ATIS sensor in three saccades in front of an
LCD showing MNIST digits. The hosts that serve it are blocked from the
cloud container, so here we emulate that recording from the plain MNIST
idx files (downloaded from the ossci S3 mirror) and always call the
result "emulated N-MNIST". `train/data_real.py` loads the real thing via
tonic on a machine that can reach it.

Pipeline, per image (fully deterministic, no RNG):

1. uint8 28x28 -> float [0, 1] -> 16x16 by area averaging
   (16 * 16 = 256 = N_AXONS, so the whole input fits one core's axons).
2. Sensor motion: the image is translated (bilinear, sub-pixel) along a
   closed triangle, like N-MNIST's three saccades:
   (0,0) -> (+d/2,+d) -> (-d/2,+d) -> (0,0), centred on its centroid,
   sampled at T+1 equally spaced points (T/3 steps per saccade).
3. DVS emulation: each pixel keeps a reference level (starts at frame 0).
   At step k the pixel emits an event iff |frame_k - ref| >= thr; the
   reference then moves toward the frame by a whole number of thr steps
   (like a DVS pixel that fires several events). ON and OFF are merged,
   and several events in one step collapse into one (the chip input is
   a bool per axon per timestep).

Output per sample: bool array (T, 256), axon index = row * 16 + col.

Default T = 24: 8 steps per saccade is enough to see every edge move
by ~1 px at least once in each of the three directions, while keeping
simulation cost low (RTL/golden runs scale linearly with T). The two-layer
net adds 1 timestep of latency (SPEC §3), so the output layer sees 23
useful timesteps.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
SIDE = 16
N_IN = SIDE * SIDE
DEFAULT_T = 24
DEFAULT_D = 2.0     # saccade amplitude in 16x16 pixels
DEFAULT_THR = 0.1   # DVS contrast threshold (linear intensity, 0..1)


# ---------------------------------------------------------------- MNIST idx
def download_mnist(root: Path = DATA_DIR / "mnist") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in MNIST_FILES.items():
        p = root / name
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(MNIST_URL + name, tmp)
            tmp.rename(p)
        got = hashlib.md5(p.read_bytes()).hexdigest()
        if got != md5:
            raise RuntimeError(f"{p}: md5 {got} != {md5}; delete it and retry")
    return root


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(split: str, root: Path = DATA_DIR / "mnist"):
    """Returns (images uint8 (N, 28, 28), labels int64 (N,))."""
    root = download_mnist(root)
    pre = "train" if split == "train" else "t10k"
    x = _read_idx(root / f"{pre}-images-idx3-ubyte.gz")
    y = _read_idx(root / f"{pre}-labels-idx1-ubyte.gz").astype(np.int64)
    return x, y


# ------------------------------------------------------- event emulation
def saccade_path(T: int, d: float = DEFAULT_D) -> np.ndarray:
    """(T+1, 2) array of (dx, dy) pixel offsets along the closed triangle."""
    verts = np.array([[0, 0], [d / 2, d], [-d / 2, d], [0, 0]], np.float64)
    verts -= verts[:3].mean(0)
    s = np.linspace(0, 3, T + 1)  # path parameter, 1 unit per saccade
    seg = np.minimum(s.astype(int), 2)
    frac = (s - seg)[:, None]
    return verts[seg] * (1 - frac) + verts[seg + 1] * frac


def images_to_events(images: np.ndarray, T: int = DEFAULT_T, d: float = DEFAULT_D,
                     thr: float = DEFAULT_THR, batch: int = 4096) -> np.ndarray:
    """uint8 (N, 28, 28) -> bool (N, T, 256). Deterministic."""
    path = torch.tensor(saccade_path(T, d), dtype=torch.float64)
    out = np.empty((len(images), T, N_IN), bool)
    for i0 in range(0, len(images), batch):
        x = torch.from_numpy(images[i0:i0 + batch].astype(np.float64) / 255.0)[:, None]
        x = F.interpolate(x, size=(SIDE, SIDE), mode="area")
        n = x.shape[0]
        frames = []
        for dx, dy in path.tolist():
            # translate image content by (+dx, +dy) pixels: sample at (u - dx, v - dy)
            theta = torch.tensor([[1, 0, -2 * dx / SIDE], [0, 1, -2 * dy / SIDE]], dtype=torch.float64)
            grid = F.affine_grid(theta.expand(n, 2, 3), (n, 1, SIDE, SIDE), align_corners=False)
            frames.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                                        align_corners=False)[:, 0].reshape(n, N_IN))
        ref = frames[0].clone()
        ev = torch.zeros(n, T, N_IN, dtype=torch.bool)
        for k in range(T):
            diff = frames[k + 1] - ref
            # small epsilon guards against float noise deciding a tie
            steps = torch.floor(diff.abs() / thr + 1e-9)
            ev[:, k] = steps >= 1
            ref = ref + torch.sign(diff) * steps * thr
        out[i0:i0 + n] = ev.numpy()
    return out


def emulated_nmnist(split: str, T: int = DEFAULT_T, d: float = DEFAULT_D, thr: float = DEFAULT_THR,
                    limit: int | None = None, cache: bool = True):
    """Emulated N-MNIST: (events bool (N, T, 256), labels int64 (N,)).

    Cached under data/emnmnist_{split}_T{T}_d{d}_thr{thr}.npz (bit-packed).
    """
    tag = f"emnmnist_{split}_T{T}_d{d:g}_thr{thr:g}"
    cp = DATA_DIR / f"{tag}.npz"
    if cache and cp.exists():
        z = np.load(cp)
        n = int(z["n"])
        ev = np.unpackbits(z["ev"], count=n * T * N_IN).reshape(n, T, N_IN).astype(bool)
        y = z["y"]
    else:
        x, y = load_mnist(split)
        ev = images_to_events(x, T, d, thr)
        if cache:
            DATA_DIR.mkdir(exist_ok=True)
            np.savez(cp, ev=np.packbits(ev.reshape(-1)), y=y, n=len(y))
    if limit is not None:
        ev, y = ev[:limit], y[:limit]
    return ev, y


def main():
    ap = argparse.ArgumentParser(description="Build the emulated N-MNIST cache and print stats")
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    ap.add_argument("--d", type=float, default=DEFAULT_D)
    ap.add_argument("--thr", type=float, default=DEFAULT_THR)
    a = ap.parse_args()
    for split in ("train", "test"):
        ev, y = emulated_nmnist(split, a.T, a.d, a.thr)
        print(f"{split}: {ev.shape}, events/sample mean {ev.sum((1, 2)).mean():.1f}, "
              f"per step {ev.sum(2).mean():.2f}, empty samples {(ev.sum((1, 2)) == 0).sum()}")


if __name__ == "__main__":
    main()
