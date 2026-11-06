"""Copy a trained NetIR + 256 emulated N-MNIST test samples into tests/train/fixtures.

    python -m train.make_fixture runs/base/net.npz
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from compiler.netir import NetIR, reference_run
from train.data import emulated_nmnist

FIX = Path(__file__).resolve().parent.parent / "tests" / "train" / "fixtures"
N = 256


def main(src: str):
    FIX.mkdir(parents=True, exist_ok=True)
    net = NetIR.load(src)
    kw = {k: net.meta[k] for k in ("T", "d", "thr") if k in net.meta}
    x, y = emulated_nmnist("test", **kw)
    x, y = x[:N], y[:N]
    counts = np.stack([reference_run(net, xi)[-1].sum(0) for xi in x])
    acc = float((counts.argmax(1) == y).mean())
    shutil.copy(src, FIX / "trained_net.npz")
    np.savez_compressed(FIX / "test_inputs.npz", x=np.packbits(x.reshape(-1)), shape=np.array(x.shape),
                        y=y, min_acc=acc)
    print(f"fixture: {N} samples, reference_run accuracy {acc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
