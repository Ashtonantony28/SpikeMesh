"""Export a trained SpikeMeshNet to the integer NetIR (compiler/netir.py)."""
from __future__ import annotations

import numpy as np
import torch

from compiler.netir import LayerIR, NetIR
from train.model import SpikeMeshNet


def export_netir(model: SpikeMeshNet, meta: dict | None = None) -> NetIR:
    """Integer NetIR whose reference_run equals model(x, quant=True) exactly.

    Uses the very same LIFLayer.int_params(quant=True) values the forward
    pass uses, so there is no second rounding path to drift.
    """
    layers = []
    with torch.no_grad():
        for L in model.layers:
            w, b, theta, vmin = L.int_params(quant=True)
            wi = w.cpu().numpy().astype(np.int64)
            bi = b.cpu().numpy().astype(np.int64)
            assert np.array_equal(wi, w.cpu().numpy()) and np.array_equal(bi, b.cpu().numpy())
            assert L.wmin <= wi.min() and wi.max() <= L.wmax
            layers.append(LayerIR(weight=wi, bias=bi, theta=int(theta), leak_shift=L.leak_shift,
                                  reset_mode=L.reset_mode, refrac=L.refrac, v_min=int(vmin)))
    m = {"readout": "argmax spike count over T, ties -> lowest index", **model.config()}
    m.update(meta or {})
    return NetIR(model.sizes[0], layers, w_bits=8, meta=m)
