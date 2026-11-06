"""Bit-exact, quantization-aware LIF network for SpikeMesh (docs/SPEC.md §2, §3).

Every layer keeps float "real" parameters (weight W, bias B, threshold
theta_real = 1.0) plus a per-layer scale s. The membrane is simulated in
*integer units* (real value / s):

    w = W / s,   b = B / s,   theta = theta_real / s,   v_min = v_min_real / s

* ``quant=False`` (float model): those values are used as-is and the leak is
  the exact ``v / 2**k``. This is the "float" accuracy reported.
* ``quant=True`` (QAT and eval): ``w = round(clamp(W/s, -128, 127))``,
  ``b = round(clamp(B/s, int16))``, ``theta = round(theta_real/s)`` clamped
  to [1, 32767], leak ``floor(v / 2**k)`` (== ``v >>> k``), all with
  straight-through gradients. With ``s = max|W| / 127`` (per layer, detached).

In quant mode every tensor in the forward pass holds an exact integer
(all magnitudes stay < 2**24 so float32 is exact: |acc| <= 256*128,
|v2| <= 32767 + 32767 + 32768 + 32768), and the update below is SPEC §2
line for line, so the eval forward is bit-exact with
``compiler.netir.reference_run`` on the exported NetIR (tests/train).

Timing (SPEC §3): layer 0 integrates input t at t; layer i > 0 integrates
layer i-1's spikes from t-1 (one timestep of latency per layer).

Readout (hardware decision): the class is the output neuron with the most
spikes over all T timesteps; ties (including all-zero) go to the lowest
index (``torch.argmax`` returns the first maximal index).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

I16_MIN, I16_MAX = -32768, 32767


class SpikeFn(torch.autograd.Function):
    """Heaviside(x >= 0) with fast-sigmoid surrogate gradient 1 / (1 + k|x|)^2."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + ctx.slope * x.abs()) ** 2, None


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def floor_ste(x):
    return x + (torch.floor(x) - x).detach()


class LIFLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, leak_shift: int = 0, reset_mode: int = 0,
                 refrac: int = 0, v_min: float | None = None, w_bits: int = 8, init_std: float = 0.2):
        super().__init__()
        assert n_in <= 256, "fan-in > N_AXONS (256) cannot be mapped to one core"
        assert 0 <= leak_shift <= 15 and reset_mode in (0, 1) and 0 <= refrac <= 15
        self.n_in, self.n_out = n_in, n_out
        self.leak_shift, self.reset_mode, self.refrac = leak_shift, reset_mode, refrac
        self.wmax = (1 << (w_bits - 1)) - 1
        self.wmin = -(1 << (w_bits - 1))
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * init_std)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("theta_real", torch.tensor(1.0))
        # v_min in real units (like theta_real); NaN = no clamp (-32768)
        self.register_buffer("v_min_real", torch.tensor(float("nan") if v_min is None else v_min))
        # > 0: fixed scale (used when loading an integer NetIR); 0: max-abs scaling
        self.register_buffer("fixed_scale", torch.tensor(0.0))

    def scale(self) -> torch.Tensor:
        if self.fixed_scale > 0:
            return self.fixed_scale
        s = self.weight.detach().abs().max() / self.wmax
        # keep theta = theta_real / s inside int16
        return torch.maximum(s, self.theta_real / I16_MAX)

    def int_params(self, quant: bool):
        """(w, b, theta, v_min) in integer units; exact integers when quant."""
        s = self.scale()
        w, b = self.weight / s, self.bias / s
        theta = self.theta_real / s
        vmin = self.v_min_real / s if not torch.isnan(self.v_min_real) else torch.tensor(float(I16_MIN))
        if quant:
            w = round_ste(torch.clamp(w, self.wmin, self.wmax))
            b = round_ste(torch.clamp(b, I16_MIN, I16_MAX))
            theta = torch.clamp(torch.round(theta), 1, I16_MAX)
            vmin = torch.clamp(torch.round(vmin), I16_MIN, I16_MAX)
        return w, b, theta.detach(), vmin.detach()

    def init_state(self, batch: int, device=None):
        z = torch.zeros(batch, self.n_out, device=device)
        return z, z.clone()  # v, refc

    def step(self, src, v, refc, params, quant: bool, slope: float):
        """One SPEC §2 update. src: (B, n_in) 0/1. Returns spike, v, refc, v2."""
        w, b, theta, vmin = params
        acc = src @ w
        ref = refc > 0
        if self.leak_shift:
            q = v / (1 << self.leak_shift)
            leak = floor_ste(q) if quant else q
        else:
            leak = 0.0
        v2 = torch.clamp(v - leak + torch.clamp(acc, I16_MIN, I16_MAX) + b, I16_MIN, I16_MAX)
        spk = SpikeFn.apply((v2 - theta) / theta, slope) * (~ref).to(v2.dtype)
        fired = spk.detach() > 0
        reset_v = (v2 - theta) if self.reset_mode == 0 else torch.zeros_like(v2)
        reset_v = v2 - (v2 - reset_v).detach()  # reset path detached (snnTorch convention)
        v_new = torch.where(fired, reset_v, torch.maximum(v2, vmin))
        v = torch.where(ref, v, v_new)
        refc = torch.where(ref, refc - 1, torch.where(fired, torch.full_like(refc, self.refrac),
                                                      torch.zeros_like(refc)))
        return spk, v, refc, v2


class SpikeMeshNet(nn.Module):
    def __init__(self, sizes=(256, 256, 10), leak_shift=(0, 0), reset_mode=(0, 0), refrac=(0, 0),
                 v_min=(None, None), init_std=None, slope: float = 5.0):
        super().__init__()
        n = len(sizes) - 1
        init_std = init_std or [0.3] * n
        self.sizes = tuple(sizes)
        self.slope = slope
        self.layers = nn.ModuleList(
            LIFLayer(sizes[i], sizes[i + 1], leak_shift[i], reset_mode[i], refrac[i], v_min[i],
                     init_std=init_std[i]) for i in range(n))

    def forward(self, x: torch.Tensor, quant: bool = True):
        """x: (B, T, n_in) 0/1 float. Returns (rasters, v2_out).

        rasters: list of (B, T, n_out) spike tensors, one per layer.
        v2_out: (B, T, n_classes) pre-reset membrane of the last layer, in
        units of its theta (for optional membrane-based training losses).
        """
        B, T, _ = x.shape
        params = [L.int_params(quant) for L in self.layers]
        state = [L.init_state(B, x.device) for L in self.layers]
        prev = [torch.zeros(B, L.n_out, device=x.device) for L in self.layers]
        rasters = [[] for _ in self.layers]
        v2s = []
        for t in range(T):
            cur = []
            for i, L in enumerate(self.layers):
                src = x[:, t] if i == 0 else prev[i - 1]
                spk, v, refc, v2 = L.step(src, *state[i], params[i], quant, self.slope)
                state[i] = (v, refc)
                rasters[i].append(spk)
                cur.append(spk)
            v2s.append(v2 / params[-1][2])
            prev = cur
        return [torch.stack(r, 1) for r in rasters], torch.stack(v2s, 1)

    @staticmethod
    def predict(out_raster: torch.Tensor) -> torch.Tensor:
        """Hardware decision: argmax of spike counts, ties -> lowest index."""
        return out_raster.sum(1).argmax(1)

    # ------------------------------------------------------------ NetIR
    def config(self) -> dict:
        return dict(sizes=list(self.sizes),
                    leak_shift=[L.leak_shift for L in self.layers],
                    reset_mode=[L.reset_mode for L in self.layers],
                    refrac=[L.refrac for L in self.layers],
                    slope=self.slope)

    @classmethod
    def from_netir(cls, net) -> "SpikeMeshNet":
        """Build a model whose quant forward reproduces an integer NetIR exactly."""
        sizes = [net.n_inputs] + [L.n_out for L in net.layers]
        m = cls(sizes, [L.leak_shift for L in net.layers], [L.reset_mode for L in net.layers],
                [L.refrac for L in net.layers], [None] * len(net.layers))
        with torch.no_grad():
            for L, ir in zip(m.layers, net.layers):
                L.weight.copy_(torch.as_tensor(np.asarray(ir.weight), dtype=torch.float32))
                L.bias.copy_(torch.as_tensor(np.asarray(ir.bias), dtype=torch.float32))
                L.theta_real.fill_(float(ir.theta))
                L.v_min_real.fill_(float(ir.v_min))
                L.fixed_scale.fill_(1.0)
        return m


def count_ce_loss(out_raster: torch.Tensor, v2_out: torch.Tensor | None, y: torch.Tensor,
                  count_gain: float = 1.0, mem_weight: float = 0.0) -> torch.Tensor:
    """Cross-entropy on spike counts (the hardware readout), optional membrane term."""
    loss = nn.functional.cross_entropy(out_raster.sum(1) * count_gain, y)
    if mem_weight and v2_out is not None:
        loss = loss + mem_weight * nn.functional.cross_entropy(v2_out.mean(1) * 4.0, y)
    return loss

