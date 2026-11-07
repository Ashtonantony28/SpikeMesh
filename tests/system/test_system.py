"""SG5: the full RTL chip (spikemesh_top + rtl/noc + the real rtl/core) is
bit-exact with the golden model end to end, driven through the host API.

Run: python3 -m pytest -q tests/system           (about 2-3 minutes)
     python3 -m pytest -q tests/system -m slow   (all 256 fixture samples)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from sysutil import (FIXTURE, HAVE_ICARUS, HAVE_VERILATOR, HERE, assert_same, fixture_inputs,
                     rand_net)
from compiler import flits as F
from compiler.compile import compile_net
from compiler.netir import NetIR, reference_run
from host.api import GoldenBackend, SpikeMeshHost
from host.rtl_backend import RtlBackend

needs_vl = pytest.mark.skipif(not HAVE_VERILATOR, reason="verilator not installed")
needs_iv = pytest.mark.skipif(not HAVE_ICARUS, reason="icarus not installed")

SMALL = dict(n=32, a=32, lanes=4, fanout=128)


# --------------------------------------------------------------------------
# (i) random NetIRs on small configs: RTL == GoldenBackend == reference_run
# --------------------------------------------------------------------------
def _random_case(seed, sparse_ok=True):
    rng = np.random.default_rng(5000 + seed)
    mx, my = [(2, 2), (3, 3)][seed % 2]
    n_cores = mx * my
    depth = 2 if n_cores == 4 else int(rng.integers(2, 4))
    sizes = [int(rng.integers(8, 33))]
    for _ in range(depth - 1):                     # hidden widths feed N_AXONS=32 axons
        sizes.append(int(rng.integers(6, 33)))
    sizes.append(int(rng.integers(3, 30 if n_cores == 4 else 60)))   # output may span 2 cores
    w_bits = 4 if seed % 4 == 3 else 8
    net = rand_net(rng, sizes, w_bits)
    probes = tuple(range(len(sizes) - 2))
    prog = compile_net(net, mx, my, SMALL["n"], SMALL["a"], w_bits=w_bits, fanout_depth=SMALL["fanout"],
                       probe_layers=probes, dense_weights=not (sparse_ok and seed % 3 == 0))
    return rng, net, prog


def _compare_samples(prog, net, be, rng, n_samples=2, T=30):
    """Run n_samples (state reset in between) on RTL and golden; compare."""
    hr = SpikeMeshHost(be)
    hg = SpikeMeshHost(GoldenBackend.for_program(prog))
    hr.load(prog)
    hg.load(prog)
    total = 0
    for s in range(n_samples):
        if s:
            hr.reset_state()
            hg.reset_state()
        x = rng.random((T, net.n_inputs)) < rng.uniform(0.1, 0.5)
        out_r, ras_r = hr.run(x, probes=True)
        out_g, ras_g = hg.run(x, probes=True)
        ref = reference_run(net, x)
        assert_same(out_r, ref[-1], f"sample {s} output (RTL vs reference_run)")
        assert_same(out_g, ref[-1], f"sample {s} output (golden vs reference_run)")
        assert set(ras_r) == set(ras_g)
        for L in ras_r:
            assert_same(ras_r[L], ras_g[L], f"sample {s} layer {L} raster (RTL vs golden)")
            assert_same(ras_r[L], ref[L], f"sample {s} layer {L} raster (RTL vs reference_run)")
            total += int(ras_r[L].sum())
    assert be.err == 0 and be.drop_cnt == 0
    return total


@needs_iv
@pytest.mark.parametrize("seed", [0, 1, 3])
def test_random_net_icarus(seed):
    # dense weights: the weight SRAM is not reset, so in 4-state Icarus a
    # sparse (dense_weights=False) image leaves X in unwritten synapses
    rng, net, prog = _random_case(seed, sparse_ok=False)
    with RtlBackend.for_program(prog, lanes=SMALL["lanes"], sim="icarus", bp=20 * (seed % 2)) as be:
        assert _compare_samples(prog, net, be, rng, n_samples=2, T=20) > 0


@needs_vl
@pytest.mark.parametrize("seed", range(4, 16))
def test_random_net_verilator(seed):
    rng, net, prog = _random_case(seed)
    with RtlBackend.for_program(prog, lanes=SMALL["lanes"], sim="verilator", bp=[0, 30, 70][seed % 3],
                                seed=seed) as be:
        assert _compare_samples(prog, net, be, rng, n_samples=3, T=40) > 0


@needs_vl
def test_backend_readback_and_cycles():
    """READ_REQ round trip through the backend: every configured weight of a
    used core reads back as golden; recv_flits syncs implicitly."""
    rng, net, prog = _random_case(4)
    s = prog.slices[0]
    addrs = [F.addr_weight(a, n, prog.n_neurons) for a in range(net.n_inputs) for n in range(s.n)]
    reqs = [F.read_req(s.x, s.y, a) for a in addrs]
    with RtlBackend.for_program(prog, lanes=SMALL["lanes"], sim="verilator") as be:
        g = GoldenBackend.for_program(prog)
        for b in (be, g):
            b.send_flits(prog.flits)
            b.send_flits(reqs)
        got, exp = be.recv_flits(), g.recv_flits()
        assert len(exp) == len(addrs)
        assert got == exp
        assert be.sync_cycles and be.sync_cycles[-1] > len(reqs)


# --------------------------------------------------------------------------
# (iii) on-chip learning: STDP core in the mesh, weights read back via READ_REQ
# --------------------------------------------------------------------------
LEARN_REGS = {8: 1, 9: 9, 10: 7, 11: 4, 12: 2, 13: 2, 14: 48, 15: 40, 16: 0x2B1D, 17: (-40) & 0xFFFF, 18: 45}


@needs_vl
@pytest.mark.parametrize("sim", ["verilator"])
def test_stdp_learning_in_mesh(sim):
    rng = np.random.default_rng(77)
    net = rand_net(rng, [24, 32, 8])
    prog = compile_net(net, 2, 2, SMALL["n"], SMALL["a"], fanout_depth=SMALL["fanout"], probe_layers=(0,))
    s0 = prog.slices[0]                                    # layer-0 core learns
    mask = 1 << (s0.y * prog.mesh_x + s0.x)
    learn_fl = [F.cfg_write(s0.x, s0.y, F.addr_reg(r), v) for r, v in LEARN_REGS.items()]
    w_addrs = [F.addr_weight(a, n, prog.n_neurons) for a in range(net.n_inputs) for n in range(s0.n)]
    reads = [F.read_req(s0.x, s0.y, a) for a in w_addrs]
    T = 16
    x = rng.random((T, net.n_inputs)) < 0.35
    results = {}
    with RtlBackend.for_program(prog, lanes=SMALL["lanes"], sim=sim, learn_mask=mask, learn_mode=1) as be:
        for name, b in (("rtl", be), ("golden", GoldenBackend.for_program(prog))):
            h = SpikeMeshHost(b)
            h.load(prog)
            b.send_flits(learn_fl)
            b.send_flits(reads)
            w0 = b.recv_flits()
            out, ras = h.run(x, probes=True)
            b.send_flits(reads)
            w1 = b.recv_flits()
            results[name] = (w0, w1, out, ras)
        assert be.err == 0
    (w0r, w1r, outr, rasr), (w0g, w1g, outg, rasg) = results["rtl"], results["golden"]
    assert w0r == w0g and len(w0r) == len(w_addrs)
    for L in rasg:
        assert_same(rasr[L], rasg[L], f"layer {L} raster during learning")
    assert_same(outr, outg, "output during learning")
    dec = lambda fl: np.array([F.decode(f)["data"] for f in fl])  # noqa: E731
    changed = int((dec(w1g) != dec(w0g)).sum())
    assert changed > 50, f"learning barely changed the weights ({changed})"
    diff = np.flatnonzero(dec(w1r) != dec(w1g))
    assert diff.size == 0, f"{diff.size} weights differ after learning, first addr idx {diff[:5].tolist()}"
    assert rasg[0].sum() > 0


# --------------------------------------------------------------------------
# (ii) + (iv) trained 256-256-10 network, emulated N-MNIST, fast path
# --------------------------------------------------------------------------
def _trained_run(n_samples: int, write_perf: bool):
    import torch

    from train.model import SpikeMeshNet

    net = NetIR.load(FIXTURE)
    x, y, _ = fixture_inputs()
    x, y = x[:n_samples], y[:n_samples]
    T = x.shape[1]
    # bit-exactness with the hidden layer probed
    prog = compile_net(net, 2, 2, probe_layers=(0,))
    m = SpikeMeshNet.from_netir(net)
    m.eval()
    with torch.no_grad():
        tr, _ = m(torch.as_tensor(x, dtype=torch.float32), quant=True)
    torch_pred = SpikeMeshNet.predict(tr[-1]).numpy()
    rtl_out = np.zeros((n_samples, T, net.layers[-1].n_out), bool)
    n_golden = min(n_samples, int(os.environ.get("SPIKEMESH_SYS_GOLDEN", "8")))
    with RtlBackend.for_program(prog, lanes=8, sim="verilator") as be:
        h = SpikeMeshHost(be)
        h.load(prog)
        hg = SpikeMeshHost(GoldenBackend.for_program(prog))
        hg.load(prog)
        for i in range(n_samples):
            if i:
                h.reset_state()
            out, ras = h.run(x[i], probes=True)
            ref = reference_run(net, x[i])
            assert_same(ras[0], ref[0], f"sample {i} hidden raster (RTL vs reference_run)")
            assert_same(out, ref[-1], f"sample {i} output raster (RTL vs reference_run)")
            assert_same(out, tr[-1][i].numpy(), f"sample {i} output raster (RTL vs integer torch)")
            if i < n_golden:                       # golden chip through flits (slower)
                if i:
                    hg.reset_state()
                og, rg = hg.run(x[i], probes=True)
                assert_same(out, og, f"sample {i} output (RTL vs GoldenBackend)")
                assert_same(ras[0], rg[0], f"sample {i} hidden (RTL vs GoldenBackend)")
            rtl_out[i] = out
        assert be.err == 0
    rtl_pred = rtl_out.sum(1).argmax(1)
    assert (rtl_pred == torch_pred).all()
    acc_rtl, acc_torch = float((rtl_pred == y).mean()), float((torch_pred == y).mean())
    assert acc_rtl == acc_torch
    if write_perf:
        _perf(net, x, y, rtl_pred, acc_rtl, acc_torch)
    return acc_rtl


def _perf(net, x, y, rtl_pred, acc_rtl, acc_torch):
    """Deployment config (no probes): cycles per timestep / sample, spikes, SOPs."""
    prog = compile_net(net, 2, 2)
    S, T = x.shape[0], x.shape[1]
    rows = []
    with RtlBackend.for_program(prog, lanes=8, sim="verilator") as be:
        h = SpikeMeshHost(be)
        h.load(prog)
        load_cycles = be.sync()
        for i in range(S):
            k0, s0 = len(be.step_cycles), len(be.sync_cycles)
            if i:
                h.reset_state()
                be.sync()
            reset_c = sum(be.step_cycles[k0:]) + sum(be.sync_cycles[s0:])
            k1 = len(be.step_cycles)
            h.run(x[i])
            step_c = be.step_cycles[k1:]
            ref = reference_run(net, x[i])
            n_in, n_hid, n_out = int(x[i].sum()), int(ref[0].sum()), int(ref[1].sum())
            n_l0, n_l1 = net.layers[0].n_out, net.layers[1].n_out
            rows.append(dict(sample=i, label=int(y[i]), pred=int(rtl_pred[i]), input_events=n_in,
                             hidden_spikes=n_hid, output_spikes=n_out, run_cycles=int(sum(step_c)),
                             reset_cycles=int(reset_c), max_step_cycles=int(max(step_c)),
                             min_step_cycles=int(min(step_c)), sops=n_in * n_l0 + n_hid * n_l1))
    run_c = np.array([r["run_cycles"] for r in rows], float)
    ev = np.array([r["input_events"] for r in rows], float)
    sops = np.array([r["sops"] for r in rows], float)
    tot = run_c + np.array([r["reset_cycles"] for r in rows], float)
    slope, icpt = np.polyfit(ev, run_c, 1)
    r2 = float(np.corrcoef(ev, run_c)[0, 1] ** 2)
    allspk = ev + np.array([r["hidden_spikes"] for r in rows], float)
    slope2, icpt2 = np.polyfit(allspk, run_c, 1)
    r2b = float(np.corrcoef(allspk, run_c)[0, 1] ** 2)
    f = 100e6
    perf = {
        "config": {"mesh": "2x2", "cores_used": len(prog.slices), "N_NEURONS": 256, "N_AXONS": 256,
                   "LANES": 8, "W_BITS": 8, "net": [256, 256, 10], "T": T, "samples": S,
                   "dataset": "emulated N-MNIST (MNIST saccades), tests/train/fixtures/test_inputs.npz",
                   "clock_for_projection_hz": f, "simulator": "Verilator 5.020 (tests/system/tb_sys.sv)"},
        "accuracy": {"rtl": acc_rtl, "integer_torch": acc_torch, "rtl_equals_torch_predictions": True},
        "load_cycles": int(load_cycles), "cfg_flits": len(prog.flits),
        "cycles_per_timestep_mean": float(run_c.sum() / (S * T)),
        "cycles_per_timestep_max": int(max(r["max_step_cycles"] for r in rows)),
        "cycles_per_timestep_min": int(min(r["min_step_cycles"] for r in rows)),
        "cycles_per_sample_mean": float(run_c.mean()),
        "cycles_per_sample_with_state_reset_mean": float(tot.mean()),
        "state_reset_cycles_mean": float(tot.mean() - run_c.mean()),
        "input_events_per_sample_mean": float(ev.mean()),
        "hidden_spikes_per_sample_mean": float(np.mean([r["hidden_spikes"] for r in rows])),
        "output_spikes_per_sample_mean": float(np.mean([r["output_spikes"] for r in rows])),
        "sops_per_sample_mean": float(sops.mean()),
        "sops_per_cycle": float(sops.sum() / run_c.sum()),
        "projected_at_100MHz": {
            "samples_per_s": float(f / run_c.mean()),
            "samples_per_s_with_state_reset": float(f / tot.mean()),
            "timesteps_per_s": float(f * S * T / run_c.sum()),
            "gsops_per_s": float(sops.sum() / run_c.sum() * f / 1e9),
            "latency_per_sample_us": float(run_c.mean() / f * 1e6),
        },
        "cycles_vs_input_events_fit": {"cycles_per_event": float(slope), "intercept_cycles": float(icpt),
                                       "r2": r2},
        "cycles_vs_all_spikes_fit": {"cycles_per_spike": float(slope2), "intercept_cycles": float(icpt2),
                                     "r2": r2b, "spikes": "input events + hidden-layer spikes"},
        "definitions": {
            "run_cycles": "sum over the T STEPs of cycles from the step pulse until drained and host output empty "
                          "(includes injecting that step's input events)",
            "sops": "synaptic operations = input_events * 256 (layer-0 fan-out) + hidden_spikes * 10",
            "reset_cycles": "SpikeMeshHost.reset_state(): 2 silent STEPs + CFG rewrite of v/refc/theta_a/pointers",
        },
    }
    (HERE / "perf.json").write_text(json.dumps(perf, indent=2) + "\n")
    keys = list(rows[0])
    (HERE / "cycles_vs_spikes.csv").write_text(
        ",".join(keys) + "\n" + "".join(",".join(str(r[k]) for k in keys) + "\n" for r in rows))
    _plot(ev, run_c, slope, icpt, r2)
    return perf


def _plot(ev, cyc, slope, icpt, r2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:            # CSV is the fallback artifact
        return
    ink, ink2, grid, c1 = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(ev, cyc / 1e3, s=14, color=c1, alpha=0.75, linewidths=0)
    xs = np.linspace(0, ev.max() * 1.05, 50)
    ax.plot(xs, (slope * xs + icpt) / 1e3, color=ink2, lw=1.2, ls="--")
    ax.text(0.02, 0.96, f"fit: {slope:.1f} cycles/event + {icpt / 1e3:.1f}k   (R² = {r2:.3f})",
            transform=ax.transAxes, va="top", color=ink2, fontsize=8)
    ax.set_xlabel("input events per sample", color=ink, fontsize=9)
    ax.set_ylabel("cycles per sample (thousands)", color=ink, fontsize=9)
    ax.set_title("SpikeMesh RTL: cycles scale with input events (T = 24)", color=ink, fontsize=10, loc="left")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, color=grid, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "cycles_vs_spikes.png")
    plt.close(fig)


@needs_vl
@pytest.mark.skipif(not FIXTURE.exists(), reason="trained fixture missing")
def test_trained_net_fast_path():
    """>= 20 emulated test samples (default 64), rasters bit-exact, accuracy == integer torch;
    writes the perf artifacts only if they do not exist yet (the committed ones
    come from the 256-sample slow run)."""
    n = int(os.environ.get("SPIKEMESH_SYS_SAMPLES", "64"))
    assert n >= 20
    _trained_run(n, write_perf=not (HERE / "perf.json").exists())


@needs_vl
@pytest.mark.slow
@pytest.mark.skipif(not FIXTURE.exists(), reason="trained fixture missing")
def test_trained_net_all_samples():
    """All 256 fixture samples (marker: slow)."""
    t0 = time.time()
    acc = _trained_run(256, write_perf=True)
    print(f"256 samples, accuracy {acc:.4f}, {time.time() - t0:.0f} s")
