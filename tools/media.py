"""Render the README demo animation: the trained 256-256-10 SNN classifying emulated N-MNIST digits
on the golden SpikeMesh chip model, driven purely through flits by the host API.

    python3 tools/media.py            # -> docs/media/snn-demo.{gif,mp4}, docs/media/snn-demo.png

The golden chip is the oracle the Verilator RTL chip matches spike for spike (tests/system). Each
frame asserts that the golden hidden and output rasters equal compiler.netir.reference_run, so the
animation cannot drift from the bit-exact model. Needs matplotlib and ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "system"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import sysutil  # noqa: E402
from compiler.compile import compile_net  # noqa: E402
from compiler.netir import NetIR, reference_run  # noqa: E402
from host.api import GoldenBackend, SpikeMeshHost  # noqa: E402

BG, PANEL, GRID, TXT, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
IN_C, HID_C, OUT_C, OK_C, BAD_C = "#58a6ff", "#d2a8ff", "#ffa657", "#3fb950", "#f85149"
OUT = ROOT / "docs" / "media"


def run_sample(net, prog, x):
    """Run one sample on the golden chip, recording per-timestep packet counts on each mesh path."""
    be = GoldenBackend.for_program(prog)
    host = SpikeMeshHost(be)
    host.load(prog)
    chip = be.chip
    traffic = []
    orig = chip.step

    def step():
        n_in = sum(1 for (cx, cy, _, _) in chip.inflight if (cx, cy) == (0, 0))
        hb = orig()
        n_mid = sum(1 for (cx, cy, _, _) in chip.inflight if (cx, cy) == (1, 0))
        traffic.append((n_in, n_mid, len(hb)))
        return hb

    chip.step = step
    out, ras = host.run(x, probes=True)
    ref = reference_run(net, x)
    assert np.array_equal(ras[0], ref[0]) and np.array_equal(out, ref[-1]), "golden chip != reference_run"
    return ras[0], out, np.array(traffic)


def frame(fig, x, t, hid, out, traffic, label, k_sample, n_samples, done):
    T = x.shape[0]
    fig.clf()
    fig.text(0.035, 0.935, "SpikeMesh: a trained spiking network classifying digits on the neuromorphic chip",
             fontsize=17, weight="bold", color=TXT)
    fig.text(0.035, 0.895, "GOLDEN-MODEL SIMULATION of the chip, driven through flits by the host API (spike-for-spike "
             "identical to the Verilator RTL)  ·  emulated N-MNIST events  ·  256-256-10 LIF network, INT8 weights",
             fontsize=9.3, color=DIM)

    # input events (current step bright, recent steps fading)
    ax = fig.add_axes([0.035, 0.40, 0.25, 0.44])
    img = np.zeros((16, 16))
    for d in range(4):
        if t - d >= 0:
            img = np.maximum(img, x[t - d].reshape(16, 16) * (1.0 - 0.28 * d))
    ax.imshow(img, cmap=LinearSegmentedColormap.from_list("i", [PANEL, IN_C]), vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(f"input events, timestep {t + 1}/{T}  ({int(x[t].sum())} events)", color=TXT, fontsize=10.5, loc="left")

    # mesh: 2x2 cores, host port west of (0,0)
    ax = fig.add_axes([0.035, 0.05, 0.25, 0.30])
    ax.set_xlim(-1.25, 2.0); ax.set_ylim(-2.0, 0.65); ax.axis("off")
    ax.set_facecolor(BG)
    n_in, n_mid, n_out = traffic[t]
    cores = {(0, 0): ("L0: 256 hidden", HID_C, hid[t].sum()), (1, 0): ("L1: 10 output", OUT_C, out[t].sum()),
             (0, 1): ("idle", GRID, 0), (1, 1): ("idle", GRID, 0)}
    for (cx, cy), (name, col, act) in cores.items():
        px, py = cx * 1.35, -cy * 1.1
        glow = min(1.0, act / (12 if cx == 0 else 1)) if col != GRID else 0
        ax.add_patch(FancyBboxPatch((px - 0.42, py - 0.32), 0.84, 0.64, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=PANEL, ec=col, lw=1.4 + 2.5 * glow, alpha=1.0))
        ax.text(px, py + 0.1, f"core ({cx},{cy})", ha="center", color=TXT if col != GRID else DIM, fontsize=9, weight="bold")
        ax.text(px, py - 0.12, name, ha="center", color=col if col != GRID else DIM, fontsize=7.4)
    for (a, b, n, col, lbl, off) in [((-1.2, 0.08), (-0.45, 0.08), n_in, IN_C, "in", 0.17),
                                     ((0.45, 0.0), (0.9, 0.0), n_mid, HID_C, "", 0.13),
                                     ((-0.45, -0.1), (-1.2, -0.1), n_out, OUT_C, "out", -0.25)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=col if n else GRID,
                                     lw=1 + min(3.0, n / 12)))
        ax.text((a[0] + b[0]) / 2, a[1] + off, f"{n}", ha="center", color=col if n else DIM, fontsize=8.5,
                family="DejaVu Sans Mono")
    ax.text(-1.22, 0.42, "host", color=DIM, fontsize=8.5)
    ax.text(-1.22, -1.68, "packets per timestep on each path; host-bound traffic includes the", color=DIM, fontsize=7.4)
    ax.text(-1.22, -1.88, "probed hidden spikes. Event-driven: no spike, no packet.", color=DIM, fontsize=7.4)

    # hidden raster
    ax = fig.add_axes([0.34, 0.47, 0.62, 0.37])
    H = np.zeros((256, T)); H[:, :t + 1] = hid[:t + 1].T
    ax.imshow(H, aspect="auto", cmap=LinearSegmentedColormap.from_list("h", [PANEL, HID_C]), vmin=0, vmax=1,
              interpolation="nearest", extent=(0.5, T + 0.5, 255.5, -0.5))
    ax.axvline(t + 1.5, color=GRID, lw=1)
    ax.set_ylabel("hidden neuron", color=DIM, fontsize=9)
    ax.set_xticks([]); ax.tick_params(colors=DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(f"hidden layer spikes ({int(hid[:t + 1].sum())} so far)", color=TXT, fontsize=10.5, loc="left")

    # output raster
    ax = fig.add_axes([0.34, 0.27, 0.62, 0.15])
    O = np.zeros((10, T)); O[:, :t + 1] = out[:t + 1].T
    ax.imshow(O, aspect="auto", cmap=LinearSegmentedColormap.from_list("o", [PANEL, OUT_C]), vmin=0, vmax=1,
              interpolation="nearest", extent=(0.5, T + 0.5, 9.5, -0.5))
    ax.axvline(t + 1.5, color=GRID, lw=1)
    ax.set_yticks(range(10)); ax.set_ylabel("digit", color=DIM, fontsize=9)
    ax.set_xlabel("timestep", color=DIM, fontsize=9)
    ax.tick_params(colors=DIM, labelsize=7.5)
    for s in ax.spines.values():
        s.set_color(GRID)

    # vote bars
    counts = out[:t + 1].sum(0)
    pred = int(np.argmax(counts)) if counts.any() else -1
    ax = fig.add_axes([0.34, 0.05, 0.40, 0.13])
    cols = [(OK_C if i == label else BAD_C) if (done and i == pred) else (OUT_C if i == pred else GRID) for i in range(10)]
    ax.bar(range(10), counts, color=cols, width=0.7)
    ax.set_xticks(range(10)); ax.set_yticks([])
    ax.set_ylim(0, max(4, out.sum(0).max() * 1.15))
    ax.tick_params(colors=DIM, labelsize=8.5)
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("output spike count per digit", color=DIM, fontsize=9, loc="left")
    fig.text(0.77, 0.15, f"sample {k_sample + 1}/{n_samples}", color=DIM, fontsize=10)
    fig.text(0.77, 0.105, f"true digit  {label}", color=TXT, fontsize=13, family="DejaVu Sans Mono")
    if done:
        fig.text(0.77, 0.06, f"predicted   {pred}  " + ("✓" if pred == label else "✗"),
                 color=OK_C if pred == label else BAD_C, fontsize=13, family="DejaVu Sans Mono", weight="bold")
    else:
        fig.text(0.77, 0.06, "predicted   " + (str(pred) if pred >= 0 else "…"), color=DIM, fontsize=13,
                 family="DejaVu Sans Mono")


def main(samples=(0, 1, 2, 3), fps=6, hold=10):
    OUT.mkdir(parents=True, exist_ok=True)
    net = NetIR.load(sysutil.FIXTURE)
    xs, ys, _ = sysutil.fixture_inputs()
    prog = compile_net(net, 2, 2, probe_layers=(0,))
    tmp = Path(tempfile.mkdtemp())
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    n = 0
    for k, i in enumerate(samples):
        hid, out, traffic = run_sample(net, prog, xs[i])
        T = xs[i].shape[0]
        for t in range(T + hold):
            frame(fig, xs[i], min(t, T - 1), hid, out, traffic, int(ys[i]), k, len(samples), t >= T)
            fig.savefig(tmp / f"f{n:04d}.png", facecolor=BG)
            if k == 1 and t == T:
                fig.savefig(OUT / "snn-demo.png", facecolor=BG)
            n += 1
    plt.close(fig)
    ff = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%04d.png")]
    subprocess.run(ff + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(OUT / "snn-demo.mp4")], check=True)
    subprocess.run(ff + ["-vf", "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];"
                         "[b][p]paletteuse=dither=none", str(OUT / "snn-demo.gif")], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
