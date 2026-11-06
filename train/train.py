"""QAT training of a SpikeMesh SNN on emulated N-MNIST (or real data via --real).

Example:
    python -m train.train --epochs 12 --float-epochs 3 --out runs/base
Writes runs/<out>/{best.pt, net.npz (NetIR), log.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train.data import DEFAULT_D, DEFAULT_T, DEFAULT_THR, emulated_nmnist
from train.export import export_netir
from train.model import SpikeMeshNet, count_ce_loss


def evaluate(model, x, y, quant, bs=1000):
    model.eval()
    correct, rate = 0, []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).float()
            r, _ = model(xb, quant=quant)
            correct += (SpikeMeshNet.predict(r[-1]).numpy() == y[i:i + bs]).sum()
            rate.append([l.mean().item() for l in r])
    return correct / len(x), np.mean(rate, 0).tolist()


def parse_list(s, n, typ):
    v = [typ(a) if a.lower() != "none" else None for a in s.split(",")]
    return v * n if len(v) == 1 else v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="256", help="comma list of hidden sizes, e.g. 256 or 256,256")
    ap.add_argument("--real", choices=["nmnist", "dvsgesture"], default=None,
                    help="train on real data cached by train/data_real.py (not available in the cloud)")
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    ap.add_argument("--d", type=float, default=DEFAULT_D)
    ap.add_argument("--thr", type=float, default=DEFAULT_THR)
    ap.add_argument("--leak", default="3", help="leak_shift per layer (one value = all)")
    ap.add_argument("--reset", default="0", help="reset_mode per layer")
    ap.add_argument("--refrac", default="0", help="refrac per layer")
    ap.add_argument("--vmin", default="-1", help="v_min per layer in units of theta, or none")
    ap.add_argument("--init-std", type=float, default=0.15)
    ap.add_argument("--slope", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--float-epochs", type=int, default=3, help="initial epochs without fake-quant")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--count-gain", type=float, default=0.5)
    ap.add_argument("--mem-weight", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="use only N training samples (smoke tests)")
    ap.add_argument("--test-limit", type=int, default=None, help="evaluate on the first N test samples only")
    ap.add_argument("--val", type=int, default=5000, help="hold out last N train samples for model selection")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/base")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.real:
        from train.data_real import load_cached
        xtr, ytr = load_cached(a.real, "train", a.T)
        xte, yte = load_cached(a.real, "test", a.T)
        dataset = f"real {a.real} (tonic, 16x16 binning) T={a.T}"
    else:
        xtr, ytr = emulated_nmnist("train", a.T, a.d, a.thr)
        xte, yte = emulated_nmnist("test", a.T, a.d, a.thr)
        dataset = f"emulated N-MNIST (MNIST saccade emulation, not real N-MNIST) T={a.T} d={a.d} thr={a.thr}"
    n_classes = int(max(ytr.max(), yte.max())) + 1
    if a.test_limit:
        xte, yte = xte[:a.test_limit], yte[:a.test_limit]
    xva, yva = xtr[-a.val:], ytr[-a.val:]
    xtr, ytr = xtr[:-a.val], ytr[:-a.val]
    if a.limit:
        xtr, ytr = xtr[:a.limit], ytr[:a.limit]
        xva, yva = xva[:a.limit // 4 + 1], yva[:a.limit // 4 + 1]

    hidden = [int(h) for h in a.hidden.split(",")]
    sizes = [256] + hidden + [n_classes]
    n = len(sizes) - 1
    model = SpikeMeshNet(sizes, parse_list(a.leak, n, int), parse_list(a.reset, n, int),
                         parse_list(a.refrac, n, int), parse_list(a.vmin, n, float),
                         init_std=[a.init_std * 16 / np.sqrt(s) for s in sizes[:-1]], slope=a.slope)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * ((len(xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)

    log = {"args": vars(a), "epochs": []}
    best = -1.0
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        quant = ep >= a.float_epochs
        model.train()
        perm = rng.permutation(len(xtr))
        tl, nb = 0.0, 0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = torch.from_numpy(xtr[idx]).float()
            yb = torch.from_numpy(ytr[idx])
            r, v2 = model(xb, quant=quant)
            loss = count_ce_loss(r[-1], v2, yb, a.count_gain, a.mem_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item(); nb += 1
        va_f, _ = evaluate(model, xva, yva, quant=False)
        va_i, rates = evaluate(model, xva, yva, quant=True)
        rec = dict(epoch=ep, quant=quant, loss=tl / nb, val_float=va_f, val_int=va_i, rates=rates,
                   minutes=(time.time() - t0) / 60)
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if va_i > best:
            best = va_i
            torch.save({"state": model.state_dict(), "config": model.config(), "args": vars(a)},
                       out / "best.pt")
        elapsed = (time.time() - t0) / 60
        if elapsed * (ep + 2) / (ep + 1) > a.max_minutes:  # next epoch would overrun
            print("time budget reached, stopping")
            break

    ck = torch.load(out / "best.pt")
    model.load_state_dict(ck["state"])
    te_f, _ = evaluate(model, xte, yte, quant=False)
    te_i, rates = evaluate(model, xte, yte, quant=True)
    net = export_netir(model, meta={"dataset": dataset, "T": a.T, "d": a.d, "thr": a.thr,
                                    "test_acc_int": te_i, "test_acc_float": te_f})
    net.save(out / "net.npz")
    log.update(test_float=te_f, test_int=te_i, test_rates=rates, best_val_int=best,
               train_minutes=(time.time() - t0) / 60,
               thetas=[L.theta for L in net.layers])
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"TEST float {te_f:.4f}  int {te_i:.4f}  gap {te_f - te_i:+.4f}  "
          f"({log['train_minutes']:.1f} min) -> {out}/net.npz")
    return log


if __name__ == "__main__":
    main()
