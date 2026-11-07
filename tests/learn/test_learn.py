"""Small, fast versions of the on-chip learning experiments (learn/)."""
from pathlib import Path

import numpy as np
import pytest

from compiler.netir import NetIR
from golden.spikemesh import Chip, Core
from learn.rstdp_adapt import AdaptCfg, AdaptChip, adapt, balanced_stream, evaluate
from learn.stdp_unsupervised import Cfg, ProtoCore, STDPNet, assign_labels, classify
from train.data import images_to_events

FIX = Path(__file__).resolve().parents[1] / "train" / "fixtures"


def _digits(n, seed=0):
    """Synthetic 'digits' (two bar shapes) as emulated event streams, no download."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, 28, 28), np.uint8)
    lab = rng.integers(0, 2, n)
    for i in range(n):
        r, c = rng.integers(6, 10, 2)
        if lab[i]:
            img[i, r:r + 14, c:c + 3] = 255          # vertical bar
        else:
            img[i, r:r + 3, c:c + 14] = 255          # horizontal bar
    return images_to_events(img), lab


# ------------------------------------------------------------- SPEC findings pinned
def test_spec_trace_decay_never_reaches_zero():
    """SPEC §6 x -= x >> k has fixed points 1..2^k-1 (reported as a SPEC issue)."""
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    for n, v in dict(learn_en=1, tau_pre_shift=2, inc_pre=64, theta=1000).items():
        k.set_reg(n, v)
    c.inject(0, 0, 0)
    for _ in range(60):
        c.step()
    assert k.x_pre[0] == 3                           # stuck forever


def test_trace_fix_prototype_decays_to_zero():
    k = ProtoCore(0, 0, 4, 4, trace_fix=True)
    for n, v in dict(learn_en=1, tau_pre_shift=2, tau_post_shift=1).items():
        k.set_reg(n, v)
    k.x_pre[:] = [64, 3, 1, 0]
    k.x_post[:] = [255, 1, 0, 9]
    for _ in range(40):
        k.learn()
    assert (k.x_pre == 0).all() and (k.x_post == 0).all()
    assert k.regs.tau_pre_shift == 2 and k.regs.tau_post_shift == 1   # registers restored


def test_reward_rounding_is_floor():
    """(r * e) >>> reward_shift floors: small positive -> 0, small negative -> -1."""
    for r, want in ((1, 0), (-1, -1)):
        c = Chip(1, 1, 2, 2)
        k = c.core(0, 0)
        for n, v in dict(learn_en=1, rstdp_en=1, reward_shift=4, theta=1000).items():
            k.set_reg(n, v)
        k.elig[:] = 1
        c.reward(0, 0, r)
        c.step()
        assert (k.w == want).all()


@pytest.mark.parametrize("seed", range(3))
def test_protocore_flags_off_is_golden(seed):
    rng = np.random.default_rng(seed)

    def build(cls):
        c = Chip(1, 1, 16, 16)
        c.cores[(0, 0)] = k = cls(0, 0, 16, 16)
        for n, v in dict(learn_en=1, stdp_a_plus=9, stdp_a_minus=7, stdp_shift=3, tau_pre_shift=2,
                         tau_post_shift=1, inc_pre=50, inc_post=70, theta=60, leak_shift=1,
                         lfsr_seed=77 + seed).items():
            k.set_reg(n, v)
        k.w[:] = np.random.default_rng(seed).integers(-10, 40, (16, 16))
        return c

    a, b = build(Core), build(ProtoCore)
    ins = rng.random((40, 16)) < 0.3
    for t in range(40):
        for c in (a, b):
            for ax in np.flatnonzero(ins[t]):
                c.inject(0, 0, int(ax))
            c.step()
        np.testing.assert_array_equal(a.core(0, 0).w, b.core(0, 0).w)
        np.testing.assert_array_equal(a.core(0, 0).v, b.core(0, 0).v)


# ------------------------------------------------------------- unsupervised STDP
def _small_cfg(**kw):
    base = dict(n_exc=12, k_inh=9, theta=3000, v_min=-3000, refrac=4, a_plus=2, a_minus=48,
                tau_pre_shift=1, tau_post_shift=1, inc_post=255, stdp_shift=8, w_init_max=30)
    base.update(kw)
    return Cfg(**base)


def test_stdp_net_learns_on_chip_only_and_is_deterministic():
    x, _ = _digits(30)
    pix = np.arange(256)
    nets = [STDPNet(_small_cfg(), pix) for _ in range(2)]
    w0 = nets[0].core.w.copy()
    for net in nets:
        for xi in x:
            net.present(xi)
    np.testing.assert_array_equal(nets[0].core.w, nets[1].core.w)       # bit-reproducible
    assert (nets[0].core.w != w0).any()                                 # learning happened
    assert nets[0].core.regs.theta == 3000                              # phase registers restored
    assert nets[0].inh_weights().max() <= nets[0].cfg.w_min + 16        # inhibition stays inhibitory
    # learning off: weights frozen
    nets[0].learning(False)
    w1 = nets[0].core.w.copy()
    for xi in x[:5]:
        nets[0].present(xi)
    np.testing.assert_array_equal(nets[0].core.w, w1)


def test_stdp_fanout_fits_memory():
    with pytest.raises(AssertionError):
        STDPNet(_small_cfg(n_exc=200, k_inh=9), np.arange(256))


def test_stdp_small_run_two_shapes():
    """Pinned small experiment: two bar orientations, 12 neurons, golden chip.

    With the trace-decay fix prototype the net separates the shapes; with
    SPEC v0.1 traces (floor never decays) it does not (see learn/README.md).
    """
    x, y = _digits(200, seed=1)
    accs = {}
    for name, kw in (("spec", {}), ("trace_fix", dict(trace_fix=True))):
        net = STDPNet(_small_cfg(**kw), np.arange(256))
        for xi in x[:120]:
            net.present(xi)
        net.learning(False)
        lab = assign_labels(net.run(x[120:150]), y[120:150], 2)
        accs[name] = (classify(net.run(x[150:]), lab, 2) == y[150:]).mean()
    assert accs["trace_fix"] >= 0.9, accs
    assert accs["spec"] <= 0.7, accs          # documents the SPEC finding; update if SPEC changes


def test_label_assignment_and_vote():
    counts = np.array([[5, 0, 1], [4, 1, 0], [0, 6, 0], [1, 5, 0]])
    y = np.array([0, 0, 1, 1])
    lab = assign_labels(counts, y, 2)
    assert lab.tolist() == [0, 1, 0]
    assert classify(counts, lab, 2).tolist() == [0, 0, 1, 1]


# ------------------------------------------------------------- R-STDP adaptation
def _fixture():
    net = NetIR.load(FIX / "trained_net.npz")
    z = np.load(FIX / "test_inputs.npz")
    shape = tuple(z["shape"])
    x = np.unpackbits(z["x"], count=int(np.prod(shape))).reshape(shape).astype(bool)
    return net, x, z["y"]


def test_rstdp_weights_change_only_with_reward():
    net, x, y = _fixture()
    ac = AdaptChip(net, AdaptCfg())
    for xi in x[:3]:
        ac.present(xi)                                   # learning on, no reward
    assert (ac.nk.w == 0).all() and ac.nk.elig[:, 0].sum() >= 0
    ac.present(x[3], reward=lambda c: 0)
    assert (ac.nk.w == 0).all()
    ac.present(x[4], reward=32)
    assert ac.nk.w[:, 0].max() > 0 and ac.nk.w[:, 1:].max() == 0      # only the head neuron
    ac2 = AdaptChip(net, AdaptCfg())
    ac2.present(x[4], reward=-32)
    assert ac2.nk.w[:, 0].min() < 0


def test_rstdp_small_adaptation_pinned():
    """Small version of learn/rstdp_adapt.py on the committed SG1 net (head = class 9).

    Stream: the fixture's class-9 samples (first 200) repeated 6x + 120 others,
    shuffled with seed 2. Seed 1 of the same stream ends with a dead head
    (all weights negative): the fragility is reported in learn/README.md.
    """
    net, x, y = _fixture()
    xt, yt = balanced_stream(x[200:], y[200:], 9, 40, np.random.default_rng(0))
    tr = np.arange(200)
    nine, old = tr[y[tr] == 9], tr[y[tr] != 9]
    idx = np.random.default_rng(2).permutation(np.concatenate([np.repeat(nine, 6), old[:120]]))
    res = []
    for _ in range(2):
        ac = AdaptChip(net, AdaptCfg())
        before = evaluate(ac, xt, yt)
        adapt(ac, x[idx], y[idx], log=lambda *a: None)
        ac.nk.bias[0] = ac.cfg.eval_bias
        res.append((before, evaluate(ac, xt, yt), ac.nk.w.copy()))
    np.testing.assert_array_equal(res[0][2], res[1][2])                 # deterministic
    before, after = res[0][0], res[0][1]
    assert before["acc_new"] == 0.0
    assert after["acc_new"] >= 0.9 and after["acc_old"] >= 0.9, after
