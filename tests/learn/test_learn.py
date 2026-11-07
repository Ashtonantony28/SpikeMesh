"""Small, fast versions of the on-chip learning experiments (learn/)."""
from pathlib import Path

import numpy as np
import pytest

from compiler.netir import NetIR
from learn.rstdp_adapt import AdaptCfg, AdaptChip, adapt, balanced_stream, evaluate
from learn.stdp_unsupervised import Cfg, STDPNet, assign_labels, classify
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


# (The SPEC v0.1 findings that these experiments produced are now SPEC v0.2
#  rules; they are pinned in tests/golden/test_v02.py.)


# ------------------------------------------------------------- unsupervised STDP
def _small_cfg(**kw):
    base = dict(n_exc=12, k_inh=9, theta=3000, v_min=-3000, refrac=4, a_plus=2, a_minus=48,
                tau_pre_shift=1, tau_post_shift=1, inc_post=255, stdp_shift=8, w_init_max=30)
    base.update(kw)
    return Cfg(**base)


def test_stdp_net_learns_on_chip_only_and_is_deterministic():
    x, _ = _digits(30)
    pix = np.arange(256)
    nets = [STDPNet(_small_cfg(th_inc=200), pix) for _ in range(2)]
    k0 = nets[0].cores[0]
    w0 = k0.w.copy()
    for net in nets:
        for xi in x:
            net.present(xi)
    np.testing.assert_array_equal(k0.w, nets[1].cores[0].w)            # bit-reproducible
    assert (k0.w != w0).any()                                           # learning happened
    assert k0.regs.theta == 3000                                        # phase registers restored
    assert (nets[0].inh_weights() == nets[0].cfg.w_inh).all()           # plastic_axon_max: fixed
    assert k0.theta_a.max() > 0                                         # adaptive threshold active
    # learning off: weights and thresholds frozen
    nets[0].learning(False)
    w1, th1 = k0.w.copy(), k0.theta_a.copy()
    for xi in x[:5]:
        nets[0].present(xi)
    np.testing.assert_array_equal(k0.w, w1)
    np.testing.assert_array_equal(k0.theta_a, th1)


def test_stdp_plastic_inhibition_drifts_without_v02_register():
    x, _ = _digits(30)
    net = STDPNet(_small_cfg(plastic_inh=False, a_plus=40), np.arange(256))
    for xi in x:
        net.present(xi)
    assert (net.inh_weights() != net.cfg.w_inh).any()


def test_stdp_fanout_fits_memory():
    with pytest.raises(AssertionError):
        STDPNet(_small_cfg(n_exc=200, k_inh=9), np.arange(256))
    with pytest.raises(AssertionError):
        STDPNet(_small_cfg(n_cores=2), np.arange(256))          # direct inhibition is 1-core


def test_stdp_two_cores_with_inhibitory_core():
    x, _ = _digits(10)
    net = STDPNet(_small_cfg(n_cores=2, inh="core", n_inh=4, k_inh=4), np.arange(256))
    counts = net.run(x)
    assert counts.shape == (10, 24) and counts.sum() > 0
    assert (net.inh_weights() == net.cfg.w_inh).all()


def test_stdp_small_run_two_shapes():
    """Pinned small experiment: two bar orientations, 12 neurons, golden chip, SPEC v0.2."""
    x, y = _digits(200, seed=1)
    net = STDPNet(_small_cfg(), np.arange(256))
    for xi in x[:120]:
        net.present(xi)
    net.learning(False)
    lab = assign_labels(net.run(x[120:150]), y[120:150], 2)
    acc = (classify(net.run(x[150:]), lab, 2) == y[150:]).mean()
    assert acc >= 0.9, acc


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


@pytest.mark.parametrize("stream_seed", [1, 2])
def test_rstdp_small_adaptation_pinned(stream_seed):
    """Small version of learn/rstdp_adapt.py on the committed SG1 net (head = class 9).

    Stream: the fixture's class-9 samples (first 200) repeated 6x + 120 others,
    shuffled. Under SPEC v0.1 (floor-rounded reward) stream seed 1 ended with a
    dead head; with v0.2's rounded reward every seed 0..4 learns.
    """
    net, x, y = _fixture()
    xt, yt = balanced_stream(x[200:], y[200:], 9, 40, np.random.default_rng(0))
    tr = np.arange(200)
    nine, old = tr[y[tr] == 9], tr[y[tr] != 9]
    idx = np.random.default_rng(stream_seed).permutation(np.concatenate([np.repeat(nine, 6), old[:120]]))
    res = []
    for _ in range(2 if stream_seed == 2 else 1):
        ac = AdaptChip(net, AdaptCfg())
        before = evaluate(ac, xt, yt)
        adapt(ac, x[idx], y[idx], log=lambda *a: None)
        ac.nk.bias[0] = ac.cfg.eval_bias
        res.append((before, evaluate(ac, xt, yt), ac.nk.w.copy()))
    if len(res) == 2:
        np.testing.assert_array_equal(res[0][2], res[1][2])             # deterministic
    before, after = res[0][0], res[0][1]
    assert before["acc_new"] == 0.0
    assert after["acc_new"] >= 0.8 and after["acc_old"] >= 0.9, after   # 6 of 7 new-class samples
