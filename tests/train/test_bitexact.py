"""SG1 bit-exactness: torch eval forward == compiler.netir.reference_run == golden Chip."""
from pathlib import Path

import numpy as np
import pytest
import torch

from compiler.netir import LayerIR, NetIR, reference_run
from golden.spikemesh import Chip, FanoutEntry, HOST_X
from train.export import export_netir
from train.model import SpikeMeshNet

FIXTURE = Path(__file__).parent / "fixtures" / "trained_net.npz"
FIXTURE_INPUTS = Path(__file__).parent / "fixtures" / "test_inputs.npz"


def fixture_inputs():
    z = np.load(FIXTURE_INPUTS)
    shape = tuple(z["shape"])
    x = np.unpackbits(z["x"], count=int(np.prod(shape))).reshape(shape).astype(bool)
    return x, z["y"], float(z["min_acc"])


def torch_rasters(model, inputs):
    """inputs: bool (N, T, n_in) -> list of bool (N, T, n_out) per layer."""
    model.eval()
    with torch.no_grad():
        r, _ = model(torch.as_tensor(inputs, dtype=torch.float32), quant=True)
    return [x.numpy().astype(bool) for x in r]


def random_float_model(rng, sizes):
    n = len(sizes) - 1
    m = SpikeMeshNet(sizes,
                     leak_shift=[int(rng.integers(0, 5)) for _ in range(n)],
                     reset_mode=[int(rng.integers(0, 2)) for _ in range(n)],
                     refrac=[int(rng.integers(0, 4)) for _ in range(n)],
                     v_min=[None if rng.random() < 0.5 else -float(rng.uniform(0.2, 2)) for _ in range(n)],
                     init_std=[float(rng.uniform(0.1, 0.6)) for _ in range(n)])
    with torch.no_grad():
        for L in m.layers:
            L.bias.uniform_(-0.1, 0.15)
    return m


def random_int_net(rng, sizes, extreme=False):
    """Integer NetIR with random registers; extreme=True provokes sat16 clipping."""
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        if extreme:
            w = rng.integers(-128, 128, (n_in, n_out))
            b = rng.integers(-32768, 32768, n_out)
            theta = int(rng.integers(1, 32768))
        else:
            w = rng.integers(-40, 80, (n_in, n_out))
            b = rng.integers(-20, 20, n_out)
            theta = int(rng.integers(20, 400))
        layers.append(LayerIR(weight=w, bias=b, theta=theta, leak_shift=int(rng.integers(0, 16)),
                              reset_mode=int(rng.integers(0, 2)), refrac=int(rng.integers(0, 16)),
                              v_min=int(rng.integers(-32768, 0)) if rng.random() < 0.5 else -32768))
    return NetIR(sizes[0], layers)


def assert_rasters_equal(a, b):
    assert len(a) == len(b)
    for li, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape, li
        mism = np.argwhere(x != y)
        assert mism.size == 0, f"layer {li}: {len(mism)} mismatches, first {mism[:3].tolist()}"


# ----------------------------------------------------------------- (a) torch == reference_run
@pytest.mark.parametrize("seed", range(6))
def test_random_float_model_export_matches_reference(seed):
    """Random float-parameter model (max-abs scaling, rounding) -> export -> reference_run."""
    rng = np.random.default_rng(seed)
    sizes = [int(rng.integers(8, 64)), int(rng.integers(8, 64)), int(rng.integers(4, 32))]
    if seed % 2:
        sizes.insert(2, int(rng.integers(8, 48)))
    m = random_float_model(rng, sizes)
    net = export_netir(m)
    x = rng.random((16, 30, sizes[0])) < rng.uniform(0.05, 0.5)
    got = torch_rasters(m, x)
    total = 0
    for i in range(len(x)):
        ref = reference_run(net, x[i])
        assert_rasters_equal([g[i] for g in got], ref)
        total += sum(r.sum() for r in ref)
    assert total > 0, "test is vacuous: no spikes at all"


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("extreme", [False, True])
def test_random_int_netir_roundtrip(seed, extreme):
    """Arbitrary integer NetIR (incl. saturation, long leak/refractory) loaded into torch."""
    rng = np.random.default_rng(100 + seed)
    sizes = [int(rng.integers(8, 256)), int(rng.integers(8, 128)), int(rng.integers(4, 32))]
    net = random_int_net(rng, sizes, extreme)
    m = SpikeMeshNet.from_netir(net)
    x = rng.random((8, 25, sizes[0])) < 0.3
    got = torch_rasters(m, x)
    for i in range(len(x)):
        assert_rasters_equal([g[i] for g in got], reference_run(net, x[i]))
    # export of the loaded model returns the identical integers
    net2 = export_netir(m)
    for a, b in zip(net.layers, net2.layers):
        assert np.array_equal(a.weight, b.weight) and np.array_equal(a.bias, b.bias)
        assert (a.theta, a.leak_shift, a.reset_mode, a.refrac, a.v_min) == \
               (b.theta, b.leak_shift, b.reset_mode, b.refrac, b.v_min)


def test_layer_latency_is_one_timestep():
    """SPEC §3: a single input event at t=0 reaches layer 1 at t=1, layer 2 at t=2."""
    one = lambda n_in, n_out: LayerIR(weight=np.full((n_in, n_out), 10), bias=np.zeros(n_out, int),
                                      theta=10)
    net = NetIR(1, [one(1, 1), one(1, 1), one(1, 1)])
    x = np.zeros((5, 1), bool); x[0, 0] = True
    r = torch_rasters(SpikeMeshNet.from_netir(net), x[None])
    assert [np.flatnonzero(l[0, :, 0]).tolist() for l in r] == [[0], [1], [2]]


def test_readout_ties_go_to_lowest_index():
    r = torch.zeros(3, 4, 5)
    r[0, :2, 3] = 1; r[0, :2, 1] = 1          # tie 1 vs 3 -> 1
    r[2, 0, 4] = 1                            # sample 1 all-zero -> 0
    assert SpikeMeshNet.predict(r).tolist() == [1, 0, 4]


def test_trained_net_matches_reference():
    """Committed trained artifact on >= 200 emulated N-MNIST test samples."""
    if not FIXTURE.exists():
        pytest.skip("fixture missing; run python -m train.make_fixture")
    net = NetIR.load(FIXTURE)
    x, y, min_acc = fixture_inputs()
    assert len(x) >= 200 and x.shape[2] == net.n_inputs
    m = SpikeMeshNet.from_netir(net)
    got = torch_rasters(m, x)
    nspk = 0
    for i in range(len(x)):
        ref = reference_run(net, x[i])
        assert_rasters_equal([g[i] for g in got], ref)
        nspk += ref[0].sum()
    assert nspk > 0
    # and the trained net actually classifies: stored labels / accuracy
    acc = (SpikeMeshNet.predict(torch.as_tensor(got[-1])).numpy() == y).mean()
    assert acc >= min_acc


# ----------------------------------------------------------------- (b) golden Chip
def run_on_chip(net: NetIR, inputs: np.ndarray, n_core: int):
    """Hand-place layer i on core i of a 2x2 mesh (like tests/golden). Returns rasters."""
    place = [(0, 0), (1, 0), (1, 1), (0, 1)][: len(net.layers)]
    chip = Chip(2, 2, n_core, n_core)
    for li, (L, (x, y)) in enumerate(zip(net.layers, place)):
        k = chip.core(x, y)
        k.w[: L.n_in, : L.n_out] = L.weight
        k.bias[: L.n_out] = L.bias
        for name in ("theta", "leak_shift", "reset_mode", "refrac", "v_min"):
            k.set_reg(name, getattr(L, name))
        for n in range(L.n_out):
            ents = [FanoutEntry(HOST_X, 0, 0)]
            if li + 1 < len(place):
                ents.insert(0, FanoutEntry(*place[li + 1], n))
            k.set_fanout(n, ents, 2 * n)
    T = inputs.shape[0]
    outs = chip.run([[(0, 0, a) for a in np.flatnonzero(inputs[t])] for t in range(T)])
    got = [np.zeros((T, L.n_out), bool) for L in net.layers]
    for t, o in enumerate(outs):
        for (sx, sy, n) in o:
            got[place.index((sx, sy))][t, n] = True
    return got


@pytest.mark.parametrize("seed", range(4))
def test_golden_chip_matches_torch(seed):
    rng = np.random.default_rng(200 + seed)
    sizes = [32, 32, 24, 10] if seed % 2 else [32, 32, 10]
    m = random_float_model(rng, sizes)
    net = export_netir(m)
    x = rng.random((4, 20, 32)) < 0.3
    got = torch_rasters(m, x)
    for i in range(len(x)):
        assert_rasters_equal(run_on_chip(net, x[i], 32), [g[i] for g in got])


def test_golden_chip_matches_trained_net():
    """Full-size trained net (256 -> 256 -> 10) on the 256x256-core golden chip."""
    if not FIXTURE.exists():
        pytest.skip("fixture missing")
    net = NetIR.load(FIXTURE)
    x = fixture_inputs()[0][:3]
    got = torch_rasters(SpikeMeshNet.from_netir(net), x)
    for i in range(len(x)):
        assert_rasters_equal(run_on_chip(net, x[i], 256), [g[i] for g in got])


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "data" / "mnist").exists(),
                    reason="MNIST not downloaded (python -m train.data)")
def test_tiny_training_run_exports_bit_exact(tmp_path):
    """End to end: train.main (float + QAT epoch) -> best.pt + net.npz; the trained
    model itself (max-abs scaling path, not from_netir) == reference_run on 200 samples."""
    from train.data import emulated_nmnist
    from train.train import main
    log = main(["--limit", "1500", "--val", "200", "--epochs", "3", "--float-epochs", "1",
                "--test-limit", "200", "--bs", "32", "--lr", "5e-3", "--leak", "2", "--count-gain", "0.3",
                "--hidden", "64", "--threads", "2", "--out", str(tmp_path)])
    ck = torch.load(tmp_path / "best.pt")
    c = ck["config"]
    n = len(c["sizes"]) - 1
    m = SpikeMeshNet(c["sizes"], c["leak_shift"], c["reset_mode"], c["refrac"], [None] * n)
    m.load_state_dict(ck["state"])
    net = NetIR.load(tmp_path / "net.npz")
    x, y = emulated_nmnist("test", limit=200)
    got = torch_rasters(m, x)
    correct = 0
    for i in range(len(x)):
        ref = reference_run(net, x[i])
        assert_rasters_equal([g[i] for g in got], ref)
        correct += ref[-1].sum(0).argmax() == y[i]
    assert correct / len(x) > 0.4          # learned something (chance = 0.1)
    assert abs(log["test_int"] - net.meta["test_acc_int"]) < 1e-12
