import numpy as np
import pytest

from compiler import flits as F
from compiler import image
from compiler.compile import CompileError, compile_net, place, Slice, _snake
from compiler.netir import LayerIR, NetIR, reference_run
from host.api import GoldenBackend, SpikeMeshHost


def rand_net(rng, sizes, w_bits=8):
    lo, hi = (-20, 40) if w_bits == 8 else (-8, 7)
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w = rng.integers(lo, hi + 1, (n_in, n_out))
        w[rng.random(w.shape) < 0.2] = 0
        layers.append(LayerIR(weight=w, bias=rng.integers(-4, 5, n_out),
                              theta=int(rng.integers(20, 80)) if w_bits == 8 else int(rng.integers(6, 20)),
                              leak_shift=int(rng.integers(0, 4)), reset_mode=int(rng.integers(0, 2)),
                              refrac=int(rng.integers(0, 3)), v_min=int(rng.choice([-32768, -50, 0]))))
    return NetIR(sizes[0], layers, w_bits=w_bits)


def run_both(net, prog, T, rng, density=0.3):
    inputs = rng.random((T, net.n_inputs)) < density
    host = SpikeMeshHost(GoldenBackend.for_program(prog))
    host.load(prog)
    return host, inputs, host.run(inputs, probes=True), reference_run(net, inputs)


# (N, A, mesh) configurations; widths both below and above N so splitting happens
CONFIGS = [(16, 64, (4, 4)), (32, 32, (3, 3)), (8, 32, (5, 5))]


@pytest.mark.parametrize("seed", range(60))
def test_golden_from_flits_matches_reference(seed):
    rng = np.random.default_rng(1000 + seed)
    N, A, (mx, my) = CONFIGS[seed % len(CONFIGS)]
    depth = int(rng.integers(1, 5))
    sizes = [int(rng.integers(4, A + 1))]
    for _ in range(depth - 1):
        sizes.append(int(rng.integers(3, A + 1)))
    sizes.append(int(rng.integers(3, 3 * N)))       # output layer may exceed N too
    w_bits = 4 if seed % 5 == 4 else 8
    net = rand_net(rng, sizes, w_bits)
    probes = tuple(range(len(sizes) - 2)) if seed % 2 else ()
    try:
        prog = compile_net(net, mx, my, N, A, w_bits=w_bits, fanout_depth=256,
                           probe_layers=probes, dense_weights=bool(seed % 3))
    except CompileError as e:  # pragma: no cover - configs chosen to fit
        pytest.fail(f"unexpected compile error {e}")
    T = 40
    host, inputs, (out, rasters), ref = run_both(net, prog, T, rng)
    np.testing.assert_array_equal(out, ref[-1])
    for L, r in rasters.items():
        np.testing.assert_array_equal(r, ref[L])


def test_nontrivial_activity_and_splitting():
    rng = np.random.default_rng(7)
    net = rand_net(rng, [40, 60, 50, 70])
    prog = compile_net(net, 4, 4, 16, 64)
    assert len(prog.slices) == 4 + 4 + 5
    _, _, (out, _), ref = run_both(net, prog, 30, rng)
    assert ref[-1].sum() > 20 and ref[1].sum() > 20
    np.testing.assert_array_equal(out, ref[-1])


def test_reset_state_reproduces_run():
    rng = np.random.default_rng(3)
    net = rand_net(rng, [30, 40, 20])
    prog = compile_net(net, 3, 3, 16, 64)
    host, inputs, (out, _), ref = run_both(net, prog, 20, rng)
    host.reset_state()
    out2 = host.run(inputs)
    np.testing.assert_array_equal(out2, ref[-1])
    np.testing.assert_array_equal(out, out2)


def test_default_sizes_single_core():
    rng = np.random.default_rng(11)
    net = rand_net(rng, [100, 200, 10])
    prog = compile_net(net, 2, 2)
    _, _, (out, _), ref = run_both(net, prog, 10, rng)
    np.testing.assert_array_equal(out, ref[-1])


# ------------------------------------------------------------------ error cases

def test_fan_in_too_big():
    net = rand_net(np.random.default_rng(0), [10, 40, 5])
    with pytest.raises(CompileError, match="fan-in 40.*N_AXONS=32"):
        compile_net(net, 4, 4, 32, 32)
    net = rand_net(np.random.default_rng(0), [40, 5])
    with pytest.raises(CompileError, match="fan-in 40 \\(network input\\)"):
        compile_net(net, 4, 4, 32, 32)


def test_mesh_too_small():
    net = rand_net(np.random.default_rng(0), [20, 30, 30])
    with pytest.raises(CompileError, match="mesh too small: network needs 4 cores"):
        compile_net(net, 1, 3, 16, 32)
    compile_net(net, 2, 2, 16, 32)


def test_fanout_depth_overflow():
    net = rand_net(np.random.default_rng(0), [10, 32, 64])
    with pytest.raises(CompileError, match="FANOUT_DEPTH=32"):
        compile_net(net, 3, 3, 32, 32, fanout_depth=32)
    compile_net(net, 3, 3, 32, 32, fanout_depth=64)


def test_fanout_count_overflow():
    # next layer of 256 neurons at N=4 -> 64 cores -> 64 entries per neuron
    net = rand_net(np.random.default_rng(0), [4, 4, 256])
    with pytest.raises(CompileError, match="64 fan-out entries"):
        compile_net(net, 15, 15, 4, 4, fanout_depth=1024)


def test_bad_params():
    net = rand_net(np.random.default_rng(0), [4, 4])
    for kw in [dict(mesh_x=16, mesh_y=1), dict(mesh_x=2, mesh_y=2, n_neurons=24),
               dict(mesh_x=2, mesh_y=2, fanout_depth=2048), dict(mesh_x=2, mesh_y=2, w_bits=4)]:
        with pytest.raises(CompileError):
            compile_net(net, **kw)


# ------------------------------------------------------------------ image, placement, addresses

def test_image_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    net = rand_net(rng, [30, 50, 40])
    prog = compile_net(net, 4, 3, 16, 64, probe_layers=(0,))
    blob = image.to_bytes(prog)
    assert len(blob) > image.HEADER_SIZE + 6 * len(prog.flits)
    assert blob[:4] == b"SMIM"
    p2 = image.from_bytes(blob)
    assert p2.flits == prog.flits
    assert p2.input_map == prog.input_map and p2.output_map == prog.output_map
    assert p2.probe_map == prog.probe_map and p2.slices == prog.slices
    assert image.to_bytes(p2) == blob
    image.save(prog, tmp_path / "x.smim")
    p3 = image.load(tmp_path / "x.smim")
    _, _, (out, _), ref = run_both(net, p3, 15, rng)
    np.testing.assert_array_equal(out, ref[-1])
    bad = bytearray(blob); bad[-1] ^= 1
    with pytest.raises(image.ImageError, match="CRC"):
        image.from_bytes(bytes(bad))
    with pytest.raises(image.ImageError):
        image.from_bytes(blob[:-1])


@pytest.mark.parametrize("seed", range(10))
def test_local_search_never_worse(seed):
    rng = np.random.default_rng(seed)
    sizes = [int(rng.integers(4, 64))] + [int(rng.integers(4, 64)) for _ in range(int(rng.integers(1, 5)))]
    net = rand_net(rng, sizes)
    prog = compile_net(net, 7, 7, 8, 64, fanout_depth=512)
    greedy = compile_net(net, 7, 7, 8, 64, fanout_depth=512, local_search=False)
    assert prog.cost_final <= prog.cost_greedy == greedy.cost_final
    rep = prog.placement_report()
    assert "greedy" in rep and str(prog.cost_final) in rep


def test_local_search_strictly_improves_a_bad_snake():
    # wide 2-layer net on a long 1-row mesh: snake puts layer 1 far away
    slices = [Slice(0, 0, 8), Slice(1, 0, 8), Slice(1, 8, 16), Slice(2, 0, 8)]
    g, f = place(slices, 6, 3, 8, (), 3)
    assert f < g
    assert len({(s.x, s.y) for s in slices}) == 4


def test_every_cfg_flit_decodes_to_valid_core_and_region():
    rng = np.random.default_rng(9)
    net = rand_net(rng, [20, 40, 30])
    N, A, depth = 16, 64, 128
    prog = compile_net(net, 3, 3, N, A, fanout_depth=depth, probe_layers=(0,))
    place_of = {(s.x, s.y): s for s in prog.slices}
    seen_cores, regions = set(), {}
    for f in prog.flits:
        d = F.decode(f)
        assert d["type"] == F.CFG_WRITE
        x, y = d["dest_x"], d["dest_y"]
        assert 0 <= x < 3 and 0 <= y < 3
        seen_cores.add((x, y))
        addr = d["addr"]
        region, low = addr >> 20, addr & 0xFFFFF
        regions.setdefault((x, y), set()).add(region)
        s = place_of.get((x, y))
        if region == 0:
            a, n = divmod(low, N)
            assert s is not None and a < net.layers[s.layer].n_in and n < s.n
        elif region == 1:
            n, fld = low >> 2, low & 3
            assert n < N
            if fld != 3:
                assert s is not None and n < s.n
        elif region == 2:
            assert s is not None and (low >> 1) < depth
        elif (low >> 16) == 6:                       # v0.2 debug space 6: theta_a[n] = 0
            assert s is not None and (low & 0xFFFF) < s.n and d["data"] == 0
        else:
            assert low in (0, 1, 2, 3, 4, 8, 22, 23, 24, 25)
            if low in (23, 24, 25):
                assert d["data"] == 0
    assert seen_cores == {(x, y) for x in range(3) for y in range(3)}
    for c, s in place_of.items():
        assert regions[c] == {0, 1, 2, 3}


def test_sparse_weights_and_init_unused_options():
    rng = np.random.default_rng(2)
    net = rand_net(rng, [20, 20])
    dense = compile_net(net, 2, 2, 32, 32)
    sparse = compile_net(net, 2, 2, 32, 32, dense_weights=False, init_unused=False)
    nnz = int((net.layers[0].weight != 0).sum())
    count_w = lambda p: sum(1 for f in p.flits if (F.decode(f)["addr"] >> 20) == 0)  # noqa: E731
    assert count_w(dense) == 20 * 20 and count_w(sparse) == nnz
    assert len(sparse.flits) < len(dense.flits)
    _, _, (out, _), ref = run_both(net, sparse, 15, rng)
    np.testing.assert_array_equal(out, ref[-1])


def test_snake_order():
    assert _snake(3, 2) == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]


def test_v02_adaptive_state_is_neutralised():
    """SPEC v0.2: theta_a is an undefined neuron memory and regs 23-25 may hold junk;
    the compiled image writes them to 0, so the golden chip still equals reference_run."""
    rng = np.random.default_rng(77)
    net = rand_net(rng, [24, 30, 12])
    prog = compile_net(net, 2, 2, 16, 32)
    backend = GoldenBackend.for_program(prog)
    for k in backend.chip.cores.values():
        k.theta_a[:] = rng.integers(0, 65536, k.N)
        k.set_reg("th_inc", 500)
        k.set_reg("th_decay_shift", 3)
        k.set_reg("plastic_axon_max", 7)
    host = SpikeMeshHost(backend)
    host.load(prog)
    for (x, y), s in {(s.x, s.y): s for s in prog.slices}.items():
        k = backend.chip.cores[(x, y)]
        assert (k.theta_a[: s.n] == 0).all()
        assert (k.regs.th_inc, k.regs.th_decay_shift, k.regs.plastic_axon_max) == (0, 0, 0)
    inputs = rng.random((30, 24)) < 0.3
    out, rasters = host.run(inputs, probes=True)
    np.testing.assert_array_equal(out, reference_run(net, inputs)[-1])
