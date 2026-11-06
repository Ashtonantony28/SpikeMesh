import numpy as np
import pytest

from golden.spikemesh import Chip, LFSR16, FanoutEntry, HOST_X
from compiler.netir import LayerIR, NetIR, reference_run


def test_lfsr_period_is_maximal():
    l = LFSR16(1)
    seen = {l.state}
    for _ in range(65534):
        seen.add(l.next())
    assert len(seen) == 65535 and l.next() == 1


def test_seed_zero_becomes_one():
    assert LFSR16(0).state == 1


def test_integrate_and_fire_subtract_reset():
    c = Chip(1, 1, 8, 8)
    k = c.core(0, 0)
    k.w[0, :] = 40
    k.set_reg("theta", 64)
    k.set_fanout(0, [FanoutEntry(HOST_X, 0, 0)], 0)
    outs = c.run([[(0, 0, 0)]] * 3)
    assert outs == [[], [(0, 0, 0)], []]
    assert k.v[0] == 56  # 40, 80->16, 56


def test_refractory_holds_v_and_discards_input():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    k.w[0, 0] = 100
    k.set_reg("theta", 50); k.set_reg("refrac", 2); k.set_reg("reset_mode", 1)
    c.run([[(0, 0, 0)]] * 4)
    # t0 spike (v=0, refc=2), t1/t2 refractory, t3 integrates 100 -> spike
    assert k.v[0] == 0 and k.refc[0] == 2


def test_leak_is_floor_shift_on_negative():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    k.v[:] = -7
    k.set_reg("leak_shift", 1); k.set_reg("theta", 1000)
    c.step()
    assert k.v[0] == -7 - (-4)  # -7 >> 1 == -4


def test_saturation():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    k.v[:] = 32000; k.bias[:] = 32767
    k.set_reg("theta", 32767); k.set_reg("reset_mode", 1)
    c.step()
    assert (k.v == 0).all()  # saturated to 32767 >= theta -> spike -> zero


def _random_net(rng, sizes, theta=40):
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers.append(LayerIR(weight=rng.integers(-20, 40, (n_in, n_out)),
                              bias=rng.integers(-3, 3, n_out), theta=theta,
                              leak_shift=int(rng.integers(0, 4)), reset_mode=int(rng.integers(0, 2)),
                              refrac=int(rng.integers(0, 3))))
    return NetIR(sizes[0], layers)


@pytest.mark.parametrize("seed", range(5))
def test_chip_matches_layer_reference(seed):
    """Hand-map a 3-layer net onto a 2x2 chip; golden chip == layer reference."""
    rng = np.random.default_rng(seed)
    net = _random_net(rng, [32, 24, 16, 10])
    T = 20
    inputs = rng.random((T, 32)) < 0.3
    ref = reference_run(net, inputs)

    chip = Chip(2, 2, 32, 32)
    place = [(0, 0), (1, 0), (1, 1)]
    for li, (L, (x, y)) in enumerate(zip(net.layers, place)):
        k = chip.core(x, y)
        k.w[: L.n_in, : L.n_out] = L.weight
        k.bias[: L.n_out] = L.bias
        for name in ("theta", "leak_shift", "reset_mode", "refrac", "v_min"):
            k.set_reg(name, getattr(L, name))
        for n in range(L.n_out):
            if li + 1 < len(place):
                nx, ny = place[li + 1]
                k.set_fanout(n, [FanoutEntry(nx, ny, n), FanoutEntry(HOST_X, 0, 0)], 2 * n)
            else:
                k.set_fanout(n, [FanoutEntry(HOST_X, 0, 0)], n)
    outs = chip.run([[(0, 0, a) for a in np.flatnonzero(inputs[t])] for t in range(T)])
    for li, (x, y) in enumerate(place):
        got = np.zeros((T, net.layers[li].n_out), bool)
        for t, o in enumerate(outs):
            for (sx, sy, n) in o:
                if (sx, sy) == (x, y):
                    got[t, n] = True
        np.testing.assert_array_equal(got, ref[li])


def test_stdp_pre_before_post_potentiates():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    for name, v in dict(learn_en=1, stdp_a_plus=8, stdp_a_minus=8, stdp_shift=4, tau_pre_shift=2,
                        tau_post_shift=2, inc_pre=64, inc_post=64, theta=1000).items():
        k.set_reg(name, v)
    k.w[:] = 0
    c.run([[(0, 0, 1)], [], []])          # pre spike on axon 1 at t0
    k.v[2] = 2000                          # force neuron 2 to fire at next update
    c.step(); c.step()                     # post spike, then LEARN potentiates
    assert k.w[1, 2] > 0 and k.w[1, 0] == 0


def test_stdp_post_before_pre_depresses():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    for name, v in dict(learn_en=1, stdp_a_plus=8, stdp_a_minus=8, stdp_shift=4, tau_pre_shift=2,
                        tau_post_shift=2, inc_pre=64, inc_post=64, theta=1000).items():
        k.set_reg(name, v)
    k.v[2] = 2000
    c.step(); c.step()                     # neuron 2 fires at t0, trace bumped at t1 LEARN
    c.run([[(0, 0, 1)], []])               # pre on axon 1 afterwards
    assert k.w[1, 2] < 0


def test_rstdp_only_changes_weights_on_reward():
    c = Chip(1, 1, 4, 4)
    k = c.core(0, 0)
    for name, v in dict(learn_en=1, rstdp_en=1, stdp_a_plus=8, stdp_a_minus=0, stdp_shift=2,
                        tau_pre_shift=2, tau_post_shift=2, inc_pre=64, inc_post=64, theta=1000,
                        reward_shift=6).items():
        k.set_reg(name, v)
    c.run([[(0, 0, 1)]])
    k.v[2] = 2000
    c.step(); c.step()
    assert (k.w == 0).all() and k.elig[1, 2] > 0
    c.reward(0, 0, 100)
    c.step()
    assert k.w[1, 2] > 0
