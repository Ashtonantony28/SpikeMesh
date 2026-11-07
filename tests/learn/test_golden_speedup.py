"""The golden LFSR / LEARN speed-up is a pure refactor: identical to straightforward loops.

The reference follows SPEC v0.2 (§6 + §9) with one loop per row / column and
one LFSR next() per draw, i.e. the original implementation plus the v0.2 rules.
"""
import numpy as np
import pytest

from golden.spikemesh import Chip, Core, LFSR16, sat


def ref_take(lfsr: LFSR16, k: int) -> np.ndarray:
    """Original implementation of LFSR16.take (Python loop over next())."""
    out = np.empty(k, dtype=np.int64)
    for i in range(k):
        out[i] = lfsr.next()
    return out


def ref_learn(self: Core):
    """Reference Core.learn: per-row / per-column loops, loop LFSR, SPEC v0.2."""
    R = self.regs
    if not R.learn_en:
        return

    def sr(m):
        s = R.stdp_shift
        rnd = ref_take(self.lfsr, m.size)
        if s == 0:
            return m.copy()
        mask = (1 << s) - 1
        return (m >> s) + ((rnd & mask) < (m & mask)).astype(np.int64)

    def decay(x, k):
        out = x.copy()
        if k:
            for i in range(len(out)):
                if out[i] > 0:
                    out[i] -= max(1, int(out[i]) >> k)
        return out

    self.x_pre = decay(self.x_pre, R.tau_pre_shift)
    self.x_post = decay(self.x_post, R.tau_post_shift)
    if R.rstdp_en:
        if R.tau_elig_shift:
            self.elig -= self.elig >> R.tau_elig_shift
        if self.pending_reward:
            re = self.pending_reward * self.elig
            if R.reward_shift:
                re = (re + (1 << (R.reward_shift - 1))) >> R.reward_shift
            self.w = sat(self.w + re, R.w_min, R.w_max)
            self.pending_reward = 0
    target = self.elig if R.rstdp_en else self.w
    lo, hi = (-32768, 32767) if R.rstdp_en else (R.w_min, R.w_max)
    for a in np.flatnonzero(self.pre_flag):
        d = sr(R.stdp_a_minus * self.x_post)
        target[a] = sat(target[a] - d, lo, hi)
    for n in np.flatnonzero(self.post_flag):
        d = sr(R.stdp_a_plus * self.x_pre)
        for a in range(self.A):
            if R.plastic_axon_max == 0 or a < R.plastic_axon_max:
                target[a, n] = min(max(target[a, n] + d[a], lo), hi)
    self.x_pre[self.pre_seen] = np.minimum(self.x_pre[self.pre_seen] + R.inc_pre, 255)
    self.x_post[self.post_flag] = np.minimum(self.x_post[self.post_flag] + R.inc_post, 255)
    self.pre_flag[:] = False
    self.pre_seen[:] = False
    self.post_flag[:] = False


class RefCore(Core):
    learn = ref_learn


@pytest.mark.parametrize("seed", [0, 1, 0xB400, 0xFFFF, 12345, 0])
def test_lfsr_take_identical(seed):
    a, b = LFSR16(seed), LFSR16(seed)
    for k in [0, 1, 7, 1000, 65535, 70000, 3]:
        np.testing.assert_array_equal(a.take(k), ref_take(b, k))
        assert a.state == b.state
        assert a.next() == b.next()


def _make_chip(core_cls, rng_seed, n=24, rstdp=False):
    rng = np.random.default_rng(rng_seed)
    chip = Chip(1, 1, n, n)
    chip.cores[(0, 0)] = k = core_cls(0, 0, n, n)
    regs = dict(learn_en=1, rstdp_en=int(rstdp), stdp_a_plus=int(rng.integers(0, 40)),
                stdp_a_minus=int(rng.integers(0, 40)), stdp_shift=int(rng.integers(0, 9)),
                tau_pre_shift=int(rng.integers(0, 5)), tau_post_shift=int(rng.integers(0, 5)),
                inc_pre=int(rng.integers(0, 200)), inc_post=int(rng.integers(0, 200)),
                lfsr_seed=int(rng.integers(0, 65536)), w_min=int(rng.integers(-128, 0)),
                w_max=int(rng.integers(1, 128)), theta=int(rng.integers(20, 200)),
                tau_elig_shift=int(rng.integers(0, 6)), reward_shift=int(rng.integers(0, 8)),
                leak_shift=2, plastic_axon_max=int(rng.choice([0, 0, int(rng.integers(1, n))])),
                th_inc=int(rng.integers(0, 30)), th_decay_shift=int(rng.integers(0, 5)))
    for name, v in regs.items():
        k.set_reg(name, v)
    k.w[:] = rng.integers(-40, 80, (n, n))
    from golden.spikemesh import FanoutEntry, HOST_X
    for i in range(n):
        k.set_fanout(i, [FanoutEntry(HOST_X, 0, 0), FanoutEntry(0, 0, (i * 7) % n)], 2 * i)
    return chip, rng


@pytest.mark.parametrize("rstdp", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_learning_identical_to_reference(seed, rstdp):
    new, rng = _make_chip(Core, seed, rstdp=rstdp)
    ref, _ = _make_chip(RefCore, seed, rstdp=rstdp)
    T = 60
    ins = rng.random((T, 24)) < 0.3
    rewards = rng.integers(-300, 300, T) * (rng.random(T) < 0.2)
    for t in range(T):
        for c in (new, ref):
            for a in np.flatnonzero(ins[t]):
                c.inject(0, 0, int(a))
            if rewards[t]:
                c.reward(0, 0, int(rewards[t]))
        assert new.step() == ref.step()
        a, b = new.core(0, 0), ref.core(0, 0)
        for f in ("w", "elig", "x_pre", "x_post", "v", "refc", "theta_a"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f"{f} t={t}")
        assert a.lfsr.state == b.lfsr.state
    assert (new.core(0, 0).w != _make_chip(Core, seed)[0].core(0, 0).w).any(), "no learning happened"
