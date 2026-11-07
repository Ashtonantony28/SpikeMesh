"""SPEC v0.2 (§9) learning rules in the golden model."""
import numpy as np

from golden.spikemesh import SPEC_VERSION, Chip, FanoutEntry, HOST_X, LFSR16

REG = 3 << 20


def dbg(space, idx):
    return REG | (space << 16) | idx


def core(n=4, **regs):
    c = Chip(1, 1, n, n)
    k = c.core(0, 0)
    for name, v in regs.items():
        k.set_reg(name, v)
    return c, k


def test_version():
    assert SPEC_VERSION == "0.2"


# rule 1 ---------------------------------------------------------------------
def test_rule1_trace_decay_reaches_zero():
    c, k = core(learn_en=1, tau_pre_shift=2, tau_post_shift=3, inc_pre=64, theta=1000)
    k.x_post[:] = [255, 7, 1, 0]
    c.inject(0, 0, 0)
    seq = []
    for _ in range(40):
        c.step()
        seq.append(int(k.x_pre[0]))
    # 64 -> 48 -> 36 -> 27 -> 21 -> 16 -> 12 -> 9 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 0
    assert seq[:16] == [64, 48, 36, 27, 21, 16, 12, 9, 7, 6, 5, 4, 3, 2, 1, 0]
    assert (k.x_post == 0).all() and k.x_pre[0] == 0


def test_rule1_zero_shift_means_no_decay():
    c, k = core(learn_en=1, inc_pre=10, theta=1000)
    c.inject(0, 0, 1)
    for _ in range(5):
        c.step()
    assert k.x_pre[1] == 10


# rule 2 ---------------------------------------------------------------------
def _reward_dw(r, e, shift, w0=0):
    c, k = core(2, learn_en=1, rstdp_en=1, reward_shift=shift, theta=1000)
    k.w[:] = w0
    k.elig[:] = e
    c.reward(0, 0, r)
    c.step()
    return int(k.w[0, 0]) - w0


def test_rule2_rounded_reward():
    assert _reward_dw(8, 1, 4) == 1          # (8 + 8) >> 4
    assert _reward_dw(7, 1, 4) == 0          # (7 + 8) >> 4
    assert _reward_dw(-8, 1, 4) == 0         # (-8 + 8) >> 4: half rounds up
    assert _reward_dw(-9, 1, 4) == -1
    assert _reward_dw(-24, 1, 4) == -1       # -1.5 -> -1 (floor(x + 0.5))
    assert _reward_dw(3, 5, 0) == 15         # shift 0: r * e
    assert _reward_dw(100, 100, 2, w0=100) == 27   # clamp to w_max = 127


# rule 3 ---------------------------------------------------------------------
def test_rule3_non_plastic_axons():
    seed = 0x3A5
    n = 8
    c, k = core(n, learn_en=1, plastic_axon_max=5, stdp_a_plus=50, stdp_a_minus=50, stdp_shift=2,
                inc_pre=100, inc_post=100, tau_pre_shift=1, tau_post_shift=1, theta=1000,
                lfsr_seed=seed)
    k.w[:] = 10
    k.x_post[:] = 80                    # depression would hit every pre row
    for a in (1, 6):                    # axon 6 is non-plastic
        c.inject(0, 0, a)
    k.v[3] = 5000                       # neuron 3 fires at t0 -> potentiation at t1
    c.step()
    assert k.cfg_read(dbg(3, 6)) == 0 and k.cfg_read(dbg(3, 1)) == 0  # flags cleared by LEARN
    assert k.x_pre[6] == 100 and k.x_pre[1] == 100                     # x_pre still bumps
    assert (k.w[6] == 10).all() and (k.w[1] < 10).all()                # only plastic row depressed
    c.step()                            # LEARN(t1): potentiation of column 3
    assert (k.w[5:, 3] == 10).all() and (k.w[:5, 3] > 0).all()
    assert k.w[1, 3] == 127 and k.w[0, 3] == 10      # x_pre[0] = 0: no change
    # LFSR draws: depression 1 row x n (axon 1 only) + potentiation n axons (all drawn)
    ref = LFSR16(seed)
    ref.take(n + n)
    assert k.lfsr.state == ref.state


def test_rule3_zero_means_all_plastic():
    c, k = core(learn_en=1, stdp_a_minus=50, stdp_shift=0, theta=1000)
    k.x_post[:] = 2
    c.inject(0, 0, 3)
    c.step()
    assert (k.w[3] == -100).all()


def test_rule3_rstdp_eligibility_also_fixed():
    c, k = core(4, learn_en=1, rstdp_en=1, plastic_axon_max=2, stdp_a_plus=8, stdp_shift=0,
                inc_pre=10, theta=1000)
    for a in range(4):
        c.inject(0, 0, a)
    c.step()
    k.v[0] = 5000
    c.step()
    c.step()
    assert (k.elig[:2, 0] > 0).all() and (k.elig[2:, 0] == 0).all()


# rule 4 ---------------------------------------------------------------------
def test_rule4_adaptive_threshold():
    c, k = core(2, theta=100, th_inc=40, th_decay_shift=2, learn_en=0)
    k.w[0, 0] = 120
    k.set_fanout(0, [FanoutEntry(HOST_X, 0, 0)], 0)
    outs = c.run([[(0, 0, 0)]] * 4)
    # t0: v2 120 >= 100 -> spike, v = 20 (subtract theta_eff 100); theta_a: 0 -> 40
    # t1: v2 140 >= 140 -> spike, v = 0; theta_a: 40 - 10 + 40 = 70
    # t2: v2 120 < 170 -> no; theta_a 70 - 17 = 53
    # t3: v2 240 >= 153 -> spike, v = 87; theta_a 53 - 13 + 40 = 80
    assert [len(o) for o in outs] == [1, 1, 0, 1]
    assert k.v[0] == 87 and k.theta_a[0] == 80
    assert k.theta_a[1] == 0


def test_rule4_decays_during_refractory_and_saturates():
    c, k = core(2, theta=32000, th_inc=60000, th_decay_shift=0, refrac=3, reset_mode=1)
    k.v[0] = 32767
    c.step()
    assert k.theta_a[0] == 60000
    c.step()                                 # theta_eff = min(32767, 92000) = 32767
    k.set_reg("th_decay_shift", 1)
    c.step()
    assert k.theta_a[0] == 30000             # decayed while refractory
    k.theta_a[1] = 65535
    k.set_reg("th_inc", 100)
    k.set_reg("th_decay_shift", 0)
    k.v[1] = 32767
    k.set_reg("theta", 1)
    c.step()
    assert k.theta_a[1] == 65535             # saturates at uint16


def test_rule4_cfg_space6():
    c, k = core(4)
    k.cfg_write(dbg(6, 2), 0x1234)
    assert k.theta_a[2] == 0x1234 and k.cfg_read(dbg(6, 2)) == 0x1234
    for r, name in ((23, "plastic_axon_max"), (24, "th_inc"), (25, "th_decay_shift")):
        k.cfg_write(REG | r, 7)
        assert getattr(k.regs, name) == 7 and k.cfg_read(REG | r) == 7


# rule 5 ---------------------------------------------------------------------
def test_rule5_rewards_dropped_while_off():
    c, k = core(2, theta=1000)
    for le, rs in ((0, 0), (1, 0), (0, 1)):
        k.set_reg("learn_en", le)
        k.set_reg("rstdp_en", rs)
        c.reward(0, 0, 50)
        assert k.pending_reward == 0
    k.set_reg("learn_en", 1)
    k.set_reg("rstdp_en", 1)
    c.reward(0, 0, -70000)
    c.reward(0, 0, 3)
    v = (-69997) & 0xFFFFFFFF
    assert k.cfg_read(dbg(5, 1)) == v & 0xFFFF and k.cfg_read(dbg(5, 2)) == v >> 16


# address map ----------------------------------------------------------------
def test_debug_spaces_and_elig_access():
    c, k = core(4, learn_en=1, theta=1000)
    k.cfg_write(dbg(1, 3), 0x1FF)
    k.cfg_write(dbg(2, 1), 77)
    assert k.x_pre[3] == 0xFF and k.cfg_read(dbg(1, 3)) == 0xFF and k.cfg_read(dbg(2, 1)) == 77
    k.cfg_write(dbg(7, 0), 999)                  # unknown space: ignored, not a register write
    assert k.regs.theta == 1000 and k.cfg_read(dbg(7, 0)) == 0
    k.cfg_write(REG | (3 << 16) | 0, 5)          # space 3 is read-only, not theta
    assert k.regs.theta == 1000
    assert k.cfg_read(dbg(5, 0)) == k.lfsr.state
    k.cfg_write(REG | 22, 1)
    k.cfg_write(1 * 4 + 2, 0xFFF0)               # region 00, axon 1 neuron 2 -> elig
    assert k.elig[1, 2] == -16 and k.w[1, 2] == 0 and k.cfg_read(6) == 0xFFF0
    k.cfg_write(REG | 22, 0)
    k.cfg_write(6, 5)
    assert k.w[1, 2] == 5 and k.cfg_read(6) == 5
