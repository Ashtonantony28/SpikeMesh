# rtl/core: SpikeMesh neurocore (SG2 inference, SG6 on-chip learning, SPEC v0.2)

Files:

| File | Content |
|---|---|
| `neurocore.sv` | The core. Port list as in SPEC §8, plus parameters `LANES` (default 8) and `ACC_W` (default `W_BITS + log2(N_AXONS) + 1`, SPEC §2), and `LEARN`:
  - 0 (default): inference only, no learning hardware.
  - 1: STDP. `rstdp_en` is ignored.
  - 2: STDP + R-STDP, which adds the eligibility memory.

  `ADAPT_TH` (default 1) builds the SPEC §9.4 adaptive threshold. With 0 there is no theta_a memory: regs 24/25 and space 6 read 0, and writes to them are accepted in 1 cycle and ignored. |
| `learn_engine.sv` | SPEC §6 learning (STDP, R-STDP). It is instantiated only when `LEARN > 0`. |
| `sram_1r1w.sv` | The only memory primitive: 1 write port and 1 registered read port on a single clock, with an optional per-lane write mask. It infers BRAM/LUTRAM on FPGA and can be replaced by a 1R1W SRAM macro wrapper for ASIC. Its contents are not reset. |

## Memories

All memories are `sram_1r1w` instances with synchronous reads.

| Memory | Depth × width | Notes |
|---|---|---|
| weights | `N_AXONS·N/LANES` × `LANES·W_BITS` | Row address `axon·G + group`, with `G = N/LANES`. CFG writes one lane through the write mask. |
| acc bank 0, 1 | `G` × `LANES·ACC_W` | One physical memory per parity (SPEC §3). The reset FSM clears both banks in `G` cycles. |
| v, bias, fan-out pointer | `N` × 16 | |
| refc | `N` × 4 | |
| fan-out dest / axon | `FANOUT_DEPTH` × 8 / × 10 | Word 0 stores `{dest_x, dest_y}`; word 1 stores `dest_axon`. |
| theta_a (ADAPT_TH=1) | `N` × 16 | SPEC §9.4. The SPEC leaves it undefined after reset; this core zeroes it during the reset clear, which then takes N cycles instead of G. |

Core registers are flip-flops. The SPEC §2 registers use their spec widths (theta 16, leak_shift 4, reset_mode 1, refrac 4, v_min 16). Learning registers 8..21 are stored as 16 bits and reset to the golden defaults: `lfsr_seed=1`, `w_min=-2^(W-1)`, `w_max=2^(W-1)-1`, all others 0. The v0.2 registers reset to 0: 23 `plastic_axon_max` (16 bits, every build), 24 `th_inc` (16 bits) and 25 `th_decay_shift` (4 bits, like `leak_shift`). Registers 24 and 25 exist only when ADAPT_TH=1.

## Microarchitecture

Three engines share the memories. Their port muxes are in one `always @*` block.

**Input engine** (FSM `IDLE / SPK / RDW / RDP`)

- **SPIKE** (2-stage pipeline). The accept cycle reads weight row group 0 and accumulator group 0 of bank `flit.parity`. The SPK state then reads groups 1..G-1, one per cycle. One cycle after each read, the write-back stage writes `acc[l] + sext(w[l])` for all LANES lanes. When the next read hits the group being written, the old value is forwarded. This can only happen when G = 1.
  - Throughput: **G = N/LANES cycles per spike** back-to-back. That is 32 for N=256/LANES=8 (measured 514 cycles for 16 spikes) and 8 for N=32/LANES=4 (130 cycles for 16 spikes).
- **CFG_WRITE**: 1 cycle. Writes to an out-of-range index are ignored. Weight data bits above W_BITS are ignored.
- **READ_REQ**: the read is issued on accept. In RDW the response is formed. In RDP it is pushed when the output FIFO has room. That is 3 cycles per read. The response is `{10, reply_x, reply_y, 1, 00000, addr[15:0], data}`. Weights read back sign-extended to 16 bits. refc, unused register bits, unmapped registers and out-of-range indices read as zero.
- **CTRL REWARD**: summed into the 32-bit `reward_acc` only while `learn_en` and `rstdp_en` are both set (SPEC §9.5); otherwise it is dropped. On LEARN<2 builds every reward is dropped. Other opcodes are consumed and ignored. READ_RESP flits arriving at a core are dropped.

**Update engine**

1. `update_start` latches `parity` and sets `upd_pending`.
2. The engine starts once nothing is still accumulating into that bank.
3. Stage R reads v/bias/refc/pointer for neuron n. At lane 0 it also reads the accumulator group. The group word is then held in the RAM output register for the other lanes.
4. Stage C runs SPEC §2 exactly: refractory, leak (`v>>>leak_shift`, none when the shift is 0), single sat16 of `v1 + sat16(acc) + bias`, threshold, subtract or zero reset, `v_min` clamp. The threshold and the subtract-reset use `theta_eff = min(32767, theta + theta_a[n])` (§9.4). theta_a is read in stage R alongside v. Stage C then writes it back for every neuron, refractory ones included: first the decay with a floor of 1, then, if the neuron fired, `+ th_inc` saturating at 65535. This runs whether or not `learn_en` is set. It writes v/refc, clears the accumulator group (at lane 0), and pushes `{n, base, count}` into a 4-entry spike FIFO when the neuron fires and count ≠ 0.
5. Stage R issues only while the spike FIFO has 2 or more free entries, so stage C never stalls.

Rate: 1 neuron/cycle.

**Emitter**

- It pops the spike FIFO and issues the first fan-out read in the same cycle. It then issues entries `base .. base+count-1` in ascending order, 1 per cycle. Fan-out addresses wrap modulo FANOUT_DEPTH.
- The next cycle forms the flit:
  - Core entry: `{00, dx, dy, ~p, 0, axon}`.
  - `dest_x == 15`: the host format `{00, 15, 0, ~p, 0, X, Y, 0, n}`, with dest_y forced to 0 (SPEC §4).
- Flits go into a 4-entry output FIFO. A read is issued only when `of_cnt + inflight + 1 ≤ 4`, so `out_ready` backpressure stalls the emitter, which stalls the update engine through the spike FIFO. Nothing is dropped. `out_flit` and `out_valid` come straight from the FIFO registers.

**UPDATE cycle counts** (no backpressure): **N + 5 cycles** when nothing fires (261 for N=256, 37 for N=32). Emission overlaps the neuron walk, so with every neuron firing one entry it is 263 / 39. In general it takes about `max(N, total fan-out flits) + 5` cycles.

**Concurrency rules** (`in_ready`)

During UPDATE, the core accepts SPIKE flits for the *other* bank (`parity ^ 1`) and CTRL flits. This keeps a mesh with recurrent or self fan-out from deadlocking: UPDATE output can loop back into the core while UPDATE is still running.

CFG and READ flits, and spikes for the bank being updated, are held off by head-of-line blocking until UPDATE ends. The host must not send CFG or READ while a STEP runs.

`in_ready` depends combinationally on `in_flit` and `in_valid`. That is legal valid/ready, but the router must not make `in_valid` depend on `in_ready`.

**LEARN** (`learn_engine.sv`, SPEC §6)

- `learn_start` latches a request. LEARN begins once no spike is mid-accumulation and UPDATE is idle. With `learn_en = 0`, or with `LEARN = 0`, it completes right away.
- **State**:
  - `x_pre[A]`: memory, 8 bit.
  - `x_post[N]`: memory, G × LANES·8.
  - Both trace memories are zeroed after reset (SPEC §5 resets traces): one x_pre entry and one x_post row per cycle, max(A, G) cycles, with `busy` high. The eligibility memory (R-STDP) is not cleared (AUDIT-1, open finding M2).
  - `pre_flag[A]`, `pre_seen[A]`, `post_flag[N]`: flip-flops. `pre_seen` is set when a spike is accepted while `learn_en` is set, and drives the x_pre bump. `pre_flag` is set as well, but only for a plastic axon (`plastic_axon_max = 0`, or `a < plastic_axon_max`). It drives depression and is what debug space 3 reads (§9.3). Both decisions are made when the spike arrives, as golden makes them. `post_flag[n]` is overwritten with `fire` for every neuron in UPDATE stage C, as golden does.
  - The 16-bit Galois LFSR: reloaded on every CFG write of `lfsr_seed` (0 becomes 1), and reset to 1.
  - LEARN=2 only: `e[a][n]` (int16), laid out like the weight memory.
- **Phases.** LEARN runs the SPEC order. Every phase drains its 1-cycle read→compute→write pipeline before the next phase starts.
  1. **DECAY**: x_pre and x_post decay, `x -= max(1, x >> k)` for x > 0 (§9.1). Skipped when both shifts are 0.
  2. **ELIG** (R-STDP only, when `tau_elig ≠ 0` or reward ≠ 0): one weight row per cycle. It computes `e -= e>>>tau_elig`, then, if reward ≠ 0, `w = clamp(w + ((r·e + 2^(rs-1)) >>> rs))`, the rounded form of §9.2 (no rounding term when rs = 0). This uses 8 lane-parallel 32×16 multipliers. The reward sum is cleared when it is applied.
  3. **DEP**: scans axons in ascending order, 1 cycle per axon. For each flagged axon it processes the row LANES weights per cycle. The LFSR advances LANES steps per cycle as an unrolled chain, with lane 0 first, which gives ascending n.
  4. **POT**: scans neurons in ascending order. For each flagged neuron it processes the column 1 weight per cycle, with axons ascending, 1 LFSR step each, and a lane-masked write. The write is suppressed for axons `a >= plastic_axon_max`, but their sr() draw still happens (§9.3).
  5. **BUMP**: trace bumps with usat8 (x_pre where `pre_seen` is set), then all flags are cleared.
- **Stochastic rounding** is exact for every 16-bit `stdp_shift`. The product is at most 24 bits, so `s ≥ 24` is evaluated as `s = 24`. The other shift amounts saturate the same way: trace shifts at 8, `tau_elig` at 15, `reward_shift` at 48. |r·e| < 2^47, so any shift ≥ 48 gives 0, even with the rounding term.
- **Cycles per LEARN:**

  ```
  2 + [max(A,G)+2 if tau_pre|tau_post] + [A·G+2 if R-STDP elig/reward]
    + (A+1 + #pre·G) + (N+1 + #post·A) + (max(A,G)+2)
  ```

  Measured at 32/32/LANES=4: 147–683 cycles for STDP. At 256/256/LANES=8 with R-STDP: 14.5k cycles with 166 pre-flags and 0 post-flags, 35.3k cycles with 144 pre and 84 post. Potentiation, at A cycles per firing neuron, dominates. Its column-major LFSR order is fixed by the spec.
- **Debug addresses** (SPEC §5 and §9.4). Region 11 `[19:16]` selects a sub-space, and `[15:0]` is the index:
  - 0: core registers, as before.
  - 1: `x_pre[a]`, read/write.
  - 2: `x_post[n]`, read/write.
  - 3: `pre_flag[a]`, read only.
  - 4: `post_flag[n]`, read only.
  - 5: index 0 is the LFSR state, index 1 the reward sum [15:0], index 2 the reward sum [31:16]. Read only.
  - Register 22 bit 0 (LEARN=2): region-00 reads and writes address `e[a][n]` instead of `w[a][n]`.
  - 6: `theta_a[n]`, read/write. This space is a core feature, not a learning one; it reads 0 when ADAPT_TH=0.
  - Spaces 1–5 read 0 when `LEARN=0`. Writes to spaces 3–5 and 7–15 are ignored. Out-of-range indices read 0. Register 22 exists only in LEARN=2 builds; in other builds it reads 0 and its writes are ignored. Golden does not gate it, so the tests use it only on LEARN=2 builds.
- **Limits.** The reward sum is held as a 32-bit int, where golden's is unbounded. `w_min`/`w_max` must lie inside the W_BITS range, because a clamped weight is stored as W_BITS bits.

**busy** = reset clear | input FSM not IDLE | write-back pending | LEARN pending/running | `update_start` | `learn_start` | UPDATE pending/active | output FIFO non-empty.

## Verification

`python -m pytest -q tests/core` runs 65 tests in about 195 s, all against golden SPEC v0.2. The SG2 inference tests are below; the SG6 learning tests are listed after them.

- **cocotb + Icarus.** The tests use `tests/core/core_tb.py`, a single-coroutine valid/ready driver. It applies random `in_valid` gaps and random `out_ready` backpressure, and loops self-addressed output spikes back into the input. Each test checks timestep-exact against `golden.spikemesh.Core`:
  - the multiset of output flits per timestep;
  - READ_REQ readback of v/refc/weights.

  Configurations:
  - N=A=32/LANES=4: 11 seeds across mixed, sparse, sat, refrac, leakneg, reset0 and reset1, plus one heavy-backpressure run.
  - Full CFG readback of every region, register reset values, and cycle measurement.
  - W_BITS=4 (LANES=8).
  - Edge cases: LANES=N (G=1, forwarding path) and LANES=1.
  - N=A=256/LANES=8: sparse random, plus `accsat` (every axon twice, |acc| > 32767 in both signs), plus cycle counts.
- **Verilator.** cocotb 2.1 needs Verilator 5.036 or newer, and 5.020 is installed. Instead, `vl_replay.py` writes a golden-generated command trace that `tb_replay.sv` replays: 4 small seeds × 40 steps, and 2 runs at 256/8 × 8 steps.
- **Learning (SG6)**, `core_tb.test_learn`, on LEARN=2/1 builds:
  - Randomized multi-step runs with random spikes and loopback, rewards before LEARN and during UPDATE, random initial traces and eligibility written through the debug addresses, and random learning registers.
  - Every timestep checks spike flits, plus x_pre, x_post, both flag arrays, the LFSR state and the reward sum. Every 3rd step checks all weights and all eligibility against golden.
  - The `learn_start` pulse sometimes lands while a spike is still accumulating.
  - Modes: stdp, rstdp, stdp_s0/rstdp_s0 (`stdp_shift=0`), *_ext (wide register values), and *_bigshift (`stdp_shift` 24..63, `reward_shift` ≥ 64).
  - Configs: 32/32/LANES=4 with 10 seeds; W4/LANES=8; LANES=N; LANES=1; N=64/A=16/LANES=2; LEARN=1; one Icarus run at 256/256/LANES=8. learn_en=0 on a LEARN=2 build reruns the inference and readback tests.
  - Verilator replay adds 3 small runs × 40 steps and 2 runs at 256/8 × 10 steps, each with a full readback of weights and eligibility.
- **SPEC v0.2 (§9)**:
  - Every random network writes `theta_a` for every neuron, plus random `th_inc` / `th_decay_shift`.
  - The `adapt` mode (5 seeds; LANES 4, 1 and N) stresses the corners, and a coverage check requires each of them to occur: theta + theta_a > 32767, firing with theta_a > 0, the decay floor, and `th_inc` saturation.
  - Learning configs draw `plastic_axon_max` from 0, values inside [1, A), A, and values above A. Readback of the flags and traces checks pre_flag versus pre_seen.
  - test_learn switches `rstdp_en` off and on between steps, and `learn_en` too when no spike is pending, so some rewards arrive while learning is off.
  - test_random on LEARN builds checks that the reward sum is still 0 after rewards were sent with learning off.
  - Readback covers registers 23–25 and every theta_a entry, plus out-of-range indices.
  - ADAPT_TH=0 builds (inference and LEARN=2) check that theta_a and registers 24/25 are ignored.
  - Test `reward_shift` ≤ 63, because golden v0.2's rounding term `1 << (rs-1)` is int64.
- **v0.2 self-check.** 25 hand mutations of the new logic, 24 killed:
  - Decay floor: removed, and applied when x = 0.
  - Reward rounding: no rounding term, and the term one bit too high.
  - Non-plastic axons:
    - POT writes their weight or eligibility.
    - `>` instead of `>=` against `plastic_axon_max`, in the POT gate and in the arrival check.
    - `plastic_axon_max` ignored when setting pre_flag.
    - The bump uses pre_flag instead of pre_seen.
    - pre_seen is not cleared.
  - Reward gating: removed entirely, and checking `rstdp_en` only.
  - Adaptive threshold:
    - theta_a ignored, and theta_eff not saturated.
    - Reset subtracts theta instead of theta_eff.
    - No decay floor, no `th_inc` saturation, and increment before the decay.
    - Refractory neurons skip the decay.
    - Out-of-range reads and writes of theta_a.
    - `th_decay_shift` corrupted.
    - An ADAPT_TH=0 build storing `th_inc`.

  The survivor caps `reward_shift` at 47 instead of 48. It differs only at r = −2^31, e = −32768, which a 32-bit reward sum can just reach and the tests do not.
- **Learning self-check.** 20 hand mutations of learning logic. 18 are killed: LFSR taps, seed-0 rule, sr compare, shift saturations, DEP lane order, POT LFSR step, clamp order, eligibility arithmetic shift, reward-zero skip, reward clear, decay skip, bump saturation, flag clear, POT trace source, R-STDP writing weights, weight sign extension, and LEARN waiting for accumulation. 2 are equivalent under the protocol: post_flag OR versus overwrite, since LEARN always clears flags before the next UPDATE; and the extra drain wait, since the scan and pipeline already give the one-cycle gap.
- **Self-check.** Hand mutations of the RTL are all caught: leak logical shift, parity not flipped, forwarding off, output FIFO overflow, missing emitter credit, spike-FIFO headroom, v_min, refractory v, both acc saturations, reset_mode, host src swap, and the in_ready UPDATE gate.

Lint: `verilator --lint-only -Wall rtl/core/neurocore.sv rtl/core/learn_engine.sv rtl/core/sram_1r1w.sv` is clean for LEARN=0/1/2 at all tested parameter sets. The only waivers are UNUSEDSIGNAL on sliced 32-bit index temporaries, the flit dest bits, and the tie-offs of unused build options.

## Synthesis (N=A=256, LANES=8, FANOUT_DEPTH=1024, Yosys 0.33, `synth_xilinx -family xcup -flatten`)

SPEC v0.2 RTL. The v0.1 numbers are in parentheses.

| Resource | LEARN=0, ADAPT_TH=0 | LEARN=0 (default ADAPT_TH=1) | LEARN=1 (STDP) | LEARN=2 (+R-STDP) |
|---|---|---|---|---|
| LUTs | 1781 | 1977 (1625) | 8272 (7653) | 12211 (11038) |
| FFs | 885 | 905 (865) | 1857 (1656) | 1924 (1627) |
| CARRY4 | 78 | 92 (76) | 651 (651) | 860 (736) |
| DSP48E2 | 0 | 0 | 8 (10) | 24 (24) |
| RAMB36E2 | 16 | 16 | 16 | 48 (+1 Mbit eligibility) |
| RAMB18E2 | 5 | 6 (5) | 6 (5) | 6 (5) |
| RAM32M16 / RAM64M8 | 26 / 4 | 26 / 4 | 34 / 12 | 34 / 12 |

v0.2 costs:
- **Adaptive threshold** (core feature, on by default): about 200 LUTs, 20 FFs and one RAMB18 per core.
- **Always present, even with ADAPT_TH=0**: about 150 LUTs and 20 FFs. This covers register 23 and the arrival compare, the reward gate, and the wider reset counter.
- **Learning builds**: about +600 LUTs (LEARN=1) and +1200 LUTs (LEARN=2). This covers the decay floor, the `pre_seen` flags (256 FFs), the POT write gate and the 50-bit rounded reward path.

LEARN=0 with ADAPT_TH=0 is the smallest core. It is within about 10 % of the SG2 inference-only core. The learning logic is mostly the lane-parallel stochastic-rounding and clamp datapath, the flag muxes, and, for LEARN=2, the 8 reward multipliers.

At SG2, generic `synth` with `sram_1r1w` blackboxed (logic only) gave 5961 cells for LEARN=0: 870 flip-flops and about 5080 gates/muxes.
