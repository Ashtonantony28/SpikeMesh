# rtl/core: SpikeMesh neurocore (SG2, inference)

Files:

| File | Content |
|---|---|
| `neurocore.sv` | The core. Port list as in SPEC §8, plus parameters `LANES` (default 8) and `ACC_W` (default `W_BITS + log2(N_AXONS) + 1`, SPEC §2). |
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

Core registers are flip-flops. The SPEC §2 registers use their spec widths (theta 16, leak_shift 4, reset_mode 1, refrac 4, v_min 16). Learning registers 8..21 are stored as 16 bits and reset to the golden defaults: `lfsr_seed=1`, `w_min=-2^(W-1)`, `w_max=2^(W-1)-1`, all others 0.

## Microarchitecture

Three engines share the memories. Their port muxes are in one `always @*` block.

**Input engine** (FSM `IDLE / SPK / RDW / RDP`)

- **SPIKE** (2-stage pipeline). The accept cycle reads weight row group 0 and accumulator group 0 of bank `flit.parity`. The SPK state then reads groups 1..G-1, one per cycle. One cycle after each read, the write-back stage writes `acc[l] + sext(w[l])` for all LANES lanes. When the next read hits the group being written, the old value is forwarded. This can only happen when G = 1.
  - Throughput: **G = N/LANES cycles per spike** back-to-back. That is 32 for N=256/LANES=8 (measured 514 cycles for 16 spikes) and 8 for N=32/LANES=4 (130 cycles for 16 spikes).
- **CFG_WRITE**: 1 cycle. Writes to an out-of-range index are ignored. Weight data bits above W_BITS are ignored.
- **READ_REQ**: the read is issued on accept. In RDW the response is formed. In RDP it is pushed when the output FIFO has room. That is 3 cycles per read. The response is `{10, reply_x, reply_y, 1, 00000, addr[15:0], data}`. Weights read back sign-extended to 16 bits. refc, unused register bits, unmapped registers and out-of-range indices read as zero.
- **CTRL REWARD**: summed into `reward_acc`, which SG6 will consume. Other opcodes are consumed and ignored. READ_RESP flits arriving at a core are dropped.

**Update engine**

1. `update_start` latches `parity` and sets `upd_pending`.
2. The engine starts once nothing is still accumulating into that bank.
3. Stage R reads v/bias/refc/pointer for neuron n. At lane 0 it also reads the accumulator group. The group word is then held in the RAM output register for the other lanes.
4. Stage C runs SPEC §2 exactly: refractory, leak (`v>>>leak_shift`, none when the shift is 0), single sat16 of `v1 + sat16(acc) + bias`, threshold, subtract or zero reset, `v_min` clamp. It writes v/refc, clears the accumulator group (at lane 0), and pushes `{n, base, count}` into a 4-entry spike FIFO when the neuron fires and count ≠ 0.
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

**LEARN**

- `learn_start` with `learn_en = 0` does nothing: `busy` is high for that cycle and the next. With `learn_en = 1` LEARN is still a 1-cycle no-op until SG6.
- Hooks for the SG6 learning engine, present but not used:
  - `hook_pre_valid/axon`: a spike accepted while `learn_en` is set.
  - `hook_post_valid/neuron`: a neuron fired in stage C while `learn_en` is set.
  - `learn_busy`: the place for the learn FSM.
  - The weight memory's write port is used only by CFG today, so the learn engine can take it.

**busy** = reset clear | input FSM not IDLE | write-back pending | learn | `update_start` | `learn_start` | UPDATE pending/active | output FIFO non-empty.

## Verification

`python -m pytest -q tests/core` runs 28 tests in about 65 s:

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
- **Self-check.** Hand mutations of the RTL are all caught: leak logical shift, parity not flipped, forwarding off, output FIFO overflow, missing emitter credit, spike-FIFO headroom, v_min, refractory v, both acc saturations, reset_mode, host src swap, and the in_ready UPDATE gate.

Lint: `verilator --lint-only -Wall rtl/core/neurocore.sv rtl/core/sram_1r1w.sv` is clean at all tested parameter sets. The only waivers are UNUSEDSIGNAL on sliced 32-bit index temporaries, the flit dest bits, and the SG6 hooks.

## Synthesis (N=A=256, LANES=8, FANOUT_DEPTH=1024, Yosys 0.33)

`synth_xilinx -family xcup -flatten` gives:

| Resource | Count |
|---|---|
| RAMB36E2 | 16 (weights) |
| RAMB18E2 | 5 |
| RAM32M16 / RAM64M8 | 26 / 4 (accumulator banks and small RAMs as LUTRAM) |
| LUTs | 1618 (LUT1–6) |
| CARRY4 | 76 |
| FFs | 865 |

Estimated LCs: 1115.

Generic `synth` with `sram_1r1w` blackboxed (logic only) gives 5961 cells: 9 SRAM instances, 870 flip-flops, and about 5080 gates/muxes. Total memory is about 565 Kbit, of which weights are 512 Kbit.
