# SpikeMesh architecture specification (v0.2)

v0.2 (2026-11-07) changes, all from the on-chip learning experiments (`learn/README.md`): trace decay floor, rounded reward, non-plastic axons, adaptive threshold, rewards dropped while R-STDP is off. See §9.

This document is the single contract shared by the PyTorch trainer (`train/`), the integer golden model (`golden/`), the compiler (`compiler/`) and the RTL (`rtl/`). If any of them disagrees with this file, that component is wrong. Changes to this file need an ADR in `docs/adr/`.

All arithmetic is integer. "sat16(x)" clamps to [-32768, 32767]; "sat8(x)" clamps to [-128, 127]; "usat8(x)" clamps to [0, 255]. ">>>" is an arithmetic (floor) right shift. Python's `>>` on ints is the same thing.

## 1. Parameters (build time)

| Name | Default | Meaning |
|---|---|---|
| `MESH_X`, `MESH_Y` | 2, 2 | Mesh size. Core (x, y), 0 ≤ x < MESH_X, 0 ≤ y < MESH_Y. Max 15 × 15. |
| `N_NEURONS` | 256 | Neurons per core (power of two, ≤ 1024). |
| `N_AXONS` | 256 | Input axons per core (power of two, ≤ 1024). |
| `W_BITS` | 8 | Weight width, signed. 8 or 4. |
| `FANOUT_DEPTH` | 1024 | Entries in a core's fan-out memory (max 1024: the pointer base is 10 bits). |

## 2. Neuron model (ADR-001)

Per-core configuration registers (shared by all neurons in the core):

| Reg | Width | Meaning |
|---|---|---|
| `theta` | int16, > 0 | Firing threshold |
| `leak_shift` | uint4 | 0 = no leak, else leak term is `v >>> leak_shift` |
| `reset_mode` | 1 bit | 0 = subtract `theta`, 1 = reset to zero |
| `refrac` | uint4 | Refractory timesteps after a spike |
| `v_min` | int16 | Lower clamp on membrane after update |

Per-neuron state and parameters: `v` (int16), `refc` (uint4 refractory counter), `bias` (int16).

Per-neuron, per-timestep input: `acc` = exact (non-saturating) sum of the weights of every synapse that received a spike for this timestep. Hardware must make the accumulator wide enough that it cannot overflow: `W_BITS + log2(N_AXONS) + 1` bits, which is exact when each axon receives at most one spike per timestep. The compiler guarantees that (each axon has one source); hand-built traffic that sends repeated spikes to one axon in one timestep must raise the RTL `ACC_W` parameter.

Update of one neuron at timestep t (exactly this order):

```
if refc > 0:
    refc = refc - 1
    spike = 0                      # acc is discarded, v unchanged
else:
    leak = 0 if leak_shift == 0 else (v >>> leak_shift)
    v1   = v - leak                # cannot overflow
    v2   = sat16(v1 + sat16(acc) + bias)
    if v2 >= theta:
        spike = 1
        v = (v2 - theta) if reset_mode == 0 else 0
        refc = refrac
    else:
        spike = 0
        v = max(v2, v_min)
acc = 0
```

Note `sat16(v1 + sat16(acc) + bias)` is evaluated in full precision and saturated once.

## 3. Timestep schedule (ADR-002)

Each core has two accumulator banks indexed by timestep parity `p = t & 1`. Every spike packet carries the parity of the timestep in which it must be *integrated*.

One call of `STEP(t)` does, in order, for the whole chip:

1. **Quiesce.** Wait until every packet tagged for timestep t (host input events for t, and spikes emitted during `UPDATE(t-1)`) has been accumulated into bank `p`.
2. **LEARN(t)** (only cores with learning enabled; see §6).
3. **UPDATE(t).** Every core updates every neuron using bank `p` and clears it. A neuron that spikes sends one packet per fan-out entry, tagged with parity `p ^ 1` (it is integrated at t+1).

The host may inject input events for t+1 (parity `p ^ 1`) at any time after `UPDATE(t)` has started. Consequences that the trainer must model:

- A spike from layer L at timestep t is integrated by layer L+1 at timestep t+1. Every layer adds one timestep of latency.
- Input events for timestep t are integrated at timestep t.

Quiescence detection in RTL: a registered global OR of all router-FIFO-non-empty, link-valid and core-busy signals; the chip is quiescent after that OR has been low for `QUIESCE_CYCLES` consecutive cycles (≥ the longest single-hop pipeline delay, default 4). Formal properties check that quiescence cannot be declared with a packet in flight.

## 4. Packets (ADR-003)

Flits are 48 bits, single-flit packets.

```
[47:46] type     00 SPIKE, 01 CFG_WRITE, 10 READ_REQ / READ_RESP (see bit 45 of payload), 11 CTRL
[45:42] dest_x   (15 = host / off-mesh port)
[41:38] dest_y
[37:0]  payload
```

| Type | Payload |
|---|---|
| SPIKE (to core) | `[37]` parity, `[9:0]` axon id |
| SPIKE (to host, dest_x = 15, dest_y = 0) | `[37]` parity, `[29:26]` src_x, `[25:22]` src_y, `[9:0]` neuron id |
| CFG_WRITE | `[37:16]` address (22 bits, §5), `[15:0]` data |
| READ_REQ | `[37]` = 0, `[36:15]` address, `[14:11]` reply_x, `[10:7]` reply_y |
| READ_RESP | `[37]` = 1, `[31:16]` low 16 bits of address, `[15:0]` data |
| CTRL | `[7:0]` opcode: 0x01 REWARD (payload `[23:8]` = signed reward), others reserved |

Routing: dimension-order XY (route X first, then Y). Host port is the west side of core (0, 0); a packet with `dest_x = 15` is routed west out of (0, 0). Host-injected packets enter at (0, 0)'s west port.

## 5. Core address map (ADR-004)

22-bit word address. `[21:20]` selects a region.

| Region | `[21:20]` | Address bits | Data |
|---|---|---|---|
| Weights | 00 | `[19:0]` = `axon * N_NEURONS + neuron` | `[W_BITS-1:0]` signed weight |
| Neuron | 01 | `[19:2]` = neuron, `[1:0]` field: 0 `v`, 1 `bias`, 2 `refc`, 3 fan-out pointer | field value; pointer = `{count[5:0], base[9:0]}` |
| Fan-out | 10 | `[19:0]` = entry index × 2 + word | word 0 = `{dest_x[3:0], dest_y[3:0], 8'b0}`, word 1 = `{6'b0, dest_axon[9:0]}` |
| Core regs | 11 | `[7:0]` = register | see below |

Core registers: 0 `theta`, 1 `leak_shift`, 2 `reset_mode`, 3 `refrac`, 4 `v_min`, 8 `learn_en`, 9 `stdp_a_plus`, 10 `stdp_a_minus`, 11 `stdp_shift`, 12 `tau_pre_shift`, 13 `tau_post_shift`, 14 `inc_pre`, 15 `inc_post`, 16 `lfsr_seed`, 17 `w_min`, 18 `w_max`, 19 `rstdp_en`, 20 `tau_elig_shift`, 21 `reward_shift`. Unused bits read as zero.

Weight writes ignore data bits above `W_BITS` (the compiler writes them as zero).

Reset values: only control state (FSMs, accumulator banks, learning flags and traces, LFSR, core registers) is reset by `rst_n`. Weights, neuron `v`/`bias`/`refc`, fan-out pointers and fan-out entries are memories with undefined contents after reset; the host must write every location it relies on, including fan-out pointer = 0 for unused neurons. (The golden model starts them at zero.)

Region 11 debug spaces (`[19:16]` selects; space 0 is the core registers above and requires `[19:16] = 0`): 1 `x_pre[a]` (r/w), 2 `x_post[n]` (r/w), 3 `pre_flag[a]` (r), 4 `post_flag[n]` (r), 5 index 0 LFSR state, 1/2 low/high half of the pending reward sum (r). Register 22 bit 0 (R-STDP builds only): region-00 accesses go to the eligibility memory instead of the weights.

Implementation limits: the pending reward sum is 32-bit signed; `w_min`/`w_max` must fit in `W_BITS`; `stdp_shift` ≤ 63 in tests (the golden mask is int64).

A neuron with fan-out count 0 sends nothing. A neuron's fan-out entries are `base .. base + count - 1`, sent in ascending order.

## 6. On-chip learning (ADR-005)

State per learning core: `x_pre[a]` (uint8, per axon), `x_post[n]` (uint8, per neuron), `pre_flag[a]`, `post_flag[n]` (1 bit), one 16-bit Galois LFSR (taps 0xB400, seeded from `lfsr_seed`, seed 0 is replaced by 1).

- `pre_flag[a]` is set when a SPIKE for timestep t arrives on axon a (it is the presynaptic spike).
- `post_flag[n]` is set when neuron n spikes during `UPDATE(t-1)`.

Stochastic rounding of a product: `sr(m) = (m >> s) + (1 if (lfsr_next() & ((1 << s) - 1)) < (m & ((1 << s) - 1)) else 0)` where `s = stdp_shift` and `lfsr_next()` advances the LFSR once and returns its new value. When `s = 0`, the LFSR still advances and the rounding term is 0.

`LEARN(t)`, executed in this exact order:

1. Decay all traces: `x_pre[a] -= x_pre[a] >> tau_pre_shift` (skip if shift = 0), same for `x_post[n]` with `tau_post_shift`.
2. Depression: for each axon a with `pre_flag[a]` set, ascending a; for each neuron n, ascending: `w[a][n] = clamp(w[a][n] - sr(a_minus * x_post[n]), w_min, w_max)`.
3. Potentiation: for each neuron n with `post_flag[n]` set, ascending n; for each axon a, ascending: `w[a][n] = clamp(w[a][n] + sr(a_plus * x_pre[a]), w_min, w_max)`.
4. Trace bumps: `x_pre[a] = usat8(x_pre[a] + inc_pre)` where `pre_flag[a]`; `x_post[n] = usat8(x_post[n] + inc_post)` where `post_flag[n]`. Clear all flags.

`sr()` is called once per weight visited in steps 2 and 3, in the order above, even when the product is zero. That fixes the LFSR sequence.

Learning in LEARN(t) runs after the weights for timestep t were already used for accumulation. Accumulation always uses weights as they were at the end of the previous LEARN.

**R-STDP (`rstdp_en = 1`).** Steps 2 and 3 write the signed delta into an eligibility memory `e[a][n]` (int16, saturating) instead of the weight. Additionally at the start of every LEARN: `e -= e >>> tau_elig_shift` for every synapse. When a CTRL REWARD packet with value r arrives, at the next LEARN (after step 1, before step 2) every weight gets `w = clamp(w + ((r * e) >>> reward_shift), w_min, w_max)`. Multiple rewards before one LEARN are summed.

## 7. Host interface

The host (cocotb testbench, Verilator harness, or FPGA host) writes CFG packets, injects SPIKE packets, issues STEP, and collects host-bound SPIKE packets. On the FPGA the host link is a 48-bit AXI-Stream in each direction plus a `step` / `step_done` handshake; UART framing is 6 bytes per flit.

## 8. RTL module interfaces (ADR-006)

All modules: single clock `clk`, active-low synchronous reset `rst_n`. Flit streams use valid/ready (a transfer happens when both are high in the same cycle). Router-to-router links use credit-based flow control internally; the core's local port is plain valid/ready.

`neurocore #(X, Y, N_NEURONS, N_AXONS, W_BITS, FANOUT_DEPTH)`:

| Port | Dir | Meaning |
|---|---|---|
| `in_valid/in_ready/in_flit[47:0]` | in | Flits addressed to this core (from the router's local output) |
| `out_valid/out_ready/out_flit[47:0]` | out | Flits from this core (to the router's local input) |
| `parity` | in | Current timestep parity p (stable during a STEP) |
| `learn_start` | in | 1-cycle pulse: run LEARN(t) |
| `update_start` | in | 1-cycle pulse: run UPDATE(t) on bank `parity`, emit spikes tagged `parity ^ 1` |
| `busy` | out | High while the core has any work pending (input flit being processed, LEARN, UPDATE, or output flits not yet accepted) |

The core accepts SPIKE, CFG_WRITE, READ_REQ and CTRL flits. During UPDATE it must keep accepting SPIKE flits for parity `p ^ 1` and CTRL flits (otherwise recurrent or self fan-out can deadlock the mesh); it may stall CFG_WRITE, READ_REQ and SPIKE flits for parity `p` until UPDATE ends. Hosts must not send CFG_WRITE or READ_REQ during a STEP. `in_ready` may depend combinationally on `in_flit`, so upstream `in_valid` must not depend on `in_ready`.

`router #(X, Y)`: five ports (N, S, E, W, L). XY routing per §4. Input FIFO depth `FIFO_DEPTH` (default 4). Round-robin output arbitration. A packet is never dropped or duplicated.

`spikemesh_top #(MESH_X, MESH_Y, ...)`: the mesh, the tick controller and the host port.

| Port | Dir | Meaning |
|---|---|---|
| `host_in_valid/ready/flit` | in | Host → chip, enters core (0,0)'s router west port |
| `host_out_valid/ready/flit` | out | Chip → host, leaves core (0,0)'s router west port |
| `step` | in | 1-cycle pulse: run STEP(t) |
| `step_done` | out | 1-cycle pulse when UPDATE(t) has finished in every core |
| `t_parity` | out | Current p |

Tick controller sequence on `step`: wait for quiescence (§3) → pulse `learn_start` → wait until no core is busy → pulse `update_start` → wait until no core is busy → pulse `step_done`, flip parity. Spikes emitted during UPDATE may still be in flight at `step_done`; the next STEP's quiesce phase absorbs them.


## 9. v0.2 learning changes (ADR-007)

These override the matching text in §2, §5 and §6.

1. **Decay floor.** Every shift decay in §6 (`x_pre`, `x_post`) and the adaptive threshold below uses `x -= max(1, x >> k)` when `x > 0` and `k > 0` (so a trace reaches 0). Eligibility decay (signed) is unchanged: `e -= e >>> tau_elig_shift`.
2. **Rounded reward.** R-STDP weight change is `(r * e + (1 << (reward_shift - 1))) >>> reward_shift` when `reward_shift > 0`, else `r * e`.
3. **Non-plastic axons.** Core register 23 `plastic_axon_max` (uint16, reset 0). When non-zero, axons `a >= plastic_axon_max` are fixed: their `pre_flag` is never set (no depression row), and in potentiation they are still visited and `sr()` is still drawn (LFSR order unchanged) but their weight (or eligibility) is not written. Their `x_pre` trace still updates.
4. **Adaptive threshold.** Per-neuron `theta_a[n]` (uint16, reset value undefined like other neuron memories; the host writes it). Core registers 24 `th_inc` (uint16, reset 0) and 25 `th_decay_shift` (uint4, reset 0). In UPDATE (§2) the comparison and subtract-reset use `theta_eff = min(32767, theta + theta_a[n])`. After the spike decision, for every neuron (including refractory ones): first decay `theta_a` per rule 1 with `th_decay_shift`, then if the neuron spiked, `theta_a = min(65535, theta_a + th_inc)`. This works whether or not `learn_en` is set. Readback / write: region 11 debug space 6, index n.
5. **Rewards while off.** A CTRL REWARD that arrives while `learn_en = 0` or `rstdp_en = 0` is dropped.
