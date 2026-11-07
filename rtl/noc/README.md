# rtl/noc + rtl/top: SpikeMesh network, tick controller, chip top (SG3)

| File | Module | Role |
|---|---|---|
| `noc_fifo.sv` | `noc_fifo` | First-word-fall-through FIFO, state = (rptr, count) |
| `router.sv` | `router` | 5-port XY router (SPEC §4, §8) |
| `noc_host_in.sv` / `noc_host_out.sv` | `noc_host_in` / `noc_host_out` | valid/ready <-> credit adapters for the host on (0,0)'s W port |
| `mesh.sv` | `mesh` | `MESH_X x MESH_Y` routers, link wiring, host adapters |
| `../top/tick_ctrl.sv` | `tick_ctrl` | STEP sequencer and quiescence detector (SPEC §3, §8) |
| `../top/spikemesh_top.sv` | `spikemesh_top` | mesh + tick controller + one `neurocore` per node |

## Coordinates and ports

Core (x, y), index `n = y*MESH_X + x`. **x grows to the east, y grows to the north.**
Router port index: 0 = N (to (x, y+1)), 1 = E (to (x+1, y)), 2 = S (to (x, y-1)), 3 = W (to (x-1, y)), 4 = L (the core).
"Input N" carries flits arriving from the north neighbour.

## Router microarchitecture

```
 in_valid/in_flit[p] --> [input FIFO p, FIFO_DEPTH] --head--> route(head) --req--> [per-output RR arbiter] --> [link register o] --> out_valid/out_flit[o]
        credit_out[p] <-- (registered pop pulse)                                       credits[o] counter <-- credit_in[o]
```

* **Input FIFOs**, one per port, `FIFO_DEPTH` deep (default 4), first-word-fall-through.
* **Route computation** on the FIFO head, combinational, dimension-order XY:
  * `dest_x == 15` (host): W while x > 0, then S while y > 0, then out of (0,0)'s W port. `dest_y` is ignored for host flits (SPEC clarification: host-bound spikes carry `dest_y = 0`).
  * Off-mesh destination (`dest_x` in [MESH_X, 14] or `dest_y >= MESH_Y`, with `dest_x != 15`): **dropped** at the first router that sees it (its ingress router). The FIFO head is popped without using an output, the sticky `err` output is set and `drop_cnt` (8 bit, saturating) counts it. The mesh never hangs on such a flit. `mesh.err` / `spikemesh_top.err` OR all routers' flags, and `drop_cnt` is summed (saturating).
  * Otherwise E/W until x matches, then N/S until y matches, then L.
* **Arbitration**: each output has a round-robin pointer over the 5 inputs; it grants the first requesting input at or after the pointer and moves the pointer to one past the winner. Every input requests one output only, so a separable allocator needs no input-side arbitration. Arbitration is work conserving (formal P6).
* **Flow control**:
  * N/E/S/W links: **credit based**. The sender holds `credits[o]` (reset to FIFO_DEPTH, the receiver's FIFO depth), sends only when it is non-zero, and the receiver pulses `credit_out` for one cycle for each flit it pops. `out_valid` is a one-cycle pulse per flit.
  * L port (core): **valid/ready** both ways. `l_in_ready = !full`; the local output is a link register that holds until `l_out_ready`. `out_valid` (the core's `in_valid`) is a register and never depends on `in_ready`. That matters because the real core's `in_ready` depends combinationally on `in_flit`.
  * Host: `noc_host_in` passes valid/ready through with a credit counter for (0,0)'s W FIFO (`host_in_ready = credits != 0`, no extra latency). `noc_host_out` is a FIFO_DEPTH-deep FIFO that returns credits to (0,0)'s W output and presents valid/ready to the host.
* **busy** = any input FIFO non-empty OR any link register valid. A flit moves from a link register (cycle c) into the next FIFO (cycle c+1), so some `busy` covers it in every cycle it is in the network (formal P10: `busy == (flits held != 0)`).

### Latency and throughput

* **2 cycles per router hop**: a flit written into an input FIFO at edge c is arbitrated in cycle c and sits in the output link register from edge c+1; the next router's FIFO captures it at edge c+2 (cocotb `hop_latency`, L to L = 2 cycles).
* A core-to-core path through h routers (including source and destination) takes 2h cycles from `out_valid&&out_ready` at the source core to `in_valid` at the destination core. The host input adds 0 cycles; the host output FIFO adds 1.
* Credit round trip = 4 cycles, so `FIFO_DEPTH = 4` sustains 1 flit per cycle per link (cocotb `arbitration_fairness`: 300 flits in 300 cycles). `FIFO_DEPTH = 2` works (it is tested on 3x2) at half the peak link rate.
* Yosys `synth -flatten` of one router (FIFO_DEPTH 4): about 5.4k generic cells.

### Ordering

Delivery is **in order per (source, destination) pair**: XY routing is deterministic (one path per pair), FIFOs keep order, and an output arbiter never reorders flits from the same input. The mesh stress test checks this.

### Deadlock freedom

1. **Routing deadlock**: XY routing permits no Y-to-X turn, so the channel dependency graph of the router-to-router links is acyclic (Dally & Seitz). Formal P7/P11 prove per router, with assume-guarantee over neighbours, that a flit that arrived travelling in Y never turns into X and that each sent flit is XY-consistent for its receiver.
2. **The one Y-to-X turn** is a host-bound flit that comes south into (0,0) and leaves west. That west channel is the host-out FIFO, a **sink** with no outgoing dependency in the mesh, so it cannot close a cycle. Host-to-host loopback (W in to W out at (0,0)) also ends in that sink.
3. **Dropped flits** are consumed at once and never wait on an output.
4. **Consumption assumption** (outside the network): the host must eventually drain `host_out`, and every core must eventually accept flits on its local output. Otherwise the mesh backs up. That is a stall, not a cycle.
5. **Message-dependent deadlock** (core-level, see "Spec notes"): if a core stops accepting input until its own output drains, and its output waits on another core that does the same, the system can deadlock even though the network cannot. The real core keeps accepting SPIKE (other parity) and CTRL flits during UPDATE, which breaks this cycle for spike traffic. CFG/READ traffic must not be sent mid-STEP.

## Tick controller (`rtl/top/tick_ctrl.sv`)

`step` (accepted only when idle) → QUIESCE → `learn_start` pulse → wait until no core is busy → `update_start` pulse → wait until no core is busy → `step_done` pulse, and `t_parity` flips on the same edge.

* **Quiescence**: `busy_r <= net_busy | (|core_busy)` (registered OR). `quiet_cnt` counts consecutive low `busy_r` cycles. Quiescence is declared in cycle c iff `quiet_cnt >= QUIESCE_CYCLES && !busy_r`, i.e. the raw OR was low in each of cycles c-1 … c-QUIESCE_CYCLES-1 (formal T2). In the top, `net_busy = mesh.busy | host_in_valid`.
* **Start guard**: `core_busy` is ignored for `START_GUARD` (2) cycles after each start pulse. Contract: a core raises `busy` within 2 cycles of seeing the pulse, or finishes by then.
* `step_done` is combinational (`WAIT_UPD && guard==0 && !core_busy`), so formal T1 (never while a core is busy) holds in the same cycle.
* `step` while running is ignored and sets the sticky `step_err` (in `spikemesh_top.err`).
* **`drained`** (top output, level): no STEP running AND the quiescence condition holds. After `step_done` it rises once every spike from UPDATE(t) has landed in a core (and been accumulated) or in the host-out FIFO. The host should also wait for `host_out_valid == 0` before it reads all outputs as complete. It replaces an "output idle N cycles" heuristic exactly. Injecting from the host makes it fall.
* Minimum STEP on an idle chip: `learn_start` +2, `update_start` +5, `step_done` +7 cycles after `step` (QUIESCE_CYCLES = 4).

**Host protocol:** all input events for timestep t must be handed over (accepted by `host_in_ready`) before `step` for t. `host_in_valid` counts as busy only while a flit is presented, so a host that pauses between its own events for t after pulsing `step` can be overtaken by quiescence. The tb_top test hit exactly this while it was being written.

## spikemesh_top parameters

`MESH_X, MESH_Y, N_NEURONS, N_AXONS, W_BITS, FANOUT_DEPTH, LANES (default 8, as in rtl/core), FIFO_DEPTH (4), QUIESCE_CYCLES (4), LEARN_MASK (225 bit, default 0), LEARN_MODE (1)`. Bit `y*MESH_X + x` of `LEARN_MASK` gives that core `LEARN = LEARN_MODE` (1 STDP, 2 STDP + R-STDP); the others get `LEARN = 0`.
Extra outputs beyond SPEC §8: `err`, `drop_cnt`, `busy` (STEP running), `drained`.

## Verification status

**Simulation** (`python3 -m pytest -q tests/noc`, Icarus 12 + cocotb 2.1, about 25 s; 13 pytest cases, 30 cocotb tests):
* `tb_router` × 6 router positions/mesh sizes: every input to every destination (incl. host and off-mesh), hop latency, round-robin fairness and 1 flit/cycle, random backpressure (4 rate mixes). Checks exactly once, correct port, in order per input/output, no credit or FIFO overflow.
* `tb_mesh` on 2x2, 3x3 (and 3x2 with FIFO_DEPTH 2): random all-to-all traffic from every core and the host with random sink backpressure down to 5%, under a deadlock bound. Also a host hotspot test and an off-mesh drop test.
* `tb_tick` (QUIESCE_CYCLES 4 and 7): ordering, waits for quiescence with gapped traffic, waits for busy cores, step while running, and a 6000-cycle random soak with invariants on every cycle (incl. `drained`).
* `tb_top` on 2x2 and 3x3 with `tests/noc/core_stub.sv`: 12 STEPs. The stub reports its per-parity receive count at `learn_start`, so a premature quiescence shows up as a count mismatch. Each test checks that every host-bound flit arrives exactly once and that `drained` rises after each `step_done`.
* Mutation checks: a quiescence controller that ignores network busy, and an arbiter that never advances its pointer, both fail the suite.
* A bug found by the tests: the round-robin index `rr + k` was computed in 3 bits and wrapped at 8, which starved input W when the pointer was 4. It is fixed (4-bit sum) and now covered by formal P6.

**Formal** (`formal/run.sh`, Yosys 0.33 `sat -tempinduct`, about 12 min; summary in `formal/build/summary.txt`). SymbiYosys is not available. yosys-smtbmc with boolector 1.5 does not work (too old), and with z3 4.8 it ran out of memory on the router.

Router properties (in `router.sv` under `ifdef FORMAL`). The only environment assumptions are at the ports: neighbours follow the credit protocol and route XY (assume-guarantee).
* P1: credits <= FIFO_DEPTH, and credits + link flit + downstream debt == FIFO_DEPTH, so a downstream FIFO cannot overflow.
* P2: an input FIFO cannot overflow (`fcount + credit_out == upstream debt`, debt <= depth).
* P3: flit-count conservation (accepted - departed - dropped == flits held).
* P4/P5: tagged flit. An arbitrary flit value injected once is never duplicated, lost or corrupted, and it leaves on the reference-XY port or is dropped only if it is off-mesh.
* P6: one grant per input, and arbitration is work conserving.
* P7/P8: no Y-to-X turn (only the host exit at (0,0)), no U-turns.
* P9: mesh-edge ports without a neighbour never fire.
* P10: `busy` == flits held.
* P11: sent flits satisfy the receiver's XY assumption.

| Check | Result |
|---|---|
| Router, FIFO_DEPTH 2: (0,0), (1,1), (2,2), (1,0) of 3x3, (0,1) of 2x2, (0,0) of 1x1 | all properties **proven unbounded** (k-induction) |
| Router (1,1) 3x3, FIFO_DEPTH 2 | BMC depth 4 from reset passes |
| Router, FIFO_DEPTH 4 | **not proven**: induction did not finish in 900 s; covered by simulation only |
| tick_ctrl, QUIESCE_CYCLES 4 and 6: T1 step_done never with a busy core; T2 quiescent/drained only after QUIESCE_CYCLES+1 quiet cycles; T3 pulse order | **proven unbounded**, plus BMC depth 30 |
| 5 mutants (RR wrap bug, lost credit, YX routing, quiescence ignoring the network, step_done ignoring busy) | all fail as expected (four with a concrete counterexample; `step_done ignoring busy` because induction cannot close) |

## Spec notes

* SPEC §4 does not define which way y grows. Here y grows to the north.
* SPEC does not define what happens to a packet sent to an off-mesh coordinate. Here it is dropped with a sticky `err` flag and counted in `drop_cnt`.
* SPEC §8 lets a core deassert `in_ready` while busy with LEARN/UPDATE. If UPDATE then blocks on `out_ready`, that can cause the message-dependent deadlock described in "Deadlock freedom" item 5. The real core avoids it for SPIKE/CTRL traffic.
* SPEC §3 says the host "may inject at any time after UPDATE(t) started". In addition, the host must hand over all events for t before `step` (see "Host protocol").
* SPEC §8 lists `neurocore` with six parameters. The top also passes `LANES` and `LEARN`, which the real core has.
