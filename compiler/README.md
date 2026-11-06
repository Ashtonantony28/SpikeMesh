# SpikeMesh compiler (SG4)

`NetIR` (from `train/`) → placement → ordered list of CFG_WRITE flits → binary image → host.

```python
from compiler.compile import compile_net
from compiler import image
from host.api import SpikeMeshHost, GoldenBackend

prog = compile_net(net, mesh_x=2, mesh_y=2)          # N=256, A=256, W_BITS=8, FANOUT_DEPTH=1024
print(prog.placement_report())
image.save(prog, "net.smim")
host = SpikeMeshHost(GoldenBackend.for_program(prog))
host.load(prog)
out = host.run(events)                                # bool (T, n_inputs) -> bool (T, n_outputs)
```

Files: `compile.py` (mapping, placement, flit generation), `image.py` (binary image),
`../host/api.py` (host + backend interface + golden backend). `netir.py` and
`flits.py` are frozen contracts. Tests: `python -m pytest tests/compiler -q`.

## Mapping rules

* **Slicing.** Layer `i` of width `W` is cut into `k = ceil(W / N_NEURONS)` contiguous,
  balanced slices (sizes differ by ≤ 1). Each slice gets its own core. A core holds
  neurons of exactly one layer, because `theta`, `leak_shift`, `reset_mode`, `refrac`,
  `v_min` are per-core registers.
* **Axons.** Every core of layer `i` receives the *whole* previous layer: global neuron
  `j` of layer `i-1` (or network input `j` for layer 0) arrives on axon `j`. Weight
  `W[j, lo+n]` goes to address `axon j * N_NEURONS + local neuron n`. Hence
  **fan-in > N_AXONS is a compile error** (a neuron can only sum its own core's axons).
* **Fan-out.** A neuron of a non-final layer gets one entry per core of the next layer
  (`dest = that core, dest_axon = global neuron index`), plus one host entry
  (`dest_x = 15`) if its layer is in `probe_layers`. Final-layer neurons get one host
  entry. Entries of local neuron `n` are at `base = n * count`. Errors: count > 63
  (6-bit count field), or `slice_size * count > FANOUT_DEPTH`.
* **Input map.** Network input `k` → `[(x, y, k) for every layer-0 core]`; the host sends
  one SPIKE flit per element with parity `t & 1`.
* **Output map.** `(src_x, src_y, local neuron)` → global output index of the final
  layer. `probe_map` maps the same key to `(layer, index)` for final + probed layers.
* **Timing.** SPEC §3 latency is reproduced naturally: the raster row `t` returned by the
  host equals `reference_run(net, inputs)[-1][t]`.
* Errors also for: mesh side outside 1..15, non-power-of-two N/A, W_BITS ∉ {4, 8},
  FANOUT_DEPTH > 1024 (pointer base is 10 bits), out-of-range weights / bias / registers,
  and more slices than cores ("mesh too small"). All raise `CompileError`.

## What the CFG stream writes (and what it assumes)

Per used core: regs 0–4 (`theta`, `leak_shift`, `reset_mode`, `refrac`, `v_min`) and
`learn_en = 0`; per used neuron `v = 0`, `bias`, `refc = 0`, fan-out pointer, both words
of each fan-out entry; weights.

* `dense_weights=True` (default, safe): writes every synapse of the used block
  (rows `0..n_in-1` × columns `0..slice-1`), zeros included. Nothing is assumed about
  weight-memory reset. Rows/columns outside the block are never read (no spikes arrive
  on those axons; those neurons have fan-out count 0). `dense_weights=False` writes only
  nonzero weights and **assumes weight memory is zero after reset**.
* `init_unused=True` (default, safe): writes fan-out pointer 0 for every spare neuron of
  a used core and for every neuron of an unused core (plus `learn_en = 0` there), so
  un-reset RAM can never emit packets. Spare neurons may still "fire" internally from
  garbage `v`/`bias`, which is harmless with count 0. `False` assumes pointers reset to 0.
* Weight data is the two's-complement value masked to `W_BITS` bits (upper bits zero).
* Flit order: core by core in (layer, slice) order: regs, neuron fields + fan-out,
  spare pointers, weights; then unused cores. Any order is legal for the golden model.

## Placement

1. **Greedy:** slices in (layer, slice) order onto cores in snake order
   (row 0 left→right, row 1 right→left, …), so consecutive layers are adjacent.
2. **Local search:** first-improvement pairwise swap over (slice, any core), empty cores
   included (a swap with an empty core is a move), repeated until a pass finds no
   improvement (≤ 50 passes). Cost = Σ fan-out entries × XY hops between source and
   destination core, plus host traffic with the host at virtual position (−1, 0)
   (west of (0,0)): `n_inputs` events per layer-0 core and one entry per final/probed
   neuron. Deterministic. `placement_report()` prints the slice table, a grid, and
   cost greedy → final. Tests check final ≤ greedy always and < on a known bad case.

## Binary image (`.smim`, little-endian)

| Off | Size | Field |
|---|---|---|
| 0 | 4 | magic `SMIM` |
| 4 | 2 | version = 1 |
| 6 | 1+1 | mesh_x, mesh_y |
| 8 | 2+2 | n_neurons, n_axons |
| 12 | 1+1 | w_bits, reserved |
| 14 | 2 | fanout_depth |
| 16 | 4 | n_inputs |
| 20 | 4 | n_flits |
| 24 | 4 | meta_len |
| 28 | 4 | CRC-32 of meta + flits |
| 32 | meta_len | JSON: layer_sizes, slices `[layer, lo, hi, x, y]`, probe_layers, options, costs |
| … | 6 × n_flits | flits, 48-bit little-endian, send order |

A device loader only needs the header and the flit block (stream flits into the host
port in order). Input/output maps are rebuilt from the slice list on load.

## Host API

`Backend` has three methods: `send_flits(list[int])`, `step()`, `recv_flits() -> list[int]`.
`GoldenBackend` implements them on `golden.spikemesh.Chip` purely by decoding flits
(CFG_WRITE → `Core.cfg_write`, SPIKE → in-flight queue with its parity, READ_REQ →
READ_RESP, CTRL REWARD). RTL (cocotb/Verilator) and FPGA backends implement the same
three methods. `SpikeMeshHost.reset_state()` silences fan-out, steps twice to drain both
accumulator banks, rewrites `v`/`refc`, and restores pointers (uses only CFG flits, so it
works on every backend).

## Limits

| Limit | Value |
|---|---|
| fan-in per layer | ≤ N_AXONS |
| cores | slices ≤ mesh_x × mesh_y, mesh side ≤ 15 |
| fan-out entries per neuron | ≤ 63 (cores of next layer + probe) |
| fan-out entries per core | slice × count ≤ FANOUT_DEPTH ≤ 1024 |
| topology | feed-forward fully-connected layers only (NetIR); no learning config yet |
