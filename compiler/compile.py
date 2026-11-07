"""NetIR -> placement -> CFG_WRITE flit program (docs/SPEC.md §3, §4, §5).

Entry point: ``compile_net(net, mesh_x, mesh_y, ...) -> Program``.

Mapping rules (see compiler/README.md for the rationale):
  * every layer is cut into ``k = ceil(width / N_NEURONS)`` balanced slices;
    a core holds exactly one slice (core registers are per layer);
  * a slice core receives the *whole* previous layer (or network input) on
    axons ``0 .. n_in-1``: global neuron j of layer i drives axon j of every
    core holding layer i+1, so fan-in > N_AXONS is a compile error;
  * a non-final neuron has one fan-out entry per core of the next layer
    (+1 host entry if its layer is probed); a final-layer neuron has one host
    entry (dest_x = 15);
  * placement: greedy snake order, then pairwise-swap local search minimising
    sum(fan-out entries x XY hops) including host traffic through (0,0) west.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compiler import flits as F
from compiler.netir import NetIR

HOST_X = F.HOST_X
MAX_FANOUT_COUNT = 63          # count field of the fan-out pointer is 6 bits (§5)
REG_THETA, REG_LEAK, REG_RESET, REG_REFRAC, REG_VMIN, REG_LEARN_EN = 0, 1, 2, 3, 4, 8
FIELD_V, FIELD_BIAS, FIELD_REFC, FIELD_PTR = 0, 1, 2, 3
# SPEC v0.2 §9: plastic_axon_max, adaptive threshold increment / decay
REG_PLASTIC_AXON_MAX, REG_TH_INC, REG_TH_DECAY = 23, 24, 25
DBG_THETA_A = 6                # region 11 debug space 6: theta_a[n] (§5, §9 rule 4)


def addr_theta_a(neuron: int) -> int:
    """Region 11, debug space 6, index n (SPEC §5/§9)."""
    return (3 << 20) | (DBG_THETA_A << 16) | (neuron & 0xFFFF)


class CompileError(ValueError):
    """The network cannot be mapped onto the requested chip."""


@dataclass
class Slice:
    layer: int
    lo: int          # first global neuron index of the layer held by this core
    hi: int          # one past the last
    x: int = -1
    y: int = -1

    @property
    def n(self) -> int:
        return self.hi - self.lo


@dataclass
class Program:
    mesh_x: int
    mesh_y: int
    n_neurons: int
    n_axons: int
    w_bits: int
    fanout_depth: int
    n_inputs: int
    layer_sizes: list[int]
    slices: list[Slice]
    flits: list[int]
    probe_layers: tuple[int, ...] = ()
    options: dict = field(default_factory=dict)
    cost_greedy: int = 0
    cost_final: int = 0
    input_map: list[list[tuple[int, int, int]]] = field(default_factory=list)
    output_map: dict[tuple[int, int, int], int] = field(default_factory=dict)
    probe_map: dict[tuple[int, int, int], tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input_map:
            self.input_map, self.output_map, self.probe_map = build_maps(
                self.n_inputs, self.layer_sizes, self.slices, self.probe_layers)

    @property
    def n_outputs(self) -> int:
        return self.layer_sizes[-1]

    @property
    def placement(self) -> dict[tuple[int, int], Slice]:
        return {(s.x, s.y): s for s in self.slices}

    def state_reset_flits(self) -> list[int]:
        """CFG writes that set v = 0, refc = 0 and theta_a = 0 for every used neuron."""
        out = []
        for s in self.slices:
            for n in range(s.n):
                out.append(F.cfg_write(s.x, s.y, F.addr_neuron(n, FIELD_V), 0))
                out.append(F.cfg_write(s.x, s.y, F.addr_neuron(n, FIELD_REFC), 0))
                out.append(F.cfg_write(s.x, s.y, addr_theta_a(n), 0))
        return out

    def pointer_flits(self, silence: bool = False) -> list[int]:
        """Fan-out pointer writes for every used neuron (count 0 if ``silence``)."""
        out = []
        cnt = _fanout_count_per_layer(self.layer_sizes, self.slices, self.probe_layers)
        for s in self.slices:
            c = 0 if silence else cnt[s.layer]
            for n in range(s.n):
                ptr = 0 if silence else (c << 10) | (n * c)
                out.append(F.cfg_write(s.x, s.y, F.addr_neuron(n, FIELD_PTR), ptr))
        return out

    def placement_report(self) -> str:
        lines = [f"SpikeMesh placement on {self.mesh_x}x{self.mesh_y} mesh "
                 f"(N={self.n_neurons}, A={self.n_axons}, FANOUT_DEPTH={self.fanout_depth})",
                 f"layers: {self.layer_sizes}, inputs: {self.n_inputs}, cores used: "
                 f"{len(self.slices)}/{self.mesh_x * self.mesh_y}, cfg flits: {len(self.flits)}"]
        grid = [["  .   " for _ in range(self.mesh_x)] for _ in range(self.mesh_y)]
        idx: dict[int, int] = {}
        for s in sorted(self.slices, key=lambda s: (s.layer, s.lo)):
            k = idx.get(s.layer, 0)
            idx[s.layer] = k + 1
            grid[s.y][s.x] = f"L{s.layer}.{k}".ljust(6)[:6]
        for s in sorted(self.slices, key=lambda s: (s.layer, s.lo)):
            lines.append(f"  layer {s.layer} neurons [{s.lo:4d},{s.hi:4d}) -> core ({s.x},{s.y})")
        lines.append("grid (row y=0 at top, host port west of (0,0)):")
        for y in range(self.mesh_y):
            lines.append("  " + " ".join(grid[y]))
        d = self.cost_greedy - self.cost_final
        pct = 100.0 * d / self.cost_greedy if self.cost_greedy else 0.0
        lines.append(f"traffic-weighted hops: greedy {self.cost_greedy} -> after local search "
                     f"{self.cost_final} ({pct:.1f}% fewer)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers

def _snake(mesh_x: int, mesh_y: int) -> list[tuple[int, int]]:
    order = []
    for y in range(mesh_y):
        xs = range(mesh_x) if y % 2 == 0 else range(mesh_x - 1, -1, -1)
        order.extend((x, y) for x in xs)
    return order


def _split(width: int, k: int) -> list[tuple[int, int]]:
    """Balanced split of range(width) into k contiguous chunks (sizes differ by <= 1)."""
    q, r = divmod(width, k)
    out, lo = [], 0
    for i in range(k):
        hi = lo + q + (1 if i < r else 0)
        out.append((lo, hi))
        lo = hi
    return out


def _fanout_count_per_layer(layer_sizes, slices, probe_layers) -> list[int]:
    n_layers = len(layer_sizes)
    cores_of = [sum(1 for s in slices if s.layer == i) for i in range(n_layers)]
    cnt = []
    for i in range(n_layers):
        if i == n_layers - 1:
            cnt.append(1)
        else:
            cnt.append(cores_of[i + 1] + (1 if i in probe_layers else 0))
    return cnt


def build_maps(n_inputs, layer_sizes, slices, probe_layers):
    layer0 = [s for s in slices if s.layer == 0]
    input_map = [[(s.x, s.y, k) for s in layer0] for k in range(n_inputs)]
    last = len(layer_sizes) - 1
    output_map, probe_map = {}, {}
    for s in slices:
        for n in range(s.n):
            if s.layer == last:
                output_map[(s.x, s.y, n)] = s.lo + n
            if s.layer == last or s.layer in probe_layers:
                probe_map[(s.x, s.y, n)] = (s.layer, s.lo + n)
    return input_map, output_map, probe_map


def _hops(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


HOST_POS = (-1, 0)   # host port sits west of core (0, 0): one hop further


class _CostModel:
    """Traffic-weighted XY hop count. Edge weight = number of fan-out entries
    (or injected input events per input index) crossing that pair of nodes."""

    def __init__(self, slices, n_inputs, probe_layers, n_layers):
        self.slices = slices
        self.adj: list[list[tuple[int, int]]] = [[] for _ in slices]   # (other slice, weight)
        self.host_w = [0] * len(slices)
        by_layer = [[i for i, s in enumerate(slices) if s.layer == L] for L in range(n_layers)]
        for i, s in enumerate(slices):
            if s.layer == 0:
                self.host_w[i] += n_inputs          # one event per input index per core
            if s.layer == n_layers - 1 or s.layer in probe_layers:
                self.host_w[i] += s.n               # one host entry per neuron
            if s.layer + 1 < n_layers:
                for j in by_layer[s.layer + 1]:
                    self.adj[i].append((j, s.n))
                    self.adj[j].append((i, s.n))

    def local(self, i, pos) -> int:
        p = pos[i]
        c = self.host_w[i] * _hops(p, HOST_POS)
        for j, w in self.adj[i]:
            c += w * _hops(p, pos[j])
        return c

    def total(self, pos) -> int:
        c = sum(self.host_w[i] * _hops(pos[i], HOST_POS) for i in range(len(pos)))
        c += sum(w * _hops(pos[i], pos[j]) for i in range(len(pos)) for j, w in self.adj[i] if i < j)
        return c


def place(slices, mesh_x, mesh_y, n_inputs, probe_layers, n_layers, local_search=True,
          max_passes=50):
    """Greedy snake placement then first-improvement pairwise swap search.
    Mutates slices' x, y. Returns (cost_greedy, cost_final)."""
    order = _snake(mesh_x, mesh_y)
    pos = [order[i] for i in range(len(slices))]
    model = _CostModel(slices, n_inputs, probe_layers, n_layers)
    greedy = model.total(pos)
    if local_search:
        occupant: dict[tuple[int, int], int] = {p: i for i, p in enumerate(pos)}
        cores = order
        for _ in range(max_passes):
            improved = False
            for i in range(len(slices)):
                for c in cores:
                    if c == pos[i]:
                        continue
                    j = occupant.get(c)
                    before = model.local(i, pos) + (model.local(j, pos) if j is not None else 0)
                    pi = pos[i]
                    pos[i] = c
                    if j is not None:
                        pos[j] = pi
                    after = model.local(i, pos) + (model.local(j, pos) if j is not None else 0)
                    if after < before:
                        occupant[c] = i
                        if j is not None:
                            occupant[pi] = j
                        else:
                            del occupant[pi]
                        improved = True
                    else:
                        pos[i] = pi
                        if j is not None:
                            pos[j] = c
            if not improved:
                break
    for s, p in zip(slices, pos):
        s.x, s.y = p
    return greedy, model.total(pos)


# --------------------------------------------------------------------------- compile

def _check_range(name, arr, lo, hi):
    arr = np.asarray(arr)
    if arr.size and (arr.min() < lo or arr.max() > hi):
        raise CompileError(f"{name} out of range [{lo}, {hi}] (got {arr.min()}..{arr.max()})")


def _pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def compile_net(net: NetIR, mesh_x: int, mesh_y: int, n_neurons: int = 256, n_axons: int = 256,
                w_bits: int = 8, fanout_depth: int = 1024, *, probe_layers=(),
                dense_weights: bool = True, init_unused: bool = True,
                local_search: bool = True) -> Program:
    """Compile a NetIR into a CFG_WRITE flit program for a mesh_x x mesh_y chip.

    probe_layers: hidden-layer indices whose neurons also get a host fan-out entry.
    dense_weights: True (default, safe) writes every synapse of the used block
        (rows 0..n_in-1, columns 0..slice-1), zeros included, so no weight
        memory reset is assumed. False writes only nonzero weights and ASSUMES
        weight memory is zero after reset.
    init_unused: True (default) writes fan-out pointer = 0 (count 0) for every
        neuron that holds no network neuron (spare neurons of used cores and all
        neurons of unused cores) and learn_en = 0 on unused cores, so that
        un-reset RAM can never emit packets. False assumes pointers reset to 0.
    """
    # ---- parameter checks
    if not (1 <= mesh_x <= 15 and 1 <= mesh_y <= 15):
        raise CompileError(f"mesh {mesh_x}x{mesh_y} invalid: each side must be 1..15 (x=15 is the host)")
    if not (_pow2(n_neurons) and n_neurons <= 1024):
        raise CompileError(f"N_NEURONS={n_neurons} must be a power of two <= 1024")
    if not (_pow2(n_axons) and n_axons <= 1024):
        raise CompileError(f"N_AXONS={n_axons} must be a power of two <= 1024")
    if w_bits not in (4, 8):
        raise CompileError(f"W_BITS={w_bits} must be 4 or 8")
    if not (1 <= fanout_depth <= 1024):
        raise CompileError(f"FANOUT_DEPTH={fanout_depth} must be 1..1024 (pointer base is 10 bits)")
    if not net.layers:
        raise CompileError("network has no layers")
    if net.w_bits > w_bits:
        raise CompileError(f"network was trained for W_BITS={net.w_bits} but chip has W_BITS={w_bits}")
    probe_layers = tuple(sorted(set(int(p) for p in probe_layers)))
    n_layers = len(net.layers)
    for p in probe_layers:
        if not 0 <= p < n_layers:
            raise CompileError(f"probe layer {p} does not exist")

    # ---- layer checks
    wlo, whi = -(1 << (w_bits - 1)), (1 << (w_bits - 1)) - 1
    prev_width = net.n_inputs
    for i, L in enumerate(net.layers):
        if L.n_in != prev_width:
            raise CompileError(f"layer {i}: weight has {L.n_in} rows but previous width is {prev_width}")
        if L.n_in > n_axons:
            src = "network input" if i == 0 else f"layer {i - 1}"
            raise CompileError(
                f"layer {i}: fan-in {L.n_in} ({src}) exceeds N_AXONS={n_axons}; a neuron can only "
                f"sum its own core's axons, so every input of a layer must fit on one core's axons")
        if L.bias.shape != (L.n_out,):
            raise CompileError(f"layer {i}: bias shape {L.bias.shape} != ({L.n_out},)")
        _check_range(f"layer {i} weight", L.weight, wlo, whi)
        _check_range(f"layer {i} bias", L.bias, -32768, 32767)
        if not 1 <= L.theta <= 32767:
            raise CompileError(f"layer {i}: theta {L.theta} must be in 1..32767")
        if not 0 <= L.leak_shift <= 15:
            raise CompileError(f"layer {i}: leak_shift {L.leak_shift} must be 0..15")
        if L.reset_mode not in (0, 1):
            raise CompileError(f"layer {i}: reset_mode {L.reset_mode} must be 0 or 1")
        if not 0 <= L.refrac <= 15:
            raise CompileError(f"layer {i}: refrac {L.refrac} must be 0..15")
        if not -32768 <= L.v_min <= 32767:
            raise CompileError(f"layer {i}: v_min {L.v_min} out of int16")
        prev_width = L.n_out

    # ---- slicing
    layer_sizes = [L.n_out for L in net.layers]
    slices: list[Slice] = []
    for i, w in enumerate(layer_sizes):
        if w == 0:
            raise CompileError(f"layer {i} has zero neurons")
        for lo, hi in _split(w, math.ceil(w / n_neurons)):
            slices.append(Slice(i, lo, hi))
    if len(slices) > mesh_x * mesh_y:
        raise CompileError(f"mesh too small: network needs {len(slices)} cores "
                           f"(layer slices {[math.ceil(w / n_neurons) for w in layer_sizes]}), "
                           f"mesh {mesh_x}x{mesh_y} has {mesh_x * mesh_y}")

    # ---- fan-out budget
    cnt = _fanout_count_per_layer(layer_sizes, slices, probe_layers)
    for s in slices:
        if cnt[s.layer] > MAX_FANOUT_COUNT:
            raise CompileError(f"layer {s.layer}: each neuron needs {cnt[s.layer]} fan-out entries "
                               f"(> {MAX_FANOUT_COUNT}, the 6-bit count limit)")
        if s.n * cnt[s.layer] > fanout_depth:
            raise CompileError(f"layer {s.layer}: core slice of {s.n} neurons x {cnt[s.layer]} "
                               f"entries = {s.n * cnt[s.layer]} fan-out entries > FANOUT_DEPTH={fanout_depth}")

    # ---- placement
    greedy, final = place(slices, mesh_x, mesh_y, net.n_inputs, probe_layers, n_layers, local_search)

    # ---- flit generation
    flits: list[int] = []
    by_layer = [[s for s in slices if s.layer == i] for i in range(n_layers)]
    wmask = (1 << w_bits) - 1
    used = set()
    for s in sorted(slices, key=lambda s: (s.layer, s.lo)):
        L = net.layers[s.layer]
        x, y = s.x, s.y
        used.add((x, y))
        cw = lambda a, d: flits.append(F.cfg_write(x, y, a, d & 0xFFFF))  # noqa: E731
        # core registers
        cw(F.addr_reg(REG_THETA), L.theta)
        cw(F.addr_reg(REG_LEAK), L.leak_shift)
        cw(F.addr_reg(REG_RESET), L.reset_mode)
        cw(F.addr_reg(REG_REFRAC), L.refrac)
        cw(F.addr_reg(REG_VMIN), L.v_min)
        cw(F.addr_reg(REG_LEARN_EN), 0)
        # v0.2: no adaptive threshold, every axon plastic (learning is off anyway)
        cw(F.addr_reg(REG_PLASTIC_AXON_MAX), 0)
        cw(F.addr_reg(REG_TH_INC), 0)
        cw(F.addr_reg(REG_TH_DECAY), 0)
        # fan-out entry list for this layer (same for every neuron)
        if s.layer == n_layers - 1:
            dests = [(HOST_X, 0, None)]
        else:
            dests = [(t.x, t.y, "axon") for t in by_layer[s.layer + 1]]
            if s.layer in probe_layers:
                dests.append((HOST_X, 0, None))
        c = len(dests)
        for n in range(s.n):
            g = s.lo + n
            cw(F.addr_neuron(n, FIELD_V), 0)
            cw(F.addr_neuron(n, FIELD_BIAS), int(L.bias[g]))
            cw(F.addr_neuron(n, FIELD_REFC), 0)
            cw(addr_theta_a(n), 0)          # theta_a is a neuron memory, undefined at reset
            base = n * c
            cw(F.addr_neuron(n, FIELD_PTR), (c << 10) | base)
            for k, (dx, dy, kind) in enumerate(dests):
                axon = g if kind == "axon" else 0
                cw(F.addr_fanout(base + k, 0), (dx << 12) | (dy << 8))
                cw(F.addr_fanout(base + k, 1), axon)
        if init_unused:
            for n in range(s.n, n_neurons):
                cw(F.addr_neuron(n, FIELD_PTR), 0)
        # weights: axon a = global index of the source neuron, column = local neuron
        W = np.asarray(L.weight)[:, s.lo:s.hi]
        for a in range(W.shape[0]):
            row = W[a]
            for n in range(s.n):
                w = int(row[n])
                if dense_weights or w != 0:
                    cw(F.addr_weight(a, n, n_neurons), w & wmask)
    if init_unused:
        for (x, y) in _snake(mesh_x, mesh_y):
            if (x, y) in used:
                continue
            flits.append(F.cfg_write(x, y, F.addr_reg(REG_LEARN_EN), 0))
            for n in range(n_neurons):
                flits.append(F.cfg_write(x, y, F.addr_neuron(n, FIELD_PTR), 0))

    return Program(mesh_x=mesh_x, mesh_y=mesh_y, n_neurons=n_neurons, n_axons=n_axons,
                   w_bits=w_bits, fanout_depth=fanout_depth, n_inputs=net.n_inputs,
                   layer_sizes=layer_sizes, slices=slices, flits=flits, probe_layers=probe_layers,
                   options=dict(dense_weights=dense_weights, init_unused=init_unused,
                                local_search=local_search),
                   cost_greedy=greedy, cost_final=final)
