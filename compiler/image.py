"""Binary config image: a Program serialised for the host loader / FPGA.

Layout (all little-endian):

    offset  size  field
    0       4     magic b"SMIM"
    4       2     version (1)
    6       1     mesh_x
    7       1     mesh_y
    8       2     n_neurons
    10      2     n_axons
    12      1     w_bits
    13      1     reserved (0)
    14      2     fanout_depth
    16      4     n_inputs
    20      4     n_flits
    24      4     meta_len  (bytes of UTF-8 JSON that follow the header)
    28      4     crc32 of (meta bytes + flit bytes)
    32      ...   meta JSON: layer_sizes, slices [[layer, lo, hi, x, y], ...],
                  probe_layers, options, cost_greedy, cost_final
    32+M    6*n   flits, each a 48-bit little-endian integer, in send order

The input/output decode maps are not stored: they are a pure function of the
slice placement and are rebuilt on load (compile.build_maps).
"""
from __future__ import annotations

import json
import struct
import zlib

from compiler.compile import Program, Slice

MAGIC = b"SMIM"
VERSION = 1
_HDR = struct.Struct("<4sHBBHHBBHIIII")
HEADER_SIZE = _HDR.size   # 32
FLIT_BYTES = 6


class ImageError(ValueError):
    pass


def to_bytes(prog: Program) -> bytes:
    meta = json.dumps({
        "layer_sizes": prog.layer_sizes,
        "slices": [[s.layer, s.lo, s.hi, s.x, s.y] for s in prog.slices],
        "probe_layers": list(prog.probe_layers),
        "options": prog.options,
        "cost_greedy": prog.cost_greedy,
        "cost_final": prog.cost_final,
    }, separators=(",", ":")).encode()
    body = b"".join(int(f).to_bytes(FLIT_BYTES, "little") for f in prog.flits)
    crc = zlib.crc32(meta + body)
    hdr = _HDR.pack(MAGIC, VERSION, prog.mesh_x, prog.mesh_y, prog.n_neurons, prog.n_axons,
                    prog.w_bits, 0, prog.fanout_depth, prog.n_inputs, len(prog.flits), len(meta), crc)
    return hdr + meta + body


def from_bytes(data: bytes) -> Program:
    if len(data) < HEADER_SIZE:
        raise ImageError("image shorter than header")
    (magic, ver, mx, my, nn, na, wb, _r, fd, n_in, n_flits, mlen, crc) = _HDR.unpack_from(data)
    if magic != MAGIC:
        raise ImageError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ImageError(f"unsupported image version {ver}")
    end = HEADER_SIZE + mlen + FLIT_BYTES * n_flits
    if len(data) != end:
        raise ImageError(f"image length {len(data)} != expected {end}")
    meta_b = data[HEADER_SIZE:HEADER_SIZE + mlen]
    body = data[HEADER_SIZE + mlen:end]
    if zlib.crc32(meta_b + body) != crc:
        raise ImageError("CRC mismatch")
    meta = json.loads(meta_b)
    flits = [int.from_bytes(body[i:i + FLIT_BYTES], "little") for i in range(0, len(body), FLIT_BYTES)]
    return Program(mesh_x=mx, mesh_y=my, n_neurons=nn, n_axons=na, w_bits=wb, fanout_depth=fd,
                   n_inputs=n_in, layer_sizes=meta["layer_sizes"],
                   slices=[Slice(*s) for s in meta["slices"]], flits=flits,
                   probe_layers=tuple(meta["probe_layers"]), options=meta["options"],
                   cost_greedy=meta["cost_greedy"], cost_final=meta["cost_final"])


def save(prog: Program, path) -> None:
    with open(path, "wb") as f:
        f.write(to_bytes(prog))


def load(path) -> Program:
    with open(path, "rb") as f:
        return from_bytes(f.read())
