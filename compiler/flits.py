"""48-bit flit encode/decode (docs/SPEC.md §4). Shared by compiler, host and testbenches."""
from __future__ import annotations

SPIKE, CFG_WRITE, READ, CTRL = 0, 1, 2, 3
HOST_X = 15
CTRL_REWARD = 0x01


def _hdr(typ: int, dx: int, dy: int) -> int:
    return (typ & 3) << 46 | (dx & 0xF) << 42 | (dy & 0xF) << 38


def spike(dx: int, dy: int, axon: int, parity: int) -> int:
    return _hdr(SPIKE, dx, dy) | (parity & 1) << 37 | (axon & 0x3FF)


def host_spike(sx: int, sy: int, neuron: int, parity: int) -> int:
    return _hdr(SPIKE, HOST_X, 0) | (parity & 1) << 37 | (sx & 0xF) << 26 | (sy & 0xF) << 22 | (neuron & 0x3FF)


def cfg_write(dx: int, dy: int, addr: int, data: int) -> int:
    return _hdr(CFG_WRITE, dx, dy) | (addr & 0x3FFFFF) << 16 | (data & 0xFFFF)


def read_req(dx: int, dy: int, addr: int, reply_x: int = HOST_X, reply_y: int = 0) -> int:
    return _hdr(READ, dx, dy) | (addr & 0x3FFFFF) << 15 | (reply_x & 0xF) << 11 | (reply_y & 0xF) << 7


def read_resp(dx: int, dy: int, addr: int, data: int) -> int:
    return _hdr(READ, dx, dy) | 1 << 37 | (addr & 0xFFFF) << 16 | (data & 0xFFFF)


def reward(dx: int, dy: int, r: int) -> int:
    return _hdr(CTRL, dx, dy) | (r & 0xFFFF) << 8 | CTRL_REWARD


def decode(f: int) -> dict:
    typ, dx, dy = f >> 46 & 3, f >> 42 & 0xF, f >> 38 & 0xF
    d = {"type": typ, "dest_x": dx, "dest_y": dy}
    if typ == SPIKE:
        d["parity"] = f >> 37 & 1
        if dx == HOST_X:
            d.update(src_x=f >> 26 & 0xF, src_y=f >> 22 & 0xF, neuron=f & 0x3FF)
        else:
            d["axon"] = f & 0x3FF
    elif typ == CFG_WRITE:
        d.update(addr=f >> 16 & 0x3FFFFF, data=f & 0xFFFF)
    elif typ == READ:
        if f >> 37 & 1:
            d.update(resp=True, addr16=f >> 16 & 0xFFFF, data=f & 0xFFFF)
        else:
            d.update(resp=False, addr=f >> 15 & 0x3FFFFF, reply_x=f >> 11 & 0xF, reply_y=f >> 7 & 0xF)
    else:
        d["opcode"] = f & 0xFF
        r = f >> 8 & 0xFFFF
        d["reward"] = r - 0x10000 if r & 0x8000 else r
    return d


# Address helpers (SPEC §5)
def addr_weight(axon: int, neuron: int, n_neurons: int) -> int:
    return axon * n_neurons + neuron


def addr_neuron(neuron: int, field: int) -> int:
    return 1 << 20 | neuron << 2 | field


def addr_fanout(entry: int, word: int) -> int:
    return 2 << 20 | entry << 1 | word


def addr_reg(reg: int) -> int:
    return 3 << 20 | reg
