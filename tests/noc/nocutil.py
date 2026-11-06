"""Shared helpers for the SpikeMesh NoC cocotb testbenches (tests/noc)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compiler import flits  # noqa: E402

import cocotb  # noqa: E402
from cocotb.clock import Clock  # noqa: E402
from cocotb.triggers import FallingEdge, RisingEdge  # noqa: E402

PN, PE, PS, PW, PL = 0, 1, 2, 3, 4
PNAME = "NESWL"
HOST_X = 15
M48 = (1 << 48) - 1


def mk(dx: int, dy: int, tag: int, typ: int = 0) -> int:
    """Test flit: header + 37-bit unique tag in payload[36:0]."""
    return (typ & 3) << 46 | (dx & 15) << 42 | (dy & 15) << 38 | (tag & ((1 << 37) - 1))


def tag_of(f: int) -> int:
    return f & ((1 << 37) - 1)


def dest_of(f: int) -> tuple[int, int]:
    return (f >> 42) & 15, (f >> 38) & 15


def ival(sig) -> int:
    """Strict read: fails on X/Z (use for control signals)."""
    return int(sig.value)


def dval(sig) -> int:
    """Data read: X/Z bits read as 0 (uninitialised flit registers)."""
    s = str(sig.value).lower()
    return int("".join("1" if c == "1" else "0" for c in s), 2)


def get_field(vec: int, i: int, w: int) -> int:
    return (vec >> (i * w)) & ((1 << w) - 1)


def set_field(vec: int, i: int, w: int, v: int) -> int:
    m = ((1 << w) - 1) << (i * w)
    return (vec & ~m) | ((v << (i * w)) & m)


def xy_route(x: int, y: int, mx: int, my: int, f: int) -> int:
    """Reference model of the router's output choice (5 = drop)."""
    dx, dy = dest_of(f)
    if dx == HOST_X:
        return PW if x > 0 else (PS if y > 0 else PW)
    if dx >= mx or dy >= my:
        return 5
    if dx > x:
        return PE
    if dx < x:
        return PW
    if dy > y:
        return PN
    if dy < y:
        return PS
    return PL


async def start(dut, cycles: int = 3, zero=("step", "net_busy", "core_busy")):
    """Start the clock (cocotb cancels it at the end of each test), drive
    the named inputs (if present) to 0 and reset."""
    for name in zero:
        if hasattr(dut, name):
            getattr(dut, name).value = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut, cycles)


async def reset(dut, cycles: int = 3):
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1


__all__ = ["flits", "dval", "mk", "tag_of", "dest_of", "ival", "get_field", "set_field", "xy_route", "start", "reset",
           "PN", "PE", "PS", "PW", "PL", "PNAME", "HOST_X", "M48"]
