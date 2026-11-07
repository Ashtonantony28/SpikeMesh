"""cocotb tests for fpga/rtl (kv260_top: spikemesh_top + uart_bridge + axis_bridge).

The real host code (host/fpga_backend.py + host/api.py) runs in a Python
thread. It talks to the simulated FPGA through fake transports:

* SimSerial: a pyserial-like object whose bytes are bit-banged onto the
  DUT's uart_rx pin, and decoded from its uart_tx pin, by cocotb coroutines.
* SimAxiIO: the AxiStreamBackend I/O object (AXI-Lite register access and
  AXI-Stream beats) serviced by cocotb coroutines.

Compared against: golden.spikemesh through host.api.GoldenBackend, and
compiler.netir.reference_run.
"""
from __future__ import annotations

import os
import queue
import random
import threading

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge

from compiler import flits as F
from compiler.compile import compile_net
from compiler.netir import LayerIR, NetIR, reference_run
from host.api import GoldenBackend, SpikeMeshHost
from host import fpga_backend as FB

MX, MY = int(os.environ.get("MX", 2)), int(os.environ.get("MY", 2))
N, A, FD = int(os.environ.get("N", 32)), int(os.environ.get("A", 32)), int(os.environ.get("FD", 64))
CPB = int(os.environ.get("CPB", 8))
SEED = int(os.environ.get("SEED", 1))
BP = float(os.environ.get("BP", 0.3))
WALL_TIMEOUT = 600.0


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.uart_rx.value = 1
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    for s in ("awvalid", "wvalid", "bready", "arvalid", "rready"):
        getattr(dut, "s_axi_" + s).value = 0
    dut.s_axi_awaddr.value = 0
    dut.s_axi_araddr.value = 0
    dut.s_axi_wdata.value = 0
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_awprot.value = 0
    dut.s_axi_arprot.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def run_thread(dut, fn, max_cycles=5_000_000):
    """Run blocking host code in a thread while the simulation advances."""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the test
            box["e"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    n = 0
    while th.is_alive():
        await ClockCycles(dut.clk, 200)
        n += 200
        if n > max_cycles:
            raise TimeoutError(f"host thread still running after {max_cycles} cycles")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def tiny_net(rng, sizes=(24, 20, 8)):
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w = rng.integers(-40, 90, (n_in, n_out))
        w[rng.random(w.shape) < 0.3] = 0
        layers.append(LayerIR(weight=w, bias=rng.integers(-4, 6, n_out), theta=int(rng.integers(40, 90)),
                              leak_shift=int(rng.integers(0, 3)), reset_mode=int(rng.integers(0, 2)),
                              refrac=int(rng.integers(0, 2)), v_min=-200))
    return NetIR(sizes[0], layers, w_bits=8)


def net_roundtrip(make_backend, log, T=12):
    """Compile a tiny 2-layer net, run it on golden and on the FPGA backend
    (probing the hidden layer too), compare against each other and against
    reference_run. Also checks READ_REQ readback and reset_state()."""
    rng = np.random.default_rng(SEED)
    net = tiny_net(rng)
    prog = compile_net(net, MX, MY, n_neurons=N, n_axons=A, fanout_depth=FD, probe_layers=(0,))
    ev = rng.random((T, net.n_inputs)) < 0.35
    gold = SpikeMeshHost(GoldenBackend.for_program(prog))
    gold.load(prog)
    g_out, g_r = gold.run(ev, probes=True)
    be = make_backend()
    assert (be.mesh_x, be.mesh_y) == (MX, MY)
    host = SpikeMeshHost(be)
    host.load(prog)
    f_out, f_r = host.run(ev, probes=True)
    ref = reference_run(net, ev)
    assert np.array_equal(f_out, g_out), f"FPGA vs golden output raster mismatch {f_out.sum(1)} {g_out.sum(1)} {[r.sum(1) for r in f_r.values()]} {[r.sum(1) for r in g_r.values()]}"
    for L in g_r:
        assert np.array_equal(f_r[L], g_r[L]), f"layer {L} raster mismatch"
        assert np.array_equal(f_r[L], ref[L]), f"layer {L} differs from reference_run"
    n_spk = int(sum(r.sum() for r in f_r.values()))
    assert n_spk > 0, "net produced no spikes; test is vacuous"
    # readback through the link: v and fan-out pointer of a few neurons, theta
    s = prog.slices[0]
    addrs = [F.addr_neuron(n, f) for n in range(4) for f in (0, 3)] + [F.addr_reg(0)]
    req = [F.read_req(s.x, s.y, a, 15, 0) for a in addrs]
    be.send_flits(req)
    got = be.recv_flits()
    gold.backend.send_flits(req)
    exp = gold.backend.recv_flits()
    assert got == exp, (got, exp)
    # second run after reset_state() must reproduce the first
    host.reset_state()
    gold.reset_state()
    f2 = host.run(ev)
    g2 = gold.run(ev)
    assert np.array_equal(f2, g2) and np.array_equal(f2, f_out)
    log(f"net 24-20-8 on {MX}x{MY}: {len(prog.flits)} cfg flits, T={T}, {n_spk} spikes, "
        f"rasters == golden == reference_run; readback {len(addrs)} OK")
    return be


# ---------------------------------------------------------------------------
# UART path
# ---------------------------------------------------------------------------
class SimSerial:
    """pyserial-like endpoint for the thread side."""

    def __init__(self):
        self.to_dut: queue.Queue[int] = queue.Queue()
        self.from_dut: queue.Queue[int] = queue.Queue()
        self.bytes_out = 0
        self.bytes_in = 0

    def write(self, data: bytes):
        for b in data:
            self.to_dut.put(b)
        self.bytes_out += len(data)

    def read(self, n: int) -> bytes:
        out = bytearray()
        try:
            out.append(self.from_dut.get(timeout=WALL_TIMEOUT))
            while len(out) < n:
                out.append(self.from_dut.get_nowait())
        except queue.Empty:
            pass
        self.bytes_in += len(out)
        return bytes(out)


async def uart_driver(dut, ser: SimSerial):
    """Bit-bang host->FPGA bytes on uart_rx (8N1, CPB clocks per bit)."""
    while True:
        try:
            b = ser.to_dut.get_nowait()
        except queue.Empty:
            await ClockCycles(dut.clk, CPB)
            continue
        await FallingEdge(dut.clk)
        for bit in [0] + [(b >> i) & 1 for i in range(8)] + [1]:
            dut.uart_rx.value = bit
            await ClockCycles(dut.clk, CPB)


async def uart_monitor(dut, ser: SimSerial):
    """Decode FPGA->host bytes from uart_tx."""
    while True:
        await FallingEdge(dut.uart_tx)
        await ClockCycles(dut.clk, CPB // 2)
        assert int(dut.uart_tx.value) == 0, "glitch on uart_tx start bit"
        b = 0
        for i in range(8):
            await ClockCycles(dut.clk, CPB)
            b |= int(dut.uart_tx.value) << i
        await ClockCycles(dut.clk, CPB)
        assert int(dut.uart_tx.value) == 1, "missing stop bit"
        ser.from_dut.put(b)


async def uart_setup(dut):
    await reset(dut)
    ser = SimSerial()
    cocotb.start_soon(uart_driver(dut, ser))
    cocotb.start_soon(uart_monitor(dut, ser))
    return ser


@cocotb.test()
async def test_uart_roundtrip(dut):
    """bytes -> words -> chip -> words -> bytes, raw (no host API)."""
    ser = await uart_setup(dut)
    # garbage partial word first: the bridge must resync on line idle
    for b in (0x12, 0x34, 0x56):
        ser.to_dut.put(b)
    await ClockCycles(dut.clk, 16 * 10 * CPB + 60 * CPB)

    def work():
        x, y = MX - 1, MY - 1
        cfg = [F.cfg_write(x, y, F.addr_reg(0), 1234), F.cfg_write(x, y, F.addr_neuron(5, 1), 0xBEEF),
               F.cfg_write(x, y, F.addr_weight(3, 7, N), 0x81)]
        rd = [F.read_req(x, y, a, 15, 3) for a in (F.addr_reg(0), F.addr_neuron(5, 1), F.addr_weight(3, 7, N))]
        ser.write(FB.pack_words(cfg + rd + [FB.cmd_word(FB.CMD_FLUSH), FB.cmd_word(FB.CMD_STATUS)]))
        buf = b""
        while len(buf) < 30:
            buf += ser.read(30 - len(buf))
        return FB.unpack_words(buf)

    words = await run_thread(dut, work)
    resp = [F.decode(w) for w in words[:3]]
    assert [r["data"] for r in resp] == [1234, 0xBEEF, 0xFF81], resp
    assert all(r["resp"] and r["dest_x"] == 15 and r["dest_y"] == 3 for r in resp)
    assert words[3] == FB.HDR | FB.EV_FLUSH_DONE
    st = FB.decode_event(words[4])
    assert st["op"] == FB.EV_STATUS and (st["mesh_x"], st["mesh_y"], st["n_neurons"]) == (MX, MY, N), st
    assert int(dut.u_uart.rx_overflow.value) == 0 and int(dut.u_uart.rx_frame_err.value) == 0
    dut._log.info(f"UART raw round trip OK: {words[:3]} status={st}")


@cocotb.test()
async def test_uart_backend_net(dut):
    """host.api + UartBackend over the simulated UART: compiled net == golden."""
    ser = await uart_setup(dut)
    be = await run_thread(dut, lambda: net_roundtrip(lambda: FB.UartBackend(ser, timeout=WALL_TIMEOUT),
                                                     dut._log.info))
    assert be.last_step is not None and be.last_step["step_cycles"] > 0

    def reset_and_check():
        be.soft_reset()
        s = be.status()
        be.send_flits([F.read_req(0, 0, F.addr_reg(0), 15, 0)])
        return s, be.recv_flits()

    s, r = await run_thread(dut, reset_and_check)
    assert F.decode(r[0])["data"] == 64, r          # theta back at its reset value
    assert s["t_parity"] == 0
    assert int(dut.u_uart.rx_overflow.value) == 0
    dut._log.info(f"UART: {ser.bytes_out} bytes out, {ser.bytes_in} bytes in; soft reset OK")


# ---------------------------------------------------------------------------
# AXI path
# ---------------------------------------------------------------------------
class SimAxiIO:
    """AxiStreamBackend I/O object; requests are executed by axi_service()."""

    def __init__(self):
        self.req: queue.Queue = queue.Queue()
        self.beats: queue.Queue = queue.Queue()

    def _call(self, *op):
        ev, box = threading.Event(), {}
        self.req.put((op, ev, box))
        if not ev.wait(WALL_TIMEOUT):
            raise TimeoutError(f"sim op {op[0]} timed out")
        return box.get("v")

    def reg_write(self, off, val):
        self._call("w", off, val)

    def reg_read(self, off):
        return self._call("r", off)

    def send(self, words):
        self._call("s", [int(w) for w in words])

    def recv_until_last(self, trigger):
        trigger()
        out = []
        while True:
            data, last = self.beats.get(timeout=WALL_TIMEOUT)
            out.append(data)
            if last:
                return out


async def axil_write(dut, off, val):
    await FallingEdge(dut.clk)
    dut.s_axi_awaddr.value = off
    dut.s_axi_wdata.value = val
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wvalid.value = 1
    aw = w = False
    while not (aw and w):
        await ReadOnly()
        aw = aw or bool(int(dut.s_axi_awready.value))
        w = w or bool(int(dut.s_axi_wready.value))
        await FallingEdge(dut.clk)
        if aw:
            dut.s_axi_awvalid.value = 0
        if w:
            dut.s_axi_wvalid.value = 0
    dut.s_axi_bready.value = 1
    while True:
        await ReadOnly()
        done = int(dut.s_axi_bvalid.value)
        await FallingEdge(dut.clk)
        if done:
            break
    dut.s_axi_bready.value = 0


async def axil_read(dut, off):
    await FallingEdge(dut.clk)
    dut.s_axi_araddr.value = off
    dut.s_axi_arvalid.value = 1
    while True:
        await ReadOnly()
        ok = int(dut.s_axi_arready.value)
        await FallingEdge(dut.clk)
        if ok:
            break
    dut.s_axi_arvalid.value = 0
    dut.s_axi_rready.value = 1
    while True:
        await ReadOnly()
        if int(dut.s_axi_rvalid.value):
            v = int(dut.s_axi_rdata.value)
            await FallingEdge(dut.clk)
            break
        await FallingEdge(dut.clk)
    dut.s_axi_rready.value = 0
    return v


async def axis_send(dut, words, rng):
    await FallingEdge(dut.clk)
    for wd in words:
        while rng.random() < 0.2:                     # random source gaps
            dut.s_axis_tvalid.value = 0
            await FallingEdge(dut.clk)
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tdata.value = wd
        while True:
            await ReadOnly()
            ok = int(dut.s_axis_tready.value)
            await FallingEdge(dut.clk)
            if ok:
                break
    dut.s_axis_tvalid.value = 0


async def axi_service(dut, io: SimAxiIO, rng):
    while True:
        try:
            (op, ev, box) = io.req.get_nowait()
        except queue.Empty:
            await RisingEdge(dut.clk)
            continue
        if op[0] == "w":
            await axil_write(dut, op[1], op[2])
        elif op[0] == "r":
            box["v"] = await axil_read(dut, op[1])
        else:
            await axis_send(dut, op[1], rng)
        ev.set()


async def axis_sink(dut, io: SimAxiIO, rng, bp):
    while True:
        await FallingEdge(dut.clk)
        rdy = rng.random() >= bp
        dut.m_axis_tready.value = int(rdy)
        await ReadOnly()
        if rdy and int(dut.m_axis_tvalid.value):
            d = int(dut.m_axis_tdata.value)
            assert d >> 48 == 0 and int(dut.m_axis_tkeep.value) == 0xFF
            io.beats.put((d, int(dut.m_axis_tlast.value)))


@cocotb.test()
async def test_axi_backend_net(dut):
    """host.api + AxiStreamBackend over simulated AXI-Lite/AXI-Stream: net == golden; registers."""
    await reset(dut)
    rng = random.Random(SEED)
    io = SimAxiIO()
    cocotb.start_soon(axi_service(dut, io, rng))
    cocotb.start_soon(axis_sink(dut, io, rng, BP))
    be = await run_thread(dut, lambda: net_roundtrip(lambda: FB.AxiStreamBackend(io), dut._log.info))
    c = await run_thread(dut, be.counters)
    T = 12 + 2 + 12                           # run + reset_state + run
    assert c["step_count"] == T, c
    assert c["last_step_cycles"] > 0 and c["busy_cycles"] >= c["last_step_cycles"], c
    assert c["in_flits"] > 0 and c["out_flits"] > 0 and c["drop_cnt"] == 0, c
    drain = int(os.environ.get("DRAIN", 0))
    if drain:       # idle-timer fallback: at least DRAIN quiet cycles after step_done
        assert c["last_drain_cycles"] >= drain, c
    assert c["status"] & 1 and not c["status"] & 0b1000, c        # done sticky, no error
    assert be.config == {"mesh_x": MX, "mesh_y": MY, "lanes": int(os.environ.get("LANES", 4)), "w_bits": 8,
                         "n_neurons": N, "n_axons": A}, be.config

    def clear_and_check():
        io.reg_write(FB.REG_STATUS, 1)
        return io.reg_read(FB.REG_STATUS)

    st = await run_thread(dut, clear_and_check)
    assert st & 1 == 0, hex(st)
    dut._log.info(f"AXI counters: {c}")
