"""FPGA backends for host/api.py (SpikeMesh on the Kria KV260).

Two transports implement the same word protocol (fpga/rtl/flit_link.sv):

* ``UartBackend``: 8N1 UART through a USB-UART adapter on a PL PMOD, 921600 baud
  by default. Every word is 6 bytes, little-endian (SPEC §7). Needs ``pyserial``,
  or any object with ``write(bytes)`` and ``read(n) -> bytes``.
* ``PynqDmaBackend``: AXI DMA (MM2S/S2MM, 64-bit beats) plus the AXI-Lite
  register block, used from PYNQ on the board. Needs ``pynq``. The protocol
  logic is in ``AxiStreamBackend``; its I/O object is swappable, which is how
  tests/fpga drives it against the RTL in simulation.

Word protocol summary. A word with header bits [47:38] all ones (type CTRL,
dest 15/15) is a link command (host -> FPGA) or event (FPGA -> host):

    commands: 0x01 STEP, 0x02 FLUSH, 0x03 STATUS, 0x04 SOFT_RESET
    events:   0x81 STEP_DONE  [8] t_parity, [9] err, [37:16] STEP cycles
              0x82 FLUSH_DONE
              0x83 STATUS     [8] t_parity, [9] err, [10] running, [23:16] drop_cnt,
                              [27:24] MESH_X, [31:28] MESH_Y, [37:32] log2 N_NEURONS
              0x84 RESET_DONE

Every chip -> host flit of UPDATE(t) arrives before STEP_DONE(t). The FPGA
drains the mesh before it emits the event, so ``step()`` + ``recv_flits()``
meet the Backend contract.
"""
from __future__ import annotations

import time
from abc import abstractmethod

from host.api import Backend

HDR = 0x3FF << 38
CMD_STEP, CMD_FLUSH, CMD_STATUS, CMD_RESET = 0x01, 0x02, 0x03, 0x04
EV_STEP_DONE, EV_FLUSH_DONE, EV_STATUS, EV_RESET_DONE = 0x81, 0x82, 0x83, 0x84
WORD_MASK = (1 << 48) - 1


class LinkError(RuntimeError):
    pass


def cmd_word(op: int) -> int:
    return HDR | (op & 0xFF)


def is_link_word(w: int) -> bool:
    return (w >> 38) & 0x3FF == 0x3FF


def pack_words(words) -> bytes:
    return b"".join((int(w) & WORD_MASK).to_bytes(6, "little") for w in words)


def unpack_words(data: bytes) -> list[int]:
    if len(data) % 6:
        raise ValueError("byte count is not a multiple of 6")
    return [int.from_bytes(data[i:i + 6], "little") for i in range(0, len(data), 6)]


def decode_event(w: int) -> dict:
    op = w & 0xFF
    d = {"op": op, "t_parity": (w >> 8) & 1, "err": (w >> 9) & 1}
    if op == EV_STEP_DONE:
        d["step_cycles"] = (w >> 16) & 0x3FFFFF
    elif op == EV_STATUS:
        d.update(running=(w >> 10) & 1, drop_cnt=(w >> 16) & 0xFF, mesh_x=(w >> 24) & 0xF,
                 mesh_y=(w >> 28) & 0xF, n_neurons=1 << ((w >> 32) & 0x3F))
    return d


class _LinkBackend(Backend):
    """Command/event logic shared by every word transport."""

    def __init__(self, mesh_x: int | None = None, mesh_y: int | None = None):
        self._rx: list[int] = []
        self.last_step: dict | None = None
        st = self.status()
        self.mesh_x = st["mesh_x"] if mesh_x is None else mesh_x
        self.mesh_y = st["mesh_y"] if mesh_y is None else mesh_y
        self.n_neurons = st["n_neurons"]
        if (self.mesh_x, self.mesh_y) != (st["mesh_x"], st["mesh_y"]):
            raise LinkError(f"bitstream is {st['mesh_x']}x{st['mesh_y']}, expected {mesh_x}x{mesh_y}")

    # transport hooks
    @abstractmethod
    def _send_words(self, words: list[int]) -> None: ...

    @abstractmethod
    def _command(self, op: int) -> int:
        """Issue command `op`, collect words until its event; return the event
        word and append the other words to self._rx."""

    # Backend interface
    def send_flits(self, flits) -> None:
        words = [int(f) & WORD_MASK for f in flits]
        bad = [w for w in words if is_link_word(w)]
        if bad:
            raise ValueError(f"flit {bad[0]:#014x} uses the reserved link header (CTRL to 15/15)")
        if words:
            self._send_words(words)

    def step(self) -> None:
        ev = self._command(CMD_STEP)
        self.last_step = decode_event(ev)
        if self.last_step["err"]:
            raise LinkError(f"chip error flag set after STEP: {self.last_step}")

    def recv_flits(self) -> list[int]:
        self._command(CMD_FLUSH)
        out, self._rx = self._rx, []
        return out

    # extras
    def status(self) -> dict:
        return decode_event(self._command(CMD_STATUS))

    def soft_reset(self) -> None:
        """Reset the chip (FSMs, accumulators, registers). Memories keep their contents."""
        self._command(CMD_RESET)
        self._rx = []


class UartBackend(_LinkBackend):
    def __init__(self, port, baud: int = 921600, timeout: float = 10.0,
                 mesh_x: int | None = None, mesh_y: int | None = None, resync: bool = True):
        if isinstance(port, str):
            try:
                import serial  # pyserial
            except ImportError as e:  # pragma: no cover - depends on host
                raise ImportError("UartBackend needs pyserial (pip install pyserial)") from e
            self.ser = serial.Serial(port, baud, timeout=0.1)
        else:
            self.ser = port            # any object with write(bytes) / read(n)
        self.timeout = timeout
        self._buf = bytearray()
        if resync:
            self.resync()
        super().__init__(mesh_x, mesh_y)

    def resync(self) -> None:
        """Let the FPGA drop any partial word (idle > 16 byte times), then discard input."""
        time.sleep(0.002)
        self._buf.clear()
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()

    def _send_words(self, words):
        data = pack_words(words)
        for i in range(0, len(data), 4096):
            self.ser.write(data[i:i + 4096])

    def _read_word(self) -> int:
        deadline = time.monotonic() + self.timeout
        while len(self._buf) < 6:
            chunk = self.ser.read(6 - len(self._buf))
            if chunk:
                self._buf.extend(chunk)
            elif time.monotonic() > deadline:
                raise LinkError(f"UART timeout ({self.timeout} s) waiting for a word")
        w = int.from_bytes(self._buf[:6], "little")
        del self._buf[:6]
        return w

    def _command(self, op):
        self._send_words([cmd_word(op)])
        want = op | 0x80
        while True:
            w = self._read_word()
            if is_link_word(w) and (w & 0xFF) == want:
                return w
            if is_link_word(w):
                raise LinkError(f"unexpected event {w:#014x} while waiting for {want:#x}")
            self._rx.append(w)


# ---------------------------------------------------------------------------
# AXI DMA path
# ---------------------------------------------------------------------------
REG_CTRL, REG_STATUS, REG_STEP_COUNT, REG_LAST_STEP, REG_LAST_DRAIN = 0x00, 0x04, 0x08, 0x0C, 0x10
REG_BUSY_CYC, REG_CYCLES, REG_IN_FLITS, REG_OUT_FLITS, REG_DROP = 0x14, 0x18, 0x1C, 0x20, 0x24
REG_ID, REG_CONFIG = 0x28, 0x2C
CTRL_STEP, CTRL_FLUSH, CTRL_RESET, CTRL_HOST_SEL = 1 << 0, 1 << 1, 1 << 3, 1 << 8
ID_VALUE = 0x534D0107


class AxiStreamBackend(_LinkBackend):
    """Protocol over an I/O object providing:

        reg_write(off, val), reg_read(off) -> int
        send(words: list[int])                      MM2S, one word per 64-bit beat
        recv_until_last(trigger) -> list[int]       start S2MM, call trigger(), return
                                                    the beats up to and including TLAST
    """

    def __init__(self, io, mesh_x: int | None = None, mesh_y: int | None = None):
        self.io = io
        ident = io.reg_read(REG_ID)
        if ident != ID_VALUE:
            raise LinkError(f"unexpected ID register {ident:#010x} (want {ID_VALUE:#010x})")
        cfg = io.reg_read(REG_CONFIG)
        self.config = {"mesh_x": cfg & 0xF, "mesh_y": (cfg >> 4) & 0xF, "lanes": 1 << ((cfg >> 8) & 0xF),
                       "w_bits": (cfg >> 12) & 0xF, "n_neurons": 1 << ((cfg >> 16) & 0xF),
                       "n_axons": 1 << ((cfg >> 20) & 0xF)}
        io.reg_write(REG_CTRL, 0)                    # HOST_SEL = 0: AXI path
        super().__init__(mesh_x, mesh_y)

    def _send_words(self, words):
        self.io.send(words)

    def _command(self, op):
        if op == CMD_STEP:
            trig = lambda: self.io.reg_write(REG_CTRL, CTRL_STEP)
        elif op == CMD_FLUSH:
            trig = lambda: self.io.reg_write(REG_CTRL, CTRL_FLUSH)
        elif op == CMD_RESET:
            trig = lambda: self.io.reg_write(REG_CTRL, CTRL_RESET)
        else:                                         # STATUS has no register; send the command word
            trig = lambda: self.io.send([cmd_word(op)])
        words = self.io.recv_until_last(trig)
        if not words or not is_link_word(words[-1]) or (words[-1] & 0xFF) != (op | 0x80):
            raise LinkError(f"command {op:#x}: stream did not end with its event: "
                            f"{[hex(w) for w in words[-3:]]}")
        self._rx.extend(words[:-1])
        return words[-1]

    def counters(self) -> dict:
        r = self.io.reg_read
        return {"step_count": r(REG_STEP_COUNT), "last_step_cycles": r(REG_LAST_STEP),
                "last_drain_cycles": r(REG_LAST_DRAIN), "busy_cycles": r(REG_BUSY_CYC),
                "cycles": r(REG_CYCLES), "in_flits": r(REG_IN_FLITS), "out_flits": r(REG_OUT_FLITS),
                "drop_cnt": r(REG_DROP), "status": r(REG_STATUS)}


class _PynqIO:  # pragma: no cover - runs on the board only
    """AXI DMA (simple mode) + MMIO via PYNQ."""

    S2MM_LENGTH = 0x58

    def __init__(self, overlay, dma_name="axi_dma_0", ctrl_name="spikemesh_0", buf_words=1 << 18):
        import numpy as np
        from pynq import allocate
        self.np = np
        self.dma = getattr(overlay, dma_name)
        self.ctrl = getattr(overlay, ctrl_name)
        self.txbuf = allocate(shape=(buf_words,), dtype=np.uint64)
        self.rxbuf = allocate(shape=(buf_words,), dtype=np.uint64)

    def reg_write(self, off, val):
        self.ctrl.write(off, val)

    def reg_read(self, off):
        return int(self.ctrl.read(off))

    def send(self, words):
        n = len(self.txbuf)
        for i in range(0, len(words), n):
            chunk = words[i:i + n]
            self.txbuf[:len(chunk)] = self.np.asarray(chunk, dtype=self.np.uint64)
            self.txbuf.flush()
            self.dma.sendchannel.transfer(self.txbuf, nbytes=8 * len(chunk))
            self.dma.sendchannel.wait()

    def recv_until_last(self, trigger):
        out: list[int] = []
        first = True
        while True:
            self.dma.recvchannel.transfer(self.rxbuf)    # armed before the trigger: no stall
            if first:
                trigger()
                first = False
            self.dma.recvchannel.wait()
            nbytes = self.dma.read(self.S2MM_LENGTH)
            self.rxbuf.invalidate()
            words = [int(w) for w in self.rxbuf[:nbytes // 8]]
            out.extend(words)
            if words and is_link_word(words[-1]) and (words[-1] & 0xFF) >= 0x81:
                return out


class PynqDmaBackend(AxiStreamBackend):  # pragma: no cover - runs on the board only
    """On the KV260 under PYNQ:  PynqDmaBackend("spikemesh.bit")."""

    def __init__(self, bitfile="spikemesh.bit", dma_name="axi_dma_0", ctrl_name="spikemesh_0",
                 mesh_x=None, mesh_y=None, buf_words=1 << 18):
        try:
            from pynq import Overlay
        except ImportError as e:
            raise ImportError("PynqDmaBackend needs the pynq package (run it on the board)") from e
        self.overlay = Overlay(bitfile)
        super().__init__(_PynqIO(self.overlay, dma_name, ctrl_name, buf_words), mesh_x, mesh_y)
