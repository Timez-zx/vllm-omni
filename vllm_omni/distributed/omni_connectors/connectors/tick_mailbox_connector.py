# SPDX-License-Identifier: Apache-2.0
"""Tick mailbox: persistent two-slot shared-memory transport (WP4-full).

Replaces the per-chunk create-shm/flock/unlink cycle of SharedMemoryConnector
with ONE persistent segment per (edge, session) holding two fixed slots,
addressed by chunk parity. Single writer (the producing stage), single reader
(the consuming stage's recv poll); a seqlock per slot makes torn reads
detectable, and a reader-maintained ACK cell gives the writer flow control.

SAFETY OVER SPEED -- the failure mode this design refuses to have is a lost
chunk. Every uncertain situation falls back to the legacy per-chunk path
(the inner SharedMemoryConnector), which the reader also consults:

  * payload larger than the slot capacity        -> legacy (e.g. the 2.2 MB
    thinker chunk-0 prefill payload)
  * slot not yet acked (reader >2 chunks behind) -> legacy, no blocking
  * any exception anywhere                       -> legacy

The ACK cell records the highest chunk id CONSUMED (regardless of which
transport carried it), so mixed-transport sequences cannot deadlock the
writer. Chunk ids are a dense per-(session, edge) counter and the reader
polls them strictly in order -- both properties are load-bearing and hold in
the existing adapter protocol.

Enabled via the factory: VLLM_OMNI_TEMPORAL_MAILBOX=1 substitutes this class
where the deploy yaml says SharedMemoryConnector, so arms share one yaml.
"""

from __future__ import annotations

import os
import struct
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from ..utils.logging import get_connector_logger
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)

_MAGIC = 0x7469636B6D623031  # "tickmb01"
# Global header: [magic u64][ack_plus1 u64]  (ack_plus1 = highest consumed
# chunk id + 1; 0 = nothing consumed yet). Written by the READER only.
_HDR_FMT = "<QQ"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
# Slot header: [seq u64][chunk_id u64][size u64][reserved u64]
_SLOT_HDR_FMT = "<QQQQ"
_SLOT_HDR_SIZE = struct.calcsize(_SLOT_HDR_FMT)

_DEFAULT_CAPACITY = 256 * 1024  # per-slot payload bytes; larger goes legacy


def _parse_key(key: str) -> tuple[str, int] | None:
    """'{external_req_id}_{stage}_{chunk}' -> (session_edge_id, chunk_id).

    external_req_id may itself contain underscores; stage and chunk are the
    last two ``_``-delimited fields. The session identity we use for the
    segment name keeps the stage field (one mailbox per edge per session).
    """
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        return None
    try:
        chunk_id = int(parts[2])
    except ValueError:
        return None
    return f"{parts[0]}_{parts[1]}", chunk_id


class _Mailbox:
    """One persistent two-slot segment. Writer or reader view."""

    def __init__(self, name: str, capacity: int, create: bool):
        self.capacity = capacity
        slot_bytes = _SLOT_HDR_SIZE + capacity
        total = _HDR_SIZE + 2 * slot_bytes
        self.slot_bytes = slot_bytes
        if create:
            try:
                self.shm = shm_pkg.SharedMemory(name=name, create=True, size=total)
                struct.pack_into(_HDR_FMT, self.shm.buf, 0, _MAGIC, 0)
            except FileExistsError:
                self.shm = shm_pkg.SharedMemory(name=name)
        else:
            self.shm = shm_pkg.SharedMemory(name=name)
        magic, _ = struct.unpack_from(_HDR_FMT, self.shm.buf, 0)
        if magic != _MAGIC:
            raise ValueError(f"mailbox {name}: bad magic")

    # -- reader-side ---------------------------------------------------- #
    def read_ack_plus1(self) -> int:
        return struct.unpack_from("<Q", self.shm.buf, 8)[0]

    def write_ack(self, chunk_id: int) -> None:
        struct.pack_into("<Q", self.shm.buf, 8, chunk_id + 1)

    def try_read(self, chunk_id: int) -> bytes | None:
        off = _HDR_SIZE + (chunk_id % 2) * self.slot_bytes
        seq1, cid, size, _ = struct.unpack_from(_SLOT_HDR_FMT, self.shm.buf, off)
        if seq1 % 2 == 1 or cid != chunk_id or size == 0 or size > self.capacity:
            return None
        payload = bytes(self.shm.buf[off + _SLOT_HDR_SIZE : off + _SLOT_HDR_SIZE + size])
        seq2 = struct.unpack_from("<Q", self.shm.buf, off)[0]
        if seq2 != seq1:
            return None  # torn read; writer was mid-update
        return payload

    # -- writer-side ---------------------------------------------------- #
    def try_write(self, chunk_id: int, payload: bytes) -> bool:
        if len(payload) > self.capacity:
            return False
        # Flow control: slot for chunk k previously held k-2, which must have
        # been consumed (ack covers consumption via EITHER transport).
        if chunk_id >= 2 and self.read_ack_plus1() < chunk_id - 1:
            return False
        off = _HDR_SIZE + (chunk_id % 2) * self.slot_bytes
        seq = struct.unpack_from("<Q", self.shm.buf, off)[0]
        struct.pack_into("<Q", self.shm.buf, off, seq + 1)  # odd: writing
        struct.pack_into("<QQQ", self.shm.buf, off + 8, chunk_id, len(payload), 0)
        self.shm.buf[off + _SLOT_HDR_SIZE : off + _SLOT_HDR_SIZE + len(payload)] = payload
        struct.pack_into("<Q", self.shm.buf, off, seq + 2)  # even: stable
        return True

    def close(self, unlink: bool = False) -> None:
        try:
            self.shm.close()
            if unlink:
                self.shm.unlink()
        except Exception:
            pass


class TickMailboxConnector(SharedMemoryConnector):
    """SharedMemoryConnector with a persistent two-slot fast path.

    Inherits the legacy per-chunk path and uses it as the universal fallback,
    so correctness never depends on the mailbox.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.capacity = int(config.get("mailbox_slot_bytes", _DEFAULT_CAPACITY) or _DEFAULT_CAPACITY)
        self._tx: dict[str, _Mailbox] = {}   # writer-side, by session_edge
        self._rx: dict[str, _Mailbox] = {}   # reader-side
        self._mb_metrics = {"mb_puts": 0, "mb_gets": 0, "fallback_puts": 0}
        logger.info("TickMailboxConnector active (slot=%dB x2 per edge-session)", self.capacity)

    @staticmethod
    def _seg_name(session_edge: str) -> str:
        # /dev/shm names have a limited charset & length; hash the tail.
        import hashlib

        return "tickmb_" + hashlib.sha1(session_edge.encode()).hexdigest()[:24]

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any):
        parsed = _parse_key(put_key)
        if parsed is None:
            return super().put(from_stage, to_stage, put_key, data)
        session_edge, chunk_id = parsed
        try:
            payload = self.serialize_obj(data)
            mb = self._tx.get(session_edge)
            if mb is None:
                mb = _Mailbox(self._seg_name(session_edge), self.capacity, create=True)
                self._tx[session_edge] = mb
            if mb.try_write(chunk_id, payload):
                self._mb_metrics["mb_puts"] += 1
                self._metrics["puts"] += 1
                self._metrics["bytes_transferred"] += len(payload)
                return True, len(payload), {"tickmb": True, "size": len(payload)}
        except Exception as e:
            logger.warning("tick mailbox put fell back for %s: %s", put_key, e)
        self._mb_metrics["fallback_puts"] += 1
        return super().put(from_stage, to_stage, put_key, data)

    def get(self, from_stage: str, to_stage: str, get_key: str, metadata=None):
        parsed = _parse_key(get_key)
        if parsed is not None:
            session_edge, chunk_id = parsed
            try:
                mb = self._rx.get(session_edge)
                if mb is None:
                    try:
                        mb = _Mailbox(self._seg_name(session_edge), self.capacity, create=False)
                        self._rx[session_edge] = mb
                    except FileNotFoundError:
                        mb = None
                if mb is not None:
                    raw = mb.try_read(chunk_id)
                    if raw is not None:
                        obj = self.deserialize_obj(raw)
                        mb.write_ack(chunk_id)
                        self._mb_metrics["mb_gets"] += 1
                        self._metrics["gets"] += 1
                        return obj, len(raw)
            except Exception as e:
                logger.warning("tick mailbox get fell back for %s: %s", get_key, e)
        result = super().get(from_stage, to_stage, get_key, metadata)
        if result is not None and parsed is not None:
            # Consumed via legacy transport: still advance the ack so the
            # writer's flow control never deadlocks on a mixed sequence.
            mb = self._rx.get(parsed[0])
            if mb is not None:
                try:
                    mb.write_ack(parsed[1])
                except Exception:
                    pass
        return result

    def cleanup(self, request_id: str) -> None:
        super().cleanup(request_id)
        for table, unlink in ((self._tx, True), (self._rx, False)):
            stale = [k for k in table if k == request_id or k.startswith(request_id + "_")]
            for k in stale:
                table.pop(k).close(unlink=unlink)

    def close(self) -> None:
        for table, unlink in ((self._tx, True), (self._rx, False)):
            for mb in table.values():
                mb.close(unlink=unlink)
            table.clear()
        super().close()

    def health(self) -> dict[str, Any]:
        return {**super().health(), **self._mb_metrics}
