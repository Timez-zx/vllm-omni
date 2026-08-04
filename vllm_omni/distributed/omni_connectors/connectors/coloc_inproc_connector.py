# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)


def _colocated_group() -> frozenset[str]:
    """Stage ids sharing one process, per VLLM_OMNI_COLOCATE_STAGES
    ("guest:host[,guest:host]"). Empty when colocation is off."""
    raw = os.environ.get("VLLM_OMNI_COLOCATE_STAGES", "").strip()
    ids: set[str] = set()
    if raw:
        for pair in raw.split(","):
            guest, _, host = pair.partition(":")
            ids.add(guest.strip())
            ids.add(host.strip())
    return frozenset(ids)


def _payload_nbytes(obj: Any) -> int:
    """Best-effort byte count so transfer telemetry stays honest -- an in-proc
    hand-off that reported 0 would make the before/after comparison lie."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_payload_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_nbytes(v) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    return 8


class ColocInProcConnector(OmniConnectorBase):
    """Edge-routing connector for colocated stages (speech-pair merge).

    One connector instance serves a stage's incoming AND outgoing edges, so
    a stage that receives cross-process but sends in-process needs routing
    PER EDGE: put/get already carry (from_stage, to_stage), and an edge whose
    both ends live in the colocated group is served from a process-global
    take-once dict (tensors pass by REFERENCE -- no msgpack, no /dev/shm, no
    copies); every other edge delegates to a plain SharedMemoryConnector.

    Measured motivation: with talker+code2wav in one process, the SHM hop's
    serialize/copy work runs inside the shared GIL and fights both engine
    loops (128-user rtf 0.63 with vocoder graphs vs 0.75 separate-process).

    Semantics preserved from the SHM path (each has a bug behind it):
      * struct payloads are stored as PLAIN DICTS via data_entry_keys.to_dict
        -- the receiver calls ``.get()`` on payloads and crashes on a struct;
      * get is non-blocking, None on miss (the recv loop is a single-pass
        poller and must not block);
      * re-put on the same key overwrites (preempted requests re-put);
      * take-once: get pops the key, so ordering stays with the key sequence;
      * cleanup(request_id) purges full/prefix/suffix key matches, both here
        and in the delegate.

    Caveat (accepted, documented): tensors are shared by reference, so a
    sender that mutated a shipped tensor before the receiver consumed it
    would corrupt the payload. The chunk builder constructs fresh tensors
    per payload (the accumulate path concatenates into new storage), so this
    does not occur today.
    """

    # Process-global: the sender's and receiver's connector INSTANCES differ,
    # the store must not.
    _store: dict[str, tuple[Any, int]] = {}
    _store_lock = threading.RLock()

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self._group = _colocated_group()
        self._delegate = SharedMemoryConnector(config)
        self._metrics = {"inproc_puts": 0, "inproc_gets": 0, "inproc_bytes": 0}
        if not self._group:
            logger.warning(
                "ColocInProcConnector constructed without VLLM_OMNI_COLOCATE_STAGES; "
                "every edge will delegate to SharedMemoryConnector"
            )

    def _intra(self, from_stage: str, to_stage: str) -> bool:
        return str(from_stage) in self._group and str(to_stage) in self._group

    def put(
        self, from_stage: str, to_stage: str, put_key: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if not self._intra(from_stage, to_stage):
            return self._delegate.put(from_stage, to_stage, put_key, data)
        try:
            payload = data
            # Mirror the serializer round-trip's ONE semantic effect (struct ->
            # plain dict, None fields dropped) without the bytes.
            from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict

            if isinstance(payload, OmniPayloadStruct):
                payload = to_dict(payload)
            size = _payload_nbytes(payload)
            with self._store_lock:
                type(self)._store[put_key] = (payload, size)
            self._metrics["inproc_puts"] += 1
            self._metrics["inproc_bytes"] += size
            return True, size, None
        except Exception as e:
            logger.error(f"ColocInProcConnector put failed for {put_key}: {e}")
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not self._intra(from_stage, to_stage):
            return self._delegate.get(from_stage, to_stage, get_key, metadata)
        with self._store_lock:
            entry = type(self)._store.pop(get_key, None)
        if entry is None:
            return None
        self._metrics["inproc_gets"] += 1
        return entry

    def cleanup(self, request_id: str) -> None:
        with self._store_lock:
            stale = [
                k
                for k in type(self)._store
                if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
            ]
            for k in stale:
                type(self)._store.pop(k, None)
        if stale:
            logger.debug("ColocInProcConnector cleanup dropped %d unconsumed payload(s)", len(stale))
        self._delegate.cleanup(request_id)

    def health(self) -> dict[str, Any]:
        with self._store_lock:
            depth = len(type(self)._store)
        return {
            "status": "healthy",
            "inproc_store_depth": depth,
            **self._metrics,
            "delegate": self._delegate.health(),
        }

    def close(self) -> None:
        self._delegate.close()
