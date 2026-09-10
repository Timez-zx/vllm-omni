# SPDX-License-Identifier: Apache-2.0
"""Opt-in, CPU-only record of actual NIXL WRITE coverage and completion.

Never imported on the normal connector path. A selected PUSH_REG carries
identity, not model data.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

WIRE_KEY = "_numerical_transfer_probe"

_NATIVE_PD_ID = re.compile(
    r"duplex-s\.([A-Za-z0-9_-]+)\.i\.(0|[1-9][0-9]*)"
    r"\.e\.(0|[1-9][0-9]*)\.r\.stage0-([0-9a-f]{8})"
)


def cache_sync_request_identity(request_id):
    """Decode the existing physical D request contract; never guess a suffix.

    Early cache-only D registration intentionally carries no duplex payload.
    Its canonical resource ID nevertheless contains the same identity as the
    later finite D request. The eight-hex unit number is bounded to32 bits;
    metadata, when present, must match exactly rather than modulo2**32.
    """
    if not isinstance(request_id, str) or not request_id.startswith("duplex-s."):
        return None
    match = _NATIVE_PD_ID.fullmatch(request_id)
    if match is None:
        raise ValueError("Invalid canonical native P/D transfer request ID")
    encoded, incarnation, epoch, sequence = match.groups()
    try:
        session = base64.b64decode(encoded + "=" * (-len(encoded) % 4),
                                   altchars=b"-_", validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid native P/D session base64/UTF-8") from error
    if not session or base64.urlsafe_b64encode(session.encode("utf-8")).decode("ascii").rstrip("=") != encoded:
        raise ValueError("Non-canonical native P/D session encoding")
    from vllm_omni.experimental.fullduplex.engine.contracts import duplex_resource_request_id
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

    fence = DuplexFence(session_id=session, incarnation=int(incarnation), epoch=int(epoch))
    canonical = duplex_resource_request_id(fence, "stage0") + f"-{int(sequence, 16):08x}"
    if request_id != canonical:
        raise ValueError("Native P/D request ID violates the resource contract")
    return {"session_id": session, "incarnation": fence.incarnation,
            "epoch": fence.epoch, "seq": int(sequence, 16)}


def _cpu_scalar(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    # Metadata tensors can be on CPU after IPC. Never synchronize CUDA just
    # to identify a request, even with this diagnostic enabled.
    device = getattr(value, "device", None)
    if device is not None and hasattr(value, "numel"):
        if str(device) != "cpu" or value.numel() != 1:
            raise ValueError("Transfer probe identity must be CPU scalar metadata")
        value = value.item()
    return value


def _identity_carriers(request):
    observations = []
    selected = None
    for name in ("model_intermediate_buffer", "additional_information"):
        raw = getattr(request, name, None)
        info = raw
        if raw is not None and not isinstance(raw, dict):
            from vllm_omni.engine.serialization import deserialize_additional_information

            info = deserialize_additional_information(raw)
        duplex = info.get("duplex") if isinstance(info, dict) else None
        if not isinstance(duplex, dict) and isinstance(info, dict):
            duplex = {key.removeprefix("duplex."): value for key, value in info.items()
                      if isinstance(key, str) and key.startswith("duplex.")}
        fields = ("data_plane", "session_id", "incarnation", "epoch", "seq")
        observation = {"carrier": name, "raw_type": type(raw).__name__,
                       "decoded_type": type(info).__name__,
                       "duplex_type": type(duplex).__name__,
                       "identity_types": {key: type(duplex.get(key)).__name__ for key in fields}
                       if isinstance(duplex, dict) else {}}
        observations.append(observation)
        if isinstance(duplex, dict) and _cpu_scalar(duplex.get("data_plane")) is True:
            selected = (name, {key: _cpu_scalar(duplex.get(key)) for key in fields})
            break
    return selected, observations


def registration_identity(request, *, layer_names, environ=None, observe=None):
    env = os.environ if environ is None else environ
    if not env.get("MINICPMO45_NUMERICAL_PROBE_DIR"):
        return None
    selected_carrier, carriers = _identity_carriers(request)
    id_identity = cache_sync_request_identity(getattr(request, "request_id", None))
    if selected_carrier is None:
        if id_identity is None:
            if observe is not None:
                observe({"decision": "missing_native_identity", "carriers": carriers})
            if str(getattr(request, "request_id", "")).startswith("duplex-"):
                raise ValueError("Native transfer probe request has no usable identity carrier")
            return None
        source = "request_id_for_cache_sync"
        identity = id_identity
    else:
        source, duplex = selected_carrier
        identity = {key: duplex.get(key) for key in ("session_id", "incarnation", "epoch", "seq")}
    if not isinstance(identity["session_id"], str) or not identity["session_id"]:
        raise ValueError("Transfer probe requires explicit native session identity")
    for key in ("incarnation", "epoch", "seq"):
        if type(identity[key]) is not int or identity[key] < 0:
            raise ValueError(f"Transfer probe missing native {key}")
    if id_identity is not None and identity != id_identity:
        raise ValueError("Native P/D metadata disagrees with canonical request identity")
    # Same selector grammar as numerical_probe.parse_seq_selection; no torch
    # import, CUDA tensor read, or request-id parsing is needed in this hook.
    selection = env.get("MINICPMO45_NUMERICAL_PROBE_SEQS", "")
    if selection:
        selected = set()
        for part in selection.split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                start = end = int(bounds[0])
            elif len(bounds) == 2:
                start, end = map(int, bounds)
            else:
                raise ValueError("Invalid transfer-probe sequence selection")
            if not 0 <= start <= end or end - start > 1000:
                raise ValueError("Invalid/unbounded transfer-probe sequence selection")
            selected.update(range(start, end + 1))
        if identity["seq"] not in selected:
            if observe is not None:
                observe({"decision": "seq_not_selected", "identity": identity,
                         "identity_source": source, "carriers": carriers})
            return None
    sessions = json.loads(env.get("MINICPMO45_NUMERICAL_PROBE_SESSIONS", "null"))
    if sessions is not None:
        if not isinstance(sessions, list) or any(not isinstance(item, str) for item in sessions):
            raise ValueError("Transfer-probe sessions must be a JSON string array")
        if identity["session_id"] not in sessions:
            if observe is not None:
                observe({"decision": "session_not_selected", "identity": identity,
                         "identity_source": source, "carriers": carriers})
            return None
    if observe is not None:
        observe({"decision": "selected", "identity": identity,
                 "identity_source": source, "carriers": carriers})
    return {
        **identity,
        "identity_source": source,
        "d_request_id": str(request.request_id),
        "write_id": uuid.uuid4().hex,
        "group_layer_names": [list(group) for group in layer_names],
    }


class TransferProbe:
    """Preserve success/failure across partial handle polls and writer races."""

    def __init__(self, directory, *, sink=None, role=None, environ=None):
        self.directory = Path(directory)
        self._sink = sink
        self._lock = threading.RLock()
        self._states = {}
        self._receivers = {}
        self._next_handle_id = 0
        self._observations = 0
        env = os.environ if environ is None else environ
        self._environ = {key: value for key, value in env.items()
                         if key.startswith("MINICPMO45_NUMERICAL_PROBE_")}
        self._environ["MINICPMO45_NUMERICAL_PROBE_DIR"] = str(directory)
        if role is not None:
            self._emit({"evidence": {}}, "kv_probe_initialized", process_role=role,
                       seq_selection=self._environ.get("MINICPMO45_NUMERICAL_PROBE_SEQS", ""))

    def registration(self, request, *, layer_names):
        def observe(fields):
            with self._lock:
                # Selected rows always leave evidence; only the first eight
                # excluded rows are recorded to diagnose carrier/selector
                # mistakes without growing logs on every unselected unit.
                if self._observations < 8 or fields["decision"] in ("selected", "missing_native_identity"):
                    self._emit({"evidence": {"d_request_id": str(request.request_id)}},
                               "kv_registration_observed", **fields)
                self._observations += 1
        return registration_identity(request, layer_names=layer_names,
                                     environ=self._environ, observe=observe)

    def _emit(self, state, event, **fields):
        record = {
            "schema_version": 1,
            "event": event,
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
            **state["evidence"],
            **fields,
        }
        if self._sink is not None:
            self._sink(record)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"kv-transfer-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def begin(self, request_id, registration, source_blocks, *, block_size):
        identity = registration.get(WIRE_KEY)
        if identity is None:
            return False
        if identity["d_request_id"] != registration["request_id"]:
            raise ValueError("Transfer probe D request identity mismatch")
        destination = registration["local_block_ids"]
        if destination and not isinstance(destination[0], (list, tuple)):
            destination = (list(destination),)
        groups = registration.get("source_block_indices")
        if groups is None:
            offset = int(registration["source_block_offset"])
            groups = [list(range(offset, offset + len(group))) for group in destination]
        if len(groups) != len(destination) or len(source_blocks) != len(destination):
            raise ValueError("Transfer probe group mismatch")
        if groups and len(identity["group_layer_names"]) != len(groups):
            raise ValueError("Transfer probe layer/group mapping mismatch")
        for logical, source, target in zip(groups, source_blocks, destination):
            if len(logical) != len(source) or len(source) != len(target):
                raise ValueError("Transfer probe block count mismatch")
            if logical != sorted(set(logical)) or any(index < 0 for index in logical):
                raise ValueError("Transfer probe needs exact ordered logical blocks")
        remote = int(registration["remote_prompt_tokens"])
        if block_size != int(registration["decode_block_size"]) or block_size <= 0:
            raise ValueError("Transfer probe block size mismatch")
        state = {
            "evidence": {
                **identity,
                "p_request_id": str(request_id),
                "block_size": block_size,
                "remote_prompt_tokens": remote,
                "matched_prefix_tokens": int(registration["matched_prefix_tokens"]),
                "source_logical_block_indices": [list(group) for group in groups],
                # These are allocator block IDs, before the upstream logical-
                # to-kernel-block expansion. Do not label them kernel IDs.
                "source_allocator_block_ids": [list(group) for group in source_blocks],
                "destination_allocator_block_ids": [list(group) for group in destination],
                "position_ranges_by_group": [
                    [[index * block_size, min((index + 1) * block_size, remote)]
                     for index in group if index * block_size < remote]
                    for group in groups
                ],
            },
            "handles": [], "missing_submission": False, "failed": False,
            "sealed": False, "done": False,
        }
        with self._lock:
            if request_id in self._states:
                raise ValueError("Transfer probe duplicate in-flight P request")
            self._states[request_id] = state
            self._emit(state, "kv_write_selected", complete=False)
        return True

    def submitted(self, request_id, handle):
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return
            if handle is None:
                state["missing_submission"] = True
                self._emit(state, "kv_write_submit_failed", complete=False, success=False)
            else:
                # Current NIXL returns an opaque nixl_xfer_handle, not an
                # integer. Never coerce it, access its private native handle,
                # or own its release. Keep a strong reference until the real
                # completion poll retires this WRITE so Python object reuse
                # cannot alias two probe IDs.
                existing = next((item for item in state["handles"]
                                 if item["object"] is handle), None)
                if existing is None:
                    self._next_handle_id += 1
                    existing = {"object": handle,
                                "probe_handle_id": f"probe-{self._next_handle_id}"}
                    state["handles"].append(existing)
                self._emit(state, "kv_write_submitted",
                           probe_handle_id=existing["probe_handle_id"],
                           handle_identity_source="probe_local_object",
                           handle_type=f"{type(handle).__module__}.{type(handle).__qualname__}",
                           complete=False)

    def failed(self, request_id):
        with self._lock:
            state = self._states.get(request_id)
            if state is not None:
                state["failed"] = True
            receiver = self._receivers.get(request_id)
            if receiver is not None:
                receiver["failed"] = True

    def register_receiver(self, registration):
        identity = registration.get(WIRE_KEY)
        if identity is None:
            return
        request_id = registration["request_id"]
        if identity["d_request_id"] != request_id:
            raise ValueError("Transfer probe receiver identity mismatch")
        with self._lock:
            state = {"evidence": dict(identity), "failed": False}
            self._receivers[request_id] = state
            self._emit(state, "kv_receive_registered", complete=False)

    def receive_ready(self, request_ids):
        with self._lock:
            for request_id in request_ids:
                state = self._receivers.pop(request_id, None)
                if state is not None:
                    # Scheduler promotion happens after this worker-side
                    # receive observation, not after P's next handle poll.
                    self._emit(state, "kv_receive_ready", complete=True,
                               success=not state["failed"])

    def completed(self, request_ids):
        with self._lock:
            for request_id in request_ids:
                state = self._states.get(request_id)
                if state is not None:
                    state["done"] = True
                    self._finish(request_id, state)

    def seal(self, request_id, *, no_write=False):
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return
            state["sealed"] = True
            if no_write:
                self._emit(state, "kv_write_not_needed", complete=True,
                           success=True, no_write=True, handle_count=0)
                del self._states[request_id]
            elif not state["handles"]:
                self._emit(state, "kv_write_not_submitted", complete=False,
                           success=False, handle_count=0)
                del self._states[request_id]
            else:
                self._finish(request_id, state)

    def _finish(self, request_id, state):
        if not (state["sealed"] and state["done"]):
            return
        success = bool(state["handles"]) and not (state["failed"] or state["missing_submission"])
        self._emit(state, "kv_write_completed", complete=True, success=success,
                   no_write=False, handle_count=len(state["handles"]))
        del self._states[request_id]
