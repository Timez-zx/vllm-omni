# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, unflatten_payload

from ..adapter import construct_next_stage_streaming_input_prompt
from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec, stage_receives_chunks
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)

# Emit one INFO line per chunk per stage edge with its size and timings. Off by default
# because this is a hot path; the running totals in `_tx_totals` are accumulated either way,
# so a caller can read them without turning logging on.
_LOG_TRANSFER = os.environ.get("VLLM_OMNI_LOG_TRANSFER", "0") not in ("0", "false", "False", "")

# [Tick engine WP4-lite] VLLM_OMNI_TEMPORAL_INLINE_SEND=1: chunk sends happen
# synchronously at the point save_async is called (T+0 of the producing step)
# instead of via the background save thread. See save_async.
_INLINE_SEND = os.environ.get("VLLM_OMNI_TEMPORAL_INLINE_SEND", "0") not in ("0", "false", "False", "")



def _request_is_prefill_only(request: Any) -> bool:
    """Late import: tts_utils lives under model_executor and importing it at module scope
    would tie this transport-level file to a model package."""
    try:
        from vllm_omni.model_executor.stage_input_processors.tts_utils import (
            request_is_prefill_only,
        )
    except Exception:
        return False
    return request_is_prefill_only(request)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        active_stream_window = int(getattr(model_config, "active_stream_window", 0) or 0)
        model_max_num_seqs = int(getattr(model_config, "max_num_seqs", self.scheduler_max_num_seqs) or 0)
        if model_max_num_seqs <= 0:
            model_max_num_seqs = self.scheduler_max_num_seqs
        self._active_window = min(active_stream_window, model_max_num_seqs) if active_stream_window > 0 else 0
        if self._active_window > 0:
            logger.info(
                "Bounded active-stream window enabled: K=%d. "
                "Multi-replica deployments require sticky per-stream routing across Stage 1 "
                "replicas (each replica owns an independent active-set; without sticky routing, "
                "a stream can be active on one replica and non-active on another and both will "
                "race to evict it).",
                self._active_window,
            )
        self.connector = self.create_connector(model_config)
        self.receives_chunks = stage_receives_chunks(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func: Callable[..., OmniPayloadStruct | None] | None = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # Accumulated transfer cost per request: bytes sent on this stage's outgoing edge
        # plus the time spent building and putting each payload. Read via tx_totals().
        self._tx_totals: dict[str, dict[str, Any]] = {}
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        # Segment-local chunk counter: incremented alongside put_req_chunk
        # but popped at segment boundaries (unlike put_req_chunk which is
        # request-global for connector key continuity).
        self.ramp_chunk_count: dict[str, int] = defaultdict(int)
        self.upstream_exhausted_requests: set[str] = set()
        self.segment_finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.requests_origin_status = {}
        # Requests whose most recently loaded chunk OPENED a new segment.
        # Two producer conventions mark that: an explicit
        # meta.replace_streaming_prompt (MiniCPM-o), or -- the Qwen3-Omni
        # append-style convention, which ships no marker -- the first
        # data-bearing chunk after a segment_finished chunk (tracked via
        # _expect_segment_opener). Such a request must re-enter scheduling
        # through the WAITING path so the runner receives it as
        # NewRequestData and runs its full segment refresh
        # (_update_streaming_request + _update_streaming_input_additional_info:
        # prompt/mrope re-init, num_processed_tokens=0). Resuming it as
        # RUNNING delivers the payload on the cached path, which refreshes
        # none of that; the stale num_processed_tokens then slices past the
        # fresh (shorter) prefill rows and the model runs on empty or
        # uninitialized embeddings. Observed as all three campaign death
        # modes: IndexError on a 0-row hidden_states, vectorized_gather
        # device-side asserts from batch-mates, and clamped-prefix audio
        # corruption with no exception.
        self._segment_replaced_reqs: set[str] = set()
        # Receive-side boundary memory: req ids whose last consumed chunk
        # ended a segment, so the next data-bearing chunk is a segment
        # opener. Survives boundary-only segments in between (those are
        # segment_finished with no data and keep the flag set).
        self._expect_segment_opener: set[str] = set()
        self._active_streams: dict[str, Any] = {}
        # Private hold-queue for non-active running requests. Restored to
        # running_queue inside restore_queues(). Avoids calling
        # waiting_queue.prepend_requests mid-step, which trips vllm's
        # per-step LogitsProcessor invariant
        # ("Cannot register new removed request after self.removed has
        #   been read").
        self._held_non_active: deque[Any] = deque()
        self.requests_num_chunks_sent: dict[str, int] = defaultdict(int)
        # Boundary-only segment cap (receive side, keyed by internal id).
        #
        # A prefill-only append ships NOTHING to this stage but the segment
        # boundary, yet the parked request still resumes and free-runs decode
        # from its 1-token placeholder until it samples its own stop -- usually
        # a few unprompted codec frames, measured up to ~950 (50 s), with the
        # next REAL segment serialized behind the junk. When a segment finishes
        # having delivered zero data-bearing chunks, there is nothing legitimate
        # to speak, so the segment is capped at one token: check_stop fires
        # FINISHED_LENGTH_CAPPED after the first sample and the segment stop
        # still ships, which is the contract that must not break (withholding
        # the segment entirely desynchronised the stages and killed the engine).
        self.segment_payload_chunks: dict[str, int] = defaultdict(int)
        self._boundary_cap_saved_max_tokens: dict[str, int] = {}
        self._pending_streaming_prefills: dict[str, dict] = {}

    @staticmethod
    def _is_truthy_scalar(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return value.numel() == 1 and bool(value.item())
        return bool(value) if value is not None else False

    @staticmethod
    def _confirmed_num_computed_tokens(request: Request) -> int:
        # vLLM async scheduling advances num_computed_tokens with output
        # placeholders before the corresponding token is committed. Connector
        # chunk send watermarks must use only committed tokens.
        num_computed = int(getattr(request, "num_computed_tokens", 0))
        num_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
        return max(0, num_computed - num_placeholders)

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0 or not self.receives_chunks:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        multimodal_output: dict[str, Any] | None = None,
        request: Request | None = None,
        is_segment_finished: bool = False,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        For streaming input request ``is_segment_finished`` marks the end
        of the current realtime input segment. It is intentionally separate
        from ``request.is_finished()``: a resumable `/v1/realtime` session
        can finish one audio segment and later continue with another segment
        under the same external request id. For other requests, it is the same
        as ``request.is_finished()``.

        Args:
            multimodal_output: Per-request multimodal output dictionary
            request: Request object
            is_segment_finished: whether the segment of request is finished
        """
        is_finished = request.is_finished() and not request.resumable

        confirmed_num_computed_tokens = self._confirmed_num_computed_tokens(request)

        # If the request is preempted, skip the already saved chunks.
        if confirmed_num_computed_tokens < self.requests_num_chunks_sent.get(request.external_req_id, 0):
            logger.warning(
                f"Enqueue save_async for request {request.external_req_id}, "
                f"request.num_computed_tokens={request.num_computed_tokens}, "
                f"request.num_output_placeholders={getattr(request, 'num_output_placeholders', 0)}, "
                f"previous_chunks_sent={self.requests_num_chunks_sent.get(request.external_req_id, 0)}"
            )
            # [Boundary-loss fix, drop point A] a task carrying the SEGMENT
            # BOUNDARY must never be silently consumed: the flag is emitted
            # exactly once (talker stop) and nothing downstream re-emits it --
            # dropping it here leaves the consumer stage waiting forever and
            # the client's turn hangs to its 240 s watchdog. Ship a
            # boundary-only chunk (payload stripped) instead of returning.
            if not (is_finished or is_segment_finished):
                return
            multimodal_output = None

        self.requests_num_chunks_sent[request.external_req_id] = confirmed_num_computed_tokens
        task = {
            "multimodal_output": multimodal_output,
            "request": request,
            "is_finished": is_finished,
            "is_segment_finished": is_segment_finished,
            # Snapshot the prefill-only decision NOW, on the scheduler's thread, while
            # request.sampling_params still belongs to the segment that produced this
            # output. The task holds a REFERENCE to the request, and by the time the save
            # thread dequeues it the next streaming update may have replaced the params.
            # Both directions of that race were hit within one hour: the marker read late
            # once said "present" for a real turn (its whole reply withheld from the
            # talker), and once said "absent" for an append (its payload shipped to stage
            # 1 as the NEXT segment's first chunk, off-by-one, ending in the talker
            # asserting in indexSelect and the engine dying). A per-segment fact must be
            # captured while the segment's state is live, not re-derived later.
            "prefill_only": _request_is_prefill_only(request),
        }
        if _INLINE_SEND:
            # [Tick engine WP4-lite] send at T+0 on the scheduler thread
            # instead of hopping through the save-thread queue. Removes one
            # thread wakeup + queue-order head-of-line from the chunk path,
            # and the enqueue-time snapshot races documented above become
            # moot (state is live at send time). Cost: payload build + shm
            # put inline -- sub-ms for decode chunks, ~10 ms once per turn
            # for the first chunk's prefill embeds, paid from tick slack.
            try:
                self._send_single_request(task)
            except Exception as e:
                logger.warning(
                    f"[OmniTransfer] inline send failed for "
                    f"{getattr(request, 'external_req_id', '?')}: {e}"
                )
            return
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    @staticmethod
    def _payload_has_data(payload: Any) -> bool:
        """True if the decoded payload carries anything beyond meta.

        msgspec omits default (None) fields on the wire, so a boundary-only
        payload decodes as {"meta": {...}} with no data key present at all.
        Values are sub-structs decoded as dicts; never bool() a raw tensor.
        """
        for key in ("embed", "hidden_states", "ids", "codes", "hidden", "latent"):
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                if value:
                    return True
            else:
                return True
        return False

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            meta = payload_data.get("meta", {})
            payload_finished = self._is_truthy_scalar(meta.get("finished"))
            payload_segment_finished = self._is_truthy_scalar(meta.get("is_segment_finished"))
            if self.model_mode == "ar":
                request.additional_information = payload_data
                replace_prompt = meta.get("replace_streaming_prompt") is True
                # Segment-opener detection for the resume in
                # _process_chunk_queue (see _segment_replaced_reqs): either
                # the producer says so explicitly (replace_prompt), or this
                # is the first data-bearing chunk after a segment_finished
                # chunk (append-style producers ship no marker).
                if replace_prompt or (
                    req_id in self._expect_segment_opener and self._payload_has_data(payload_data)
                ):
                    self._segment_replaced_reqs.add(req_id)
                    self._expect_segment_opener.discard(req_id)
                if getattr(request, "resumable", False) and (chunk_id > 0 or replace_prompt):
                    # For new streaming input segment, we should update prompt from payload
                    construct_next_stage_streaming_input_prompt(payload_data, request)

                # Boundary-only segment cap. Structural, not a plumbed flag: the
                # prefill-only marker itself was lost on this path twice (see the
                # snapshot comment in _send_single_request), but "the segment
                # finished and no chunk of it carried data" is readable right
                # here and identifies the same set of segments. Mutating
                # request.max_tokens is race-free at this point: the request is
                # parked in WAITING_FOR_CHUNK while this thread runs, and
                # _finished_load_reqs.add below is what makes it runnable again.
                has_data = self._payload_has_data(payload_data)
                if has_data:
                    self.segment_payload_chunks[req_id] += 1
                    saved = self._boundary_cap_saved_max_tokens.pop(req_id, None)
                    if saved is not None:
                        request.max_tokens = saved
                        # Presence probe for the RESTORE side: a missed restore
                        # caps a real reply at one token, which must be visible.
                        logger.info(
                            "[boundary-cap] restored max_tokens=%d for req %s",
                            saved, req_id,
                        )
                if payload_segment_finished:
                    if not has_data and self.segment_payload_chunks.get(req_id, 0) == 0:
                        if req_id not in self._boundary_cap_saved_max_tokens:
                            self._boundary_cap_saved_max_tokens[req_id] = request.max_tokens
                        request.max_tokens = 1
                        logger.info(
                            "[boundary-cap] boundary-only segment: capping req %s "
                            "at 1 token (was %d)",
                            req_id, self._boundary_cap_saved_max_tokens[req_id],
                        )
                    self.segment_payload_chunks.pop(req_id, None)

                if payload_finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)
                    # The next data-bearing chunk for this request opens a
                    # new segment (boundary-only segments in between keep
                    # this set: they are segment_finished with no data).
                    self._expect_segment_opener.add(req_id)
            else:
                if payload_finished:
                    self.upstream_exhausted_requests.add(req_id)
                    request.resumable = False
                if payload_segment_finished:
                    self.segment_finished_requests.add(req_id)

                new_ids = payload_data.get("codes", {}).get("audio")
                has_tensor_codes = isinstance(new_ids, torch.Tensor)
                use_tensor_codes = has_tensor_codes and new_ids.ndim >= 2
                if use_tensor_codes:
                    request.prompt_token_ids = [0] if new_ids.numel() > 0 else []
                elif has_tensor_codes:
                    new_ids = new_ids.tolist()
                elif new_ids is None:
                    new_ids = []
                    request.prompt_token_ids = new_ids
                if not use_tensor_codes:
                    request.prompt_token_ids = new_ids
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key == "codes":
                        if use_tensor_codes and isinstance(value, dict):
                            existing_sub = info.get(key)
                            merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                            merged_sub.update(value)
                            info[key] = merged_sub
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                has_new_ids = bool(new_ids.numel()) if use_tensor_codes else bool(new_ids)
                if not has_new_ids and payload_segment_finished:
                    # Preserve an explicit scheduler boundary even when it
                    # contains no new codec frames.
                    request.prompt_token_ids = [0]
                if not has_new_ids and not payload_finished and not payload_segment_finished:
                    # The base recv loop treats False as "not ready yet" and
                    # requeues the request. Do not mark an empty non-terminal
                    # chunk as ready, otherwise Stage1 can consume before the
                    # first DAC frame arrives.
                    return False

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _record_tx(
        self,
        *,
        external_req_id: str,
        stage_id: int,
        next_stage_id: int,
        chunk_id: int,
        size: int,
        build_ms: float,
        put_ms: float,
    ) -> None:
        """Record what a stage-to-stage payload actually cost to build and send.

        Until now nothing measured this edge. ``StageRequestStats`` hardcodes
        ``rx_transfer_bytes=0`` / ``rx_decode_time_ms=0.0``, and
        ``Orchestrator._emit_tx_edge`` passes a literal 0 to the size histogram with a
        docstring noting that a follow-up should "plumb that from the connector adapter".
        This is that measurement, taken where the numbers already exist: ``connector.put``
        returns the serialized size, and both phases can be timed in place.

        Why it is worth measuring rather than assuming negligible: for a multi-stage omni
        pipeline the payload is per-position, not per-request. The Qwen3-Omni thinker ships
        the prompt's embeddings *and* last-layer hidden states to the talker -- two
        [L, hidden] bf16 tensors, i.e. 4 * hidden bytes per prompt token. At hidden=2048
        that is 8 KB/token, so a 35k-token prompt is ~290 MB on a single edge, for a single
        turn. Measured on one H100 at roughly 225 MB/s effective (detach+cpu, serialize,
        shared-memory write), that is over a second of wall clock sitting between the
        thinker's last token and the talker's first -- which the logs previously reported
        as ``transfers=[0->1=0.00ms]``.

        Kept deliberately cheap: two ``perf_counter`` calls per chunk and a dict update.
        The per-chunk log line is opt-in via ``VLLM_OMNI_LOG_TRANSFER=1`` because this runs
        once per chunk per stage edge; the running totals are always accumulated so a
        caller can read them without turning logging on.
        """
        acc = self._tx_totals.setdefault(external_req_id, {"bytes": 0, "build_ms": 0.0, "put_ms": 0.0, "chunks": 0})
        acc["bytes"] += int(size or 0)
        acc["build_ms"] += float(build_ms)
        acc["put_ms"] += float(put_ms)
        acc["chunks"] += 1
        if _LOG_TRANSFER:
            logger.info(
                "[OmniTransfer] req=%s edge=%d->%d chunk=%d bytes=%d build_ms=%.2f put_ms=%.2f "
                "cum_bytes=%d cum_put_ms=%.1f",
                external_req_id,
                stage_id,
                next_stage_id,
                chunk_id,
                int(size or 0),
                build_ms,
                put_ms,
                acc["bytes"],
                acc["put_ms"],
            )

    def tx_totals(self, external_req_id: str) -> dict[str, Any] | None:
        """Accumulated transfer cost for a request, or None if nothing was sent for it.

        Exposed so a caller that does have a metrics handle can report a real size instead
        of the placeholder zero. The adapter itself lives in the engine-core process and has
        no route to the orchestrator's aggregator, which is why the plumbing stops here.
        """
        return self._tx_totals.get(external_req_id)

    def _send_single_request(self, task: dict):
        raw_mm = task["multimodal_output"]
        multimodal_output = unflatten_payload(raw_mm) if isinstance(raw_mm, Mapping) else raw_mm
        request = task["request"]
        is_finished = task["is_finished"]
        is_segment_finished = task["is_segment_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id

        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data: OmniPayloadStruct | None = None
        _t_build0 = time.perf_counter()
        # Skip the payload BUILD for a prefill-only append, using the snapshot taken at
        # enqueue time. Reading the live request here does not work: sampling_params (which
        # carries the marker) has already been replaced by the next streaming update by the
        # time this thread runs, so the processor's own check silently never fires -- observed,
        # with `shipping nothing` absent from the log while the crash it prevents happened.
        #
        # Only the build is skipped. The boundary marker, the chunk id and the key namespace
        # all proceed exactly as before: withholding THOSE is what four earlier attempts did,
        # and it left stage 1 unable to re-admit the talker cleanly. What must not be shipped
        # is the append's PREFILL tensor mislabelled `embed.decode` -- the runner copies it
        # into a one-row decode slot and raises
        # `output with shape [1, 1024] doesn't match the broadcast shape [222, 1024]`.
        _prefill_only = bool(task.get("prefill_only"))
        if _prefill_only:
            logger.info(
                "[prefill-only] skipping payload build, stage %s -> %s, req %s (boundary still ships)",
                stage_id, next_stage_id, external_req_id,
            )
        if self.custom_process_next_stage_input_func and not _prefill_only:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    multimodal_output=multimodal_output,
                    request=request,
                    # Existing processors use is_finished as a flush signal.
                    # Terminal stops no longer count as segment boundaries
                    # (is_segment_finished is False when the request finishes,
                    # see #5383), but the processor must still flush its
                    # accumulated tail on the terminal chunk — otherwise the
                    # downstream stage receives the finished marker without
                    # the final payload (#5413).
                    is_finished=is_segment_finished or is_finished,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")
        _build_ms = (time.perf_counter() - _t_build0) * 1000.0

        if payload_data is None:
            if not (is_segment_finished or is_finished):
                return
            if _prefill_only and chunk_id == 0:
                # A prefill-only FIRST chunk: a compression shadow's seed. For mid-session
                # appends the boundary must still ship -- withholding it desyncs a talker
                # that already runs this request (the four earlier attempts documented
                # above) -- but at chunk 0 the talker has never seen the request, and a
                # tensor-less boundary would BE its bring-up payload: an untested shape.
                # Suppressing the ship (put_req_chunk only increments on a successful put,
                # below) means stage 1 first hears of this request from the first REAL
                # turn, as a normal full chunk 0.
                logger.info(
                    "[prefill-only] chunk-0 boundary suppressed, stage %s -> %s, req %s",
                    stage_id, next_stage_id, external_req_id,
                )
                return
            # Segment/request finish markers must still reach downstream even when
            # the processor has no tensor payload.
            payload_data = OmniPayloadStruct()
        if payload_data.meta is None:
            payload_data.meta = MetaStruct()
        payload_data.meta.finished = torch.tensor(is_finished, dtype=torch.bool)
        if payload_data.meta.is_segment_finished is None:
            payload_data.meta.is_segment_finished = torch.tensor(is_segment_finished, dtype=torch.bool)

        _t_put0 = time.perf_counter()
        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )
        # [Boundary-loss fix, drop point B] a failed put normally just loses
        # one data chunk (bad but survivable); losing the chunk that carries
        # finished/is_segment_finished hangs the consumer stage forever, and
        # nothing re-emits it. Retry flagged chunks, loudly.
        if not success and (is_finished or is_segment_finished):
            for _attempt in range(3):
                time.sleep(0.005 * (_attempt + 1))
                success, size, metadata = self.connector.put(
                    from_stage=str(stage_id),
                    to_stage=str(next_stage_id),
                    put_key=connector_put_key,
                    data=payload_data,
                )
                if success:
                    break
            if not success:
                logger.error(
                    "[OmniTransfer] BOUNDARY chunk PUT FAILED after retries: %s "
                    "(stage %s->%s) -- downstream segment will hang",
                    connector_put_key, stage_id, next_stage_id,
                )
        _put_ms = (time.perf_counter() - _t_put0) * 1000.0

        if success:
            self.put_req_chunk[external_req_id] += 1
            self.ramp_chunk_count[external_req_id] += 1
            # Not a real conflict: upstream's ramp counter and this branch's transfer
            # timing are independent additions on the same success path, so both stay.
            # The timing exists because stage-to-stage transfer cost used to be reported
            # as a hardcoded zero, which made "is the speech stage the bottleneck?"
            # unanswerable -- see workflow.md section 9.
            self._record_tx(
                external_req_id=external_req_id,
                stage_id=stage_id,
                next_stage_id=next_stage_id,
                chunk_id=chunk_id,
                size=size,
                build_ms=_build_ms,
                put_ms=_put_ms,
            )
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            # Sender uses struct attr access here; the receive path in
            # `_load_one_request` / `_update_request_payload` reads dict keys.
            # That asymmetry is intentional: `OmniMsgpackDecoder` is type-erased
            # (no target type), so the wire round-trips struct -> dict. If you
            # change the schema, update both ends — see test_wire_round_trip.
            finished_flag = payload_data.meta.finished if payload_data.meta is not None else None
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id)

        if is_segment_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            self.requests_num_chunks_sent.pop(external_req_id, None)
            self.ramp_chunk_count.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    def is_done_receiving_chunks(self, request_id: str) -> bool:
        """Return True if the request should stop polling upstream chunks.

        Covers both the whole-request marker (``upstream_exhausted_requests``)
        and the per-segment marker (``segment_finished_requests``) used while
        waiting for the next streaming input slice. Neither means this
        stage's own generation is done -- see vllm-project/vllm-omni#5349.
        """
        return request_id in self.upstream_exhausted_requests or request_id in self.segment_finished_requests

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Must also purge the request from the chunk-parking deques
        (``waiting_for_chunk_waiting_requests`` / ``_running_requests`` /
        ``_held_non_active``): otherwise a caller that calls
        ``restore_queues()`` without ``scheduler_requests`` (e.g. a unit
        test, or any future caller not synced with the scheduler's own
        request-removal timing) would re-admit an already-finished
        request into the visible queue, which ``_promote_active_streams``
        would then FIFO-promote ahead of genuinely-waiting requests. See
        vllm-project/vllm-omni#5349's active-stream-window tests.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self._active_streams.pop(request_id, None)
        self.upstream_exhausted_requests.discard(request_id)
        self.segment_finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.segment_payload_chunks.pop(request_id, None)
        self._boundary_cap_saved_max_tokens.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self._segment_replaced_reqs.discard(request_id)
        self._expect_segment_opener.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)
        self._discard_from_chunk_deque(self.waiting_for_chunk_waiting_requests, request_id)
        self._discard_from_chunk_deque(self.waiting_for_chunk_running_requests, request_id)
        self._discard_from_chunk_deque(self._held_non_active, request_id)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    @staticmethod
    def _discard_from_chunk_deque(deque_list: deque[Any], request_id: str) -> None:
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id != request_id:
                deque_list.append(request)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        self.requests_num_chunks_sent.pop(external_req_id, None)
        self.ramp_chunk_count.pop(external_req_id, None)
        self._pending_streaming_prefills.pop(external_req_id, None)
        # Log the request's total before dropping it -- for a long streaming session this is
        # the only place the accumulated cost of the edge is ever visible.
        totals = self._tx_totals.pop(external_req_id, None)
        if _LOG_TRANSFER and totals:
            logger.info(
                "[OmniTransfer] req=%s TOTAL chunks=%d bytes=%d build_ms=%.1f put_ms=%.1f",
                external_req_id,
                totals["chunks"],
                totals["bytes"],
                totals["build_ms"],
                totals["put_ms"],
            )

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

        # [Boundary-loss fix, hygiene] unconsumed connector segments used to
        # outlive the request forever: this adapter never told the connector
        # to reclaim them, so every aborted/rolled session leaked its unread
        # /dev/shm segments and one 0-byte lockfile per chunk key (measured:
        # 475 leaked lockfiles after one day's experiments). Best-effort by
        # contract -- the connector matches keys by request-id prefix.
        try:
            self.connector.cleanup(external_req_id)
        except Exception:
            logger.debug("connector cleanup failed for %s", external_req_id, exc_info=True)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        *,
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Process pending chunks for waiting and running queues.

        When ``scheduler_requests`` is provided, purges any
        ``waiting_for_chunk_*_requests`` deque entries whose
        ``request_id`` is no longer tracked by it (e.g. after a
        mid-flight abort that ran ``Scheduler._free_request``) before
        processing chunks. Without this purge, ``restore_queues`` would
        later re-inject the freed ``Request`` onto ``running_queue`` and
        the worker's ``_update_states`` would crash with ``KeyError``
        reading ``self.requests[req_id]``. See vllm-project/vllm-omni#3736.

        ``scheduler_requests`` is keyword-only and optional; production
        schedulers always pass their live request map, while legacy
        callers that don't track aborts may omit it to keep the prior
        (unguarded) behaviour.
        """
        if not self.receives_chunks:
            return
        if self.connector.stage_id == 0:
            return

        # Purge deque entries whose request was freed mid-flight (abort →
        # Scheduler._free_request) before any chunk processing, so neither
        # the legacy nor the active-stream path can re-inject a zombie
        # Request onto the queues. See vllm-project/vllm-omni#3736.
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)

        if self._active_window <= 0:
            self._process_chunk_queue_legacy(
                waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
            )
            self._process_chunk_queue_legacy(
                running_queue,
                self.waiting_for_chunk_running_requests,
                RequestStatus.RUNNING,
                self._finished_load_reqs,
            )
            while len(running_queue) > self.scheduler_max_num_seqs:
                request = running_queue.pop()
                request.status = RequestStatus.PREEMPTED
                waiting_queue.prepend_requests([request])
            return

        self._promote_active_streams(running_queue)
        self._promote_active_streams(waiting_queue)
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        self._promote_active_streams(waiting_queue)
        self._preempt_non_active_running(waiting_queue, running_queue)

    def _promote_active_streams(self, queue: Any) -> None:
        if len(self._active_streams) >= self._active_window:
            return
        for request in list(queue):
            if len(self._active_streams) >= self._active_window:
                return
            request_id = request.request_id
            if request_id in self._active_streams:
                continue
            # Iterating the existing queue preserves FIFO admission.
            self._active_streams[request_id] = request

    def _ensure_active_stream(self, request: Request) -> bool:
        if self._active_window <= 0:
            return True
        request_id = request.request_id
        if request_id in self._active_streams:
            self._active_streams[request_id] = request
            return True
        if len(self._active_streams) >= self._active_window:
            return False
        self._active_streams[request_id] = request
        return True

    @property
    def num_running_waiting_for_chunk(self) -> int:
        """Count running requests temporarily removed while awaiting a chunk."""
        return len(self.waiting_for_chunk_running_requests)

    def _preempt_non_active_running(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        # Hold non-active running requests in a private deque rather than
        # routing them back through waiting_queue. Routing through the
        # vllm RequestQueue mid-step triggers
        #   "Cannot register new removed request after self.removed has
        #    been read"
        # in vllm.v1.sample.logits_processor.state when the persistent
        # batch was already snapshotted. They are returned to
        # running_queue in restore_queues() so the next scheduler tick
        # re-evaluates them through _promote_active_streams.
        index = len(running_queue) - 1
        while index >= 0:
            request = running_queue[index]
            if request.request_id in self._active_streams:
                index -= 1
                continue
            request = running_queue.pop(index)
            self._held_non_active.append(request)
            index -= 1

    def _resume_loaded_request(self, request: Request, queue: Any, target_status: RequestStatus) -> None:
        """Restore a request whose async chunk load just completed.

        Normal case: flip back to the status of the queue it parked from.

        Segment-boundary case: if the loaded payload REPLACED the streaming
        prompt (new segment), a RUNNING resume is forbidden -- it would ship
        the payload to the runner on the cached path, which performs none of
        the per-segment state refresh (see _segment_replaced_reqs). Route the
        request through WAITING instead: pull it off the running queue and
        park it in waiting_for_chunk_waiting_requests, which restore_queues()
        re-admits into the waiting queue after this scheduler pass; the next
        pass then emits it as NewRequestData with the fresh payload attached.
        """
        req_id = request.request_id
        self.requests_with_ready_chunks.add(req_id)
        replaced = req_id in self._segment_replaced_reqs
        self._segment_replaced_reqs.discard(req_id)
        if replaced and target_status == RequestStatus.RUNNING:
            # Loud on purpose: this is the exact interleaving that used to
            # kill the engine (stale num_processed_tokens -> empty prefill
            # rows -> IndexError / device-side assert). A count here is the
            # proof the guard is earning its keep.
            logger.warning(
                "[OmniTransfer] req %s: new-segment payload landed while parked "
                "from RUNNING; rerouting through WAITING so the runner refreshes "
                "segment state (NewRequestData path)",
                req_id,
            )
            request.status = RequestStatus.WAITING
            try:
                queue.remove(request)
            except ValueError:
                pass
            self.requests_origin_status[req_id] = RequestStatus.WAITING
            self.waiting_for_chunk_waiting_requests.append(request)
            return
        request.status = target_status

    def _process_chunk_queue_legacy(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    finished_load_reqs.remove(request.request_id)
                    self._resume_loaded_request(request, queue, target_status)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _purge_untracked_chunk_requests(
        self,
        deque_list: deque[Any],
        scheduler_requests: dict[str, Request],
    ) -> None:
        """Drop deque entries whose ``request_id`` is not in
        ``scheduler_requests`` and reclaim their receiver-side state.

        Handles requests that were aborted mid-flight while parked in a
        chunk-transfer deque: ``Scheduler._free_request`` deleted the
        entry from ``scheduler.requests`` but the deque still holds a
        reference to the now-freed ``Request``. Order of survivors is
        preserved.
        """
        if not deque_list:
            return
        for _ in range(len(deque_list)):
            request = deque_list.popleft()
            if request.request_id in scheduler_requests:
                deque_list.append(request)
            else:
                self.cleanup_receiver(request.request_id)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        scheduler_requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.

        Re-runs the zombie purge first to close the race window where an
        abort fires *between* ``process_pending_chunks`` and the
        ``finally``-clause ``restore_queues`` call. Without the second
        purge, ``running_queue.extend(...)`` would still re-inject a
        freed ``Request`` and crash the worker on the next tick.

        ``scheduler_requests`` is optional for back-compat with legacy
        callers (older tests pass only the two queue arguments). When
        provided, it gates both the deque purge and the per-request
        admit checks below; when ``None``, the purge is skipped and
        every parked request is restored unconditionally (the
        pre-purge behavior).
        """
        if not self.receives_chunks:
            return
        if scheduler_requests is not None:
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_waiting_requests, scheduler_requests)
            self._purge_untracked_chunk_requests(self.waiting_for_chunk_running_requests, scheduler_requests)
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            if scheduler_requests is None or request.request_id in scheduler_requests:
                waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            live_running_requests = [
                request
                for request in self.waiting_for_chunk_running_requests
                if scheduler_requests is None or request.request_id in scheduler_requests
            ]
            running_queue.extend(live_running_requests)
        self.waiting_for_chunk_running_requests = deque()

        if self._held_non_active:
            running_queue.extend(self._held_non_active)
            self._held_non_active = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if not self.receives_chunks:
            return
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return

        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request and additional_info:
                request.additional_information = None

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if not self._ensure_active_stream(request):
                if target_status == RequestStatus.WAITING:
                    # A non-active placeholder must not remain visible to the
                    # scheduler: it has no connector payload yet, so running
                    # it would execute the downstream model with empty
                    # additional_information. Park it until restore_queues()
                    # and retry admission on the next scheduler tick.
                    queue.remove(request)
                    waiting_for_chunk_list.append(request)
                continue
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if self.is_done_receiving_chunks(request.request_id):
                    request.additional_information = None
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    finished_load_reqs.remove(request.request_id)
                    self._resume_loaded_request(request, queue, target_status)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)

        request_ids = set(request_ids)

        self.waiting_for_chunk_waiting_requests = deque(
            request for request in self.waiting_for_chunk_waiting_requests if request.request_id not in request_ids
        )
        self.waiting_for_chunk_running_requests = deque(
            request for request in self.waiting_for_chunk_running_requests if request.request_id not in request_ids
        )
        self._held_non_active = deque(
            request for request in self._held_non_active if request.request_id not in request_ids
        )

        for req_id in request_ids:
            self._active_streams.pop(req_id, None)
            self.requests_with_ready_chunks.discard(req_id)
            self.upstream_exhausted_requests.discard(req_id)
            self._finished_load_reqs.discard(req_id)
            self._cancelled_load_reqs.add(req_id)

        return []
