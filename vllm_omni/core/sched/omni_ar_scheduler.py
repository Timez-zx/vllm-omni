from __future__ import annotations

import os
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from time import monotonic, time
from typing import Any

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.core.sched.output import OmniCachedRequestData
from vllm_omni.core.sched.utils import omni_routed_experts_for_request
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.experimental.fullduplex.engine.intermediate import (
    NATIVE_LAST_PROMPT_TOKEN_KEY,
    NATIVE_PROMPT_LEN_KEY,
    NATIVE_PROMPT_TOKEN_IDS_KEY,
    NATIVE_SEGMENT_TOKEN_IDS_KEY,
)
from vllm_omni.outputs import OmniConnectorOutput

logger = init_logger(__name__)
_MAX_KV_LINEAGE_SNAPSHOTS = 4096
_LOG_SCHED_DIAG = os.environ.get("VLLM_OMNI_LOG_SCHED_DIAG", "0") not in ("0", "", "false", "False")
_LOG_HANDOFF_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "0") not in ("0", "", "false", "False")
_LOG_CORE_STEP_DIAG = os.environ.get("VLLM_OMNI_LOG_CORE_STEP_DIAG", "0") not in (
    "0",
    "",
    "false",
    "False",
)
_DIAG_STAGE_RAW = os.environ.get("VLLM_OMNI_DIAG_STAGE")
_DIAG_STAGES = (
    None
    if _DIAG_STAGE_RAW is None
    else frozenset(stage.strip() for stage in _DIAG_STAGE_RAW.split(",") if stage.strip())
)

_KV_TRANSFER_EVIDENCE_FIELDS = (
    "kv_transfer_selected_blocks",
    "kv_transfer_selected_tokens",
    "kv_transfer_selected_bytes",
    "kv_transfer_write_submit_to_d_ready_ms",
)


def _prefill_microbatch_window_s(stage_id: object) -> float:
    """Experimental stage-0 scheduler admission window, disabled by default."""
    if str(stage_id) != "0":
        return 0.0
    raw = os.environ.get("VLLM_OMNI_PREFILL_MICROBATCH_WINDOW_MS", "0")
    try:
        window_ms = float(raw)
    except ValueError:
        logger.warning(
            "Invalid VLLM_OMNI_PREFILL_MICROBATCH_WINDOW_MS=%r; disabling scheduler microbatching",
            raw,
        )
        return 0.0
    if window_ms < 0:
        logger.warning(
            "Negative VLLM_OMNI_PREFILL_MICROBATCH_WINDOW_MS=%r; disabling scheduler microbatching",
            raw,
        )
        return 0.0
    return min(window_ms, 1000.0) / 1000.0


def _diagnostic_tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return 0
    if isinstance(value, dict):
        return sum(_diagnostic_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_diagnostic_tensor_bytes(item) for item in value)
    return 0


def _compact_native_duplex_prompt_metadata(
    multimodal_output: Any,
    *,
    current_segment_token_ids: Iterable[int] | None = None,
    native_segment: bool = False,
) -> Any:
    """Replace MiniCPM's O(context) prompt snapshot with boundary metadata.

    The scheduler still owns the complete request prompt and KV identity.  The
    model-provided copy exists only to help the downstream Talker locate the
    current segment, so prompt length plus the final prompt token is sufficient.
    Unknown/legacy payload shapes are returned unchanged.
    """
    if not isinstance(multimodal_output, dict):
        return multimodal_output
    raw_prompt_ids = multimodal_output.get(NATIVE_PROMPT_TOKEN_IDS_KEY)
    if raw_prompt_ids is None:
        # Decode steps do not rerun media preprocessing, so they need not
        # repeat the prompt snapshot emitted by the unit's prefill step.
        # Nevertheless the terminal segment IDs MUST replace the previous
        # unit's IDs; otherwise an earlier listen decision masks new speech.
        if native_segment and current_segment_token_ids is not None:
            return {
                **multimodal_output,
                NATIVE_SEGMENT_TOKEN_IDS_KEY: torch.tensor(list(current_segment_token_ids), dtype=torch.int64),
            }
        return multimodal_output
    tensor_payload = isinstance(raw_prompt_ids, torch.Tensor)
    if hasattr(raw_prompt_ids, "detach"):
        raw_prompt_ids = raw_prompt_ids.detach().cpu().tolist()
    if isinstance(raw_prompt_ids, tuple):
        raw_prompt_ids = list(raw_prompt_ids)
    if (
        isinstance(raw_prompt_ids, list)
        and len(raw_prompt_ids) == 1
        and isinstance(raw_prompt_ids[0], (list, tuple))
    ):
        raw_prompt_ids = list(raw_prompt_ids[0])
    if not isinstance(raw_prompt_ids, list):
        return multimodal_output
    try:
        prompt_ids = [int(token_id) for token_id in raw_prompt_ids]
    except (TypeError, ValueError):
        return multimodal_output

    compact = dict(multimodal_output)
    compact.pop(NATIVE_PROMPT_TOKEN_IDS_KEY, None)
    compact[NATIVE_PROMPT_LEN_KEY] = len(prompt_ids)
    compact[NATIVE_LAST_PROMPT_TOKEN_KEY] = prompt_ids[-1] if prompt_ids else None
    if current_segment_token_ids is not None:
        compact[NATIVE_SEGMENT_TOKEN_IDS_KEY] = [int(token_id) for token_id in current_segment_token_ids]
    if tensor_payload:
        # EngineCore's multimodal wire channel is dict[str, Tensor]. Preserve
        # that contract when compacting a model output; Python ints/lists are
        # valid inside local bridge payloads but fail typed msgpack decoding.
        for key in (NATIVE_PROMPT_LEN_KEY, NATIVE_LAST_PROMPT_TOKEN_KEY, NATIVE_SEGMENT_TOKEN_IDS_KEY):
            if key not in compact:
                continue
            if compact[key] is None:
                compact.pop(key)
            else:
                compact[key] = torch.tensor(compact[key], dtype=torch.int64)
    return compact


LOG_DUPLEX_CADENCE = os.environ.get("VLLM_OMNI_LOG_DUPLEX_CADENCE", "0") in {"1", "units"}
LOG_DUPLEX_STEPS = os.environ.get("VLLM_OMNI_LOG_DUPLEX_CADENCE", "0") == "1"


class SampledLogprobContractError(RuntimeError):
    """The model runner returned unusable sampled-token logprobs."""


def _slice_sampled_logprobs(logprobs: Any, req_index: int, sampled_token_ids: list[int]) -> Any:
    """Slice and validate the sampled-token logprobs for one AR request."""
    if logprobs is None:
        raise SampledLogprobContractError("AR logprobs were requested, but the model runner returned none")

    sliced = logprobs.slice_request(req_index, len(sampled_token_ids))
    token_rows = np.asarray(sliced.logprob_token_ids)
    value_rows = np.asarray(sliced.logprobs)
    expected_rows = len(sampled_token_ids)

    if token_rows.ndim != 2 or value_rows.ndim != 2:
        raise SampledLogprobContractError(
            "AR sampled-token logprobs must be rank-2 arrays, "
            f"got token_ids={token_rows.shape} logprobs={value_rows.shape}"
        )
    if token_rows.shape[0] != expected_rows or value_rows.shape[0] != expected_rows:
        raise SampledLogprobContractError(
            "AR sampled-token logprob row count does not match generated tokens: "
            f"tokens={expected_rows} token_id_rows={token_rows.shape[0]} "
            f"logprob_rows={value_rows.shape[0]}"
        )
    if expected_rows == 0:
        return sliced
    if token_rows.shape[1] == 0 or value_rows.shape[1] == 0:
        raise SampledLogprobContractError("AR sampled-token logprob rows are empty")

    sampled = np.asarray(sampled_token_ids)
    if not np.array_equal(token_rows[:, 0], sampled):
        mismatch = np.flatnonzero(token_rows[:, 0] != sampled)
        first = int(mismatch[0])
        raise SampledLogprobContractError(
            "AR sampled-token logprobs are misaligned: "
            f"row={first} generated_token={int(sampled[first])} "
            f"logprob_token={int(token_rows[first, 0])}"
        )
    if not np.isfinite(value_rows[:, 0]).all():
        bad_rows = np.flatnonzero(~np.isfinite(value_rows[:, 0])).tolist()
        raise SampledLogprobContractError(f"AR sampled-token logprobs contain non-finite values at rows {bad_rows}")
    return sliced


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Diagnostic-only scheduler admission timestamps. Request.arrival_time
        # starts before the request reaches EngineCore, so it cannot isolate
        # time spent in the scheduler's own waiting queue.
        self._diag_scheduler_admit_mono: dict[str, float] = {}

        # Controlled A/B hook for short finite prefills. While the oldest
        # stage-0 request is younger than this bound, schedule() continues to
        # process running work but temporarily hides the waiting queue. The
        # EngineCore loop remains free to drain newly arrived requests, so the
        # next admission can contain a real multi-request batch.
        stage_id = getattr(self.vllm_config.model_config, "stage_id", None)
        self._prefill_microbatch_window_s = _prefill_microbatch_window_s(stage_id)
        self._prefill_scheduler_admit_mono: dict[str, float] = {}

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        self.finished_req_ids_dict = defaultdict(set)

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        self.input_coordinator: OmniSchedulingCoordinator | None = None
        if uses_full_payload_input_coordinator(model_config):
            self.input_coordinator = OmniSchedulingCoordinator(
                stage_id=getattr(model_config, "stage_id", 0),
            )
        self._latest_omni_connector_output: OmniConnectorOutput | None = None
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}
        # Hash-only metadata for completed finite requests. This registry owns
        # no KV blocks; normal prefix-cache eviction remains authoritative.
        self.kv_cache_manager._omni_kv_lineage_lock = threading.Lock()
        self.kv_cache_manager._omni_kv_lineage_snapshots = OrderedDict()
        # Observation-only generation counters for duplex latency traces.
        self._duplex_admit_generation: dict[str, int] = defaultdict(int)
        self._duplex_schedule_step: dict[tuple[str, int], int] = defaultdict(int)
        self._duplex_inflight_steps: dict[str, deque[tuple[int, int]]] = defaultdict(deque)

    def _update_from_kv_xfer_finished(
        self,
        kv_connector_output: KVConnectorOutput,
    ) -> None:
        """Apply connector completions without freeing live duplex-P KV.

        Upstream treats every ``finished_sending`` notification as the end of
        a finite P request. Native duplex publishes the current prefix at a
        resumable segment boundary, so the notification only releases the
        connector lease; the live request still owns and extends those blocks.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            request = self.requests.get(req_id)
            if request is None:
                # A session may be aborted while an already-submitted NIXL
                # operation is completing. Its scheduler blocks were released
                # by abort; this late connector acknowledgement owns no live
                # request state and must not terminate the EngineCore.
                logger.debug(
                    "Ignoring late KV receive completion for removed request %s",
                    req_id,
                )
                continue
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            elif RequestStatus.is_finished(request.status):
                self._free_blocks(request)

        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            request = self.requests.get(req_id)
            if request is None:
                logger.debug(
                    "Ignoring late KV send completion for removed request %s",
                    req_id,
                )
                continue
            if RequestStatus.is_finished(request.status):
                self._free_blocks(request)
            else:
                logger.debug(
                    "Retaining live resumable request blocks after P/D segment send: %s",
                    req_id,
                )

    def _kv_lineage_registry(
        self,
    ) -> tuple[
        threading.Lock,
        OrderedDict[tuple[str, int], tuple[tuple[object, ...], int, int]],
    ]:
        manager = self.kv_cache_manager
        # Some unit tests construct the scheduler with __new__.
        if not hasattr(manager, "_omni_kv_lineage_lock"):
            manager._omni_kv_lineage_lock = threading.Lock()
            manager._omni_kv_lineage_snapshots = OrderedDict()
        return manager._omni_kv_lineage_lock, manager._omni_kv_lineage_snapshots

    def prepare_kv_lineage_request(self, request: Any) -> None:
        """Resolve an opaque parent handle before Request hashing begins."""
        lineage_id = getattr(request, "kv_lineage_id", None)
        parent_revision = int(getattr(request, "kv_lineage_parent_revision", 0))
        if not lineage_id or parent_revision <= 0:
            return
        lineage_lock, lineage_snapshots = self._kv_lineage_registry()
        key = (lineage_id, parent_revision)
        with lineage_lock:
            snapshot = lineage_snapshots.get(key)
            if snapshot is not None:
                lineage_snapshots.move_to_end(key)
        if snapshot is None:
            return
        hashes, num_computed_tokens, hash_block_size = snapshot
        request.kv_lineage_snapshot_block_hashes = list(hashes)
        request.kv_lineage_snapshot_num_computed_tokens = num_computed_tokens
        request.kv_lineage_snapshot_hash_block_size = hash_block_size

    def _store_kv_lineage_snapshot(
        self,
        lineage_id: str,
        revision: int,
        block_hashes: list[object],
        num_computed_tokens: int,
        hash_block_size: int,
    ) -> None:
        if revision <= 0 or num_computed_tokens <= 0 or hash_block_size <= 0:
            return
        full_blocks = min(len(block_hashes), num_computed_tokens // hash_block_size)
        snapshot = (tuple(block_hashes[:full_blocks]), num_computed_tokens, hash_block_size)
        key = (lineage_id, revision)
        lineage_lock, lineage_snapshots = self._kv_lineage_registry()
        with lineage_lock:
            lineage_snapshots[key] = snapshot
            lineage_snapshots.move_to_end(key)
            while len(lineage_snapshots) > _MAX_KV_LINEAGE_SNAPSHOTS:
                lineage_snapshots.popitem(last=False)

    def prepare_direct_pd_cache_sync(self, request: Request) -> tuple[str, Any | None]:
        """Allocate a D prefix-cache import without admitting an inference request.

        Returns ``("blocked", None)`` when cache capacity is temporarily
        unavailable, ``("full_hit", metadata)`` when D already owns the full
        prefix, or ``("loading", metadata)`` after staging a NIXL delta import.
        The caller passes the metadata directly to the worker control path.
        """
        connector = self.connector
        if connector is None:
            raise RuntimeError("Direct P/D cache sync requires a KV connector")
        if request.request_id in self.requests:
            raise RuntimeError(f"Direct P/D cache sync collides with inference request {request.request_id}")

        computed_blocks, local_tokens, _ = self.kv_cache_manager.get_computed_blocks(request)
        external_tokens, load_async = connector.get_num_new_matched_tokens(request, local_tokens)
        if external_tokens is None:
            return "blocked", None

        remote_tokens = local_tokens + external_tokens
        if external_tokens == 0:
            # NixlDeltaPushConnector emits an empty registration here so P can
            # release its request-scoped lease even though D needs no bytes.
            connector.update_state_after_alloc(request, computed_blocks, 0)
            self._snapshot_pd_transfer_evidence(request)
            metadata = connector.build_connector_meta(SchedulerOutput.make_empty())
            hash_block_size = int(getattr(self.kv_cache_manager.block_pool, "hash_block_size", 0))
            lineage_id = getattr(request, "kv_lineage_id", None)
            revision = int(getattr(request, "kv_lineage_revision", 0))
            if lineage_id:
                self._store_kv_lineage_snapshot(
                    lineage_id,
                    revision,
                    request.block_hashes,
                    remote_tokens,
                    hash_block_size,
                )
            # Stop the D heartbeat; no allocated request block table exists on
            # this path, so the ordinary scheduler finalizer cannot be used.
            request.status = RequestStatus.FINISHED_STOPPED
            connector.request_finished(request, [])
            return "full_hit", metadata

        if not load_async:
            raise RuntimeError("Direct P/D cache sync expected an asynchronous external KV load")

        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            0,
            num_new_computed_tokens=local_tokens,
            new_computed_blocks=computed_blocks,
            num_lookahead_tokens=0,
            num_external_computed_tokens=external_tokens,
            delay_cache_blocks=True,
            full_sequence_must_fit=True,
            has_scheduled_reqs=bool(self.running),
        )
        if new_blocks is None:
            return "blocked", None

        connector.update_state_after_alloc(
            request,
            self.kv_cache_manager.get_blocks(request.request_id),
            external_tokens,
        )
        request.num_computed_tokens = remote_tokens
        metadata = connector.build_connector_meta(SchedulerOutput.make_empty())
        return "loading", metadata

    @staticmethod
    def _snapshot_pd_transfer_evidence(request: Request) -> None:
        """Retain connector-proven scalars past cache-sync finalization."""
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict) or not any(
            name in params for name in _KV_TRANSFER_EVIDENCE_FIELDS
        ):
            return
        request.pd_transfer_evidence = {
            name: params.get(name, -1)
            for name in _KV_TRANSFER_EVIDENCE_FIELDS
        }

    def complete_direct_pd_cache_sync(self, request: Request) -> None:
        """Commit a completed cache-only import to D's prefix cache."""
        connector = self.connector
        if connector is None:
            raise RuntimeError("Direct P/D cache sync requires a KV connector")
        connector.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
        self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

        # This is a cache-only import, not an inference request. Preserve ALL
        # imported KV, including the partial block. Only formal D admission
        # can decide whether its own prompt needs a final-token replay. Native
        # MiniCPM appends P's sampled token: backing up here would recompute an
        # audio embedding as a plain placeholder token and corrupt generation.

        lineage_id = getattr(request, "kv_lineage_id", None)
        revision = int(getattr(request, "kv_lineage_revision", 0))
        hash_block_size = int(getattr(self.kv_cache_manager.block_pool, "hash_block_size", 0))
        if lineage_id:
            self._store_kv_lineage_snapshot(
                lineage_id,
                revision,
                request.block_hashes,
                request.num_computed_tokens,
                hash_block_size,
            )

        request.status = RequestStatus.FINISHED_STOPPED
        # request_finished() owns connector cleanup and may clear or replace
        # its control dictionary.  The paired formal D request still needs
        # the exact allocation/WRITE evidence for its completion witness.
        self._snapshot_pd_transfer_evidence(request)
        self._connector_finished(request)
        if not bool(getattr(request, "pd_cache_sync_retain", False)):
            self.kv_cache_manager.free(request)

    def release_direct_pd_cache_sync(self, request: Request) -> None:
        """Release a retained cache-only request while preserving cached KV."""
        self.kv_cache_manager.free(request)

    def fail_direct_pd_cache_sync(self, request: Request) -> None:
        """Release scheduler-side resources after a cache-only import error."""
        request.status = RequestStatus.FINISHED_ABORTED
        try:
            self._connector_finished(request)
        finally:
            self.kv_cache_manager.free(request)

    def _init_duplex_cadence_counters(self) -> None:
        """Lazily initialize counters for tests that construct schedulers with ``__new__``."""
        if not hasattr(self, "_duplex_admit_generation"):
            self._duplex_admit_generation = defaultdict(int)
            self._duplex_schedule_step = defaultdict(int)
            self._duplex_inflight_steps = defaultdict(deque)

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when orchestrator will not run stage 1+ for this request (e.g. text-only).

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _should_defer_waiting_admission(self) -> bool:
        window_s = getattr(self, "_prefill_microbatch_window_s", 0.0)
        if window_s <= 0 or not self.waiting:
            return False
        admitted = getattr(self, "_prefill_scheduler_admit_mono", {})
        oldest = min(
            (admitted.get(request.request_id, 0.0) for request in self.waiting),
            default=0.0,
        )
        return oldest > 0 and monotonic() - oldest < window_s

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self.transfer_triggered_requests.add(request.request_id)

                self._mark_request_for_kv_transfer(request.request_id, confirmed_computed)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    # Defer the stop until KV extraction completes so that
                    # the kv_ready signal can be emitted while the request
                    # is still alive.  The request will be stopped on the
                    # next scheduler step after extraction ack arrives.
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                self.transfer_triggered_requests.add(request.request_id)

                try:
                    idx = new_token_ids.index(target_token_id)
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = confirmed_computed - tokens_to_exclude
                except ValueError:
                    snapshot_len = confirmed_computed

                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        return False

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        # Upstream does not emit a per-request warning for this path. Keep
        # capacity audits from silently treating KV recomputation as ordinary
        # input work, especially for native multimodal streaming requests.
        logger.warning(
            "[kv-preemption] stage=%s request=%s preempted computed=%d "
            "prompt=%d tokens=%d free_blocks=%d",
            self.vllm_config.model_config.stage_id,
            request.request_id,
            request.num_computed_tokens,
            request.num_prompt_tokens,
            request.num_tokens,
            self.kv_cache_manager.block_pool.get_num_free_blocks(),
        )
        super()._preempt_request(request, timestamp)

    def add_request(self, request: Request) -> None:
        if getattr(self, "_prefill_microbatch_window_s", 0.0) > 0:
            self._prefill_scheduler_admit_mono[request.request_id] = monotonic()
        if _LOG_SCHED_DIAG:
            self._diag_scheduler_admit_mono[request.request_id] = monotonic()
        if LOG_DUPLEX_CADENCE:
            self._init_duplex_cadence_counters()
            self._duplex_admit_generation[request.request_id] += 1
            # Preserve native non-P/D input identity as well as finite-P/D
            # request identity. Protocol audio deltas are not one-to-one with
            # real input units and cannot measure sustained input progress.
            input_seq: int | str = "-"
            input_origin = "-"
            input_unit_index: int | str = "-"
            model_buffer = getattr(request, "model_intermediate_buffer", None)
            duplex = model_buffer.get("duplex") if isinstance(model_buffer, dict) else None
            if isinstance(duplex, dict):
                input_seq = duplex.get("seq", "-")
                payload = duplex.get("payload")
                if isinstance(payload, dict):
                    input_origin = (
                        "continuation"
                        if payload.get("duplex_input_source") == "auto_continuation"
                        else "client"
                    )
                    input_unit_index = payload.get("input_unit_index", "-")
            logger.info(
                "[duplex_cadence] stage=%s ADMIT req=%s generation=%s "
                "admit_epoch=%.6f prompt_tokens=%s input_seq=%s input_origin=%s input_unit_index=%s",
                self.vllm_config.model_config.stage_id,
                request.request_id,
                self._duplex_admit_generation[request.request_id],
                time(),
                getattr(request, "num_prompt_tokens", "?"),
                input_seq,
                input_origin,
                input_unit_index,
            )
        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        core_step_start = monotonic() if _LOG_CORE_STEP_DIAG else 0.0
        # Remove FINISHED_ABORTED requests before the upstream scheduler sees
        # them. Upstream vllm raises RuntimeError on this status; omni allows
        # async abort (e.g. client disconnect during TTS streaming) to leave
        # requests in the waiting/running queues temporarily.
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if getattr(req, "status", None) == RequestStatus.FINISHED_ABORTED:
                    queue.remove(req)
        self._consume_pending_connector_output(model_mode="ar")
        self._process_pending_input_timeouts()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting, self.running, scheduler_requests=self.requests
            )

        original_waiting = None
        if self._should_defer_waiting_admission():
            original_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)

        before_base_schedule = monotonic() if _LOG_CORE_STEP_DIAG else 0.0
        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            if original_waiting is not None:
                deferred_waiting = list(self.waiting)
                if deferred_waiting:
                    original_waiting.prepend_requests(deferred_waiting)
                self.waiting = original_waiting
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(
                    self.waiting,
                    self.running,
                    scheduler_requests=self.requests,
                )
            if self.input_coordinator:
                self.input_coordinator.restore_queues(self.waiting)

        if _LOG_HANDOFF_DIAG and not scheduler_output.num_scheduled_tokens:
            stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
            if _DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES:
                now = monotonic()
                last = getattr(self, "_pd_blocked_diag_last", 0.0)
                blocked = [*self.waiting, *self.skipped_waiting]
                if blocked and now - last >= 0.5:
                    self._pd_blocked_diag_last = now
                    request = blocked[0]
                    params = getattr(request, "kv_transfer_params", None)
                    logger.info(
                        "[SCHED-BLOCKED-DIAG] stage=%s req=%s status=%s "
                        "computed=%d prompt=%d tokens=%d running=%d waiting=%d "
                        "skipped=%d remote_prefill=%s remote_prompt=%s",
                        stage_id,
                        request.request_id,
                        request.status,
                        int(request.num_computed_tokens),
                        int(request.num_prompt_tokens),
                        int(request.num_tokens),
                        len(self.running),
                        len(self.waiting),
                        len(self.skipped_waiting),
                        params.get("do_remote_prefill") if params else None,
                        params.get("remote_prompt_tokens") if params else None,
                    )
        after_base_schedule = monotonic() if _LOG_CORE_STEP_DIAG else 0.0

        # A normal vLLM recompute preemption discards the request's KV and
        # resets its computed-token cursor.  MiniCPM P deliberately owns only
        # the newest suffix on the steady path, so switch the preempted live
        # request to its exact compact recovery lineage before it is resumed.
        for req_id in scheduler_output.preempted_req_ids:
            request = self.requests.get(req_id)
            if request is not None:
                self._rebase_preempted_minicpmo_duplex_request(request)

        if getattr(self, "_prefill_microbatch_window_s", 0.0) > 0:
            for scheduled in scheduler_output.scheduled_new_reqs:
                self._prefill_scheduler_admit_mono.pop(scheduled.req_id, None)

        if _LOG_SCHED_DIAG:
            stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
            diag_stage_matches = _DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES
        else:
            diag_stage_matches = False
        if diag_stage_matches:
            now = monotonic()
            for scheduled in scheduler_output.scheduled_new_reqs:
                req_id = scheduled.req_id
                request = self.requests.get(req_id)
                queued_ts = None
                scheduled_ts = None
                if request is not None:
                    for event in request.events:
                        if event.type == EngineCoreEventType.QUEUED and queued_ts is None:
                            queued_ts = event.timestamp
                        elif event.type == EngineCoreEventType.SCHEDULED and scheduled_ts is None:
                            scheduled_ts = event.timestamp
                queue_ms = (
                    (scheduled_ts - queued_ts) * 1000.0
                    if queued_ts is not None and scheduled_ts is not None
                    else (time() - request.arrival_time) * 1000.0
                    if request is not None
                    else -1.0
                )
                scheduler_admit_mono = self._diag_scheduler_admit_mono.pop(req_id, None)
                scheduler_queue_ms = (now - scheduler_admit_mono) * 1000.0 if scheduler_admit_mono is not None else -1.0
                logger.info(
                    "[SCHED-DIAG] stage=%s mono=%.6f req=%s queue_ms=%.3f "
                    "scheduler_queue_ms=%.3f "
                    "prompt=%d cached=%d scheduled=%d waiting=%d running=%d",
                    stage_id,
                    now,
                    req_id,
                    queue_ms,
                    scheduler_queue_ms,
                    len(scheduled.prompt_token_ids),
                    int(scheduled.num_computed_tokens),
                    int(scheduler_output.num_scheduled_tokens.get(req_id, 0)),
                    len(self.waiting),
                    len(self.running),
                )
        if LOG_DUPLEX_STEPS and scheduler_output.num_scheduled_tokens:
            self._init_duplex_cadence_counters()
            scheduled_epoch = time()
            batch_reqs = len(scheduler_output.num_scheduled_tokens)
            batch_tokens = scheduler_output.total_num_scheduled_tokens
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                generation = self._duplex_admit_generation.get(req_id, 0)
                key = (req_id, generation)
                self._duplex_schedule_step[key] += 1
                step = self._duplex_schedule_step[key]
                self._duplex_inflight_steps[req_id].append((generation, step))
                logger.info(
                    "[duplex_cadence] stage=%s SCHEDULE req=%s generation=%s step=%s "
                    "schedule_epoch=%.6f scheduled_tokens=%s batch_reqs=%s batch_tokens=%s",
                    self.vllm_config.model_config.stage_id,
                    req_id,
                    generation,
                    step,
                    scheduled_epoch,
                    scheduled_tokens,
                    batch_reqs,
                    batch_tokens,
                )
        cached = scheduler_output.scheduled_cached_reqs
        rebased_resumed_req_ids = {
            req_id
            for req_id in cached.resumed_req_ids
            if req_id in self.requests
            and bool(
                getattr(
                    self.requests[req_id],
                    "_minicpmo_duplex_preemption_rebased",
                    False,
                )
            )
        }
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    prompt_is_token_ids=nr.prompt_is_token_ids,
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                    model_intermediate_buffer=(
                        getattr(request, "model_intermediate_buffer", None) if request else None
                    ),
                    pd_prefill_payload=(getattr(request, "pd_prefill_payload", None) if request else None),
                )
                new_list.append(omni_nr)

            # Base CachedRequestData does not carry a changed prompt on
            # preemption resume; the worker would otherwise retain the old
            # cumulative prompt length while the scheduler owns the compact
            # rebase.  OmniCachedRequestData synchronizes that identity for all
            # resumed requests (ordinary cached-running requests stay empty).
            resumed_prompt_token_ids = {
                req_id: list(self.requests[req_id].prompt_token_ids or ())
                for req_id in rebased_resumed_req_ids
            }
            omni_cached = OmniCachedRequestData(
                req_ids=cached.req_ids,
                resumed_req_ids=cached.resumed_req_ids,
                new_token_ids=cached.new_token_ids,
                all_token_ids=cached.all_token_ids,
                new_block_ids=cached.new_block_ids,
                num_computed_tokens=cached.num_computed_tokens,
                num_output_tokens=cached.num_output_tokens,
                prompt_token_ids=resumed_prompt_token_ids,
                additional_information={},
            )
            # Construct both wrappers before publishing either one.  This
            # avoids a half-wrapped SchedulerOutput if dataclass construction
            # or a future serialization field check raises.
            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            scheduler_output.scheduled_cached_reqs = omni_cached
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception as exc:
            if rebased_resumed_req_ids:
                # The scheduler request now owns a compact recompute lineage.
                # Continuing without delivering its replacement prompt to the
                # worker would silently split their token/KV identities.
                raise RuntimeError(
                    "Failed to publish a compact MiniCPM-o preemption rebase "
                    "to the worker for requests "
                    f"{sorted(rebased_resumed_req_ids)}"
                ) from exc
            logger.exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        # Wrap in omni scheduler output to carry transfer metadata.
        result = self._wrap_omni_scheduler_output(
            scheduler_output,
            finished_requests_needing_kv_transfer=finished_reqs,
        )
        if _LOG_CORE_STEP_DIAG and result.num_scheduled_tokens:
            done = monotonic()
            stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
            if _DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES:
                logger.info(
                    "[CORE-STEP-DIAG] event=scheduler-return stage=%s mono=%.6f "
                    "reqs=%s pre_base_ms=%.3f base_ms=%.3f post_base_ms=%.3f total_ms=%.3f",
                    stage_id,
                    done,
                    ",".join(result.num_scheduled_tokens),
                    (before_base_schedule - core_step_start) * 1000.0,
                    (after_base_schedule - before_base_schedule) * 1000.0,
                    (done - after_base_schedule) * 1000.0,
                    (done - core_step_start) * 1000.0,
                )
        return result

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        handoff_diag_start = monotonic()
        if LOG_DUPLEX_STEPS and scheduler_output.num_scheduled_tokens:
            self._init_duplex_cadence_counters()
            runner_done_epoch = time()
            for req_id in scheduler_output.num_scheduled_tokens:
                inflight = self._duplex_inflight_steps.get(req_id)
                if inflight:
                    generation, step = inflight.popleft()
                else:
                    generation = self._duplex_admit_generation.get(req_id, 0)
                    step = self._duplex_schedule_step.get((req_id, generation), 0)
                logger.info(
                    "[duplex_cadence] stage=%s RUNNER_DONE req=%s generation=%s "
                    "step=%s runner_done_epoch=%.6f",
                    self.vllm_config.model_config.stage_id,
                    req_id,
                    generation,
                    step,
                    runner_done_epoch,
                )
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        inter_stage_outputs = getattr(model_runner_output, "inter_stage_outputs", None)
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        # Keep the vLLM async-scheduler block-lifetime fence in sync.  This
        # method replaces Scheduler.update_from_output(), so new upstream
        # lifecycle steps must be mirrored here: without advancing the
        # processed sequence, normal request completion only moves blocks into
        # ``deferred_frees`` and they are never returned to the pool.
        total_scheduled_tokens = getattr(
            scheduler_output,
            "total_num_scheduled_tokens",
            sum(num_scheduled_tokens.values()),
        )
        if getattr(self, "defer_block_free", False) and total_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Pre-process KV extraction acks so that the per-request loop below
        # can see up-to-date active_kv_transfers state and emit kv_ready
        # signals while requests are still alive (before any deferred stop).
        kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    self.active_kv_transfers.discard(req_id)
                    req = self.requests.get(req_id)
                    if req is not None and not req.is_finished():
                        outputs[req.client_index].append(
                            OmniEngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=[],
                                kv_transfer_params={"kv_ready": True},
                            )
                        )
                except Exception:
                    init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is not None:
                # vLLM 0.26: settle the in-flight tokens counted in schedule().
                # Must happen before the skips below — failed-KV-load and
                # already-finished requests were incremented too, and the two
                # readers (allocate_slots, _connector_finished) clamp with
                # max(0, computed - in_flight), so a leaked counter silently
                # freezes sliding-window block freeing.
                request.num_in_flight_tokens -= num_tokens_scheduled
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
            status_before_stop = request.status
            new_logprobs = None
            logprob_validation_failed = False

            # Validate before mutating request token state. A bad runner output
            # is request-local: terminate only this request and keep processing
            # the rest of the batch.
            if (
                generated_token_ids
                and request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
            ):
                try:
                    new_logprobs = _slice_sampled_logprobs(logprobs, req_index, generated_token_ids)
                except SampledLogprobContractError as exc:
                    logger.error("Invalid AR sampled-token logprobs for request %s: %s", req_id, exc)
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = str(exc)
                    request.resumable = False
                    generated_token_ids = []
                    logprob_validation_failed = True

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = logprob_validation_failed
            is_segment_finished = False
            finished = False
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            mm_output = mm_outputs[req_index] if mm_outputs else None
            inter_stage_output = inter_stage_outputs[req_index] if inter_stage_outputs else None
            kv_transfer_params = None
            finish_reason = None
            routed_experts = None

            # Check for stop and update request status.
            if bool(getattr(request, "prefill_only", False)) and (
                self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens
            ):
                # The prompt KV is now materialized. Do not commit the sampled
                # next token: this request exists only to populate reusable
                # prefix blocks and must not create model-visible output.
                request.status = RequestStatus.FINISHED_STOPPED
                new_token_ids = []
                stopped = True
            elif new_token_ids:
                num_sampled_tokens = len(new_token_ids)
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
                if new_logprobs is not None and len(new_token_ids) < num_sampled_tokens:
                    # A mid-step stop (e.g. spec-decode tokens sampled past
                    # EOS) trims new_token_ids after the validation slice
                    # above; re-slice so the emitted logprob rows stay 1:1
                    # with the emitted tokens, as upstream vLLM does by
                    # slicing after the trim.
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # If criteria returns True, it means we must STOP the request.
            # If criteria returns False, it might have triggered a background
            # transfer (e.g. prefill finished / special token) but continues decoding.
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            # Capture the complete finite/resumable segment before the stop
            # handler clears a live session's output-token list.
            current_segment_token_ids = (
                list(getattr(request, "output_token_ids", ()))
                if stopped
                else None
            )

            if stopped:
                if model_runner_output.routed_experts is not None:
                    routed_experts = omni_routed_experts_for_request(model_runner_output.routed_experts, request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                if LOG_DUPLEX_CADENCE:
                    logger.info(
                        "[duplex_cadence] stage=%s UNIT_DONE req=%s generation=%s done_epoch=%.6f "
                        "context_tokens=%s output_tokens=%s max_tokens=%s last_token=%s "
                        "stop_reason=%s finish_reason=%s",
                        self.vllm_config.model_config.stage_id,
                        req_id,
                        self._duplex_admit_generation.get(req_id, 0),
                        time(),
                        request.num_prompt_tokens,
                        request.num_output_tokens,
                        request.max_tokens,
                        new_token_ids[-1] if new_token_ids else None,
                        request.stop_reason,
                        finish_reason,
                    )
                # Native duplex P keeps one resumable request so the model's
                # streaming encoder/session state survives across input units.
                # A P/D connector normally publishes KV only from the terminal
                # request_finished path, which a resumable segment never takes.
                # Publish the completed prefix at this segment boundary while
                # retaining ownership of the live request's blocks.  Per-session
                # recurrence prevents another segment from overwriting this
                # lease before its paired D segment completes.
                segment_kv_params = getattr(request, "kv_transfer_params", None)
                publish_pd_segment = bool(
                    getattr(request, "resumable", False)
                    and isinstance(segment_kv_params, dict)
                    and segment_kv_params.get("do_remote_decode")
                    and self.connector is not None
                )
                if publish_pd_segment:
                    _, kv_transfer_params = self._connector_finished(request)
                    if kv_transfer_params is None:
                        raise RuntimeError(
                            "P/D streaming segment ended without connector metadata: "
                            f"request={request.request_id}"
                        )
                    # The paired D completion gates the next P segment, so
                    # the API already owns the previous published prefix.
                    # Echo only its extension, not the full logical history.
                    prompt_ids = request.prompt_token_ids or ()
                    end = min(len(prompt_ids), request.num_computed_tokens)
                    start = getattr(request, "_omni_pd_published_prompt_tokens", 0)
                    if not 0 <= start <= end:
                        start = 0
                    kv_transfer_params["remote_prompt_token_offset"] = start
                    kv_transfer_params["remote_prompt_token_ids"] = list(prompt_ids[start:end])
                    request._omni_pd_published_prompt_tokens = end
                else:
                    # Native duplex D normally uses one finite request per
                    # slot, while compatibility paths may retain a streaming
                    # request.  In both cases the physical transfer belonged
                    # to the preceding cache-sync request, so surface only its
                    # immutable evidence here without disturbing connector
                    # state or re-running request_finished().
                    request_kv_params = getattr(
                        request,
                        "pd_transfer_evidence",
                        None,
                    )
                    if not isinstance(request_kv_params, dict):
                        request_kv_params = getattr(
                            request,
                            "kv_transfer_params",
                            None,
                        )
                    if isinstance(request_kv_params, dict) and any(
                        name in request_kv_params
                        for name in _KV_TRANSFER_EVIDENCE_FIELDS
                    ):
                        kv_transfer_params = {
                            name: request_kv_params.get(name, -1)
                            for name in _KV_TRANSFER_EVIDENCE_FIELDS
                        }
                finished = self._handle_stopped_request(request)
                is_segment_finished = not finished
                if finished:
                    request.resumable = False
                if not finished:
                    # for streaming input request only
                    if self.chunk_transfer_adapter:
                        if self.vllm_config.model_config.stage_id != 0:
                            # Downstream async-chunk stages receive real payloads from the
                            # connector. This update only resumes polling for the next segment.
                            self.chunk_transfer_adapter.segment_finished_requests.discard(request.request_id)
                    outstanding_async_tokens = request.num_output_placeholders
                    if outstanding_async_tokens > 0:
                        # Discard only outputs that are already in flight and
                        # roll back their optimistic computed-token accounting.
                        request.async_tokens_to_discard = outstanding_async_tokens
                        request.num_computed_tokens -= outstanding_async_tokens
                        request.num_output_placeholders = 0
                    request.spec_token_ids = []
                    request._output_token_ids.clear()
                if finished:
                    final_kv_transfer_params, _ = self._free_request(request)
                    if final_kv_transfer_params is not None:
                        kv_transfer_params = final_kv_transfer_params
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            mm_output = _compact_native_duplex_prompt_metadata(
                mm_output,
                current_segment_token_ids=current_segment_token_ids,
                native_segment=bool(
                    self.vllm_config.model_config.stage_id == 0
                    and isinstance(getattr(request, "model_intermediate_buffer", None), dict)
                    and isinstance(request.model_intermediate_buffer.get("duplex"), dict)
                    and request.model_intermediate_buffer["duplex"].get("data_plane")
                ),
            )

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or mm_output is not None or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        multimodal_output=mm_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        is_segment_finished=is_segment_finished,
                        new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id, None),
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if self.chunk_transfer_adapter is not None and (
                inter_stage_output is not None or is_segment_finished or finished
            ):
                self.chunk_transfer_adapter.save_async(
                    inter_stage_output,
                    request,
                    is_segment_finished,
                )

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup_receiver(
                        request.request_id,
                    )

        # [Omni] Cleanup state for finished requests
        for req in stopped_running_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # Same for preempted
        for req in stopped_preempted_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if scheduler_kv_connector_stats is not None and not scheduler_kv_connector_stats.is_empty():
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        # FIXME: finished_req_ids_dict is unconditionally initialized as
        # defaultdict(set) in __init__ (not gated by include_finished_set).
        # This branch is therefore always eligible once any client_index is
        # populated; revisit when wiring streaming-only / upstream semantics.
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                eco = engine_core_outputs.get(client_index)
                if eco is None:
                    eco = EngineCoreOutputs()
                    engine_core_outputs[client_index] = eco
                emitted = {o.request_id for o in eco.outputs}
                for req_id in finished_set:
                    if req_id not in emitted:
                        eco.outputs.append(EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT))
                eco.finished_requests = finished_set
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        self._capture_omni_connector_output(model_runner_output)

        # Free blocks that were held for transfer (kv_ready and
        # active_kv_transfers updates already done before the per-request loop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            self.active_kv_transfers.discard(req_id)
                            self.pending_stop_after_extraction.discard(req_id)
                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
                except Exception:
                    init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        if _LOG_HANDOFF_DIAG:
            stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
            diag_stage_matches = _DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES
        else:
            diag_stage_matches = False
        if diag_stage_matches:
            handoff_diag_end = monotonic()
            req_ids = [
                output.request_id
                for client_outputs in engine_core_outputs.values()
                for output in client_outputs.outputs
            ]
            if req_ids:
                logger.info(
                    "[HANDOFF-DIAG] event=core-output-ready stage=%s wall=%.6f mono=%.6f reqs=%s "
                    "payload_mib=%.3f scheduler_update_ms=%.3f",
                    stage_id,
                    time(),
                    handoff_diag_end,
                    ",".join(req_ids),
                    _diagnostic_tensor_bytes(mm_outputs) / float(1 << 20),
                    (handoff_diag_end - handoff_diag_start) * 1000.0,
                )

        return engine_core_outputs

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            The Request objects that were aborted. Will not include any that
            were already finished.
        """
        # TODO(yrr): chunk transfer adapter & input_coordinator unified to one
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        # Realign stale ``request.status`` (chunk-transfer-adapter's
        # ``requests_origin_status`` table doesn't follow the
        # ``waiting → running`` admit transition; without this, an abort
        # arriving between admit and the next deque round-trip leaves
        # the request in ``self.running`` with ``status=WAITING`` and
        # upstream ``Scheduler.finish_requests`` silently fails to
        # release the worker's ``input_batch`` slot -- after
        # ``max_num_seqs`` such aborts new requests hang at
        # ``chunks=0``). Only the ``async_chunk`` path triggers the
        # staleness; with ``async_chunk`` disabled this is a cheap O(n)
        # no-op over an already-aligned set, kept unconditional so the
        # abort path stays uniform across configurations. See
        # ``OmniSchedulerMixin._realign_request_status_to_queues`` and
        # #3774 discussion.
        self._realign_request_status_to_queues(request_ids)

        finished = super().finish_requests(request_ids, finished_status)

        # Defensive post-finish purge: belt-and-suspenders to the
        # realignment above. Even after realign + ``super()``, corner
        # cases (mid-transition status, connector cleanups that pop
        # from ``self.requests`` without unwinding ``self.running``)
        # can leave already-finished or untracked entries in
        # ``self.running``. Sweep them now so the worker's
        # ``input_batch`` slot never pins a freed request and starves
        # new admissions. See ``OmniSchedulerMixin._purge_finished_from_running``.
        self._purge_finished_from_running()

        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            for request in finished:
                self._free_input_coordinator_request(request.request_id)
        return finished

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        if outstanding_async_tokens > 0:
            # Async scheduling may already have sampled the previous
            # segment's next token. Drop that late token instead of
            # appending it to the new streaming segment.
            session.async_tokens_to_discard = 1
            session.num_computed_tokens -= session.num_output_placeholders
            session.num_output_placeholders = 0
            session.spec_token_ids = []
        stage_id = self.vllm_config.model_config.stage_id
        if self.chunk_transfer_adapter and self.chunk_transfer_adapter.receives_chunks:
            self.chunk_transfer_adapter.requests_num_chunks_sent.pop(session.external_req_id, None)
            if stage_id != 0:
                # Downstream async-chunk stages receive real payloads from the
                # connector. This update only resumes polling for the next segment.
                self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
                # Do not replace prompt/additional_information here; the next
                # upstream chunk will populate them in chunk transfer adapter.
                session.arrival_time = update.arrival_time
                session.sampling_params = update.sampling_params
                if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                session.status = RequestStatus.WAITING
                if session in self.skipped_waiting:
                    self.skipped_waiting.remove_requests((session,))
                    self._enqueue_waiting_request(session)

                if self.log_stats:
                    session.record_event(EngineCoreEventType.QUEUED)
                return
        update_infos = (
            getattr(update, "model_intermediate_buffer", None),
            getattr(update, "additional_information", None),
        )
        replace_streaming_prompt = any(
            isinstance(info, dict)
            and isinstance(info.get("meta"), dict)
            and info["meta"].get("replace_streaming_prompt") is True
            for info in update_infos
        )
        if replace_streaming_prompt:
            # A rebase rewrites prefix identity, even at an equal/greater length.
            session._omni_pd_published_prompt_tokens = 0
            self._replace_streaming_session(session, update)
            return
        super()._update_request_as_session(session, update)
        if hasattr(update, "model_intermediate_buffer"):
            session.model_intermediate_buffer = update.model_intermediate_buffer

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        lineage_id = getattr(request, "kv_lineage_id", None)
        lineage_revision = int(getattr(request, "kv_lineage_revision", 0))
        if lineage_id and lineage_revision > 0 and request.status != RequestStatus.FINISHED_ABORTED:
            confirmed_computed = self._get_confirmed_num_computed_tokens(request)
            hash_block_size = int(getattr(self.kv_cache_manager.block_pool, "hash_block_size", 0))
            self._store_kv_lineage_snapshot(
                lineage_id,
                lineage_revision,
                request.block_hashes,
                confirmed_computed,
                hash_block_size,
            )
            logger.debug(
                "[kv-lineage] store id=%s parent=%d revision=%d computed=%d hashes=%d snapshot_found=%s seeded=%d",
                lineage_id,
                int(getattr(request, "kv_lineage_parent_revision", 0)),
                lineage_revision,
                confirmed_computed,
                len(request.block_hashes),
                bool(getattr(request, "kv_lineage_snapshot_found", False)),
                int(getattr(request, "kv_lineage_seeded_tokens", 0)),
            )

        self._omits_kv_transfer_cache.pop(request.request_id, None)

        # [Upstream compat] Discard request from in-flight prefills set added
        # upstream for routed-experts in-flight reservation tracking.
        # Use getattr for safety with test __new__ code paths.
        getattr(self, "_inflight_prefills", set()).discard(request)

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # Mirror the generation scheduler's try/finally pattern so the
        # input_coordinator entry is always pruned along every return path,
        # including the early returns for in-flight / waiting KV transfers
        # below. _free_input_coordinator_request is a no-op when the
        # coordinator is None, so the unconditional finally is safe.
        try:
            # 2. Omni Specific: Check if we need to transfer KV
            if self._should_transfer_kv_for_request(request_id):
                already_triggered = request_id in self.transfer_triggered_requests
                is_active = request_id in self.active_kv_transfers

                if already_triggered:
                    if is_active:
                        # It triggered but hasn't finished yet. We MUST wait.
                        logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                        self.waiting_for_transfer_free.add(request_id)
                        kv_xfer_params = None
                        return kv_xfer_params, None
                    elif request_id in self.waiting_for_transfer_free:
                        # Blocks held until KV extraction completes in a future step.
                        return None, None
                    else:
                        logger.debug(
                            f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                            "Freeing immediately."
                        )
                else:
                    self.waiting_for_transfer_free.add(request_id)
                    confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                    self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                    # Return KV transfer metadata so it propagates to RequestOutput
                    if request_id in self.requests_needing_kv_transfer:
                        transfer_data = self.requests_needing_kv_transfer[request_id]
                        kv_xfer_params = {
                            "past_key_values": transfer_data["block_ids"],
                            "kv_metadata": {
                                "seq_len": transfer_data["seq_len"],
                                "block_ids": transfer_data["block_ids"],
                            },
                        }
                        # Also update request.additional_information for good measure
                        add_info = getattr(request, "additional_information", None)
                        # If additional_information is an AdditionalInformationPayload-like object,
                        # unpack it into a plain dict.
                        if (
                            add_info is not None
                            and hasattr(add_info, "entries")
                            and isinstance(getattr(add_info, "entries"), dict)
                        ):
                            request.additional_information = deserialize_additional_information(add_info)
                            add_info = request.additional_information
                        if add_info is None:
                            request.additional_information = {}
                            add_info = request.additional_information
                        if isinstance(add_info, dict):
                            add_info.update(kv_xfer_params)

                    return kv_xfer_params, None

            # 3. Standard Freeing
            delay_free_blocks |= connector_delay_free_blocks
            if not delay_free_blocks:
                self._free_blocks(request)

            return kv_xfer_params, None
        finally:
            self._free_input_coordinator_request(request_id)
            # Normal completion runs through here, not finish_requests()
            # (the abort path) -- see vllm-project/vllm-omni#5349.
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.cleanup_receiver(request_id)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        if not need_send:
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""
