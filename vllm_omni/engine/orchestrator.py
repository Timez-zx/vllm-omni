"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.

Distributed membership (replica attach/detach, hub monitoring) is
handled by :class:`MembershipController`, which is injected optionally.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time as _time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.distributed.omni_connectors.utils.config import stage_receives_chunks
from vllm_omni.engine import OmniEngineCoreRequest, OmniPDPrefillPayload
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.duplexomni_pipeline import (
    DuplexOmniPipelineCoordinator,
    DuplexOmniPipelineIdentity,
    inject_codec_history,
    pipeline_identity_from_prompt,
)
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
    InteractionMessage,
    OutputMessage,
    PhysicalDCompletionWitnessMessage,
    RegisterRemoteReplicaMessage,
    ShutdownRequestMessage,
    StageMetricsMessage,
    StageSubmissionMessage,
    UnregisterRemoteReplicaMessage,
)
from vllm_omni.engine.orchestrator_monitor import create_orch_monitor, replica_key
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.experimental.fullduplex.engine.intermediate import (
    NATIVE_LAST_PROMPT_TOKEN_KEY,
    NATIVE_PROMPT_LEN_KEY,
    NATIVE_PROMPT_TOKEN_IDS_KEY,
    NATIVE_SEGMENT_TOKEN_IDS_KEY,
)
from vllm_omni.experimental.fullduplex.minicpmo45.sampling_state import (
    SAMPLING_STATE_KEY,
    SAMPLING_STATE_WIRE_KEY,
    unpack_sampling_state,
)
from vllm_omni.metrics.prometheus import OmniRequestCounter
from vllm_omni.metrics.stat_logger import OmniPrometheusStatLogger
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_LOG_HANDOFF_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "0") not in ("0", "", "false", "False")
_MINICPMO_PD_ONLY_DIAGNOSTIC = os.environ.get(
    "VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC",
    "0",
) not in ("0", "", "false", "False")
_DIAG_STAGE_RAW = os.environ.get("VLLM_OMNI_DIAG_STAGE")
_DIAG_STAGES = (
    None
    if _DIAG_STAGE_RAW is None
    else frozenset(stage.strip() for stage in _DIAG_STAGE_RAW.split(",") if stage.strip())
)


def _pd_snapshot_cache_bytes() -> int:
    raw = os.environ.get("VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES", str(8 << 30))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES=%r; using 8 GiB", raw)
        return 8 << 30


def _pd_snapshot_max_chunks() -> int:
    raw = os.environ.get("VLLM_OMNI_PD_SNAPSHOT_MAX_CHUNKS", "16")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid VLLM_OMNI_PD_SNAPSHOT_MAX_CHUNKS=%r; using 16", raw)
        return 16


def _extend_native_pd_feedback_budget(prompt: dict[str, Any], feedback_token_ids: list[int]) -> None:
    """Reserve the prompt rows Stage0 uses to replay D feedback on P.

    The final D token is the unit terminator and replaces the native boundary;
    only the preceding feedback rows add scheduler-visible prompt work.
    """
    model_buffer = prompt.get("model_intermediate_buffer")
    duplex = model_buffer.get("duplex") if isinstance(model_buffer, dict) else None
    if not isinstance(duplex, dict):
        return
    feedback = [int(token_id) for token_id in feedback_token_ids]
    duplex["pd_feedback_token_ids"] = feedback
    from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
        duplex_feedback_scheduler_rows,
    )

    extra_rows = duplex_feedback_scheduler_rows(feedback)
    if extra_rows == 0:
        return
    prompt_ids = prompt.get("prompt_token_ids")
    if not isinstance(prompt_ids, (list, tuple)):
        return
    existing_ids = [int(token_id) for token_id in prompt_ids]
    raw_scheduler_token = duplex.get("scheduler_token_id")
    try:
        scheduler_token = int(raw_scheduler_token)
    except (TypeError, ValueError):
        scheduler_token = existing_ids[-1] if existing_ids else 0
    prompt["prompt_token_ids"] = [*existing_ids, *([scheduler_token] * extra_rows)]
    raw_budget = duplex.get("scheduler_token_budget")
    try:
        scheduler_budget = int(raw_budget)
    except (TypeError, ValueError):
        scheduler_budget = len(existing_ids)
    duplex["scheduler_token_budget"] = scheduler_budget + extra_rows


if TYPE_CHECKING:
    from vllm_omni.experimental.fullduplex.engine.contracts import (
        DuplexControlPlanePort,
        DuplexOutputContext,
        DuplexOutputDecision,
        DuplexRequestIdentity,
        DuplexRuntimeExtension,
        DuplexStageRequestContext,
        DuplexStageSubmission,
        DuplexStageSubmissionResult,
    )
    from vllm_omni.experimental.fullduplex.engine.duplex_session import (
        DuplexSessionRuntimeManager,
        DuplexSessionRuntimeState,
    )
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


def _build_terminal_empty_output(
    request_id: str,
    *,
    final_output_type: str | None,
    audio_sample_rate: int = 24000,
) -> RequestOutput:
    """Build a terminal empty output when no downstream stage input exists."""
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    if final_output_type == "audio":
        completion.multimodal_output = {
            "audio": torch.zeros((0,), dtype=torch.float32),
            "sr": audio_sample_rate,
        }
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    raw_additional_information = prompt.get("additional_information")
    model_intermediate_buffer = prompt.get("model_intermediate_buffer")
    pd_prefill_payload = prompt.get("pd_prefill_payload")
    wire_payload: dict[str, Any] | None = None
    if isinstance(raw_additional_information, dict):
        wire_payload = dict(raw_additional_information)
    additional_info_payload = serialize_additional_information(
        wire_payload,
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=prompt.get("cache_salt"),
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
        model_intermediate_buffer=model_intermediate_buffer if isinstance(model_intermediate_buffer, dict) else None,
        pd_prefill_payload=pd_prefill_payload if isinstance(pd_prefill_payload, OmniPDPrefillPayload) else None,
        cache_token_ids=prompt.get("cache_token_ids"),
        prefill_only=prompt.get("prefill_only") is True,
        kv_lineage_id=prompt.get("kv_lineage_id"),
        kv_lineage_parent_revision=int(prompt.get("kv_lineage_parent_revision", 0)),
        kv_lineage_revision=int(prompt.get("kv_lineage_revision", 0)),
        kv_lineage_prefix_tokens=max(0, int(prompt.get("kv_lineage_prefix_tokens", 0))),
    )


@dataclass
class _PDPrefillSnapshot:
    revision: int
    prompt_token_ids: tuple[int, ...]
    output: dict[str, Any]
    nbytes: int
    # Leading chunks that have already been copied into dedicated shared
    # slabs. Future revisions must reuse these slabs rather than repeatedly
    # concatenating the full history.
    packed_prefix_chunks: int = 0


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    final_output_stage_ids: set[int] = field(default_factory=set)
    finished_final_output_stage_ids: set[int] = field(default_factory=set)

    # Wall-clock timestamp when the client-facing engine request was accepted.
    request_timestamp: float = 0.0

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None
    pd_mrope_feature_metadata: list[dict[str, Any]] = field(default_factory=list)
    pd_prefill_multimodal_output: dict[str, Any] | None = None
    pd_prefill_parent_snapshot: _PDPrefillSnapshot | None = None
    pd_prefill_lineage_id: str | None = None
    pd_prefill_revision: int = 0
    pd_prefill_max_parent_rows: int = 0
    pd_prefill_prompt_token_ids: tuple[int, ...] = ()
    # A finite arrival-prefill has completed on P and is being imported into
    # D's ordinary prefix cache through the cache-only control path. It never
    # enters D's inference scheduler; the flag tracks the pending background
    # cache import after the client-visible P-ready ACK.
    pd_decode_cache_sync_pending: bool = False
    # The client-visible prefill-only request completes as soon as P has
    # materialized the reusable snapshot.  D cache population continues in a
    # request-scoped background task and must not emit a second terminal item.
    pd_prefill_ready_emitted: bool = False
    # D may allocate/register its request-scoped destination blocks while P
    # is still computing.  Arrival imports become disposable prefix cache;
    # formal-query imports stay pinned until the paired D ADD is admitted.
    pd_early_cache_sync_task: asyncio.Task[dict[str, Any]] | None = None
    pd_early_cache_sync_result: dict[str, Any] | None = None
    pd_early_cache_sync_error: BaseException | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())

    # Per-request pipeline timing accumulator (milliseconds)
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    duplex_identity: DuplexRequestIdentity | None = None
    duplex_stage_fences: dict[int, DuplexFence] = field(default_factory=dict)
    duplex_config_generation: int = -1
    duplexomni_pipeline_identity: DuplexOmniPipelineIdentity | None = None
    running_counter_registered: bool = False


@dataclass
class _DuplexOmniPendingTalker:
    output: Any
    replica_id: int
    is_streaming_session: bool
    is_final_update: bool


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Tokens from the current raw segment boundary. The vLLM output processor
    # does not guarantee that EngineCoreOutput.new_token_ids survives on the
    # processed RequestOutput used by the routing layer.
    segment_token_ids: list[int] = field(default_factory=list)
    segment_output_metadata: dict[str, Any] = field(default_factory=dict)
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)
    # Synchronous stage-transition capability installed by the orchestrator
    # while the downstream input processor consumes upstream token output.
    source_token_decoder: Callable[..., str] | None = None


class _OrchestratorDuplexStagePort:
    """Adapts generic stage pools to the model-neutral duplex control plane."""

    def __init__(
        self,
        *,
        stage_pools: list[StagePool],
        request_states: dict[str, OrchestratorRequestState],
        running_counter: OmniRequestCounter | None,
        cleanup_request_ids: Callable[..., Any],
        async_chunk: bool,
        prewarm_async_chunk_stages: Callable[
            [str, Any, OrchestratorRequestState],
            Awaitable[None],
        ],
        schedule_pd_early_cache_sync: Callable[..., None] | None = None,
        pd_pair: tuple[int, int] | None = None,
    ) -> None:
        self._stage_pools = stage_pools
        self._request_states = request_states
        self._running_counter = running_counter
        self._cleanup_request_ids = cleanup_request_ids
        self._async_chunk = async_chunk
        self._pd_pair = pd_pair
        self._prewarm_async_chunk_stages = prewarm_async_chunk_stages
        self._schedule_pd_early_cache_sync = schedule_pd_early_cache_sync

    @staticmethod
    def _native_pd_prefix_prediction(
        prompt: dict[str, Any],
        bridge: dict[str, Any],
        *,
        already_submitted: bool,
    ) -> list[int]:
        """Predict P's prompt prefix before its segment starts executing."""
        delta_ids = [int(token_id) for token_id in prompt.get("prompt_token_ids", ())]
        if not already_submitted:
            return delta_ids

        model_buffer = prompt.get("model_intermediate_buffer")
        meta = model_buffer.get("meta") if isinstance(model_buffer, dict) else None
        replace_prompt = isinstance(meta, dict) and meta.get("replace_streaming_prompt") is True
        if replace_prompt:
            predicted = delta_ids
            if meta.get("retain_streaming_output_tokens") is True:
                retained = bridge.get("pd_duplex_prefill_sample_token_ids")
                if isinstance(retained, (list, tuple)) and retained:
                    try:
                        offset = int(meta.get("retained_output_insert_offset", len(predicted)))
                    except (TypeError, ValueError):
                        offset = len(predicted)
                    offset = max(0, min(offset, len(predicted)))
                    predicted[offset:offset] = [int(token_id) for token_id in retained]
            return predicted

        previous = bridge.get("pd_duplex_remote_prompt_token_ids")
        if not isinstance(previous, (list, tuple)):
            return delta_ids
        # P's typed token IDs are already normalized. Copy the list, not a
        # Python int() loop over the full session history on every unit.
        return [*previous, *delta_ids]

    @property
    def stage_count(self) -> int:
        return len(self._stage_pools)

    def sampling_defaults(self) -> tuple[object, ...]:
        return tuple(pool.stage_client.default_sampling_params for pool in self._stage_pools)

    @staticmethod
    def _sync_bridge_state(
        request_state: OrchestratorRequestState,
        context: DuplexStageRequestContext,
    ) -> None:
        duplex_state = request_state.streaming.bridge_states.setdefault("duplex", {})
        if not isinstance(duplex_state, dict):
            duplex_state = {}
            request_state.streaming.bridge_states["duplex"] = duplex_state
        previous_epoch = duplex_state.get("epoch")
        if not isinstance(duplex_state.get("model_turn_id"), int) or previous_epoch != context.fence.epoch:
            duplex_state["model_turn_id"] = context.fence.turn_id
        duplex_state.update(
            {
                "session_id": context.session_id,
                "fence": context.fence,
                "incarnation": context.fence.incarnation,
                "epoch": context.fence.epoch,
                "turn_id": context.fence.turn_id,
                "response_seq": context.fence.response_seq,
                "session_config": dict(context.session_config),
                "runtime_config": dict(context.runtime_config),
            }
        )

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        from vllm_omni.experimental.fullduplex.engine.contracts import DuplexRequestIdentity

        request_state = self._request_states.get(context.request_id)
        if request_state is None:
            request_state = OrchestratorRequestState(
                request_id=context.request_id,
                prompt=None,
                sampling_params_list=list(context.sampling_params),
                final_stage_id=context.final_stage_id,
                duplex_config_generation=context.config_generation,
            )
            request_state.streaming.enabled = True
            self._request_states[context.request_id] = request_state
        elif request_state.duplex_config_generation != context.config_generation:
            request_state.sampling_params_list = list(context.sampling_params)
            request_state.duplex_config_generation = context.config_generation
        request_state.duplex_identity = DuplexRequestIdentity(
            session_id=context.session_id,
            fence=context.fence,
        )
        self._sync_bridge_state(request_state, context)

    def _validate_native_pd_logical_context(self, context: DuplexStageRequestContext, prompt_tokens: int) -> None:
        """Reject one session before ingress; never recycle its history/KV.

        The P seed becomes D's last prompt token. Reserve it and D's complete
        finite generation budget so a unit cannot be truncated by the logical
        limit. With the default D budget of 20, this conservatively reserves 21
        tokens. Physical sliding-window bounds do not extend this limit.
        """
        from vllm_omni.experimental.fullduplex.engine.contracts import DuplexContextLimitError

        assert self._pd_pair is not None
        p_stage, d_stage = self._pd_pair
        limit = min(self._stage_pools[i].stage_vllm_config.model_config.max_model_len for i in (p_stage, d_stage))
        decode_budget = getattr(context.sampling_params[d_stage], "max_tokens", None)
        if not isinstance(decode_budget, int) or isinstance(decode_budget, bool) or decode_budget < 1:
            raise ValueError("Native P/D context admission requires a positive finite D max_tokens")
        generation_tokens = 1 + decode_budget
        if prompt_tokens + generation_tokens > limit:
            raise DuplexContextLimitError(
                prompt_tokens=prompt_tokens, generation_tokens=generation_tokens, max_model_len=limit
            )

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        from vllm_omni.experimental.fullduplex.engine.contracts import DuplexStageSubmissionResult

        context = submission.context
        request_state = self._request_states.get(context.request_id)
        if request_state is None:
            raise RuntimeError(f"duplex request was not preregistered: {context.request_id}")
        original_prompt = dict(submission.prompt)
        prompt = original_prompt
        slot_ready_epoch = _time.time()
        slot_wait_started = _time.monotonic()
        is_pd_prefill = self._pd_pair is not None and context.stage_id == self._pd_pair[0]
        pd_predicted_prompt_token_ids: list[int] | None = None
        if is_pd_prefill:
            bridge = request_state.streaming.bridge_states
            if submission.already_submitted:
                decode_ready = bridge.get("pd_duplex_decode_ready")
                if isinstance(decode_ready, asyncio.Event):
                    await decode_ready.wait()
                route_error = bridge.get("pd_duplex_prefill_raw_error")
                if route_error is not None:
                    raise RuntimeError(f"previous native P/D route failed for {context.request_id}: {route_error}")
                if self._request_states.get(context.request_id) is not request_state:
                    raise RuntimeError(f"duplex request was closed while waiting for D: {context.request_id}")
                feedback = bridge.get("pd_duplex_feedback_token_ids", [])
                if feedback:
                    prompt = copy.deepcopy(original_prompt)
                    _extend_native_pd_feedback_budget(prompt, list(feedback))
                    policy_state = bridge.get("pd_duplex_feedback_sampling_state")
                    if policy_state is None:
                        raise RuntimeError("Native D feedback is missing sampling state")
                    prompt["model_intermediate_buffer"]["duplex"]["pd_feedback_sampling_state"] = policy_state
            pd_predicted_prompt_token_ids = self._native_pd_prefix_prediction(
                prompt,
                bridge,
                already_submitted=submission.already_submitted,
            )
            # This exception stays in handle_append's per-operation error
            # path. Do not defer the check to the GPU runner: its fixed-size
            # token buffer would otherwise raise and kill unrelated sessions.
            self._validate_native_pd_logical_context(context, len(pd_predicted_prompt_token_ids))
            if submission.already_submitted and feedback:
                bridge.pop("pd_duplex_feedback_token_ids", None)
                bridge.pop("pd_duplex_feedback_sampling_state", None)
            # One request per session may be in P→D flight.  This is the
            # model's recurrence dependency, not a global admission gate:
            # unrelated sessions continue independently.
            bridge["pd_duplex_decode_ready"] = asyncio.Event()
            model_buffer = original_prompt.get("model_intermediate_buffer")
            duplex = model_buffer.get("duplex") if isinstance(model_buffer, dict) else None
            seq = duplex.get("seq") if isinstance(duplex, dict) else None
            payload = duplex.get("payload") if isinstance(duplex, dict) else None
            input_source = (
                "auto_continuation"
                if isinstance(payload, dict) and payload.get("duplex_input_source") == "auto_continuation"
                else "real_input"
            )
            raw_input_unit_index = payload.get("input_unit_index") if isinstance(payload, dict) else None
            try:
                input_unit_index = (
                    int(raw_input_unit_index)
                    if raw_input_unit_index is not None and not isinstance(raw_input_unit_index, bool)
                    else None
                )
            except (TypeError, ValueError):
                input_unit_index = None
            video_frames = payload.get("video_frames") if isinstance(payload, dict) else None
            input_video_frames = len(video_frames) if isinstance(video_frames, list) else 0
            bridge["pd_duplex_active_slot"] = {
                "seq": seq,
                "input_unit_index": input_unit_index,
                "source": input_source,
                "input_video_frames": input_video_frames,
                "arrival_video_frames": 0,
                "vision_fallback_frames": 0,
                "arrival_audio_units": 0,
                "audio_fallback_units": 0,
                "local_cached_tokens": -1,
                "external_cached_tokens": -1,
                "computed_tokens": -1,
                "kv_transfer_selected_blocks": -1,
                "kv_transfer_selected_tokens": -1,
                "kv_transfer_selected_bytes": -1,
                "kv_transfer_write_submit_to_d_ready_ms": -1.0,
                "ready_epoch": slot_ready_epoch,
                "wait_previous_d_ms": (_time.monotonic() - slot_wait_started) * 1000.0,
            }
            # P exposes a resumable segment boundary as a raw EngineCore
            # output.  FINAL_ONLY intentionally emits no processed output for
            # that boundary, so the raw-output path owns exactly one P->D
            # dispatch for this slot.
            bridge["pd_duplex_prefill_raw_routed"] = False
            bridge.pop("pd_duplex_prefill_raw_scheduled_id", None)
            bridge.pop("pd_duplex_prefill_raw_routed_id", None)
            if seq is None:
                seq = int(bridge.get("pd_duplex_decode_sequence", 0)) + 1
            decode_engine_req_id = f"{context.request_id}-{int(seq) & 0xFFFFFFFF:08x}"
            bridge["pd_duplex_decode_sequence"] = int(seq)
            bridge["pd_decode_engine_request_id"] = decode_engine_req_id
            bridge["pd_decode_transfer_id"] = f"xfer-{decode_engine_req_id}"
            request_state.pd_early_cache_sync_task = None
            request_state.pd_early_cache_sync_result = None
            request_state.pd_early_cache_sync_error = None
        stage_sampling_params = context.stage_sampling_params
        if is_pd_prefill and isinstance(stage_sampling_params, SamplingParams):
            stage_sampling_params = stage_sampling_params.clone()
            stage_sampling_params.max_tokens = 1
            stage_sampling_params.stop = []
            stage_sampling_params.stop_token_ids = []
            stage_sampling_params.include_stop_str_in_output = False
            extra_args = dict(stage_sampling_params.extra_args or {})
            kv_params = dict(extra_args.get("kv_transfer_params") or {})
            kv_params.update(
                {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "transfer_id": f"xfer-{context.request_id}",
                }
            )
            extra_args["kv_transfer_params"] = kv_params
            stage_sampling_params.extra_args = extra_args
        request = build_engine_core_request_from_tokens(
            request_id=context.request_id,
            prompt=prompt,
            params=stage_sampling_params,
            model_config=self._stage_pools[context.stage_id].stage_vllm_config.model_config,
            resumable=True,
        )
        request.external_req_id = request.request_id
        # The native duplex control plane submits token deltas directly to
        # stage 0, bypassing _handle_add_request/_handle_streaming_update.
        # Preserve the unmodified stage-0 delta for the paired D request: P's
        # scheduler may extend its own persistent prompt, while D must receive
        # the same media delta through its independent persistent request.
        request_state.prompt = prompt
        request_state.streaming.bridge_states["pd_decode_prompt"] = original_prompt
        pool = self._stage_pools[context.stage_id]
        if submission.already_submitted:
            replica_id = await pool.submit_update(context.request_id, request_state, request)
        else:
            replica_id = await pool.submit_initial(context.request_id, request_state, request, prompt_text=None)
            if self._async_chunk and context.stage_id == 0:
                await self._prewarm_async_chunk_stages(
                    context.request_id,
                    request,
                    request_state,
                )
        if is_pd_prefill and self._schedule_pd_early_cache_sync is not None:
            bridge = request_state.streaming.bridge_states
            assert pd_predicted_prompt_token_ids is not None
            self._schedule_pd_early_cache_sync(
                context.request_id,
                request_state,
                engine_request_id=bridge["pd_decode_engine_request_id"],
                prompt_token_ids=pd_predicted_prompt_token_ids,
            )
        request_state.duplex_stage_fences[context.stage_id] = context.fence
        request_state.stage_submit_ts[context.stage_id] = _time.time()
        if not request_state.running_counter_registered and self._running_counter is not None:
            self._running_counter.increment()
            request_state.running_counter_registered = True
        return DuplexStageSubmissionResult(
            request_id=context.request_id,
            stage_id=context.stage_id,
            replica_id=replica_id,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        await self._cleanup_request_ids(request_ids, abort=abort)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    # Class-level defaults so tests that bypass __init__ via object.__new__
    # don't AttributeError when transfer / counter emit paths access them.
    _running_counter: OmniRequestCounter | None = None
    _transfer_emitter: Any = None
    _stat_logger: OmniPrometheusStatLogger | None = None
    duplex_control_plane: DuplexControlPlanePort | None = None

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[EngineQueueMessage],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
        membership_controller: MembershipController | None = None,
        running_counter: OmniRequestCounter | None = None,
        transfer_emitter: Any = None,
        log_stats: bool = False,
        enable_orch_monitor: bool = False,
        duplex_runtime_extension: DuplexRuntimeExtension | None = None,
        enable_duplex_control: bool = False,
        duplex_session_config: DuplexSessionRuntimeConfig | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools
        self._orch_monitor = create_orch_monitor(
            enabled=enable_orch_monitor,
            replica_sampler=self._sample_replica_metrics,
        )
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.available_replica_ids():
                self._orch_monitor.register_replica(stage_id, replica_id)

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_prefill_remote: dict[str, Any] | None = None
        self._pd_kv_params: dict[str, Any] = {}
        # Native duplex keeps one logical application request alive while D
        # executes one ordinary finite engine request per model unit.  Map
        # those physical D ids back to the logical session id for raw-output
        # routing; processed RequestOutput already uses external_req_id.
        self._pd_decode_request_aliases: dict[str, str] = {}
        # Arrival requests are linear within each application session, so only
        # the newest conditioning snapshot for a lineage is needed.
        self._pd_prefill_snapshots: OrderedDict[str, _PDPrefillSnapshot] = OrderedDict()
        self._pd_prefill_snapshot_bytes = 0
        self._pd_prefill_snapshot_limit_bytes = _pd_snapshot_cache_bytes()
        self._pd_prefill_snapshot_max_chunks = _pd_snapshot_max_chunks()
        self._pd_snapshot_hidden_layer = 24
        # Cache-sync completion is request-scoped.  Never await it from the
        # single stage-output polling loop: doing so serializes unrelated
        # sessions and lets already-produced stage outputs pile up behind one
        # slow sync. Cache sync remains request-scoped while each application
        # session continues to send complete canonical prompts.
        self._pd_cache_sync_tasks: dict[str, asyncio.Task[Any]] = {}
        self._background_collective_rpc_tasks: set[asyncio.Task[None]] = set()
        # Raw resumable-P boundaries must not await D admission from the one
        # global stage-output poller.  Ownership remains request-scoped: the
        # model recurrence permits at most one P->D route per logical session,
        # while unrelated sessions may submit D independently.
        self._pd_raw_route_tasks: dict[str, asyncio.Task[None]] = {}
        # P output routing is kept FIFO but runs independently from the global
        # stage poller. Snapshot compaction can then yield to D/Talker output
        # consumption instead of blocking every stage on the orchestrator
        # event loop.
        self._pd_prefill_output_queue: asyncio.Queue[tuple[int, int, list[Any], float]] | None = None
        # Arrival-prefill requests may populate vLLM's sender-side media
        # cache, leaving the final request with hash-only feature references.
        # Retain only the tiny values needed to reconstruct M-RoPE on D.
        self._pd_mrope_values_by_identifier: dict[str, dict[str, Any]] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")
            prefill_remote = pd_config.get("prefill_remote")
            self._pd_prefill_remote = dict(prefill_remote) if isinstance(prefill_remote, dict) else None
            self._pd_snapshot_hidden_layer = int(pd_config.get("snapshot_hidden_layer", 24))
        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._init_metrics_state(stage_pools, running_counter, transfer_emitter, log_stats=log_stats)

        self._cfg_tracker = CfgCompanionTracker()
        self._stage_input_processors: dict[int, Any] = {}
        self._duplexomni_pipeline = DuplexOmniPipelineCoordinator()
        self._duplexomni_pending_talker: dict[str, _DuplexOmniPendingTalker] = {}

        self.duplex_control_plane: DuplexControlPlanePort | None = None
        self._duplex_reaper_interval_s = 1.0
        if enable_duplex_control:
            from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import DuplexControlPlane
            from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseConfig

            runtime_session_config = duplex_session_config or DuplexSessionRuntimeConfig()
            self._duplex_reaper_interval_s = runtime_session_config.reaper_interval_s

            self.duplex_control_plane = DuplexControlPlane(
                extension=duplex_runtime_extension,
                stage_port=_OrchestratorDuplexStagePort(
                    stage_pools=self.stage_pools,
                    request_states=self.request_states,
                    running_counter=self._running_counter,
                    cleanup_request_ids=self._cleanup_request_ids,
                    async_chunk=self.async_chunk,
                    pd_pair=self._pd_pair,
                    prewarm_async_chunk_stages=self._prewarm_async_chunk_stages,
                    schedule_pd_early_cache_sync=self._schedule_pd_early_cache_sync,
                ),
                result_sink=self.rpc_async_queue,
                lifecycle_sink=self.output_async_queue,
                lease_config=DuplexLeaseConfig(
                    idle_ttl_s=runtime_session_config.idle_ttl_s,
                    disconnect_grace_s=runtime_session_config.disconnect_grace_s,
                ),
                max_sessions=runtime_session_config.max_sessions,
                completed_append_limit=runtime_session_config.completed_append_cache_size,
            )

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False
        self._fatal_error: str | None = None
        self._fatal_error_stage_id: int | None = None

        # Distributed membership (optional, injected by DistStageRuntime)
        self._membership = membership_controller

    def _init_metrics_state(
        self,
        stage_pools: list[StagePool],
        running_counter: OmniRequestCounter | None,
        transfer_emitter: Any,
        log_stats: bool = False,
    ) -> None:
        """Wire up all metric-related orchestrator state.

        Sets ``self._running_counter`` and ``self._transfer_emitter``
        (both optional, used by request-add / forward paths), builds the
        ``(stage_id, replica_id) ↔ engine_idx`` lookup used at record() time,
        and best-effort constructs the ``OmniPrometheusStatLogger`` wrap
        that exposes ~37 upstream ``vllm:*`` families with per-(stage,
        replica) labels. Failure to build the wrap is logged and metrics
        are simply disabled — orchestrator construction continues so unit
        tests with a minimal ``vllm_config`` still pass.

        ``log_stats=False`` short-circuits the wrap entirely so the
        ~65 upstream ``vllm:*`` families are not registered in the
        Prometheus default registry at all. The per-step record() path
        already no-ops on ``scheduler_stats is None`` (which is what
        the upstream scheduler returns when its own log_stats is False),
        so this gate is mainly to keep the ``/metrics`` surface clean
        when the user did not request stats.
        """
        self._running_counter = running_counter
        self._transfer_emitter = transfer_emitter

        # Flat engine_idx ↔ (stage, replica) maps. The reverse map is
        # consulted at record() time to translate the orchestrator's
        # (stage_id, replica_id) loop variables into an engine_idx the
        # underlying PrometheusStatLogger can address.
        stage_replica_map: dict[int, tuple[str, str]] = {}
        self._stage_replica_to_engine_idx: dict[tuple[int, int], int] = {}
        flat_idx = 0
        for stage_id, pool in enumerate(stage_pools):
            for replica_id in range(pool.num_replicas):
                stage_replica_map[flat_idx] = (str(stage_id), str(replica_id))
                self._stage_replica_to_engine_idx[(stage_id, replica_id)] = flat_idx
                flat_idx += 1

        if not log_stats:
            self._stat_logger = None
            return

        vllm_config_for_stats = next(
            (p.stage_vllm_config for p in stage_pools if p.stage_vllm_config is not None),
            None,
        )
        if vllm_config_for_stats is None:
            self._stat_logger = None
            return
        try:
            self._stat_logger = OmniPrometheusStatLogger(
                vllm_config=vllm_config_for_stats,
                stage_replica_map=stage_replica_map,
            )
        except Exception:
            # Minimal vllm_config in unit-test contexts can lack fields the
            # upstream PrometheusStatLogger expects. Skip wrap rather than
            # break orchestrator construction.
            logger.exception("[Orchestrator] OmniPrometheusStatLogger init failed; metrics wrap disabled")
            self._stat_logger = None

    @property
    def duplex_sessions(self) -> DuplexSessionRuntimeManager:
        if self.duplex_control_plane is None:
            raise RuntimeError("duplex control plane is disabled")
        return self.duplex_control_plane.sessions

    def _require_duplex_control_plane(self) -> DuplexControlPlanePort:
        if self.duplex_control_plane is None:
            raise RuntimeError("duplex control plane is disabled")
        return self.duplex_control_plane

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        if self._pd_pair is not None:
            self._pd_prefill_output_queue = asyncio.Queue()

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        # Start membership watcher if distributed mode is active.
        membership_watcher: asyncio.Task[None] | None = None
        if self._membership is not None:
            self._membership.install_unregister_handlers(
                output_queue=self.output_async_queue,
                cleanup_callback=lambda ids: self._cleanup_request_ids(ids, abort=True),
            )
            membership_watcher = self._membership.start()

        tasks = [request_task, output_task]
        if self._pd_prefill_output_queue is not None:
            tasks.append(
                asyncio.create_task(
                    self._pd_prefill_output_worker(),
                    name="orchestrator-pd-prefill-output-worker",
                )
            )
        if self.duplex_control_plane is not None:
            tasks.append(asyncio.create_task(self._duplex_reaper_loop(), name="orchestrator-duplex-reaper"))
        if membership_watcher is not None:
            tasks.append(membership_watcher)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except EngineDeadError as e:
            # EngineDeadError from _orchestration_loop means the diffusion
            # engine died.  All pending requests were already notified and
            # _shutdown_event was already set by the loop's handler.
            # During teardown this is expected; the finally block handles
            # proper cleanup.  Do not re-raise.
            logger.info("[Orchestrator] Engine dead during shutdown: %s", e)
            if self._fatal_error is None:
                self._fatal_error = str(e) or "Stage engine died"
            await self.rpc_async_queue.put(
                ErrorMessage(
                    error=self._fatal_error or str(e),
                    fatal=True,
                    stage_id=self._fatal_error_stage_id,
                )
            )
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

            await self._cancel_pd_raw_route_tasks(
                reason="orchestrator shutdown",
            )

            cache_sync_tasks = list(self._pd_cache_sync_tasks.values())
            for task in cache_sync_tasks:
                if not task.done():
                    task.cancel()
            if cache_sync_tasks:
                await asyncio.gather(*cache_sync_tasks, return_exceptions=True)
            self._pd_cache_sync_tasks.clear()

            if self.duplex_control_plane is not None:
                await self.duplex_control_plane.shutdown()

            if self._fatal_error is not None:
                await self._drain_pending_requests_on_fatal()

            if self._membership is not None:
                await self._membership.drain_tasks(timeout=10.0)
                self._membership.shutdown()

            self._orch_monitor.flush()
            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # ---- Request handling ----

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.type

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif self.duplex_control_plane is not None and self.duplex_control_plane.accepts(msg):
                self.duplex_control_plane.dispatch(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "interaction":
                await self._handle_interaction(msg)
            elif msg_type == "collective_rpc":
                if msg.method == "preencode_minicpmo45_vision":
                    self._schedule_background_collective_rpc(msg)
                else:
                    await self._handle_collective_rpc(msg)
            elif isinstance(msg, RegisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_register(msg.stage_id, msg.replica_id)
                    self._orch_monitor.register_replica(msg.stage_id, msg.replica_id)
            elif isinstance(msg, UnregisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_unregister(msg.stage_id, msg.input_addr)
            elif isinstance(msg, ShutdownRequestMessage):
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                # Pre-mark stage clients as shutting down to prevent
                # proc_monitor daemon threads from flagging normal
                # process exit as EngineDeadError during teardown.
                for pool in self.stage_pools:
                    for client in pool.clients:
                        if hasattr(client, "_shutting_down"):
                            client._shutting_down = True
                # Stage teardown runs once in run()'s finally after the
                # orchestration loop observes _shutdown_event and exits.
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    async def _duplex_reaper_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._duplex_reaper_interval_s,
                )
            except TimeoutError:
                plane = self.duplex_control_plane
                if plane is not None:
                    try:
                        await plane.reap_expired()
                    except Exception:
                        logger.exception("[Orchestrator] Duplex expiry cleanup failed; retrying on next tick")

    async def _handle_add_request(self, msg: StageSubmissionMessage) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg.request_id
        prompt = msg.prompt
        original_prompt = msg.original_prompt
        sampling_params_list = msg.sampling_params_list
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg.final_stage_id
        final_output_stage_ids = set(msg.final_output_stage_ids or [final_stage_id])

        logger.debug(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            request_timestamp=float(msg.request_timestamp or _time.time()),
            mm_features=getattr(prompt, "mm_features", None),
        )
        pipeline_identity = pipeline_identity_from_prompt(original_prompt)
        if pipeline_identity is not None:
            self._duplexomni_pipeline.register(request_id, pipeline_identity, original_prompt)
            req_state.duplexomni_pipeline_identity = pipeline_identity
        if self._pd_pair is not None:
            req_state.pd_mrope_feature_metadata = self._capture_pd_mrope_metadata(req_state.mm_features)
            self._prepare_pd_prefill_snapshot_request(prompt, req_state)
        self.request_states[request_id] = req_state
        self._register_running_request(req_state)
        req_state.streaming.enabled = bool(getattr(prompt, "resumable", False))
        enqueue_ts = msg.enqueue_ts
        if enqueue_ts > 0:
            req_state.pipeline_timings["queue_wait_ms"] = (_time.perf_counter() - enqueue_ts) * 1000.0
        preprocess_ms = msg.preprocess_ms
        if preprocess_ms > 0:
            req_state.pipeline_timings["preprocess_ms"] = preprocess_ms
        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[0].submit_initial(
            request_id,
            req_state,
            prompt,
            prompt_text=msg.output_prompt_text,
        )
        if self._pd_pair is not None:
            self._schedule_pd_early_cache_sync(request_id, req_state)

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: StageSubmissionMessage) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg.request_id
        request = msg.prompt
        final_stage_id = msg.final_stage_id
        req_state = self.request_states.get(request_id)
        if req_state is None:
            # Streaming updates always follow the first-chunk add_request
            # through the same ordered submission queue, so an unknown id
            # here can only mean the request already finished or was
            # aborted (e.g. the client disconnected). Re-adding it would
            # resurrect a headless session that keeps cycling through the
            # stages with nobody consuming its outputs (issue #4271).
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s; dropping (request finished or aborted)",
                request_id,
            )
            return

        if msg.sampling_params_list:
            req_state.sampling_params_list = msg.sampling_params_list

        req_state.streaming.enabled = True
        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[stage_id].submit_update(
            request_id,
            req_state,
            request,
            prompt_text=msg.output_prompt_text,
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, request, req_state)

    async def _handle_add_companion(self, msg: AddCompanionRequestMessage) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg.companion_id
        parent_id = msg.parent_id
        role = msg.role
        companion_prompt = msg.prompt
        sampling_params_list = msg.sampling_params_list

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
            final_output_stage_ids={0},
            request_timestamp=parent_state.request_timestamp,
        )
        self.request_states[companion_id] = companion_state
        companion_state.stage_submit_ts[0] = _time.time()
        companion_replica_id = await self.stage_pools[0].submit_initial(
            companion_id,
            companion_state,
            companion_prompt,
            prompt_text=msg.companion_prompt_text,
            affinity_request_id=parent_id,
        )

        logger.debug(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            companion_replica_id,
        )

    async def _handle_abort(self, msg: AbortRequestMessage) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg.request_ids
        # _cleanup_request_ids is CFG-aware: it expands aborted parents to
        # their companions and fails a deferred parent whose companion is
        # aborted before its output arrived.
        await self._cleanup_request_ids(list(request_ids), abort=True)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _handle_interaction(self, msg: InteractionMessage) -> None:
        """Handle a midway interaction for an active streaming diffusion request."""
        stage_id = 0
        request_id = msg.request_id
        event_id = msg.interaction.get("event_id")
        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.info("[Orchestrator] Dropping interaction for inactive req %s", request_id)
            await self.output_async_queue.put(
                ErrorMessage(
                    error=f"No active request for interaction: {request_id}",
                    fatal=False,
                    request_id=request_id,
                    event_id=event_id,
                    stage_id=stage_id,
                )
            )
            return

        try:
            await self.stage_pools[stage_id].submit_interaction(request_id, msg.interaction)
        except Exception as exc:
            logger.info(
                "[Orchestrator] Failed interaction for req %s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            await self.output_async_queue.put(
                ErrorMessage(
                    error=f"Failed interaction for request {request_id}: {exc}",
                    fatal=False,
                    request_id=request_id,
                    event_id=event_id,
                    stage_id=stage_id,
                )
            )

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Forward abort requests to all stage pools."""
        if not request_ids:
            return
        pd_pair = getattr(self, "_pd_pair", None)
        aliases = getattr(self, "_pd_decode_request_aliases", {})
        request_states = getattr(self, "request_states", {})
        if pd_pair is not None:
            _, d_stage_id = pd_pair
            d_pool = self.stage_pools[d_stage_id]
            abort_physical = getattr(
                d_pool,
                "abort_engine_requests_for_binding",
                None,
            )
            if callable(abort_physical):
                for request_id in request_ids:
                    physical_ids = [
                        engine_request_id
                        for engine_request_id, logical_request_id in aliases.items()
                        if logical_request_id == request_id
                    ]
                    req_state = request_states.get(request_id)
                    if req_state is not None:
                        active_id = req_state.streaming.bridge_states.get("pd_decode_engine_request_id")
                        if isinstance(active_id, str) and active_id:
                            physical_ids.append(active_id)
                    await abort_physical(request_id, physical_ids)
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)
            pool.release_bindings(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        for pool in self.stage_pools:
            pool.release_bindings(request_ids)

    async def _handle_collective_rpc(self, msg: CollectiveRPCRequestMessage) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg.rpc_id
        method = msg.method
        timeout = msg.timeout
        args = tuple(msg.args)
        kwargs = dict(msg.kwargs or {})
        requested_stage_ids = msg.stage_ids

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if not (0 <= lid < self.num_stages):
                    logger.warning("[Orchestrator] collective_rpc: ignoring invalid stage_id %s", lid)
                    continue
                target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            for replica_id in pool.live_replica_ids():
                stage_result = await pool.collective_rpc(
                    replica_id=replica_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
                stage_ids.append(pool.stage_id)
                results.append(stage_result)

        await self.rpc_async_queue.put(
            CollectiveRPCResultMessage(
                rpc_id=rpc_id,
                method=method,
                stage_ids=stage_ids,
                results=results,
            )
        )

    def _schedule_background_collective_rpc(
        self,
        msg: CollectiveRPCRequestMessage,
    ) -> None:
        """Keep speculative preprocessing from blocking request ingress."""
        tasks = getattr(self, "_background_collective_rpc_tasks", None)
        if tasks is None:
            tasks = self._background_collective_rpc_tasks = set()
        task = asyncio.create_task(
            self._handle_collective_rpc(msg),
            name=f"orchestrator-rpc-{msg.method}-{msg.rpc_id}",
        )
        tasks.add(task)

        def _discard(done: asyncio.Task[None]) -> None:
            tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error(
                    "[Orchestrator] background collective_rpc(%s) failed: %s",
                    msg.method,
                    error,
                )

        task.add_done_callback(_discard)

    # ---- Orchestration loop ----

    def _sample_replica_metrics(self) -> dict[str, tuple[int, int]]:
        samples: dict[str, tuple[int, int]] = {}
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.live_replica_ids():
                key = replica_key(stage_id, replica_id)
                samples[key] = pool.replica_monitor_sample(replica_id)
        return samples

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _pd_prefill_output_worker(self) -> None:
        """Route P outputs in order without blocking other stage consumers."""
        queue = self._pd_prefill_output_queue
        if queue is None:
            return
        while not self._shutdown_event.is_set() or not queue.empty():
            try:
                stage_id, replica_id, outputs, enqueued_mono = await asyncio.wait_for(
                    queue.get(),
                    timeout=0.1,
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            route_started = _time.monotonic()
            try:
                await self._handle_processed_outputs(stage_id, replica_id, outputs)
            finally:
                queue.task_done()
            if _LOG_HANDOFF_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES):
                route_done = _time.monotonic()
                logger.info(
                    "[ORCH-P-ROUTE-DIAG] mono=%.6f reqs=%s queue_wait_ms=%.3f route_ms=%.3f qsize_after=%d",
                    route_done,
                    ",".join(str(getattr(output, "request_id", "?")) for output in outputs),
                    (route_started - enqueued_mono) * 1000.0,
                    (route_done - route_started) * 1000.0,
                    queue.qsize(),
                )

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id in pool.available_replica_ids():
                    if self._shutdown_event.is_set():
                        return

                    if pool.stage_type == "diffusion":
                        output = pool.poll_diffusion_output(replica_id)
                        if output is None:
                            continue

                        pool.record_output_timestamps([output])
                        await self._handle_processed_outputs(stage_id, replica_id, [output])
                        idle = False
                    else:
                        try:
                            output_step_start = _time.perf_counter()
                            # Each AsyncMPClient already has a persistent socket
                            # reader. Do not serialize a 1 ms timeout across
                            # every empty stage before consuming a ready stage.
                            raw_outputs = pool.poll_llm_raw_output_nowait(replica_id)
                            if raw_outputs is None:
                                continue
                            output_poll_done = _time.perf_counter()

                            if _LOG_HANDOFF_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES):
                                recv_wall = _time.time()
                                for diagnostic_output in raw_outputs.outputs:
                                    diagnostic_engine_req_id = getattr(
                                        diagnostic_output,
                                        "request_id",
                                        None,
                                    )
                                    diagnostic_req_id = self._pd_decode_request_aliases.get(
                                        diagnostic_engine_req_id,
                                        diagnostic_engine_req_id,
                                    )
                                    diagnostic_state = self.request_states.get(diagnostic_req_id)
                                    submit_wall = (
                                        diagnostic_state.stage_submit_ts.get(stage_id)
                                        if diagnostic_state is not None
                                        else None
                                    )
                                    logger.info(
                                        "[HANDOFF-DIAG] event=orchestrator-raw-recv stage=%s wall=%.6f "
                                        "req=%s since_submit_ms=%.3f",
                                        stage_id,
                                        recv_wall,
                                        diagnostic_req_id,
                                        (recv_wall - submit_wall) * 1000.0 if submit_wall is not None else -1.0,
                                    )

                            await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)
                            output_kv_done = _time.perf_counter()
                            for eco in raw_outputs.outputs:
                                engine_req_id = getattr(eco, "request_id", None)
                                logical_req_id = self._pd_decode_request_aliases.get(
                                    engine_req_id,
                                    engine_req_id,
                                )
                                req_state = self.request_states.get(logical_req_id)
                                if req_state is None:
                                    continue
                                if (
                                    self._pd_pair is not None
                                    and stage_id == self._pd_pair[0]
                                    and not self._is_duplex_session_request(req_state)
                                ):
                                    raw_mm = self._completion_multimodal_output(eco, None)
                                    if raw_mm:
                                        req_state.pd_prefill_multimodal_output = self._accumulate_pd_prefill_output(
                                            req_state.pd_prefill_multimodal_output,
                                            raw_mm,
                                        )
                                if not req_state.streaming.enabled:
                                    continue
                                req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
                                req_state.streaming.segment_token_ids = (
                                    self._coerce_int_list(getattr(eco, "new_token_ids", None))
                                    if req_state.streaming.segment_finished
                                    else []
                                )
                                if (
                                    self._pd_pair is not None
                                    and stage_id == self._pd_pair[1]
                                    and self._is_duplex_session_request(req_state)
                                ):
                                    bridge = req_state.streaming.bridge_states
                                    active_slot = bridge.get("pd_duplex_active_slot")
                                    prefill_stats = getattr(eco, "prefill_stats", None)
                                    if isinstance(active_slot, dict) and prefill_stats is not None:
                                        prompt_tokens = self._coerce_int(
                                            getattr(prefill_stats, "num_prompt_tokens", None)
                                        )
                                        local_cached_tokens = self._coerce_int(
                                            getattr(prefill_stats, "num_local_cached_tokens", None)
                                        )
                                        external_cached_tokens = self._coerce_int(
                                            getattr(prefill_stats, "num_external_cached_tokens", None)
                                        )
                                        computed_tokens = self._coerce_int(
                                            getattr(prefill_stats, "num_computed_tokens", None)
                                        )
                                        cached_tokens = self._coerce_int(
                                            getattr(prefill_stats, "num_cached_tokens", None)
                                        )
                                        counts = (
                                            prompt_tokens,
                                            local_cached_tokens,
                                            external_cached_tokens,
                                            computed_tokens,
                                            cached_tokens,
                                        )
                                        if (
                                            all(value is not None and value >= 0 for value in counts)
                                            and prompt_tokens is not None
                                            and local_cached_tokens is not None
                                            and external_cached_tokens is not None
                                            and computed_tokens is not None
                                            and cached_tokens is not None
                                            and local_cached_tokens + external_cached_tokens == cached_tokens
                                            and computed_tokens + cached_tokens == prompt_tokens
                                        ):
                                            active_slot.update(
                                                {
                                                    "prompt_tokens": prompt_tokens,
                                                    "local_cached_tokens": local_cached_tokens,
                                                    "external_cached_tokens": external_cached_tokens,
                                                    "computed_tokens": computed_tokens,
                                                }
                                            )
                                    if isinstance(active_slot, dict):
                                        self._update_native_pd_transfer_evidence(
                                            active_slot,
                                            getattr(eco, "kv_transfer_params", None),
                                        )
                                    segment_tokens = bridge.setdefault(
                                        "pd_duplex_decode_segment_token_ids",
                                        [],
                                    )
                                    if not isinstance(segment_tokens, list):
                                        segment_tokens = []
                                        bridge["pd_duplex_decode_segment_token_ids"] = segment_tokens
                                    segment_tokens.extend(self._coerce_int_list(getattr(eco, "new_token_ids", None)))
                                    decode_boundary = (
                                        req_state.streaming.segment_finished
                                        or getattr(eco, "finish_reason", None) is not None
                                    )
                                    if decode_boundary:
                                        # D is finite per slot, but the
                                        # application request remains a live
                                        # duplex session.  Normalize its
                                        # finite finish into the same segment
                                        # boundary used by a resumable
                                        # Thinker so generic routing does not
                                        # tear down a silent/listen slot.
                                        req_state.streaming.segment_finished = True
                                        if _LOG_HANDOFF_DIAG:
                                            diagnostic_mm = self._completion_multimodal_output(
                                                eco,
                                                None,
                                            )
                                            diagnostic_shapes = {
                                                str(key): tuple(value.shape)
                                                if isinstance(value, torch.Tensor)
                                                else type(value).__name__
                                                for key, value in diagnostic_mm.items()
                                            }
                                            logger.info(
                                                "[MINICPM-PD-D-OUTPUT] req=%s raw_mm=%s",
                                                req_state.request_id,
                                                diagnostic_shapes,
                                            )
                                        bridge["pd_duplex_feedback_token_ids"] = [
                                            *bridge.pop(
                                                "pd_duplex_prefill_sample_token_ids",
                                                [],
                                            ),
                                            *segment_tokens,
                                        ]
                                        if bridge.get("pd_duplex_terminal_replay", False):
                                            if (len(segment_tokens) != 1
                                                    or bridge["pd_duplex_feedback_token_ids"] != segment_tokens * 2):
                                                raise RuntimeError("Invalid native P terminal replay on D")
                                            bridge["pd_duplex_feedback_token_ids"] = list(segment_tokens)
                                        self._capture_native_pd_sampling_feedback(eco, req_state)
                                        segment_tokens.clear()
                                        active_slot = bridge.pop(
                                            "pd_duplex_active_slot",
                                            None,
                                        )
                                        if isinstance(active_slot, dict):
                                            done_epoch = _time.time()
                                            await self._emit_native_duplex_d_completion_witness(
                                                stage_id=stage_id,
                                                replica_id=replica_id,
                                                engine_request_id=str(engine_req_id),
                                                req_state=req_state,
                                                active_slot=active_slot,
                                                completed_epoch_s=done_epoch,
                                            )
                                            ready_epoch = float(
                                                active_slot.get(
                                                    "ready_epoch",
                                                    done_epoch,
                                                )
                                            )
                                            if _LOG_HANDOFF_DIAG:
                                                logger.info(
                                                    "[minicpm_pd_slot] req=%s seq=%s "
                                                    "ready_epoch=%.6f done_epoch=%.6f "
                                                    "e2e_ms=%.3f wait_previous_d_ms=%.3f",
                                                    req_state.request_id,
                                                    active_slot.get("seq"),
                                                    ready_epoch,
                                                    done_epoch,
                                                    (done_epoch - ready_epoch) * 1000.0,
                                                    float(
                                                        active_slot.get(
                                                            "wait_previous_d_ms",
                                                            0.0,
                                                        )
                                                    ),
                                                )
                                        decode_ready = bridge.get("pd_duplex_decode_ready")
                                        if isinstance(decode_ready, asyncio.Event):
                                            decode_ready.set()
                                raw_mm = self._completion_multimodal_output(eco, None)
                                req_state.streaming.segment_output_metadata = (
                                    dict(raw_mm)
                                    if req_state.streaming.segment_finished and isinstance(raw_mm, dict)
                                    else {}
                                )
                                req_state.streaming.new_prompt_len_snapshot = getattr(
                                    eco,
                                    "new_prompt_len_snapshot",
                                    None,
                                )
                                if req_state.streaming.enabled:
                                    await self._apply_raw_terminal_stage_finish(stage_id, eco, req_state)
                                if (
                                    req_state.streaming.segment_finished
                                    and self._pd_pair is not None
                                    and stage_id == self._pd_pair[0]
                                    and self._is_duplex_session_request(req_state)
                                ):
                                    self._schedule_native_duplex_pd_prefill_raw(
                                        stage_id,
                                        replica_id,
                                        eco,
                                        req_state,
                                    )
                            # OmniSchedulerMixin.make_stats() already throttles
                            # per-scheduler at 1 Hz, so raw_outputs.scheduler_stats
                            # being non-None means this replica passed its own gate.
                            # A second global throttle here would drop stats for
                            # other (stage, replica) pairs in the same 1s window.
                            record_stats = self._stat_logger is not None and raw_outputs.scheduler_stats is not None
                            iteration_stats = IterationStats() if record_stats else None
                            raw_output = await pool.process_llm_raw_outputs(
                                replica_id,
                                raw_outputs,
                                iteration_stats=iteration_stats,
                            )
                            output_process_done = _time.perf_counter()
                            if record_stats:
                                self._stat_logger.record(
                                    raw_outputs.scheduler_stats,
                                    iteration_stats,
                                    engine_idx=self._stage_replica_to_engine_idx[(stage_id, replica_id)],
                                )
                        except asyncio.CancelledError:
                            raise
                        except EngineDeadError as e:
                            logger.error(
                                "[Orchestrator] Stage-%s replica-%s is dead: %s",
                                stage_id,
                                replica_id,
                                e,
                            )
                            affected_request_ids = pool.mark_replica_unavailable(replica_id)
                            closed_sessions = (
                                self.duplex_control_plane.close_sessions_for_request_ids(
                                    affected_request_ids,
                                    abort=False,
                                )
                                if self.duplex_control_plane is not None
                                else {}
                            )
                            for session_id, stale_request_ids in closed_sessions.items():
                                affected_request_ids.extend(stale_request_ids)
                                logger.warning(
                                    "[Orchestrator] closed duplex session %s after stage-%s replica-%s died; "
                                    "stale_request_ids=%s",
                                    session_id,
                                    stage_id,
                                    replica_id,
                                    stale_request_ids,
                                )
                            affected_request_ids = list(dict.fromkeys(affected_request_ids))
                            if pool.available_replica_ids():
                                for req_id in affected_request_ids:
                                    await self.output_async_queue.put(
                                        ErrorMessage(
                                            error=str(e),
                                            fatal=False,
                                            request_id=req_id,
                                            stage_id=stage_id,
                                        )
                                    )
                                await self._cleanup_request_ids(
                                    affected_request_ids,
                                    close_duplex_sessions=True,
                                )
                                continue

                            self._fatal_error = str(e)
                            self._fatal_error_stage_id = stage_id
                            for req_id in affected_request_ids:
                                await self.output_async_queue.put(
                                    ErrorMessage(
                                        error=str(e),
                                        fatal=True,
                                        request_id=req_id,
                                        stage_id=stage_id,
                                    )
                                )
                            await self._cleanup_request_ids(
                                affected_request_ids,
                                close_duplex_sessions=True,
                            )
                            self._shutdown_event.set()
                            raise
                        except Exception:
                            if self._shutdown_event.is_set():
                                return
                            logger.exception(
                                "[Orchestrator] Stage-%s replica-%s processing failed",
                                stage_id,
                                replica_id,
                            )
                            raise

                        pd_prefill_stage = self._pd_pair[0] if self._pd_pair is not None else None
                        if stage_id == pd_prefill_stage and self._pd_prefill_output_queue is not None:
                            self._pd_prefill_output_queue.put_nowait(
                                (stage_id, replica_id, raw_output, _time.monotonic())
                            )
                        else:
                            await self._handle_processed_outputs(stage_id, replica_id, raw_output)
                        if _LOG_HANDOFF_DIAG and (_DIAG_STAGES is None or str(stage_id) in _DIAG_STAGES):
                            output_route_done = _time.perf_counter()
                            request_ids = ",".join(
                                str(getattr(output, "request_id", "?")) for output in raw_outputs.outputs
                            )
                            queue_size, _ = pool.replica_monitor_sample(replica_id)
                            logger.info(
                                "[ORCH-OUTPUT-DIAG] mono=%.6f stage=%s reqs=%s poll_ms=%.3f "
                                "kv_ms=%.3f process_ms=%.3f route_ms=%.3f total_ms=%.3f "
                                "qsize_after=%d",
                                _time.monotonic(),
                                stage_id,
                                request_ids,
                                (output_poll_done - output_step_start) * 1000.0,
                                (output_kv_done - output_poll_done) * 1000.0,
                                (output_process_done - output_kv_done) * 1000.0,
                                (output_route_done - output_process_done) * 1000.0,
                                (output_route_done - output_step_start) * 1000.0,
                                queue_size,
                            )
                        idle = False

            self._orch_monitor.note_loop(idle=idle)
            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _handle_processed_outputs(self, stage_id: int, replica_id: int, outputs: list[Any]) -> None:
        """Route processed stage outputs produced by one stage poll."""
        pool = self.stage_pools[stage_id]
        for output in outputs:
            req_state = self.request_states.get(output.request_id)
            if req_state is None:
                logger.warning(
                    "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                    output.request_id,
                    stage_id,
                    list(self.request_states.keys()),
                )
                continue

            if getattr(output, "error", None) is not None:
                await self._handle_stage_error(stage_id, output)
                continue

            stage_metrics = None
            segment_finished = req_state.streaming.enabled and req_state.streaming.segment_finished
            if output.finished or segment_finished:
                stage_metrics = pool.build_stage_metrics(
                    [output],
                    submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                    request_timestamp=req_state.request_timestamp,
                    replica_id=replica_id,
                    sampling_params=req_state.sampling_params_list[stage_id],
                )
                stage_metrics.pipeline_timings = dict(req_state.pipeline_timings)

            await self._route_output(stage_id, replica_id, output, req_state, stage_metrics)

    @classmethod
    def _update_native_pd_transfer_evidence(
        cls,
        active_slot: dict[str, Any],
        kv_transfer_params: Any,
    ) -> None:
        """Copy only connector-proven request scalars into the active slot."""
        if not isinstance(kv_transfer_params, dict):
            return
        for name in (
            "kv_transfer_selected_blocks",
            "kv_transfer_selected_tokens",
            "kv_transfer_selected_bytes",
        ):
            value = cls._coerce_int(kv_transfer_params.get(name))
            if value is not None and value >= 0:
                active_slot[name] = value

        raw_ms = kv_transfer_params.get("kv_transfer_write_submit_to_d_ready_ms")
        if isinstance(raw_ms, int | float) and not isinstance(raw_ms, bool):
            value_ms = float(raw_ms)
            if value_ms >= 0.0:
                active_slot["kv_transfer_write_submit_to_d_ready_ms"] = value_ms

    async def _emit_native_duplex_d_completion_witness(
        self,
        *,
        stage_id: int,
        replica_id: int,
        engine_request_id: str,
        req_state: OrchestratorRequestState,
        active_slot: dict[str, Any],
        completed_epoch_s: float,
    ) -> None:
        """Emit exactly one observer record at a finite native-D boundary."""
        sequence = self._coerce_int(active_slot.get("seq"))
        if sequence is None:
            _, separator, raw_sequence = engine_request_id.rpartition("-")
            try:
                sequence = int(raw_sequence, 16) if separator and len(raw_sequence) == 8 else None
            except ValueError:
                sequence = None
        if sequence is None:
            logger.warning(
                "[Orchestrator][PD] cannot identify physical D completion req=%s",
                engine_request_id,
            )
            return

        bridge = req_state.streaming.bridge_states
        last_sequence = self._coerce_int(bridge.get("pd_duplex_last_completion_witness_sequence"))
        if last_sequence is not None and sequence <= last_sequence:
            return

        submit_epoch_s = float(req_state.stage_submit_ts.get(stage_id, completed_epoch_s))
        raw_input_unit_index = self._coerce_int(active_slot.get("input_unit_index"))
        source = active_slot.get("source")
        if source not in ("real_input", "auto_continuation"):
            source = "real_input"
        prompt_tokens = self._coerce_int(active_slot.get("prompt_tokens"))
        local_cached_tokens = self._coerce_int(active_slot.get("local_cached_tokens"))
        external_cached_tokens = self._coerce_int(active_slot.get("external_cached_tokens"))
        computed_tokens = self._coerce_int(active_slot.get("computed_tokens"))
        kv_transfer_selected_blocks = self._coerce_int(active_slot.get("kv_transfer_selected_blocks"))
        kv_transfer_selected_tokens = self._coerce_int(active_slot.get("kv_transfer_selected_tokens"))
        kv_transfer_selected_bytes = self._coerce_int(active_slot.get("kv_transfer_selected_bytes"))
        raw_write_to_ready_ms = active_slot.get("kv_transfer_write_submit_to_d_ready_ms")
        prompt_tokens = prompt_tokens if prompt_tokens is not None and prompt_tokens >= 0 else -1
        local_cached_tokens = (
            local_cached_tokens if local_cached_tokens is not None and local_cached_tokens >= 0 else -1
        )
        external_cached_tokens = (
            external_cached_tokens if external_cached_tokens is not None and external_cached_tokens >= 0 else -1
        )
        computed_tokens = computed_tokens if computed_tokens is not None and computed_tokens >= 0 else -1
        kv_transfer_selected_blocks = (
            kv_transfer_selected_blocks
            if kv_transfer_selected_blocks is not None and kv_transfer_selected_blocks >= 0
            else -1
        )
        kv_transfer_selected_tokens = (
            kv_transfer_selected_tokens
            if kv_transfer_selected_tokens is not None and kv_transfer_selected_tokens >= 0
            else -1
        )
        kv_transfer_selected_bytes = (
            kv_transfer_selected_bytes
            if kv_transfer_selected_bytes is not None and kv_transfer_selected_bytes >= 0
            else -1
        )
        kv_transfer_write_submit_to_d_ready_ms = (
            float(raw_write_to_ready_ms)
            if isinstance(raw_write_to_ready_ms, int | float)
            and not isinstance(raw_write_to_ready_ms, bool)
            and float(raw_write_to_ready_ms) >= 0.0
            else -1.0
        )
        cached_tokens = (
            local_cached_tokens + external_cached_tokens
            if local_cached_tokens >= 0 and external_cached_tokens >= 0
            else -1
        )
        await self.output_async_queue.put(
            PhysicalDCompletionWitnessMessage(
                request_id=req_state.request_id,
                stage_id=stage_id,
                replica_id=replica_id,
                engine_request_id=engine_request_id,
                physical_sequence=sequence,
                input_unit_index=(raw_input_unit_index if raw_input_unit_index is not None else sequence),
                source=source,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                local_cached_tokens=local_cached_tokens,
                external_cached_tokens=external_cached_tokens,
                computed_tokens=computed_tokens,
                # Core batch identity is not exposed at this raw boundary.
                batch_id=0,
                submit_epoch_s=submit_epoch_s,
                completed_epoch_s=completed_epoch_s,
                service_ms=max(0.0, (completed_epoch_s - submit_epoch_s) * 1000.0),
                input_video_frames=max(
                    0,
                    self._coerce_int(active_slot.get("input_video_frames")) or 0,
                ),
                arrival_video_frames=max(
                    0,
                    self._coerce_int(active_slot.get("arrival_video_frames")) or 0,
                ),
                vision_fallback_frames=max(
                    0,
                    self._coerce_int(active_slot.get("vision_fallback_frames")) or 0,
                ),
                arrival_audio_units=max(
                    0,
                    self._coerce_int(active_slot.get("arrival_audio_units")) or 0,
                ),
                audio_fallback_units=max(
                    0,
                    self._coerce_int(active_slot.get("audio_fallback_units")) or 0,
                ),
                kv_transfer_selected_blocks=kv_transfer_selected_blocks,
                kv_transfer_selected_tokens=kv_transfer_selected_tokens,
                kv_transfer_selected_bytes=kv_transfer_selected_bytes,
                kv_transfer_write_submit_to_d_ready_ms=(kv_transfer_write_submit_to_d_ready_ms),
            )
        )
        bridge["pd_duplex_last_completion_witness_sequence"] = sequence

    async def _handle_stage_error(self, stage_id: int, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        if self._cfg_tracker.is_companion(output.request_id):
            parent_id = self._cfg_tracker.get_parent_id(output.request_id) or output.request_id
        else:
            parent_id = output.request_id
        await self.output_async_queue.put(
            ErrorMessage(
                request_id=parent_id,
                stage_id=stage_id,
                error=output.error,
                status_code=getattr(output, "error_status_code", None),
                error_type=getattr(output, "error_type", None),
            )
        )
        await self._cleanup_request_ids(
            [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)],
            abort=True,
            close_duplex_sessions=True,
        )

    # ---- Shared helpers ----

    async def _cleanup_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        close_duplex_sessions: bool = False,
    ) -> None:
        """Release pool bindings and logical request state for the given ids.

        CFG-aware: cleaning a parent releases its tracker state and pulls its
        companions into the batch; cleaning a companion whose parent is NOT in
        the batch and whose output never arrived fails that parent (its bundle
        can never complete). Every teardown path (stage error, abort, replica
        loss, membership unregister) funnels through here, so tracker state
        cannot outlive its requests.
        """
        if not request_ids:
            return

        cleanup_ids = list(dict.fromkeys(request_ids))
        batch = set(cleanup_ids)
        orphaned_parents: dict[str, str] = {}
        for rid in cleanup_ids:
            pid = self._cfg_tracker.get_parent_id(rid)
            if pid is not None and pid not in batch and not self._cfg_tracker.is_companion_done(rid):
                orphaned_parents.setdefault(pid, rid)
        for pid, cid in orphaned_parents.items():
            deferred = self._cfg_tracker.pop_pending_parent(pid)
            await self.output_async_queue.put(
                ErrorMessage(
                    request_id=pid,
                    stage_id=deferred["stage_id"] if deferred is not None else None,
                    error=f"CFG companion {cid} was aborted or lost before its outputs arrived",
                )
            )
            batch.add(pid)
            cleanup_ids.append(pid)
        for rid in list(cleanup_ids):
            for cid in self._cfg_tracker.cleanup_parent(rid):
                if cid not in batch:
                    batch.add(cid)
                    cleanup_ids.append(cid)
        closing_session_ids: list[str] = []
        if close_duplex_sessions and self.duplex_control_plane is not None:
            closed_sessions = self.duplex_control_plane.close_sessions_for_request_ids(
                cleanup_ids,
                abort=abort,
                cleanup_in_progress=True,
            )
            closing_session_ids.extend(closed_sessions)
            for session_id, stale_request_ids in closed_sessions.items():
                logger.info(
                    "[Orchestrator] closed duplex session %s while cleaning failed request ids %s",
                    session_id,
                    stale_request_ids,
                )
                cleanup_ids.extend(stale_request_ids)
            cleanup_ids = list(dict.fromkeys(cleanup_ids))

        cleanup_reason = (
            "duplex request aborted"
            if abort
            else "duplex request closed"
            if close_duplex_sessions
            else "duplex request cleaned up"
        )
        for request_id in cleanup_ids:
            req_state = self.request_states.get(request_id)
            if req_state is not None and self._is_duplex_session_request(req_state):
                self._signal_native_duplex_pd_route_error(
                    req_state,
                    cleanup_reason,
                )
        await self._cancel_pd_raw_route_tasks(
            cleanup_ids,
            reason=cleanup_reason,
        )

        try:
            if abort:
                await self._abort_request_ids(cleanup_ids)
            self._release_request_bindings(cleanup_ids)
            for request_id in cleanup_ids:
                cache_sync_task = getattr(self, "_pd_cache_sync_tasks", {}).get(request_id)
                if cache_sync_task is not None and cache_sync_task is not asyncio.current_task():
                    cache_sync_task.cancel()
                req_state = self.request_states.get(request_id)
                native_cache_sync_task = req_state.pd_early_cache_sync_task if req_state is not None else None
                if (
                    native_cache_sync_task is not None
                    and native_cache_sync_task is not asyncio.current_task()
                    and not native_cache_sync_task.done()
                ):
                    native_cache_sync_task.cancel()
                self._pd_kv_params.pop(request_id, None)
                for engine_req_id, logical_req_id in list(self._pd_decode_request_aliases.items()):
                    if logical_req_id == request_id:
                        self._pd_decode_request_aliases.pop(engine_req_id, None)
                self._duplexomni_pending_talker.pop(request_id, None)
                self._duplexomni_pipeline.release_request(request_id)
                req_state = self.request_states.pop(request_id, None)
                if req_state is not None and req_state.running_counter_registered and self._running_counter is not None:
                    self._running_counter.decrement()
                    req_state.running_counter_registered = False
        except BaseException:
            if closing_session_ids and self.duplex_control_plane is not None:
                self.duplex_control_plane.defer_request_cleanups(closing_session_ids)
            raise
        if closing_session_ids and self.duplex_control_plane is not None:
            self.duplex_control_plane.finalize_closed_sessions(closing_session_ids)

    async def _apply_raw_terminal_stage_finish(
        self,
        stage_id: int,
        eco: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Record session-level finish markers dropped by the streaming output processor.

        Streaming segment stops set ``is_segment_finished=True`` and are handled
        via processed outputs. Session termination (e.g. ``finish_requests`` after
        ``resumable=False``) emits a terminal ``finish_reason`` with
        ``is_segment_finished=False``, but vLLM's output processor may remove the
        request state before that EngineCoreOutput is processed.

        Only update ``finished_final_output_stage_ids`` here. Request cleanup stays
        in ``_route_output`` so downstream async-chunk stages can still deliver
        outputs after stage-0 session end.
        """
        if getattr(eco, "finish_reason", None) is None:
            return
        if getattr(eco, "is_segment_finished", False):
            return

        final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
        if stage_id not in final_output_stage_ids:
            return
        req_state.finished_final_output_stage_ids.add(stage_id)

    def _maybe_clone_diffusion_params_for_cfg(self, request_id: str, params: Any) -> Any:
        """Attach CFG companion ids to diffusion sampling params when needed."""
        companion_request_ids = self._cfg_tracker.get_companion_request_ids(request_id)
        if not companion_request_ids:
            return params

        import copy

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not isinstance(params, OmniDiffusionSamplingParams):
            return params

        params = copy.deepcopy(params)
        params.cfg_kv_request_ids = companion_request_ids
        return params

    def _duplex_session_for_req_state(self, req_state: OrchestratorRequestState) -> DuplexSessionRuntimeState | None:
        if self.duplex_control_plane is None:
            return None
        return self.duplex_control_plane.session_for_identity(req_state.duplex_identity)

    def _record_duplex_stage_submission(
        self,
        stage_id: int,
        request_id: str,
        replica_id: int,
        req_state: OrchestratorRequestState,
    ) -> None:
        del replica_id
        identity = req_state.duplex_identity
        session = self._duplex_session_for_req_state(req_state)
        if identity is None or session is None:
            return
        req_state.duplex_stage_fences[stage_id] = identity.fence
        session.bind_stage_request(stage_id, request_id, fence=identity.fence)
        req_state.stage_submit_ts[stage_id] = _time.time()
        self._register_running_request(req_state)

    def _register_running_request(self, req_state: OrchestratorRequestState) -> None:
        if req_state.running_counter_registered or self._running_counter is None:
            return
        self._running_counter.increment()
        req_state.running_counter_registered = True

    def _duplexomni_thinker_output_stage(self) -> int:
        """Return the Thinker stage whose completed output feeds Talker."""
        return self._pd_pair[1] if self._pd_pair is not None else 0

    @staticmethod
    def _duplexomni_output_codec(output: Any) -> tuple[Any, bool]:
        completions = getattr(output, "outputs", None)
        completion = completions[0] if isinstance(completions, list) and completions else None
        metadata = Orchestrator._completion_multimodal_output(output, completion)
        codes = metadata.get("codec_codes")
        if isinstance(codes, list) and len(codes) == 1:
            codes = codes[0]
        valid = metadata.get("duplexomni_valid_turn", False)
        if isinstance(valid, list) and valid:
            valid = valid[-1]
        if hasattr(valid, "item"):
            valid = valid.item()
        return codes, bool(valid)

    def _prepare_duplexomni_talker_prompt(self, req_state: OrchestratorRequestState) -> None:
        identity = req_state.duplexomni_pipeline_identity
        if identity is None:
            return
        history_indices, codec_history = self._duplexomni_pipeline.history_for(identity)
        req_state.prompt = inject_codec_history(
            req_state.prompt,
            history_indices=history_indices,
            codec_history=codec_history,
        )
        prompt_item = req_state.prompt[0] if isinstance(req_state.prompt, list) else req_state.prompt
        logger.info(
            "[DuplexOmni] prepared Talker req=%s slot=%d prompt_tokens=%d history_turns=%d",
            req_state.request_id,
            identity.slot,
            len(prompt_item.get("prompt_token_ids", [])) if isinstance(prompt_item, dict) else -1,
            len(history_indices),
        )

    async def _release_duplexomni_talker_successor(
        self,
        identity: DuplexOmniPipelineIdentity,
    ) -> None:
        successor_id = self._duplexomni_pipeline.successor_request_id(identity)
        if successor_id is None:
            return
        pending = self._duplexomni_pending_talker.pop(successor_id, None)
        if pending is None:
            return
        successor_state = self.request_states.get(successor_id)
        if successor_state is None:
            return
        self._prepare_duplexomni_talker_prompt(successor_state)
        await self._forward_to_next_stage(
            successor_id,
            self._duplexomni_thinker_output_stage(),
            pending.output,
            successor_state,
            src_replica_id=pending.replica_id,
            is_streaming_session=pending.is_streaming_session,
            is_final_update=pending.is_final_update,
        )

    async def _complete_duplexomni_pipeline_slot(
        self,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        identity = req_state.duplexomni_pipeline_identity
        if identity is None:
            return
        codes, valid_turn = self._duplexomni_output_codec(output)
        if valid_turn and codes is None:
            raise RuntimeError("DuplexOmni pipeline received a valid Talker turn without codec history")
        self._duplexomni_pipeline.complete(
            identity,
            codec_codes=codes,
            valid_turn=valid_turn,
        )
        await self._release_duplexomni_talker_successor(identity)
        self._duplexomni_pipeline.close_if_final(identity)

    async def _forward_or_defer_duplexomni_talker(
        self,
        req_id: str,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        replica_id: int,
        is_streaming_session: bool,
        is_final_update: bool,
    ) -> None:
        identity = req_state.duplexomni_pipeline_identity
        if identity is None:
            await self._forward_to_next_stage(
                req_id,
                self._duplexomni_thinker_output_stage(),
                output,
                req_state,
                src_replica_id=replica_id,
                is_streaming_session=is_streaming_session,
                is_final_update=is_final_update,
            )
            return
        if not self._duplexomni_pipeline.talker_ready(identity):
            self._duplexomni_pending_talker[req_id] = _DuplexOmniPendingTalker(
                output=output,
                replica_id=replica_id,
                is_streaming_session=is_streaming_session,
                is_final_update=is_final_update,
            )
            return
        self._prepare_duplexomni_talker_prompt(req_state)
        await self._forward_to_next_stage(
            req_id,
            self._duplexomni_thinker_output_stage(),
            output,
            req_state,
            src_replica_id=replica_id,
            is_streaming_session=is_streaming_session,
            is_final_update=is_final_update,
        )

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)
        if (
            finished
            and self._pd_pair is not None
            and stage_id == self._pd_pair[1]
            and self._is_duplex_session_request(req_state)
        ):
            self._ensure_native_duplex_pd_talker_metadata(output, req_state)
        # CFG companion: stash output so parent can bundle [parent, *companions]
        # into source_outputs for the bridge (e.g. thinker2imagegen).
        if finished and self._cfg_tracker.is_companion(req_id):
            self._cfg_tracker.set_companion_output(req_id, output)
            await self._handle_cfg_companion_ready(req_id)
            await self._cleanup_request_ids([req_id])
            return

        if (
            finished
            and req_state.pd_decode_cache_sync_pending
            and self._pd_pair is not None
            and stage_id == self._pd_pair[1]
        ):
            # Compatibility fallback for an older D backend that still emits
            # a finite inference output. The direct cache-only path completes
            # and acknowledges inside _forward_to_next_stage instead.
            req_state.pd_decode_cache_sync_pending = False
            elapsed_ms = max(
                0.0,
                (_time.time() - req_state.stage_submit_ts.get(stage_id, _time.time())) * 1000.0,
            )
            logger.info(
                "[Orchestrator][PD cache-sync] completed req=%s lineage=%s revision=%d d_ms=%.3f",
                req_id,
                req_state.pd_prefill_lineage_id,
                req_state.pd_prefill_revision,
                elapsed_ms,
            )
            if req_state.pd_prefill_ready_emitted:
                await self._cleanup_request_ids([req_id])
                return
            final_stage_id = req_state.final_stage_id
            final_pool = self.stage_pools[final_stage_id]
            terminal_output = _build_terminal_empty_output(
                req_id,
                final_output_type=getattr(final_pool.stage_client, "final_output_type", None),
                audio_sample_rate=final_pool._infer_audio_sample_rate(),
            )
            terminal_ts = _time.time()
            req_state.stage_submit_ts[final_stage_id] = terminal_ts
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=final_stage_id,
                    replica_id=replica_id,
                    engine_outputs=terminal_output,
                    metrics=None,
                    finished=True,
                    stage_submit_ts=terminal_ts,
                )
            )
            await self._cleanup_request_ids([req_id])
            return

        request_finished = False
        if (
            finished
            and self.stage_pools[stage_id].final_output
            and not (req_state.streaming.enabled and req_state.streaming.segment_finished)
        ):
            req_state.finished_final_output_stage_ids.add(stage_id)
            final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
            request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)
        # Duplex Thinker segment boundaries are not client-visible outputs:
        # direct decisions are emitted by the model runtime extension below,
        # while spoken content flows through the next stage. Forwarding
        # the raw Thinker output as well injects one cumulative-text,
        # no-audio message per unit that every downstream consumer must
        # filter out again (the official implementation returns exactly one
        # result per audio chunk).
        is_duplex_thinker_segment = (
            stage_id == self._duplexomni_thinker_output_stage()
            and self._is_duplex_session_request(req_state)
            and req_state.streaming.segment_finished
        )
        if self.stage_pools[stage_id].final_output and not is_duplex_thinker_segment:
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    engine_outputs=output,
                    metrics=stage_metrics,
                    finished=(
                        request_finished
                        or (self._is_duplex_session_request(req_state) and req_state.streaming.segment_finished)
                    ),
                    stage_submit_ts=submit_ts,
                )
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                StageMetricsMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    metrics=stage_metrics,
                    stage_submit_ts=submit_ts,
                )
            )

        pd_prefill_boundary = finished or (req_state.streaming.enabled and req_state.streaming.segment_finished)
        bridge = req_state.streaming.bridge_states
        current_d_request_id = bridge.get("pd_decode_engine_request_id")
        native_raw_route_owned = bool(
            bridge.get("pd_duplex_prefill_raw_routed", False)
            or (
                isinstance(current_d_request_id, str)
                and current_d_request_id
                and current_d_request_id
                in (
                    bridge.get("pd_duplex_prefill_raw_scheduled_id"),
                    bridge.get("pd_duplex_prefill_raw_routed_id"),
                )
            )
        )
        if (
            self._pd_pair is not None
            and stage_id == self._pd_pair[0]
            and self._is_duplex_session_request(req_state)
            and native_raw_route_owned
        ):
            # Native duplex P segments are dispatched from their raw boundary.
            # A backend that also materializes a processed FINAL_ONLY object
            # must not submit the same finite D request twice.
            return
        if self._pd_pair is not None and pd_prefill_boundary and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
                if self._is_duplex_session_request(req_state):
                    self._prepare_native_duplex_pd_decode(output, req_state)
            # Raw EngineCore outputs are accumulated above because a chunked
            # prefill exposes only one hidden-state slice per engine step.  A
            # processed-only backend may not expose those raw slices, so keep
            # this final-output fallback without duplicating an accumulated
            # snapshot.
            is_native_duplex_pd = self._is_duplex_session_request(req_state)
            if not is_native_duplex_pd and req_state.pd_prefill_multimodal_output is None:
                processed_mm = self._completion_multimodal_output(output, None)
                if processed_mm:
                    req_state.pd_prefill_multimodal_output = self._accumulate_pd_prefill_output(
                        None,
                        processed_mm,
                    )
            # MiniCPM's native Talker consumes only D's generated-segment
            # hidden rows; unlike Qwen's finite-request bridge it does not
            # need a full P-side prompt-hidden snapshot.  Avoid retaining and
            # copying that O(context) tensor on every 1 s duplex unit.
            snapshot_cached = False if is_native_duplex_pd else await self._materialize_pd_prefill_snapshot(req_state)

            # A cache-population request must stop at P.  In a non-split
            # pipeline the Thinker itself is the text output stage; after a
            # split, text belongs to D, so ordinary modality routing would
            # accidentally submit this silent warm-up to D.
            prompt = req_state.prompt
            prefill_only = isinstance(prompt, dict) and prompt.get("prefill_only") is True
            if prefill_only:
                # Do not acknowledge a lineage revision that the orchestrator
                # cannot serve to the next finite request.  The application
                # advances its revision only after this request completes, so
                # a request-scoped error preserves the last valid parent.
                if req_state.pd_prefill_lineage_id and not snapshot_cached:
                    error = "Thinker-P finished a prefill-only request without a usable lineage snapshot"
                    logger.error(
                        "[Orchestrator][PD snapshot] req=%s lineage=%s revision=%d: %s",
                        req_id,
                        req_state.pd_prefill_lineage_id,
                        req_state.pd_prefill_revision,
                        error,
                    )
                    await self.output_async_queue.put(
                        ErrorMessage(
                            request_id=req_id,
                            stage_id=stage_id,
                            error=error,
                            error_type="PDSnapshotError",
                        )
                    )
                    await self._cleanup_request_ids([req_id])
                    return
                req_state.pd_decode_cache_sync_pending = True
                logger.info(
                    "[Orchestrator][PD cache-sync] submit req=%s lineage=%s "
                    "parent_revision=%d revision=%d prompt_tokens=%d",
                    req_id,
                    req_state.pd_prefill_lineage_id,
                    int(getattr(prompt, "kv_lineage_parent_revision", 0))
                    if not isinstance(prompt, dict)
                    else int(prompt.get("kv_lineage_parent_revision", 0)),
                    req_state.pd_prefill_revision,
                    len(req_state.pd_prefill_prompt_token_ids),
                )
                early_task = req_state.pd_early_cache_sync_task
                early_failed = bool(
                    early_task is not None
                    and early_task.done()
                    and not (
                        req_state.pd_early_cache_sync_result and req_state.pd_early_cache_sync_result.get("ok") is True
                    )
                )
                if early_task is None or early_failed:
                    # Compatibility/failure fallback when a static P endpoint
                    # was unavailable or an eager registration failed.
                    self._schedule_pd_cache_sync(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        src_replica_id=replica_id,
                    )
                # P has finished and its snapshot/prefix blocks are now a
                # valid parent for the next request in this lineage.  Do not
                # keep the application session blocked on D's disposable
                # cache import: a later foreground request can send the
                # cumulative suffix from whatever prefix D still owns.
                final_pool = self.stage_pools[req_state.final_stage_id]
                terminal_output = _build_terminal_empty_output(
                    req_id,
                    final_output_type=getattr(final_pool.stage_client, "final_output_type", None),
                    audio_sample_rate=final_pool._infer_audio_sample_rate(),
                )
                req_state.pd_prefill_ready_emitted = True
                await self.output_async_queue.put(
                    OutputMessage(
                        request_id=req_id,
                        stage_id=stage_id,
                        replica_id=replica_id,
                        engine_outputs=terminal_output,
                        metrics=stage_metrics,
                        finished=True,
                        stage_submit_ts=submit_ts,
                    )
                )
                if (
                    early_task is not None
                    and early_task.done()
                    and not early_failed
                    and self.request_states.get(req_id) is req_state
                ):
                    await self._cleanup_request_ids([req_id])
                return

        # P samples the first native token; D continues it (or acknowledges a
        # P terminator). Only the assembled D boundary is client-visible.
        thinker_output_stage = self._duplexomni_thinker_output_stage()
        duplex_output_decision = (
            self._duplex_output_decision(stage_id, output, req_state) if stage_id == thinker_output_stage else None
        )
        if (
            duplex_output_decision is None
            and _MINICPMO_PD_ONLY_DIAGNOSTIC
            and self._pd_pair is not None
            and stage_id == self._pd_pair[1]
            and self._is_duplex_session_request(req_state)
            and req_state.streaming.segment_finished
        ):
            # Diagnostic control: preserve the complete finite D request and
            # its feedback into the next P unit, but terminate the stage graph
            # here. This removes Talker/Code2Wav work (and therefore their
            # contention with an auxiliary-GPU Encoder) without changing the
            # measured P/D workload.
            from vllm_omni.experimental.fullduplex.engine.contracts import (
                DuplexOutputAction,
                DuplexOutputDecision,
            )

            duplex_output_decision = DuplexOutputDecision(
                action=DuplexOutputAction.DIRECT_RESPONSE,
                metadata={
                    "duplex_direct_response": True,
                    "duplex_pd_only_diagnostic": True,
                },
                final_output_type="text",
            )
        if duplex_output_decision is not None:
            await self._emit_duplex_direct_output(
                stage_id,
                req_id,
                output,
                duplex_output_decision,
                stage_metrics,
                submit_ts,
            )
            return

        if finished and stage_id == req_state.final_stage_id and req_state.duplexomni_pipeline_identity is not None:
            await self._complete_duplexomni_pipeline_slot(output, req_state)

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and (not self.async_chunk or not self._stage_receives_async_chunks(stage_id + 1))
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                stage_params = req_state.sampling_params_list[stage_id]
                native_duplex_pd_decode_segment = bool(
                    self._pd_pair is not None
                    and stage_id == self._pd_pair[1]
                    and self._is_duplex_session_request(req_state)
                )
                final_only_finished = (
                    req_state.streaming.enabled
                    and finished
                    and getattr(stage_params, "output_kind", None) == RequestOutputKind.FINAL_ONLY
                    # D is intentionally finite per model unit, but that
                    # physical finish is only a session segment boundary.
                    # Keep the Talker request resumable so a later spoken
                    # unit queues behind the current audio instead of racing
                    # a duplicate finite request under the session id.
                    and not native_duplex_pd_decode_segment
                )
                if (
                    stage_id == self._duplexomni_thinker_output_stage()
                    and req_state.duplexomni_pipeline_identity is not None
                ):
                    await self._forward_or_defer_duplexomni_talker(
                        req_id,
                        output,
                        req_state,
                        replica_id=replica_id,
                        is_streaming_session=req_state.streaming.enabled,
                        is_final_update=final_only_finished,
                    )
                else:
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        src_replica_id=replica_id,
                        is_streaming_session=req_state.streaming.enabled,
                        is_final_update=final_only_finished,
                    )
                if (
                    req_state.streaming.enabled
                    and finished
                    and not final_only_finished
                    and not self._is_duplex_session_request(req_state)
                ):
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        src_replica_id=replica_id,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if request_finished and not self._is_duplex_session_request(req_state):
            # The terminal stage can finish before an async-chunk upstream
            # stage has drained every already-produced output.  Merely
            # dropping request state here leaves that upstream request live:
            # its late chunks consume compute, race connector cleanup, and are
            # then discarded as outputs for an unknown request.  Once every
            # client-visible final-output stage has finished, none of that
            # residual work can contribute to the response, so stop it before
            # releasing bindings and connector/request state.
            await self._cleanup_request_ids(
                [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                abort=True,
            )

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    def _stage_receives_async_chunks(self, stage_id: int) -> bool:
        """Whether a stage's connector supplies its runtime inputs."""
        pool = self.stage_pools[stage_id]
        model_config = getattr(pool.stage_vllm_config, "model_config", None)
        return stage_receives_chunks(model_config)

    def _get_stage_input_processor(self, stage_id: int) -> Any:
        processor = self._stage_input_processors.get(stage_id)
        if processor is None:
            from vllm_omni.engine.stage_init_utils import build_stage0_input_processor

            processor = build_stage0_input_processor(self.stage_pools[stage_id].stage_vllm_config)
            self._stage_input_processors[stage_id] = processor
        return processor

    def _upgrade_processed_stage_request(self, request: Any, raw_prompt: Any) -> Any:
        prompt_embeds = getattr(request, "prompt_embeds", None)
        additional_information = None

        if isinstance(raw_prompt, dict):
            if prompt_embeds is None:
                raw_prompt_embeds = raw_prompt.get("prompt_embeds")
                if isinstance(raw_prompt_embeds, torch.Tensor):
                    prompt_embeds = raw_prompt_embeds
            additional_information = serialize_additional_information(
                raw_prompt.get("additional_information"),
                log_prefix="Orchestrator stage input",
            )

        if prompt_embeds is None and additional_information is None:
            return request

        return OmniEngineCoreRequest.from_request(
            request,
            prompt_embeds=prompt_embeds,
            additional_information=additional_information,
        )

    def _next_stage_input_is_tokens(self, next_input: Any) -> bool:
        return isinstance(next_input, dict) and "prompt_token_ids" in next_input

    def _build_next_stage_request(
        self,
        req_id: str,
        next_stage_id: int,
        next_input: Any,
        params: SamplingParams | PoolingParams,
        *,
        mm_features: list | None = None,
        resumable: bool = False,
    ) -> Any:
        next_pool = self.stage_pools[next_stage_id]
        if self._next_stage_input_is_tokens(next_input):
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=next_pool.stage_vllm_config.model_config,
                mm_features=mm_features,
                resumable=resumable,
            )
            request.external_req_id = request.request_id
            return request

        processor = self._get_stage_input_processor(next_stage_id)
        request = processor.process_inputs(
            request_id=req_id,
            prompt=next_input,
            params=params,
            supported_tasks=("generate",),
            arrival_time=_time.time(),
            resumable=resumable,
        )
        request = self._upgrade_processed_stage_request(request, next_input)
        request.external_req_id = req_id
        return request

    @staticmethod
    def _duplex_output_context(
        req_state: OrchestratorRequestState,
        *,
        stage_id: int | None = None,
    ) -> DuplexOutputContext | None:
        identity = req_state.duplex_identity
        if identity is None:
            return None
        from vllm_omni.experimental.fullduplex.engine.contracts import (
            DuplexOutputContext,
            DuplexRequestIdentity,
        )

        fence = req_state.duplex_stage_fences.get(stage_id, identity.fence) if stage_id is not None else identity.fence
        return DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=identity.session_id,
                fence=fence,
            ),
            final_stage_id=req_state.final_stage_id,
            segment_finished=req_state.streaming.enabled and req_state.streaming.segment_finished,
            segment_token_ids=tuple(req_state.streaming.segment_token_ids),
            segment_output_metadata=req_state.streaming.segment_output_metadata,
        )

    @staticmethod
    def _is_duplex_session_request(req_state: OrchestratorRequestState) -> bool:
        return req_state.duplex_identity is not None

    @staticmethod
    def _duplex_fence_for_req_state(
        req_state: OrchestratorRequestState,
        *,
        stage_id: int | None = None,
    ) -> DuplexFence | None:
        context = Orchestrator._duplex_output_context(req_state, stage_id=stage_id)
        return context.identity.fence if context is not None else None

    def _duplex_output_decision(
        self,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> DuplexOutputDecision | None:
        if self.duplex_control_plane is None:
            return None
        context = self._duplex_output_context(req_state, stage_id=stage_id)
        decision = self.duplex_control_plane.decide_output(
            stage_id,
            output,
            context,
        )
        return decision

    async def _emit_duplex_direct_output(
        self,
        stage_id: int,
        req_id: str,
        output: Any,
        decision: DuplexOutputDecision,
        stage_metrics: Any,
        submit_ts: float | None,
    ) -> None:
        action = getattr(decision.action, "value", decision.action)
        if action != "direct_response":
            raise ValueError(f"Unsupported duplex output action: {action}")
        from vllm_omni.experimental.fullduplex.output import attach_duplex_output_decision

        engine_output = attach_duplex_output_decision(
            OmniRequestOutput(
                request_id=req_id,
                finished=True,
                stage_id=stage_id,
                final_output_type=decision.final_output_type,
                request_output=output,
            ),
            decision,
        )
        await self.output_async_queue.put(
            OutputMessage(
                request_id=req_id,
                stage_id=stage_id,
                engine_outputs=engine_output,
                metrics=stage_metrics,
                finished=True,
                stage_submit_ts=submit_ts,
            )
        )

    @staticmethod
    def _completion_multimodal_output(output: Any, completion: Any) -> Mapping[str, Any]:
        mm_output = getattr(output, "multimodal_output", None)
        if isinstance(mm_output, Mapping):
            return mm_output
        mm_output = getattr(completion, "multimodal_output", None) if completion is not None else None
        return mm_output if isinstance(mm_output, Mapping) else {}

    def _capture_pd_mrope_metadata(self, features: Any) -> list[dict[str, Any]]:
        """Capture media layout without retaining media tensors for D."""
        metadata: list[dict[str, Any]] = []
        for feature in features or ():
            mm_position = getattr(feature, "mm_position", None)
            if mm_position is None:
                continue
            identifier = str(getattr(feature, "identifier", ""))
            item = getattr(feature, "data", None)
            values: dict[str, Any] = {}
            if item is not None:
                for key in (
                    "image_grid_thw",
                    "video_grid_thw",
                    "second_per_grid_ts",
                    "use_audio_in_video",
                    "audio_feature_lengths",
                ):
                    elem = item.get(key)
                    value = getattr(elem, "data", None)
                    if value is None:
                        continue
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    values[key] = value
                if values and identifier:
                    self._pd_mrope_values_by_identifier[identifier] = values
            elif identifier:
                values = self._pd_mrope_values_by_identifier.get(identifier, {})
            if values:
                metadata.append(
                    {
                        "modality": str(feature.modality),
                        "identifier": identifier,
                        "offset": int(mm_position.offset),
                        "length": int(mm_position.length),
                        "values": values,
                    }
                )
        return metadata

    @staticmethod
    def _build_pd_mrope_features(metadata: Any) -> list[MultiModalFeatureSpec]:
        """Rebuild metadata-only features for D's Qwen3-Omni M-RoPE path."""
        if not isinstance(metadata, list):
            return []
        features: list[MultiModalFeatureSpec] = []
        for index, raw in enumerate(metadata):
            if not isinstance(raw, dict):
                continue
            values = raw.get("values")
            if not isinstance(values, dict):
                continue
            fields = {
                key: MultiModalFieldElem(
                    data=torch.as_tensor(value),
                    field=MultiModalBatchedField(keep_on_cpu=True),
                )
                for key, value in values.items()
            }
            identifier = f"pd-mrope:{raw.get('identifier', index)}"
            features.append(
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem(fields),
                    modality=str(raw["modality"]),
                    identifier=identifier,
                    mm_position=PlaceholderRange(
                        offset=int(raw["offset"]),
                        length=int(raw["length"]),
                    ),
                    mm_hash=identifier,
                )
            )
        return features

    def _prepare_pd_prefill_snapshot_request(
        self,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Select full or delta P output before submitting a finite request."""
        lineage_id = getattr(stage0_request, "kv_lineage_id", None)
        parent_revision = int(getattr(stage0_request, "kv_lineage_parent_revision", 0))
        revision = int(getattr(stage0_request, "kv_lineage_revision", 0))
        prefix_tokens = max(0, int(getattr(stage0_request, "kv_lineage_prefix_tokens", 0)))
        prompt_ids = tuple(int(token) for token in (getattr(stage0_request, "prompt_token_ids", None) or ()))

        req_state.pd_prefill_lineage_id = lineage_id if isinstance(lineage_id, str) else None
        req_state.pd_prefill_revision = revision
        req_state.pd_prefill_prompt_token_ids = prompt_ids

        parent = self._pd_prefill_snapshots.get(lineage_id) if isinstance(lineage_id, str) else None
        usable_parent_rows = 0
        if parent is not None and parent.revision == parent_revision and prefix_tokens > 0:
            usable_parent_rows = min(prefix_tokens, len(parent.prompt_token_ids))
            if parent.prompt_token_ids[:usable_parent_rows] != prompt_ids[:usable_parent_rows]:
                usable_parent_rows = 0

        if usable_parent_rows > 0 and parent is not None:
            req_state.pd_prefill_parent_snapshot = parent
            req_state.pd_prefill_max_parent_rows = usable_parent_rows
            self._pd_prefill_snapshots.move_to_end(lineage_id)
            snapshot_mode = "delta"
        else:
            snapshot_mode = "full"

        buffer = getattr(stage0_request, "model_intermediate_buffer", None)
        buffer = dict(buffer) if isinstance(buffer, dict) else {}
        meta = buffer.get("meta")
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["pd_prefill_snapshot_mode"] = snapshot_mode
        meta["pd_prefill_snapshot_parent_rows"] = usable_parent_rows
        buffer["meta"] = meta
        stage0_request.model_intermediate_buffer = buffer
        logger.info(
            "[Orchestrator][PD snapshot] req=%s mode=%s prompt_rows=%d reusable_parent_rows=%d",
            req_state.request_id,
            snapshot_mode,
            len(prompt_ids),
            usable_parent_rows,
        )

    @staticmethod
    def _pd_snapshot_nbytes(output: dict[str, Any]) -> int:
        total = 0
        stack: list[Any] = [output]
        seen: set[int] = set()
        while stack:
            value = stack.pop()
            if isinstance(value, torch.Tensor):
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    total += int(value.nbytes)
            elif isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        return total

    def _cache_pd_prefill_snapshot(
        self,
        req_state: OrchestratorRequestState,
        output: dict[str, Any],
        *,
        packed_prefix_chunks: int = 0,
    ) -> bool:
        lineage_id = req_state.pd_prefill_lineage_id
        prompt_ids = req_state.pd_prefill_prompt_token_ids
        limit_bytes = int(getattr(self, "_pd_prefill_snapshot_limit_bytes", 0))
        if not lineage_id or not prompt_ids or limit_bytes <= 0:
            return False

        nbytes = self._pd_snapshot_nbytes(output)
        if nbytes <= 0 or nbytes > limit_bytes:
            return False
        if not hasattr(self, "_pd_prefill_snapshots"):
            self._pd_prefill_snapshots = OrderedDict()
            self._pd_prefill_snapshot_bytes = 0
        previous = self._pd_prefill_snapshots.pop(lineage_id, None)
        if previous is not None:
            self._pd_prefill_snapshot_bytes -= previous.nbytes
        snapshot = _PDPrefillSnapshot(
            revision=req_state.pd_prefill_revision,
            prompt_token_ids=prompt_ids,
            output=output,
            nbytes=nbytes,
            packed_prefix_chunks=max(0, int(packed_prefix_chunks)),
        )
        self._pd_prefill_snapshots[lineage_id] = snapshot
        self._pd_prefill_snapshot_bytes += nbytes
        while self._pd_prefill_snapshots and self._pd_prefill_snapshot_bytes > limit_bytes:
            _, evicted = self._pd_prefill_snapshots.popitem(last=False)
            self._pd_prefill_snapshot_bytes -= evicted.nbytes
        return self._pd_prefill_snapshots.get(lineage_id) is snapshot

    @staticmethod
    def _pd_snapshot_layer_chunks(output: dict[str, Any], layer: int) -> tuple[torch.Tensor, ...]:
        hidden = output.get("hidden_states")
        layers = hidden.get("layers") if isinstance(hidden, dict) else None
        value = layers.get(layer, layers.get(str(layer))) if isinstance(layers, dict) else None
        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, (list, tuple)) and all(isinstance(chunk, torch.Tensor) for chunk in value):
            return tuple(value)
        return ()

    @staticmethod
    def _slice_pd_snapshot_chunks(chunks: tuple[torch.Tensor, ...], rows: int) -> tuple[torch.Tensor, ...]:
        remaining = rows
        selected: list[torch.Tensor] = []
        for chunk in chunks:
            if remaining <= 0:
                break
            take = min(remaining, int(chunk.shape[0]))
            if take > 0:
                selected.append(chunk[:take])
                remaining -= take
        if remaining > 0:
            return ()
        return tuple(selected)

    async def _materialize_pd_prefill_snapshot(self, req_state: OrchestratorRequestState) -> bool:
        """Merge a P cache-hit tail with the lineage's latest snapshot."""
        diag_start = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
        current = req_state.pd_prefill_multimodal_output
        if not isinstance(current, dict):
            return False
        hidden = current.get("hidden_states")
        layers = hidden.get("layers") if isinstance(hidden, dict) else None
        if not isinstance(layers, dict):
            return False

        hidden_layer = int(getattr(self, "_pd_snapshot_hidden_layer", 24))
        layer_0 = layers.get(0, layers.get("0"))
        layer_hidden = layers.get(hidden_layer, layers.get(str(hidden_layer)))
        if not isinstance(layer_0, torch.Tensor) or not isinstance(layer_hidden, torch.Tensor):
            return False
        prompt_rows = len(req_state.pd_prefill_prompt_token_ids)
        current_rows = min(int(layer_0.shape[0]), int(layer_hidden.shape[0]))
        if current_rows > prompt_rows:
            raise RuntimeError(
                f"[Orchestrator][PD] P snapshot has {current_rows} rows for a {prompt_rows}-token prompt"
            )

        packed_prefix_chunks = 0
        if current_rows < prompt_rows:
            prefix_rows = prompt_rows - current_rows
            parent = req_state.pd_prefill_parent_snapshot
            if parent is None or prefix_rows > req_state.pd_prefill_max_parent_rows:
                raise RuntimeError(
                    "[Orchestrator][PD] delta P snapshot lacks an exact parent: "
                    f"req={req_state.request_id} prefix_rows={prefix_rows} "
                    f"available={req_state.pd_prefill_max_parent_rows}"
                )
            parent_0_chunks = self._slice_pd_snapshot_chunks(
                self._pd_snapshot_layer_chunks(parent.output, 0),
                prefix_rows,
            )
            parent_hidden_chunks = self._slice_pd_snapshot_chunks(
                self._pd_snapshot_layer_chunks(parent.output, hidden_layer),
                prefix_rows,
            )
            if not parent_0_chunks or not parent_hidden_chunks:
                raise RuntimeError(f"[Orchestrator][PD] parent P snapshot is incomplete for req={req_state.request_id}")
            packed_prefix_chunks = min(
                max(0, int(parent.packed_prefix_chunks)),
                len(parent_0_chunks),
                len(parent_hidden_chunks),
            )
            layer_0_chunks = (*parent_0_chunks, layer_0[:current_rows])
            layer_hidden_chunks = (*parent_hidden_chunks, layer_hidden[:current_rows])
        else:
            prefix_rows = 0
            layer_0_chunks = (layer_0[:current_rows],)
            layer_hidden_chunks = (layer_hidden[:current_rows],)

        embeds = current.get("embed")
        if not isinstance(embeds, dict) and req_state.pd_prefill_parent_snapshot is not None:
            embeds = req_state.pd_prefill_parent_snapshot.output.get("embed")
        cached_output: dict[str, Any] = {
            "hidden_states": {"layers": {0: layer_0_chunks, hidden_layer: layer_hidden_chunks}},
        }
        if isinstance(embeds, dict):
            cached_output["embed"] = embeds

        prefill_only = isinstance(req_state.prompt, dict) and req_state.prompt.get("prefill_only") is True
        max_chunks = int(getattr(self, "_pd_prefill_snapshot_max_chunks", 16))
        chunks_before = max(len(layer_0_chunks), len(layer_hidden_chunks))
        unpacked_chunks_before = max(
            len(layer_0_chunks) - packed_prefix_chunks,
            len(layer_hidden_chunks) - packed_prefix_chunks,
        )
        compact_ms = 0.0
        if unpacked_chunks_before > max_chunks:
            if len(layer_0_chunks) != len(layer_hidden_chunks):
                raise RuntimeError(
                    "[Orchestrator][PD] snapshot layers have mismatched chunk boundaries: "
                    f"req={req_state.request_id} layer0={len(layer_0_chunks)} "
                    f"layer{hidden_layer}={len(layer_hidden_chunks)}"
                )
            compact_start = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
            # Pack only the newly accumulated suffix into a shared slab.
            # Previously packed prefix slabs remain immutable and are reused
            # by later revisions, so no token is recopied on every threshold.
            compacted_layer_0, compacted_layer_hidden = await asyncio.to_thread(
                self._materialize_pd_snapshot_pair,
                layer_0_chunks[packed_prefix_chunks:],
                layer_hidden_chunks[packed_prefix_chunks:],
            )
            layer_0_chunks = (
                *layer_0_chunks[:packed_prefix_chunks],
                compacted_layer_0,
            )
            layer_hidden_chunks = (
                *layer_hidden_chunks[:packed_prefix_chunks],
                compacted_layer_hidden,
            )
            packed_prefix_chunks += 1
            compacted: dict[str, Any] = {
                "hidden_states": {
                    "layers": {
                        0: layer_0_chunks,
                        hidden_layer: layer_hidden_chunks,
                    }
                }
            }
            if isinstance(embeds, dict):
                compacted["embed"] = embeds
            cached_output = compacted
            if _LOG_HANDOFF_DIAG:
                compact_ms = (_time.monotonic() - compact_start) * 1000.0
        if not prefill_only:
            # Keep P's shared output chunks intact.  D already concatenates
            # the prompt conditioning with its first decode rows for Talker,
            # so materializing a contiguous copy here is redundant.
            req_state.pd_prefill_multimodal_output = cached_output
        cache_start = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
        snapshot_cached = self._cache_pd_prefill_snapshot(
            req_state,
            cached_output,
            packed_prefix_chunks=packed_prefix_chunks,
        )
        if _LOG_HANDOFF_DIAG:
            diag_end = _time.monotonic()
            logger.info(
                "[PD-SNAPSHOT-DIAG] req=%s prompt_rows=%d delta_rows=%d "
                "chunks_before=%d unpacked_chunks_before=%d compact_ms=%.3f "
                "cache_ms=%.3f total_ms=%.3f",
                req_state.request_id,
                prompt_rows,
                current_rows,
                chunks_before,
                unpacked_chunks_before,
                compact_ms,
                (diag_end - cache_start) * 1000.0,
                (diag_end - diag_start) * 1000.0,
            )
        logger.info(
            "[Orchestrator][PD snapshot] req=%s assembled prefix_rows=%d delta_rows=%d "
            "deferred=%s cached=%s cache_chunks=%d cache_mib=%.1f",
            req_state.request_id,
            prefix_rows,
            current_rows,
            prefill_only,
            snapshot_cached,
            max(
                len(self._pd_snapshot_layer_chunks(cached_output, 0)),
                len(self._pd_snapshot_layer_chunks(cached_output, hidden_layer)),
            ),
            float(getattr(self, "_pd_prefill_snapshot_bytes", 0)) / float(1 << 20),
        )
        return snapshot_cached

    @classmethod
    def _materialize_pd_snapshot_pair(
        cls,
        layer_0_chunks: tuple[torch.Tensor, ...],
        layer_hidden_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact both immutable snapshot layers off the event-loop thread."""
        return (
            cls._materialize_pd_snapshot_layer(layer_0_chunks, shared=True),
            cls._materialize_pd_snapshot_layer(layer_hidden_chunks, shared=True),
        )

    @staticmethod
    def _materialize_pd_snapshot_layer(
        chunks: tuple[torch.Tensor, ...],
        *,
        shared: bool = False,
    ) -> torch.Tensor:
        """Coalesce a snapshot into one contiguous CPU tensor."""
        if len(chunks) == 1:
            output = chunks[0]
            if shared and output.device.type == "cpu" and not output.is_shared():
                output.share_memory_()
            return output

        first = chunks[0]
        shape = (sum(int(chunk.shape[0]) for chunk in chunks), *first.shape[1:])
        if shared and first.device.type == "cpu":
            # ``torch.empty(shape).share_memory_()`` first allocates ordinary
            # storage and then copies it into shared storage.  Build the tensor
            # directly on a shared storage instead, then fill it once via cat.
            numel = 1
            for dim in shape:
                numel *= int(dim)
            storage = torch.UntypedStorage._new_shared(
                numel * first.element_size(),
                device="cpu",
            )
            output = torch.empty(0, dtype=first.dtype, device="cpu").set_(storage, 0, shape)
        else:
            output = torch.empty(shape, dtype=first.dtype, device=first.device)
        torch.cat(chunks, dim=0, out=output)
        return output

    @staticmethod
    def _ensure_shared_pd_snapshot_chunks(
        chunks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Ensure a local D request can encode every snapshot chunk by handle."""
        shared_chunks: list[torch.Tensor] = []
        for chunk in chunks:
            chunk = chunk.detach().cpu()
            if not chunk.is_shared():
                chunk.share_memory_()
            shared_chunks.append(chunk)
        return tuple(shared_chunks)

    @staticmethod
    def _accumulate_pd_prefill_output(
        accumulated: dict[str, Any] | None,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Accumulate only the P tensors required by the D-to-Talker edge."""
        current = unflatten_payload(current)
        hidden = current.get("hidden_states")
        layers = hidden.get("layers") if isinstance(hidden, dict) else None
        if not isinstance(layers, dict):
            return accumulated

        selected_layers: dict[int, torch.Tensor] = {}
        for key in (0, 24, 48, "0", "24", "48"):
            value = layers.get(key)
            if isinstance(value, torch.Tensor):
                selected_layers[int(key)] = value.detach().cpu()
        if not selected_layers:
            return accumulated

        if accumulated is None:
            accumulated = {"hidden_states": {"layers": {}}}
        destination = accumulated.setdefault("hidden_states", {}).setdefault("layers", {})
        for key, value in selected_layers.items():
            previous = destination.get(key)
            destination[key] = torch.cat((previous, value), dim=0) if isinstance(previous, torch.Tensor) else value

        embeds = current.get("embed")
        if isinstance(embeds, dict):
            destination_embeds = accumulated.setdefault("embed", {})
            for key in ("tts_bos", "tts_eos", "tts_pad"):
                value = embeds.get(key)
                if isinstance(value, torch.Tensor):
                    destination_embeds.setdefault(key, value.detach().cpu())
        return accumulated

    @classmethod
    def _coerce_int_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1).tolist()
            except Exception:
                return []
        if not isinstance(value, (list, tuple)):
            return []
        out: list[int] = []
        for item in value:
            token_id = cls._coerce_int(item)
            if token_id is not None:
                out.append(token_id)
        return out

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1)
                if value.numel() == 0:
                    return None
                value = value[0].item()
            except Exception:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if (stage_id + 1) in parent_state.stage_submit_ts:
            return

        await self._forward_to_next_stage(
            parent_id,
            stage_id,
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        stage_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
        if self.async_chunk:
            return

        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue

            req_id = self._pd_decode_request_aliases.get(
                raw_output.request_id,
                raw_output.request_id,
            )
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                # kv_ready only says the companion's KV hit the connector; its
                # processed output has not been stashed yet. Counting it as
                # done here let the parent pass the all_companions_done gate
                # and dispatch with 0/N companion outputs (degraded CFG).
                # Done-marking now happens solely on the processed-output path
                # in _route_output, after set_companion_output.
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if (stage_id + 1) in req_state.stage_submit_ts:
                continue

            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_pd_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build decode-side sampling params with KV transfer params for PD routing.

        Clones the sampling params and injects kv_transfer_params that tell the
        decode engine where to pull the KV cache from (prefill engine's bootstrap addr).
        """
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}

        # Get KV params captured from the prefill output (must include remote_request_id).
        kv_prefill_params = self._pd_kv_params.pop(req_id, None)
        if not kv_prefill_params or "remote_request_id" not in kv_prefill_params:
            raise RuntimeError(
                f"[Orchestrator][PD] Missing prefill kv_transfer_params.remote_request_id for req={req_id}"
            )

        kv_prefill_params = dict(kv_prefill_params)
        # This local scheduler identity is used to build D's prompt and must
        # not be copied into the connector control payload.
        remote_prompt_token_ids = kv_prefill_params.pop(
            "remote_prompt_token_ids",
            None,
        )
        remote_prompt_offset = kv_prefill_params.pop("remote_prompt_token_offset", 0)
        decode_kv_params: dict[str, Any] = {
            "transfer_id": f"xfer-{req_id}",
        }
        if isinstance(remote_prompt_token_ids, (list, tuple)):
            decode_kv_params["remote_prompt_tokens"] = remote_prompt_offset + len(remote_prompt_token_ids)

        if self._pd_bootstrap_addr:
            decode_kv_params["remote_bootstrap_addr"] = self._pd_bootstrap_addr

        if self._pd_prefill_engine_id:
            decode_kv_params["remote_engine_id"] = self._pd_prefill_engine_id

        # Overlay params from prefill side (includes remote_request_id set by monkey patch).
        decode_kv_params.update(kv_prefill_params)

        # Ensure these flags are set correctly after any overlay.
        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        sp.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[Orchestrator][PD] decode kv_transfer_params for req=%s: %s",
            req_id,
            decode_kv_params,
        )
        return sp

    def _prepare_native_duplex_pd_decode(
        self,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Build the finite D prompt from one completed native P segment."""
        req_id = req_state.request_id
        kv_params = getattr(output, "kv_transfer_params", None)
        if not isinstance(kv_params, dict):
            raise RuntimeError(f"[Orchestrator][PD] native P boundary lacks KV metadata for req={req_id}")
        remote_prompt_token_ids = kv_params.get("remote_prompt_token_ids")
        if not isinstance(remote_prompt_token_ids, (list, tuple)):
            raise RuntimeError(f"[Orchestrator][PD] native P boundary lacks remote prompt tokens for req={req_id}")

        offset = kv_params.get("remote_prompt_token_offset", 0)
        if type(offset) is not int or offset < 0:
            raise RuntimeError(f"Invalid native P prompt offset for req={req_id}: {offset!r}")
        if offset:
            previous = req_state.streaming.bridge_states.get("pd_duplex_remote_prompt_token_ids")
            if not isinstance(previous, (list, tuple)) or len(previous) != offset:
                raise RuntimeError(f"Native P prompt delta has no matching prefix for req={req_id}: offset={offset}")
            remote_prompt_token_ids = [*previous, *remote_prompt_token_ids]

        self._pd_kv_params[req_id] = dict(kv_params)
        actual_prefix_ids = (
            remote_prompt_token_ids
            if isinstance(remote_prompt_token_ids, list)
            else [int(token_id) for token_id in remote_prompt_token_ids]
        )
        # This route runs in a detached task. The shared streaming scratchpad
        # can already describe a later Talker/Code2Wav poll when we get here.
        # Read the immutable P boundary itself, never another stage's tokens.
        raw_ids = getattr(output, "new_token_ids", None)
        if raw_ids is None:
            completions = getattr(output, "outputs", None)
            if isinstance(completions, (list, tuple)) and len(completions) == 1:
                raw_ids = getattr(completions[0], "token_ids", None)
        p_sampled_ids = self._coerce_int_list(raw_ids)
        if len(p_sampled_ids) != 1:
            raise RuntimeError(
                f"Native P boundary must own exactly one sampled token for req={req_id}; "
                f"boundary_tokens={p_sampled_ids[:21]}"
            )
        decode_prompt: dict[str, Any] = {
            "prompt_token_ids": [
                *actual_prefix_ids,
                *p_sampled_ids,
            ],
        }
        source_prompt = req_state.prompt
        if isinstance(source_prompt, dict):
            for key in ("additional_information", "cache_salt"):
                if key in source_prompt:
                    decode_prompt[key] = copy.deepcopy(source_prompt[key])
            source_model_buffer = source_prompt.get("model_intermediate_buffer")
            if isinstance(source_model_buffer, dict):
                # The native D runner imports P's cumulative AV KV and
                # deliberately ignores ``duplex.payload``.  Avoid copying and
                # serializing the raw audio/video a second time on every slot;
                # retain only the compact control metadata D and Talker use.
                decode_model_buffer = {
                    key: copy.deepcopy(value) for key, value in source_model_buffer.items() if key != "duplex"
                }
                source_duplex = source_model_buffer.get("duplex")
                if isinstance(source_duplex, dict):
                    decode_model_buffer["duplex"] = {
                        key: copy.deepcopy(value) for key, value in source_duplex.items() if key != "payload"
                    }
                    source_payload = source_duplex.get("payload") or {}
                    decode_model_buffer["duplex"]["payload"] = {
                        key: source_payload[key] for key in ("force_listen", "is_speech") if key in source_payload
                    }
                decode_prompt["model_intermediate_buffer"] = decode_model_buffer
        bridge = req_state.streaming.bridge_states
        bridge["pd_duplex_remote_prompt_token_ids"] = actual_prefix_ids
        active_slot = bridge.get("pd_duplex_active_slot")
        slot_seq = self._coerce_int(active_slot.get("seq")) if isinstance(active_slot, dict) else None
        if slot_seq is None:
            slot_seq = int(bridge.get("pd_duplex_decode_sequence", 0)) + 1
        bridge["pd_duplex_decode_sequence"] = slot_seq
        # A logical duplex session owns one resumable P request, but D must
        # see a fresh finite request on every slot.  The trailing eight-hex
        # suffix is intentional: NIXL's push matcher strips that suffix and
        # therefore pairs this D request with the persistent P request while
        # keeping all D scheduler/connector state request-scoped.
        expected_engine_req_id = f"{req_id}-{slot_seq & 0xFFFFFFFF:08x}"
        decode_engine_req_id = bridge.get("pd_decode_engine_request_id")
        if decode_engine_req_id != expected_engine_req_id:
            decode_engine_req_id = expected_engine_req_id
        bridge["pd_decode_prompt"] = decode_prompt
        bridge["pd_decode_prompt_len"] = len(decode_prompt["prompt_token_ids"])
        bridge["pd_decode_last_prompt_token_id"] = (
            int(decode_prompt["prompt_token_ids"][-1]) if decode_prompt["prompt_token_ids"] else None
        )
        decode_model_buffer = decode_prompt.get("model_intermediate_buffer")
        decode_duplex = decode_model_buffer.get("duplex") if isinstance(decode_model_buffer, dict) else None
        decode_special_ids = decode_duplex.get("special_token_ids") if isinstance(decode_duplex, dict) else None
        if isinstance(decode_special_ids, dict):
            bridge["pd_duplex_special_token_ids"] = dict(decode_special_ids)
        bridge["pd_decode_engine_request_id"] = decode_engine_req_id
        bridge["pd_decode_transfer_id"] = f"xfer-{decode_engine_req_id}"
        bridge["pd_duplex_prefill_sample_token_ids"] = p_sampled_ids
        if isinstance(active_slot, dict):
            active_slot["prompt_tokens"] = len(decode_prompt["prompt_token_ids"])
        p_mm_output = self._completion_multimodal_output(output, None)
        policy_snapshot = p_mm_output.get(SAMPLING_STATE_WIRE_KEY)
        if policy_snapshot is None:
            nested_meta = p_mm_output.get("meta")
            policy_snapshot = nested_meta.get(SAMPLING_STATE_KEY) if isinstance(nested_meta, dict) else None
        policy_state, identity = unpack_sampling_state(policy_snapshot)
        decode_duplex = decode_prompt["model_intermediate_buffer"]["duplex"]
        decode_duplex["pd_media_prefix_tokens"] = len(actual_prefix_ids)
        expected_identity = (
            int(decode_duplex.get("incarnation", 0)),
            decode_duplex.get("epoch"),
            decode_duplex.get("seq"),
        )
        if identity != expected_identity or policy_state.current_segment_output_tokens != p_sampled_ids:
            raise RuntimeError(
                "Native P sampling state does not match the KV handoff boundary: "
                f"expected_identity={expected_identity} actual_identity={identity} "
                f"boundary_tokens={p_sampled_ids} "
                f"state_tokens={policy_state.current_segment_output_tokens[:21]}"
            )
        # The full prompt is released immediately after D submission. Only
        # retain the small identity needed to validate its later completion.
        bridge["pd_duplex_sampling_identity"] = expected_identity
        decode_duplex["payload"][SAMPLING_STATE_KEY] = (
            policy_snapshot.tolist() if isinstance(policy_snapshot, torch.Tensor) else list(policy_snapshot)
        )
        if p_mm_output:
            p_mm_output = unflatten_payload(dict(p_mm_output))
            if isinstance(active_slot, dict):
                audit_sources = [p_mm_output]
                nested_audit = p_mm_output.get("duplex")
                if isinstance(nested_audit, dict):
                    audit_sources.append(nested_audit)
                for wire_name, slot_name in (
                    ("duplex_input_video_frames", "input_video_frames"),
                    ("duplex_arrival_video_frames", "arrival_video_frames"),
                    ("duplex_vision_fallback_frames", "vision_fallback_frames"),
                    ("duplex_arrival_audio_units", "arrival_audio_units"),
                    ("duplex_audio_fallback_units", "audio_fallback_units"),
                ):
                    for audit_source in audit_sources:
                        audit_value = self._coerce_int(audit_source.get(wire_name))
                        if audit_value is not None:
                            active_slot[slot_name] = max(0, audit_value)
                            break
            p_meta = p_mm_output.get("meta")
            p_special = p_mm_output.get("special_token_ids")
            special_ids: dict[str, int] = {}
            for source in (
                p_special if isinstance(p_special, dict) else {},
                p_meta if isinstance(p_meta, dict) else {},
            ):
                for key, value in source.items():
                    value = self._coerce_int(value)
                    if isinstance(key, str) and key.endswith("_token_id") and value is not None and value >= 0:
                        special_ids[key] = value
            for key, value in p_mm_output.items():
                if not (isinstance(key, str) and key.startswith("meta.") and key.endswith("_token_id")):
                    continue
                value = self._coerce_int(value)
                if value is not None and value >= 0:
                    special_ids[key.removeprefix("meta.")] = value
            if special_ids:
                bridge["pd_duplex_special_token_ids"] = special_ids
        special_ids = bridge.get("pd_duplex_special_token_ids", {})
        bridge["pd_duplex_terminal_replay"] = len(p_sampled_ids) == 1 and p_sampled_ids[0] in {
            special_ids.get(key, -1) for key in (
                "listen_token_id", "chunk_eos_token_id", "chunk_tts_eos_token_id")
        }

    @staticmethod
    def _capture_native_pd_sampling_feedback(output: Any, req_state: OrchestratorRequestState) -> None:
        """Require D's exact post-sample policy state before admitting the next P unit."""
        metadata = Orchestrator._completion_multimodal_output(output, None)
        value = metadata.get(SAMPLING_STATE_WIRE_KEY)
        if value is None:
            nested = metadata.get("meta")
            value = nested.get(SAMPLING_STATE_KEY) if isinstance(nested, dict) else None
        state, identity = unpack_sampling_state(value)
        bridge = req_state.streaming.bridge_states
        expected = bridge["pd_duplex_sampling_identity"]
        if identity != expected or state.current_segment_output_tokens != bridge["pd_duplex_feedback_token_ids"]:
            raise RuntimeError("Native D sampling state does not match completed segment")
        bridge["pd_duplex_feedback_sampling_state"] = value.tolist() if isinstance(value, torch.Tensor) else list(value)

    @staticmethod
    def _ensure_native_duplex_pd_talker_metadata(
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Attach compact native handoff metadata to a finite D output.

        D receives a token-only prompt and imports the media prefix KV.  Its
        accumulated latent is authoritative, while the prompt-boundary and
        tokenizer IDs remain application metadata.  The Talker needs only the
        boundary and current D segment, never another full cached-prompt copy.
        """
        completions = getattr(output, "outputs", None)
        completion = completions[0] if isinstance(completions, list) and completions else None
        mm_output = getattr(completion, "multimodal_output", None)
        metadata = getattr(mm_output, "metadata", None)
        if not isinstance(metadata, dict):
            return

        bridge = req_state.streaming.bridge_states
        prompt_len = bridge.get("pd_decode_prompt_len")
        last_prompt_token = bridge.get("pd_decode_last_prompt_token_id")
        if not isinstance(prompt_len, int):
            # Compatibility for request states created before compact prompt
            # metadata was introduced.
            decode_prompt = bridge.get("pd_decode_prompt")
            prompt_ids = decode_prompt.get("prompt_token_ids") if isinstance(decode_prompt, dict) else None
            if not isinstance(prompt_ids, (list, tuple)):
                return
            prompt_len = len(prompt_ids)
            last_prompt_token = int(prompt_ids[-1]) if prompt_ids else None
        if prompt_len < 0:
            return
        metadata["duplex_pd_decode"] = True
        metadata.pop(NATIVE_PROMPT_TOKEN_IDS_KEY, None)
        metadata.setdefault(NATIVE_PROMPT_LEN_KEY, prompt_len)
        metadata.setdefault(
            NATIVE_LAST_PROMPT_TOKEN_KEY,
            last_prompt_token,
        )
        if metadata.get(NATIVE_SEGMENT_TOKEN_IDS_KEY) is None:
            segment_ids = getattr(completion, "token_ids", None)
            if isinstance(segment_ids, (list, tuple)):
                metadata[NATIVE_SEGMENT_TOKEN_IDS_KEY] = [int(token_id) for token_id in segment_ids]

        special_ids = bridge.get("pd_duplex_special_token_ids")
        if isinstance(special_ids, dict):
            existing = metadata.get("special_token_ids")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(
                {
                    str(key): int(value)
                    for key, value in special_ids.items()
                    if isinstance(key, str) and isinstance(value, int) and value >= 0
                }
            )
            metadata["special_token_ids"] = merged

    @staticmethod
    def _signal_native_duplex_pd_route_error(
        req_state: OrchestratorRequestState,
        error: str,
    ) -> None:
        """Wake a same-session append that is waiting for the previous D."""
        bridge = req_state.streaming.bridge_states
        bridge.setdefault("pd_duplex_prefill_raw_error", error)
        decode_ready = bridge.get("pd_duplex_decode_ready")
        if isinstance(decode_ready, asyncio.Event):
            decode_ready.set()

    async def _cancel_pd_raw_route_tasks(
        self,
        request_ids: list[str] | None = None,
        *,
        reason: str,
    ) -> None:
        """Cancel and join owned raw P->D route tasks before state teardown."""
        tasks = getattr(self, "_pd_raw_route_tasks", None)
        if not tasks:
            return
        selected_ids = list(tasks) if request_ids is None else list(dict.fromkeys(request_ids))
        current = asyncio.current_task()
        to_join: list[asyncio.Task[None]] = []
        for request_id in selected_ids:
            task = tasks.get(request_id)
            if task is None:
                continue
            req_state = getattr(self, "request_states", {}).get(request_id)
            if req_state is not None:
                self._signal_native_duplex_pd_route_error(req_state, reason)
            if task is current:
                continue
            if not task.done():
                task.cancel()
            to_join.append(task)
        if to_join:
            await asyncio.gather(*to_join, return_exceptions=True)
        for request_id in selected_ids:
            task = tasks.get(request_id)
            if task is not None and task is not current and task.done():
                tasks.pop(request_id, None)

    def _schedule_native_duplex_pd_prefill_raw(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Own one raw P->D route task without blocking global output polling."""
        req_id = req_state.request_id
        if self.request_states.get(req_id) is not req_state:
            return
        bridge = req_state.streaming.bridge_states
        physical_id = bridge.get("pd_decode_engine_request_id")
        if not isinstance(physical_id, str) or not physical_id:
            raise RuntimeError(f"[Orchestrator][PD] native P boundary has no physical D id for req={req_id}")
        if (
            bridge.get("pd_duplex_prefill_raw_scheduled_id") == physical_id
            or bridge.get("pd_duplex_prefill_raw_routed_id") == physical_id
            or bridge.get("pd_duplex_prefill_raw_routed", False)
        ):
            return

        tasks = getattr(self, "_pd_raw_route_tasks", None)
        if tasks is None:
            tasks = self._pd_raw_route_tasks = {}
        previous = tasks.get(req_id)
        if previous is not None and not previous.done():
            raise RuntimeError(f"duplicate in-flight native P/D route for req={req_id} physical={physical_id}")
        if previous is not None:
            if not previous.cancelled():
                previous.exception()
            tasks.pop(req_id, None)

        bridge["pd_duplex_prefill_raw_scheduled_id"] = physical_id
        task = asyncio.create_task(
            self._run_native_duplex_pd_prefill_raw(
                stage_id,
                replica_id,
                output,
                req_state,
                physical_id=physical_id,
            ),
            name=f"orchestrator-native-pd-route-{physical_id}",
        )
        tasks[req_id] = task

        def _discard(done: asyncio.Task[None]) -> None:
            if tasks.get(req_id) is done:
                tasks.pop(req_id, None)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error(
                    "[Orchestrator][PD] raw route task failed during error cleanup req=%s physical=%s: %s",
                    req_id,
                    physical_id,
                    error,
                )

        task.add_done_callback(_discard)

    async def _run_native_duplex_pd_prefill_raw(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        physical_id: str,
    ) -> None:
        req_id = req_state.request_id
        try:
            if self.request_states.get(req_id) is not req_state:
                return
            if req_state.streaming.bridge_states.get("pd_decode_engine_request_id") != physical_id:
                raise RuntimeError(f"stale native P/D route identity for req={req_id} physical={physical_id}")
            await self._route_native_duplex_pd_prefill_raw(
                stage_id,
                replica_id,
                output,
                req_state,
                physical_id=physical_id,
            )
        except asyncio.CancelledError:
            raise
        except EngineDeadError as exc:
            await self._handle_native_duplex_pd_route_engine_dead(
                stage_id,
                exc,
                req_state,
            )
        except Exception as exc:
            if self.request_states.get(req_id) is not req_state:
                return
            error_text = f"{type(exc).__name__}: {exc}"
            self._signal_native_duplex_pd_route_error(req_state, error_text)
            await self.output_async_queue.put(
                ErrorMessage(
                    request_id=req_id,
                    stage_id=self._pd_pair[1] if self._pd_pair is not None else stage_id + 1,
                    error=str(exc),
                    error_type="PDRawRouteError",
                )
            )
            try:
                await self._cleanup_request_ids(
                    [req_id],
                    abort=True,
                    close_duplex_sessions=True,
                )
            except Exception:
                logger.exception(
                    "[Orchestrator][PD] cleanup after raw route failure failed req=%s physical=%s",
                    req_id,
                    physical_id,
                )
        finally:
            bridge = req_state.streaming.bridge_states
            if bridge.get("pd_duplex_prefill_raw_scheduled_id") == physical_id:
                bridge.pop("pd_duplex_prefill_raw_scheduled_id", None)

    async def _handle_native_duplex_pd_route_engine_dead(
        self,
        prefill_stage_id: int,
        exc: EngineDeadError,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Quarantine a D replica that dies during detached admission."""
        req_id = req_state.request_id
        d_stage_id = self._pd_pair[1] if self._pd_pair is not None else prefill_stage_id + 1
        d_pool = self.stage_pools[d_stage_id]
        available_before = d_pool.available_replica_ids()
        failed_replica_id = getattr(exc, "vllm_omni_replica_id", None)
        failed_stage_id = getattr(exc, "vllm_omni_stage_id", d_stage_id)
        if failed_stage_id != d_stage_id:
            failed_replica_id = None
        if failed_replica_id is None:
            failed_replica_id = d_pool.get_bound_replica_id(req_id)
        if failed_replica_id is None and len(available_before) == 1:
            # Native Duplex P/D currently deploys one D replica.  If an
            # unannotated admission error reaches us, the sole live replica is
            # necessarily the failed target.
            failed_replica_id = available_before[0]

        affected_request_ids: list[str] = []
        if isinstance(failed_replica_id, int):
            affected_request_ids.extend(d_pool.mark_replica_unavailable(failed_replica_id))
        if self.request_states.get(req_id) is req_state:
            affected_request_ids.insert(0, req_id)
            self._signal_native_duplex_pd_route_error(
                req_state,
                f"{type(exc).__name__}: {exc}",
            )
        affected_request_ids = list(dict.fromkeys(affected_request_ids))

        fatal = not d_pool.available_replica_ids()
        if fatal:
            self._fatal_error = str(exc) or "Decode stage engine died during admission"
            self._fatal_error_stage_id = d_stage_id

        message_request_ids: list[str | None] = affected_request_ids or ([None] if fatal else [])
        for affected_request_id in message_request_ids:
            await self.output_async_queue.put(
                ErrorMessage(
                    request_id=affected_request_id,
                    stage_id=d_stage_id,
                    error=str(exc),
                    error_type="EngineDeadError",
                    fatal=fatal,
                )
            )

        try:
            await self._cleanup_request_ids(
                affected_request_ids,
                close_duplex_sessions=True,
            )
        except Exception:
            logger.exception(
                "[Orchestrator][PD] cleanup after D admission death failed req=%s stage=%s replica=%s",
                req_id,
                d_stage_id,
                failed_replica_id,
            )
        finally:
            if fatal:
                self._shutdown_event.set()

    async def _route_native_duplex_pd_prefill_raw(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        physical_id: str | None = None,
    ) -> None:
        """Dispatch D at the raw resumable P boundary.

        vLLM's FINAL_ONLY output processor waits for request termination, while
        native duplex P remains alive across slots.  The raw segment boundary
        is therefore the authoritative point at which P's KV is publishable.
        """
        bridge = req_state.streaming.bridge_states
        physical_id = physical_id or bridge.get("pd_decode_engine_request_id")
        if not isinstance(physical_id, str) or not physical_id:
            raise RuntimeError(
                f"[Orchestrator][PD] native P boundary has no physical D id for req={req_state.request_id}"
            )
        if bridge.get("pd_duplex_prefill_raw_routed_id") == physical_id:
            return
        if self.request_states.get(req_state.request_id) is not req_state:
            return
        if bridge.get("pd_decode_engine_request_id") != physical_id:
            raise RuntimeError(f"stale native P/D route identity for req={req_state.request_id} physical={physical_id}")
        self._prepare_native_duplex_pd_decode(output, req_state)
        if self.request_states.get(req_state.request_id) is not req_state:
            return
        if bridge.get("pd_decode_engine_request_id") != physical_id:
            raise RuntimeError(
                f"native P/D route identity changed during preparation "
                f"req={req_state.request_id} physical={physical_id}"
            )
        try:
            await self._forward_to_next_stage(
                req_state.request_id,
                stage_id,
                output,
                req_state,
                src_replica_id=replica_id,
                is_streaming_session=True,
                is_final_update=False,
            )
        except Exception:
            bridge["pd_duplex_prefill_raw_routed"] = False
            if bridge.get("pd_duplex_prefill_raw_routed_id") == physical_id:
                bridge.pop("pd_duplex_prefill_raw_routed_id", None)
            raise
        bridge["pd_duplex_prefill_raw_routed"] = True
        bridge["pd_duplex_prefill_raw_routed_id"] = physical_id

    def _build_pd_local_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build D params after an early cache-only import completed.

        The paired P request has already populated D's ordinary prefix cache,
        so admitting the finite decode request with another remote-prefill
        registration would repeat the P/D handshake and strand P's completed
        transfer state.  D locally computes only the final non-cacheable block.
        """
        sp = sp.clone()
        if sp.extra_args is not None:
            sp.extra_args = dict(sp.extra_args)
            sp.extra_args.pop("kv_transfer_params", None)
        self._pd_kv_params.pop(req_id, None)
        return sp

    def _build_pd_early_cache_params(
        self,
        remote_req_id: str,
        transfer_req_id: str,
        sp: Any,
    ) -> Any | None:
        remote = self._pd_prefill_remote
        if not isinstance(remote, dict):
            return None
        required = ("remote_engine_id", "remote_host", "remote_port", "tp_size")
        if any(remote.get(key) is None for key in required):
            return None
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}
        else:
            sp.extra_args = dict(sp.extra_args)
        sp.extra_args["kv_transfer_params"] = {
            **remote,
            "remote_request_id": remote_req_id,
            "transfer_id": f"xfer-{transfer_req_id}",
            "do_remote_prefill": True,
            "do_remote_decode": False,
        }
        return sp

    @staticmethod
    def _pd_decode_inputs(req_state: OrchestratorRequestState) -> list[dict[str, Any]]:
        original_prompt = req_state.streaming.bridge_states.get(
            "pd_decode_prompt",
            req_state.prompt,
        )
        raw_inputs = [original_prompt] if not isinstance(original_prompt, list) else original_prompt
        decode_inputs: list[dict[str, Any]] = []
        for decode_input in raw_inputs:
            if isinstance(decode_input, dict):
                # Never attach a P snapshot or mutate the canonical prompt in
                # place; the early cache request and later decode must hash the
                # same prompt identity.
                decode_inputs.append(dict(decode_input))
                continue
            prompt_token_ids = getattr(decode_input, "prompt_token_ids", None)
            if prompt_token_ids is None:
                raise TypeError(
                    "[Orchestrator][PD] decode input must be dict or have prompt_token_ids, "
                    f"got {type(decode_input).__name__} for req={req_state.request_id}"
                )
            decode_inputs.append({"prompt_token_ids": list(prompt_token_ids)})
        return decode_inputs

    def _build_pd_early_cache_request(
        self,
        req_id: str,
        req_state: OrchestratorRequestState,
        *,
        engine_request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> OmniEngineCoreRequest | None:
        if self._pd_pair is None:
            return None
        _, d_stage = self._pd_pair
        engine_request_id = engine_request_id or req_id
        params = self._build_pd_early_cache_params(
            req_id,
            engine_request_id,
            req_state.sampling_params_list[d_stage],
        )
        if params is None:
            return None
        decode_inputs = (
            [{"prompt_token_ids": prompt_token_ids}]
            if prompt_token_ids is not None
            else self._pd_decode_inputs(req_state)
        )
        if prompt_token_ids is not None and isinstance(req_state.prompt, dict):
            # Predicted native P prompts contain AV placeholders. Dropping
            # their salt here would import another session's KV during early
            # registration even if the later formal D request is salted.
            if cache_salt := req_state.prompt.get("cache_salt"):
                decode_inputs[0]["cache_salt"] = cache_salt
        if len(decode_inputs) != 1:
            # The current Thinker path creates exactly one finite request.  Do
            # not silently pre-register only part of a batched prompt.
            return None
        pd_mrope_features = self._build_pd_mrope_features(req_state.pd_mrope_feature_metadata)
        request = build_engine_core_request_from_tokens(
            request_id=engine_request_id,
            prompt=decode_inputs[0],
            params=params,
            model_config=self.stage_pools[d_stage].stage_vllm_config.model_config,
            mm_features=pd_mrope_features,
        )
        request.external_req_id = request.request_id
        # Arrival-prefill requests end after P and only populate disposable D
        # prefix cache.  A formal query continues to D with the same finite
        # request id, so pin its imported blocks until that ADD is admitted.
        request.pd_cache_sync_retain = not (
            isinstance(req_state.prompt, dict) and req_state.prompt.get("prefill_only") is True
        )
        return request

    def _emit_tx_edge(
        self,
        *,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_pool: StagePool,
        request_id: str,
        tx_ms: float,
    ) -> None:
        """Emit per-edge transfer_tx_s + transfer_size_bytes histograms.

        ``tx_ms`` is the orchestrator-side wall-clock spent in ``next_pool.
        submit_*`` (serialize + queue submit to the receiving worker). Best-
        effort size_bytes left at 0 — orchestrator doesn't have a cheap handle
        on the serialized payload size; a follow-up can plumb that from the
        connector adapter.
        """
        if self._transfer_emitter is None:
            return
        to_replica = to_pool.get_bound_replica_id(request_id)
        if to_replica is None:
            return
        try:
            self._transfer_emitter.observe_size(from_stage, from_replica, to_stage, to_replica, 0)
            self._transfer_emitter.observe_tx_time(from_stage, from_replica, to_stage, to_replica, tx_ms / 1000.0)
        except Exception:
            logger.debug(
                "[Orchestrator] transfer_tx emit failed for edge %d->%d req=%s",
                from_stage,
                to_stage,
                request_id,
                exc_info=True,
            )

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int | None = None,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
        pd_cache_sync: bool = False,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = src_stage_id + 1
        next_pool = self.stage_pools[next_logical]
        next_client = next_pool.stage_client
        params = req_state.sampling_params_list[next_logical]
        source_outputs = [output]
        next_stage_resumable = is_streaming_session and not is_final_update
        already_submitted = self._next_stage_already_submitted(src_stage_id, req_state)
        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)
        _t_submit_start = _time.perf_counter()

        if next_pool.stage_type == "diffusion":
            # Gate: never dispatch with an incomplete CFG bundle. Checked
            # non-destructively BEFORE popping — a pop-then-redefer would lose
            # the partial outputs. all_companions_done is trustworthy here
            # because done-marking happens only after set_companion_output.
            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, output, src_stage_id)
                logger.info(
                    "[Orchestrator] req=%s: CFG companion outputs not all stashed yet; re-deferring parent",
                    req_id,
                )
                return
            # Peek, don't pop: outputs stay stashed until cleanup_parent so a
            # streaming re-submission bundles the complete set again rather
            # than an empty one.
            companion_outputs = self._cfg_tracker.get_companion_outputs(req_id)
            expected = len(self._cfg_tracker.get_companion_request_ids(req_id))
            if expected > len(companion_outputs):
                # Companions are done but outputs are missing — inconsistent
                # tracker state (should be unreachable with peek semantics).
                # Fail the request rather than degrade CFG conditioning.
                logger.error(
                    "[Orchestrator] req=%s: only %d/%d CFG companion outputs available; "
                    "failing the request instead of dispatching degraded CFG",
                    req_id,
                    len(companion_outputs),
                    expected,
                )
                await self.output_async_queue.put(
                    ErrorMessage(
                        request_id=req_id,
                        stage_id=src_stage_id,
                        error=(
                            f"CFG companion outputs incomplete ({len(companion_outputs)}/{expected}); "
                            "request aborted to avoid degraded CFG conditioning"
                        ),
                    )
                )
                await self._cleanup_request_ids(
                    [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                    abort=True,
                )
                return
            diffusion_source_outputs = [output, *companion_outputs]
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                _fn = next_client.custom_process_input_func
                _extra_kwargs: dict[str, Any] = {}
                # TODO: replace signature probe with explicit kwarg contract.
                try:
                    import inspect as _inspect

                    if "sampling_params" in _inspect.signature(_fn).parameters:
                        _extra_kwargs["sampling_params"] = params
                except (TypeError, ValueError):
                    pass
                diffusion_prompt = _fn(
                    diffusion_source_outputs,
                    req_state.prompt,
                    requires_multimodal_data,
                    **_extra_kwargs,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                req_state.pipeline_timings["ar2diffusion_ms"] = _dt_ar2d
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    src_stage_id,
                    next_logical,
                )
                if diffusion_prompt is None:
                    error_output = OmniRequestOutput.from_error(
                        req_id,
                        f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                    )
                    logger.warning(
                        "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                        "routing terminal error output",
                        req_id,
                        src_stage_id,
                        next_logical,
                    )
                    await self.output_async_queue.put(
                        OutputMessage(
                            request_id=req_id,
                            stage_id=next_logical,
                            engine_outputs=error_output,
                            metrics=None,
                            finished=True,
                        )
                    )
                    await self._cleanup_request_ids(
                        [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                    )
                    return
                if isinstance(diffusion_prompt, list):
                    if not diffusion_prompt:
                        error_output = OmniRequestOutput.from_error(
                            req_id,
                            f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                        )
                        logger.warning(
                            "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                            "routing terminal error output",
                            req_id,
                            src_stage_id,
                            next_logical,
                        )
                        await self.output_async_queue.put(
                            OutputMessage(
                                request_id=req_id,
                                stage_id=next_logical,
                                engine_outputs=error_output,
                                metrics=None,
                                finished=True,
                            )
                        )
                        await self._cleanup_request_ids(
                            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                        )
                        return
                    if len(diffusion_prompt) == 1:
                        diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, diffusion_prompt)
            else:
                replica_id = await next_pool.submit_initial(
                    req_id,
                    req_state,
                    diffusion_prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                            request_id=req_id,
                        )
                    },
                    params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                )
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
            )
            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (src_stage_id, next_logical) == self._pd_pair:
            native_duplex_pd = self._is_duplex_session_request(req_state)
            if native_duplex_pd:
                # P owns the persistent streaming state.  D is a finite
                # decode request per model unit so every unit can import the
                # newly published P prefix instead of locally recomputing the
                # media append through a resumable D request.
                next_stage_resumable = False
                already_submitted = False
            prepared_locally = bool(
                not native_duplex_pd
                and not pd_cache_sync
                and req_state.pd_early_cache_sync_result
                and req_state.pd_early_cache_sync_result.get("ok") is True
            )
            params = (
                self._build_pd_local_decode_params(req_id, params)
                if prepared_locally
                else self._build_pd_decode_params(req_id, params)
            )
            decode_engine_req_id = req_id
            if native_duplex_pd:
                bridge = req_state.streaming.bridge_states
                candidate = bridge.get("pd_decode_engine_request_id")
                if not isinstance(candidate, str) or not candidate:
                    raise RuntimeError(f"[Orchestrator][PD] native D request lacks a physical slot id for req={req_id}")
                decode_engine_req_id = candidate
                extra_args = getattr(params, "extra_args", None)
                kv_params = extra_args.get("kv_transfer_params") if isinstance(extra_args, dict) else None
                if isinstance(kv_params, dict):
                    kv_params["transfer_id"] = bridge.get(
                        "pd_decode_transfer_id",
                        f"xfer-{decode_engine_req_id}",
                    )

            # This token-only D request bypasses the ordinary input processor,
            # which normally installs tokenizer-derived EOS and stop metadata.
            # Mirror that initialization so D terminates at Qwen's chat EOS
            # instead of running to the configured max_tokens limit.
            decode_processor = self.stage_pools[next_logical].output_processor
            decode_tokenizer = getattr(decode_processor, "tokenizer", None)
            if isinstance(params, SamplingParams) and decode_tokenizer is not None:
                params.update_from_generation_config(
                    {},
                    getattr(decode_tokenizer, "eos_token_id", None),
                )
                params.update_from_tokenizer(decode_tokenizer)

            # Use the original user prompt for the decode stage (not processed embeddings).
            decode_inputs = self._pd_decode_inputs(req_state)

            pd_mrope_features = self._build_pd_mrope_features(req_state.pd_mrope_feature_metadata)
            expected_mrope_features = sum(
                getattr(feature, "modality", None) in ("image", "video", "audio")
                for feature in (req_state.mm_features or ())
            )
            if len(pd_mrope_features) != expected_mrope_features:
                raise RuntimeError(
                    "[Orchestrator][PD] incomplete M-RoPE metadata for decode "
                    f"req={req_id}: rebuilt={len(pd_mrope_features)} expected={expected_mrope_features}"
                )

            prefill_snapshot = req_state.pd_prefill_multimodal_output
            if not pd_cache_sync and isinstance(prefill_snapshot, dict):
                embeds = prefill_snapshot.get("embed")
                embeds = embeds if isinstance(embeds, dict) else {}

                hidden_layer = int(getattr(self, "_pd_snapshot_hidden_layer", 24))
                layer_0_chunks = self._pd_snapshot_layer_chunks(prefill_snapshot, 0)
                layer_hidden_chunks = self._pd_snapshot_layer_chunks(
                    prefill_snapshot,
                    hidden_layer,
                )
                if layer_0_chunks and layer_hidden_chunks:
                    for decode_input in decode_inputs:
                        prompt_ids = decode_input.get("prompt_token_ids") or []
                        available_rows = min(
                            sum(int(chunk.shape[0]) for chunk in layer_0_chunks),
                            sum(int(chunk.shape[0]) for chunk in layer_hidden_chunks),
                        )
                        if available_rows < len(prompt_ids):
                            logger.warning(
                                "[Orchestrator][PD] incomplete P snapshot req=%s rows=%d prompt_tokens=%d",
                                req_id,
                                available_rows,
                                len(prompt_ids),
                            )
                            continue
                        selected_layer_0 = self._slice_pd_snapshot_chunks(layer_0_chunks, len(prompt_ids))
                        selected_layer_hidden = self._slice_pd_snapshot_chunks(
                            layer_hidden_chunks,
                            len(prompt_ids),
                        )
                        decode_input["pd_prefill_payload"] = OmniPDPrefillPayload(
                            prompt_layer_0_chunks=self._ensure_shared_pd_snapshot_chunks(selected_layer_0),
                            # ``prompt_layer_24`` is the wire-compatible name
                            # for the model-selected Talker hidden layer. For
                            # DuplexOmni it carries final layer 48.
                            prompt_layer_24_chunks=self._ensure_shared_pd_snapshot_chunks(selected_layer_hidden),
                            tts_bos=embeds.get("tts_bos"),
                            tts_eos=embeds.get("tts_eos"),
                            tts_pad=embeds.get("tts_pad"),
                        )
                        logger.info(
                            "[Orchestrator][PD] attached P snapshot req=%s rows=%d prompt_tokens=%d",
                            req_id,
                            available_rows,
                            len(prompt_ids),
                        )
                else:
                    logger.warning(
                        "[Orchestrator][PD] P output lacks Talker conditioning layers for req=%s; "
                        "D can decode but audio handoff may be incomplete",
                        req_id,
                    )

            for decode_input in decode_inputs:
                request = build_engine_core_request_from_tokens(
                    request_id=decode_engine_req_id,
                    prompt=decode_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    mm_features=pd_mrope_features,
                    resumable=next_stage_resumable,
                )
                request.external_req_id = req_id
                if native_duplex_pd:
                    self._pd_decode_request_aliases[decode_engine_req_id] = req_id
                if pd_cache_sync:
                    req_state.stage_submit_ts[next_logical] = _time.time()
                    try:
                        replica_id, sync_result = await next_pool.submit_pd_cache_sync(
                            req_id,
                            request,
                        )
                    except Exception as exc:
                        req_state.pd_decode_cache_sync_pending = False
                        logger.exception(
                            "[Orchestrator][PD cache-sync] failed req=%s lineage=%s revision=%d",
                            req_id,
                            req_state.pd_prefill_lineage_id,
                            req_state.pd_prefill_revision,
                        )
                        if req_state.pd_prefill_ready_emitted:
                            # P-ready was already reported successfully. D's
                            # cache is disposable, and a later foreground
                            # request can transfer the cumulative suffix from
                            # D's last valid prefix. Do not emit a contradictory
                            # second terminal result for this request.
                            await self._cleanup_request_ids([req_id])
                            return
                        await self.output_async_queue.put(
                            ErrorMessage(
                                request_id=req_id,
                                stage_id=next_logical,
                                error=str(exc),
                                error_type="PDCacheSyncError",
                            )
                        )
                        await self._cleanup_request_ids([req_id])
                        return

                    req_state.pd_decode_cache_sync_pending = False
                    logger.info(
                        "[Orchestrator][PD cache-sync] completed req=%s lineage=%s revision=%d d_ms=%.3f full_hit=%s",
                        req_id,
                        req_state.pd_prefill_lineage_id,
                        req_state.pd_prefill_revision,
                        float(sync_result.get("cache_sync_ms", 0.0)) if isinstance(sync_result, dict) else 0.0,
                        bool(sync_result.get("full_hit", False)) if isinstance(sync_result, dict) else False,
                    )
                    if req_state.pd_prefill_ready_emitted:
                        await self._cleanup_request_ids([req_id])
                        return
                    final_stage_id = req_state.final_stage_id
                    final_pool = self.stage_pools[final_stage_id]
                    terminal_output = _build_terminal_empty_output(
                        req_id,
                        final_output_type=getattr(final_pool.stage_client, "final_output_type", None),
                        audio_sample_rate=final_pool._infer_audio_sample_rate(),
                    )
                    terminal_ts = _time.time()
                    req_state.stage_submit_ts[final_stage_id] = terminal_ts
                    await self.output_async_queue.put(
                        OutputMessage(
                            request_id=req_id,
                            stage_id=final_stage_id,
                            replica_id=replica_id,
                            engine_outputs=terminal_output,
                            metrics=None,
                            finished=True,
                            stage_submit_ts=terminal_ts,
                        )
                    )
                    await self._cleanup_request_ids([req_id])
                    return
                if already_submitted:
                    replica_id = await next_pool.submit_update(req_id, req_state, request)
                else:
                    replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
                self._record_duplex_stage_submission(
                    next_logical,
                    req_id,
                    replica_id,
                    req_state,
                )

            if native_duplex_pd:
                # D owns an independent serialized EngineCoreRequest after the
                # successful submit. Keep only the two scalars Talker needs;
                # retaining another full prompt in the session bridge would
                # add O(context) memory to every live session.
                req_state.streaming.bridge_states.pop("pd_decode_prompt", None)

            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        if req_state.pd_prefill_multimodal_output is not None:
            req_state.streaming.bridge_states.setdefault(
                "pd_prefill_multimodal_output_by_req",
                {},
            )[req_id] = req_state.pd_prefill_multimodal_output

        previous_decoder = req_state.streaming.source_token_decoder
        source_processor = self.stage_pools[src_stage_id].output_processor
        tokenizer = getattr(source_processor, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            req_state.streaming.source_token_decoder = decode

        try:
            next_inputs = next_client.process_engine_inputs(
                source_outputs,
                req_state.prompt,
                streaming_context=req_state.streaming,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_logical,
            )
            raise
        finally:
            req_state.streaming.source_token_decoder = previous_decoder

        if not next_inputs:
            if (
                self._pd_pair is not None
                and src_stage_id == self._pd_pair[1]
                and self._is_duplex_session_request(req_state)
                and req_state.streaming.segment_finished
            ):
                logger.debug(
                    "[Orchestrator] req=%s native D slot produced no Talker input; keeping duplex session alive",
                    req_id,
                )
                return
            if not getattr(output, "finished", False):
                logger.debug(
                    "[Orchestrator] req=%s stage-%s produced no inputs for stage-%s; waiting for more outputs",
                    req_id,
                    src_stage_id,
                    next_logical,
                )
                return

            final_stage_id = req_state.final_stage_id
            final_pool = self.stage_pools[final_stage_id]
            final_output_type = getattr(final_pool.stage_client, "final_output_type", None)
            terminal_output = _build_terminal_empty_output(
                req_id,
                final_output_type=final_output_type,
                audio_sample_rate=final_pool._infer_audio_sample_rate(),
            )
            submit_ts = _time.time()
            req_state.stage_submit_ts[final_stage_id] = submit_ts
            logger.info(
                "[Orchestrator] req=%s stage-%s produced no terminal inputs for stage-%s; "
                "returning empty %s output from final stage-%s",
                req_id,
                src_stage_id,
                next_logical,
                final_output_type or "text",
                final_stage_id,
            )
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=final_stage_id,
                    replica_id=0,
                    engine_outputs=terminal_output,
                    metrics=None,
                    finished=True,
                    stage_submit_ts=submit_ts,
                )
            )
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])
            return

        # Build and submit requests for each input
        for next_input in next_inputs:
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            model_stage = getattr(getattr(next_pool.stage_vllm_config, "model_config", None), "model_stage", None)
            mm_features = req_state.mm_features if model_stage == "thinker" else None
            request = self._build_next_stage_request(
                req_id,
                next_logical,
                next_input,
                params=params,
                mm_features=mm_features,
                resumable=next_stage_resumable,
            )

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, request)
            else:
                replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
            )

        req_state.stage_submit_ts[next_logical] = _time.time()
        _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
        self._emit_tx_edge(
            from_stage=src_stage_id,
            from_replica=src_replica_id if src_replica_id is not None else 0,
            to_stage=next_logical,
            to_pool=next_pool,
            request_id=req_id,
            tx_ms=_tx_ms,
        )

    def _schedule_pd_early_cache_sync(
        self,
        req_id: str,
        req_state: OrchestratorRequestState,
        *,
        engine_request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        """Pre-register D while the paired finite P request is running."""
        request = self._build_pd_early_cache_request(
            req_id,
            req_state,
            engine_request_id=engine_request_id,
            prompt_token_ids=prompt_token_ids,
        )
        if request is None or self._pd_pair is None:
            return

        task_key = engine_request_id or req_id
        tasks = self._pd_cache_sync_tasks
        if task_key in tasks and not tasks[task_key].done():
            raise RuntimeError(f"duplicate in-flight P/D cache preparation for request {task_key}")

        task = asyncio.create_task(
            self._run_pd_early_cache_sync(
                req_id,
                task_key,
                request,
                req_state,
            ),
            name=f"orchestrator-pd-early-cache-{task_key}",
        )
        req_state.pd_early_cache_sync_task = task
        tasks[task_key] = task

        def _discard(done: asyncio.Task[Any]) -> None:
            if tasks.get(task_key) is done:
                tasks.pop(task_key, None)

        task.add_done_callback(_discard)

    async def _run_pd_early_cache_sync(
        self,
        req_id: str,
        task_key: str,
        request: OmniEngineCoreRequest,
        req_state: OrchestratorRequestState,
    ) -> dict[str, Any]:
        assert self._pd_pair is not None
        _, d_stage = self._pd_pair
        started = _time.monotonic()
        try:
            registered = _time.monotonic()
            replica_id, result = await self.stage_pools[d_stage].submit_pd_cache_sync(
                task_key,
                request,
            )
            completed = _time.monotonic()
            normalized = dict(result) if isinstance(result, dict) else {"result": result}
            normalized.update(ok=True, replica_id=replica_id)
            req_state.pd_early_cache_sync_result = normalized
            log_ready = logger.info if _LOG_HANDOFF_DIAG else logger.debug
            log_ready(
                "[PD-EARLY-D] request=%s lineage=%s revision=%d submit_setup_ms=%.3f "
                "register_to_ready_ms=%.3f total_ms=%.3f full_hit=%s",
                req_id,
                req_state.pd_prefill_lineage_id,
                req_state.pd_prefill_revision,
                (registered - started) * 1000.0,
                (completed - registered) * 1000.0,
                (completed - started) * 1000.0,
                bool(normalized.get("full_hit", False)),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            req_state.pd_early_cache_sync_error = exc
            normalized = {"ok": False, "error": str(exc)}
            logger.warning(
                "[PD-EARLY-D] request=%s lineage=%s revision=%d failed after %.3fms: %s",
                req_id,
                req_state.pd_prefill_lineage_id,
                req_state.pd_prefill_revision,
                (_time.monotonic() - started) * 1000.0,
                exc,
            )
        finally:
            req_state.pd_decode_cache_sync_pending = False

        if req_state.pd_prefill_ready_emitted:
            await self._cleanup_request_ids([req_id])
        return normalized

    def _schedule_pd_cache_sync(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int,
    ) -> None:
        """Run one arrival cache sync without blocking global output polling."""
        tasks = getattr(self, "_pd_cache_sync_tasks", None)
        if tasks is None:
            # Some focused unit tests construct the orchestrator without
            # invoking __init__.
            tasks = self._pd_cache_sync_tasks = {}

        previous = tasks.get(req_id)
        if previous is not None and not previous.done():
            raise RuntimeError(f"duplicate in-flight P/D cache sync for request {req_id}")

        task = asyncio.create_task(
            self._run_pd_cache_sync(
                req_id,
                stage_id,
                output,
                req_state,
                src_replica_id=src_replica_id,
            ),
            name=f"orchestrator-pd-cache-sync-{req_id}",
        )
        tasks[req_id] = task

        def _discard(done: asyncio.Task[None]) -> None:
            if tasks.get(req_id) is done:
                tasks.pop(req_id, None)

        task.add_done_callback(_discard)

    async def _run_pd_cache_sync(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int,
    ) -> None:
        try:
            await self._forward_to_next_stage(
                req_id,
                stage_id,
                output,
                req_state,
                src_replica_id=src_replica_id,
                pd_cache_sync=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            req_state.pd_decode_cache_sync_pending = False
            logger.exception(
                "[Orchestrator][PD cache-sync] unexpected forwarding failure req=%s lineage=%s revision=%d",
                req_id,
                req_state.pd_prefill_lineage_id,
                req_state.pd_prefill_revision,
            )
            # An explicit abort may already have removed the request while the
            # control operation was completing.  Do not resurrect it with a
            # late error message.
            if self.request_states.get(req_id) is not req_state:
                return
            if req_state.pd_prefill_ready_emitted:
                # The client has already consumed the one P-ready terminal.
                # Treat background cache warming as best-effort and preserve
                # the last valid D prefix for cumulative recovery.
                await self._cleanup_request_ids([req_id], abort=True)
                return
            await self.output_async_queue.put(
                ErrorMessage(
                    request_id=req_id,
                    stage_id=stage_id + 1,
                    error=str(exc),
                    error_type="PDCacheSyncError",
                )
            )
            await self._cleanup_request_ids([req_id], abort=True)

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        pd_pair = getattr(self, "_pd_pair", None)
        for next_stage_id in range(1, req_state.final_stage_id + 1):
            if (
                _MINICPMO_PD_ONLY_DIAGNOSTIC
                and pd_pair is not None
                and self._is_duplex_session_request(req_state)
                and next_stage_id > pd_pair[1]
            ):
                # The D-sink control never consumes downstream outputs. Do not
                # even pre-submit persistent Talker/Code2Wav placeholders.
                continue
            if pd_pair is not None and next_stage_id == pd_pair[1] and self._is_duplex_session_request(req_state):
                # D is finite per slot and can only be admitted after P has
                # published the matching cumulative prefix.
                continue
            next_pool = self.stage_pools[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]
            if not self._stage_receives_async_chunks(next_stage_id):
                # Outgoing-only stages receive their first real input from the
                # orchestrator. Pre-submitting a placeholder lets it race and
                # execute before that conditioning payload arrives.
                continue

            req_state.stage_submit_ts[next_stage_id] = _time.time()
            _t_submit_start = _time.perf_counter()

            if next_pool.stage_type == "diffusion":
                replica_id = await next_pool.submit_initial(
                    request_id,
                    req_state,
                    req_state.prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_pool.stage_client, "engine_input_source", None) or [next_stage_id - 1]),
                            request_id=request_id,
                        )
                    },
                )
            else:
                import copy

                try:
                    from vllm_omni.distributed.omni_connectors.adapter import (
                        compute_talker_prompt_cache_ids,
                    )

                    talker_cache_ids = compute_talker_prompt_cache_ids(prompt_token_ids)
                except Exception:
                    # Exact source-token cache identities are currently
                    # available only for model families whose Talker prompt
                    # is a deterministic projection of the Thinker prompt.
                    # MiniCPM's native Talker is conditioned by generated
                    # hidden rows instead, so retain only the scheduler length
                    # and isolate the placeholder with the request cache salt.
                    try:
                        from vllm_omni.distributed.omni_connectors.adapter import (
                            compute_talker_prompt_ids_length,
                        )

                        talker_prompt_len = max(
                            1,
                            compute_talker_prompt_ids_length(prompt_token_ids),
                        )
                    except Exception:
                        talker_prompt_len = max(1, len(prompt_token_ids))
                    talker_cache_ids = [0] * talker_prompt_len

                original_prompt = req_state.prompt
                if isinstance(original_prompt, dict):
                    base_input = copy.deepcopy(original_prompt)
                else:
                    base_input = {}

                talker_prompt_len = max(1, len(talker_cache_ids))
                base_input["prompt_token_ids"] = [0] * talker_prompt_len
                base_input["cache_token_ids"] = talker_cache_ids or [0]
                # Thinker and Talker have different token/KV spaces. Never
                # leak the Thinker handle into the downstream stage.
                base_input.pop("kv_lineage_id", None)
                base_input.pop("kv_lineage_parent_revision", None)
                base_input.pop("kv_lineage_revision", None)
                base_input.pop("kv_lineage_prefix_tokens", None)
                # This is application-session isolation for a disposable
                # engine cache, not engine-owned session state.  A malformed
                # or non-live prompt gets a request-local salt so placeholder
                # fallbacks can never collide.
                talker_cache_salt = base_input.pop("talker_cache_salt", None)
                base_input["cache_salt"] = talker_cache_salt or f"talker-request:{request_id}"
                base_input["multi_modal_data"] = None
                base_input["mm_processor_kwargs"] = None
                downstream_resumable = bool(getattr(stage0_request, "resumable", req_state.streaming.enabled))
                request = build_engine_core_request_from_tokens(
                    request_id=request_id,
                    prompt=base_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    resumable=downstream_resumable,
                )
                request.external_req_id = request.request_id
                replica_id = await next_pool.submit_initial(
                    request_id,
                    req_state,
                    request,
                    prompt_text=None,
                )
            self._record_duplex_stage_submission(
                next_stage_id,
                request_id,
                replica_id,
                req_state,
            )

            # async_chunk pre-submit fires per stage edge (N-1 -> N). Source
            # replica is stage 0's bound replica (single-replica thinker in
            # all current configs); fall back to 0 if unknown.
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            src_replica = self.stage_pools[next_stage_id - 1].get_bound_replica_id(request_id)
            self._emit_tx_edge(
                from_stage=next_stage_id - 1,
                from_replica=src_replica if src_replica is not None else 0,
                to_stage=next_stage_id,
                to_pool=next_pool,
                request_id=request_id,
                tx_ms=_tx_ms,
            )

    def _build_kv_sender_info(
        self,
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(self.stage_pools):
                continue

            sender_pool = self.stage_pools[sender_stage_id]
            sender_stage = sender_pool.get_bound_client(request_id) if request_id is not None else None
            if sender_stage is None:
                sender_stage = sender_pool.stage_client
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    # ---- Shutdown / lifecycle ----

    async def _drain_pending_requests_on_fatal(self) -> None:
        """Drain the request queue and broadcast fatal errors for any
        pending add_request messages that were never processed.

        Called from the ``run()`` finally block when a fatal error
        (e.g. ``EngineDeadError``) caused the orchestrator to shut down
        before the request handler could process all queued messages.
        Also broadcasts for any already-tracked requests still in
        ``request_states`` that were not yet notified.
        """
        assert self._fatal_error is not None

        notified: set[str] = set()

        # 1) Drain pending messages from the request queue.
        while True:
            try:
                msg = self.request_async_queue.get_nowait()
            except Exception:
                break
            if msg.type == "add_request":
                req_id = msg.request_id
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
                notified.add(req_id)

        # 2) Broadcast for any tracked requests not already notified
        #    (e.g. request was registered but the EngineDeadError handler
        #    missed it because it wasn't submitted to the dead stage yet).
        for req_id in list(self.request_states):
            if req_id not in notified:
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
            self.request_states.pop(req_id, None)

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.live_num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            for replica_id in pool.live_replica_ids():
                pool.shutdown_replica(replica_id)
