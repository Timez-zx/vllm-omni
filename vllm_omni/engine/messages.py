from __future__ import annotations

from typing import Literal

import msgspec
from vllm.inputs import PromptType
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.inputs.data import OmniInteractionPrompt, OmniSamplingParams
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.outputs import OmniRequestOutput


class EngineQueueMessage(msgspec.Struct, forbid_unknown_fields=True):
    pass


class StageSubmissionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["add_request", "streaming_update"]
    request_id: str
    prompt: EngineCoreRequest | PromptType
    original_prompt: EngineCoreRequest | PromptType
    output_prompt_text: object | None
    sampling_params_list: list[OmniSamplingParams]
    final_stage_id: int
    preprocess_ms: float
    request_timestamp: float
    enqueue_ts: float
    final_output_stage_ids: list[int] | None = None


class AddCompanionRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["add_companion_request"] = "add_companion_request"
    companion_id: str
    parent_id: str
    role: str
    prompt: EngineCoreRequest
    companion_prompt_text: object | None
    sampling_params_list: list[OmniSamplingParams]


class AbortRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["abort"] = "abort"
    request_ids: list[str]


class InteractionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["interaction"] = "interaction"
    request_id: str
    interaction: OmniInteractionPrompt


class CollectiveRPCRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["collective_rpc"] = "collective_rpc"
    rpc_id: str
    method: str
    timeout: float | None = None
    args: tuple[object, ...]
    kwargs: dict[str, object]
    stage_ids: list[int] | None


class ShutdownRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["shutdown"] = "shutdown"


class RegisterRemoteReplicaMessage(EngineQueueMessage, kw_only=True):
    type: Literal["register_remote_replica"] = "register_remote_replica"
    stage_id: int
    replica_id: int


class UnregisterRemoteReplicaMessage(EngineQueueMessage, kw_only=True):
    type: Literal["unregister_remote_replica"] = "unregister_remote_replica"
    stage_id: int
    input_addr: str


class ErrorMessage(EngineQueueMessage, kw_only=True):
    type: Literal["error"] = "error"
    error: str
    status_code: int | None = None
    error_type: str | None = None
    fatal: bool = False
    request_id: str | None = None
    stage_id: int | None = None
    event_id: str | None = None  # for interactions on diffusion generation requests


class OutputMessage(EngineQueueMessage, kw_only=True):
    type: Literal["output"] = "output"
    request_id: str
    stage_id: int
    replica_id: int | None = None
    engine_outputs: OmniRequestOutput
    metrics: StageRequestMetrics | None = None
    finished: bool
    stage_submit_ts: float | None = None


class StageMetricsMessage(EngineQueueMessage, kw_only=True):
    type: Literal["stage_metrics"] = "stage_metrics"
    request_id: str
    stage_id: int
    replica_id: int | None = None
    metrics: StageRequestMetrics
    stage_submit_ts: float | None = None


class PhysicalDCompletionWitnessMessage(EngineQueueMessage, kw_only=True, frozen=True):
    """Observer-only record for one completed physical duplex D request.

    This message contains scalars only.  It is routed to the owning duplex
    request queue, but must be removed before the model output projector.
    """

    type: Literal["physical_d_completion_witness"] = "physical_d_completion_witness"
    request_id: str
    stage_id: int
    replica_id: int | None = None
    engine_request_id: str
    physical_sequence: int
    input_unit_index: int
    source: Literal["real_input", "auto_continuation"]
    prompt_tokens: int
    cached_tokens: int
    local_cached_tokens: int
    external_cached_tokens: int
    computed_tokens: int
    batch_id: int
    submit_epoch_s: float
    completed_epoch_s: float
    service_ms: float
    input_video_frames: int
    arrival_video_frames: int
    vision_fallback_frames: int
    arrival_audio_units: int
    audio_fallback_units: int
    # Request-scoped NIXL delta evidence. ``selected_tokens`` is the exact
    # semantic suffix supplied by P and therefore matches
    # ``external_cached_tokens`` when both are available. ``selected_blocks``
    # and ``selected_bytes`` describe the block-granular WRITE (including its
    # final-block padding); bytes are aggregated across the D tensor-parallel
    # ranks. A negative value means the connector could not prove the scalar
    # without adding hot-path synchronization or a new wire round trip.
    kv_transfer_selected_blocks: int = -1
    kv_transfer_selected_tokens: int = -1
    kv_transfer_selected_bytes: int = -1
    kv_transfer_write_submit_to_d_ready_ms: float = -1.0


class CollectiveRPCResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["collective_rpc_result"] = "collective_rpc_result"
    rpc_id: str
    method: str
    stage_ids: list[int]
    results: list[object]

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return ("collective", self.rpc_id)
