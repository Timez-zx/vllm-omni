"""
Engine components for vLLM-Omni.
"""

from typing import Any

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Three supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
      - scalar: a Python scalar (msgspec-serializable)
    Exactly one of (tensor_data, list_data, scalar_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None

    # Scalar form
    scalar_data: Any | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniPDPrefillPayload(msgspec.Struct):
    """P-side prompt tensors needed by Talker after Thinker P/D split.

    KV transfer is sufficient for Thinker decode, but Talker also consumes the
    Thinker prompt embeddings and an intermediate hidden layer.  Keep those
    tensors in a typed payload so vLLM's tensor-aware request serialization is
    preserved without making them durable engine or session state.
    """

    # The outer EngineCoreRequest already carries the identical canonical
    # prompt. New P/D requests omit this duplicate list and the Talker bridge
    # falls back to request.prompt_token_ids; keep the field for wire
    # compatibility with older callers.
    prompt_token_ids: list[int] | None = None
    # Legacy contiguous form.  New local P/D requests use the chunk fields
    # below so the orchestrator can forward P's already-shared output buffers
    # without materializing another full-prompt copy.
    prompt_layer_0: torch.Tensor | None = None
    prompt_layer_24: torch.Tensor | None = None
    prompt_layer_0_chunks: tuple[torch.Tensor, ...] | None = None
    prompt_layer_24_chunks: tuple[torch.Tensor, ...] | None = None
    tts_bos: torch.Tensor | None = None
    tts_eos: torch.Tensor | None = None
    tts_pad: torch.Tensor | None = None


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for additional
    information payloads, enabling direct transfer of pre-computed data
    between pipeline stages.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.

    Attributes:
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None
    # Runner-owned runtime payload. This is materialized directly into
    # GPUModelRunner.model_intermediate_buffer instead of using the deprecated
    # additional_information request transport.
    model_intermediate_buffer: dict[str, Any] | None = None
    # Disposable P-side snapshot carried only by the paired finite D request.
    pd_prefill_payload: OmniPDPrefillPayload | None = None
    # Optional prompt identity used only by the KV block hasher.  Some Omni
    # stages execute placeholder token ids because real conditioning arrives
    # as embeddings after admission; their cache identity must not be coupled
    # to those executable placeholders.
    cache_token_ids: list[int] | None = None
    # Finish immediately after the prompt forward. Used by disposable cache
    # population requests; it is request policy, not persistent engine state.
    prefill_only: bool = False
    # Opaque, disposable KV lineage metadata.  The application still sends a
    # complete canonical prompt; these fields only let the engine reuse the
    # already-hashed prefix from a completed finite request.
    kv_lineage_id: str | None = None
    kv_lineage_parent_revision: int = 0
    kv_lineage_revision: int = 0
    kv_lineage_prefix_tokens: int = 0
    # Populated inside StageEngineCoreProc from its scheduler-owned registry;
    # never serialized by the application or treated as correctness state.
    kv_lineage_snapshot_block_hashes: list[bytes] | None = None
    kv_lineage_snapshot_num_computed_tokens: int = 0
    kv_lineage_snapshot_hash_block_size: int = 0
    # Keep a completed cache-only D import referenced until the paired finite
    # decode request reaches D's Core.  This closes the eviction window
    # between early P->D transfer completion and ordinary scheduler admission.
    pd_cache_sync_retain: bool = False

    @classmethod
    def from_request(
        cls,
        request: EngineCoreRequest,
        *,
        prompt_embeds: torch.Tensor | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        model_intermediate_buffer: dict[str, Any] | None = None,
        pd_prefill_payload: OmniPDPrefillPayload | None = None,
        prefill_only: bool | None = None,
        kv_lineage_id: str | None = None,
        kv_lineage_parent_revision: int | None = None,
        kv_lineage_revision: int | None = None,
        kv_lineage_prefix_tokens: int | None = None,
    ) -> "OmniEngineCoreRequest":
        """Clone an EngineCoreRequest into an OmniEngineCoreRequest with optional payload overrides."""

        if prompt_embeds is None:
            prompt_embeds = request.prompt_embeds
        if additional_information is None:
            additional_information = getattr(request, "additional_information", None)
        if model_intermediate_buffer is None:
            model_intermediate_buffer = getattr(request, "model_intermediate_buffer", None)
        if pd_prefill_payload is None:
            pd_prefill_payload = getattr(request, "pd_prefill_payload", None)
        if prefill_only is None:
            prefill_only = bool(getattr(request, "prefill_only", False))
        if kv_lineage_id is None:
            kv_lineage_id = getattr(request, "kv_lineage_id", None)
        if kv_lineage_parent_revision is None:
            kv_lineage_parent_revision = int(getattr(request, "kv_lineage_parent_revision", 0))
        if kv_lineage_revision is None:
            kv_lineage_revision = int(getattr(request, "kv_lineage_revision", 0))
        if kv_lineage_prefix_tokens is None:
            kv_lineage_prefix_tokens = int(getattr(request, "kv_lineage_prefix_tokens", 0))

        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            data_parallel_rank=request.data_parallel_rank,
            prompt_embeds=prompt_embeds,
            client_index=request.client_index,
            current_wave=request.current_wave,
            priority=request.priority,
            trace_headers=request.trace_headers,
            resumable=request.resumable,
            external_req_id=request.external_req_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
            additional_information=additional_information,
            model_intermediate_buffer=model_intermediate_buffer,
            pd_prefill_payload=pd_prefill_payload,
            cache_token_ids=getattr(request, "cache_token_ids", None),
            prefill_only=prefill_only,
            kv_lineage_id=kv_lineage_id,
            kv_lineage_parent_revision=kv_lineage_parent_revision,
            kv_lineage_revision=kv_lineage_revision,
            kv_lineage_prefix_tokens=kv_lineage_prefix_tokens,
            kv_lineage_snapshot_block_hashes=getattr(request, "kv_lineage_snapshot_block_hashes", None),
            kv_lineage_snapshot_num_computed_tokens=getattr(request, "kv_lineage_snapshot_num_computed_tokens", 0),
            kv_lineage_snapshot_hash_block_size=getattr(request, "kv_lineage_snapshot_hash_block_size", 0),
            pd_cache_sync_retain=bool(getattr(request, "pd_cache_sync_retain", False)),
        )


class OmniEngineCoreOutput(EngineCoreOutput):
    # Dedicated channel for multimodal outputs (image/audio/latent).
    # pooling_output is inherited from EngineCoreOutput as torch.Tensor | None
    # and retains its original vLLM semantics for pooling/embedding tasks.
    multimodal_output: dict[str, torch.Tensor] | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
