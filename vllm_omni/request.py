from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_omni.engine import (
    AdditionalInformationPayload,
    OmniEngineCoreRequest,
    OmniPDPrefillPayload,
    PromptEmbedsPayload,
)


class OmniRequest(Request):
    """Request class for omni models, extending the base Request.

    This class extends the base vLLM Request with support for prompt
    embeddings and additional information payloads, enabling direct
    transfer of pre-computed embeddings between stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload.
            Used for direct transfer of embeddings between stages.
        additional_information: Optional additional information payload
            containing tensors or lists to be passed along with the request.
    """

    def __init__(
        self,
        *args,
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None = None,
        # Optional external request ID for tracking
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        model_intermediate_buffer: dict | None = None,
        pd_prefill_payload: OmniPDPrefillPayload | None = None,
        cache_token_ids: list[int] | None = None,
        prefill_only: bool = False,
        kv_lineage_id: str | None = None,
        kv_lineage_parent_revision: int = 0,
        kv_lineage_revision: int = 0,
        kv_lineage_prefix_tokens: int = 0,
        kv_lineage_snapshot_block_hashes: list[object] | None = None,
        kv_lineage_snapshot_num_computed_tokens: int = 0,
        kv_lineage_snapshot_hash_block_size: int = 0,
        **kwargs,
    ):
        self.cache_token_ids: list[int] | None = None
        self.kv_lineage_id = kv_lineage_id
        self.kv_lineage_parent_revision = kv_lineage_parent_revision
        self.kv_lineage_revision = kv_lineage_revision
        self.kv_lineage_prefix_tokens = max(0, kv_lineage_prefix_tokens)
        self.kv_lineage_seeded_tokens = 0
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = self._maybe_decode_prompt_embeds(prompt_embeds)

        block_hasher = kwargs.get("block_hasher")
        snapshot_found = bool(
            kv_lineage_snapshot_block_hashes
            and kv_lineage_snapshot_num_computed_tokens > 0
            and kv_lineage_snapshot_hash_block_size > 0
            and cache_token_ids is None
            and prompt_embeds is None
        )
        self.kv_lineage_snapshot_found = snapshot_found
        if not snapshot_found or block_hasher is None:
            super().__init__(*args, **kwargs)
        else:
            # Avoid hashing the known prefix in Request.__init__. Reinstall the
            # hasher after seeding, then hash only full blocks after the LCP.
            kwargs["block_hasher"] = None
            super().__init__(*args, **kwargs)
            self._block_hasher = block_hasher
            reusable_tokens = min(
                self.kv_lineage_prefix_tokens,
                kv_lineage_snapshot_num_computed_tokens,
                self.num_prompt_tokens,
            )
            reusable_blocks = min(
                len(kv_lineage_snapshot_block_hashes),
                reusable_tokens // kv_lineage_snapshot_hash_block_size,
            )
            # Upstream's incremental hasher treats a non-zero start as decode
            # extension and begins MM scanning at the final feature. That is
            # valid for one appended media item, but not when arrival
            # coalescing adds two or more items. Fall back to full hashing in
            # that uncommon case so no MM identifier can be skipped.
            reusable_boundary = reusable_blocks * kv_lineage_snapshot_hash_block_size
            tail_mm_features = sum(
                feature.mm_position.offset + feature.mm_position.length > reusable_boundary
                for feature in self.mm_features
            )
            if tail_mm_features > 1:
                reusable_blocks = 0
            if reusable_blocks:
                self.block_hashes.extend(kv_lineage_snapshot_block_hashes[:reusable_blocks])
                self.kv_lineage_seeded_tokens = reusable_blocks * kv_lineage_snapshot_hash_block_size
            self.update_block_hashes()
        if cache_token_ids is not None:
            if len(cache_token_ids) != self.num_prompt_tokens:
                raise ValueError(
                    "cache_token_ids must match the executable prompt length: "
                    f"{len(cache_token_ids)} != {self.num_prompt_tokens}"
                )
            self.cache_token_ids = list(cache_token_ids)
            # ``Request.__init__`` hashed the executable placeholders before
            # this Omni-only field was installed. Rebuild with the independent
            # conditioning lineage.
            self.block_hashes.clear()
            self.update_block_hashes()
        # Preserve serialized prompt embeddings payload (optional)
        self.prompt_embeds_payload: PromptEmbedsPayload | None = (
            prompt_embeds if isinstance(prompt_embeds, PromptEmbedsPayload) else None
        )
        # Optional external request ID for tracking
        self.external_req_id: str | None = external_req_id
        # Serialized additional information payload (optional)
        self.additional_information: AdditionalInformationPayload | None = additional_information
        # Runner-owned runtime payload.
        self.model_intermediate_buffer: dict | None = model_intermediate_buffer
        self.pd_prefill_payload = pd_prefill_payload
        self.prefill_only = prefill_only

    @staticmethod
    def _maybe_decode_prompt_embeds(
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(prompt_embeds, PromptEmbedsPayload):
            dtype = getattr(np, prompt_embeds.dtype)
            arr = np.frombuffer(prompt_embeds.data, dtype=dtype)
            arr = arr.reshape(prompt_embeds.shape)
            return torch.from_numpy(arr)
        return prompt_embeds

    @classmethod
    def from_engine_core_request(
        cls,
        request: OmniEngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        """Create an OmniRequest from an OmniEngineCoreRequest.

        Args:
            request: The OmniEngineCoreRequest to convert
            block_hasher: Optional function to compute block hashes for
                prefix caching

        Returns:
            OmniRequest instance created from the engine core request
        """
        return cls(
            request_id=request.request_id,
            # Optional external request ID for tracking
            external_req_id=request.external_req_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            additional_information=request.additional_information,
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
            pd_prefill_payload=getattr(request, "pd_prefill_payload", None),
            cache_token_ids=getattr(request, "cache_token_ids", None),
            prefill_only=getattr(request, "prefill_only", False),
            kv_lineage_id=getattr(request, "kv_lineage_id", None),
            kv_lineage_parent_revision=getattr(request, "kv_lineage_parent_revision", 0),
            kv_lineage_revision=getattr(request, "kv_lineage_revision", 0),
            kv_lineage_prefix_tokens=getattr(request, "kv_lineage_prefix_tokens", 0),
            kv_lineage_snapshot_block_hashes=getattr(request, "kv_lineage_snapshot_block_hashes", None),
            kv_lineage_snapshot_num_computed_tokens=getattr(request, "kv_lineage_snapshot_num_computed_tokens", 0),
            kv_lineage_snapshot_hash_block_size=getattr(request, "kv_lineage_snapshot_hash_block_size", 0),
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    def update_block_hashes(self) -> None:
        """Hash the conditioning lineage without changing executable ids."""
        cache_token_ids = getattr(self, "cache_token_ids", None)
        if cache_token_ids is None or self._block_hasher is None:
            super().update_block_hashes()
            return

        execution_ids = self._all_token_ids
        execution_view = self.all_token_ids
        cache_ids = [*cache_token_ids, *self._output_token_ids]
        self._all_token_ids = cache_ids
        self.all_token_ids = ConstantList(cache_ids)
        try:
            self.block_hashes.extend(self._block_hasher(self))
        finally:
            self._all_token_ids = execution_ids
            self.all_token_ids = execution_view


@dataclass
class OmniStreamingUpdate:
    """
    Override: add additional information
    Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    additional_information: AdditionalInformationPayload | None = None
    model_intermediate_buffer: dict | None = None

    @classmethod
    def from_request(cls, request: "Request") -> "OmniStreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
            additional_information=request.additional_information,
            model_intermediate_buffer=getattr(request, "model_intermediate_buffer", None),
        )
