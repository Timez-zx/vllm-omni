from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from numbers import Integral
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer, SlowIncrementalDetokenizer
from vllm.v1.engine.output_processor import OutputProcessor as VLLMOutputProcessor
from vllm.v1.engine.output_processor import (
    OutputProcessorOutput,
    RequestOutputCollector,
    RequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import IterationStats, RequestStateStats

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    release_native_segment_content,
    replace_snapshot_keys,
)
from vllm_omni.outputs.output_modality import OutputModality, get_accumulation_strategy

logger = init_logger(__name__)

_ENGINE_INPUT_AUDIT_KEYS = (
    "duplex_input_video_frames",
    "duplex_arrival_video_frames",
    "duplex_vision_fallback_frames",
    "duplex_arrival_audio_units",
    "duplex_audio_fallback_units",
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _modality_to_type_string(modality: OutputModality) -> str:
    """Convert an OutputModality flag to a lowercase type string."""
    try:
        if OutputModality.AUDIO in modality:
            return "audio"
        if OutputModality.IMAGE in modality:
            return "image"
        if OutputModality.LATENT in modality:
            return "latent"
    except TypeError:
        # Flag identity mismatch (e.g. after module reload in tests).
        name = getattr(modality, "name", "") or ""
        lowered = name.lower()
        if "audio" in lowered:
            return "audio"
        if "image" in lowered:
            return "image"
        if "latent" in lowered:
            return "latent"
    return "text"


def _mean_time_per_output_token_ms(stats: RequestStateStats) -> float:
    output_intervals = stats.num_generation_tokens - 1
    if output_intervals <= 0:
        return 0.0
    decode_time_s = max(stats.last_token_ts - stats.first_token_ts, 0.0)
    return decode_time_s * 1000.0 / float(output_intervals)


class OmniRequestState(RequestState):
    """Request state for omni models, tracking multimodal outputs.

    Extends the base RequestState with support for accumulating
    multimodal tensor outputs (e.g., images, audio, latents) that
    are produced incrementally during generation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # arrival_time is always passed as a keyword argument by the sole caller
        # (from_new_request).  The *args fallback via positional index was removed
        # because upstream RequestState.__init__ reorders parameters between
        # releases, making args[12] fragile.
        arrival_time = kwargs.get("arrival_time")
        super().__init__(*args, **kwargs)
        self.native_text_stats = RequestStateStats(arrival_time=float(arrival_time or 0.0))
        # Omni-specific: multimodal output accumulation
        # TODO: mm_type is per-request, not per-key. If a model ever produces
        # both audio and latent outputs, the modality type would flip on each
        # add_multimodal_tensor() call. Consider tracking per-key modality
        # types (e.g. dict[str, str]) for future multi-output models.
        self.mm_type: str | None = None
        self.mm_accumulated: MultimodalPayload = MultimodalPayload()
        # ``RequestOutput.request_id`` is deliberately the external/logical
        # request id.  A duplex session can therefore emit many finite engine
        # requests whose outputs all carry the same public id.  Keep the
        # current physical prefill snapshot on the request state so every
        # emitted output can expose an exact engine-completion identity.
        self.engine_prefill_stats: dict[str, int] | None = None
        self.engine_input_audit: dict[str, int] = {}

    def apply_streaming_update(self, update) -> None:
        super().apply_streaming_update(update)
        self.native_text_stats = RequestStateStats(arrival_time=float(update.arrival_time or 0.0))
        self.engine_prefill_stats = None
        self.engine_input_audit = {}

    def record_engine_prefill_stats(self, prefill_stats: Any | None) -> None:
        """Snapshot vLLM's prefill counters for the current physical input."""
        if prefill_stats is None:
            return
        fields = (
            "num_prompt_tokens",
            "num_computed_tokens",
            "num_cached_tokens",
            "num_local_cached_tokens",
            "num_external_cached_tokens",
            "num_cache_creation_tokens",
        )
        self.engine_prefill_stats = {
            name: int(getattr(prefill_stats, name, 0) or 0) for name in fields
        }

    def annotate_engine_request_output(
        self,
        request_output: RequestOutput | PoolingRequestOutput,
    ) -> RequestOutput | PoolingRequestOutput:
        """Attach non-accumulating physical-request metadata to an output."""
        request_output.engine_request_id = self.request_id
        request_output.engine_prompt_tokens = int(self.prompt_len)
        request_output.engine_num_cached_tokens = int(self.num_cached_tokens)
        request_output.engine_prefill_stats = (
            dict(self.engine_prefill_stats)
            if self.engine_prefill_stats is not None
            else None
        )
        request_output.engine_input_audit = dict(self.engine_input_audit)
        return request_output

    def record_engine_input_audit(self, payload: object) -> None:
        """Keep fixed-size counters from the current physical runner input."""
        if not isinstance(payload, Mapping):
            return
        candidate = payload.get("duplex")
        values = candidate if isinstance(candidate, Mapping) else payload
        for name in _ENGINE_INPUT_AUDIT_KEYS:
            value = self._coerce_nonnegative_scalar_count(values.get(name))
            if value is not None:
                self.engine_input_audit[name] = max(
                    self.engine_input_audit.get(name, 0),
                    value,
                )

    @staticmethod
    def _coerce_nonnegative_scalar_count(value: object) -> int | None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, Integral):
            return None
        result = int(value)
        return result if result >= 0 else None

    @staticmethod
    def _without_engine_input_audit(payload: object) -> object:
        """Remove observer-only scalars before multimodal accumulation/handoff."""
        if not isinstance(payload, Mapping):
            return payload
        cleaned = dict(payload)
        for name in _ENGINE_INPUT_AUDIT_KEYS:
            cleaned.pop(name, None)
        nested = cleaned.get("duplex")
        if isinstance(nested, Mapping):
            nested_cleaned = dict(nested)
            for name in _ENGINE_INPUT_AUDIT_KEYS:
                nested_cleaned.pop(name, None)
            if nested_cleaned:
                cleaned["duplex"] = nested_cleaned
            else:
                cleaned.pop("duplex", None)
        return cleaned

    def add_multimodal_tensor(self, payload: Any | None, mm_type: str | None) -> None:
        """Accumulate a multimodal tensor payload into the request state.

        Normalizes incoming payloads (dict or raw tensor) into a
        MultimodalPayload and merges with any previously accumulated data.
        Uses list-based deferred concatenation to avoid O(n²) repeated
        torch.cat calls.
        """
        if payload is None:
            return
        self.record_engine_input_audit(payload)
        payload = self._without_engine_input_audit(payload)
        if isinstance(payload, Mapping) and not payload:
            return
        try:
            if mm_type:
                self.mm_type = mm_type.lower()
            modality_key = self.mm_type or "hidden"

            incoming = MultimodalPayload.from_raw(payload, modality_key)
            if incoming is not None:
                replace_snapshot_keys(self.mm_accumulated, incoming)
                self.mm_accumulated = self.mm_accumulated.merged_with(incoming)
        except (ValueError, TypeError, RuntimeError):
            logger.exception("Error accumulating multimodal tensor")

    def _consolidate_multimodal_tensors(self) -> None:
        """Consolidate accumulated tensor lists into single tensors.

        Uses TensorAccumulationStrategy derived from the output modality
        to determine concatenation behavior. Metadata values always use
        REPLACE (keep latest).
        """
        if self.mm_accumulated.is_empty:
            return

        try:
            modality = OutputModality.from_string(self.mm_type)
        except (ValueError, KeyError):
            modality = OutputModality.TEXT
        strategy = get_accumulation_strategy(modality)

        try:
            self.mm_accumulated.consolidate_tensors(strategy)
            self.mm_accumulated.consolidate_metadata()
        except (RuntimeError, TypeError, KeyError):
            logger.exception("Error consolidating multimodal tensors")

    # Override: do not route to pooling-only path; always create completion
    # outputs, and attach pooling_result into the CompletionOutput.
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: Any = None,
    ) -> OmniRequestOutput | PoolingRequestOutput | None:
        """Create a request output from generation results.

        Creates a RequestOutput or PoolingRequestOutput from the generated
        tokens and accumulated multimodal outputs. Attaches multimodal
        tensors to the completion output if available.

        Args:
            new_token_ids: List of newly generated token IDs
            pooling_output: Optional pooling output tensor
            finish_reason: Optional finish reason indicating why generation stopped
            stop_reason: Optional stop reason (token ID or stop string)
            kv_transfer_params: Optional KV cache transfer parameters
            routed_experts: Optional MoE routed-expert ids for this step,
                attached to the completion output for generation stages

        Returns:
            OmniRequestOutput or PoolingRequestOutput if output should be
            emitted (based on finish status and output kind), None otherwise
        """
        # Pooling-only requests should follow base behavior.
        if self.detokenizer is None and pooling_output is not None:
            request_output = super().make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
            )
            return (
                self.annotate_engine_request_output(request_output)
                if request_output is not None
                else None
            )

        is_delta = self.output_kind == RequestOutputKind.DELTA
        if is_delta and finish_reason is not None and is_non_final_delta_audio_chunk(self.mm_accumulated, self.mm_type):
            finish_reason = None
            stop_reason = None

        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            return None

        # Consolidate accumulated tensors when finishing, or for every
        # CUMULATIVE step so the consumer always sees a single tensor.
        if finished or not is_delta:
            self._consolidate_multimodal_tensors()

        if self.stream_interval > 1 and self.detokenizer is not None:
            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[self.sent_tokens_offset :]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason, routed_experts)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self.annotate_engine_request_output(
            self._new_request_output(
                external_req_id,
                outputs,
                finished,
                kv_transfer_params,
            )
        )

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        routed_experts: Any = None,
    ) -> MultimodalCompletionOutput | CompletionOutput:
        """Create a completion output with multimodal data attached.

        Returns a MultimodalCompletionOutput when multimodal data has been
        accumulated, otherwise returns the base CompletionOutput.  Snapshots
        the accumulated payload before draining DELTA-mode modality keys so
        that callers receive an immutable view.
        """
        # When there is no detokenizer (generation stages), build a minimal
        # CompletionOutput directly since upstream asserts detokenizer != None.
        if self.detokenizer is None:
            finished = finish_reason is not None
            base_output = CompletionOutput(
                index=self.request_index,
                text="",
                token_ids=token_ids,
                logprobs=None,
                cumulative_logprob=None,
                finish_reason=str(finish_reason) if finished else None,
                stop_reason=stop_reason if finished else None,
                routed_experts=routed_experts,
            )
        else:
            base_output = super()._new_completion_output(token_ids, finish_reason, stop_reason)

        # Always provide cumulative token IDs for inter-stage processors.
        if self.detokenizer is not None:
            base_output.cumulative_token_ids = list(self.detokenizer.output_token_ids)
        else:
            base_output.cumulative_token_ids = list(token_ids)

        # Attach cumulative_text only at the final step for inter-stage use.
        if finish_reason is not None and hasattr(self.detokenizer, "output_text"):
            base_output.cumulative_text = self.detokenizer.output_text

        try:
            if self.mm_accumulated and not self.mm_accumulated.is_empty:
                # Snapshot: copy current tensors/metadata so drain doesn't
                # mutate the payload already handed to the caller.
                # Unflatten dotted keys (e.g. "hidden_states.layer_0") back
                # to nested dicts so downstream consumers (thinker2talker etc.)
                # can access with .get("hidden_states", {}).get("layers", {}).
                merged = {**self.mm_accumulated.tensors, **self.mm_accumulated.metadata}
                nested = unflatten_payload(merged)
                snapshot = MultimodalPayload.from_dict(nested) or MultimodalPayload()
                kwargs = {f.name: getattr(base_output, f.name) for f in dataclass_fields(CompletionOutput)}
                output = MultimodalCompletionOutput(
                    multimodal_output=snapshot,
                    **kwargs,
                )
                output.cumulative_token_ids = base_output.cumulative_token_ids
                if hasattr(base_output, "cumulative_text"):
                    output.cumulative_text = base_output.cumulative_text

                # DELTA mode: drain modality keys (e.g. audio) so the next
                # step only sees freshly accumulated data for those keys and
                # their client-facing per-chunk metadata.
                if self.output_kind == RequestOutputKind.DELTA:
                    drain_delta_payload(self.mm_accumulated)

                return output
        except (RuntimeError, TypeError, AttributeError):
            logger.exception("Error creating MultimodalCompletionOutput")
        return base_output

    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list,
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        """Create request output, handling no-detokenizer generation stages.

        Upstream asserts ``self.logprobs_processor is not None`` which fails
        for generation stages (talker, code2wav) that have no tokenizer.
        When the logprobs processor is absent we build the RequestOutput
        directly with ``prompt_logprobs=None``.
        """
        if self.logprobs_processor is not None:
            return super()._new_request_output(
                external_req_id,
                outputs,
                finished,
                kv_transfer_params,
            )

        # No-detokenizer path: build RequestOutput directly.
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        if prompt_token_ids is None:
            prompt_token_ids = []

        return RequestOutput(
            request_id=external_req_id,
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            metrics=self.stats,
        )


class MultimodalOutputProcessor(VLLMOutputProcessor):
    """Handles multimodal output processing.

    Captures multimodal outputs from OmniEngineCoreOutput and accumulates
    them as MultimodalPayload in OmniRequestState, before delegating to
    the base vLLM OutputProcessor for text handling.

    The data flow is:
    1. For each EngineCoreOutput with multimodal_output:
       - Capture into OmniRequestState.add_multimodal_tensor()
    2. Base vLLM OutputProcessor handles text detokenization
    3. On finish, _consolidate_multimodal_tensors() concatenates accumulated
       tensors using strategy-based dispatch
    4. _new_completion_output() returns MultimodalCompletionOutput
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
        engine_core_output_type: str | None = None,
        output_modality: OutputModality = OutputModality.TEXT,
    ):
        """Initialize the multimodal output processor.

        Args:
            tokenizer: Tokenizer for detokenizing text outputs
            log_stats: Whether to log statistics
            stream_interval: Stream interval for output generation
            engine_core_output_type: Optional output type string (e.g.,
                "image", "audio", "latent"). Converted to OutputModality
                internally. Kept for backward compatibility with
                stage_init_utils.
            output_modality: Type-safe output modality flag. Used to tag
                multimodal outputs with the correct modality key when
                per-output type info is unavailable.
        """
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
            tracing_enabled=tracing_enabled,
        )
        # Convert string-based engine_core_output_type to OutputModality
        if engine_core_output_type is not None:
            self.output_modality = OutputModality.from_string(engine_core_output_type)
        else:
            self.output_modality = output_modality
        self.engine_core_output_type = engine_core_output_type
        self._native_text_metrics_by_request: dict[str, dict[str, Any]] = {}

    def _native_text_metric_record(self, request_id: str) -> dict[str, Any]:
        return self._native_text_metrics_by_request.setdefault(
            request_id,
            {
                "num_generation_tokens": 0,
                "vllm_ttft_ms": 0.0,
                "vllm_queue_ms": 0.0,
                "vllm_prefill_ms": 0.0,
                "vllm_tpot_ms": 0.0,
                "vllm_itl_ms": 0.0,
                "vllm_itls_ms": [],
            },
        )

    def pop_native_text_metrics(self, request_id: str) -> dict[str, Any]:
        return self._native_text_metrics_by_request.pop(request_id, {})

    def abort_requests(self, request_ids, internal: bool) -> list[str]:
        request_ids = list(request_ids)
        for request_id in request_ids:
            if internal:
                req_state = self.request_states.get(request_id)
                if req_state is not None:
                    self._native_text_metrics_by_request.pop(req_state.external_req_id, None)
            else:
                self._native_text_metrics_by_request.pop(request_id, None)
        return super().abort_requests(request_ids, internal)

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        """Add a new request to be processed.

        Creates an OmniRequestState for the request and registers it
        for output processing.

        Args:
            request: Engine core request to add
            prompt: Optional prompt string for the request
            parent_req: Optional parent request for parallel sampling
            request_index: Index of the request in the batch
            queue: Optional queue for collecting outputs

        Raises:
            ValueError: If the request ID is already registered
        """
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = OmniRequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        buffer = getattr(request, "model_intermediate_buffer", None)
        if (
            isinstance(buffer, dict)
            and isinstance(buffer.get("duplex"), dict)
            and isinstance(req_state.detokenizer, FastIncrementalDetokenizer)
        ):
            # Native DecodeStream primes from the full logical prompt on its
            # first step, even when GPU KV is a bounded sliding window. Reuse
            # vLLM's bounded-prefix incremental path for recurring duplex
            # requests; preserve text, token counts and stop-string handling.
            req_state.detokenizer = SlowIncrementalDetokenizer(self.tokenizer, request)
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req
        self.external_req_ids[req_state.external_req_id].append(request_id)

    def remove_request(self, request_id: str) -> None:
        """Rollback one previously registered request if it was never submitted."""
        req_state = self.request_states.pop(request_id, None)
        if req_state is None:
            return

        external_req_id = getattr(req_state, "external_req_id", None)
        if external_req_id is not None:
            self._native_text_metrics_by_request.pop(external_req_id, None)
            request_ids = self.external_req_ids.get(external_req_id)
            if request_ids is not None:
                self.external_req_ids[external_req_id] = [rid for rid in request_ids if rid != request_id]
                if not self.external_req_ids[external_req_id]:
                    self.external_req_ids.pop(external_req_id, None)

        parent_req = getattr(req_state, "parent_req", None)
        if parent_req is not None:
            self.parent_requests.pop(parent_req.request_id, None)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        default_mm_type = _modality_to_type_string(self.output_modality)

        # Separate outputs that upstream can handle (has detokenizer or
        # pooling) from multimodal-only outputs (no detokenizer, no pooling)
        # that would trigger upstream's `assert detokenizer is not None`.
        upstream_outputs: list[EngineCoreOutput] = []
        mm_only_outputs: list[EngineCoreOutput] = []
        completed_native_states: list[OmniRequestState] = []

        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None:
                continue

            # Accumulate multimodal tensors regardless of path.
            if isinstance(req_state, OmniRequestState):
                mm_output = getattr(eco, "multimodal_output", None)
                if mm_output is not None:
                    mm_type = getattr(eco, "output_type", None) or default_mm_type
                    req_state.add_multimodal_tensor(mm_output, mm_type)
                if (
                    getattr(eco, "is_segment_finished", False)
                    and "duplex_segment_token_ids" in req_state.mm_accumulated
                ):
                    completed_native_states.append(req_state)

            # Route: if no detokenizer and no pooling output, handle locally
            # to avoid upstream's assert on detokenizer.
            if req_state.detokenizer is None and eco.pooling_output is None:
                mm_only_outputs.append(eco)
            else:
                upstream_outputs.append(eco)

        # Handle multimodal-only outputs (generation stages) locally.
        mm_request_outputs = self._process_mm_only_outputs(mm_only_outputs)

        # Delegate text/pooling outputs to upstream.
        processed = super().process_outputs(
            upstream_outputs,
            engine_core_timestamp=engine_core_timestamp,
            iteration_stats=iteration_stats,
        )
        processed.request_outputs.extend(mm_request_outputs)
        for state in completed_native_states:
            release_native_segment_content(state.mm_accumulated)
        return processed

    def _process_mm_only_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
    ) -> list[OmniRequestOutput | PoolingRequestOutput]:
        """Handle outputs from generation stages that have no detokenizer.

        These cannot go through upstream process_outputs because it asserts
        detokenizer is not None when pooling_output is None.
        """
        request_outputs: list[OmniRequestOutput | PoolingRequestOutput] = []
        for eco in engine_core_outputs:
            req_state = self.request_states.get(eco.request_id)
            if req_state is None or not isinstance(req_state, OmniRequestState):
                continue

            new_token_ids = eco.new_token_ids
            finish_reason = eco.finish_reason
            stop_reason = eco.stop_reason
            kv_transfer_params = eco.kv_transfer_params
            routed_experts = eco.routed_experts
            prefill_stats = getattr(eco, "prefill_stats", None)
            req_state.record_engine_prefill_stats(prefill_stats)
            if prefill_stats is not None:
                req_state.num_cached_tokens = int(prefill_stats.num_cached_tokens)
                req_state.num_cache_creation_tokens = int(
                    prefill_stats.num_cache_creation_tokens
                )
            else:
                req_state.num_cached_tokens = int(
                    getattr(eco, "num_cached_tokens", 0) or 0
                )
            req_state.is_prefilling = False

            is_non_final_audio_chunk = (
                finish_reason is not None
                and req_state.output_kind == RequestOutputKind.DELTA
                and is_non_final_delta_audio_chunk(req_state.mm_accumulated, req_state.mm_type)
            )
            if request_output := req_state.make_request_output(
                new_token_ids,
                None,  # pooling_output
                finish_reason,
                stop_reason,
                kv_transfer_params,
                routed_experts,
            ):
                if req_state.queue is not None:
                    req_state.queue.put(request_output)
                else:
                    request_outputs.append(request_output)

            is_segment_finished = bool(getattr(eco, "is_segment_finished", False))
            if finish_reason is not None and not is_segment_finished and not is_non_final_audio_chunk:
                self._finish_request(req_state)
        return request_outputs

    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        was_prefilling = req_state.is_prefilling
        native_stats = req_state.native_text_stats if isinstance(req_state, OmniRequestState) else None
        previous_last_token_ts = native_stats.last_token_ts if native_stats is not None else 0.0
        if isinstance(req_state, OmniRequestState):
            req_state.record_engine_prefill_stats(
                getattr(engine_core_output, "prefill_stats", None)
            )

        # NOTE: We pass ``None`` for  *iteration_stats* to the parent so that
        # the upstream's ``_update_stats_from_output`` logs stats via its own
        # ``RequestStateStats`` (req_state.stats) without touching the omni
        # ``IterationStats`` object.  Below we re-apply the same update using
        # *native_stats* (the omni-specific ``RequestStateStats`` attached to
        # ``OmniRequestState``), which correctly feeds the omni metrics path.
        # If the upstream adds new logic in ``_update_stats_from_output`` (e.g.
        # additional stat fields), this override must be kept in sync.
        super()._update_stats_from_output(
            req_state,
            engine_core_output,
            engine_core_timestamp,
            None,
        )

        if iteration_stats is None or engine_core_timestamp is None or native_stats is None:
            return

        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            was_prefilling,
            native_stats,
            self.lora_states,
            req_state.lora_name,
        )
        record = self._native_text_metric_record(req_state.external_req_id)
        record["num_generation_tokens"] = int(native_stats.num_generation_tokens)
        if was_prefilling:
            record["vllm_ttft_ms"] = max(float(native_stats.first_token_latency) * 1000.0, 0.0)
            if native_stats.queued_ts > 0 and native_stats.scheduled_ts > 0:
                record["vllm_queue_ms"] = max(
                    float(native_stats.scheduled_ts - native_stats.queued_ts) * 1000.0,
                    0.0,
                )
            if native_stats.scheduled_ts > 0 and native_stats.first_token_ts > 0:
                record["vllm_prefill_ms"] = max(
                    float(native_stats.first_token_ts - native_stats.scheduled_ts) * 1000.0,
                    0.0,
                )
            return

        if previous_last_token_ts > 0:
            itl_ms = max((engine_core_timestamp - previous_last_token_ts) * 1000.0, 0.0)
            itls_ms = record.setdefault("vllm_itls_ms", [])
            itls_ms.append(itl_ms)
            record["vllm_itl_ms"] = sum(itls_ms) / float(len(itls_ms))
        record["vllm_tpot_ms"] = _mean_time_per_output_token_ms(native_stats)

    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        super()._update_stats_from_finished(req_state, finish_reason, None)

        native_stats = req_state.native_text_stats if isinstance(req_state, OmniRequestState) else None
        if iteration_stats is None or native_stats is None or finish_reason is None:
            return

        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=native_stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        if not iteration_stats.finished_requests:
            return

        finished_request = iteration_stats.finished_requests[-1]
        if finished_request.request_id != req_state.external_req_id:
            return
        self._native_text_metric_record(req_state.external_req_id)["vllm_tpot_ms"] = (
            float(finished_request.mean_time_per_output_token) * 1000.0
        )
