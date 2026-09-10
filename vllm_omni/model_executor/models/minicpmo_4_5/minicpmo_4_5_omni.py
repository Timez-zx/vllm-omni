# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
#
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import cached_property
from threading import Lock
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.minicpmo45.sampling_state import (
    SAMPLING_STATE_KEY,
    SAMPLING_STATE_WIRE_KEY,
    pack_sampling_state,
    resume_sampling_rng,
    snapshot_sampling_rng,
    unpack_sampling_state,
)
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplerOutput, DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMDummyInputsBuilder,
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_MINICPMO45_BATCHED_VISION_KEY = "_minicpmo45_batched_vision"
_MINICPMO45_LOG_PREP_DIAG = os.environ.get("MINICPMO45_LOG_PREP_DIAG", "0") not in ("0", "", "false", "False")


@dataclass
class _MiniCPMFilteredDraw:
    logits: torch.Tensor
    top_k: int
    top_p: float
    generator: torch.Generator | None
    token_ids: dict[str, int] | None = None
    repetition_tokens: tuple[int, ...] = ()
    temperature: float = 1.0
    greedy: bool = False


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMO45OmniLLMMultiModalProcessor,
    info=MiniCPMO45OmniLLMProcessingInfo,
    dummy_inputs=MiniCPMO45OmniLLMDummyInputsBuilder,
)
class MiniCPMO45OmniForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """MiniCPM-o 4.5 Omni model for conditional generation.

    Three-stage pipeline:
    - thinker (model_stage="llm"): image / video / audio encoders + 3D
      resampler + the omni LLM that emits text + hidden states.
    - talker  (model_stage="tts"): native continuous MiniCPMTTS AR that emits
      codec-token deltas for the separate Code2Wav stage.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"
        if modality.startswith("audio"):
            return "(<audio>./</audio>)"
        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Audio, vision and P can enter the lazy runtime concurrently. Publish
        # exactly one helper, otherwise a later constructor discards the first
        # encoder's arrival cache and the first frame is encoded again on P.
        self._minicpmo45_helper_init_lock = Lock()
        self.have_multimodal_outputs = True
        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Store configs
        self.config = config
        self.multimodal_config = multimodal_config
        self._minicpmo_pd_prefill = bool(getattr(config, "vllm_omni_minicpmo_pd_prefill", False))
        self._minicpmo_pd_decode = bool(getattr(config, "vllm_omni_minicpmo_pd_decode", False))
        # Native P/D transfers the reusable prompt through the model KV cache.
        # P has no downstream tensor consumer. D->Talker needs the hidden rows
        # produced by every finite decode step, but only for that step: the
        # output processor accumulates those scheduled tails into one segment.
        # Keep D's generic ``hidden`` payload enabled for that purpose while
        # disabling the separate O(context) CPU tensor cache on both stages.
        # Native KV prefix caching remains enabled and authoritative for model
        # execution.
        self._minicpmo_pd_thinker = self._minicpmo_pd_prefill or self._minicpmo_pd_decode
        self._minicpmo45_numerical_probe_dir = os.environ.get("MINICPMO45_NUMERICAL_PROBE_DIR")
        self.omni_pooler_payload_include_hidden = self._minicpmo_pd_decode or not self._minicpmo_pd_thinker
        self.requires_full_prefix_cached_hidden_states = not self._minicpmo_pd_thinker
        self.requires_full_prefix_cached_multimodal_outputs = not self._minicpmo_pd_thinker
        from vllm_omni.experimental.fullduplex.minicpmo45.compat import (
            patch_minicpmo_remote_config,
        )

        patch_minicpmo_remote_config(config)

        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "llm":
            # Initialize thinker model (image preprocessing + vision encoder + 3D resampler)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniLLMForConditionalGeneration"],
            )
            self.model = self.thinker
            if self._minicpmo_pd_thinker:
                # Some runner paths query the inner registered model rather
                # than this pipeline wrapper.
                self.thinker.omni_pooler_payload_include_hidden = self._minicpmo_pd_decode
                self.thinker.requires_full_prefix_cached_hidden_states = False
                self.thinker.requires_full_prefix_cached_multimodal_outputs = False
            self.talker = None

        elif self.model_stage == "tts":
            self.thinker = None
            # The Talker is always the runner-owned continuous codec producer.
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=config,
                # Use registry architecture key
                architectures=["MiniCPMO45OmniTTSForConditionalGeneration"],
            )
            # Initialize multimodal components if needed
            if hasattr(self.talker, "init_multi_modal"):
                self.talker.init_multi_modal(config)
            self.model = self.talker
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}. Must be one of: 'llm', 'tts'")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors)
            if self.model_stage == "llm" and self.thinker is not None
            else self.talker.make_empty_intermediate_tensors
            if self.talker is not None
            else lambda: None
        )

        self._language_model_names = ["model"]
        self.prefer_model_sampler = self.model_stage in {"llm", "tts"}
        # Both AR stages require model-specific embeddings.  The Thinker uses
        # preprocess for duplex audio, while the Talker converts the
        # tts_token_ids/tts_hidden_states handoff into its conditioning
        # embeddings and initializes request-local codec generation state.
        self.has_preprocess = self.model_stage in {"llm", "tts"}

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        from vllm.v1.sample.sampler import Sampler

        return Sampler()

    def prepare_duplex_sampling(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        rows: tuple[DuplexSamplingRow, ...],
    ) -> None:
        """Apply MiniCPM duplex policy before the standard model sampler."""
        del sampling_metadata
        self._minicpmo45_active_duplex_rows = [row.row_idx for row in rows]
        self._minicpmo45_sampling_rows = rows
        self._minicpmo45_duplex_row_requests = {row.row_idx: row.request_id for row in rows}
        self._minicpmo45_duplex_row_sessions = {
            row.row_idx: (row.session_id, row.incarnation) for row in rows if row.session_id is not None
        }
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        if not isinstance(request_sessions, dict):
            request_sessions = {}
            self._minicpmo45_duplex_request_sessions = request_sessions
        request_sessions.update(
            {row.request_id: (row.session_id, row.incarnation) for row in rows if row.session_id is not None}
        )
        self._minicpmo45_duplex_row_payloads = {row.row_idx: row.payload for row in rows if row.payload is not None}
        self._minicpmo45_duplex_row_max_tokens = {
            row.row_idx: row.max_tokens for row in rows if row.max_tokens is not None
        }
        if self.model_stage != "llm" or not rows or logits.ndim != 2:
            return

        if getattr(self, "_minicpmo_pd_decode", False):
            states = getattr(self, "_minicpmo45_pd_sampling_states", None)
            if states is None:
                states = self._minicpmo45_pd_sampling_states = {}
            for row in rows:
                if not row.should_sample:
                    continue
                if row.request_id not in states:
                    payload = row.payload or {}
                    state, identity = unpack_sampling_state(payload.get(SAMPLING_STATE_KEY))
                    if identity != (row.incarnation, row.epoch, row.seq):
                        raise ValueError("Stale MiniCPM P/D sampling state identity")
                    states[row.request_id] = state
            # The P policy has already applied explicit force-listen and sampled
            # the first token. D must continue it, not force LISTEN a second time.
            return

        token_ids = self._minicpmo45_native_duplex_token_ids()
        listen_id = int(token_ids.get("listen_token_id", -1))
        if listen_id < 0 or listen_id >= logits.shape[-1]:
            return

        force_listen_segments = getattr(
            self,
            "_minicpmo45_force_listen_applied_segments",
            None,
        )
        if not isinstance(force_listen_segments, set):
            force_listen_segments = set()
            self._minicpmo45_force_listen_applied_segments = force_listen_segments
        for row in rows:
            if not row.should_sample:
                continue
            row_idx = row.row_idx
            if row_idx < 0 or row_idx >= logits.shape[0]:
                continue
            payload = row.payload
            if not isinstance(payload, dict):
                continue
            state = self._minicpmo45_duplex_state_for_row(row_idx)
            if getattr(state, "current_segment_output_tokens", None):
                # Explicit force-listen is evaluated once per unit, not again
                # when TURN_EOS occurs inside that unit.
                continue
            force_listen = payload.get("force_listen") is True
            segment_key = (row.request_id, row.seq if row.seq is not None else -1)
            # The native model may initiate a reply over silence. An input RMS
            # tag must neither force LISTEN nor overwrite its previous control
            # decision; only an explicit caller/model-config instruction may.
            if not force_listen:
                continue
            if force_listen and segment_key in force_listen_segments:
                continue
            if force_listen:
                logits[row_idx, :] = float("-inf")
                logits[row_idx, listen_id] = 0.0
                force_listen_segments.add(segment_key)

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move the thinker and talker to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        embed_fn = getattr(self.model, "get_input_embeddings", None)
        if callable(embed_fn):
            try:
                return embed_fn(input_ids, multimodal_embeddings)
            except TypeError:
                embeddings = embed_fn()
                if callable(embeddings):
                    return embeddings(input_ids)
            except AttributeError:
                pass

        embed_tokens = getattr(getattr(getattr(self.model, "llm", None), "model", None), "embed_tokens", None)
        if callable(embed_tokens):
            return embed_tokens(input_ids)

        raise AttributeError(f"{type(self.model).__name__} does not expose token embeddings")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "tts":
            return self.get_input_embeddings(input_ids)
        return super().embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    @staticmethod
    def _slice_duplex_prompt_delta(
        delta_embeds: torch.Tensor,
        delta_token_ids: list[int],
        *,
        prompt_len: int,
        token_offset: int,
        span_len: int,
        rebase_prompt: bool,
        scheduler_token_budget: object,
    ) -> tuple[torch.Tensor, list[int], int]:
        """Return the exact scheduler span from one Stage0-owned suffix.

        Steady duplex appends retain their prefix exclusively in engine KV.
        Materializing placeholder embeddings for that prefix is both
        O(context) and semantically wrong after a cache miss.  A first append
        or explicit rollover is a rebase and therefore owns the whole prompt.
        """
        delta_len = int(delta_embeds.shape[0])
        if delta_len != len(delta_token_ids):
            raise RuntimeError(
                f"MiniCPM-o duplex embedding/token row mismatch: embeddings={delta_len}, tokens={len(delta_token_ids)}"
            )
        if prompt_len < 0 or token_offset < 0 or span_len < 0:
            raise RuntimeError(
                "MiniCPM-o duplex received negative scheduler coordinates: "
                f"prompt_len={prompt_len}, token_offset={token_offset}, span_len={span_len}"
            )

        if rebase_prompt:
            if prompt_len != delta_len:
                raise RuntimeError(
                    "MiniCPM-o duplex rebase must materialize the complete prompt: "
                    f"scheduler={prompt_len}, embeddings={delta_len}"
                )
            delta_start = 0
        else:
            if isinstance(scheduler_token_budget, bool) or not isinstance(scheduler_token_budget, int):
                raise RuntimeError("MiniCPM-o duplex steady append requires an exact scheduler token budget")
            if scheduler_token_budget != delta_len:
                raise RuntimeError(
                    "MiniCPM-o duplex append scheduler budget mismatch: "
                    f"scheduler={scheduler_token_budget}, embeddings={delta_len}; "
                    "refusing to pad or truncate"
                )
            delta_start = prompt_len - delta_len
            if delta_start < 0:
                raise RuntimeError(
                    "MiniCPM-o duplex append is longer than the scheduler prompt: "
                    f"prompt={prompt_len}, delta={delta_len}"
                )

        if token_offset < delta_start:
            raise RuntimeError(
                "MiniCPM-o duplex prefix KV is unavailable; an exact context "
                f"rebase is required (offset={token_offset}, delta_start={delta_start})"
            )
        relative_offset = token_offset - delta_start
        relative_end = relative_offset + span_len
        if relative_end > delta_len:
            raise RuntimeError(
                "MiniCPM-o duplex scheduler span exceeds the materialized delta: "
                f"relative=[{relative_offset}, {relative_end}), delta={delta_len}"
            )
        return (
            delta_embeds[relative_offset:relative_end],
            delta_token_ids[relative_offset:relative_end],
            delta_start,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Model-runner data-plane hook for MiniCPM-o 4.5 duplex audio.

        The scheduler owns the request, block table, attention metadata, KV,
        and sampler. This hook only turns the current duplex audio append into
        the prompt embeddings consumed by the normal runner forward.
        """
        if self.model_stage == "tts":
            return self.talker.preprocess(input_ids=input_ids, input_embeds=input_embeds, **kwargs)
        if self.model_stage != "llm":
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        duplex = kwargs.get("duplex")
        if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        if self._minicpmo_pd_decode:
            media_prefix = duplex.get("pd_media_prefix_tokens")
            token_offset = kwargs.get("duplex_token_offset")
            if media_prefix is not None and (token_offset is None or int(token_offset) < int(media_prefix)):
                raise RuntimeError(
                    "MiniCPM D must import the complete media KV prefix; "
                    "refusing to recompute audio/video placeholders as text embeddings "
                    f"(offset={token_offset}, media_prefix={media_prefix})"
                )
            # P already encoded the cumulative AV prompt and D loads its KV.
            # P's first sampled decision token is D's sole local prompt
            # suffix, so a normal token embedding is exact. Re-running the AV
            # processor here would both duplicate prefill and defeat P/D.
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            decode_info = {key: value for key, value in duplex.items() if key != "payload"}
            payload = duplex.get("payload") or {}
            decode_info["payload"] = {
                key: payload[key] for key in (SAMPLING_STATE_KEY, "force_listen", "is_speech") if key in payload
            }
            decode_info.setdefault(
                "special_token_ids",
                self._minicpmo45_native_duplex_token_ids(),
            )
            return input_ids, embeds, {"duplex": decode_info}

        prompt_len_meta = kwargs.get("duplex_prompt_len")
        token_offset_meta = kwargs.get("duplex_token_offset", 0)
        if (
            isinstance(prompt_len_meta, int)
            and isinstance(token_offset_meta, int)
            and token_offset_meta >= prompt_len_meta
        ):
            # Decode step of the resumable duplex request: input_ids are the
            # runner-sampled tokens and the normal embedding lookup is the
            # correct input. Slicing the (prompt-only) duplex embeddings here
            # would come up empty and pad-fill, feeding a </unit> embedding in
            # place of every sampled token and corrupting generation.
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {}

        helper = self._duplex_data_plane_helper()
        session_id = str(duplex.get("session_id") or "")
        try:
            incarnation = int(duplex.get("incarnation", 0))
        except (TypeError, ValueError):
            incarnation = 0
        payload = duplex.get("payload")
        if not session_id or not isinstance(payload, dict):
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": "bad_duplex_payload"}}

        # Detect an engine prefix replay before mutating Stage0's streaming
        # audio/vision state.  A steady request owns only its newly appended
        # suffix.  After a vLLM preemption, the scheduler explicitly replaces
        # the cumulative prompt with Stage0's bounded compact recovery prompt.
        # No other offset regression is accepted.
        context_rollover = bool(payload.get("duplex_context_rollover", False))
        scheduler_token_budget = duplex.get("scheduler_token_budget")
        engine_rebase = False
        try:
            prompt_len_for_preflight = int(prompt_len_meta) if prompt_len_meta is not None else None
            token_offset_for_preflight = max(0, int(token_offset_meta))
        except (TypeError, ValueError):
            prompt_len_for_preflight = None
            token_offset_for_preflight = 0
        if not context_rollover and prompt_len_for_preflight is not None:
            if isinstance(scheduler_token_budget, bool) or not isinstance(
                scheduler_token_budget,
                int,
            ):
                raise RuntimeError(
                    "MiniCPM-o duplex append requires an exact integer scheduler token budget before Stage0 consumption"
                )
            expected_delta_start = prompt_len_for_preflight - scheduler_token_budget
            if expected_delta_start < 0:
                raise RuntimeError(
                    "MiniCPM-o duplex scheduler budget exceeds the prompt length: "
                    f"prompt={prompt_len_for_preflight}, budget={scheduler_token_budget}"
                )
            if token_offset_for_preflight < expected_delta_start:
                raw_rebase_prefix = duplex.get(
                    "compact_rebase_prefix_tokens",
                )
                if (
                    isinstance(raw_rebase_prefix, bool)
                    or not isinstance(raw_rebase_prefix, int)
                    or raw_rebase_prefix <= 0
                ):
                    raise RuntimeError(
                        "MiniCPM-o duplex prefix KV is unavailable and no "
                        "exact compact rebase was declared "
                        f"(offset={token_offset_for_preflight}, "
                        f"delta_start={expected_delta_start})"
                    )
                expected_rebase_prompt_len = raw_rebase_prefix + scheduler_token_budget
                if prompt_len_for_preflight != expected_rebase_prompt_len:
                    raise RuntimeError(
                        "MiniCPM-o duplex prefix KV is unavailable but the "
                        "scheduler did not install the exact compact rebase "
                        f"prompt: prompt={prompt_len_for_preflight}, "
                        f"expected={expected_rebase_prompt_len}"
                    )
                engine_rebase = True

        session_config = duplex.get("session_config")
        session_config = dict(session_config) if isinstance(session_config, dict) else {}
        runtime_config = duplex.get("runtime_config")
        runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
        state = helper.get_or_create_session_state(
            session_id,
            incarnation,
            session_config=session_config,
            runtime_config=runtime_config,
        )

        seq = duplex.get("seq")
        try:
            seq = int(seq) if seq is not None else None
        except (TypeError, ValueError):
            seq = None
        epoch = duplex.get("epoch")
        try:
            epoch = int(epoch) if epoch is not None else None
        except (TypeError, ValueError):
            epoch = None
        append_identity = (epoch, seq) if seq is not None else None
        prepared_append_replay = bool(
            append_identity is not None
            and getattr(state, "prepared_append_identity", None) == append_identity
            and getattr(state, "prepared_inputs_embeds", None) is not None
        )
        pd_feedback = duplex.get("pd_feedback_token_ids")
        if isinstance(pd_feedback, list) and seq is not None:
            helper.apply_pd_decode_feedback(
                state,
                [int(token_id) for token_id in pd_feedback],
                epoch=epoch,
                seq=seq,
                force_listen=bool(payload.get("force_listen", False)),
                sampling_state=duplex.get("pd_feedback_sampling_state"),
            )
        video_frames = None
        preprocessed_vision = None
        preencoded_vision = None
        preprocessed_audio = None
        audio_waveform = None
        vision_input_source = "request_fallback"
        audio_input_source = "request_fallback"
        batched_vision = kwargs.get(_MINICPMO45_BATCHED_VISION_KEY)
        if isinstance(batched_vision, dict):
            identity = (
                session_id,
                incarnation,
                epoch,
                seq,
            )
            if batched_vision.get("identity") == identity:
                # ``preprocess_batch`` has already decoded this exact payload
                # before preparing its transactional Mel plan.  Reuse that
                # immutable ndarray rather than base64-decoding and allocating
                # the same PCM a second time on the request-local path.
                audio_waveform = batched_vision.get("audio_waveform")
                preprocessed_audio = batched_vision.get("audio_plan")
                raw_audio_source = batched_vision.get("audio_source")
                if isinstance(raw_audio_source, str):
                    audio_input_source = raw_audio_source
                candidate_frames = batched_vision.get("video_frames")
                if isinstance(candidate_frames, list):
                    video_frames = candidate_frames
                    raw_vision_source = batched_vision.get("vision_source")
                    if isinstance(raw_vision_source, str):
                        vision_input_source = raw_vision_source
                    preprocessed_vision = batched_vision.get("processed")
                    candidate_blocks = batched_vision.get("frame_blocks")
                    if isinstance(candidate_blocks, list):
                        preencoded_vision = candidate_blocks
            # The runner calls preprocess_batch() immediately before the
            # per-request preprocess loop. Release the speculative CPU payloads
            # after this request consumes (or rejects) them; a retry prepares
            # them again from its authoritative request payload.
            batched_vision.clear()
        audio_decode_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        if audio_waveform is None and not prepared_append_replay:
            audio_waveform = helper._decode_audio_payload(payload)
        audio_decode_ms = (time.perf_counter() - audio_decode_start) * 1000.0 if _MINICPMO45_LOG_PREP_DIAG else 0.0
        if video_frames is None and not prepared_append_replay:
            try:
                video_frames = helper._decode_video_frames_payload(payload)
            except ValueError as exc:
                embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
                return input_ids, embeds, {"duplex": {"prefill_success": False, "reason": str(exc)}}
        result = helper._stage_prefill_embeddings_only(
            state,
            audio_waveform,
            video_frames=video_frames,
            max_slice_nums=payload.get("max_slice_nums", 1),
            preprocessed_vision=preprocessed_vision,
            preencoded_vision=preencoded_vision,
            preprocessed_audio=preprocessed_audio,
            epoch=epoch,
            seq=seq,
            is_speech=bool(payload.get("is_speech", False)),
            final=bool(duplex.get("final")),
            context_rollover=context_rollover,
            engine_rebase=engine_rebase,
            vision_input_source=vision_input_source,
        )
        update_result = dict(result)
        update_result.pop("inputs_embeds", None)
        update_result.pop("input_token_ids", None)
        if result.get("success") is not True:
            if engine_rebase:
                raise RuntimeError(
                    "MiniCPM-o duplex compact engine rebase failed without "
                    f"mutating the session: {result.get('reason', 'unknown')}"
                )
            embeds = input_embeds if input_embeds is not None else self.get_input_embeddings(input_ids)
            return input_ids, embeds, {"duplex": update_result}
        if _MINICPMO45_LOG_PREP_DIAG:
            diag = result.get("prep_diag")
            if isinstance(diag, dict):
                logger.info(
                    "[MINICPM-PREP] req=%s audio_decode_ms=%.3f "
                    "vision_consume_ms=%.3f audio_feature_ms=%.3f "
                    "audio_encoder_ms=%.3f assembly_ms=%.3f stage_total_ms=%.3f "
                    "audio_source=%s",
                    kwargs.get("request_id", "?"),
                    audio_decode_ms,
                    float(diag.get("vision_consume_ms", 0.0)),
                    float(diag.get("audio_feature_ms", 0.0)),
                    float(diag.get("audio_encoder_ms", 0.0)),
                    float(diag.get("assembly_ms", 0.0)),
                    float(diag.get("stage_total_ms", 0.0)),
                    audio_input_source,
                )
            if video_frames:
                logger.info(
                    "[MINICPM-FRAME-CONSUMED] req=%s seq=%s frames=%d source=%s done_epoch=%.6f",
                    kwargs.get("request_id", "?"),
                    seq,
                    len(video_frames),
                    "arrival" if preencoded_vision is not None else "formal_fallback",
                    time.time(),
                )

        target_dtype = (
            input_embeds.dtype if input_embeds is not None else self.get_input_embeddings(input_ids[:1]).dtype
        )
        delta_embeds = result["inputs_embeds"].to(device=input_ids.device, dtype=target_dtype)
        delta_input_token_ids = [int(token_id) for token_id in (result.get("input_token_ids") or [])]
        prompt_len = kwargs.get("duplex_prompt_len")
        try:
            prompt_len = int(prompt_len) if prompt_len is not None else int(delta_embeds.shape[0])
        except (TypeError, ValueError):
            prompt_len = int(delta_embeds.shape[0])

        span_len = int(input_ids.shape[0])
        token_offset = kwargs.get("duplex_token_offset", 0)
        try:
            token_offset = max(0, int(token_offset))
        except (TypeError, ValueError):
            token_offset = 0
        if not result.get("rebase_prompt", False) and scheduler_token_budget != len(delta_input_token_ids):
            logger.error(
                "[duplex-budget] session=%s seq=%s budget=%s rows=%d feedback=%s delta_tail=%s",
                session_id,
                seq,
                scheduler_token_budget,
                len(delta_input_token_ids),
                pd_feedback,
                delta_input_token_ids[-16:],
            )
        req_embeds, input_token_ids, delta_start = self._slice_duplex_prompt_delta(
            delta_embeds,
            delta_input_token_ids,
            prompt_len=prompt_len,
            token_offset=token_offset,
            span_len=span_len,
            rebase_prompt=bool(result.get("rebase_prompt", False)),
            scheduler_token_budget=scheduler_token_budget,
        )
        req_input_ids = torch.tensor(input_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        update_result["duplex_prompt_delta_start"] = delta_start
        update_result["duplex_prompt_delta_token_ids"] = delta_input_token_ids
        update_result["duplex_prompt_len"] = prompt_len
        if not getattr(self, "_minicpmo_pd_prefill", False):
            # The native non-P/D Talker bridge still consumes a full prompt
            # length marker. Keep that compatibility list off the P path;
            # unlike embeddings, it is never sent through the Thinker runner.
            pad_token_id = helper.stage_padding_token_id()
            update_result["duplex_prompt_token_ids"] = [pad_token_id] * delta_start + delta_input_token_ids
        return req_input_ids, req_embeds, {"duplex": update_result}

    @torch.inference_mode()
    def preencode_duplex_vision(
        self,
        jobs: list[dict[str, object]],
    ) -> dict[str, object]:
        """Encode arrival-side camera frames without running the Thinker LLM."""
        total_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        if self.model_stage != "llm" or self._minicpmo_pd_decode:
            return {"supported": False, "encoded_frames": 0}
        helper = self._duplex_data_plane_helper()
        process_image = getattr(helper.processor, "process_image", None)
        encode_batch = getattr(helper, "_stage_vision_embeddings_batch", None)
        cache_embeddings = getattr(helper, "cache_arrival_vision_embeddings", None)
        if not callable(process_image) or not callable(encode_batch) or not callable(cache_embeddings):
            return {"supported": False, "encoded_frames": 0}

        def prepare(job: dict[str, object]):
            session_id = job.get("session_id")
            raw_frames = job.get("video_frames")
            raw_ids = job.get("preencode_ids")
            if (
                not isinstance(session_id, str)
                or not session_id
                or not isinstance(raw_frames, list)
                or not raw_frames
                or not isinstance(raw_ids, list)
                or len(raw_ids) != len(raw_frames)
                or not all(isinstance(value, str) and value for value in raw_ids)
            ):
                return None
            try:
                incarnation = int(job.get("incarnation", 0))
                epoch = int(job.get("epoch", 0))
            except (TypeError, ValueError):
                return None
            raw_limits = job.get("max_slice_nums", 1)
            if isinstance(raw_limits, int) and not isinstance(raw_limits, bool):
                max_slice_nums = max(1, raw_limits)
            elif (
                isinstance(raw_limits, list)
                and len(raw_limits) == len(raw_frames)
                and raw_limits
                and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_limits)
                and len({max(1, value) for value in raw_limits}) == 1
            ):
                max_slice_nums = max(1, raw_limits[0])
            else:
                return None
            payload = {
                "video_frames": list(raw_frames),
                "max_slice_nums": max_slice_nums,
            }
            try:
                frames = helper._decode_video_frames_payload(payload)
                if not frames:
                    return None
                processed = process_image(frames, max_slice_nums=max_slice_nums)
                if processed is None:
                    return None
            except Exception:  # noqa: BLE001 - speculative fallback is request-local
                return None
            return (
                session_id,
                incarnation,
                epoch,
                list(raw_ids),
                processed,
            )

        executor = getattr(self, "_minicpmo45_vision_prepare_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="minicpmo-vision-prepare",
            )
            self._minicpmo45_vision_prepare_executor = executor
        cpu_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        prepared = [item for item in executor.map(prepare, jobs) if item is not None]
        cpu_prepare_ms = (time.perf_counter() - cpu_start) * 1000.0 if _MINICPMO45_LOG_PREP_DIAG else 0.0
        if not prepared:
            return {"supported": True, "encoded_frames": 0}
        try:
            microbatch_size = max(
                1,
                int(os.environ.get("MINICPMO45_VISION_ENCODER_BATCH_SIZE", "8")),
            )
        except ValueError:
            microbatch_size = 8
        vision_device = None
        if _MINICPMO45_LOG_PREP_DIAG:
            thinker = getattr(self, "thinker", None)
            vision_module = getattr(thinker, "vpm", None)
            if vision_module is not None:
                vision_device = next(vision_module.parameters()).device
            torch.accelerator.synchronize(vision_device)
            vision_start = time.perf_counter()
        encoded_batch = encode_batch(
            [item[4] for item in prepared],
            microbatch_size=microbatch_size,
        )
        if _MINICPMO45_LOG_PREP_DIAG:
            torch.accelerator.synchronize(vision_device)
            vision_encoder_ms = (time.perf_counter() - vision_start) * 1000.0
        else:
            vision_encoder_ms = 0.0
        if not isinstance(encoded_batch, list) or len(encoded_batch) != len(prepared):
            return {"supported": True, "encoded_frames": 0}
        cache_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        encoded_frames = 0
        for item, frame_blocks in zip(prepared, encoded_batch, strict=True):
            session_id, incarnation, epoch, preencode_ids, _ = item
            if not isinstance(frame_blocks, list):
                continue
            encoded_frames += cache_embeddings(
                session_id=session_id,
                incarnation=incarnation,
                epoch=epoch,
                preencode_ids=preencode_ids,
                frame_blocks=frame_blocks,
            )
        if _MINICPMO45_LOG_PREP_DIAG:
            cache_ms = (time.perf_counter() - cache_start) * 1000.0
            cache_size = getattr(helper, "arrival_vision_cache_size", None)
            cache_entries = int(cache_size()) if callable(cache_size) else -1
            logger.info(
                "[MINICPM-PREP-ARRIVAL] jobs=%d encoded_frames=%d "
                "cpu_prepare_ms=%.3f vision_encoder_ms=%.3f cache_ms=%.3f "
                "total_ms=%.3f done_epoch=%.6f cache_entries=%d",
                len(prepared),
                encoded_frames,
                cpu_prepare_ms,
                vision_encoder_ms,
                cache_ms,
                (time.perf_counter() - total_start) * 1000.0,
                time.time(),
                cache_entries,
            )
        return {
            "supported": True,
            "encoded_jobs": len(prepared),
            "encoded_frames": encoded_frames,
        }

    @torch.inference_mode()
    def preencode_duplex_audio(
        self,
        jobs: list[dict[str, object]],
    ) -> dict[str, object]:
        """Advance GPU2-owned streaming-audio lineages ahead of formal P."""
        if self.model_stage != "llm" or self._minicpmo_pd_decode:
            return {"supported": False, "encoded_jobs": 0, "job_results": {}}
        helper = self._duplex_data_plane_helper()
        preencode = getattr(helper, "preencode_arrival_audio", None)
        if not callable(preencode):
            return {"supported": False, "encoded_jobs": 0, "job_results": {}}
        return preencode(jobs)

    @torch.inference_mode()
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Prepare independent native-duplex AV inputs across requests.

        The generic Omni runner invokes ``preprocess`` once per request. That
        remains the transactional commit point for stateful streaming audio.
        Before it runs, independent sessions can prepare exact Mel features in
        parallel and batch equal-shaped Whisper/vision encoder work.  The
        speculative results are consumed once by the request-local path; stale
        or failed results fall back without advancing session state.
        """
        del device
        batch_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        if self.model_stage != "llm" or self._minicpmo_pd_decode:
            return

        helper = self._duplex_data_plane_helper()
        sessions = getattr(helper, "sessions", None)
        get_or_create_state = getattr(helper, "get_or_create_session_state", None)
        pending: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any], int, Any]] = []
        arrival_hits: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any], int, Any]] = []
        audio_items: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any], int, Any]] = []
        claimed_audio_sessions: set[tuple[str, int]] = set()

        for req_id in req_ids:
            info = model_intermediate_buffer.get(req_id)
            if not isinstance(info, dict):
                continue
            # Drop a value left by an interrupted preprocess pass before
            # considering the current append.
            info.pop(_MINICPMO45_BATCHED_VISION_KEY, None)
            duplex = info.get("duplex")
            if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
                continue
            payload = duplex.get("payload")
            if not isinstance(payload, dict):
                continue

            session_id = str(duplex.get("session_id") or "")
            if not session_id:
                continue
            try:
                incarnation = int(duplex.get("incarnation", 0))
            except (TypeError, ValueError):
                incarnation = 0
            try:
                epoch_raw = duplex.get("epoch")
                epoch = int(epoch_raw) if epoch_raw is not None else None
            except (TypeError, ValueError):
                epoch = None
            try:
                seq_raw = duplex.get("seq")
                seq = int(seq_raw) if seq_raw is not None else None
            except (TypeError, ValueError):
                seq = None

            append_identity = (epoch, seq) if seq is not None else None
            session_config = duplex.get("session_config")
            session_config = dict(session_config) if isinstance(session_config, dict) else {}
            runtime_config = duplex.get("runtime_config")
            runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}
            if callable(get_or_create_state):
                state = get_or_create_state(
                    session_id,
                    incarnation,
                    session_config=session_config,
                    runtime_config=runtime_config,
                )
            else:
                state = sessions.get((session_id, incarnation)) if isinstance(sessions, dict) else None
            if (
                append_identity is not None
                and state is not None
                and getattr(state, "prepared_append_identity", None) == append_identity
            ):
                continue

            identity = (session_id, incarnation, epoch, seq)
            item = (info, identity, payload, 1, state)
            audio_arrival_hit = False
            raw_audio_preencode_id = payload.get("audio_preencode_id")
            raw_audio_preencode_seq = payload.get("audio_preencode_seq")
            try:
                audio_preencode_seq = int(raw_audio_preencode_seq) if raw_audio_preencode_seq is not None else None
            except (TypeError, ValueError):
                audio_preencode_seq = None
            if isinstance(raw_audio_preencode_id, str) and raw_audio_preencode_id and audio_preencode_seq is not None:
                take_preencoded_audio = getattr(
                    helper,
                    "take_arrival_audio_append",
                    None,
                )
                audio_plan = (
                    take_preencoded_audio(
                        session_id=session_id,
                        incarnation=incarnation,
                        epoch=epoch,
                        audio_preencode_seq=audio_preencode_seq,
                        audio_preencode_id=raw_audio_preencode_id,
                    )
                    if callable(take_preencoded_audio)
                    else None
                )
                if audio_plan is not None:
                    cached = info.get(_MINICPMO45_BATCHED_VISION_KEY)
                    if not isinstance(cached, dict) or cached.get("identity") != identity:
                        cached = {"identity": identity}
                        info[_MINICPMO45_BATCHED_VISION_KEY] = cached
                    cached["audio_plan"] = audio_plan
                    cached["audio_source"] = "arrival"
                    audio_arrival_hit = True
                else:
                    retire_preencoded_audio = getattr(
                        helper,
                        "retire_arrival_audio_append",
                        None,
                    )
                    if callable(retire_preencoded_audio) and epoch is not None:
                        retire_preencoded_audio(
                            session_id=session_id,
                            incarnation=incarnation,
                            epoch=epoch,
                            audio_preencode_seq=audio_preencode_seq,
                            audio_preencode_id=raw_audio_preencode_id,
                        )
            # The Mel processor and Whisper KV are session state.  Normal
            # serving admits one append per session, but defensively speculate
            # only the first if multiple appends ever share a runner batch.
            # Later appends then take the exact serial fallback after the first
            # request commits, rather than racing the same processor snapshot.
            audio_session = (session_id, incarnation)
            if not audio_arrival_hit and state is not None and audio_session not in claimed_audio_sessions:
                claimed_audio_sessions.add(audio_session)
                audio_items.append(item)

            raw_frames = payload.get("video_frames")
            if not isinstance(raw_frames, list) or not raw_frames:
                continue

            raw_limits = payload.get("max_slice_nums", 1)
            if isinstance(raw_limits, int) and not isinstance(raw_limits, bool):
                uniform_limit = max(1, raw_limits)
            elif (
                isinstance(raw_limits, list)
                and len(raw_limits) == len(raw_frames)
                and raw_limits
                and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_limits)
                and len({max(1, value) for value in raw_limits}) == 1
            ):
                # The live protocol carries one entry per frame even when all
                # frames use the same HD-slicing limit.  Preserve heterogeneous
                # per-frame limits on the exact serial fallback path.
                uniform_limit = max(1, raw_limits[0])
            else:
                continue
            item = (info, identity, payload, uniform_limit, state)
            raw_preencode_ids = payload.get("video_preencode_ids")
            if (
                isinstance(raw_preencode_ids, list)
                and len(raw_preencode_ids) == len(raw_frames)
                and all(isinstance(preencode_id, str) and preencode_id for preencode_id in raw_preencode_ids)
                and len(set(raw_preencode_ids)) == len(raw_preencode_ids)
            ):
                take_preencoded = getattr(helper, "take_arrival_vision_embeddings", None)
                frame_blocks = (
                    take_preencoded(
                        session_id=session_id,
                        incarnation=incarnation,
                        epoch=epoch,
                        preencode_ids=list(raw_preencode_ids),
                    )
                    if callable(take_preencoded)
                    else None
                )
                if isinstance(frame_blocks, list) and len(frame_blocks) == len(raw_frames):
                    cached = info.get(_MINICPMO45_BATCHED_VISION_KEY)
                    if not isinstance(cached, dict) or cached.get("identity") != identity:
                        cached = {"identity": identity}
                        info[_MINICPMO45_BATCHED_VISION_KEY] = cached
                    cached.update(
                        {
                            "video_frames": list(raw_frames),
                            "frame_blocks": frame_blocks,
                            "vision_source": "arrival",
                        }
                    )
                    arrival_hits.append(item)
                    continue
                retire_preencoded = getattr(
                    helper,
                    "retire_arrival_vision_embeddings",
                    None,
                )
                if callable(retire_preencoded):
                    retire_preencoded(
                        session_id=session_id,
                        incarnation=incarnation,
                        epoch=epoch,
                        preencode_ids=list(raw_preencode_ids),
                    )
            pending.append(item)

        vision_items = [*pending, *arrival_hits]
        if _MINICPMO45_LOG_PREP_DIAG and arrival_hits:
            logger.info(
                "[MINICPM-PREP-ARRIVAL-HIT] requests=%d",
                len(arrival_hits),
            )
        # Encoder batching needs at least two independent requests.  Arrival
        # vision hits remain installed even when this hook has no other work.
        prepare_vision_batch = len(vision_items) >= 2
        prepare_audio_batch = len(audio_items) >= 2
        if not prepare_vision_batch and not prepare_audio_batch:
            return

        def prepare_vision(
            item: tuple[
                dict[str, Any],
                tuple[Any, ...],
                dict[str, Any],
                int,
                Any,
            ],
        ):
            info, identity, payload, max_slice_nums, state = item
            try:
                frames = helper._decode_video_frames_payload(payload)
                if not frames:
                    return None
                process_image = getattr(helper.processor, "process_image", None)
                if not callable(process_image):
                    return None
                processed = process_image(frames, max_slice_nums=max_slice_nums)
                if processed is None:
                    return None
            except Exception:  # noqa: BLE001 - preserve serial error handling
                # Preserve the existing per-request error path and do not let
                # one malformed frame poison other sessions.
                return None
            return info, identity, frames, processed, state

        def prepare_audio(
            item: tuple[
                dict[str, Any],
                tuple[Any, ...],
                dict[str, Any],
                int,
                Any,
            ],
        ):
            _info, identity, payload, _max_slice_nums, state = item
            if state is None:
                return identity, None, None
            try:
                audio_waveform = helper._decode_audio_payload(payload)
                return (
                    identity,
                    audio_waveform,
                    helper._prepare_streaming_audio_append(
                        state,
                        audio_waveform,
                    ),
                )
            except Exception:  # noqa: BLE001 - preserve serial fallback
                return identity, None, None

        # Audio Mel extraction and image decoding are independent. Submit the
        # former first, then let it continue on CPU while the main thread runs
        # the batched vision tower on GPU.
        audio_futures = {}
        if audio_items:
            audio_executor = getattr(self, "_minicpmo45_audio_prepare_executor", None)
            if audio_executor is None:
                audio_executor = ThreadPoolExecutor(
                    max_workers=8,
                    thread_name_prefix="minicpmo-audio-prepare",
                )
                self._minicpmo45_audio_prepare_executor = audio_executor
            audio_futures = {item[1]: audio_executor.submit(prepare_audio, item) for item in audio_items}

        cpu_prepare_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        prepared_items = []
        if prepare_vision_batch and pending:
            executor = getattr(self, "_minicpmo45_vision_prepare_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=8,
                    thread_name_prefix="minicpmo-vision-prepare",
                )
                self._minicpmo45_vision_prepare_executor = executor
            prepared_items = [prepared for prepared in executor.map(prepare_vision, pending) if prepared is not None]
        cpu_prepare_ms = (time.perf_counter() - cpu_prepare_start) * 1000.0 if _MINICPMO45_LOG_PREP_DIAG else 0.0

        frame_blocks_batch = None
        vision_batch_ms = 0.0
        encode_batch = getattr(helper, "_stage_vision_embeddings_batch", None)
        if callable(encode_batch) and prepared_items:
            try:
                microbatch_size = max(
                    1,
                    int(os.environ.get("MINICPMO45_VISION_ENCODER_BATCH_SIZE", "8")),
                )
            except ValueError:
                microbatch_size = 8
            if _MINICPMO45_LOG_PREP_DIAG:
                torch.accelerator.synchronize()
                vision_batch_start = time.perf_counter()
            frame_blocks_batch = encode_batch(
                [prepared[3] for prepared in prepared_items],
                microbatch_size=microbatch_size,
            )
            if _MINICPMO45_LOG_PREP_DIAG:
                torch.accelerator.synchronize()
                vision_batch_ms = (time.perf_counter() - vision_batch_start) * 1000.0
            if not isinstance(frame_blocks_batch, list) or len(frame_blocks_batch) != len(prepared_items):
                frame_blocks_batch = None

        audio_wait_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        audio_results: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        for identity, future in audio_futures.items():
            try:
                _returned_identity, audio_waveform, audio_plan = future.result()
            except Exception:  # noqa: BLE001 - preserve serial fallback
                audio_waveform = None
                audio_plan = None
            audio_results[identity] = (audio_waveform, audio_plan)
        audio_wait_ms = (time.perf_counter() - audio_wait_start) * 1000.0 if _MINICPMO45_LOG_PREP_DIAG else 0.0
        audio_batch_start = time.perf_counter() if _MINICPMO45_LOG_PREP_DIAG else 0.0
        prepared_audio = [
            (audio_results[item[1]][1], item[4])
            for item in audio_items
            if item[1] in audio_results and audio_results[item[1]][1] is not None and item[4] is not None
        ]
        if len(prepared_audio) >= 2:
            helper._stage_audio_embeddings_batch(prepared_audio)
        if _MINICPMO45_LOG_PREP_DIAG:
            torch.accelerator.synchronize()
            audio_batch_ms = (time.perf_counter() - audio_batch_start) * 1000.0

        for index, prepared in enumerate(prepared_items):
            info, identity, frames, processed, _state = prepared
            cached = info.get(_MINICPMO45_BATCHED_VISION_KEY)
            if not isinstance(cached, dict) or cached.get("identity") != identity:
                cached = {"identity": identity}
                info[_MINICPMO45_BATCHED_VISION_KEY] = cached
            cached.update(
                {
                    "video_frames": frames,
                    "processed": processed,
                    "vision_source": "runner_fallback",
                }
            )
            if frame_blocks_batch is not None:
                request_blocks = frame_blocks_batch[index]
                if isinstance(request_blocks, list) and len(request_blocks) == len(frames):
                    cached["frame_blocks"] = request_blocks

        # Install decoded PCM and the optional transactional Mel/Whisper plan
        # independently of vision.  This covers audio-only slots and prevents
        # a second request-local decode for ordinary AV slots.
        for info, identity, _payload, _max_slice_nums, _state in audio_items:
            audio_result = audio_results.get(identity)
            if audio_result is None or audio_result[0] is None:
                continue
            cached = info.get(_MINICPMO45_BATCHED_VISION_KEY)
            if not isinstance(cached, dict) or cached.get("identity") != identity:
                cached = {"identity": identity}
                info[_MINICPMO45_BATCHED_VISION_KEY] = cached
            audio_waveform, audio_plan = audio_result
            cached["audio_waveform"] = audio_waveform
            if audio_plan is not None:
                cached["audio_plan"] = audio_plan
        if _MINICPMO45_LOG_PREP_DIAG:
            request_count = len({item[1] for item in [*vision_items, *audio_items]})
            logger.info(
                "[MINICPM-PREP-BATCH] reqs=%d cpu_image_ms=%.3f "
                "vision_batch_ms=%.3f audio_wait_ms=%.3f "
                "audio_batch_ms=%.3f total_ms=%.3f",
                request_count,
                cpu_prepare_ms,
                vision_batch_ms,
                audio_wait_ms,
                audio_batch_ms,
                (time.perf_counter() - batch_start) * 1000.0,
            )

    def _duplex_data_plane_helper(self):
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        if helper is not None:
            return helper
        with self._minicpmo45_helper_init_lock:
            helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
            if helper is None:
                from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import MiniCPMO45Stage0DuplexRuntime

                model_path = getattr(getattr(self.vllm_config, "model_config", None), "model", None)
                device = str(self._module_device(self.thinker if self.thinker is not None else self))
                helper = MiniCPMO45Stage0DuplexRuntime(self, model_path=model_path, device=device)
                self._minicpmo45_duplex_data_plane_helper = helper
            return helper

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to the active stage submodule when it implements MM encoding.
        mm_fn = getattr(self.model, "get_multimodal_embeddings", None)
        if mm_fn is not None:
            return mm_fn(**kwargs)
        return []

    def embed_multimodal(self, **kwargs: object):
        """vLLM V1 encoder profiling calls this; the inherited Protocol stub returns None."""
        return self.get_multimodal_embeddings(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Forward pass for MiniCPM-o Omni model.

        Workflow:
        1) Thinker (model_stage="llm"): Image / video / audio encoders +
           3D resampler + omni LLM → text + hidden states.
        2) Talker (model_stage="tts"): native MiniCPMTTS AR → codec deltas.
        """
        if self.model_stage == "llm":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            # TODO: Remove this hack when NPU supports batched inputs properly
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            thinker_dev = self._module_device(self.thinker)

            # if input_ids is None, set it to a zero tensor
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(0)
                added_batch_dim = True

            # Ensure inputs on thinker's device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            if current_omni_platform.is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions.ndim > 1 else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )
            else:
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                thinker_positions = positions[0] if positions is not None and added_batch_dim else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )

            # vLLM's fused residual updates may modify inputs_embeds in-place.
            # A correctness probe must snapshot inputs BEFORE that forward.
            probe_embeds = None
            if getattr(self, "_minicpmo_pd_thinker", False) and getattr(self, "_minicpmo45_numerical_probe_dir", None):
                from vllm_omni.experimental.fullduplex.minicpmo45.numerical_probe import capture_forward, selected_rows

                if selected_rows(kwargs.get("minicpmo_numerical_probe_rows")):
                    probe_embeds = thinker_inputs_embeds.detach().clone()
                    if self._minicpmo_pd_decode:
                        capture_forward(
                            self,
                            None,
                            thinker_positions,
                            None,
                            None,
                            rows=kwargs.get("minicpmo_numerical_probe_rows"),
                            phase="before_backbone_forward",
                        )

            # Layer I/O is a separate opt-in eager correctness diagnostic.
            # Default execution never installs hooks or copies layer tensors.
            if probe_embeds is not None and os.environ.get("MINICPMO45_NUMERICAL_PROBE_LAYER_IO") == "1":
                from vllm_omni.experimental.fullduplex.minicpmo45.numerical_probe import capture_layer0_io

                with capture_layer0_io(self, thinker_positions, rows=kwargs.get("minicpmo_numerical_probe_rows")):
                    thinker_output = self.thinker(
                        input_ids=thinker_input_ids,
                        positions=thinker_positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=thinker_inputs_embeds,
                        **kwargs,
                    )
            else:
                thinker_output = self.thinker(
                    input_ids=thinker_input_ids,
                    positions=thinker_positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=thinker_inputs_embeds,
                    **kwargs,
                )

            input_embedding_states = None
            if isinstance(thinker_output, tuple):
                input_embedding_states, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Prepare hidden states for downstream stages
            # Ensure correct shape: (batch_size, seq_len, hidden_dim)
            if added_batch_dim:
                text_hidden_states = text_hidden_states.squeeze(0)

            if probe_embeds is not None:
                from vllm_omni.experimental.fullduplex.minicpmo45.numerical_probe import capture_forward

                capture_forward(
                    self,
                    thinker_input_ids,
                    thinker_positions,
                    probe_embeds,
                    text_hidden_states,
                    rows=kwargs.get("minicpmo_numerical_probe_rows"),
                )

            # D's generated hidden rows are sufficient for MiniCPM's native
            # duplex Thinker->Talker bridge. P therefore exports KV only; an
            # O(context) hidden-state snapshot would duplicate host memory and
            # copy work every 1 s unit without affecting native audio output.
            # In native P/D, D's generic scheduled-hidden payload is the
            # canonical delta.  Emitting the same tensor again as ``latent``
            # duplicates every D->Core IPC payload; the output processor maps
            # generic ``hidden`` to the stage's latent modality.
            multimodal_outputs = {} if getattr(self, "_minicpmo_pd_thinker", False) else {"latent": text_hidden_states}
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                duplex_rows = []
                for req_info in runtime_info:
                    duplex_info = req_info.get("duplex") if isinstance(req_info, dict) else None
                    duplex_rows.append(duplex_info if isinstance(duplex_info, dict) else {})

                prompt_rows = []
                for duplex_info in duplex_rows:
                    prompt_token_ids = duplex_info.get("duplex_prompt_token_ids")
                    # This is a complete per-handoff snapshot, not a generated
                    # tensor delta. Keep it as row-local metadata so output
                    # accumulation replaces the previous value instead of
                    # attempting to concatenate variable-length prompts.
                    prompt_rows.append(list(prompt_token_ids) if isinstance(prompt_token_ids, list) else None)
                if any(row is not None for row in prompt_rows):
                    multimodal_outputs["duplex_prompt_token_ids"] = prompt_rows

                # Fixed-size, per-request input audit metadata. Lists are
                # intentional: the AR output builder selects one item by
                # request index before converting the wire value to a scalar
                # CPU tensor. These counters never enter prefix-cache tensor
                # reconstruction and add no sequence-length-dependent copy.
                for audit_key in (
                    "duplex_input_video_frames",
                    "duplex_arrival_video_frames",
                    "duplex_vision_fallback_frames",
                    "duplex_arrival_audio_units",
                    "duplex_audio_fallback_units",
                ):
                    audit_rows = [max(0, int(duplex_info.get(audit_key, 0) or 0)) for duplex_info in duplex_rows]
                    # Presence is significant even when every row is zero:
                    # formal sidecar audits must distinguish an observed
                    # zero fallback count from a counter that was never
                    # emitted by stage 0.
                    if any(audit_key in duplex_info for duplex_info in duplex_rows):
                        multimodal_outputs[audit_key] = audit_rows

                special_keys = {
                    key
                    for duplex_info in duplex_rows
                    for key, value in (
                        duplex_info.get("special_token_ids", {}).items()
                        if isinstance(duplex_info.get("special_token_ids"), dict)
                        else ()
                    )
                    if isinstance(key, str) and isinstance(value, int) and value >= 0
                }
                if special_keys:
                    # Known CPU IDs are metadata, not GPU model inputs. Keep
                    # their one-element shape for the existing tensor wire
                    # format, but do not enqueue H2D copies (and synchronization)
                    # after each backbone forward just to copy them back out.
                    multimodal_outputs["meta"] = {
                        key: [
                            (int(value),)
                            if isinstance(value, int) and value >= 0
                            else None
                            for duplex_info in duplex_rows
                            for value in [
                                (
                                    duplex_info.get("special_token_ids", {}).get(key)
                                    if isinstance(duplex_info.get("special_token_ids"), dict)
                                    else None
                                )
                            ]
                        ]
                        for key in sorted(special_keys)
                    }
            return OmniOutput(
                text_hidden_states=text_hidden_states,
                multimodal_outputs=multimodal_outputs,
            )

        # Talker stage: runner-owned native AR only. Waveform generation belongs
        # to the separate Code2Wav stage.
        if self.model_stage == "tts":
            return self.talker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        raise ValueError(f"Unsupported model stage: {self.model_stage}")

    @property
    def requires_request_sample_eligibility(self):
        # The runner sees this outer model, not the inner Talker. Without
        # forwarding the capability, incomplete prefill chunks would still
        # advance the model-owned codec sampler.
        return self.model_stage == "tts"

    @property
    def supports_omni_decode_step_metadata(self):
        # The existing runner hook executes even when the model uses CUDA
        # graph replay. Keep it entirely disabled for ordinary serving.
        from vllm_omni.experimental.fullduplex.minicpmo45 import speech_probe

        return self.model_stage == "tts" and speech_probe.ENABLED

    def update_decode_step_metadata(self, **kwargs):
        if self.supports_omni_decode_step_metadata:
            self.talker.capture_speech_step_metadata(**kwargs)

    def make_omni_output(self, model_outputs, **kwargs):
        if self.model_stage != "tts":
            return model_outputs
        return self.talker.make_omni_output(model_outputs, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use model for logits computation
        return self.model.compute_logits(hidden_states)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        decode_states = getattr(self, "_minicpmo45_pd_sampling_states", {})
        for request_id in finished_req_ids:
            decode_states.pop(request_id, None)
        request_sessions = getattr(self, "_minicpmo45_duplex_request_sessions", None)
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        if isinstance(request_sessions, dict):
            for request_id in finished_req_ids:
                session_key = request_sessions.pop(request_id, None)
                if session_key is not None and isinstance(sessions, dict):
                    sessions.pop(session_key, None)
                    discard_preencoded = getattr(helper, "discard_arrival_vision_session", None)
                    if callable(discard_preencoded):
                        discard_preencoded(*session_key)
                    discard_audio = getattr(helper, "discard_arrival_audio_session", None)
                    if callable(discard_audio):
                        discard_audio(*session_key)
        forced_segments = getattr(self, "_minicpmo45_force_listen_applied_segments", None)
        if isinstance(forced_segments, set):
            finished = set(finished_req_ids)
            completed_segments = {segment for segment in forced_segments if segment[0] in finished}
            forced_segments.difference_update(completed_segments)
        if hasattr(self.model, "on_requests_finished"):
            self.model.on_requests_finished(finished_req_ids)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        native_duplex = self._sample_minicpmo45_native_duplex_stage0(
            logits,
            sampling_metadata,
            duplex_rows=getattr(self, "_minicpmo45_active_duplex_rows", None),
        )
        if native_duplex is not None:
            return native_duplex
        if self.model_stage == "tts":
            return self.model.sample(logits, sampling_metadata)
        return None

    def _sample_minicpmo45_native_duplex_stage0(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        duplex_rows: list[int] | None = None,
    ) -> SamplerOutput | None:
        if self.model_stage != "llm" or logits.ndim != 2 or logits.shape[0] == 0:
            return None
        token_ids = self._minicpmo45_native_duplex_token_ids()
        unit_id = token_ids.get("unit_token_id", -1)
        if unit_id < 0:
            return None
        native_rows = self._minicpmo45_native_duplex_prompt_rows(
            sampling_metadata,
            unit_id,
            logits.shape[0],
            duplex_rows=duplex_rows,
        )
        if not native_rows or len(native_rows) != logits.shape[0]:
            return None

        batch_size = int(logits.shape[0])
        temperatures = self._sampling_metadata_values(sampling_metadata, "temperature", batch_size, 0.7)
        top_ks = self._sampling_metadata_values(sampling_metadata, "top_k", batch_size, 100)
        top_ps = self._sampling_metadata_values(sampling_metadata, "top_p", batch_size, 0.8)

        rng_rows = {row.row_idx: row for row in getattr(self, "_minicpmo45_sampling_rows", ())}
        sampled_ids: list[int] = [0] * batch_size
        # Only reorder independent rows. With an unseeded/shared generator,
        # retain native row-major RNG draws (boundary then text for each row).
        generators = getattr(sampling_metadata, "generators", {})
        row_generators = [generators.get(i) for i in range(batch_size)]
        states = [self._minicpmo45_duplex_state_for_row(i) for i in range(batch_size)]
        owned_states = [id(state) for state in states if state is not None]
        batch_feedback = (
            logits.is_cuda and batch_size > 1
            and len(set(owned_states)) == len(owned_states)
            and (getattr(sampling_metadata, "all_greedy", False)
                 or (all(g is not None for g in row_generators)
                     and len({id(g) for g in row_generators}) == batch_size))
        )
        pending: list[tuple[int, Generator[torch.Tensor | _MiniCPMFilteredDraw, int | None, int], int | None]] = []
        row_rng: dict[int, Any] = {}
        skipped_request_ids: list[str] = []
        for row_idx in range(logits.shape[0]):
            row = rng_rows.get(row_idx)
            if row is not None and not row.should_sample:
                # Intermediate prefill chunks have no model decision yet.
                # vLLM drops this placeholder using its existing discard mask.
                skipped_request_ids.append(row.request_id)
                continue
            state = self._minicpmo45_duplex_state_for_row(row_idx)
            generator = None
            if getattr(self, "_minicpmo_pd_thinker", False) and not getattr(sampling_metadata, "all_greedy", False):
                generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
                row = rng_rows.get(row_idx)
                if row is not None:
                    identity = (row.request_id, row.incarnation, row.epoch, row.seq)
                    resume_sampling_rng(state, generator, identity)
            segment = getattr(state, "current_segment_output_tokens", [])
            if (
                getattr(self, "_minicpmo_pd_decode", False)
                and len(segment) == 1
                and segment[0]
                in {
                    token_ids.get(key, -1)
                    for key in ("listen_token_id", "chunk_eos_token_id", "chunk_tts_eos_token_id")
                }
            ):
                # P already ended this unit. D only installs KV and feeds that
                # terminator (as the native decoder does); expose the same
                # boundary without sampling or recording a second decision.
                snapshot_sampling_rng(state, generator)
                sampled_ids[row_idx] = segment[0]
                continue
            sample_row = (
                self._sample_minicpmo45_native_duplex_row_steps if batch_feedback
                else self._sample_minicpmo45_native_duplex_row
            )
            sampled = sample_row(
                logits[row_idx : row_idx + 1],
                sampling_metadata,
                row_idx=row_idx,
                token_ids=token_ids,
                temperature=temperatures[row_idx],
                top_k=int(top_ks[row_idx]),
                top_p=top_ps[row_idx],
            )
            if batch_feedback:
                pending.append((row_idx, sampled, None))
                row_rng[row_idx] = generator
                continue
            self._record_minicpmo45_duplex_terminator(row_idx, sampled, token_ids)
            snapshot_sampling_rng(state, generator)
            sampled_ids[row_idx] = sampled
        # Run the same per-row CUDA sampling operations, but collect decisions
        # once per phase instead of .item() twice per user. No extra RNG draw
        # for forced boundaries, P-ended units, or intermediate prefill chunks.
        while pending:
            decisions = []
            for row_idx, steps, feedback in pending:
                try:
                    decision = steps.send(feedback)
                except StopIteration as finished:
                    sampled_ids[row_idx] = finished.value
                    self._record_minicpmo45_duplex_terminator(row_idx, finished.value, token_ids)
                    snapshot_sampling_rng(states[row_idx], row_rng[row_idx])
                else:
                    decisions.append((row_idx, steps, decision))
            draws = self._materialize_minicpmo45_draws([d for _, _, d in decisions])
            values = torch.cat([d.reshape(-1) for d in draws]).cpu().tolist() if draws else []
            pending = [(i, steps, int(value)) for (i, steps, _), value in zip(decisions, values)]
        return DuplexSamplerOutput(
            sampled_token_ids=torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32).unsqueeze(-1),
            logprobs_tensors=None,
            skipped_sampling_request_ids=tuple(skipped_request_ids),
        )

    def _sample_minicpmo45_native_duplex_row(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata, **kwargs: Any,
    ) -> int:
        """Native row-major fallback for shared/unseeded RNG and CPU callers."""
        steps = self._sample_minicpmo45_native_duplex_row_steps(logits, sampling_metadata, **kwargs)
        feedback = None
        while True:
            try:
                draw = self._materialize_minicpmo45_draws([steps.send(feedback)])[0]
                feedback = int(draw.item())
            except StopIteration as finished:
                return finished.value

    def _sample_minicpmo45_native_duplex_row_steps(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        row_idx: int,
        token_ids: dict[str, int],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Generator[torch.Tensor | _MiniCPMFilteredDraw, int | None, int]:
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        generator = getattr(sampling_metadata, "generators", {}).get(row_idx)
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None) or []
        raw_recent_tokens = output_token_ids[row_idx] if row_idx < len(output_token_ids) else []
        recent_tokens = [int(token_id) for token_id in raw_recent_tokens if isinstance(token_id, int) and token_id >= 0]
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if getattr(self, "_minicpmo_pd_decode", False) and state is not None:
            # Include P's first sampled token in native chunk/character limits.
            recent_tokens = list(state.current_segment_output_tokens)
        if chunk_eos_id >= 0 and chunk_eos_id < logits.shape[-1]:
            max_speak_tokens = int(
                getattr(
                    self,
                    "max_new_speak_tokens_per_chunk",
                    MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK,
                )
                or MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK
            )
            request_max_tokens = self._minicpmo45_duplex_row_request_max_tokens(row_idx)
            effective_max_speak_tokens = max_speak_tokens
            if request_max_tokens is not None:
                effective_max_speak_tokens = min(effective_max_speak_tokens, request_max_tokens)
            if len(recent_tokens) >= max(1, effective_max_speak_tokens - 1):
                return int(chunk_eos_id)

            # Match the released StreamDecoder: first sample the original
            # distribution only to preserve the model's own chunk boundary.
            # If it does not choose chunk_eos, mask that token before the
            # normal text/listen/turn sampling pass below.
            if getattr(sampling_metadata, "all_greedy", False):
                boundary_sample = yield torch.argmax(logits, dim=-1)
            else:
                boundary_sample = yield _MiniCPMFilteredDraw(logits, 0, 1.0, generator)
            if boundary_sample == chunk_eos_id:
                return int(chunk_eos_id)

        generated_tokens = getattr(state, "generated_tokens", None)
        repetition_tokens = generated_tokens if generated_tokens else recent_tokens
        sampled = yield _MiniCPMFilteredDraw(
            logits, top_k, top_p, generator, token_ids=token_ids,
            repetition_tokens=tuple(repetition_tokens[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE:]),
            temperature=temperature,
            greedy=bool(getattr(sampling_metadata, "all_greedy", False) or temperature <= 0),
        )
        self._record_minicpmo45_duplex_generation_token(row_idx, sampled)
        return self._finalize_minicpmo45_native_duplex_sample(
            row_idx,
            sampled,
            token_ids,
        )

    def _materialize_minicpmo45_draws(
        self, decisions: list[torch.Tensor | _MiniCPMFilteredDraw]
    ) -> list[torch.Tensor]:
        # The caller only groups independent session RNG streams. Shared or
        # unseeded streams retain the original one-row, boundary-then-text
        # order. Native batched draws retain each independent row's RNG stream.
        outputs: dict[int, torch.Tensor] = {}
        groups: dict[tuple, list[tuple[int, _MiniCPMFilteredDraw]]] = {}
        for i, decision in enumerate(decisions):
            if isinstance(decision, torch.Tensor):
                outputs[i] = decision
            else:
                policy = None if decision.token_ids is None else tuple(sorted(decision.token_ids.items()))
                key = (decision.top_k, decision.top_p, decision.temperature, decision.greedy, policy)
                groups.setdefault(key, []).append((i, decision))
        for (top_k, top_p, temperature, greedy, _), group in groups.items():
            logits = group[0][1].logits if len(group) == 1 else torch.cat([d.logits for _, d in group])
            token_ids = group[0][1].token_ids
            if token_ids is not None:
                if len(group) == 1:
                    logits = logits.clone()
                self._mask_minicpmo45_forbidden_tokens(logits, token_ids)
                if len(group) == 1:
                    self._apply_minicpmo45_repetition_penalty(logits, group[0][1].repetition_tokens, 1.05)
                else:
                    pairs = [
                        (row, token) for row, (_, decision) in enumerate(group)
                        for token in set(decision.repetition_tokens)
                        if 0 <= token < logits.shape[-1]
                    ]
                    if pairs:
                        rows, columns = torch.tensor(pairs, device=logits.device, dtype=torch.long).unbind(1)
                        logits[rows, columns] = logits[rows, columns] / 1.05
                if not greedy:
                    logits = logits / temperature
            if greedy:
                draws = torch.argmax(logits, dim=-1).reshape(-1, 1)
                for row, (i, _) in enumerate(group):
                    outputs[i] = draws[row : row + 1]
                continue
            probs = F.softmax(self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p), dim=-1)
            generators = {row: decision.generator for row, (_, decision) in enumerate(group)}
            if (
                probs.is_cuda and len(group) > 1
                and all(g is not None for g in generators.values())
                and len({id(g) for g in generators.values()}) == len(group)
            ):
                draws = random_sample(probs, generators, use_fp64_gumbel=False).reshape(-1, 1)
                for row, (i, _) in enumerate(group):
                    outputs[i] = draws[row : row + 1]
                continue
            for row, (i, decision) in enumerate(group):
                outputs[i] = self._draw_minicpmo45_token(probs[row : row + 1], decision.generator)
        return [outputs[i] for i in range(len(decisions))]

    @staticmethod
    def _draw_minicpmo45_token(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if not probs.is_cuda:
            return torch.multinomial(probs, num_samples=1, generator=generator)
        # These are internal softmax probabilities. Reuse vLLM's draw for the
        # same exponential noise/RNG sequence without multinomial's repeated
        # full-vocabulary validity reductions. The probability buffer is dead
        # after this draw; random_sample may safely modify it in place.
        return random_sample(
            probs, {0: generator} if generator is not None else {}, use_fp64_gumbel=False
        ).reshape(1, 1)

    @staticmethod
    def _apply_minicpmo45_repetition_penalty(
        logits: torch.Tensor, history: list[int] | tuple[int, ...], penalty: float,
    ) -> None:
        # Preserve the released decoder's rule: divide either sign once per
        # distinct token in the recent history, including already-masked logits.
        # A scalar CUDA update per token otherwise launches hundreds of kernels
        # for every row of every decode step. No RNG or sampling policy changes.
        token_ids = [
            token_id
            for token_id in set(history[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :])
            if 0 <= token_id < logits.shape[-1]
        ]
        if token_ids:
            indices = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
            logits[0, indices] = logits[0, indices] / penalty

    def _minicpmo45_duplex_state_for_row(self, row_idx: int):
        if getattr(self, "_minicpmo_pd_decode", False):
            request_id = getattr(self, "_minicpmo45_duplex_row_requests", {}).get(row_idx)
            return getattr(self, "_minicpmo45_pd_sampling_states", {}).get(request_id)
        row_sessions = getattr(self, "_minicpmo45_duplex_row_sessions", None)
        session_key = row_sessions.get(row_idx) if isinstance(row_sessions, dict) else None
        if not session_key:
            return None
        helper = getattr(self, "_minicpmo45_duplex_data_plane_helper", None)
        sessions = getattr(helper, "sessions", None) if helper is not None else None
        return sessions.get(session_key) if isinstance(sessions, dict) else None

    def snapshot_duplex_sampling_outputs(self, request_ids: list[str]) -> dict[str, Any]:
        """Freeze CPU policy state after sampling, before async cleanup/output."""
        if not getattr(self, "_minicpmo_pd_thinker", False):
            return {}
        rows = {row.request_id: row for row in getattr(self, "_minicpmo45_sampling_rows", ())}
        snapshots = []
        for request_id in request_ids:
            row = rows.get(request_id)
            state = (
                self._minicpmo45_duplex_state_for_row(row.row_idx) if row is not None and row.should_sample else None
            )
            snapshots.append(
                # This is integer metadata, not a tensor for model execution.
                # A frozen tuple also avoids CPU tensor cloning/serialization
                # repeatedly releasing and reacquiring the GIL on output threads.
                tuple(pack_sampling_state(state, incarnation=row.incarnation, epoch=row.epoch, seq=row.seq))
                if state is not None
                else None
            )
        return {SAMPLING_STATE_WIRE_KEY: snapshots} if any(item is not None for item in snapshots) else {}

    def _minicpmo45_duplex_payload_for_row(self, row_idx: int) -> dict[str, Any] | None:
        row_payloads = getattr(self, "_minicpmo45_duplex_row_payloads", None)
        payload = row_payloads.get(row_idx) if isinstance(row_payloads, dict) else None
        return payload if isinstance(payload, dict) else None

    def _minicpmo45_duplex_row_request_max_tokens(self, row_idx: int) -> int | None:
        row_max_tokens = getattr(self, "_minicpmo45_duplex_row_max_tokens", None)
        value = row_max_tokens.get(row_idx) if isinstance(row_max_tokens, dict) else None
        try:
            max_tokens = int(value)
        except (TypeError, ValueError):
            return None
        return max_tokens if max_tokens > 0 else None

    def _finalize_minicpmo45_native_duplex_sample(
        self,
        row_idx: int,
        sampled: int,
        token_ids: dict[str, int],
    ) -> int:
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        if (
            sampled == listen_id
            and 0 <= tts_bos_id
            and state is not None
            and not getattr(state, "current_turn_ended", True)
            and not force_listen
        ):
            return int(tts_bos_id)
        return int(sampled)

    def _record_minicpmo45_duplex_generation_token(self, row_idx: int, sampled: int) -> None:
        """Track tokens returned by the model-policy decoder.

        The released StreamDecoder is constructed without special_token_ids, so
        every normally decoded token participates in repetition penalty. Forced
        listen bypasses that decoder, while chunk_eos boundary decisions return
        before this method is called.
        """
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if state is None:
            return
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        if isinstance(payload, dict) and payload.get("force_listen") is True:
            return
        generated_tokens = getattr(state, "generated_tokens", None)
        if not isinstance(generated_tokens, list):
            generated_tokens = []
            state.generated_tokens = generated_tokens
        generated_tokens.append(int(sampled))
        history_size = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
        del generated_tokens[:-history_size]

    def _record_minicpmo45_duplex_terminator(self, row_idx: int, sampled: int, token_ids: dict[str, int]) -> None:
        """Remember sampled unit state for the next append.

        The scheduler session update discards the final sampled token of a
        segment before the next streaming update, but the official duplex
        format feeds it (terminator + </unit>) into the KV at every unit
        boundary, and the model's listen/speak policy depends on seeing its own
        past decisions. Non-terminators clear the turn-ended latch."""
        state = self._minicpmo45_duplex_state_for_row(row_idx)
        if state is None:
            return
        segment_tokens = getattr(state, "current_segment_output_tokens", None)
        if not isinstance(segment_tokens, list):
            segment_tokens = []
            state.current_segment_output_tokens = segment_tokens
        segment_tokens.append(int(sampled))
        payload = self._minicpmo45_duplex_payload_for_row(row_idx)
        force_listen = isinstance(payload, dict) and payload.get("force_listen") is True
        listen_id = token_ids.get("listen_token_id", -1)
        tts_bos_id = token_ids.get("tts_bos_token_id", -1)
        chunk_eos_id = token_ids.get("chunk_eos_token_id", -1)
        chunk_tts_eos_id = token_ids.get("chunk_tts_eos_token_id", -1)
        turn_eos_id = token_ids.get("turn_eos_token_id", -1)
        terminators = {listen_id, chunk_eos_id, chunk_tts_eos_id, turn_eos_id}
        if sampled in terminators:
            state.pending_terminator_token = int(sampled)
            state.last_terminator_token = int(sampled)
            if sampled == turn_eos_id or (sampled == listen_id and force_listen):
                state.current_turn_ended = True
                with suppress(Exception):
                    state.pending_speech_response_open = False
            return
        if (
            sampled == tts_bos_id
            and getattr(state, "current_turn_ended", True)
            and getattr(state, "pending_speech_context", False)
        ):
            with suppress(Exception):
                state.pending_speech_response_open = True
        elif getattr(state, "pending_speech_response_open", False):
            with suppress(Exception):
                state.pending_speech_context = False
                state.pending_speech_response_open = False
        elif getattr(state, "current_turn_ended", True):
            with suppress(Exception):
                state.pending_speech_context = False
        state.pending_terminator_token = None
        state.last_terminator_token = None
        state.current_turn_ended = False

    def _minicpmo45_tokenizer(self):
        if hasattr(self, "_minicpmo45_tokenizer_cache"):
            return self._minicpmo45_tokenizer_cache
        tokenizer = None
        get_tokenizer = getattr(getattr(self, "thinker", None), "get_tokenizer", None)
        if callable(get_tokenizer):
            tokenizer = get_tokenizer()
        if tokenizer is None:
            try:
                from vllm.tokenizers import cached_tokenizer_from_config

                tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
            except Exception:
                pass
        self._minicpmo45_tokenizer_cache = tokenizer
        return tokenizer

    def _minicpmo45_native_duplex_token_ids(self) -> dict[str, int]:
        cached = getattr(self, "_minicpmo45_native_duplex_token_ids_cache", None)
        if isinstance(cached, dict):
            return cached
        tokenizer = self._minicpmo45_tokenizer()
        cached = MiniCPMO45DuplexPolicy.token_ids_from_tokenizer(tokenizer)
        self._minicpmo45_native_duplex_token_ids_cache = cached
        return cached

    def _minicpmo45_native_duplex_prompt_rows(
        self,
        sampling_metadata: SamplingMetadata,
        unit_id: int,
        batch_size: int,
        *,
        duplex_rows: list[int] | None = None,
    ) -> list[int]:
        if duplex_rows is not None:
            rows: list[int] = []
            for row in duplex_rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                if 0 <= row_idx < batch_size:
                    rows.append(row_idx)
            return rows

        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return []
        if prompt_token_ids.ndim == 1:
            prompt_token_ids = prompt_token_ids.unsqueeze(0)
        rows: list[int] = []
        for row_idx in range(min(batch_size, int(prompt_token_ids.shape[0]))):
            row = prompt_token_ids[row_idx]
            if torch.count_nonzero(row == unit_id).item() >= 2:
                rows.append(row_idx)
        return rows

    def _minicpmo45_native_forbidden_token_ids(self, token_ids: dict[str, int]) -> list[int]:
        tokenizer = self._minicpmo45_tokenizer()
        bad_token_ids = getattr(tokenizer, "bad_token_ids", []) if tokenizer is not None else []
        return MiniCPMO45DuplexPolicy.native_forbidden_token_ids(token_ids, bad_token_ids=bad_token_ids)

    def _mask_minicpmo45_forbidden_tokens(self, logits: torch.Tensor, token_ids: dict[str, int]) -> None:
        # Cache only derived indices, never session state. The content key also
        # detects an in-place tokenizer/config change; dtype does not affect IDs.
        forbidden = tuple(self._minicpmo45_native_forbidden_token_ids(token_ids))
        vocab_size = logits.shape[-1]
        key = (logits.device, vocab_size, forbidden)
        cached = getattr(self, "_minicpmo45_forbidden_index_cache", None)
        if cached is None or cached[0] != key:
            indices = torch.tensor(
                [token_id for token_id in forbidden if 0 <= token_id < vocab_size],
                dtype=torch.long,
                device=logits.device,
            )
            cached = self._minicpmo45_forbidden_index_cache = (key, indices)
        if cached[1].numel():
            logits.index_fill_(-1, cached[1], float("-inf"))

    def _minicpmo45_native_special_token_ids(self, token_ids: dict[str, int]) -> set[int]:
        tokenizer = self._minicpmo45_tokenizer()
        return MiniCPMO45DuplexPolicy.native_special_token_ids(
            token_ids,
            tokenizer_special_ids=getattr(tokenizer, "all_special_ids", []) if tokenizer is not None else [],
        )

    @staticmethod
    def _sampling_metadata_values(
        sampling_metadata: SamplingMetadata,
        name: str,
        batch_size: int,
        default: float,
    ) -> list[float]:
        """Read a sampling field with at most one device-to-host sync."""
        if batch_size <= 0:
            return []
        value = getattr(sampling_metadata, name, None)
        if value is None:
            return [default] * batch_size
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return [default] * batch_size
            raw_values = value.detach().reshape(-1).cpu().tolist()
            return [float(raw_values[min(row_idx, len(raw_values) - 1)]) for row_idx in range(batch_size)]
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            scalar = default
        return [scalar] * batch_size

    @staticmethod
    def _top_k_top_p_filter(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0 and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []

        # MiniCPM-o checkpoint prefixes → stage mapping:
        #   thinker: vpm, resampler, llm, apm, audio_projection_layer
        #   talker:  tts (native MiniCPMTTS AR codec producer)
        for k, v in weights:
            if k.startswith(("vpm.", "resampler.", "llm.", "apm.", "audio_projection_layer.")):
                thinker_weights.append((k, v))
            elif k.startswith("tts."):
                talker_weights.append((k, v))
            else:
                logger.warning("Unknown weight prefix: %s, skipping", k)

        # Load thinker weights
        if self.thinker is not None and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker is not None and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)

        return loaded_weights
