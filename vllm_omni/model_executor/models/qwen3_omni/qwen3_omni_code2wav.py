# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
)
from vllm.config import VllmConfig  # type: ignore
from vllm.logger import init_logger  # type: ignore
from vllm.model_executor.models.utils import (  # type: ignore
    AutoWeightsLoader,
    WeightsMapper,
)

from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta

logger = init_logger(__name__)

# [live-vllm P5] Conv window margin in codec frames for the windowed streaming
# decode: 10 = the conv/upsample stack's MEASURED left receptive field
# (autograd probe: emitted samples of frame q depend on input frames
# [q-10, q+1]) + 1 frame covering the 555-sample tail refill and the
# transposed-conv lookahead. The 25-frame left_context the caller ships is
# sized for the PRE-TRANSFORMER's attention and still applies there.
_CONV_WINDOW_MARGIN_FRAMES = 11
try:
    from vllm_omni.core.sched.temporal_pacing import live_env_on as _live_env_on

    _STREAM_VOCODER = _live_env_on("VLLM_OMNI_STREAM_VOCODER")
except Exception:  # pragma: no cover -- standalone/partial installs
    _STREAM_VOCODER = os.environ.get("VLLM_OMNI_STREAM_VOCODER", "0") not in (
        "0", "", "false", "False")


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.

    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform

    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]

    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """

    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config: Qwen3OmniMoeCode2WavConfig = vllm_config.model_config.hf_config

        # Calculate total upsampling factor
        self.total_upsample = np.prod(self.config.upsample_rates + self.config.upsampling_ratios)

        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(self.config)

        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )

        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [Qwen3OmniMoeCausalConvNet(self.config.hidden_size, self.config.decoder_dim, kernel_size=7)]

        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))

        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

        # CUDA Graph support — reuses CUDAGraphDecoderWrapper from Qwen3-TTS
        self._cudagraph_enabled = False
        self._cudagraph_wrapper = None

    def precompute_snake_caches(self):
        """Precompute exp(alpha) and 1/(exp(beta)+eps) for all SnakeBeta modules."""
        count = 0
        for module in self.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()
                count += 1
        if count > 0:
            logger.info("Precomputed exp caches for %d SnakeBeta activations", count)

    def enable_cudagraph(
        self,
        device: torch.device | None = None,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ):
        """Enable CUDA graph acceleration (same pattern as Qwen3-TTS Code2Wav)."""
        from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
            CUDAGraphDecoderWrapper,
        )

        if device is None:
            device = next(self.parameters()).device
        if device.type != "cuda":
            logger.warning("Cannot enable CUDA Graph: not on CUDA device (got %s)", device)
            return

        wrapper = CUDAGraphDecoderWrapper(
            decoder=self,
            num_quantizers=self.config.num_quantizers,
            enabled=True,
        )
        try:
            wrapper.warmup(
                device,
                dtype=torch.long,
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )
        except Exception:
            self._cudagraph_wrapper = None
            self._cudagraph_enabled = False
            raise
        self._cudagraph_wrapper = wrapper
        self._cudagraph_enabled = True
        logger.info(
            "CUDA Graph enabled for Code2Wav: num_quantizers=%d, sizes=%s",
            self.config.num_quantizers,
            self._cudagraph_wrapper.capture_sizes,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes

        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")
        return self._forward_convs(self._forward_features(codes))

    def _forward_features(self, codes: torch.Tensor) -> torch.Tensor:
        """Stages 1+2: code embedding + pre-transformer. [B,Q,T] -> [B,T,H].

        Runs at codec-frame rate (12.5 Hz) -- the cheap half. Split out so the
        streaming path can window the CONV stack without touching attention
        semantics (the 25-frame left context is sized for attention, and its
        truncated-window recompute is the quality-approved behavior).
        """
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        return self.pre_transformer(inputs_embeds=hidden).last_hidden_state

    def _forward_convs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stages 3+4: upsampling + decoder + clamp. [B,T,H] -> [B,1,samples].

        Runs at up-to-1920x upsampled resolution -- the FLOPs-dominant half.
        Output length is T*total_upsample - 555 (the causal stack's structural
        right trim; see chunked_decode_streaming's tail-refill note).
        """
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.
        When CUDA graphs are enabled, delegates chunk-level decoding to the
        CUDAGraphDecoderWrapper for reduced kernel launch overhead.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        # Use CUDA graph wrapper for chunk-level decode when available
        if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
            batch_wav = self._cudagraph_wrapper.chunked_decode_with_cudagraph(codes, chunk_size, left_context_size)
        else:
            wavs = []
            start_index = 0

            while start_index < codes.shape[-1]:
                end_index = min(start_index + chunk_size, codes.shape[-1])
                context_size = left_context_size if start_index >= left_context_size else start_index

                # Extract chunk with left context
                codes_chunk = codes[..., start_index - context_size : end_index]

                # Decode chunk
                wav_chunk = self(codes_chunk)

                # Remove context from output (context_size * total_upsample samples)
                wavs.append(wav_chunk[..., context_size * self.total_upsample :])

                start_index = end_index

            batch_wav = torch.cat(wavs, dim=-1)

        if seq_token_counts is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        result = []
        for idx, code_seq_len in enumerate(code_seq_lens):
            wav_chunk = batch_wav[idx, :, : code_seq_len * self.total_upsample]
            result.append(wav_chunk)
        return result

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        left_context_size: list[int] | None = None,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        No longer need chunk size here, which is different from chunked_decode

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        meta_ok = bool(
            left_context_size and seq_token_counts
            and len(left_context_size) == len(seq_token_counts)
        )
        if not meta_ok:
            logger.warning_once(
                "chunked_decode_streaming: missing/invalid left_context_size or seq_token_counts; "
                "defaulting to left_context_size=zeros(len(codes)). This is expected during cudagraph warmup."
            )
            left_context_size = [0] * codes.shape[0]
        # [live-vllm P5] windowed streaming decode: same features, conv stack
        # only over the last (new + margin) frames per row. Precedence vs the
        # CUDA-graph wrapper: the wrapper only ever CAPTURES batch=1 -- at
        # B>1 it falls back to the eager full-window forward internally, so
        # the windowed path strictly wins there (measured 1.85x at B=8). At
        # B=1 the captured graph keeps its launch-overhead win; leave it.
        _graphs_active = self._cudagraph_enabled and self._cudagraph_wrapper is not None
        if (_STREAM_VOCODER and meta_ok
                and (not _graphs_active or codes.shape[0] > 1)):
            lens = [n // self.config.num_quantizers for n in seq_token_counts]
            return self._windowed_streaming_decode(codes, left_context_size, lens)
        # Decode chunk
        wavs = []
        if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
            batch_wav = self._cudagraph_wrapper.decode(codes)
        else:
            batch_wav = self(codes)
        if seq_token_counts is not None:
            code_seq_lens = [n // self.config.num_quantizers for n in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        for idx, code_seq_len in enumerate(code_seq_lens):
            # The eager decoder (self(codes)) emits a fixed `tail` fewer samples than
            # code_seq_len*total_upsample: the causal conv stack trims the sequence end
            # because it lacks right context. The original slice assumed full length, so
            # under async_chunk every streaming chunk silently dropped its last `tail`
            # samples -> a ~23ms gap at every chunk boundary + cumulative ~1.2% time
            # compression. Shift the start back by `tail` so this chunk refills the gap the
            # previous chunk left (those frames are re-decoded here with proper context).
            # `tail` is measured per call, so this is a no-op only when the decoder returns
            # the nominal full length (tail==0, e.g. a C==0 decoder like Qwen3-TTS); for
            # Qwen3-Omni Code2Wav both the eager forward AND the CUDA-graph wrapper
            # (_trim_replay_output -> actual*upsample - 555) are short, so it applies to both.
            tail = max(0, int(code_seq_len * self.total_upsample) - batch_wav.shape[-1])
            start = max(0, left_context_size[idx] * self.total_upsample - tail)
            wav_chunk = batch_wav[idx, :, start : code_seq_len * self.total_upsample]
            wavs.append(wav_chunk)
        return wavs

    def _windowed_streaming_decode(
        self,
        codes: torch.Tensor,
        left_context_size: list[int],
        code_seq_lens: list[int],
    ) -> list[torch.Tensor]:
        """[live-vllm P5] Full-window features, WINDOWED conv stack.

        The pre-transformer (12.5 Hz, cheap) still sees each row's whole
        left_context+new window -- attention semantics identical to the
        quality-approved path. The conv/upsample stack (up to 1920x resolution,
        the FLOPs) then processes only the last ``new + _CONV_WINDOW_MARGIN``
        frames per row: its measured left receptive field is 10 codec frames,
        so the emitted samples -- the same global span
        [left*U - 555, seq*U - 555) the full path emits -- are computed from
        exactly the same inputs. Steady-state chunks drop from 29 to 15 conv
        frames (-48% on the dominant half); the conv window is also CONSTANT
        across rows regardless of left-context, so mixed batches pack tighter
        than the full path.
        """
        U = int(self.total_upsample)
        feats = self._forward_features(codes)  # [B, maxT, H]
        win: list[int] = []
        for lc, sl in zip(left_context_size, code_seq_lens):
            new = max(0, sl - lc)
            win.append(min(sl, new + _CONV_WINDOW_MARGIN_FRAMES))
        max_w = max(win) if win else 0
        if max_w <= 0:
            return [feats.new_zeros((1, 0)) for _ in code_seq_lens]
        sliced = feats.new_zeros((feats.shape[0], max_w, feats.shape[-1]))
        for i, (sl, w) in enumerate(zip(code_seq_lens, win)):
            if w > 0:
                sliced[i, :w] = feats[i, sl - w : sl]
        wav = self._forward_convs(sliced)  # [B, 1, max_w*U - 555]
        outs: list[torch.Tensor] = []
        for i, (lc, sl, w) in enumerate(zip(left_context_size, code_seq_lens, win)):
            new = max(0, sl - lc)
            # Row-valid output ends at w*U - 555: the structural trim excises
            # exactly the right-context-dependent tail, so nothing emitted can
            # see a shorter row's zero padding. Emit the last new*U samples --
            # the same tail-refill-shifted span the full path ships.
            valid = w * U - 555
            start = max(0, valid - new * U)
            outs.append(wav[i, :, start:valid])
        return outs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "talker."],  # Already loaded above
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            logger.error("Error logging model load summary")

        return loaded
