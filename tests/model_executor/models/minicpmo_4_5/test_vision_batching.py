from dataclasses import dataclass

import pytest
import torch

from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _VisionEncoderOutput:
    last_hidden_state: torch.Tensor


@dataclass
class _VisionConfig:
    vision_batch_size: int


@dataclass
class _AudioConfig:
    audio_chunk_length: float = 0
    audio_pool_step: int = 1


@dataclass
class _FakeConv:
    weight: torch.Tensor


@dataclass
class _AudioEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


class _FakeVisionEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> _VisionEncoderOutput:
        batch_size = pixel_values.shape[0]
        self.batch_sizes.append(batch_size)
        assert patch_attention_mask.shape[0] == batch_size
        assert tgt_sizes.shape[0] == batch_size

        per_item = pixel_values.sum(dim=(1, 2, 3))
        hidden_states = per_item[:, None, None].expand(-1, 3, 4).clone()
        return _VisionEncoderOutput(last_hidden_state=hidden_states)


class _FakeResampler:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, hidden_states: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(hidden_states.shape[0])
        return hidden_states + tgt_sizes.sum(dim=1)[:, None, None]


class _FakeAudioEncoder:
    def __init__(self) -> None:
        self.conv1 = _FakeConv(weight=torch.empty(1, dtype=torch.float32))
        self.output_hidden_states: list[bool] = []

    def __call__(
        self,
        input_features: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> _AudioEncoderOutput:
        self.output_hidden_states.append(output_hidden_states)
        assert attention_mask.shape == (1, 1, 2, 2)
        final = torch.full((1, 2, 4), 2.0)
        earlier = torch.full((1, 2, 4), 1.0)
        return _AudioEncoderOutput(
            last_hidden_state=final,
            hidden_states=(earlier, final) if output_hidden_states else None,
        )


class _FakeVisionModel:
    def __init__(self, vision_batch_size: int) -> None:
        self.config = _VisionConfig(vision_batch_size=vision_batch_size)
        self.vpm = _FakeVisionEncoder()
        self.resampler = _FakeResampler()


class _FakeAudioModel:
    def __init__(self, audio_encoder: _FakeAudioEncoder, audio_encoder_layer: int) -> None:
        self.config = _AudioConfig()
        self.apm = audio_encoder
        self.audio_projection_layer = torch.nn.Identity()
        self.audio_avg_pooler = torch.nn.Identity()
        self.audio_encoder_layer = audio_encoder_layer

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        return input_lengths_after_cnn, input_lengths_after_pooling.to(dtype=torch.int32)


def _run_vision_encoder(vision_batch_size: int) -> tuple[torch.Tensor, list[int], list[int]]:
    model = _FakeVisionModel(vision_batch_size)
    pixel_values = [
        torch.full((3, 2, length), fill_value=index + 1, dtype=torch.float32)
        for index, length in enumerate((4, 3, 2, 4, 1))
    ]
    tgt_sizes = torch.tensor([[1, 2], [1, 2], [1, 1], [2, 2], [1, 1]])

    result = MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(
        model,
        {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes},
    )
    return result, model.vpm.batch_sizes, model.resampler.batch_sizes


def test_vision_encoder_batches_vpm_before_resampling() -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(2)
    unchunked, unchunked_vision_batches, unchunked_resampler_batches = _run_vision_encoder(16)

    assert vision_batches == [2, 2, 1]
    assert resampler_batches == [5]
    assert unchunked_vision_batches == [5]
    assert unchunked_resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


@pytest.mark.parametrize("vision_batch_size", [0, -2])
def test_vision_encoder_clamps_nonpositive_batch_size(vision_batch_size: int) -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(vision_batch_size)
    unchunked, _, _ = _run_vision_encoder(16)

    assert vision_batches == [1, 1, 1, 1, 1]
    assert resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


class _FakeShapeBucketVisionTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vpm = torch.nn.Linear(1, 1, bias=False)
        self.calls: list[tuple[list[tuple[int, ...]], list[list[int]], list[int]]] = []

    def get_vision_hidden_states(self, data: dict[str, object]) -> torch.Tensor:
        pixels = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]
        assert isinstance(pixels, list)
        assert isinstance(tgt_sizes, torch.Tensor)
        tags = [int(pixel[0, 0, 0].item()) for pixel in pixels]
        self.calls.append(
            (
                [tuple(pixel.shape) for pixel in pixels],
                tgt_sizes.tolist(),
                tags,
            )
        )
        return torch.stack(
            [torch.full((64, 2), float(tag)) for tag in tags],
            dim=0,
        )


def _shape_bucket_processed(*tags: int) -> dict[str, object]:
    assert len(tags) == 3
    return {
        "pixel_values": [
            [
                torch.full((3, 14, 14448), float(tags[0])),
                torch.full((3, 14, 14280), float(tags[1])),
                torch.full((3, 14, 14280), float(tags[2])),
            ]
        ],
        "tgt_sizes": [torch.tensor([[43, 24], [30, 34], [30, 34]])],
    }


def test_stage0_vision_batch_groups_equal_shapes_and_restores_request_order() -> None:
    target = _FakeShapeBucketVisionTarget()
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = target
    runtime.thinker = target

    result = runtime._stage_vision_embeddings_batch(
        [_shape_bucket_processed(1, 2, 3), _shape_bucket_processed(4, 5, 6)],
        microbatch_size=2,
    )

    assert result is not None
    assert [int(block[0, 0].item()) for request in result for frame in request for block in frame] == [1, 2, 3, 4, 5, 6]
    assert target.calls == [
        (
            [(3, 14, 14448), (3, 14, 14448)],
            [[43, 24], [43, 24]],
            [1, 4],
        ),
        (
            [(3, 14, 14280), (3, 14, 14280)],
            [[30, 34], [30, 34]],
            [2, 3],
        ),
        (
            [(3, 14, 14280), (3, 14, 14280)],
            [[30, 34], [30, 34]],
            [5, 6],
        ),
    ]


def test_arrival_vision_cache_is_epoch_fenced_and_atomically_consumed() -> None:
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    first = [torch.full((2, 2), 1.0)]
    second = [torch.full((2, 2), 2.0)]

    assert (
        runtime.cache_arrival_vision_embeddings(
            session_id="session-a",
            incarnation=3,
            epoch=4,
            preencode_ids=["frame-1", "frame-2"],
            frame_blocks=[first, second],
        )
        == 2
    )
    assert (
        runtime.take_arrival_vision_embeddings(
            session_id="session-a",
            incarnation=3,
            epoch=4,
            preencode_ids=["frame-1", "missing"],
        )
        is None
    )
    cached = runtime.take_arrival_vision_embeddings(
        session_id="session-a",
        incarnation=3,
        epoch=4,
        preencode_ids=["frame-1", "frame-2"],
    )
    assert cached is not None
    torch.testing.assert_close(cached[0][0], first[0])
    torch.testing.assert_close(cached[1][0], second[0])
    assert (
        runtime.take_arrival_vision_embeddings(
            session_id="session-a",
            incarnation=3,
            epoch=4,
            preencode_ids=["frame-1"],
        )
        is None
    )

    runtime.cache_arrival_vision_embeddings(
        session_id="session-a",
        incarnation=3,
        epoch=5,
        preencode_ids=["new-frame"],
        frame_blocks=[first],
    )
    assert (
        runtime.take_arrival_vision_embeddings(
            session_id="session-a",
            incarnation=3,
            epoch=4,
            preencode_ids=["new-frame"],
        )
        is None
    )


@pytest.mark.parametrize(
    ("audio_encoder_layer", "expected_value", "expected_hidden_states"),
    [(-1, 2.0, False), (0, 1.0, True)],
)
def test_audio_encoder_retains_layers_only_for_nonfinal_selection(
    audio_encoder_layer: int,
    expected_value: float,
    expected_hidden_states: bool,
) -> None:
    audio_encoder = _FakeAudioEncoder()
    model = _FakeAudioModel(audio_encoder, audio_encoder_layer)

    result = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states(
        model,
        {
            "audio_features": torch.zeros((1, 80, 4)),
            "audio_feature_lens": [torch.tensor([4])],
        },
    )

    assert audio_encoder.output_hidden_states == [expected_hidden_states]
    torch.testing.assert_close(result[0], torch.full((2, 4), expected_value))


def test_uniform_vision_fast_path_matches_full_mask_path() -> None:
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        num_channels=3,
    )
    config._attn_implementation = "eager"
    vision = SiglipVisionTransformer(config).eval()
    resampler = Resampler(
        num_queries=4,
        embed_dim=16,
        num_heads=2,
        kv_dim=16,
        max_size=(4, 4),
    ).eval()
    pixels = torch.randn(3, 3, 14, 56)
    target_sizes = torch.tensor([[2, 2]] * 3)
    full_mask = torch.ones(3, 1, 4, dtype=torch.bool)

    with torch.inference_mode():
        regular = vision(
            pixels,
            patch_attention_mask=full_mask,
            tgt_sizes=target_sizes,
        ).last_hidden_state
        uniform = vision.forward_uniform(pixels, 2, 2).last_hidden_state
        regular_resampled = resampler(regular, target_sizes)
        uniform_resampled = resampler.forward_uniform(uniform, 2, 2)

    torch.testing.assert_close(uniform, regular, rtol=0, atol=0)
    torch.testing.assert_close(uniform_resampled, regular_resampled, rtol=0, atol=0)


class _StreamingAudioHarness:
    audio_cache_seq_length = staticmethod(MiniCPMO45OmniLLMForConditionalGeneration.audio_cache_seq_length)
    _audio_self_attention_cache = staticmethod(MiniCPMO45OmniLLMForConditionalGeneration._audio_self_attention_cache)
    combine_audio_past_key_values = classmethod(
        MiniCPMO45OmniLLMForConditionalGeneration.combine_audio_past_key_values.__func__
    )
    split_audio_past_key_values = staticmethod(MiniCPMO45OmniLLMForConditionalGeneration.split_audio_past_key_values)
    get_audio_embedding_streaming_batch = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming_batch

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        output_lengths = (input_lengths - 1) // 2 + 1
        return output_lengths, output_lengths


def test_streaming_audio_batch_matches_independent_cached_sessions() -> None:
    from transformers.models.whisper.modeling_whisper import WhisperConfig

    config = WhisperConfig(
        num_mel_bins=8,
        d_model=16,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        max_source_positions=64,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    config._attn_implementation = "eager"
    harness = _StreamingAudioHarness()
    harness.apm = MiniCPMWhisperEncoder(config).eval()
    harness.audio_projection_layer = torch.nn.Identity()
    harness.audio_avg_pooler = torch.nn.Identity()
    harness.audio_encoder_layer = -1
    first = torch.randn(2, 8, 17)
    second = torch.randn(2, 8, 17)
    lengths = [torch.tensor([17]), torch.tensor([17])]

    with torch.inference_mode():
        batched_first, batched_cache = harness.get_audio_embedding_streaming_batch(
            {"audio_features": first, "audio_feature_lens": lengths}
        )
        serial_first = []
        serial_caches = []
        for row in range(2):
            output, cache = harness.get_audio_embedding_streaming_batch(
                {
                    "audio_features": first[row : row + 1],
                    "audio_feature_lens": [lengths[row]],
                }
            )
            serial_first.append(output[0][0])
            serial_caches.append(cache)

        split_cache = harness.split_audio_past_key_values(batched_cache, 2)
        batched_second, batched_cache = harness.get_audio_embedding_streaming_batch(
            {"audio_features": second, "audio_feature_lens": lengths},
            past_key_values=harness.combine_audio_past_key_values(split_cache),
        )
        serial_second = []
        for row in range(2):
            output, _cache = harness.get_audio_embedding_streaming_batch(
                {
                    "audio_features": second[row : row + 1],
                    "audio_feature_lens": [lengths[row]],
                },
                past_key_values=serial_caches[row],
            )
            serial_second.append(output[0][0])

    for row in range(2):
        torch.testing.assert_close(
            batched_first[row][0],
            serial_first[row],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            batched_second[row][0],
            serial_second[row],
            rtol=1e-5,
            atol=1e-5,
        )
    assert harness.audio_cache_seq_length(batched_cache) == 18
