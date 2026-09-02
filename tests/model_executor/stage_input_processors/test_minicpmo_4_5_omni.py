from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _extract_first_audio_ref,
    llm2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _output(
    *,
    prompt_ids: list[int],
    output_ids: list[int],
    latent: torch.Tensor,
    multimodal_output: dict | None = None,
    token_list: list[int] | None = None,
):
    mm_output = dict(multimodal_output or {})
    mm_output["latent"] = latent
    completion = SimpleNamespace(
        token_ids=output_ids if token_list is None else token_list,
        text="hello",
        multimodal_output=mm_output,
    )
    return SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=prompt_ids,
        outputs=[completion],
    )


def test_extract_first_audio_ref_accepts_dict_stereo_audio() -> None:
    ref = _extract_first_audio_ref(
        {
            "audio": {
                "array": [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                "sampling_rate": 16000,
            }
        }
    )

    assert ref is not None
    waveform, sample_rate = ref
    assert sample_rate == 16000
    assert torch.allclose(waveform, torch.tensor([1.5, 3.5, 5.5]))


def test_plain_chat_handoff_owns_talker_prompt_contract() -> None:
    prompt_ids = [101, 102]
    output_ids = [11, 12]
    latent = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    converted = llm2tts(
        [_output(prompt_ids=prompt_ids, output_ids=output_ids, latent=latent)],
        prompt=[{}],
    )[0]

    info = converted["model_intermediate_buffer"]
    assert info["ids"]["tts"] == output_ids
    assert torch.equal(torch.tensor(info["hidden_states"]["tts"]), latent[2:4])
    assert converted["prompt_token_ids"] == [0, 0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 4


def test_llm2tts_carries_request_ref_audio() -> None:
    latent = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    source = _output(
        prompt_ids=[101, 9001],
        output_ids=[11, 12, 9002],
        latent=latent,
        multimodal_output={
            "meta": {
                "tts_bos_token_id": 9001,
                "tts_eos_token_id": 9002,
            }
        },
    )
    ref_waveform = torch.tensor([0.1, 0.2, 0.3])

    converted = llm2tts(
        [source],
        prompt=[{"multi_modal_data": {"audio": (ref_waveform, 22050)}}],
    )[0]

    info = converted["model_intermediate_buffer"]
    assert info["codes"]["ref"] == ref_waveform.tolist()
    assert info["meta"]["ref_audio_sr"] == 22050
    assert info["ids"]["tts"] == [11, 12]


def test_native_duplex_speak_segment_reaches_split_talker() -> None:
    prompt_ids = [101, 102]
    output_ids = [9304, 21, 22, 9308]
    latent = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        latent=latent,
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "epoch": 3,
                "model_turn_id": 7,
            }
        }
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)[0]

    info = converted["model_intermediate_buffer"]
    assert info["native_duplex"] is True
    assert info["ids"]["tts"] == [21, 22]
    assert converted["prompt_token_ids"] == [0, 0, 0]
    assert info["meta"]["replace_streaming_prompt"] is True
    assert info["meta"]["next_stage_prompt_len"] == 3
    assert info["meta"]["turn_start"] is True
    assert info["meta"]["segment_end"] is True
    assert info["duplex"]["epoch"] == 3
    assert info["duplex"]["turn_id"] == 7


def test_native_duplex_compact_handoff_does_not_copy_full_prompt() -> None:
    prompt_ids = [101] * 1023 + [9304]
    segment_ids = [21, 22, 9308]
    latent = torch.arange((len(prompt_ids) + len(segment_ids)) * 2, dtype=torch.float32).reshape(-1, 2)
    source = _output(
        prompt_ids=prompt_ids,
        output_ids=[999],
        latent=latent,
        multimodal_output={
            "duplex_prompt_len": len(prompt_ids),
            "duplex_last_prompt_token_id": 9304,
            "duplex_segment_token_ids": segment_ids,
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-1",
                "incarnation": 2,
                "epoch": 3,
                "model_turn_id": 7,
            }
        }
    )

    converted = llm2tts([source], prompt=[{}], _streaming_context=context)[0]

    info = converted["model_intermediate_buffer"]
    assert "prompt_token_ids" not in info
    assert "llm_output_token_ids" not in info
    assert "prompt" not in info["ids"]
    assert "output" not in info["ids"]
    assert info["ids"]["tts"] == [21, 22]
    assert info["meta"]["prompt_len"] == len(prompt_ids)
    assert info["meta"]["last_prompt_token"] == 9304
    assert info["meta"]["current_segment_token_ids"] == segment_ids
    assert torch.equal(torch.tensor(info["hidden_states"]["tts"]), latent[-3:-1])


def test_native_duplex_ref_audio_is_published_once_per_session() -> None:
    token_ids = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    segment_ids = [9304, 21, 9308]
    source = _output(
        prompt_ids=[101, 102],
        output_ids=segment_ids,
        latent=torch.zeros((5, 2)),
        multimodal_output={
            "duplex_prompt_len": 2,
            "duplex_last_prompt_token_id": 102,
            "duplex_segment_token_ids": segment_ids,
            "meta": token_ids,
        },
    )
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "session-ref",
                "incarnation": 4,
                "epoch": 1,
                "model_turn_id": 2,
                "runtime_config": {
                    "ref_audio_data": "zczMPc3MTD6amZk+",
                    "ref_audio_format": "pcm_f32le",
                    "ref_audio_sample_rate_hz": 16000,
                    "keep": "value",
                },
                "session_config": {
                    "extra_body": {
                        "ref_audio_data": "duplicate-inline-payload",
                        "keep_nested": True,
                    }
                },
            }
        }
    )
    prompt = [{}]

    first = llm2tts([source], prompt=prompt, _streaming_context=context)[0]["model_intermediate_buffer"]
    second = llm2tts([source], prompt=prompt, _streaming_context=context)[0]["model_intermediate_buffer"]

    assert first["codes"]["ref"] == pytest.approx([0.1, 0.2, 0.3])
    assert first["meta"]["ref_audio_sr"] == 16000
    assert first["meta"]["ref_audio_handle"] == "minicpmo45-ref:session-ref:4"
    assert "codes" not in second
    assert second["meta"]["ref_audio_handle"] == first["meta"]["ref_audio_handle"]
    assert first["duplex"]["runtime_config"] == {"keep": "value"}
    assert first["duplex"]["session_config"]["extra_body"] == {"keep_nested": True}


def test_native_duplex_continuation_appends_only_new_talker_condition() -> None:
    prompt_ids = [101, 102]
    token_ids = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "epoch": 3,
                "model_turn_id": 7,
            }
        }
    )

    first_ids = [9304, 21, 22, 9308]
    first = _output(
        prompt_ids=prompt_ids,
        output_ids=first_ids,
        latent=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )
    second_ids = [*first_ids, 9304, 23, 24, 9308]
    second = _output(
        prompt_ids=prompt_ids,
        output_ids=second_ids,
        latent=torch.arange(40, dtype=torch.float32).reshape(10, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )
    third_ids = [*second_ids, 9304, 25, 26, 9308]
    third = _output(
        prompt_ids=prompt_ids,
        output_ids=third_ids,
        latent=torch.arange(56, dtype=torch.float32).reshape(14, 4),
        multimodal_output={
            "duplex_prompt_token_ids": prompt_ids,
            "meta": token_ids,
        },
    )

    first_input = llm2tts([first], prompt=[{}], _streaming_context=context)[0]
    second_input = llm2tts([second], prompt=[{}], _streaming_context=context)[0]
    context.bridge_states["duplex"]["model_turn_id"] = 8
    third_input = llm2tts([third], prompt=[{}], _streaming_context=context)[0]

    assert first_input["model_intermediate_buffer"]["ids"]["tts"] == [21, 22]
    assert second_input["model_intermediate_buffer"]["ids"]["tts"] == [23, 24]
    assert third_input["model_intermediate_buffer"]["ids"]["tts"] == [25, 26]
    assert first_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["turn_start"] is False
    assert third_input["model_intermediate_buffer"]["meta"]["turn_start"] is True
    assert first_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert "replace_streaming_prompt" not in second_input["model_intermediate_buffer"]["meta"]
    assert third_input["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    assert second_input["model_intermediate_buffer"]["meta"]["next_stage_prompt_len"] == 3
    assert second_input["prompt_token_ids"] == [0, 0, 0]


def test_native_duplex_transcript_decodes_the_talker_condition_slice() -> None:
    prompt_ids = [101, 102]
    metadata = {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }
    token_text = {21: "杭州", 22: "和", 23: "州和", 24: "上海之间", 25: "大"}

    def output(token_ids: list[int]):
        return _output(
            prompt_ids=prompt_ids,
            output_ids=token_ids,
            latent=torch.zeros((len(prompt_ids) + len(token_ids), 1)),
            multimodal_output={
                "duplex_prompt_token_ids": prompt_ids,
                "meta": metadata,
            },
        )

    context = SimpleNamespace(
        bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7}},
        source_token_decoder=lambda ids, **_: "".join(token_text.get(int(token_id), "") for token_id in ids),
    )
    first_ids = [9304, 21, 22, 9308]
    first_info = llm2tts([output(first_ids)], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    second_info = llm2tts(
        [output([*first_ids, 23, 24, 25, 9308])],
        prompt=[{}],
        _streaming_context=context,
    )[0]["model_intermediate_buffer"]

    assert first_info["meta"]["native_duplex_segment_text"] == "杭州和"
    assert second_info["ids"]["tts"] == [24, 25]
    assert second_info["meta"]["native_duplex_segment_text"] == "上海之间大"


def test_native_duplex_requires_tokenizer_boundary_metadata() -> None:
    latent = torch.zeros((3, 4))
    source = _output(
        prompt_ids=[101],
        output_ids=[21, 22],
        latent=latent,
        multimodal_output={"duplex_prompt_token_ids": [101]},
    )

    with pytest.raises(ValueError, match="tokenizer-derived.*metadata"):
        llm2tts([source], prompt=[{}], _streaming_context=SimpleNamespace(bridge_states={}))


def test_llm2tts_does_not_alias_live_thinker_token_list() -> None:
    live_tokens = [11, 12]
    latent = torch.zeros((3, 4))
    source = _output(
        prompt_ids=[101],
        output_ids=list(live_tokens),
        latent=latent,
        token_list=live_tokens,
    )

    converted = llm2tts([source], prompt=[{}])[0]
    live_tokens.append(13)

    assert converted["model_intermediate_buffer"]["ids"]["output"] == [11, 12]
