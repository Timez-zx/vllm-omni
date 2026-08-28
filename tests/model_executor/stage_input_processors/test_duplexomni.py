from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.engine import OmniPDPrefillPayload
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors import duplexomni as duplex


def _prompt_ids() -> list[int]:
    # system, user, historical assistant, current user, generation prefix
    return [
        duplex.IM_START,
        8948,
        198,
        101,
        duplex.IM_END,
        198,
        duplex.IM_START,
        872,
        198,
        102,
        duplex.IM_END,
        198,
        duplex.IM_START,
        duplex.ASSISTANT,
        198,
        201,
        202,
        duplex.IM_END,
        198,
        duplex.IM_START,
        872,
        198,
        103,
        duplex.IM_END,
        198,
        duplex.IM_START,
        duplex.ASSISTANT,
        198,
    ]


def _history(turns: int = 1) -> torch.Tensor:
    values = torch.arange(turns * 16 * 6, dtype=torch.long) % 2048
    return values.reshape(turns, 16, 6)


def test_assistant_ranges_exclude_headers_and_im_end() -> None:
    assert duplex._assistant_content_ranges(_prompt_ids()) == [(15, 17)]


def test_conditioning_layout_matches_historical_and_current_rows() -> None:
    prompt = _prompt_ids()
    output = [301, 302, 303, duplex.IM_END]
    positions, lengths = duplex._conditioning_layout(
        prompt,
        prompt + output,
        available_rows=len(prompt) + len(output) - 1,
    )
    assert positions == [15, 16, len(prompt), len(prompt) + 1, len(prompt) + 2]
    assert lengths == [2, 3]


def test_conditioning_layout_can_skip_an_invalid_historical_talker_turn() -> None:
    first = _prompt_ids()
    # Turn the generation prefix into a second historical assistant turn and
    # append a fresh user + assistant generation prefix.
    prompt = [*first, 401, 402, duplex.IM_END, 198, duplex.IM_START, 872, 198, 403, duplex.IM_END, 198]
    prompt += [duplex.IM_START, duplex.ASSISTANT, 198]
    output = [501, 502, duplex.IM_END]
    positions, lengths = duplex._conditioning_layout(
        prompt,
        prompt + output,
        available_rows=len(prompt) + len(output) - 1,
        history_indices=[1],
    )

    second_start, second_end = duplex._assistant_content_ranges(prompt)[1]
    assert positions == [*range(second_start, second_end), len(prompt), len(prompt) + 1]
    assert lengths == [second_end - second_start, 2]


def test_codec_history_contract_is_strict() -> None:
    assert duplex._normalize_codec_history([]).shape == (0, 16, 6)
    assert duplex._normalize_codec_history(_history()).shape == (1, 16, 6)
    with pytest.raises(ValueError, match="shape"):
        duplex._normalize_codec_history(torch.zeros(1, 16, 5))
    with pytest.raises(ValueError, match="outside"):
        duplex._normalize_codec_history(torch.full((1, 16, 6), 2048))


def test_full_payload_selects_only_training_aligned_assistant_rows() -> None:
    prompt = _prompt_ids()
    output = [301, 302, 303, duplex.IM_END]
    total = len(prompt) + len(output)
    layer0 = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    layer24 = layer0 + 1000
    request = SimpleNamespace(
        prompt_token_ids=prompt,
        all_token_ids=prompt + output,
        output_token_ids=output,
        additional_information={"codes": {"ref": _history()}},
    )

    payload = duplex.thinker2talker_full_payload(
        None,
        {
            "hidden_states.layer_0": layer0,
            f"hidden_states.layer_{duplex.FINAL_THINKER_LAYER}": layer24,
        },
        request,
    )

    assert payload is not None
    expected_positions = torch.tensor([15, 16, len(prompt), len(prompt) + 1, len(prompt) + 2])
    assert torch.equal(payload["embed"]["duplex_conditioning"], layer0[expected_positions])
    assert torch.equal(payload["hidden_states"]["duplex_conditioning"], layer24[expected_positions])
    assert payload["ids"]["duplex_conditioning_lengths"] == [2, 3]
    assert tuple(payload["codes"]["ref"].shape) == (1, 16, 6)


def test_full_payload_reads_materialized_runner_codec_history() -> None:
    prompt = _prompt_ids()
    output = [301, 302, 303, duplex.IM_END]
    total = len(prompt) + len(output)
    layer0 = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
    request = SimpleNamespace(
        prompt_token_ids=prompt,
        all_token_ids=prompt + output,
        output_token_ids=output,
        # This is the shape used by vLLM's CachedRequestState after the omni
        # runner has decoded the EngineCore transport payload.
        additional_information_cpu={
            "codes": {"ref": _history()},
            "ids": {"duplex_history_indices": [0]},
        },
    )

    payload = duplex.thinker2talker_full_payload(
        None,
        {
            "hidden_states.layer_0": layer0,
            f"hidden_states.layer_{duplex.FINAL_THINKER_LAYER}": layer0 + 1000,
        },
        request,
    )

    assert payload is not None
    assert torch.equal(payload["codes"]["ref"], _history())


def test_pd_full_payload_combines_layer48_prompt_snapshot_with_decode_rows() -> None:
    prompt = _prompt_ids()
    output = [301, 302, 303, duplex.IM_END]
    prompt_layer_0 = torch.arange(len(prompt) * 2, dtype=torch.float32).reshape(len(prompt), 2)
    prompt_layer_48 = prompt_layer_0 + 1000
    decode_layer_0 = torch.arange(len(output) * 2, dtype=torch.float32).reshape(len(output), 2) + 2000
    decode_layer_48 = decode_layer_0 + 1000
    request = SimpleNamespace(
        request_id="pd-slot",
        prompt_token_ids=prompt,
        # Remote-KV CachedRequestState may retain its pre-decode snapshot here.
        all_token_ids=prompt,
        output_token_ids=output,
        additional_information_cpu={"codes": {"ref": _history()}},
        pd_prefill_payload=OmniPDPrefillPayload(
            prompt_layer_0_chunks=(prompt_layer_0[:10], prompt_layer_0[10:]),
            # The legacy wire field carries DuplexOmni's selected layer 48.
            prompt_layer_24_chunks=(prompt_layer_48,),
        ),
    )

    payload = duplex.thinker2talker_full_payload(
        None,
        {
            "hidden_states.layer_0": decode_layer_0,
            f"hidden_states.layer_{duplex.FINAL_THINKER_LAYER}": decode_layer_48,
        },
        request,
    )

    assert payload is not None
    expected_layer_0 = torch.cat(
        (prompt_layer_0[[15, 16]], decode_layer_0[1:]),
        dim=0,
    )
    expected_layer_48 = torch.cat(
        (prompt_layer_48[[15, 16]], decode_layer_48[1:]),
        dim=0,
    )
    assert torch.equal(payload["embed"]["duplex_conditioning"], expected_layer_0)
    assert torch.equal(
        payload["hidden_states"]["duplex_conditioning"],
        expected_layer_48,
    )
    assert payload["ids"]["duplex_conditioning_lengths"] == [2, 3]


def test_token_only_prompt_has_exact_official_history_scaffold() -> None:
    prompt = _prompt_ids()
    generated = [301, 302, 303, duplex.IM_END]
    source = SimpleNamespace(
        prompt_token_ids=prompt,
        outputs=[SimpleNamespace(cumulative_token_ids=generated)],
    )
    history = _history()

    result = duplex.thinker2talker_token_only(
        [source],
        prompt={"additional_information": {"codes": {"ref": history}}, "cache_salt": "session:0"},
    )[0]

    ids = result["prompt_token_ids"]
    assert len(ids) == 2 + 1 + 6 + 1 + 3 + 1
    assert ids[2:10] == [duplex.CODEC_BOS, *history[0, 0].tolist(), duplex.CODEC_EOS]
    assert ids[-1] == duplex.CODEC_BOS
    assert all(0 <= token < duplex.CODEC_PAD for token in [*ids[:2], *ids[10:13]])
    # Dynamic conditioning no longer uses an all-CODEC_PAD cache identity.
    assert ids[:2] != [duplex.CODEC_PAD] * 2
    assert ids[10:13] != [duplex.CODEC_PAD] * 3
    assert result["cache_salt"] == "session:0"

    repeated = duplex.thinker2talker_token_only(
        [source],
        prompt={"additional_information": {"codes": {"ref": history}}, "cache_salt": "session:0"},
    )[0]
    assert repeated["prompt_token_ids"] == ids

    other_lineage = duplex.thinker2talker_token_only(
        [source],
        prompt={"additional_information": {"codes": {"ref": history}}, "cache_salt": "session:1"},
    )[0]
    assert other_lineage["prompt_token_ids"][:2] != ids[:2]


def test_token_only_uses_processed_prompt_when_public_output_omits_prompt_ids() -> None:
    prompt_ids = _prompt_ids()
    generated = [301, 302, 303, duplex.IM_END]
    # OmniRequestOutput exposes generated tokens but not prompt_token_ids.
    source = SimpleNamespace(outputs=[SimpleNamespace(cumulative_token_ids=generated)])
    history = _history()

    result = duplex.thinker2talker_token_only(
        [source],
        prompt={
            "prompt_token_ids": prompt_ids,
            "additional_information": {
                "codes": {"ref": history},
                "ids": {"duplex_history_indices": [0]},
            },
            "cache_salt": "session:0",
        },
    )[0]

    # Two historical conditioning tokens + BOS/6 codec/EOS, followed by the
    # three-token current conditioning and its BOS.
    assert len(result["prompt_token_ids"]) == 14
    assert result["prompt_token_ids"][2:10] == [
        duplex.CODEC_BOS,
        *history[0, 0].tolist(),
        duplex.CODEC_EOS,
    ]


def test_code2wav_payload_requires_exactly_six_frames() -> None:
    rows = torch.arange(7 * 16, dtype=torch.long).reshape(7, 16) % 2048
    request = SimpleNamespace(request_id="slice", output_token_ids=[1, 2, 3, 4, 5, 6, duplex.CODEC_EOS])
    payload = duplex.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)
    assert payload is not None
    assert len(payload["codes"]["audio"]) == 16 * 6
    assert payload["meta"]["duplexomni"] is True
    assert payload["meta"]["return_codec_codes"] is True

    with pytest.raises(RuntimeError, match="at least 6"):
        duplex.talker2code2wav_full_payload(
            None,
            {"codes.audio": rows[:6]},
            SimpleNamespace(request_id="short", output_token_ids=[1, 2, 3, 4, 5, duplex.CODEC_EOS]),
        )

    overrun = duplex.talker2code2wav_full_payload(
        None,
        {"codes.audio": rows},
        SimpleNamespace(request_id="overrun", output_token_ids=[1, 2, 3, 4, 5, 6, 7]),
    )
    assert overrun is not None
    assert len(overrun["codes"]["audio"]) == 16 * 6
    assert overrun["meta"]["duplexomni_valid_turn"] is False


class _FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_projection = nn.Identity()
        self.hidden_projection = nn.Identity()
        self.base_codec = nn.Embedding(2048, 4)
        residual = [nn.Embedding(2048, 4) for _ in range(15)]
        self.code_predictor = SimpleNamespace(model=SimpleNamespace(codec_embedding=residual))
        with torch.no_grad():
            self.base_codec.weight.fill_(1.0)
            for index, layer in enumerate(residual, start=1):
                layer.weight.fill_(float(index + 1))

    def embed_input_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.base_codec(ids)


def _fake_model() -> SimpleNamespace:
    return SimpleNamespace(
        talker=_FakeTalker(),
        talker_config=SimpleNamespace(codec_pad_id=2148, codec_bos_id=2149, codec_eos_token_id=2150),
        embed_codec_bos_token=torch.full((4,), 300.0),
        embed_codec_eos_token=torch.full((4,), 400.0),
        _module_device=lambda _module: torch.device("cpu"),
    )


def test_talker_prefill_builds_and_prefix_slices_exact_scaffold() -> None:
    model = _fake_model()
    history = torch.zeros((1, 16, 6), dtype=torch.long)
    payload = {
        "embed": {"duplex_conditioning": torch.ones((3, 4))},
        "hidden_states": {"duplex_conditioning": torch.full((3, 4), 2.0)},
        "ids": {"duplex_conditioning_lengths": [2, 1]},
        "codes": {"ref": history},
        "meta": {"duplexomni": True, "num_processed_tokens": 0},
    }
    expected_ids = [2148, 2148, 2149, 0, 0, 0, 0, 0, 0, 2150, 2148, 2149]

    ids, embeds, update = Qwen3OmniMoeForConditionalGeneration._duplexomni_talker_preprocess_prefill(
        model,
        torch.zeros(len(expected_ids), dtype=torch.long),
        torch.zeros((len(expected_ids), 4)),
        payload,
    )

    assert ids.tolist() == expected_ids
    assert embeds.shape == (len(expected_ids), 4)
    assert torch.equal(embeds[:2], torch.full((2, 4), 3.0))
    # Base codec embedding (1) plus residual groups (2..16).
    assert torch.equal(embeds[3:9], torch.full((6, 4), float(sum(range(1, 17)))))
    assert update["meta"]["duplexomni"] is True

    payload["meta"]["num_processed_tokens"] = 3
    sliced_ids, sliced_embeds, _ = Qwen3OmniMoeForConditionalGeneration._duplexomni_talker_preprocess_prefill(
        model,
        torch.zeros(4, dtype=torch.long),
        torch.zeros((4, 4)),
        payload,
    )
    assert sliced_ids.tolist() == expected_ids[3:7]
    assert torch.equal(sliced_embeds, embeds[3:7])


def test_pipelined_talker_selects_history_after_thinker_has_finished() -> None:
    model = _fake_model()
    # The Thinker exported conditioning for two historical assistant turns
    # plus the current turn before the preceding Talker result was available.
    payload = {
        "embed": {"duplex_conditioning": torch.ones((6, 4))},
        "hidden_states": {"duplex_conditioning": torch.full((6, 4), 2.0)},
        "ids": {
            "duplex_conditioning_lengths": [2, 2, 2],
            "duplex_history_indices": [1],
        },
        "codes": {"ref": _history()},
        "meta": {
            "duplexomni": True,
            "duplexomni_pipeline": True,
            "num_processed_tokens": 0,
        },
    }

    ids, embeds, _ = Qwen3OmniMoeForConditionalGeneration._duplexomni_talker_preprocess_prefill(
        model,
        torch.zeros(13, dtype=torch.long),
        torch.zeros((13, 4)),
        payload,
    )

    # Only historical turn 1 and the current turn survive; turn 0 had no
    # valid codec result and therefore cannot be part of the Talker prefix.
    assert ids.tolist() == [2148, 2148, 2149, *_history()[0, 0].tolist(), 2150, 2148, 2148, 2149]
    assert embeds.shape == (13, 4)
