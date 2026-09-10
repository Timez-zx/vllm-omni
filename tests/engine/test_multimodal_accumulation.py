# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    release_native_segment_content,
    replace_snapshot_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_consumed_native_latent_does_not_grow_across_units():
    payload = MultimodalPayload.from_dict(
        {
            "latent": torch.ones(211, 4),
            "duplex_segment_token_ids": torch.tensor([151706, 42, 151718]),
            "duplex_prompt_len": torch.tensor(32000),
        }
    )
    emitted = dict(payload)
    release_native_segment_content(payload)
    assert "latent" not in payload
    assert "duplex_segment_token_ids" not in payload
    assert emitted["latent"].shape == (211, 4)
    assert "duplex_prompt_len" in payload
    next_unit = payload.merged_with(MultimodalPayload.from_dict({"latent": torch.ones(212, 4)}))
    assert next_unit["latent"].shape == (212, 4)


def test_ordinary_cumulative_latent_is_not_released():
    payload = MultimodalPayload.from_dict({"latent": torch.ones(211, 4)})
    release_native_segment_content(payload)
    assert payload["latent"].shape == (211, 4)


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "meta.segment_end": torch.tensor([0]),
            "meta.tts_is_last_chunk": torch.tensor([0]),
            "meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "meta.segment_end": torch.tensor([1]),
            "meta.tts_is_last_chunk": torch.tensor([1]),
            "meta.turn_end": torch.tensor([1]),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)

    assert not is_non_final_delta_audio_chunk(merged, "audio")

    drain_delta_payload(merged)

    assert "audio" not in merged
    assert "meta.segment_end" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"
