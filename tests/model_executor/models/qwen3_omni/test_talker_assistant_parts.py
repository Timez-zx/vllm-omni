from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Talker:
    hidden_size = 4

    @staticmethod
    def text_projection(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def embed_input_ids(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones((token_ids.shape[0], _Talker.hidden_size), dtype=torch.bfloat16)


@pytest.mark.parametrize("assistant_rows", [0, 1, 2, 3, 4])
def test_short_assistant_segment_keeps_fixed_talker_control_prefix(assistant_rows: int) -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = _Talker()
    model.config = SimpleNamespace(
        tts_pad_token_id=99,
        talker_config=SimpleNamespace(
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=4,
            codec_bos_id=5,
            text_config=SimpleNamespace(hidden_size=_Talker.hidden_size),
        ),
    )
    thinker_embed = torch.arange(
        assistant_rows * _Talker.hidden_size,
        dtype=torch.bfloat16,
    ).reshape(assistant_rows, _Talker.hidden_size)
    special = torch.ones((1, _Talker.hidden_size), dtype=torch.bfloat16)

    input_embeds, input_ids, trailing = model._get_talker_assistant_parts(
        0,
        assistant_rows,
        6,
        thinker_embed,
        special,
        special,
        special,
    )

    assert input_embeds.shape == (9, _Talker.hidden_size)
    assert input_ids.shape == (9,)
    assert trailing.shape[1:] == (_Talker.hidden_size,)
