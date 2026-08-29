from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _helper() -> MiniCPMO45Stage0DuplexRuntime:
    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper.listen_token_id = 90
    helper.chunk_eos_token_id = 91
    helper.chunk_tts_eos_token_id = 92
    helper.turn_eos_token_id = 93
    return helper


def test_pd_feedback_replays_decode_prefix_and_terminator() -> None:
    state = _MiniCPMO45Stage0SessionState(session_id="session")

    _helper().apply_pd_decode_feedback(
        state,
        [11, 12, 93],
        epoch=0,
        seq=2,
    )

    assert state.pd_feedback_token_ids == [11, 12, 93]
    assert state.current_segment_output_tokens == [11, 12, 93]
    assert state.pending_terminator_token == 93
    assert state.current_turn_ended is True


def test_pd_feedback_retry_is_idempotent() -> None:
    state = _MiniCPMO45Stage0SessionState(session_id="session")
    helper = _helper()
    helper.apply_pd_decode_feedback(state, [11, 90], epoch=0, seq=2)
    helper.apply_pd_decode_feedback(state, [999], epoch=0, seq=2)

    assert state.pd_feedback_token_ids == [11, 90]
    assert state.pending_terminator_token == 90
