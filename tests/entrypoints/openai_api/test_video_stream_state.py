# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.video_stream_base import StreamingVideoSessionConfig
from vllm_omni.entrypoints.openai.video_stream_state import (
    SegmentKind,
    SegmentLedger,
    SessionIdentity,
)


def test_session_config_accepts_identity_and_hybrid_baseline() -> None:
    config = StreamingVideoSessionConfig(session_id="bench-user-7", evict_engine_request_after_turn=True)
    assert config.session_id == "bench-user-7"
    assert config.session_scoped_request is True
    assert config.evict_engine_request_after_turn is True


def test_session_config_rejects_empty_identity() -> None:
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(session_id="")


def test_identity_is_monotonic_across_request_epochs() -> None:
    identity = SessionIdentity(session_id="client-7", incarnation="inc-1")
    epoch0 = identity.allocate_epoch()
    epoch1 = identity.allocate_epoch()
    first = identity.new_segment(epoch=epoch0, kind=SegmentKind.TURN, turn_id=identity.turn_id)
    identity.advance_turn(first.turn_id)
    second = identity.new_segment(epoch=epoch1, kind=SegmentKind.TURN, turn_id=identity.turn_id)

    assert (epoch0, epoch1) == (0, 1)
    assert (first.segment_id, second.segment_id) == (0, 1)
    assert (first.turn_id, second.turn_id) == (0, 1)
    assert second.event_fields() == {
        "session_id": "client-7",
        "incarnation": "inc-1",
        "epoch": 1,
        "segment_id": 1,
        "segment_kind": "turn",
        "turn_id": 1,
    }


def test_reconnect_gets_a_new_incarnation() -> None:
    first = SessionIdentity(session_id="same-client")
    second = SessionIdentity(session_id="same-client")
    assert first.incarnation != second.incarnation


def test_ledger_attributes_in_submission_order() -> None:
    identity = SessionIdentity(session_id="s", incarnation="i")
    epoch = identity.allocate_epoch()
    seed = identity.new_segment(epoch=epoch, kind=SegmentKind.SHADOW_SEED)
    turn = identity.new_segment(epoch=epoch, kind=SegmentKind.TURN, turn_id=0)
    ledger = SegmentLedger(epoch=epoch)
    ledger.submit(seed)
    ledger.submit(turn)

    assert ledger.current() is seed
    assert ledger.finish_current() is seed
    assert ledger.current() is turn
    assert ledger.finish_current() is turn
    assert ledger.current() is None


def test_ledger_rejects_stale_epoch() -> None:
    identity = SessionIdentity(session_id="s", incarnation="i")
    stale_epoch = identity.allocate_epoch()
    active_epoch = identity.allocate_epoch()
    stale = identity.new_segment(epoch=stale_epoch, kind=SegmentKind.TURN, turn_id=0)
    with pytest.raises(RuntimeError, match="segment epoch"):
        SegmentLedger(epoch=active_epoch).submit(stale)


def test_turn_fence_rejects_double_completion() -> None:
    identity = SessionIdentity(session_id="s", incarnation="i")
    identity.advance_turn(0)
    with pytest.raises(RuntimeError, match="turn fence mismatch"):
        identity.advance_turn(0)
