# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)


def _projector():
    def encode(audio, _rate, _format, _speed):
        if audio is None or np.asarray(audio).size == 0:
            return None
        return f"audio-{int(np.asarray(audio).reshape(-1)[0])}"

    return MiniCPMO45DataPlaneSession(encode)


def _context(turn=0, epoch=0, *, active=False, native=True):
    return MiniCPMO45DataPlaneContext(
        epoch=epoch,
        turn_id=turn,
        active_response_turn_id=turn if active else None,
        active_response_id="response" if active else None,
        auto_responds=native,
    )


def _output(request="request-a", turn=0, epoch=0, *, audio=None, text="", terminal=False, native=True):
    metadata = {"sr": 24000}
    if native:
        metadata.update(
            {
                "meta.duplex_turn_id": turn,
                "meta.turn_end": terminal,
                "meta.tts_is_last_chunk": terminal,
            }
        )
        if epoch is not None:
            metadata["meta.duplex_epoch"] = epoch
    if audio is not None:
        # Generation-stage chunks are lists, not cumulative waveform tensors.
        metadata["audio"] = [np.full(2400, audio, dtype=np.float32)]
    return SimpleNamespace(
        request_id=request,
        finished=False,
        outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
        multimodal_output=metadata,
    )


def _run(projector, output, context):
    return list(projector.project_output(output, context=context))


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("new_audio", [None, 2])
def test_pending_audio_never_acquires_another_turns_text(terminal, new_audio):
    projector = _projector()
    # Only the legacy path without a verified epoch still buffers unbound
    # audio. Explicit native owners no longer need text to emit audio.
    assert _run(projector, _output(audio=1, epoch=None), _context()) == []
    if terminal:
        _run(projector, _output(terminal=True, epoch=None), _context())
        assert not projector.has_pending_audio("request-a")
    results = _run(projector, _output(turn=1, text="new text", audio=new_audio), _context(turn=1))
    assert all(result.get("audio_data") != "audio-1" for result in results)
    assert all(result.get("model_turn_id") == 1 for result in results)
    if new_audio is not None:
        assert results[0]["audio_data"] == "audio-2"
        assert results[0]["text"] == "new text"
    assert not projector.has_pending_audio("request-a")


def test_legacy_same_turn_delayed_text_still_releases_audio():
    projector = _projector()
    assert _run(projector, _output(audio=1, epoch=None), _context()) == []
    results = _run(projector, _output(text="same turn", terminal=True, epoch=None), _context())
    assert results[0]["audio_data"] == "audio-1"
    assert results[0]["text"] == "same turn"
    assert results[0]["end_of_turn"] is True


def test_bound_response_keeps_audio_only_continuation_and_terminal_drain():
    projector = _projector()
    assert _run(projector, _output(audio=1, epoch=None), _context()) == []
    results = _run(projector, _output(terminal=True, epoch=None), _context(active=True))
    assert results[0]["audio_data"] == "audio-1"
    assert results[0]["end_of_turn"] is True
    direct = _run(projector, _output(turn=1, audio=2), _context(turn=1, active=True))
    assert direct[0]["audio_data"] == "audio-2"
    assert direct[0]["text"] == ""


def test_late_turn_output_cannot_clear_or_label_new_turn_buffer():
    projector = _projector()
    assert _run(projector, _output(turn=1, audio=2, epoch=None), _context()) == []
    assert _run(projector, _output(turn=0, audio=1, text="old", epoch=None), _context()) == []
    results = _run(projector, _output(turn=1, text="new", epoch=None), _context())
    assert results[0]["audio_data"] == "audio-2"
    assert results[0]["text"] == "new"
    assert results[0]["model_turn_id"] == 1


def test_epoch_reuse_resets_turn_terminal_and_pending_audio_state():
    projector = _projector()
    _run(projector, _output(audio=1, epoch=None), _context())
    _run(projector, _output(terminal=True, epoch=None), _context())
    results = _run(projector, _output(epoch=1, audio=2), _context(epoch=1))
    assert results[0]["audio_data"] == "audio-2"
    assert not projector.has_pending_audio("request-a")
    assert _run(projector, _output(epoch=0, text="old"), _context(epoch=1)) == []
    results = _run(projector, _output(epoch=1, text="new", audio=3), _context(epoch=1))
    assert results[0]["audio_data"] == "audio-3"
    assert results[0]["text"] == "new"


def test_terminal_rejects_late_audio_in_same_completed_turn():
    projector = _projector()
    _run(projector, _output(audio=1, text="done", terminal=True), _context())
    assert _run(projector, _output(audio=1, text="late"), _context()) == []
    assert not projector.has_pending_audio("request-a")


def test_two_requests_keep_separate_pending_audio_and_close_cleanup():
    projector = _projector()
    for request, audio in (("duplex-user-a-stage0", 1), ("duplex-user-b-stage0", 2)):
        assert _run(projector, _output(request=request, audio=audio, epoch=None), _context()) == []
    projector.close_session("user-a")
    assert not projector.has_request("duplex-user-a-stage0")
    results = _run(projector, _output(request="duplex-user-b-stage0", text="B", epoch=None), _context())
    assert results[0]["audio_data"] == "audio-2"
    assert results[0]["text"] == "B"


def test_non_native_audio_without_text_is_not_buffered():
    projector = _projector()
    results = _run(projector, _output(audio=1, native=False), _context(native=False))
    assert results[0]["audio_data"] == "audio-1"
    assert not projector.has_pending_audio("request-a")


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("segment_end", [False, True])
def test_owned_native_audio_needs_no_text_and_obeys_real_turn_end(terminal, segment_end):
    projector = _projector()
    output = _output(audio=1, terminal=terminal)
    output.multimodal_output["meta.tts_is_last_chunk"] = segment_end
    results = _run(projector, output, _context())

    assert len(results) == 1
    assert results[0]["audio_data"] == "audio-1"
    assert results[0]["text"] == ""
    assert results[0]["end_of_turn"] is terminal
    assert results[0]["abort_data_plane_request"] is segment_end
    assert results[0]["model_turn_id"] == 0
    assert not projector.has_pending_audio("request-a")


def test_owned_native_listen_never_emits_audio():
    projector = _projector()
    output = _output(audio=1)
    output.finished = True
    output.multimodal_output["duplex_native_decision"] = "listen"

    results = _run(projector, output, _context())

    assert len(results) == 1
    assert results[0]["is_listen"] is True
    assert not results[0].get("audio_data")
    assert not projector.has_pending_audio("request-a")


def test_owned_native_audio_remains_isolated_across_users_turns_and_epochs():
    projector = _projector()
    for request, audio in (("duplex-user-a-stage0", 1), ("duplex-user-b-stage0", 2)):
        results = _run(projector, _output(request=request, audio=audio, turn=1), _context(turn=1))
        assert results[0]["audio_data"] == f"audio-{audio}"
        assert results[0]["model_turn_id"] == 1
        assert not projector.has_pending_audio(request)

    request = "duplex-user-a-stage0"
    for output in (_output(request=request, turn=0, audio=9), _output(request=request, epoch=1, turn=1, audio=9)):
        assert _run(projector, output, _context(turn=1)) == []
    projector.close_session("user-a")
    assert not projector.has_request(request)
    results = _run(projector, _output(request="duplex-user-b-stage0", turn=1, audio=3), _context(turn=1))
    assert results[0]["audio_data"] == "audio-3"


def test_legacy_audio_without_any_owner_keeps_text_buffering():
    projector = _projector()
    assert _run(projector, _output(audio=1, native=False), _context()) == []
    results = _run(projector, _output(text="legacy", native=False), _context())
    assert results[0]["audio_data"] == "audio-1"
    assert results[0]["text"] == "legacy"
