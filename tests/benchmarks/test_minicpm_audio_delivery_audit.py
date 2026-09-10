import base64
from types import SimpleNamespace

from benchmarks.minicpmo.continuous_av import _audio_cadence
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector


def audio(response, duration_ms):
    return {
        "type": "response.audio.delta",
        "response_id": response,
        "delta": base64.b64encode(bytes(duration_ms * 48)).decode(),
        "sample_rate_hz": 24000,
    }


def audit(events):
    collector = RealtimeEventCollector()
    for timestamp, event in events:
        collector.add(event, received_at_s=timestamp)
    return _audio_cadence(SimpleNamespace(events=collector))


def test_gap_between_distinct_responses_is_not_audio_starvation():
    result = audit(
        [
            (0, audio("one", 1000)),
            (0.1, {"type": "response.audio.done", "response_id": "one"}),
            (10, audio("two", 1000)),
            (10.1, {"type": "response.audio.done", "response_id": "two"}),
        ]
    )
    assert result["observation_complete"]
    assert result["underruns_with_200ms_buffer"] == 0
    assert result["playback_slack_samples_ms"] == []


def test_buffered_audio_is_accumulated_not_reset_for_each_chunk():
    result = audit([(0, audio("one", 1000)), (0.1, audio("one", 1000)), (1.8, audio("one", 1000))])
    # The adjacent gap is 1.7 s, but 2 s of audio was already buffered.
    assert result["underruns_with_200ms_buffer"] == 0
    assert not result["observation_complete"]
    assert result["unfinished_speech_response_ids"] == ["one"]


def test_real_starvation_and_missing_first_audio_are_visible():
    result = audit(
        [
            (0, audio("one", 1000)),
            (1.5, audio("one", 1000)),
            (2, {"type": "response.audio.done", "response_id": "one"}),
            (3, {"type": "response.speak", "response_id": "two"}),
        ]
    )
    assert result["underruns_with_200ms_buffer"] == 1
    assert result["underrun_duration_ms"]["max"] == 300
    assert result["unfinished_speech_response_ids"] == ["two"]
    assert not result["observation_complete"]


def test_silence_and_malformed_audio_cannot_certify_audio_delivery():
    assert not audit([])["observation_complete"]
    result = audit([(0, {"type": "response.audio.delta", "response_id": "one", "delta": "!"})])
    assert not result["observation_complete"]
    assert result["malformed_chunks"] == 1
