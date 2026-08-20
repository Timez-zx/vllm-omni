# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import wave
from pathlib import Path

import pytest

from benchmarks.live_agent.web_client.continuous_av_workload import (
    AUDIO_CADENCE_MS,
    AUDIO_RATE,
    ECHO_GUARD_MS,
    ENDPOINT_SILENCE_MS,
    VIDEO_INTERVAL_MS,
    build_user_plan,
    load_audio_manifest,
    make_room_tone_chunks,
    plan_sha256,
)


def _write_wav(path: Path, *, samples: int = 3200, rate: int = AUDIO_RATE) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x01\x00" * samples)


def _manifest(tmp_path: Path) -> Path:
    for name in ("a1.wav", "a2.wav", "b1.wav", "b2.wav"):
        _write_wav(tmp_path / name)
    records = [
        {"id": "a1", "speaker": "alice", "transcript": "What do you see?", "audio": "a1.wav"},
        {"id": "a2", "speaker": "alice", "transcript": "Has it changed?", "audio": "a2.wav"},
        {"id": "b1", "speaker": "bob", "transcript": "Describe the scene.", "audio": "b1.wav"},
        {"id": "b2", "speaker": "bob", "transcript": "What should I notice?", "audio": "b2.wav"},
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def test_manifest_loads_real_pcm_and_pins_provenance(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    utterances, meta = load_audio_manifest(path)

    assert len(utterances) == 4
    assert meta["audio_speakers"] == ["alice", "bob"]
    assert meta["audio_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(utterance.duration_s == pytest.approx(0.2) for utterance in utterances)


def test_manifest_rejects_audio_that_does_not_match_browser_contract(tmp_path: Path) -> None:
    _write_wav(tmp_path / "bad.wav", rate=8000)
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"id": "bad", "speaker": "alice", "transcript": "bad", "audio": "bad.wav"}))

    with pytest.raises(ValueError, match="expected 16000 Hz"):
        load_audio_manifest(path)


def test_user_plans_are_deterministic_and_keep_one_speaker_per_session(tmp_path: Path) -> None:
    utterances, _ = load_audio_manifest(_manifest(tmp_path))
    kwargs = {
        "rep": 0,
        "users": 2,
        "turns": 2,
        "seed": 7,
        "utterances": utterances,
        "frame_count": 101,
        "stagger_s": (0.0, 40.0),
    }
    first = build_user_plan(uid=0, **kwargs)
    repeated = build_user_plan(uid=0, **kwargs)
    second = build_user_plan(uid=1, **kwargs)

    assert first == repeated
    assert first.turns[0].think_s == 0.0
    assert {turn.utterance.speaker for turn in first.turns} == {"alice"}
    assert {turn.utterance.speaker for turn in second.turns} == {"bob"}
    assert first.frame_start_offset != second.frame_start_offset
    assert plan_sha256([first, second]) == plan_sha256([repeated, second])


def test_room_tone_is_deterministic_quiet_and_user_specific() -> None:
    first = make_room_tone_chunks(variant=101)
    repeated = make_room_tone_chunks(variant=101)
    second = make_room_tone_chunks(variant=102)

    assert first == repeated
    assert first != second
    assert len(first) == 32
    assert all(len(chunk) == AUDIO_RATE * AUDIO_CADENCE_MS // 1000 * 2 for chunk in first)
    samples = memoryview(first[0]).cast("h")
    assert max(abs(value) for value in samples) <= 96


def test_workload_constants_match_the_browser_client() -> None:
    assert AUDIO_CADENCE_MS == 200
    assert VIDEO_INTERVAL_MS == 500
    assert ENDPOINT_SILENCE_MS == 700
    assert ECHO_GUARD_MS == 300
