# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

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
from benchmarks.live_agent.web_client.mu_bench import PLAYBACK_PREBUFFER_S, summarize
from benchmarks.live_agent.web_client.prepare_slurp_davis import main as prepare_corpus


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


def test_reused_speaker_sessions_take_disjoint_recording_windows(tmp_path: Path) -> None:
    records = []
    for index in range(6):
        name = f"a{index}.wav"
        _write_wav(tmp_path / name)
        records.append(
            {
                "id": f"a{index}",
                "speaker": "alice",
                "transcript": f"request {index}",
                "audio": name,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records))
    utterances, _ = load_audio_manifest(manifest)
    kwargs = {
        "rep": 0,
        "users": 2,
        "turns": 3,
        "seed": 7,
        "utterances": utterances,
        "frame_count": 101,
        "stagger_s": (0.0, 40.0),
    }

    first = build_user_plan(uid=0, **kwargs)
    second = build_user_plan(uid=1, **kwargs)

    first_ids = {turn.utterance.utterance_id for turn in first.turns}
    second_ids = {turn.utterance.utterance_id for turn in second.turns}
    assert not first_ids & second_ids


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
    assert PLAYBACK_PREBUFFER_S == 1.4


def test_capacity_uses_audible_playback_start_and_accepts_short_released_reply() -> None:
    record = {
        "turn": 1,
        "status": "ok",
        "ttfa_ms": 200.0,
        "ttft_ms": 100.0,
        "playback_start_ms": 360.0,
        "stall_max_ms": 0.0,
        "rtf_deliver": 2.0,
        "input_audio_s": 1.0,
        "audio_s": 0.8,
    }
    media = {
        "frames_sent": 1,
        "mic_chunks_sent": 1,
        "mic_audio_s_sent": 0.2,
        "mic_speech_s_sent": 0.2,
        "mic_ambient_s_sent": 0.0,
        "mic_paused_s": 0.8,
    }
    user = SimpleNamespace(
        name="u0",
        protocol_mismatches=0,
        stray_audio=0,
        errors=[],
        rolls=0,
        stats=lambda: media,
    )

    summary = summarize([record], [user], {"turns_per_user": 1}, "", warmup_turns=0)

    assert summary["playback_start_p99_ms"] == 360.0
    assert summary["capacity_pass"] is True


def test_slurp_davis_preparation_emits_browser_compatible_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations = tmp_path / "slurp"
    dataset = annotations / "dataset" / "slurp"
    dataset.mkdir(parents=True)
    audio_root = tmp_path / "slurp_real"
    audio_root.mkdir()
    recordings = []
    metadata: dict[str, dict] = {}
    for speaker_index, speaker in enumerate(("FE-001", "MO-002")):
        metadata_recordings = {}
        for recording_index in range(2):
            suffix = "-headset" if speaker_index == 0 else ""
            filename = f"audio-{speaker_index}-{recording_index}{suffix}.flac"
            sf.write(audio_root / filename, np.zeros(AUDIO_RATE, dtype=np.float32), AUDIO_RATE)
            recordings.append({"file": filename, "wer": 0.25 * recording_index, "status": "correct"})
            metadata_recordings[filename] = {"status": "correct", "usrid": speaker}
        metadata[str(speaker_index)] = {"recordings": metadata_recordings}
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    utterance = {
        "slurp_id": 1,
        "sentence": "set alarm",
        "scenario": "alarm",
        "recordings": recordings,
    }
    (dataset / "train.jsonl").write_text(json.dumps(utterance) + "\n")
    (dataset / "devel.jsonl").write_text("")
    (dataset / "test.jsonl").write_text("")

    davis = tmp_path / "davis" / "sequence"
    davis.mkdir(parents=True)
    for index in range(24):
        (davis / f"{index:05d}.jpg").write_bytes(f"jpeg-{index}".encode())
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_slurp_davis.py",
            "--slurp-annotations",
            str(annotations),
            "--slurp-audio",
            str(audio_root),
            "--davis-jpegs",
            str(davis.parent),
            "--out",
            str(output),
            "--speakers",
            "2",
            "--utterances-per-speaker",
            "2",
        ],
    )

    assert prepare_corpus() == 0
    loaded, meta = load_audio_manifest(output / "audio_manifest.jsonl")
    assert len(loaded) == 4
    assert meta["audio_speakers"] == ["FE-001", "MO-002"]
    assert len(list((output / "frames").glob("*.jpg"))) == 2
    provenance = json.loads((output / "corpus_provenance.json").read_text())
    assert provenance["audio"]["license"] == "CC BY-NC 4.0"
    assert provenance["audio"]["mic_condition_speakers"] == {"close": 1, "distant": 1}
    assert provenance["audio"]["source_asr_wer_nonzero_share"] == 0.5
    assert provenance["video"]["effective_fps"] == 2.0
