from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks/duplexomni" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_control_parser_accepts_json_and_python_dict() -> None:
    harness = _module("duplexomni_single_user", "single_user.py")
    expected = {"asr": "hi", "tts": "hello", "tts_control": "", "system2_control": "[WAIT]"}
    assert harness._parse_controls(f"```json\n{expected!r}\n```") == expected
    assert harness._parse_controls('{"asr":"hi","tts":"hello","tts_control":"","system2_control":"[WAIT]"}') == expected


def test_fp8_comparison_checks_semantics_not_waveform_identity() -> None:
    compare = _module("duplexomni_compare", "compare_runs.py")
    common = {
        "session_id": "same-session",
        "slot_ms": 480,
        "input_modalities": ["audio", "image"],
        "video_frame_interval_slots": 4,
    }
    slot = {
        "controls": {"asr": "hello", "tts": "hi", "tts_control": "", "system2_control": "[WAIT]"},
        "codec_shape": [16, 6],
        "valid_turn": True,
        "audio": {"duration_ms": 480.0, "peak": 0.5},
    }
    result = compare.compare(
        common | {"slots": [slot]},
        common | {"slots": [slot | {"audio": {"duration_ms": 480.0, "peak": 0.9}}]},
    )
    assert result["pass"] is True


def test_fp8_comparison_rejects_a_different_workload() -> None:
    compare = _module("duplexomni_compare_workload", "compare_runs.py")
    base = {
        "session_id": "session",
        "slot_ms": 480,
        "input_modalities": ["audio", "image"],
        "video_frame_interval_slots": 4,
        "slots": [{}],
    }

    result = compare.compare(base, base | {"video_frame_interval_slots": 1})

    assert result["input_match"] is False
    assert result["pass"] is False


def test_default_video_cadence_is_one_frame_per_four_slots() -> None:
    harness = _module("duplexomni_single_user_video_cadence", "single_user.py")

    assert [harness._slot_has_video(index, False, 4) for index in range(8)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert harness._slot_has_video(0, True, 4) is False


def test_single_user_supplies_all_multi_user_defaults(tmp_path, monkeypatch) -> None:
    harness = _module("duplexomni_single_user_defaults", "single_user.py")
    captured = {}

    async def fake_run(args):
        captured["args"] = args
        return {
            "errors": [],
            "sessions": [
                {
                    "session_id": "single",
                    "records": [],
                    "compressions": [],
                }
            ],
            "video_frames": {"sent": 0, "acked": 0, "accepted": 0},
        }

    fake_multi_user = types.ModuleType("multi_user")
    fake_multi_user.run = fake_run
    monkeypatch.setitem(sys.modules, "multi_user", fake_multi_user)
    args = types.SimpleNamespace(
        url="http://127.0.0.1:8092",
        model="DuplexOmni",
        label="fp8",
        output=tmp_path,
        slots=1,
        audio=None,
        image=None,
        audio_only=True,
        video_frame_interval=1,
        from_s2="",
        system_prompt="test",
        session_id="single",
        context_trigger_tokens=6144,
        max_tokens=999,
        timeout=10.0,
    )

    harness.run(args)

    forwarded = captured["args"]
    assert forwarded.prefill_only_users == 0
    assert forwarded.media_seed == 0
    assert forwarded.shared_media_across_users is False


def test_multi_user_phases_are_reproducible_bounded_and_periodic() -> None:
    harness = _module("duplexomni_multi_user_phase", "multi_user.py")

    first = harness._session_specs(32, seed=17, prefix="load")
    second = harness._session_specs(32, seed=17, prefix="load")

    assert first == second
    assert all(0.0 <= spec.phase_ms < harness.SLOT_MS for spec in first)
    assert len({spec.phase_ms for spec in first}) == 32
    assert harness._scheduled_ms(first[0], 3) - harness._scheduled_ms(first[0], 2) == harness.SLOT_MS


def test_multi_user_summary_counts_e2e_deadline_misses() -> None:
    harness = _module("duplexomni_multi_user_summary", "multi_user.py")
    records = [
        {
            "e2e_slot_latency_ms": 120.0,
            "request_latency_ms": 100.0,
            "app_queue_ms": 20.0,
            "deadline_miss": False,
        },
        {
            "e2e_slot_latency_ms": 600.0,
            "request_latency_ms": 400.0,
            "app_queue_ms": 200.0,
            "deadline_miss": True,
        },
    ]

    summary = harness._latency_summary(records)

    assert summary["deadline_ms"] == harness.SLOT_MS
    assert summary["deadline_misses"] == 1
    assert summary["deadline_miss_rate"] == 0.5
    assert summary["e2e_slot_latency"]["max_ms"] == 600.0


def test_multi_user_client_sends_only_media_to_the_server_session() -> None:
    harness = _module("duplexomni_multi_user_transport", "multi_user.py")
    samples = harness.np.arange(harness.SLOT_SAMPLES, dtype=harness.np.int16)

    pcm = harness._audio_pcm_slot(samples, 0)

    assert len(pcm) == harness.SLOT_SAMPLES * 2
    assert harness._websocket_uri("http://127.0.0.1:8092") == (
        "ws://127.0.0.1:8092/v1/video/chat/stream"
    )


def test_multi_user_media_is_distinct_without_changing_payload_shape() -> None:
    harness = _module("duplexomni_multi_user_media_identity", "multi_user.py")
    samples = harness.np.arange(harness.SLOT_SAMPLES, dtype=harness.np.int16)
    pcm = samples.astype("<i2").tobytes()
    image = harness._image_bytes(None, 0)

    pcm_user_0 = harness._personalize_audio_pcm(pcm, 0)
    pcm_user_1 = harness._personalize_audio_pcm(pcm, 1)
    image_user_0 = harness._personalize_image_jpeg(image, 0)
    image_user_1 = harness._personalize_image_jpeg(image, 1)

    assert len(pcm_user_0) == len(pcm_user_1) == len(pcm)
    assert pcm_user_0 != pcm_user_1
    assert image_user_0 != image_user_1
