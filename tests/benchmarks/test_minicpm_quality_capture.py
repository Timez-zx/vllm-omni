import base64
import json
import wave

import pytest

from benchmarks.minicpmo.quality_capture import export_quality_capture
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector


def test_quality_capture_preserves_full_text_pcm_and_timing(tmp_path):
    collector = RealtimeEventCollector()
    for i in range(30):
        collector.add({"type": "response.audio_transcript.delta", "response_id": "one",
                       "delta": str(i) + " "}, received_at_s=100 + i)
    pcm = b"\x01\x00" * 480
    collector.add({"type": "response.audio.delta", "response_id": "one",
                   "delta": base64.b64encode(pcm).decode(), "sample_rate_hz": 24000},
                  received_at_s=131)
    original = list(collector.events)
    export_quality_capture(collector, tmp_path / "quality", origin_s=100, metadata={})
    result = json.loads((tmp_path / "quality/responses.json").read_text())
    response = result["responses"][0]
    assert response["text"] == "".join(str(i) + " " for i in range(30))
    assert response["first_event_s"] == 0
    assert response["last_event_s"] == 31
    with wave.open(response["wav"]) as wav:
        assert wav.getframerate() == 24000
        assert wav.readframes(480) == pcm
    assert collector.events == original
    assert result["event_count"] == 31
    with pytest.raises(FileExistsError):
        export_quality_capture(collector, tmp_path / "quality", origin_s=100, metadata={})
