"""Opt-in, post-measurement evidence export; never changes model input/output."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from vllm_omni.experimental.fullduplex.client import write_pcm16_wav


def export_quality_capture(collector, out: Path, *, origin_s: float, metadata: dict) -> dict:
    """Keep all events and exact PCM, after the receiver has stopped.

    Per-response WAVs concatenate chunks without network/playback gaps. Event
    timestamps retain those gaps separately; these WAVs are not latency data.
    """
    out.mkdir(parents=True, exist_ok=False)
    events = []
    responses = {}
    for event, received in zip(collector.events, collector.event_received_at_s, strict=True):
        events.append({"relative_s": received - origin_s, "event": event})
        rid = collector.response_id(event)
        if not rid:
            continue
        response = responses.setdefault(rid, {
            "response_id": rid, "first_event_s": received - origin_s,
            "text_deltas": [], "transcripts": [], "pcm_parts": [], "rates": set(),
        })
        response["last_event_s"] = received - origin_s
        kind = str(event.get("type", ""))
        if ("transcript" in kind or "text" in kind) and kind.endswith(".delta"):
            response["text_deltas"].append(str(event.get("delta", "")))
        if ("transcript" in kind or "text" in kind) and kind.endswith(".done"):
            response["transcripts"].append(str(event.get("transcript", event.get("text", ""))))
        if kind == "response.audio.delta":
            raw = event.get("delta") or event.get("audio")
            response["pcm_parts"].append(base64.b64decode(raw, validate=True))
            response["rates"].add(int(event.get("sample_rate_hz", collector.output_sample_rate_hz)))
    (out / "events.json").write_text(json.dumps(events, ensure_ascii=False))
    exported = []
    for index, response in enumerate(responses.values()):
        pcm = b"".join(response.pop("pcm_parts"))
        rates = response.pop("rates")
        response["text"] = "".join(response["text_deltas"])
        if pcm:
            if len(rates) != 1 or len(pcm) % 2:
                raise ValueError("Cannot export malformed or mixed-rate PCM as a single WAV")
            rate = rates.pop()
            path = out / f"response-{index:04d}.wav"
            write_pcm16_wav(path, pcm, sample_rate_hz=rate)
            response.update(wav=str(path), sample_rate_hz=rate, audio_s=len(pcm) / (2 * rate))
        exported.append(response)
    result = {
        "metadata": metadata,
        "definition": "Full received events; exact per-response PCM with delivery gaps removed only in WAVs",
        "event_count": len(events), "responses": exported,
    }
    (out / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / "transcript.txt").write_text("\n\n".join(
        f"[{r['first_event_s']:.3f}–{r['last_event_s']:.3f} s] {r['response_id']}\n{r['text']}"
        for r in exported if r["text"]
    ))
    return {"directory": str(out), "events": len(events), "responses": len(exported)}
