#!/usr/bin/env python3
"""Compare DuplexOmni FP8 correctness against the BF16 baseline."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _similarity(left: Any, right: Any) -> float:
    return difflib.SequenceMatcher(None, str(left).strip(), str(right).strip()).ratio()


def compare(bf16: dict[str, Any], fp8: dict[str, Any]) -> dict[str, Any]:
    baseline = bf16.get("slots") or []
    candidate = fp8.get("slots") or []
    input_keys = ("session_id", "slot_ms", "input_modalities", "video_frame_interval_slots")
    input_match = all(bf16.get(key) == fp8.get(key) for key in input_keys)
    slots: list[dict[str, Any]] = []
    overall_pass = input_match and len(baseline) == len(candidate) and bool(baseline)
    for index, (base, quant) in enumerate(zip(baseline, candidate)):
        base_control = base.get("controls") or {}
        quant_control = quant.get("controls") or {}
        semantic_scores = [_similarity(base_control.get(key, ""), quant_control.get(key, "")) for key in ("asr", "tts")]
        action_match = all(
            str(base_control.get(key, "")) == str(quant_control.get(key, ""))
            for key in ("tts_control", "system2_control")
        )
        speaking_match = bool(base_control.get("tts")) == bool(quant_control.get("tts"))
        codec_ok = base.get("codec_shape") == [16, 6] and quant.get("codec_shape") == [16, 6]
        eos_decision_match = base.get("valid_turn") == quant.get("valid_turn")
        audio_ok = all(
            400.0 <= float(item.get("audio", {}).get("duration_ms", 0.0)) <= 560.0
            and 0.0 <= float(item.get("audio", {}).get("peak", -1.0)) <= 1.0
            for item in (base, quant)
        )
        slot_pass = (
            min(semantic_scores) >= 0.5
            and action_match
            and speaking_match
            and codec_ok
            and eos_decision_match
            and audio_ok
        )
        overall_pass = overall_pass and slot_pass
        slots.append(
            {
                "slot": index,
                "asr_similarity": semantic_scores[0],
                "tts_similarity": semantic_scores[1],
                "control_actions_match": action_match,
                "speaking_decision_match": speaking_match,
                "codec_shape_ok": codec_ok,
                "eos_decision_match": eos_decision_match,
                "audio_sanity_ok": audio_ok,
                "pass": slot_pass,
            }
        )
    return {
        "pass": overall_pass,
        "bf16_slots": len(baseline),
        "fp8_slots": len(candidate),
        "input_match": input_match,
        "criteria": {
            "asr_tts_similarity_min": 0.5,
            "control_actions": "exact",
            "speaking_decision": "exact",
            "codec_shape": [16, 6],
            "eos_decision": "exact match",
            "audio_duration_ms": [400, 560],
        },
        "slots": slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(_load(args.bf16), _load(args.fp8))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
