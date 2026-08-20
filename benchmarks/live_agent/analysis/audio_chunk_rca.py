#!/usr/bin/env python3
"""Attribute continuous-AV playback-start tail across the speech pipeline.

Requires a diagnostic run with:

  VLLM_OMNI_LOG_SCHED_STEPS=1
  VLLM_OMNI_LOG_REQ_STEPS=1
  VLLM_OMNI_LOG_AUDIO_CHUNKS=1

The canonical Qwen3-Omni deploy emits four codec frames in the first audio
chunk and 25 new frames in the second.  The first-to-second emit interval is
therefore the useful unit: it separates a fixed number of Talker AR steps,
time where Talker has no upstream chunk, and Code2Wav latency.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from collections import defaultdict, deque
from typing import Any

_SCHED_STEP = re.compile(
    r"\[SCHED-STEP\] stage=(\d+) mono=([0-9.]+) nreq=(\d+) ntok=(\d+).*"
    r"irecv=(\d+)/(\d+)/(\d+)/(\d+)"
)
_REQ_STEP = re.compile(r"\[REQ-STEP\] stage=(\d+) mono=([0-9.]+) reqs=(.*)")
_CHUNK_EMIT = re.compile(r"\[CHUNK-EMIT\] rid=(\S+) chunk_id=(\d+) frames=(\d+) mono=([0-9.]+)")
_AUDIO_CHUNK = re.compile(r"\[AUDIO-CHUNK\] stage=2 req=(\S+) ts=[0-9.]+ mono=([0-9.]+) frames=(\d+)")
_TURN_RECV = re.compile(r"\[turnprobe\] recv rid=(\S+) turn=(\d+) mono=([0-9.]+)")


def nearest_rank(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "max": max(values) if values else None,
    }


def _request_ids(raw: str) -> set[str]:
    return {item.rsplit(":", 1)[0] for item in raw.split(",") if item}


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_log(text: str) -> dict[str, Any]:
    sched: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for match in _SCHED_STEP.finditer(text):
        stage, mono, nreq, ntok, hit, miss, skip, dupe = match.groups()
        sched[int(stage)].append(
            {
                "mono": float(mono),
                "nreq": int(nreq),
                "ntok": int(ntok),
                "hit": int(hit),
                "miss": int(miss),
                "skip": int(skip),
                "dupe": int(dupe),
            }
        )
    for steps in sched.values():
        steps.sort(key=lambda item: item["mono"])

    req_steps: dict[int, list[tuple[float, set[str]]]] = defaultdict(list)
    for match in _REQ_STEP.finditer(text):
        stage, mono, requests = match.groups()
        req_steps[int(stage)].append((float(mono), _request_ids(requests)))
    for steps in req_steps.values():
        steps.sort(key=lambda item: item[0])

    emits = [
        {
            "rid": match.group(1),
            "chunk_id": int(match.group(2)),
            "frames": int(match.group(3)),
            "mono": float(match.group(4)),
        }
        for match in _CHUNK_EMIT.finditer(text)
    ]
    emits.sort(key=lambda item: item["mono"])

    audio = [
        {
            "rid": match.group(1),
            "mono": float(match.group(2)),
            "frames": int(match.group(3)),
        }
        for match in _AUDIO_CHUNK.finditer(text)
    ]
    audio.sort(key=lambda item: item["mono"])

    turns = [
        {"rid": match.group(1), "turn0": int(match.group(2)), "mono": float(match.group(3))}
        for match in _TURN_RECV.finditer(text)
    ]
    turns.sort(key=lambda item: item["mono"])
    return {"sched": sched, "req_steps": req_steps, "emits": emits, "audio": audio, "turns": turns}


def _match_turn_records(turn_probes: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    for probe in turn_probes:
        candidates = [
            (abs(float(record["t_q"]) - probe["mono"]), index, record)
            for index, record in enumerate(records)
            if index not in used and int(record["turn"]) == probe["turn0"] + 1
        ]
        if not candidates:
            continue
        delta, index, record = min(candidates)
        if delta > 0.1:
            continue
        used.add(index)
        matched.append({**probe, "record": record, "query_clock_delta_ms": delta * 1000.0})
    return matched


def _pair_code2wav(emits: list[dict[str, Any]], audio: list[dict[str, Any]]) -> list[float]:
    pending: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in audio:
        pending[item["rid"]].append(item)
    latencies: list[float] = []
    for emit in emits:
        queue = pending[emit["rid"]]
        while queue and queue[0]["mono"] < emit["mono"]:
            queue.popleft()
        if queue:
            latencies.append((queue.popleft()["mono"] - emit["mono"]) * 1000.0)
    return latencies


def _decode_step_intervals(sched_steps: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    by_batch: dict[int, list[float]] = defaultdict(list)
    for current, following in zip(sched_steps, sched_steps[1:]):
        interval_ms = (following["mono"] - current["mono"]) * 1000.0
        if current["ntok"] == current["nreq"] and 0.0 < interval_ms < 100.0:
            by_batch[current["nreq"]].append(interval_ms)
    return {str(batch): distribution(values) for batch, values in sorted(by_batch.items())}


def _find_turn_segments(
    parsed: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    warmup_turns: int,
    wait_threshold_ms: float,
) -> tuple[list[dict[str, Any]], int]:
    matched = _match_turn_records(parsed["turns"], records)
    stage0_steps = parsed["req_steps"].get(0, [])
    stage1_steps = parsed["req_steps"].get(1, [])
    stage1_sched = parsed["sched"].get(1, [])
    segments: list[dict[str, Any]] = []

    for turn in matched:
        record = turn["record"]
        if int(record["turn"]) <= warmup_turns or record.get("status") != "ok":
            continue
        query = float(record["t_q"])
        first_audio = float(record["t_fa"])
        done = float(record["t_done"])
        first_candidates = [
            emit
            for emit in parsed["emits"]
            if emit["chunk_id"] == 0
            and emit["rid"].startswith(turn["rid"] + "-")
            and query <= emit["mono"] <= first_audio + 0.1
        ]
        if not first_candidates:
            continue
        first = min(first_candidates, key=lambda emit: abs(first_audio - emit["mono"]))
        second_candidates = [
            emit
            for emit in parsed["emits"]
            if emit["rid"] == first["rid"] and emit["chunk_id"] == 1 and first["mono"] < emit["mono"] <= done + 0.1
        ]
        if not second_candidates:
            continue
        second = min(second_candidates, key=lambda emit: emit["mono"])
        request_step_times = [
            mono
            for mono, request_ids in stage1_steps
            if first["mono"] < mono <= second["mono"] and first["rid"] in request_ids
        ]
        gaps = [(left, right) for left, right in zip(request_step_times, request_step_times[1:])]
        long_gaps = [(left, right) for left, right in gaps if (right - left) * 1000.0 > wait_threshold_ms]
        wait_excess_ms = sum(max(0.0, (right - left) * 1000.0 - wait_threshold_ms) for left, right in gaps)
        stage0_same = sum(
            any(left < mono < right and first["rid"] in request_ids for mono, request_ids in stage0_steps)
            for left, right in long_gaps
        )
        batches = [step["nreq"] for step in stage1_sched if first["mono"] < step["mono"] <= second["mono"]]
        segments.append(
            {
                "user": record.get("user"),
                "turn": record.get("turn"),
                "request_id": first["rid"],
                "gap_ms": (second["mono"] - first["mono"]) * 1000.0,
                "talker_request_steps": len(request_step_times),
                "wait_excess_ms": wait_excess_ms,
                "long_gaps": len(long_gaps),
                "long_gaps_with_same_thinker_request": stage0_same,
                "mean_talker_batch": sum(batches) / len(batches) if batches else None,
                "max_talker_batch": max(batches) if batches else None,
                "first_frames": first["frames"],
                "second_frames": second["frames"],
            }
        )
    return segments, len(matched)


def analyze_cell(path: pathlib.Path, *, wait_threshold_ms: float = 25.0) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text())
    records = _load_jsonl(path / "turns.jsonl")
    parsed = _parse_log((path / "engine.log").read_text(errors="replace"))
    warmup_turns = int(summary.get("warmup_turns", 0))
    segments, matched_turns = _find_turn_segments(
        parsed,
        records,
        warmup_turns=warmup_turns,
        wait_threshold_ms=wait_threshold_ms,
    )
    long_gaps = sum(int(segment["long_gaps"]) for segment in segments)
    same_thinker = sum(int(segment["long_gaps_with_same_thinker_request"]) for segment in segments)
    final_stage1 = parsed["sched"].get(1, [])[-1] if parsed["sched"].get(1) else None
    hits = int(final_stage1["hit"]) if final_stage1 else 0
    misses = int(final_stage1["miss"]) if final_stage1 else 0
    expected = int(summary.get("expected_measured_turns", 0))
    return {
        "path": str(path.resolve()),
        "users": summary.get("users"),
        "source_commit": summary.get("source_commit"),
        "source_dirty": summary.get("source_dirty"),
        "session_config_overrides": summary.get("session_config_overrides"),
        "expected_measured_turns": expected,
        "matched_turns": matched_turns,
        "mapped_second_chunks": len(segments),
        "mapped_second_chunk_coverage": len(segments) / expected if expected else None,
        "ttfa_ms": {
            "p50": summary.get("ttfa_p50_ms"),
            "p99": summary.get("ttfa_p99_ms"),
        },
        "playback_start_ms": {
            "p50": summary.get("playback_start_p50_ms"),
            "p99": summary.get("playback_start_p99_ms"),
        },
        "stall_p99_ms": summary.get("stall_max_ms_p99"),
        "second_chunk_gap_ms": distribution([float(segment["gap_ms"]) for segment in segments]),
        "talker_request_steps": distribution([float(segment["talker_request_steps"]) for segment in segments]),
        "talker_wait_excess_ms": distribution([float(segment["wait_excess_ms"]) for segment in segments]),
        "long_gap_same_thinker_request_share": same_thinker / long_gaps if long_gaps else None,
        "thinker_decode_step_ms_by_batch": _decode_step_intervals(parsed["sched"].get(0, [])),
        "talker_decode_step_ms_by_batch": _decode_step_intervals(parsed["sched"].get(1, [])),
        "thinker_to_talker_inline_receive": {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if hits + misses else None,
        },
        "code2wav_emit_to_audio_ms": distribution(_pair_code2wav(parsed["emits"], parsed["audio"])),
        "chunk_emit_count": len(parsed["emits"]),
        "audio_chunk_count": len(parsed["audio"]),
        "segments": segments,
        "wait_threshold_ms": wait_threshold_ms,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Repeat for each baseline or ablation cell.",
    )
    parser.add_argument("--wait-threshold-ms", type=float, default=25.0)
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    for raw in args.cell:
        if "=" not in raw:
            parser.error(f"--cell must be LABEL=DIR, got {raw!r}")
        label, directory = raw.split("=", 1)
        reports[label] = analyze_cell(pathlib.Path(directory), wait_threshold_ms=args.wait_threshold_ms)

    print(
        f"{'cell':<18} {'TTFA p50/p99':>15} {'play p99':>10} {'chunk2 p50/p99':>18} "
        f"{'wait p50/p99':>16} {'irecv hit':>10} {'c2w p99':>9} {'mapped':>9}"
    )
    for label, report in reports.items():
        ttfa = report["ttfa_ms"]
        gap = report["second_chunk_gap_ms"]
        wait = report["talker_wait_excess_ms"]
        receive = report["thinker_to_talker_inline_receive"]
        code2wav = report["code2wav_emit_to_audio_ms"]
        hit_rate = receive["hit_rate"]
        print(
            f"{label:<18} {_fmt(ttfa['p50']) + '/' + _fmt(ttfa['p99']):>15} "
            f"{_fmt(report['playback_start_ms']['p99']):>10} "
            f"{_fmt(gap['p50']) + '/' + _fmt(gap['p99']):>18} "
            f"{_fmt(wait['p50']) + '/' + _fmt(wait['p99']):>16} "
            f"{('-' if hit_rate is None else f'{100 * hit_rate:.1f}%'):>10} "
            f"{_fmt(code2wav['p99']):>9} "
            f"{report['mapped_second_chunks']}/{report['expected_measured_turns']:>3}"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
        print(f"full detail -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
