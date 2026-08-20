#!/usr/bin/env python3
"""Verify paired continuous-AV cells processed identical media per turn."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

_LEDGER = re.compile(
    r"\[media-ledger\] sid=(?P<sid>\S*) turn=(?P<turn>\d+) "
    r"frames_selected=(?P<frames_selected>\d+) "
    r"frames_submitted=(?P<frames_submitted>\d+) "
    r"frame_selected_sha=(?P<frame_selected_sha>[0-9a-f]{64}) "
    r"frame_submitted_sha=(?P<frame_submitted_sha>[0-9a-f]{64}) "
    r"audio_selected_bytes=(?P<audio_selected_bytes>\d+) "
    r"audio_submitted_bytes=(?P<audio_submitted_bytes>\d+) "
    r"audio_selected_sha=(?P<audio_selected_sha>[0-9a-f]{64}) "
    r"audio_submitted_sha=(?P<audio_submitted_sha>[0-9a-f]{64}) "
    r"prefill_chunks=(?P<prefill_chunks>[-0-9,]+) "
    r"prefill_tokens=(?P<prefill_tokens>\d+) "
    r"prefill_token_sha=(?P<prefill_token_sha>[0-9a-f]{64}) "
    r"dropped=(?P<dropped>\d+)"
)


def parse_log(path: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for match in _LEDGER.finditer(path.read_text(errors="replace")):
        raw = match.groupdict()
        record: dict[str, Any] = {
            **raw,
            "turn": int(raw["turn"]),
            "frames_selected": int(raw["frames_selected"]),
            "frames_submitted": int(raw["frames_submitted"]),
            "audio_selected_bytes": int(raw["audio_selected_bytes"]),
            "audio_submitted_bytes": int(raw["audio_submitted_bytes"]),
            "prefill_tokens": int(raw["prefill_tokens"]),
            "dropped": int(raw["dropped"]),
            "prefill_chunk_sizes": (
                [] if raw["prefill_chunks"] == "-" else [int(value) for value in raw["prefill_chunks"].split(",")]
            ),
        }
        key = (record["sid"], record["turn"])
        if key in records:
            raise ValueError(f"duplicate media ledger for {key} in {path}")
        records[key] = record
    if not records:
        raise ValueError(f"no [media-ledger] records in {path}")
    return records


def summarize(records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    frame_exact = [
        record["frames_selected"] == record["frames_submitted"]
        and record["frame_selected_sha"] == record["frame_submitted_sha"]
        for record in records.values()
    ]
    audio_exact = [
        record["audio_selected_bytes"] == record["audio_submitted_bytes"]
        and record["audio_selected_sha"] == record["audio_submitted_sha"]
        for record in records.values()
    ]
    return {
        "turns": len(records),
        "frames_selected": sum(record["frames_selected"] for record in records.values()),
        "frames_submitted": sum(record["frames_submitted"] for record in records.values()),
        "dropped_frames": sum(record["dropped"] for record in records.values()),
        "frame_ledger_exact_turns": sum(frame_exact),
        "audio_selected_bytes": sum(record["audio_selected_bytes"] for record in records.values()),
        "audio_submitted_bytes": sum(record["audio_submitted_bytes"] for record in records.values()),
        "audio_ledger_exact_turns": sum(audio_exact),
        "prefill_chunks": sum(len(record["prefill_chunk_sizes"]) for record in records.values()),
        "prefill_tokens": sum(record["prefill_tokens"] for record in records.values()),
        "cell_media_exact": all(frame_exact) and all(audio_exact),
    }


def compare(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    common = sorted(baseline_keys & candidate_keys)
    frame_matches = sum(
        baseline[key]["frames_selected"] == candidate[key]["frames_selected"]
        and baseline[key]["frame_selected_sha"] == candidate[key]["frame_selected_sha"]
        for key in common
    )
    audio_matches = sum(
        baseline[key]["audio_selected_bytes"] == candidate[key]["audio_selected_bytes"]
        and baseline[key]["audio_selected_sha"] == candidate[key]["audio_selected_sha"]
        for key in common
    )
    media_matches = sum(
        baseline[key]["frames_selected"] == candidate[key]["frames_selected"]
        and baseline[key]["frame_selected_sha"] == candidate[key]["frame_selected_sha"]
        and baseline[key]["audio_selected_bytes"] == candidate[key]["audio_selected_bytes"]
        and baseline[key]["audio_selected_sha"] == candidate[key]["audio_selected_sha"]
        for key in common
    )
    return {
        "baseline_turns": len(baseline_keys),
        "candidate_turns": len(candidate_keys),
        "paired_turns": len(common),
        "missing_from_candidate": len(baseline_keys - candidate_keys),
        "extra_in_candidate": len(candidate_keys - baseline_keys),
        "frame_input_exact_turns": frame_matches,
        "audio_input_exact_turns": audio_matches,
        "media_input_exact_turns": media_matches,
        "paired_media_exact": bool(common)
        and len(common) == len(baseline_keys) == len(candidate_keys)
        and media_matches == len(common),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--baseline", default="arrival")
    parser.add_argument("--json-out", type=pathlib.Path)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="return non-zero unless every cell and cross-arm media ledger is exact",
    )
    args = parser.parse_args()

    cells: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    client_summaries: dict[str, dict[str, Any]] = {}
    for raw in args.cell:
        if "=" not in raw:
            parser.error(f"--cell must be LABEL=DIR, got {raw!r}")
        label, directory = raw.split("=", 1)
        cell_dir = pathlib.Path(directory)
        cells[label] = parse_log(cell_dir / "engine.log")
        client_summaries[label] = json.loads((cell_dir / "summary.json").read_text())
    if args.baseline not in cells:
        parser.error(f"baseline {args.baseline!r} was not supplied")

    report = {
        "cells": {
            label: {
                **summarize(records),
                "client_frames_sent": client_summaries[label].get("frames_sent"),
                "client_audio_chunks_sent": client_summaries[label].get("mic_chunks_sent"),
                "input_trace_mode": client_summaries[label].get("input_trace_mode"),
                "input_trace_sha256": client_summaries[label].get("input_trace_sha256"),
                "replay_schedule_slips": client_summaries[label].get("replay_schedule_slips", 0),
            }
            for label, records in cells.items()
        },
        "comparisons": {
            label: compare(cells[args.baseline], records) for label, records in cells.items() if label != args.baseline
        },
    }
    for label, summary in report["cells"].items():
        print(
            f"{label}: turns={summary['turns']} "
            f"frames={summary['frames_submitted']}/{summary['frames_selected']} "
            f"audio={summary['audio_submitted_bytes']}/{summary['audio_selected_bytes']} "
            f"drops={summary['dropped_frames']} slips={summary['replay_schedule_slips']} "
            f"exact={summary['cell_media_exact']}"
        )
    for label, comparison in report["comparisons"].items():
        print(
            f"{args.baseline} vs {label}: paired={comparison['paired_turns']} "
            f"frame_exact={comparison['frame_input_exact_turns']} "
            f"audio_exact={comparison['audio_input_exact_turns']} "
            f"media_exact={comparison['paired_media_exact']}"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.require_exact:
        trace_hashes = {summary["input_trace_sha256"] for summary in report["cells"].values()}
        cells_exact = (
            all(
                summary["cell_media_exact"] and summary["replay_schedule_slips"] == 0
                for summary in report["cells"].values()
            )
            and len(trace_hashes) == 1
        )
        comparisons_exact = all(comparison["paired_media_exact"] for comparison in report["comparisons"].values())
        if not cells_exact or not comparisons_exact:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
