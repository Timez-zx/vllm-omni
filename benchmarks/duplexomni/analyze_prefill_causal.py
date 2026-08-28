#!/usr/bin/env python3
"""Analyze fixed-probe / prefill-only interference experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def _role(request_id: str, probes: set[str], backgrounds: set[str]) -> str | None:
    if any(request_id.startswith(prefix) for prefix in probes):
        return "probe"
    if any(request_id.startswith(prefix) for prefix in backgrounds):
        return "prefill_only"
    return None


def _load_iterations(log_path: Path) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    marker = "[PD-ITER] "
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            item = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if str(item.get("stage")) == "0":
            iterations.append(item)
    return iterations


def analyze(manifest_path: Path, log_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for session in manifest["sessions"] for record in session["records"]]
    probe_records = [record for record in records if record.get("workload_role") == "probe"]
    background_records = [
        record for record in records if record.get("workload_role") == "prefill_only"
    ]
    probes = {str(record["request_id"]) for record in probe_records}
    backgrounds = {str(record["request_id"]) for record in background_records}

    all_iterations = _load_iterations(log_path)
    by_sequence = {int(item["seq"]): item for item in all_iterations}

    def iteration_relevant(item: dict[str, Any]) -> bool:
        requests = item.get("prefill", []) + item.get("decode", []) + item.get("encoder", [])
        return any(
            _role(str(request.get("id", "")), probes, backgrounds) is not None
            for request in requests
        )

    iterations = [item for item in all_iterations if iteration_relevant(item)]
    if not iterations:
        raise RuntimeError("no trace iterations matched this manifest")

    def has_role_prefill(item: dict[str, Any], role: str) -> bool:
        return any(
            _role(str(request.get("id", "")), probes, backgrounds) == role
            for request in item.get("prefill", [])
        )

    gaps: list[dict[str, Any]] = []
    for current in iterations:
        current_sequence = int(current["seq"])
        for gap in current.get("decode_gaps", []):
            if _role(str(gap.get("id", "")), probes, backgrounds) != "probe":
                continue
            previous_sequence = int(gap["after_seq"])
            interval = [
                by_sequence[sequence]
                for sequence in range(previous_sequence, current_sequence)
                if sequence in by_sequence
            ]
            background_exposed = any(
                has_role_prefill(item, "prefill_only") for item in interval
            )
            probe_prefill_exposed = any(has_role_prefill(item, "probe") for item in interval)
            gaps.append(
                {
                    "ms": float(gap["ms"]),
                    "background_exposed": background_exposed,
                    "probe_prefill_exposed": probe_prefill_exposed,
                }
            )

    all_gaps = [float(gap["ms"]) for gap in gaps]
    background_exposed = [
        float(gap["ms"]) for gap in gaps if gap["background_exposed"]
    ]
    no_background = [
        float(gap["ms"]) for gap in gaps if not gap["background_exposed"]
    ]
    clean = [
        float(gap["ms"])
        for gap in gaps
        if not gap["background_exposed"] and not gap["probe_prefill_exposed"]
    ]

    tail_background_share: dict[str, float | None] = {}
    for percentile in (90, 95, 99):
        threshold = _percentile(all_gaps, percentile)
        tail = [gap for gap in gaps if threshold is not None and float(gap["ms"]) >= threshold]
        tail_background_share[f"p{percentile}"] = (
            sum(bool(gap["background_exposed"]) for gap in tail) / len(tail) if tail else None
        )

    background_prefill_iterations = [
        item for item in iterations if has_role_prefill(item, "prefill_only")
    ]
    background_decode_entries = [
        request
        for item in iterations
        for request in item.get("decode", [])
        if _role(str(request.get("id", "")), probes, backgrounds) == "prefill_only"
    ]
    mixed_iterations = [
        item
        for item in background_prefill_iterations
        if any(
            _role(str(request.get("id", "")), probes, backgrounds) == "probe"
            for request in item.get("decode", [])
        )
    ]
    probe_decode_only_iterations = [
        item
        for item in iterations
        if item.get("decode")
        and not item.get("prefill")
        and any(
            _role(str(request.get("id", "")), probes, backgrounds) == "probe"
            for request in item.get("decode", [])
        )
    ]

    trace_duration_s = (
        (max(float(item["mono"]) for item in iterations) - min(float(item["mono"]) for item in iterations))
        if len(iterations) > 1
        else 0.0
    )
    thinker_latency = [float(record["thinker_latency_ms"]) for record in probe_records]
    request_latency = [float(record["request_latency_ms"]) for record in probe_records]
    app_queue = [float(record["app_queue_ms"]) for record in probe_records]

    return {
        "manifest": str(manifest_path),
        "probe_users": int(manifest.get("probe_users", manifest["users"])),
        "prefill_only_users": int(manifest.get("prefill_only_users", 0)),
        "probe_requests": len(probe_records),
        "prefill_only_requests": len(background_records),
        "trace_duration_s": trace_duration_s,
        "probe_latency_ms": {
            "thinker": _distribution(thinker_latency),
            "request": _distribution(request_latency),
            "app_queue": _distribution(app_queue),
        },
        "probe_decode_gap_ms": {
            "all": _distribution(all_gaps),
            "no_background_prefill": _distribution(no_background),
            "background_prefill_exposed": _distribution(background_exposed),
            "clean_no_prefill": _distribution(clean),
        },
        "decode_tail_background_prefill_share": tail_background_share,
        "background_prefill": {
            "batches": len(background_prefill_iterations),
            "mixed_with_probe_decode_batches": len(mixed_iterations),
            "tokens": sum(
                int(request.get("tokens", 0))
                for item in background_prefill_iterations
                for request in item.get("prefill", [])
                if _role(str(request.get("id", "")), probes, backgrounds) == "prefill_only"
            ),
            "encoder_inputs": sum(
                int(request.get("items", 0))
                for item in background_prefill_iterations
                for request in item.get("encoder", [])
                if _role(str(request.get("id", "")), probes, backgrounds) == "prefill_only"
            ),
            "decode_entries": len(background_decode_entries),
            "rate_requests_per_s": (
                len(background_records) / trace_duration_s if trace_duration_s else None
            ),
        },
        "iteration_gpu_ms": {
            "background_prefill_mixed_with_probe_decode": _distribution(
                [float(item["gpu_ms"]) for item in mixed_iterations]
            ),
            "probe_decode_only": _distribution(
                [float(item["gpu_ms"]) for item in probe_decode_only_iterations]
            ),
        },
        "valid_control": len(background_decode_entries) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.manifest, args.server_log)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["valid_control"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
