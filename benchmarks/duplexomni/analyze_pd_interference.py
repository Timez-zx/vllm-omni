#!/usr/bin/env python3
"""Attribute Thinker decode tail to prefill at scheduler-iteration granularity."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
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


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def _load_request_ids(manifest_path: Path) -> tuple[set[str], int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for session in manifest["sessions"] for record in session["records"]]
    expected_decode_gaps = sum(
        max(
            0,
            int(
                (((record.get("metrics") or {}).get("stage_metrics") or {}).get("0") or {}).get(
                    "num_tokens_out", 0
                )
            )
            - 1,
        )
        for record in records
    )
    return {str(record["request_id"]) for record in records}, len(records), expected_decode_gaps


def _base_request_id(stage_request_id: str, request_ids: set[str]) -> str | None:
    # Stage request IDs append a stage-local suffix to the external request ID.
    return next((request_id for request_id in request_ids if stage_request_id.startswith(request_id)), None)


def _load_iterations(log_path: Path, request_ids: set[str]) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "[PD-ITER] "
        if marker not in line:
            continue
        try:
            iteration = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        requests = iteration.get("prefill", []) + iteration.get("decode", []) + iteration.get("encoder", [])
        if any(_base_request_id(str(request["id"]), request_ids) for request in requests):
            iterations.append(iteration)
    iterations.sort(key=lambda item: int(item["seq"]))
    return iterations


def _decode_context(iteration: dict[str, Any]) -> float:
    contexts = [float(request["computed"]) for request in iteration["decode"]]
    return statistics.median(contexts) if contexts else 0.0


def analyze(manifest_path: Path, log_path: Path) -> dict[str, Any]:
    request_ids, record_count, expected_decode_gaps = _load_request_ids(manifest_path)
    iterations = _load_iterations(log_path, request_ids)
    if not iterations:
        raise RuntimeError("no matching [PD-ITER] records found")

    by_sequence = {int(iteration["seq"]): iteration for iteration in iterations}
    by_kind = Counter(str(iteration["kind"]) for iteration in iterations)
    by_prefill_source = Counter(str(iteration.get("prefill_source", "unknown")) for iteration in iterations)
    batch_gpu = {
        kind: _distribution([float(item["gpu_ms"]) for item in iterations if item["kind"] == kind])
        for kind in ("decode_only", "mixed", "prefill_only")
    }
    batch_prepare = {
        source: _distribution(
            [
                float(item["prepare_ms"])
                for item in iterations
                if item.get("prefill_source", "unknown") == source
            ]
        )
        for source in ("none", "tokens", "encoder", "tokens_and_encoder")
    }

    gaps: list[dict[str, Any]] = []
    for current in iterations:
        current_sequence = int(current["seq"])
        for gap in current.get("decode_gaps", []):
            base_id = _base_request_id(str(gap["id"]), request_ids)
            previous_sequence = int(gap["after_seq"])
            previous = by_sequence.get(previous_sequence)
            if base_id is None or previous is None:
                continue
            interval_batches = [
                by_sequence[sequence]
                for sequence in range(previous_sequence, current_sequence)
                if sequence in by_sequence
            ]
            sources: set[str] = set()
            for item in interval_batches:
                if int(item.get("prefill_tokens", 0)) > 0:
                    sources.add("tokens")
                if int(item.get("encoder_inputs", 0)) > 0:
                    sources.add("encoder")
            gaps.append(
                {
                    "request_id": base_id,
                    "ms": float(gap["ms"]),
                    "prefill_exposed": bool(sources),
                    "prefill_sources": sorted(sources),
                    "previous": previous,
                }
            )

    clean = [float(gap["ms"]) for gap in gaps if not gap["prefill_exposed"]]
    exposed = [float(gap["ms"]) for gap in gaps if gap["prefill_exposed"]]
    gap_summary = {
        "clean": _distribution(clean),
        "prefill_exposed": _distribution(exposed),
    }
    source_gap_summary = {
        source: _distribution(
            [float(gap["ms"]) for gap in gaps if gap["prefill_sources"] == sources]
        )
        for source, sources in {
            "token_prefill_only": ["tokens"],
            "encoder_prefill_only": ["encoder"],
            "token_and_encoder_prefill": ["encoder", "tokens"],
        }.items()
    }

    tail_prefill_share: dict[str, float | None] = {}
    all_gap_ms = [float(gap["ms"]) for gap in gaps]
    for percentile in (90, 95, 99):
        threshold = _percentile(all_gap_ms, percentile)
        tail = [gap for gap in gaps if threshold is not None and float(gap["ms"]) >= threshold]
        tail_prefill_share[f"p{percentile}"] = (
            sum(bool(gap["prefill_exposed"]) for gap in tail) / len(tail) if tail else None
        )

    decode_only = [iteration for iteration in iterations if iteration["kind"] == "decode_only"]
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for mixed in (
        iteration
        for iteration in iterations
        if iteration["kind"] == "mixed" and int(iteration.get("prefill_tokens", 0)) > 0
    ):
        candidates = [
            decode
            for decode in decode_only
            if len(decode["decode"]) == len(mixed["decode"])
        ]
        if not candidates:
            continue
        match = min(candidates, key=lambda decode: abs(_decode_context(decode) - _decode_context(mixed)))
        matched_pairs.append((mixed, match, abs(_decode_context(match) - _decode_context(mixed))))

    strict_pairs = [(mixed, clean_batch) for mixed, clean_batch, delta in matched_pairs if delta <= 32]
    matched_summary = {
        "pairs": len(matched_pairs),
        "strict_context_delta_le_32_pairs": len(strict_pairs),
        "strict_mixed_gpu_ms_p50": _percentile([float(mixed["gpu_ms"]) for mixed, _ in strict_pairs], 50),
        "strict_decode_only_gpu_ms_p50": _percentile(
            [float(clean_batch["gpu_ms"]) for _, clean_batch in strict_pairs], 50
        ),
    }
    strict_clean = matched_summary["strict_decode_only_gpu_ms_p50"]
    strict_mixed = matched_summary["strict_mixed_gpu_ms_p50"]
    matched_summary["strict_mixed_to_decode_only_ratio"] = (
        strict_mixed / strict_clean if strict_mixed is not None and strict_clean else None
    )

    baseline_by_decode_count: dict[int, float] = {}
    for decode_count in range(1, 1 + max(len(item["decode"]) for item in iterations)):
        values = [
            float(gap["ms"])
            for gap in gaps
            if not gap["prefill_exposed"] and len(gap["previous"]["decode"]) == decode_count
        ]
        if values:
            baseline_by_decode_count[decode_count] = statistics.median(values)
    exposed_gaps = [gap for gap in gaps if gap["prefill_exposed"]]
    matched_exposed_gaps = [
        gap
        for gap in exposed_gaps
        if len(gap["previous"]["decode"]) in baseline_by_decode_count
    ]
    excess_ms = sum(
        max(
            0.0,
            float(gap["ms"]) - baseline_by_decode_count[len(gap["previous"]["decode"])],
        )
        for gap in matched_exposed_gaps
    )
    total_gap_ms = sum(all_gap_ms)
    # Absolute upper bound: even if every millisecond of every exposed gap
    # were caused by prefill, attribution cannot exceed the exposed gap sum.
    excess_upper_bound_ms = sum(float(gap["ms"]) for gap in exposed_gaps)

    total_gpu_ms = sum(float(iteration["gpu_ms"]) for iteration in iterations)
    matched_gpu_excess_ms = sum(
        max(0.0, float(mixed["gpu_ms"]) - float(clean_batch["gpu_ms"]))
        for mixed, clean_batch, _ in matched_pairs
    )
    token_mixed_count = sum(
        iteration["kind"] == "mixed" and int(iteration.get("prefill_tokens", 0)) > 0
        for iteration in iterations
    )

    clean_p99 = gap_summary["clean"]["p99"]
    exposed_p50 = gap_summary["prefill_exposed"]["p50"]
    p99_share = tail_prefill_share["p99"]
    return {
        "requests": record_count,
        "trace_coverage_of_expected_decode_gaps": len(gaps) / expected_decode_gaps if expected_decode_gaps else None,
        "iteration_count": len(iterations),
        "batch_counts": dict(by_kind),
        "batch_counts_by_prefill_source": dict(by_prefill_source),
        "gpu_forward_ms": batch_gpu,
        "prepare_ms_by_prefill_source": batch_prepare,
        "decode_gap_ms": gap_summary,
        "decode_gap_ms_by_prefill_source": source_gap_summary,
        "decode_tail_prefill_share": tail_prefill_share,
        "matched_same_decode_count_and_context": matched_summary,
        "attribution": {
            "decode_gap_samples_matched": len(matched_exposed_gaps),
            "decode_gap_sample_coverage": (
                len(matched_exposed_gaps) / len(exposed_gaps) if exposed_gaps else None
            ),
            "prefill_excess_decode_gap_ms_lower_bound": excess_ms,
            "prefill_excess_decode_gap_ms_upper_bound": excess_upper_bound_ms,
            "prefill_excess_share_of_observed_decode_gap_time_bounds": (
                [excess_ms / total_gap_ms, excess_upper_bound_ms / total_gap_ms]
                if total_gap_ms
                else None
            ),
            "token_mixed_batches_matched": len(matched_pairs),
            "token_mixed_batch_coverage": len(matched_pairs) / token_mixed_count if token_mixed_count else None,
            "prefill_excess_gpu_ms_matched_lower_bound": matched_gpu_excess_ms,
            "prefill_excess_share_of_forward_gpu_time_matched_lower_bound": (
                matched_gpu_excess_ms / total_gpu_ms if total_gpu_ms else None
            ),
        },
        "conclusion": {
            "prefill_dominates_decode_tail": bool(
                p99_share is not None
                and p99_share >= 0.9
                and clean_p99 is not None
                and exposed_p50 is not None
                and exposed_p50 > clean_p99
            ),
            "prefill_is_majority_of_total_decode_gap_time": (
                True
                if total_gap_ms and excess_ms / total_gap_ms > 0.5
                else False
                if total_gap_ms and excess_upper_bound_ms / total_gap_ms < 0.5
                else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.manifest, args.server_log)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
