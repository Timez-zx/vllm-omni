#!/usr/bin/env python3
"""Summarize DuplexOmni capacity manifests and aligned GPU telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SLOT_MS = 480.0


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
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stage_value(record: dict[str, Any], stage: int, field: str) -> float | None:
    metrics = record.get("metrics") or {}
    stages = metrics.get("stage_metrics") or {}
    stage_metrics = stages.get(str(stage)) or {}
    return _number(stage_metrics.get(field))


def _component_values(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "thinker_to_first": [],
        "thinker_pre_submit_lower_bound": [],
        "talker_added": [],
        "code2wav_added": [],
        "api_return_added": [],
        "non_stage_total": [],
        "client_json_encode": [],
        "client_http_wait": [],
        "client_json_decode": [],
        "client_request_mib": [],
        "thinker_engine": [],
        "talker_engine": [],
        "code2wav_engine": [],
        "thinker_ttft": [],
        "talker_ttft": [],
        "thinker_tpot": [],
        "talker_tpot": [],
    }
    for record in records:
        stage0 = _stage_value(record, 0, "serving_time_to_first_output_ms")
        stage1 = _stage_value(record, 1, "serving_time_to_first_output_ms")
        stage2 = _stage_value(record, 2, "serving_time_to_first_output_ms")
        request = _number(record.get("request_latency_ms"))
        stage_engine_values: list[float] = []
        if stage0 is not None:
            result["thinker_to_first"].append(stage0)
        if stage0 is not None and stage1 is not None:
            result["talker_added"].append(max(0.0, stage1 - stage0))
        if stage1 is not None and stage2 is not None:
            result["code2wav_added"].append(max(0.0, stage2 - stage1))
        if stage2 is not None and request is not None:
            result["api_return_added"].append(max(0.0, request - stage2))
        for stage, name in ((0, "thinker"), (1, "talker"), (2, "code2wav")):
            engine = _stage_value(record, stage, "stage_gen_time_ms")
            if engine is not None:
                result[f"{name}_engine"].append(engine)
                stage_engine_values.append(engine)
                if stage == 0 and stage0 is not None:
                    # Stage generation runs from stage submission to final
                    # output, whereas serving-to-first starts at request
                    # arrival. Subtracting the longer submit-to-final interval
                    # therefore gives a conservative lower bound on work or
                    # waiting before stage submission.
                    result["thinker_pre_submit_lower_bound"].append(max(0.0, stage0 - engine))
        for stage, name in ((0, "thinker"), (1, "talker")):
            ttft = _stage_value(record, stage, "vllm_ttft_ms")
            if ttft is not None:
                result[f"{name}_ttft"].append(ttft)
            tpot = _stage_value(record, stage, "vllm_tpot_ms")
            if tpot is not None:
                result[f"{name}_tpot"].append(tpot)
        if request is not None and len(stage_engine_values) == 3:
            result["non_stage_total"].append(max(0.0, request - sum(stage_engine_values)))
        client_timing = record.get("client_timing") or {}
        for field, name in (
            ("json_encode_ms", "client_json_encode"),
            ("http_wait_ms", "client_http_wait"),
            ("json_decode_ms", "client_json_decode"),
        ):
            value = _number(client_timing.get(field))
            if value is not None:
                result[name].append(value)
        request_bytes = _number(client_timing.get("request_bytes"))
        if request_bytes is not None:
            result["client_request_mib"].append(request_bytes / 2**20)
    return result


def _gpu_summary(path: Path, start: float, end: float) -> dict[str, Any]:
    if not path.is_file():
        return {}
    by_gpu: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            timestamp = _number(row.get("monotonic_s"))
            if row.get("k") != "sample" or timestamp is None or not start <= timestamp <= end:
                continue
            by_gpu.setdefault(int(row["gpu"]), []).append(row)
    fields = (
        "gpu_busy_pct",
        "sm_active_pct",
        "sm_occupancy_pct",
        "tensor_active_pct",
        "dram_active_pct",
        "power_w",
    )
    result: dict[str, Any] = {}
    for gpu, rows in sorted(by_gpu.items()):
        result[str(gpu)] = {
            field: _distribution(
                [value for row in rows if (value := _number(row.get(field))) is not None]
            )
            for field in fields
        }
    return result


def analyze(label: str, directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = [record for session in manifest["sessions"] for record in session["records"]]
    e2e = [float(record["e2e_slot_latency_ms"]) for record in records]
    request = [float(record["request_latency_ms"]) for record in records]
    queue = [float(record["app_queue_ms"]) for record in records]
    misses = sum(bool(record["deadline_miss"]) for record in records)
    slots = int(manifest["slots_per_user"])
    third = max(1, slots // 3)
    first_queue = [float(record["app_queue_ms"]) for record in records if int(record["slot"]) < third]
    last_queue = [float(record["app_queue_ms"]) for record in records if int(record["slot"]) >= slots - third]
    queue_growth = (_percentile(last_queue, 50) or 0.0) - (_percentile(first_queue, 50) or 0.0)
    components = _component_values(records)
    start = float(manifest["benchmark_start_monotonic_s"])
    end = start + max(float(record["response_ready_ms"]) for record in records) / 1000.0
    request_p50 = _percentile(request, 50) or 0.0
    return {
        "label": label,
        "users": manifest["users"],
        "slots_per_user": slots,
        "records": len(records),
        "prompt_tokens_max": max(int(record["usage"].get("prompt_tokens") or 0) for record in records),
        "e2e_ms": _distribution(e2e),
        "request_ms": _distribution(request),
        "app_queue_ms": _distribution(queue),
        "deadline_miss_rate": misses / len(records),
        "queue_growth_first_to_last_p50_ms": queue_growth,
        "strict_realtime": (_percentile(e2e, 99) or float("inf")) <= SLOT_MS and misses / len(records) <= 0.01,
        "service_p50_exceeds_slot": request_p50 > SLOT_MS,
        # A median backlog increase larger than one whole slot is direct
        # evidence that the run is not keeping up with its input clock. This
        # remains valid when bursty work leaves request p50 just below 480 ms.
        "throughput_collapse": queue_growth > SLOT_MS,
        "components_ms": {name: _distribution(values) for name, values in components.items()},
        "gpu": _gpu_summary(directory / "gpu.jsonl", start, end),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        metavar="LABEL=DIR",
        help="capacity run label and directory containing manifest.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for value in args.runs:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise SystemExit(f"invalid run {value!r}; expected LABEL=DIR")
        reports.append(analyze(label, Path(raw_path)))
    rendered = json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
