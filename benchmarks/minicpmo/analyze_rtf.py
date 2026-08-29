"""Compute MiniCPM native-duplex real-time factors from server traces."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ADMIT = re.compile(
    r"\[duplex_cadence\] stage=(\d+) ADMIT req=(\S+) generation=(\d+) "
    r"admit_epoch=([0-9]+(?:\.[0-9]+)?)"
)
SCHEDULE = re.compile(
    r"\[duplex_cadence\] stage=(\d+) SCHEDULE req=(\S+) generation=(\d+) "
    r"step=(\d+) schedule_epoch=([0-9]+(?:\.[0-9]+)?) scheduled_tokens=(\d+) "
    r"batch_reqs=(\d+) batch_tokens=(\d+)"
)
RUNNER_DONE = re.compile(
    r"\[duplex_cadence\] stage=(\d+) RUNNER_DONE req=(\S+) generation=(\d+) "
    r"step=(\d+) runner_done_epoch=([0-9]+(?:\.[0-9]+)?)"
)
DONE = re.compile(
    r"\[duplex_stage\] stage=(\d+) DONE req=(\S+) "
    r"submit_epoch=([0-9]+(?:\.[0-9]+)?) done_epoch=([0-9]+(?:\.[0-9]+)?) "
    r"service_ms=([0-9]+(?:\.[0-9]+)?) tokens_in=(\d+) tokens_out=(\d+) "
    r"output_units=(\d+) audio_s=([0-9]+(?:\.[0-9]+)?) "
    r"ttft_ms=([0-9]+(?:\.[0-9]+)?) tpot_ms=([0-9]+(?:\.[0-9]+)?)"
)
PD_SLOT_DONE = re.compile(
    r"\[minicpm_pd_slot\] req=(\S+) seq=(\S+) "
    r"ready_epoch=([0-9]+(?:\.[0-9]+)?) done_epoch=([0-9]+(?:\.[0-9]+)?) "
    r"e2e_ms=([0-9]+(?:\.[0-9]+)?) wait_previous_d_ms=([0-9]+(?:\.[0-9]+)?)"
)
MODEL_UNIT_MS = 1000.0
PD_DECODE_REQUEST = re.compile(r"^(?P<logical>.+)-(?P<slot>[0-9a-f]{8})$")


def _cadence_identity(stage: int, request_id: str, generation: int) -> tuple[str, int]:
    """Map MiniCPM's per-slot physical D id back to its logical session id."""
    if stage == 1 and (match := PD_DECODE_REQUEST.fullmatch(request_id)) is not None:
        return match.group("logical"), int(match.group("slot"), 16)
    return request_id, generation


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    index = max(0, math.ceil(q * len(clean)) - 1)
    return round(clean[min(index, len(clean) - 1)], 3)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": _percentile(values, 1.0),
    }


def _rtf_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "min": _percentile(values, 0.0),
        "p01": _percentile(values, 0.01),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": _percentile(values, 1.0),
    }


def _parse(
    log_path: Path, started: float, ended: float
) -> tuple[
    dict[tuple[int, str, int], float],
    dict[tuple[int, str, int], dict[int, dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    admits: dict[tuple[int, str, int], float] = {}
    schedules: dict[tuple[int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    runner_results: dict[tuple[int, str, int], dict[int, float]] = defaultdict(dict)
    completions: list[dict[str, Any]] = []
    pd_slots: list[dict[str, Any]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        pd_slot = PD_SLOT_DONE.search(line)
        if pd_slot is not None:
            done_epoch = float(pd_slot.group(4))
            if started <= done_epoch <= ended:
                pd_slots.append(
                    {
                        "request_id": pd_slot.group(1),
                        "seq": pd_slot.group(2),
                        "ready_epoch": float(pd_slot.group(3)),
                        "done_epoch": done_epoch,
                        "e2e_ms": float(pd_slot.group(5)),
                        "wait_previous_d_ms": float(pd_slot.group(6)),
                    }
                )
            continue
        admit = ADMIT.search(line)
        if admit is not None:
            timestamp = float(admit.group(4))
            if started <= timestamp <= ended:
                stage = int(admit.group(1))
                request_id, generation = _cadence_identity(
                    stage, admit.group(2), int(admit.group(3))
                )
                admits[(stage, request_id, generation)] = timestamp
            continue
        schedule = SCHEDULE.search(line)
        if schedule is not None:
            timestamp = float(schedule.group(5))
            if started <= timestamp <= ended:
                stage = int(schedule.group(1))
                request_id, generation = _cadence_identity(
                    stage, schedule.group(2), int(schedule.group(3))
                )
                key = (stage, request_id, generation)
                schedules[key][int(schedule.group(4))] = {
                    "schedule_epoch": timestamp,
                    "scheduled_tokens": int(schedule.group(6)),
                    "batch_reqs": int(schedule.group(7)),
                    "batch_tokens": int(schedule.group(8)),
                }
            continue
        runner_done = RUNNER_DONE.search(line)
        if runner_done is not None:
            timestamp = float(runner_done.group(5))
            if started <= timestamp <= ended:
                stage = int(runner_done.group(1))
                request_id, generation = _cadence_identity(
                    stage, runner_done.group(2), int(runner_done.group(3))
                )
                key = (stage, request_id, generation)
                runner_results[key][int(runner_done.group(4))] = timestamp
            continue
        done = DONE.search(line)
        if done is None:
            continue
        done_epoch = float(done.group(4))
        if started <= done_epoch <= ended:
            completions.append(
                {
                    "stage": int(done.group(1)),
                    "request_id": done.group(2),
                    "submit_epoch": float(done.group(3)),
                    "done_epoch": done_epoch,
                    "service_ms": float(done.group(5)),
                    "tokens_in": int(done.group(6)),
                    "tokens_out": int(done.group(7)),
                    "output_units": int(done.group(8)),
                    "audio_s": float(done.group(9)),
                    "ttft_ms": float(done.group(10)),
                    "tpot_ms": float(done.group(11)),
                }
            )
    # Streaming stages can expose multiple cumulative snapshots for one stage
    # submission. Keep the last snapshot; treating each snapshot as a new unit
    # would double-count work and make partial audio look slower than real time.
    latest: dict[tuple[int, str, float], dict[str, Any]] = {}
    for completion in completions:
        key = (
            completion["stage"],
            completion["request_id"],
            round(completion["submit_epoch"], 6),
        )
        previous = latest.get(key)
        if previous is None or completion["done_epoch"] > previous["done_epoch"]:
            latest[key] = completion
    for key, steps in schedules.items():
        for step, runner_done_epoch in runner_results.get(key, {}).items():
            if step in steps:
                steps[step]["runner_done_epoch"] = runner_done_epoch
    return admits, schedules, list(latest.values()), pd_slots


def _pair_admits(
    admits: dict[tuple[int, str, int], float],
    schedules: dict[tuple[int, str, int], dict[int, dict[str, Any]]],
    completions: list[dict[str, Any]],
) -> None:
    generations: dict[tuple[int, str], list[int]] = defaultdict(list)
    for stage, request_id, generation in admits:
        generations[(stage, request_id)].append(generation)
    for values in generations.values():
        values.sort()
    next_index: dict[tuple[int, str], int] = defaultdict(int)
    for completion in sorted(completions, key=lambda item: item["done_epoch"]):
        key = (completion["stage"], completion["request_id"])
        request_generations = generations.get(key, [])
        index = next_index[key]
        while index < len(request_generations):
            generation = request_generations[index]
            admit_epoch = admits[(key[0], key[1], generation)]
            if admit_epoch + 0.001 >= completion["submit_epoch"]:
                break
            index += 1
        if index >= len(request_generations):
            continue
        generation = request_generations[index]
        admit_epoch = admits[(key[0], key[1], generation)]
        if admit_epoch > completion["done_epoch"]:
            continue
        next_index[key] = index + 1
        completion["generation"] = generation
        completion["admit_epoch"] = admit_epoch
        completion["dispatch_ms"] = max((admit_epoch - completion["submit_epoch"]) * 1000.0, 0.0)
        completion["core_to_done_ms"] = max((completion["done_epoch"] - admit_epoch) * 1000.0, 0.0)
        unit_steps = [
            value
            for _, value in sorted(schedules.get((key[0], key[1], generation), {}).items())
            if admit_epoch <= value["schedule_epoch"] <= completion["done_epoch"]
        ]
        if not unit_steps:
            continue
        first = unit_steps[0]
        completion.update(first)
        completion["scheduler_steps"] = len(unit_steps)
        completion["scheduler_queue_ms"] = (first["schedule_epoch"] - admit_epoch) * 1000.0
        completion["scheduled_to_done_ms"] = (completion["done_epoch"] - first["schedule_epoch"]) * 1000.0
        runner_ms = [
            (step["runner_done_epoch"] - step["schedule_epoch"]) * 1000.0
            for step in unit_steps
            if "runner_done_epoch" in step and step["runner_done_epoch"] <= completion["done_epoch"]
        ]
        completion["runner_steps_paired"] = len(runner_ms)
        completion["runner_ms_sum"] = sum(runner_ms)
        if runner_ms:
            completion["first_runner_ms"] = runner_ms[0]
            completion["decode_runner_ms_sum"] = sum(runner_ms[1:])
            completion["runner_ms_per_step"] = sum(runner_ms) / len(runner_ms)
        paired_steps = [
            step
            for step in unit_steps
            if "runner_done_epoch" in step and step["runner_done_epoch"] <= completion["done_epoch"]
        ]
        decode_steps = paired_steps[1:]
        if decode_steps:
            mixed_decode_steps = [
                step for step in decode_steps if step["batch_tokens"] > step["batch_reqs"]
            ]
            mixed_decode_step_runner_ms = [
                (step["runner_done_epoch"] - step["schedule_epoch"]) * 1000.0
                for step in mixed_decode_steps
            ]
            completion["mixed_decode_steps"] = len(mixed_decode_steps)
            completion["mixed_decode_step_rate"] = len(mixed_decode_steps) / len(decode_steps)
            completion["mixed_decode_step_runner_ms"] = mixed_decode_step_runner_ms
            completion["mixed_decode_runner_ms_sum"] = sum(mixed_decode_step_runner_ms)
            decode_only_steps = [step for step in decode_steps if step not in mixed_decode_steps]
            decode_only_step_runner_ms = [
                (step["runner_done_epoch"] - step["schedule_epoch"]) * 1000.0
                for step in decode_only_steps
            ]
            completion["decode_only_step_runner_ms"] = decode_only_step_runner_ms
            completion["decode_only_runner_ms_sum"] = sum(decode_only_step_runner_ms)
            if mixed_decode_steps:
                completion["mixed_decode_runner_ms_per_step"] = (
                    completion["mixed_decode_runner_ms_sum"] / len(mixed_decode_steps)
                )
            if decode_only_steps:
                completion["decode_only_runner_ms_per_step"] = (
                    completion["decode_only_runner_ms_sum"] / len(decode_only_steps)
                )
        completion["inter_step_ms_sum"] = sum(
            max(next_step["schedule_epoch"] - step["runner_done_epoch"], 0.0) * 1000.0
            for step, next_step in zip(paired_steps, paired_steps[1:])
        )
        last = paired_steps[-1] if paired_steps else None
        if last is not None:
            completion["result_exposure_ms"] = max(
                (completion["done_epoch"] - last["runner_done_epoch"]) * 1000.0,
                0.0,
            )


def _stage_summary(
    stage: int,
    completions: list[dict[str, Any]],
    *,
    periodic: bool,
) -> dict[str, Any]:
    records = [item for item in completions if item["stage"] == stage]
    service_ms = [item["service_ms"] for item in records]
    unit_rtf = (
        [MODEL_UNIT_MS / value for value in service_ms if value > 0]
        if periodic
        else []
    )
    audio_rtf = [item["audio_s"] * 1000.0 / item["service_ms"] for item in records if item["audio_s"] > 0]
    result: dict[str, Any] = {
        "requests": len(records),
        "service_ms": _latency_summary(service_ms),
        "unit_rtf": _rtf_summary(unit_rtf),
        "unit_rtf_le_1": sum(value <= 1.0 for value in unit_rtf),
        "unit_rtf_le_1_rate": round(sum(value <= 1.0 for value in unit_rtf) / len(unit_rtf), 4)
        if unit_rtf
        else None,
        "ttft_ms": _latency_summary([item["ttft_ms"] for item in records if item["ttft_ms"] > 0]),
        "dispatch_to_core_ms": _latency_summary(
            [item["dispatch_ms"] for item in records if "dispatch_ms" in item]
        ),
        "core_admit_to_done_ms": _latency_summary(
            [item["core_to_done_ms"] for item in records if "core_to_done_ms" in item]
        ),
        "scheduler_queue_ms": _latency_summary(
            [item["scheduler_queue_ms"] for item in records if "scheduler_queue_ms" in item]
        ),
        "scheduled_to_done_ms": _latency_summary(
            [item["scheduled_to_done_ms"] for item in records if "scheduled_to_done_ms" in item]
        ),
        "runner_ms_sum": _latency_summary(
            [item["runner_ms_sum"] for item in records if "runner_ms_sum" in item]
        ),
        "first_runner_ms": _latency_summary(
            [item["first_runner_ms"] for item in records if "first_runner_ms" in item]
        ),
        "decode_runner_ms_sum": _latency_summary(
            [item["decode_runner_ms_sum"] for item in records if "decode_runner_ms_sum" in item]
        ),
        "runner_ms_per_step": _latency_summary(
            [item["runner_ms_per_step"] for item in records if "runner_ms_per_step" in item]
        ),
        "mixed_decode_step_rate": _rtf_summary(
            [item["mixed_decode_step_rate"] for item in records if "mixed_decode_step_rate" in item]
        ),
        "mixed_decode_runner_ms_sum": _latency_summary(
            [item["mixed_decode_runner_ms_sum"] for item in records if "mixed_decode_runner_ms_sum" in item]
        ),
        "decode_only_runner_ms_sum": _latency_summary(
            [item["decode_only_runner_ms_sum"] for item in records if "decode_only_runner_ms_sum" in item]
        ),
        "mixed_decode_runner_ms_per_step": _latency_summary(
            [
                item["mixed_decode_runner_ms_per_step"]
                for item in records
                if "mixed_decode_runner_ms_per_step" in item
            ]
        ),
        "decode_only_runner_ms_per_step": _latency_summary(
            [
                item["decode_only_runner_ms_per_step"]
                for item in records
                if "decode_only_runner_ms_per_step" in item
            ]
        ),
        "mixed_decode_step_runner_ms": _latency_summary(
            [
                value
                for item in records
                for value in item.get("mixed_decode_step_runner_ms", [])
            ]
        ),
        "decode_only_step_runner_ms": _latency_summary(
            [
                value
                for item in records
                for value in item.get("decode_only_step_runner_ms", [])
            ]
        ),
        "inter_step_ms_sum": _latency_summary(
            [item["inter_step_ms_sum"] for item in records if "inter_step_ms_sum" in item]
        ),
        "result_exposure_ms": _latency_summary(
            [item["result_exposure_ms"] for item in records if "result_exposure_ms" in item]
        ),
        "scheduler_steps": _latency_summary(
            [float(item["scheduler_steps"]) for item in records if "scheduler_steps" in item]
        ),
        "first_schedule_batch_reqs": _latency_summary(
            [float(item["batch_reqs"]) for item in records if "batch_reqs" in item]
        ),
        "first_schedule_batch_tokens": _latency_summary(
            [float(item["batch_tokens"]) for item in records if "batch_tokens" in item]
        ),
        "tokens_in": _latency_summary([float(item["tokens_in"]) for item in records]),
        "tokens_out": _latency_summary([float(item["tokens_out"]) for item in records]),
    }
    if audio_rtf:
        result["persistent_audio_wall_ratio"] = _rtf_summary(audio_rtf)
        result["audio_duration_s"] = _latency_summary([item["audio_s"] for item in records if item["audio_s"] > 0])
        result["persistent_stage_note"] = (
            "This request spans session idle time; do not use its wall ratio as per-unit RTF."
        )
    if records:
        threshold = _percentile(service_ms, 0.95)
        tail = [item for item in records if threshold is not None and item["service_ms"] >= threshold]
        component_names = (
            "dispatch_ms",
            "scheduler_queue_ms",
            "runner_ms_sum",
            "first_runner_ms",
            "decode_runner_ms_sum",
            "mixed_decode_runner_ms_sum",
            "decode_only_runner_ms_sum",
            "mixed_decode_step_rate",
            "inter_step_ms_sum",
            "result_exposure_ms",
            "scheduler_steps",
            "batch_reqs",
            "batch_tokens",
        )
        tail_result: dict[str, Any] = {
            "count": len(tail),
            "service_threshold_ms": threshold,
            "service_ms": _latency_summary([item["service_ms"] for item in tail]),
        }
        for name in component_names:
            values = [float(item[name]) for item in tail if name in item]
            tail_result[name] = _latency_summary(values)
        runner_values = [
            item["runner_ms_sum"] / item["service_ms"]
            for item in tail
            if item.get("runner_ms_sum") is not None and item["service_ms"] > 0
        ]
        tail_result["runner_service_share"] = _rtf_summary(runner_values)
        result["slowest_5pct"] = tail_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = json.loads(Path(args.run_json).read_text())
    started = float(run["started_epoch_s"]) - 0.25
    ended = float(run["ended_epoch_s"]) + 0.25
    admits, schedules, completions, pd_slots = _parse(
        Path(args.server_log),
        started,
        ended,
    )
    _pair_admits(admits, schedules, completions)
    stage_ids = sorted({item["stage"] for item in completions})
    is_pd = bool(pd_slots) or 3 in stage_ids
    periodic_stage_ids = (0, 1, 2) if is_pd else (0, 1)
    stages = {
        str(stage): _stage_summary(
            stage,
            completions,
            periodic=stage in periodic_stage_ids,
        )
        for stage in stage_ids
    }
    periodic_stages = {
        str(stage): stages[str(stage)]
        for stage in periodic_stage_ids
        if str(stage) in stages
    }
    periodic_stage_p05 = {
        stage: (summary.get("unit_rtf", {}).get("p05") or 0.0)
        for stage, summary in periodic_stages.items()
    }
    strict_pass = bool(periodic_stages) and all(
        summary.get("unit_rtf_le_1", 0) == 0 for summary in periodic_stages.values()
    )
    p95_pass = bool(periodic_stage_p05) and all(
        value > 1.0 for value in periodic_stage_p05.values()
    )
    pd_slot_e2e = [slot["e2e_ms"] for slot in pd_slots]
    pd_slot_summary = {
        "count": len(pd_slot_e2e),
        "e2e_ms": _latency_summary(pd_slot_e2e),
        "e2e_over_1000ms": sum(value >= MODEL_UNIT_MS for value in pd_slot_e2e),
        "wait_previous_d_ms": _latency_summary(
            [slot["wait_previous_d_ms"] for slot in pd_slots]
        ),
    }
    if is_pd:
        strict_pass = strict_pass and bool(pd_slot_e2e) and all(
            value < MODEL_UNIT_MS for value in pd_slot_e2e
        )
        p95_e2e = _percentile(pd_slot_e2e, 0.95)
        p95_pass = p95_pass and p95_e2e is not None and p95_e2e < MODEL_UNIT_MS
    result = {
        "run_json": str(Path(args.run_json).resolve()),
        "definition": "RTF = 1 second model unit / stage service time; RTF > 1 is real-time",
        "capacity_slo": (
            "every P, D, and Talker unit has RTF > 1 and every input-ready-to-D "
            "slot latency is below 1 second; Code2Wav persistent-session wall time is excluded"
            if is_pd
            else "every observed Thinker and Talker unit has RTF > 1; "
            "Code2Wav persistent-session wall time is excluded"
        ),
        "capacity_pass": strict_pass,
        "p95_capacity_pass": p95_pass,
        "limiting_periodic_stage": min(periodic_stage_p05, key=periodic_stage_p05.get)
        if periodic_stage_p05
        else None,
        "stages": stages,
    }
    if is_pd:
        result["pd_slot"] = pd_slot_summary
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
