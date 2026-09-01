"""Compute MiniCPM native-duplex real-time factors from server traces."""

from __future__ import annotations

import argparse
import base64
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
VISION_ARRIVAL_BATCH = re.compile(
    r"\[MINICPM-PREP-ARRIVAL\] jobs=(\d+) encoded_frames=(\d+) "
    r"cpu_prepare_ms=([0-9]+(?:\.[0-9]+)?) "
    r"vision_encoder_ms=([0-9]+(?:\.[0-9]+)?) "
    r"cache_ms=([0-9]+(?:\.[0-9]+)?) total_ms=([0-9]+(?:\.[0-9]+)?) "
    r"done_epoch=([0-9]+(?:\.[0-9]+)?)"
)
VISION_ARRIVAL_DISPATCH = re.compile(
    r"\[MINICPM-PREP-ARRIVAL-(?:READY|ACCEPTED|SUBMITTED)\] "
    r"success=(True|False) "
    r"arrival_to_(?:ready|accepted|submitted)_ms=([0-9]+(?:\.[0-9]+)?) "
    r"done_epoch=([0-9]+(?:\.[0-9]+)?)"
)
VISION_FORMAL_WAIT = re.compile(
    r"\[MINICPM-PREP-FORMAL-WAIT\] tasks=(\d+) "
    r"wait_ms=([0-9]+(?:\.[0-9]+)?) "
    r"(?:\S+\s+)*"
    r"done_epoch=([0-9]+(?:\.[0-9]+)?)"
)
FRAME_CONSUMED = re.compile(
    r"\[MINICPM-FRAME-CONSUMED\] req=(\S+) seq=(\d+) "
    r"frames=(\d+) source=(\S+) done_epoch=([0-9]+(?:\.[0-9]+)?)"
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


def _rtf_miss_count(values: list[float]) -> int:
    """Count model units whose service rate is slower than real time."""
    return sum(value < 1.0 for value in values)


def _request_session_id(request_id: str) -> str | None:
    """Decode the session id embedded in a current duplex resource id."""
    parts = request_id.split(".")
    if len(parts) != 8 or parts[0] != "duplex-s":
        return None
    encoded = parts[1]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _measurement_sequence_ranges(run: dict[str, Any]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for user in run.get("users", []):
        if not isinstance(user, dict):
            continue
        session_id = user.get("session_id")
        start = user.get("formal_seq_start")
        end = user.get("formal_seq_end")
        if (
            isinstance(session_id, str)
            and session_id
            and isinstance(start, int)
            and isinstance(end, int)
            and 1 <= start <= end
        ):
            ranges[session_id] = (start, end)
    return ranges


def _sequence_in_measurement(
    request_id: str,
    sequence: int,
    ranges: dict[str, tuple[int, int]],
) -> bool:
    if not ranges:
        return False
    session_id = _request_session_id(request_id)
    bounds = ranges.get(session_id) if session_id is not None else None
    return bounds is not None and bounds[0] <= sequence <= bounds[1]


def _pd_long_horizon_summary(
    pd_slots: list[dict[str, Any]],
    input_units_per_session: int | None,
    measurement_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Summarize sustained per-session P-to-D progress.

    Per-unit tail RTF is useful for jitter, but it is too strict for capacity:
    a slow unit can be recovered by later fast units. Stream RTF compares the
    completed one-second input budget with wall time from the first input-ready
    event through the last D completion. It therefore includes backlog growth
    without counting the same queue interval once per delayed unit.

    The mean-latency budget RTF is retained only as a responsiveness diagnostic.
    It sums per-unit latencies, so it is not a throughput/capacity metric.
    """
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for slot in pd_slots:
        try:
            sequence = int(slot["seq"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(slot["request_id"]), sequence)
        previous = latest.get(key)
        if previous is None or slot["done_epoch"] > previous["done_epoch"]:
            latest[key] = slot

    by_session: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (request_id, sequence), slot in latest.items():
        by_session[request_id].append((sequence, slot))

    latency_budget_rtfs: list[float] = []
    stream_rtfs: list[float] = []
    cadence_rtfs: list[float] = []
    completed_units: list[float] = []
    terminal_lag_ms: list[float] = []
    lag_growth_ms: list[float] = []
    lag_growth_per_unit_ms: list[float] = []
    terminal_stream_backlog_ms: list[float] = []
    complete_sessions = 0
    all_e2e_ms: list[float] = []
    for rows in by_session.values():
        rows.sort(key=lambda item: item[0])
        e2e_ms = [float(slot["e2e_ms"]) for _, slot in rows]
        if not e2e_ms:
            continue
        all_e2e_ms.extend(e2e_ms)
        completed_units.append(float(len(rows)))
        latency_budget_rtfs.append(len(rows) * MODEL_UNIT_MS / sum(e2e_ms))
        first_sequence, first = rows[0]
        last_sequence, last = rows[-1]
        stream_wall_ms = (
            float(last["done_epoch"]) - float(first["ready_epoch"])
        ) * 1000.0
        if stream_wall_ms > 0:
            stream_rtfs.append(len(rows) * MODEL_UNIT_MS / stream_wall_ms)
            terminal_stream_backlog_ms.append(
                stream_wall_ms - len(rows) * MODEL_UNIT_MS
            )
        terminal_lag_ms.append(float(last["e2e_ms"]))
        growth = float(last["e2e_ms"]) - float(first["e2e_ms"])
        lag_growth_ms.append(growth)
        sequence_span = last_sequence - first_sequence
        if sequence_span > 0:
            completion_span_ms = (float(last["done_epoch"]) - float(first["done_epoch"])) * 1000.0
            if completion_span_ms > 0:
                cadence_rtfs.append(sequence_span * MODEL_UNIT_MS / completion_span_ms)
            lag_growth_per_unit_ms.append(growth / sequence_span)
        session_id = _request_session_id(str(first["request_id"]))
        bounds = (measurement_ranges or {}).get(session_id) if session_id is not None else None
        complete = (
            bounds is not None
            and len(rows) == bounds[1] - bounds[0] + 1
            and first_sequence == bounds[0]
            and last_sequence == bounds[1]
        )
        if bounds is None:
            complete = (
                input_units_per_session is not None
                and len(rows) == input_units_per_session
                and first_sequence == 1
                and last_sequence == input_units_per_session
            )
        if complete:
            complete_sessions += 1

    latency_budget_rtf = (
        len(all_e2e_ms) * MODEL_UNIT_MS / sum(all_e2e_ms)
        if all_e2e_ms
        else None
    )
    return {
        "definition": (
            "stream_rtf = completed one-second input budget / wall time from "
            "the first input-ready event through the last D completion; "
            "values >= 1 sustain real time"
        ),
        "sessions": len(by_session),
        "complete_sessions": complete_sessions,
        "expected_units_per_session": input_units_per_session,
        "completed_units_per_session": _latency_summary(completed_units),
        "per_session_stream_rtf": _rtf_summary(stream_rtfs),
        "completion_cadence_rtf": _rtf_summary(cadence_rtfs),
        "terminal_stream_backlog_ms": _latency_summary(
            terminal_stream_backlog_ms
        ),
        "mean_latency_budget_rtf": (
            round(latency_budget_rtf, 4)
            if latency_budget_rtf is not None
            else None
        ),
        "per_session_mean_latency_budget_rtf": _rtf_summary(
            latency_budget_rtfs
        ),
        "terminal_e2e_ms": _latency_summary(terminal_lag_ms),
        "e2e_growth_ms": _latency_summary(lag_growth_ms),
        "e2e_growth_per_unit_ms": _latency_summary(lag_growth_per_unit_ms),
    }


def _pd_chain_summary(
    completions: list[dict[str, Any]],
    pd_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Decompose matched P-ready -> P -> post-P/D completion latency."""
    stage_records: dict[tuple[int, str, int], dict[str, Any]] = {}
    for completion in completions:
        generation = completion.get("generation")
        if completion["stage"] not in (0, 1) or not isinstance(generation, int):
            continue
        key = (completion["stage"], completion["request_id"], generation)
        previous = stage_records.get(key)
        if previous is None or completion["done_epoch"] > previous["done_epoch"]:
            stage_records[key] = completion

    ready_to_p_submit_ms: list[float] = []
    p_service_ms: list[float] = []
    p_done_to_d_done_ms: list[float] = []
    d_service_ms: list[float] = []
    chain_ms: list[float] = []
    for slot in pd_slots:
        try:
            generation = int(slot["seq"])
        except (KeyError, TypeError, ValueError):
            continue
        request_id = str(slot["request_id"])
        p_record = stage_records.get((0, request_id, generation))
        d_record = stage_records.get((1, request_id, generation))
        if p_record is None or d_record is None:
            continue
        ready_epoch = float(slot["ready_epoch"])
        ready_to_p_submit_ms.append(
            max((float(p_record["submit_epoch"]) - ready_epoch) * 1000.0, 0.0)
        )
        p_service_ms.append(float(p_record["service_ms"]))
        p_done_to_d_done_ms.append(
            max(
                (float(d_record["done_epoch"]) - float(p_record["done_epoch"]))
                * 1000.0,
                0.0,
            )
        )
        d_service_ms.append(float(d_record["service_ms"]))
        chain_ms.append(float(slot["e2e_ms"]))
    return {
        "matched_units": len(chain_ms),
        "ready_to_p_submit_ms": _latency_summary(ready_to_p_submit_ms),
        "p_service_ms": _latency_summary(p_service_ms),
        "p_done_to_d_done_ms": _latency_summary(p_done_to_d_done_ms),
        "d_service_ms": _latency_summary(d_service_ms),
        "p_ready_to_d_done_ms": _latency_summary(chain_ms),
    }


def _pd_context_cycle_summary(
    completions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure steady context cycles between consecutive D prompt resets.

    MiniCPM P/D creates a finite D request for every model unit.  Its reported
    prompt length therefore exposes context rollover directly: the active
    prompt drops from tens of thousands of tokens to the retained unit.  A
    rollover-to-rollover interval compares the same context phase and avoids
    making a long-run result depend on where the finite benchmark stops.
    """
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for completion in completions:
        generation = completion.get("generation")
        if completion["stage"] != 1 or not isinstance(generation, int):
            continue
        by_session[str(completion["request_id"])].append(completion)

    rollover_counts: list[float] = []
    cycle_spans: list[float] = []
    cycle_rtfs: list[float] = []
    sessions_with_rollover = 0
    sessions_with_complete_cycle = 0
    for records in by_session.values():
        records.sort(key=lambda item: int(item["generation"]))
        rollovers: list[dict[str, Any]] = []
        for previous, current in zip(records, records[1:]):
            previous_tokens = int(previous["tokens_in"])
            current_tokens = int(current["tokens_in"])
            if previous_tokens >= 4096 and current_tokens * 2 < previous_tokens:
                rollovers.append(current)
        rollover_counts.append(float(len(rollovers)))
        if rollovers:
            sessions_with_rollover += 1
        if len(rollovers) >= 2:
            sessions_with_complete_cycle += 1
        for first, last in zip(rollovers, rollovers[1:]):
            sequence_span = int(last["generation"]) - int(first["generation"])
            wall_ms = (float(last["done_epoch"]) - float(first["done_epoch"])) * 1000.0
            if sequence_span <= 0 or wall_ms <= 0:
                continue
            cycle_spans.append(float(sequence_span))
            cycle_rtfs.append(sequence_span * MODEL_UNIT_MS / wall_ms)
    return {
        "definition": (
            "context_cycle_rtf = model-unit duration between consecutive D "
            "prompt resets / matching D-completion wall time"
        ),
        "sessions": len(by_session),
        "sessions_with_rollover": sessions_with_rollover,
        "sessions_with_complete_cycle": sessions_with_complete_cycle,
        "rollovers_per_session": _latency_summary(rollover_counts),
        "cycle_span_units": _latency_summary(cycle_spans),
        "context_cycle_rtf": _rtf_summary(cycle_rtfs),
    }


def _parse_vision_timing(
    log_path: Path,
    started: float,
    ended: float,
) -> dict[str, Any]:
    batches: list[dict[str, float | int]] = []
    ready_ms: list[float] = []
    ready_successes = 0
    formal_wait_ms: list[float] = []
    formal_wait_tasks = 0
    for line in log_path.read_text(errors="replace").splitlines():
        batch = VISION_ARRIVAL_BATCH.search(line)
        if batch is not None and started <= float(batch.group(7)) <= ended:
            batches.append(
                {
                    "jobs": int(batch.group(1)),
                    "encoded_frames": int(batch.group(2)),
                    "cpu_prepare_ms": float(batch.group(3)),
                    "vision_encoder_ms": float(batch.group(4)),
                    "cache_ms": float(batch.group(5)),
                    "total_ms": float(batch.group(6)),
                }
            )
            continue
        dispatch = VISION_ARRIVAL_DISPATCH.search(line)
        if dispatch is not None and started <= float(dispatch.group(3)) <= ended:
            ready_successes += dispatch.group(1) == "True"
            ready_ms.append(float(dispatch.group(2)))
            continue
        formal_wait = VISION_FORMAL_WAIT.search(line)
        if formal_wait is not None and started <= float(formal_wait.group(3)) <= ended:
            formal_wait_tasks += int(formal_wait.group(1))
            formal_wait_ms.append(float(formal_wait.group(2)))

    by_batch_size: dict[str, dict[str, Any]] = {}
    for size in sorted({int(batch["jobs"]) for batch in batches}):
        selected = [batch for batch in batches if int(batch["jobs"]) == size]
        by_batch_size[str(size)] = {
            "batches": len(selected),
            "vision_encoder_ms": _latency_summary(
                [float(batch["vision_encoder_ms"]) for batch in selected]
            ),
            "total_ms": _latency_summary(
                [float(batch["total_ms"]) for batch in selected]
            ),
        }
    return {
        "batches": len(batches),
        "jobs": sum(int(batch["jobs"]) for batch in batches),
        "encoded_frames": sum(int(batch["encoded_frames"]) for batch in batches),
        "cpu_prepare_ms": _latency_summary(
            [float(batch["cpu_prepare_ms"]) for batch in batches]
        ),
        "vision_encoder_ms": _latency_summary(
            [float(batch["vision_encoder_ms"]) for batch in batches]
        ),
        "cache_ms": _latency_summary(
            [float(batch["cache_ms"]) for batch in batches]
        ),
        "batch_total_ms": _latency_summary(
            [float(batch["total_ms"]) for batch in batches]
        ),
        "by_batch_size": by_batch_size,
        "arrival_to_dispatch_ms": _latency_summary(ready_ms),
        "arrival_dispatch_successes": ready_successes,
        # Backward-compatible aliases for archived READY traces.
        "arrival_to_ready_ms": _latency_summary(ready_ms),
        "arrival_ready_successes": ready_successes,
        "formal_wait_calls": len(formal_wait_ms),
        "formal_wait_tasks": formal_wait_tasks,
        "formal_wait_ms": _latency_summary(formal_wait_ms),
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


def _input_units_per_session(run: dict[str, Any]) -> int | None:
    """Return the common number of client-generated model units, if known."""
    units = {
        int(user["units_sent"])
        for user in run.get("users", [])
        if isinstance(user, dict) and isinstance(user.get("units_sent"), int)
    }
    if len(units) != 1:
        return None
    value = units.pop()
    return value if value > 0 else None


def _expected_input_units(
    run: dict[str, Any],
    units_per_session: int | None,
) -> int | None:
    if units_per_session is None:
        return None
    users = run.get("users")
    if not isinstance(users, list) or not users:
        return None
    return units_per_session * len(users)


def _measurement_is_complete(
    run: dict[str, Any],
    expected_input_units: int | None,
    completed_input_units: dict[str, int],
    *,
    is_pd: bool,
) -> bool:
    """Use the terminal Thinker stage as the causal completion witness.

    In P/D mode every finite D unit can only be emitted after its matching P
    unit has completed and its KV is available.  StagePool's P-side DONE
    metric can be coalesced when adjacent resumable segments finish close
    together, so requiring both log counters creates false incomplete runs.
    The D counter is the stronger end-to-end witness; non-P/D runs use stage 0.
    """
    witness_stage = "1" if is_pd else "0"
    return (
        expected_input_units is not None
        and int(run.get("failed_users", 0)) == 0
        and completed_input_units.get(witness_stage) == expected_input_units
    )


def _periodic_pd_records(
    completions: list[dict[str, Any]],
    pd_slots: list[dict[str, Any]],
    input_units_per_session: int | None,
    measurement_ranges: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude setup and post-stream continuation work from input-unit SLOs.

    MiniCPM may run generation 0 while a session is admitted and may create
    autonomous continuation slots after the last client input.  Both consume
    real engine capacity, but neither is one of the client's measured 1-second
    AV units.  Mixing them into input-ready latency makes adjacent capacity
    runs depend on how many continuation slots happen to finish before the
    benchmark's wall-clock cutoff.
    """
    if input_units_per_session is None and not measurement_ranges:
        return completions, pd_slots

    periodic_completions: list[dict[str, Any]] = []
    for completion in completions:
        if completion["stage"] not in (0, 1):
            periodic_completions.append(completion)
            continue
        generation = completion.get("generation")
        in_range = (
            isinstance(generation, int)
            and measurement_ranges
            and _sequence_in_measurement(
                str(completion.get("request_id", "")),
                generation,
                measurement_ranges,
            )
        )
        if isinstance(generation, int) and (
            in_range
            or (not measurement_ranges and 1 <= generation <= input_units_per_session)
        ):
            periodic_completions.append(completion)

    periodic_slots: list[dict[str, Any]] = []
    for slot in pd_slots:
        try:
            sequence = int(slot["seq"])
        except (TypeError, ValueError):
            continue
        in_range = bool(
            measurement_ranges
            and _sequence_in_measurement(
                str(slot.get("request_id", "")),
                sequence,
                measurement_ranges,
            )
        )
        if in_range or (not measurement_ranges and 1 <= sequence <= input_units_per_session):
            periodic_slots.append(slot)
    return periodic_completions, periodic_slots


def _parse_frame_audit(
    log_path: Path,
    started: float,
    ended: float,
    measurement_ranges: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    latest: dict[tuple[str, int], tuple[int, str]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = FRAME_CONSUMED.search(line)
        if match is None:
            continue
        done_epoch = float(match.group(5))
        if not started <= done_epoch <= ended:
            continue
        request_id = match.group(1)
        sequence = int(match.group(2))
        if measurement_ranges and not _sequence_in_measurement(
            request_id,
            sequence,
            measurement_ranges,
        ):
            continue
        latest[(request_id, sequence)] = (int(match.group(3)), match.group(4))
    by_source: dict[str, int] = defaultdict(int)
    for frames, source in latest.values():
        by_source[source] += frames
    return {
        "units": len(latest),
        "frames_consumed": sum(frames for frames, _ in latest.values()),
        "by_source": dict(sorted(by_source.items())),
    }


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
        "accumulated_unit_rtf": round(
            len(service_ms) * MODEL_UNIT_MS / sum(service_ms), 4
        )
        if periodic and service_ms and sum(service_ms) > 0
        else None,
        "unit_rtf": _rtf_summary(unit_rtf),
        "unit_rtf_lt_1": _rtf_miss_count(unit_rtf),
        "unit_rtf_lt_1_rate": round(_rtf_miss_count(unit_rtf) / len(unit_rtf), 4)
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
    server_log = Path(args.server_log)
    admits, schedules, completions, pd_slots = _parse(
        server_log,
        started,
        ended,
    )
    vision_timing = _parse_vision_timing(server_log, started, ended)
    _pair_admits(admits, schedules, completions)
    all_completions = completions
    all_pd_slots = pd_slots
    input_units_per_session = _input_units_per_session(run)
    measurement_ranges = _measurement_sequence_ranges(run)
    completions, pd_slots = _periodic_pd_records(
        completions,
        pd_slots,
        input_units_per_session,
        measurement_ranges,
    )
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
    rtf_pass = bool(periodic_stages) and all(
        summary.get("unit_rtf_lt_1", 0) == 0 for summary in periodic_stages.values()
    )
    p95_pass = bool(periodic_stage_p05) and all(
        value >= 1.0 for value in periodic_stage_p05.values()
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
    expected_input_units = _expected_input_units(run, input_units_per_session)
    required_input_stages = (0, 1) if is_pd else (0,)
    completed_input_units = {
        str(stage): sum(item["stage"] == stage for item in completions)
        for stage in required_input_stages
    }
    measurement_complete = _measurement_is_complete(
        run,
        expected_input_units,
        completed_input_units,
        is_pd=is_pd,
    )
    frame_audit = _parse_frame_audit(
        server_log,
        started,
        ended,
        measurement_ranges,
    )
    expected_frames = run.get("frames_sent")
    frame_audit_required = run.get("config", {}).get("frame_audit_required") is True
    frame_audit_complete = bool(
        isinstance(expected_frames, int)
        and expected_frames >= 0
        and frame_audit["frames_consumed"] == expected_frames
        and frame_audit["units"] == expected_input_units
    )
    if frame_audit_required:
        measurement_complete = measurement_complete and frame_audit_complete
    pd_long_horizon = (
        _pd_long_horizon_summary(
            pd_slots,
            input_units_per_session,
            measurement_ranges,
        )
        if is_pd
        else None
    )
    expected_sessions = len(run.get("users", []))
    long_horizon_rtf_pass = bool(
        pd_long_horizon
        and pd_long_horizon.get("per_session_stream_rtf")
        and float(
            pd_long_horizon.get("per_session_stream_rtf", {}).get("min")
            or 0.0
        )
        >= 1.0
        and int(pd_long_horizon.get("complete_sessions", 0)) == expected_sessions
    )
    primary_capacity_pass = (
        long_horizon_rtf_pass and measurement_complete
        if is_pd
        else rtf_pass and measurement_complete
    )
    result = {
        "run_json": str(Path(args.run_json).resolve()),
        "definition": "RTF = 1 second model unit / stage service time; RTF >= 1 is real-time",
        "capacity_slo": (
            "the complete long-session input set finishes without errors and "
            "every session's stream RTF is >= 1 from first input-ready through "
            "last D completion; per-unit P/D/Talker RTF remains a tail "
            "diagnostic; setup "
            "and post-stream autonomous continuations are excluded; Code2Wav "
            "persistent-session wall time is excluded"
            if is_pd
            else "every observed Thinker and Talker unit has RTF >= 1; "
            "Code2Wav persistent-session wall time is excluded; capacity_pass "
            "also requires a complete, error-free measurement"
        ),
        "rtf_capacity_pass": rtf_pass,
        "strict_unit_rtf_capacity_pass": rtf_pass and measurement_complete,
        "long_horizon_rtf_capacity_pass": long_horizon_rtf_pass and measurement_complete,
        "capacity_pass": primary_capacity_pass,
        "p95_capacity_pass": p95_pass and measurement_complete,
        "measurement_complete": measurement_complete,
        "expected_input_units": expected_input_units,
        "completed_input_units": completed_input_units,
        "completion_witness_stage": 1 if is_pd else 0,
        "failed_users": int(run.get("failed_users", 0)),
        "frame_audit": {
            **frame_audit,
            "expected_frames": expected_frames,
            "required": frame_audit_required,
            "complete": frame_audit_complete,
        },
        "limiting_periodic_stage": min(periodic_stage_p05, key=periodic_stage_p05.get)
        if periodic_stage_p05
        else None,
        "stages": stages,
    }
    if is_pd:
        result["pd_slot"] = pd_slot_summary
        result["pd_long_horizon"] = pd_long_horizon
        result["pd_chain"] = _pd_chain_summary(completions, pd_slots)
        result["pd_context_cycles"] = _pd_context_cycle_summary(completions)
        result["vision_arrival"] = vision_timing
        result["measurement_scope"] = {
            "input_units_per_session": input_units_per_session,
            "per_session_sequence_ranges": {
                session_id: {"start": bounds[0], "end": bounds[1]}
                for session_id, bounds in sorted(measurement_ranges.items())
            },
            "periodic_stage_requests": {
                str(stage): sum(item["stage"] == stage for item in completions)
                for stage in (0, 1)
            },
            "excluded_stage_requests": {
                str(stage): sum(item["stage"] == stage for item in all_completions)
                - sum(item["stage"] == stage for item in completions)
                for stage in (0, 1)
            },
            "periodic_pd_slots": len(pd_slots),
            "excluded_pd_slots": len(all_pd_slots) - len(pd_slots),
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
