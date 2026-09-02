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
PREFIX_CACHE = re.compile(r"\[prefix-cache\] request=(\S+) hit_tokens=(\d+) prompt_tokens=(\d+)")
PROVENANCE_MARKER = re.compile(r"\[benchmark-provenance\] run_id=([0-9a-f]+)")
TRUNCATED_APPEND = re.compile(r"MiniCPM-o duplex append produced (\d+) embeddings .* reserved only (\d+) prompt slots")
DIAGNOSTIC_MARKERS = {
    "duplex_cadence": re.compile(r"\[(?:duplex_cadence|duplex_stage)\]"),
    "prep_diag": re.compile(r"\[MINICPM-PREP"),
    "handoff_diag": re.compile(r"\[(?:HANDOFF|INGRESS)-DIAG\]"),
    "core_step_diag": re.compile(r"\[CORE-STEP-DIAG\]"),
    "runner_diag": re.compile(r"\[(?:RUNNER-DIAG|STEP-GPU|PD-ITER)\]"),
    "gpu_probe": re.compile(r"\[(?:ENC-GPU|MTP-GPU|SPAN)\]"),
    "audio_chunk_log": re.compile(r"\[AUDIO-CHUNK\]"),
    "nixl_connector_diag": re.compile(r"\[(?:NIXL-[PD]-TRACE|NIXL-PUSH-DIAG|nixl-delta-(?:load|push))\]"),
    "pd_slot": re.compile(r"\[minicpm_pd_slot\]"),
    "prefix_cache_log": re.compile(r"\[prefix-cache\]"),
    "orchestrator_lag": re.compile(r"\[orch-lag\]"),
}
RUNTIME_FAILURE_MARKERS = {
    "pd_prefix_changed": re.compile(r"Prepared P/D prefix changed before activation", re.IGNORECASE),
    "remote_prefill_fallback": re.compile(r"ordinary remote-prefill path", re.IGNORECASE),
    "formal_frame_fallback": re.compile(r"source=formal_fallback"),
    "nixl_failure": re.compile(
        r"(?:NIXL|nixl).*(?:handshake|transfer|registration).*(?:failed|error)",
        re.IGNORECASE,
    ),
}
PREEMPTION_OR_EVICTION = re.compile(
    r"(?:"
    r"(?:request|sequence|kv|cache).*(?:preempted|evicted)"
    r"|\bPreemptions:\s*[1-9]\d*"
    r")",
    re.IGNORECASE,
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
        # Physical D sequence also counts autonomous continuations. Capacity
        # is indexed by real client input, so prefer the explicit input-unit
        # range and retain the old names only for artifact compatibility.
        start = user.get("formal_input_unit_start", user.get("formal_seq_start"))
        end = user.get("formal_input_unit_end", user.get("formal_seq_end"))
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
    completed one-second input budget with wall time from the first media
    arrival through the last D completion. It therefore includes input
    aggregation and backlog growth without counting the same queue interval
    once per delayed unit.

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
    terminal_ready_to_d_ms: list[float] = []
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
        stream_wall_ms = (float(last["done_epoch"]) - float(first["ready_epoch"])) * 1000.0
        if stream_wall_ms > 0:
            stream_rtfs.append(len(rows) * MODEL_UNIT_MS / stream_wall_ms)
            terminal_stream_backlog_ms.append(stream_wall_ms - len(rows) * MODEL_UNIT_MS)
        terminal_lag_ms.append(float(last["e2e_ms"]))
        last_ready_to_d_ms = last.get("ready_to_d_ms")
        if isinstance(last_ready_to_d_ms, int | float):
            terminal_ready_to_d_ms.append(float(last_ready_to_d_ms))
        growth = float(last["e2e_ms"]) - float(first["e2e_ms"])
        lag_growth_ms.append(growth)
        sequence_span = last_sequence - first_sequence
        if sequence_span > 0:
            completion_span_ms = (float(last["done_epoch"]) - float(first["done_epoch"])) * 1000.0
            if completion_span_ms > 0:
                cadence_rtfs.append(sequence_span * MODEL_UNIT_MS / completion_span_ms)
            lag_growth_per_unit_ms.append(growth / sequence_span)
        raw_request_id = str(first["request_id"])
        session_id = _request_session_id(raw_request_id)
        if session_id is None and raw_request_id in (measurement_ranges or {}):
            session_id = raw_request_id
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

    latency_budget_rtf = len(all_e2e_ms) * MODEL_UNIT_MS / sum(all_e2e_ms) if all_e2e_ms else None
    return {
        "definition": (
            "stream_rtf = completed one-second input budget / wall time from "
            "the first media arrival through the last D completion; "
            "values >= 1 sustain real time"
        ),
        "sessions": len(by_session),
        "complete_sessions": complete_sessions,
        "expected_units_per_session": input_units_per_session,
        "completed_units_per_session": _latency_summary(completed_units),
        "per_session_stream_rtf": _rtf_summary(stream_rtfs),
        "completion_cadence_rtf": _rtf_summary(cadence_rtfs),
        "terminal_stream_backlog_ms": _latency_summary(terminal_stream_backlog_ms),
        "mean_latency_budget_rtf": (round(latency_budget_rtf, 4) if latency_budget_rtf is not None else None),
        "per_session_mean_latency_budget_rtf": _rtf_summary(latency_budget_rtfs),
        "terminal_e2e_ms": _latency_summary(terminal_lag_ms),
        "terminal_input_start_e2e_ms": _latency_summary(terminal_lag_ms),
        "terminal_ready_to_d_ms": _latency_summary(terminal_ready_to_d_ms),
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
    input_start_chain_ms: list[float] = []
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
        ready_epoch = float(slot.get("model_unit_ready_epoch", slot["ready_epoch"]))
        ready_to_p_submit_ms.append(max((float(p_record["submit_epoch"]) - ready_epoch) * 1000.0, 0.0))
        p_service_ms.append(float(p_record["service_ms"]))
        p_done_to_d_done_ms.append(
            max(
                (float(d_record["done_epoch"]) - float(p_record["done_epoch"])) * 1000.0,
                0.0,
            )
        )
        d_service_ms.append(float(d_record["service_ms"]))
        chain_ms.append(float(slot.get("ready_to_d_ms", slot["e2e_ms"])))
        input_start_chain_ms.append(float(slot["e2e_ms"]))
    return {
        "matched_units": len(chain_ms),
        "ready_to_p_submit_ms": _latency_summary(ready_to_p_submit_ms),
        "model_ready_to_p_submit_ms": _latency_summary(ready_to_p_submit_ms),
        "p_service_ms": _latency_summary(p_service_ms),
        "p_done_to_d_done_ms": _latency_summary(p_done_to_d_done_ms),
        "d_service_ms": _latency_summary(d_service_ms),
        "p_ready_to_d_done_ms": _latency_summary(chain_ms),
        "model_ready_to_d_done_ms": _latency_summary(chain_ms),
        "input_start_to_d_done_ms": _latency_summary(input_start_chain_ms),
    }


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0.0:
        return None
    return round(
        sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator,
        4,
    )


def _pd_session_recurrence_summary(
    pd_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate inherited same-session D backlog from current-slot work.

    A physical D request's submit time is reconstructed as completion minus
    its measured D service interval. For every consecutive pair of formal
    input units, ``max(current_ready, previous_done)`` is the earliest point
    at which the current unit is both runnable and clear of the session's D
    ordering dependency. This gives the exact recurrence:

      ready-to-D = inherited previous-D wait
                 + current pre-D time after the ordering barrier
                 + current D service.

    The first formal unit of each session is excluded because its preceding
    preconditioning D completion is intentionally outside the formal witness.
    """
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for slot in pd_slots:
        try:
            sequence = int(slot["seq"])
        except (KeyError, TypeError, ValueError):
            continue
        ready = slot.get("model_unit_ready_epoch")
        done = slot.get("done_epoch")
        d_service_ms = slot.get("d_service_ms")
        if not all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in (ready, done, d_service_ms)
        ):
            continue
        key = (str(slot["request_id"]), sequence)
        previous = latest.get(key)
        if previous is None or float(done) > float(previous["done_epoch"]):
            latest[key] = slot

    by_session: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (request_id, sequence), slot in latest.items():
        by_session[request_id].append((sequence, slot))

    rows: list[dict[str, float]] = []
    sequence_gap_count = 0
    d_submit_order_violations = 0
    for slots in by_session.values():
        slots.sort(key=lambda item: item[0])
        for (previous_sequence, previous), (sequence, current) in zip(
            slots,
            slots[1:],
            strict=False,
        ):
            if sequence != previous_sequence + 1:
                sequence_gap_count += 1
                continue
            ready = float(current["model_unit_ready_epoch"])
            done = float(current["done_epoch"])
            previous_done = float(previous["done_epoch"])
            d_service_ms = float(current["d_service_ms"])
            d_submit = done - d_service_ms / 1000.0
            barrier = max(ready, previous_done)
            fresh_pre_d_ms = (d_submit - barrier) * 1000.0
            if fresh_pre_d_ms < -0.5:
                d_submit_order_violations += 1
            fresh_pre_d_ms = max(fresh_pre_d_ms, 0.0)
            inherited_wait_ms = max((previous_done - ready) * 1000.0, 0.0)
            ready_to_d_ms = max((done - ready) * 1000.0, 0.0)
            fresh_serial_cycle_ms = max((done - barrier) * 1000.0, 0.0)
            identity_error_ms = abs(ready_to_d_ms - inherited_wait_ms - fresh_pre_d_ms - d_service_ms)
            rows.append(
                {
                    "ready_to_d_ms": ready_to_d_ms,
                    "inherited_previous_d_wait_ms": inherited_wait_ms,
                    "current_pre_d_after_barrier_ms": fresh_pre_d_ms,
                    "current_d_service_ms": d_service_ms,
                    "fresh_serial_cycle_ms": fresh_serial_cycle_ms,
                    "identity_error_ms": identity_error_ms,
                }
            )

    ready_to_d = [row["ready_to_d_ms"] for row in rows]
    inherited = [row["inherited_previous_d_wait_ms"] for row in rows]
    fresh_pre_d = [row["current_pre_d_after_barrier_ms"] for row in rows]
    d_service = [row["current_d_service_ms"] for row in rows]
    fresh_cycle = [row["fresh_serial_cycle_ms"] for row in rows]
    top_count = max(1, math.ceil(len(rows) * 0.01)) if rows else 0
    top_rows = sorted(rows, key=lambda row: row["ready_to_d_ms"], reverse=True)[:top_count]
    top_ready = [row["ready_to_d_ms"] for row in top_rows]
    top_inherited = [row["inherited_previous_d_wait_ms"] for row in top_rows]
    top_fresh_pre_d = [row["current_pre_d_after_barrier_ms"] for row in top_rows]
    top_d_service = [row["current_d_service_ms"] for row in top_rows]
    top_mean = statistics.fmean(top_ready) if top_ready else 0.0

    def _mean_share(values: list[float]) -> float | None:
        if not values or top_mean <= 0.0:
            return None
        return round(statistics.fmean(values) / top_mean, 4)

    return {
        "matched_consecutive_transitions": len(rows),
        "sessions": len(by_session),
        "excluded_first_formal_units": len(by_session),
        "sequence_gap_count": sequence_gap_count,
        "d_submit_order_violations": d_submit_order_violations,
        "identity_max_abs_error_ms": (round(max(row["identity_error_ms"] for row in rows), 6) if rows else None),
        "definition": {
            "inherited_previous_d_wait_ms": ("max(previous D completion - current model-unit ready, 0)"),
            "current_pre_d_after_barrier_ms": ("inferred D submit - max(current ready, previous D completion)"),
            "fresh_serial_cycle_ms": ("current D completion - max(current ready, previous D completion)"),
            "identity": ("ready_to_d = inherited_previous_d_wait + current_pre_d_after_barrier + current_d_service"),
        },
        "previous_d_incomplete_at_ready": sum(value > 0.0 for value in inherited),
        "previous_d_wait_over_1000ms": sum(value > MODEL_UNIT_MS for value in inherited),
        "all_transitions": {
            "ready_to_d_ms": _latency_summary(ready_to_d),
            "inherited_previous_d_wait_ms": _latency_summary(inherited),
            "current_pre_d_after_barrier_ms": _latency_summary(fresh_pre_d),
            "current_d_service_ms": _latency_summary(d_service),
            "fresh_serial_cycle_ms": _latency_summary(fresh_cycle),
            "ready_to_d_correlation": {
                "inherited_previous_d_wait": _pearson_correlation(ready_to_d, inherited),
                "current_pre_d_after_barrier": _pearson_correlation(ready_to_d, fresh_pre_d),
                "current_d_service": _pearson_correlation(ready_to_d, d_service),
            },
        },
        "top_1pct_ready_to_d": {
            "count": len(top_rows),
            "ready_to_d_ms": _latency_summary(top_ready),
            "inherited_previous_d_wait_ms": _latency_summary(top_inherited),
            "current_pre_d_after_barrier_ms": _latency_summary(top_fresh_pre_d),
            "current_d_service_ms": _latency_summary(top_d_service),
            "mean_fraction": {
                "inherited_previous_d_wait": _mean_share(top_inherited),
                "current_pre_d_after_barrier": _mean_share(top_fresh_pre_d),
                "current_d_service": _mean_share(top_d_service),
            },
        },
    }


def _pd_context_cycle_summary(
    completions: list[dict[str, Any]],
    pd_slots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure steady context cycles between consecutive D prompt resets.

    MiniCPM P/D creates a finite D request for every model unit.  Its reported
    prompt length therefore exposes context rollover directly: the active
    prompt drops from tens of thousands of tokens to the retained unit.  A
    rollover-to-rollover interval compares the same context phase and avoids
    making a long-run result depend on where the finite benchmark stops.
    """
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # A clean formal run deliberately disables per-request cadence logs. The
    # fixed-size physical-D witness still exposes prompt length and exact real
    # input identity, which is sufficient to observe rollover without
    # perturbing the hot path.
    for slot in pd_slots or []:
        prompt_tokens = slot.get("prompt_tokens")
        input_unit_index = slot.get("input_unit_index")
        if not isinstance(prompt_tokens, int) or not isinstance(input_unit_index, int):
            continue
        by_session[str(slot["request_id"])].append(
            {
                "generation": input_unit_index,
                "tokens_in": prompt_tokens,
                "done_epoch": float(slot["done_epoch"]),
            }
        )
    if not by_session:
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
            "vision_encoder_ms": _latency_summary([float(batch["vision_encoder_ms"]) for batch in selected]),
            "total_ms": _latency_summary([float(batch["total_ms"]) for batch in selected]),
        }
    return {
        "batches": len(batches),
        "jobs": sum(int(batch["jobs"]) for batch in batches),
        "encoded_frames": sum(int(batch["encoded_frames"]) for batch in batches),
        "cpu_prepare_ms": _latency_summary([float(batch["cpu_prepare_ms"]) for batch in batches]),
        "vision_encoder_ms": _latency_summary([float(batch["vision_encoder_ms"]) for batch in batches]),
        "cache_ms": _latency_summary([float(batch["cache_ms"]) for batch in batches]),
        "batch_total_ms": _latency_summary([float(batch["total_ms"]) for batch in batches]),
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
                request_id, generation = _cadence_identity(stage, admit.group(2), int(admit.group(3)))
                admits[(stage, request_id, generation)] = timestamp
            continue
        schedule = SCHEDULE.search(line)
        if schedule is not None:
            timestamp = float(schedule.group(5))
            if started <= timestamp <= ended:
                stage = int(schedule.group(1))
                request_id, generation = _cadence_identity(stage, schedule.group(2), int(schedule.group(3)))
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
                request_id, generation = _cadence_identity(stage, runner_done.group(2), int(runner_done.group(3)))
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
            mixed_decode_steps = [step for step in decode_steps if step["batch_tokens"] > step["batch_reqs"]]
            mixed_decode_step_runner_ms = [
                (step["runner_done_epoch"] - step["schedule_epoch"]) * 1000.0 for step in mixed_decode_steps
            ]
            completion["mixed_decode_steps"] = len(mixed_decode_steps)
            completion["mixed_decode_step_rate"] = len(mixed_decode_steps) / len(decode_steps)
            completion["mixed_decode_step_runner_ms"] = mixed_decode_step_runner_ms
            completion["mixed_decode_runner_ms_sum"] = sum(mixed_decode_step_runner_ms)
            decode_only_steps = [step for step in decode_steps if step not in mixed_decode_steps]
            decode_only_step_runner_ms = [
                (step["runner_done_epoch"] - step["schedule_epoch"]) * 1000.0 for step in decode_only_steps
            ]
            completion["decode_only_step_runner_ms"] = decode_only_step_runner_ms
            completion["decode_only_runner_ms_sum"] = sum(decode_only_step_runner_ms)
            if mixed_decode_steps:
                completion["mixed_decode_runner_ms_per_step"] = completion["mixed_decode_runner_ms_sum"] / len(
                    mixed_decode_steps
                )
            if decode_only_steps:
                completion["decode_only_runner_ms_per_step"] = completion["decode_only_runner_ms_sum"] / len(
                    decode_only_steps
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


def _client_pd_completion_slots(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert client-visible exact D identities into the normal slot schema."""
    slots: list[dict[str, Any]] = []
    for user in run.get("users", []):
        if not isinstance(user, dict):
            continue
        session_id = user.get("session_id")
        witness = user.get("pd_completion_witness")
        records = witness.get("records") if isinstance(witness, dict) else None
        if not isinstance(session_id, str) or not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            physical_sequence = record.get("sequence")
            input_unit_index = record.get("input_unit_index", physical_sequence)
            first_media_arrival = record.get("first_media_arrival_at_s")
            if not isinstance(first_media_arrival, int | float):
                # Pre dual-origin artifacts used ``ready_at_s`` for the same
                # first-media/input-start clock.
                first_media_arrival = record.get("ready_at_s")
            model_unit_ready = record.get("model_unit_ready_at_s")
            has_dual_origin = isinstance(model_unit_ready, int | float)
            done = record.get("done_at_s")
            input_start_e2e_ms = record.get("input_start_e2e_ms")
            if not isinstance(input_start_e2e_ms, int | float):
                input_start_e2e_ms = record.get("e2e_ms")
            if not (
                isinstance(input_unit_index, int)
                and not isinstance(input_unit_index, bool)
                and isinstance(first_media_arrival, int | float)
                and isinstance(done, int | float)
            ):
                continue
            if not isinstance(input_start_e2e_ms, int | float):
                input_start_e2e_ms = round(
                    max(
                        (float(done) - float(first_media_arrival)) * 1000.0,
                        0.0,
                    ),
                    3,
                )
            ready_to_d_ms = record.get("ready_to_d_ms")
            if not isinstance(ready_to_d_ms, int | float) and isinstance(model_unit_ready, int | float):
                ready_to_d_ms = round(
                    max(
                        (float(done) - float(model_unit_ready)) * 1000.0,
                        0.0,
                    ),
                    3,
                )
            input_aggregation_ms = record.get("input_aggregation_ms")
            if not isinstance(input_aggregation_ms, int | float) and isinstance(model_unit_ready, int | float):
                input_aggregation_ms = round(
                    max(
                        (float(model_unit_ready) - float(first_media_arrival)) * 1000.0,
                        0.0,
                    ),
                    3,
                )
            slot = {
                "request_id": session_id,
                "physical_request_id": record.get("request_id"),
                # Capacity is indexed by real input identity.  Physical D
                # sequence may also include auto-continuation slots.
                "seq": str(input_unit_index),
                "input_unit_index": int(input_unit_index),
                "physical_seq": (
                    int(physical_sequence)
                    if isinstance(physical_sequence, int) and not isinstance(physical_sequence, bool)
                    else None
                ),
                # Keep ``ready_epoch`` on its historical first-media origin
                # so stream/cadence RTF does not change.
                "ready_epoch": float(first_media_arrival),
                "first_media_arrival_epoch": float(first_media_arrival),
                "done_epoch": float(done),
                "e2e_ms": float(input_start_e2e_ms),
                "input_start_e2e_ms": float(input_start_e2e_ms),
                "timing_schema": ("dual_input_origin" if has_dual_origin else "legacy_single_origin"),
                "source": "client_physical_d_completion_witness",
            }
            if isinstance(model_unit_ready, int | float):
                slot["model_unit_ready_epoch"] = float(model_unit_ready)
            if isinstance(ready_to_d_ms, int | float):
                slot["ready_to_d_ms"] = float(ready_to_d_ms)
            if isinstance(input_aggregation_ms, int | float):
                slot["input_aggregation_ms"] = float(input_aggregation_ms)
            for name in (
                "prompt_tokens",
                "cached_tokens",
                "local_cached_tokens",
                "external_cached_tokens",
                "computed_tokens",
                "uncached_suffix_tokens",
                "d_service_ms",
                "kv_transfer_selected_blocks",
                "kv_transfer_selected_tokens",
                "kv_transfer_selected_bytes",
                "kv_transfer_write_submit_to_d_ready_ms",
            ):
                value = record.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    slot[name] = value
            slots.append(slot)
    return slots


def _physical_d_kv_transfer_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the connector's request-scoped delta evidence.

    The semantic token count is exact. Blocks and bytes describe the padded
    physical WRITE, so they need not be derivable from the token count. The
    connector currently cannot expose the cross-process write-submit clock
    without extra synchronization; exactly ``-1`` is therefore a valid
    unavailable value and is never replaced with another interval here.
    """
    selected_blocks: list[float] = []
    selected_tokens: list[float] = []
    selected_bytes: list[float] = []
    remote_replay_tokens: list[float] = []
    write_submit_to_d_ready_ms: list[float] = []
    unavailable_write_timing = 0
    full_hits = 0
    non_full_hits = 0
    mismatches: list[dict[str, Any]] = []

    for record in records:
        external = record.get("external_cached_tokens")
        blocks = record.get("kv_transfer_selected_blocks", -1)
        tokens = record.get("kv_transfer_selected_tokens", -1)
        byte_count = record.get("kv_transfer_selected_bytes", -1)
        write_ms = record.get("kv_transfer_write_submit_to_d_ready_ms", -1.0)
        reasons: list[str] = []

        external_valid = bool(isinstance(external, int) and not isinstance(external, bool) and external >= 0)
        blocks_valid = bool(isinstance(blocks, int) and not isinstance(blocks, bool) and blocks >= 0)
        tokens_valid = bool(isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0)
        bytes_valid = bool(isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count >= 0)
        write_valid = bool(
            isinstance(write_ms, int | float)
            and not isinstance(write_ms, bool)
            and math.isfinite(float(write_ms))
            and (float(write_ms) >= 0.0 or float(write_ms) == -1.0)
        )

        replay_tokens: int | None = None
        if external_valid and tokens_valid:
            replay_tokens = int(tokens) - int(external)

        if not external_valid:
            reasons.append("external_cached_tokens_unavailable")
        if not tokens_valid:
            reasons.append("selected_tokens_unavailable_or_negative")
        if not blocks_valid:
            reasons.append("selected_blocks_unavailable_or_negative")
        if not bytes_valid:
            reasons.append("selected_bytes_unavailable_or_negative")
        if not write_valid:
            reasons.append("invalid_write_submit_to_d_ready_ms")

        if tokens_valid:
            if tokens == 0:
                full_hits += 1
                if not external_valid or external != 0:
                    reasons.append("full_hit_external_cached_tokens_nonzero")
                if blocks_valid and blocks != 0:
                    reasons.append("full_hit_selected_nonzero_blocks")
                if bytes_valid and byte_count != 0:
                    reasons.append("full_hit_selected_nonzero_bytes")
            else:
                non_full_hits += 1
                # Direct cache-sync imports P's complete prefix. D then
                # deliberately replays the final imported token before its
                # first decode step, so PrefillStats counts one fewer token as
                # external cache than NIXL physically selected and wrote.
                if replay_tokens != 1:
                    reasons.append("selected_tokens_not_external_plus_one_replay")
                if blocks_valid and blocks == 0:
                    reasons.append("non_full_hit_missing_selected_blocks")
                if bytes_valid and byte_count == 0:
                    reasons.append("non_full_hit_missing_selected_bytes")

        if blocks_valid:
            selected_blocks.append(float(blocks))
        if tokens_valid:
            selected_tokens.append(float(tokens))
        if bytes_valid:
            selected_bytes.append(float(byte_count))
        if replay_tokens is not None and replay_tokens >= 0:
            remote_replay_tokens.append(float(replay_tokens))
        if write_valid and float(write_ms) >= 0.0:
            write_submit_to_d_ready_ms.append(float(write_ms))
        elif write_valid:
            unavailable_write_timing += 1

        if reasons:
            mismatches.append(
                {
                    "session_id": str(record.get("request_id", "")),
                    "sequence": record.get("sequence"),
                    "external_cached_tokens": external,
                    "selected_blocks": blocks,
                    "selected_tokens": tokens,
                    "remote_replay_tokens": replay_tokens,
                    "selected_bytes": byte_count,
                    "write_submit_to_d_ready_ms": write_ms,
                    "reasons": reasons,
                }
            )

    return {
        "definition": {
            "selected_tokens": (
                "exact semantic P-to-D delta tokens, including the final remote token that D deliberately replays"
            ),
            "remote_replay_tokens": (
                "selected_tokens - external_cached_tokens; 1 for a delta import and 0 for a full D-local hit"
            ),
            "selected_blocks": "block-granular WRITE selection including tail padding",
            "selected_bytes": "WRITE bytes aggregated across D tensor-parallel ranks",
            "write_submit_to_d_ready_ms": ("connector write submission to D-ready; -1 means unavailable"),
        },
        "records": len(records),
        "valid_records": len(records) - len(mismatches),
        "full_hit_records": full_hits,
        "non_full_hit_records": non_full_hits,
        "selected_blocks": _latency_summary(selected_blocks),
        "selected_tokens": _latency_summary(selected_tokens),
        "remote_replay_tokens": _latency_summary(remote_replay_tokens),
        "selected_bytes": _latency_summary(selected_bytes),
        "write_submit_to_d_ready_ms": _latency_summary(write_submit_to_d_ready_ms),
        "write_timing_unavailable_records": unavailable_write_timing,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "valid": bool(records) and not mismatches,
    }


def _client_physical_measurement_keys(
    slots: list[dict[str, Any]],
) -> set[tuple[str, int]]:
    """Return exact ``(session, physical D sequence)`` diagnostic join keys."""
    keys: set[tuple[str, int]] = set()
    for slot in slots:
        session_id = slot.get("request_id")
        physical_sequence = slot.get("physical_seq")
        if isinstance(session_id, str) and isinstance(physical_sequence, int):
            keys.add((session_id, physical_sequence))
    return keys


def _physical_record_is_measured(
    request_id: str,
    physical_sequence: int,
    keys: set[tuple[str, int]],
) -> bool:
    session_id = _request_session_id(request_id)
    if session_id is None:
        session_id = request_id
    return (session_id, physical_sequence) in keys


def _client_pd_measurement_is_complete(
    run: dict[str, Any],
    expected_input_units: int | None,
) -> bool:
    witness = run.get("pd_completion_witness")
    return bool(
        isinstance(witness, dict)
        and expected_input_units is not None
        and witness.get("source")
        in {
            "client-visible engine stage_metrics",
            "client-visible physical D completion witness",
        }
        and witness.get("complete") is True
        and witness.get("expected") == expected_input_units
        and witness.get("completed") == expected_input_units
        and int(run.get("failed_users", 0)) == 0
    )


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


def _decode_prefix_identity(request_id: str) -> tuple[str, int] | None:
    match = PD_DECODE_REQUEST.fullmatch(request_id)
    if match is None:
        return None
    session_id = _request_session_id(match.group("logical"))
    if session_id is None:
        return None
    return session_id, int(match.group("slot"), 16)


def _formal_capacity_workload_validity(run: dict[str, Any]) -> dict[str, Any]:
    """Classify whether the client workload can support a capacity claim.

    Short and synchronized runs remain useful development measurements. They
    must not, however, silently acquire the same ``benchmark_valid`` status as
    the long, randomized production workload.
    """
    config = run.get("config")
    config = config if isinstance(config, dict) else {}
    users = [user for user in run.get("users", []) if isinstance(user, dict)]
    duration_s = config.get("duration_s")
    phase_window_s = config.get("phase_window_s")
    phases = [user.get("phase_s") for user in users]
    numeric_phases = [
        float(phase) for phase in phases if isinstance(phase, int | float) and not isinstance(phase, bool)
    ]
    valid_phase_bounds = bool(
        isinstance(phase_window_s, int | float)
        and not isinstance(phase_window_s, bool)
        and phase_window_s > 0
        and len(numeric_phases) == len(users)
        and users
        # The client archives rounded phases, so the upper endpoint may appear
        # exactly equal to the configured half-open sampling window.
        and all(0.0 <= phase <= float(phase_window_s) for phase in numeric_phases)
    )
    # One session has no cross-session synchronization to detect. For a
    # multi-user capacity run, require the sampled phases to be observably
    # dispersed in the archived artifact rather than trusting the profile name.
    phases_dispersed = bool(
        valid_phase_bounds and (len(users) == 1 or len({round(phase, 6) for phase in numeric_phases}) > 1)
    )
    checks = {
        "production_workload_profile": config.get("workload_profile") == "production",
        "duration_at_least_180s": bool(
            isinstance(duration_s, int | float) and not isinstance(duration_s, bool) and duration_s >= 180
        ),
        "force_listen_disabled": config.get("force_listen_count") is None,
        "randomized_session_phases": phases_dispersed,
        "server_trace_diagnostics_not_requested": (config.get("server_trace_frame_audit_requested") is not True),
    }
    violations = [name for name, passed in checks.items() if not passed]
    diagnostic_control = bool(
        config.get("workload_profile") == "synchronized"
        or config.get("force_listen_count") is not None
        or config.get("server_trace_frame_audit_requested") is True
    )
    return {
        "valid": not violations,
        "classification": (
            "formal_capacity"
            if not violations
            else "diagnostic_control"
            if diagnostic_control
            else "development_screening"
        ),
        "checks": checks,
        "violations": violations,
        "requirements": {
            "minimum_duration_s": 180,
            "workload_profile": "production",
            "phase_policy": "randomized and dispersed within a non-zero phase window",
            "force_listen_count": None,
            "server_trace_diagnostics": False,
        },
        "note": (
            "Development/screening and diagnostic-control runs remain fully "
            "analyzable, but cannot produce a valid formal capacity result."
        ),
    }


def _benchmark_cleanliness(
    log_path: Path,
    run: dict[str, Any],
    measurement_ranges: dict[str, tuple[int, int]],
    expected_input_units: int | None,
    measurement_complete: bool,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail-closed validity checks that run entirely after measurement.

    The server log must be fresh for one server lifetime.  Warning messages do
    not carry a machine-readable epoch, so silently reusing a previous log
    would make truncation/fallback checks ambiguous.
    """
    provenance_present = isinstance(provenance, dict)
    run_id = provenance.get("run_id") if provenance_present else None
    all_lines = log_path.read_text(errors="replace").splitlines()
    marker_indices = [
        index
        for index, line in enumerate(all_lines)
        if (match := PROVENANCE_MARKER.search(line)) is not None and match.group(1) == run_id
    ]
    log_matches_provenance = bool(isinstance(run_id, str) and run_id and len(marker_indices) == 1)
    if log_matches_provenance:
        marker_start = marker_indices[0]
        marker_end = next(
            (
                index
                for index in range(marker_start + 1, len(all_lines))
                if PROVENANCE_MARKER.search(all_lines[index]) is not None
            ),
            len(all_lines),
        )
        lines = all_lines[marker_start:marker_end]
    else:
        lines = all_lines
    diagnostic_counts = {name: 0 for name in DIAGNOSTIC_MARKERS}
    failure_counts = {name: 0 for name in RUNTIME_FAILURE_MARKERS}
    truncations: list[dict[str, int]] = []
    logged_prefix_records = 0
    logged_prefix_mismatches: list[dict[str, int | str]] = []
    preemption_or_eviction = 0
    for line in lines:
        for name, marker in DIAGNOSTIC_MARKERS.items():
            diagnostic_counts[name] += marker.search(line) is not None
        for name, marker in RUNTIME_FAILURE_MARKERS.items():
            failure_counts[name] += marker.search(line) is not None
        preemption_or_eviction += PREEMPTION_OR_EVICTION.search(line) is not None
        if match := TRUNCATED_APPEND.search(line):
            truncations.append({"produced": int(match.group(1)), "reserved": int(match.group(2))})
        match = PREFIX_CACHE.search(line)
        if match is None:
            continue
        identity = _decode_prefix_identity(match.group(1))
        if identity is None:
            continue
        session_id, sequence = identity
        bounds = measurement_ranges.get(session_id)
        if bounds is None or not bounds[0] <= sequence <= bounds[1]:
            continue
        logged_prefix_records += 1
        hit_tokens = int(match.group(2))
        prompt_tokens = int(match.group(3))
        if prompt_tokens - hit_tokens != 2:
            logged_prefix_mismatches.append(
                {
                    "session_id": session_id,
                    "sequence": sequence,
                    "hit_tokens": hit_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            )

    provenance_diagnostic_state = provenance.get("diagnostics", {}) if provenance_present else {}
    provenance_diagnostics = provenance_diagnostic_state.get("enabled", [])
    provenance_cli_diagnostics = provenance_diagnostic_state.get("cli_enabled", [])
    formal_provenance = bool(provenance.get("formal")) if provenance_present else False
    import_matches_checkout = False
    if provenance_present:
        repo_value = provenance.get("git", {}).get("repo")
        omni_origin = provenance.get("imports", {}).get("vllm_omni")
        if isinstance(repo_value, str) and isinstance(omni_origin, str):
            try:
                Path(omni_origin).resolve().relative_to(Path(repo_value).resolve())
                import_matches_checkout = True
            except ValueError:
                pass
    diagnostics_off = bool(
        provenance_present
        and formal_provenance
        and not provenance_diagnostics
        and not provenance_cli_diagnostics
        and provenance_diagnostic_state.get("all_disabled") is True
        and not any(diagnostic_counts.values())
    )
    server_git = provenance.get("git", {}) if provenance_present else {}
    server_head = server_git.get("head") if isinstance(server_git, dict) else None
    server_source_clean = bool(isinstance(server_git, dict) and server_git.get("dirty") is False)
    server_commit_captured = bool(isinstance(server_head, str) and server_head)
    deploy_config = provenance.get("deploy_config", {}) if provenance_present else {}
    deploy_path = deploy_config.get("resolved_path") if isinstance(deploy_config, dict) else None
    deploy_sha256 = deploy_config.get("sha256") if isinstance(deploy_config, dict) else None
    deploy_config_captured = bool(
        isinstance(deploy_config, dict)
        and deploy_config.get("exists") is True
        and isinstance(deploy_path, str)
        and Path(deploy_path).is_absolute()
        and isinstance(deploy_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", deploy_sha256)
    )
    client_provenance = run.get("client_provenance")
    client_provenance_present = isinstance(client_provenance, dict)
    client_git = client_provenance.get("git", {}) if client_provenance_present else {}
    client_head = client_git.get("head") if isinstance(client_git, dict) else None
    client_commit_captured = bool(isinstance(client_head, str) and client_head)
    client_source_clean = bool(isinstance(client_git, dict) and client_git.get("dirty") is False)
    client_diagnostic_state = client_provenance.get("diagnostics", {}) if client_provenance_present else {}
    client_diagnostics_off = bool(
        isinstance(client_diagnostic_state, dict)
        and client_diagnostic_state.get("all_disabled") is True
        and not client_diagnostic_state.get("enabled", [])
        and not client_diagnostic_state.get("cli_enabled", [])
    )
    client_server_commit_match = bool(server_commit_captured and client_commit_captured and client_head == server_head)
    client_prefix_records = 0
    client_prefix_mismatches: list[dict[str, int | str]] = []
    client_transfer_records: list[dict[str, Any]] = []
    for user in run.get("users", []):
        witness = user.get("pd_completion_witness") if isinstance(user, dict) else None
        records = witness.get("records") if isinstance(witness, dict) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            client_transfer_records.append(record)
            prompt_tokens = record.get("prompt_tokens")
            cached_tokens = record.get("cached_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(cached_tokens, int):
                continue
            client_prefix_records += 1
            local_cached_tokens = record.get("local_cached_tokens")
            external_cached_tokens = record.get("external_cached_tokens")
            computed_tokens = record.get("computed_tokens")
            has_exact_split = all(
                name in record
                for name in (
                    "local_cached_tokens",
                    "external_cached_tokens",
                    "computed_tokens",
                )
            )
            exact_values = (
                prompt_tokens,
                cached_tokens,
                local_cached_tokens,
                external_cached_tokens,
                computed_tokens,
            )
            exact_split_valid = bool(
                has_exact_split
                and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in exact_values)
                and isinstance(local_cached_tokens, int)
                and isinstance(external_cached_tokens, int)
                and isinstance(computed_tokens, int)
                and local_cached_tokens + external_cached_tokens == cached_tokens
                and computed_tokens + cached_tokens == prompt_tokens
                # D always replays the final imported P token. Most slots
                # also append P's sampled token, while valid EOS/control
                # boundaries can omit that append. Both leave a bounded
                # one- or two-token suffix and preserve exact accounting.
                and computed_tokens in (1, 2)
            )
            legacy_split_valid = bool(
                not has_exact_split and prompt_tokens >= 0 and cached_tokens >= 0 and prompt_tokens - cached_tokens == 2
            )
            if not (exact_split_valid or legacy_split_valid):
                client_prefix_mismatches.append(
                    {
                        "session_id": str(user.get("session_id", "")),
                        "sequence": int(record.get("sequence") or 0),
                        "input_unit_index": int(record.get("input_unit_index") or 0),
                        "hit_tokens": cached_tokens,
                        "prompt_tokens": prompt_tokens,
                        "local_cached_tokens": (
                            local_cached_tokens
                            if isinstance(local_cached_tokens, int) and not isinstance(local_cached_tokens, bool)
                            else -1
                        ),
                        "external_cached_tokens": (
                            external_cached_tokens
                            if isinstance(external_cached_tokens, int) and not isinstance(external_cached_tokens, bool)
                            else -1
                        ),
                        "computed_tokens": (
                            computed_tokens
                            if isinstance(computed_tokens, int) and not isinstance(computed_tokens, bool)
                            else -1
                        ),
                    }
                )
    if client_prefix_records:
        prefix_source = "client_physical_completion_witness"
        prefix_records = client_prefix_records
        prefix_mismatches = client_prefix_mismatches
    else:
        prefix_source = "instrumented_server_log"
        prefix_records = logged_prefix_records
        prefix_mismatches = logged_prefix_mismatches
    expected_prefix_records = expected_input_units
    prefix_complete = bool(
        expected_prefix_records is not None and prefix_records == expected_prefix_records and not prefix_mismatches
    )
    transfer_evidence = _physical_d_kv_transfer_summary(client_transfer_records)
    transfer_evidence_complete = bool(
        expected_input_units is not None
        and transfer_evidence["records"] == expected_input_units
        and transfer_evidence["valid"] is True
    )
    input_stream_complete = run.get("input_stream_complete") is True
    client_frame_audit = _client_frame_audit(run)
    client_audio_sidecar_audit = _client_audio_sidecar_audit(run)
    expected_frames = run.get("frames_sent")
    client_frame_audit_complete = bool(
        client_frame_audit is not None
        and isinstance(expected_frames, int)
        and expected_input_units is not None
        and client_frame_audit["units"] == expected_input_units
        and client_frame_audit["frames_consumed"] == expected_frames
        and client_frame_audit["by_source"].get("arrival") == expected_frames
        and client_frame_audit["by_source"].get("formal_fallback") == 0
    )
    audio_sidecar_audit_required = bool(
        client_audio_sidecar_audit is not None
        or run.get("config", {}).get("audio_sidecar_audit_required") is True
    )
    client_audio_sidecar_complete = bool(
        client_audio_sidecar_audit is not None
        and expected_input_units is not None
        and client_audio_sidecar_audit["audited_records"] == expected_input_units
        and client_audio_sidecar_audit["arrival_audio_units"] == expected_input_units
        and client_audio_sidecar_audit["audio_fallback_units"] == 0
        and client_audio_sidecar_audit["malformed_records"] == 0
    )
    checks = {
        "provenance_captured": provenance_present,
        "server_log_matches_provenance": log_matches_provenance,
        "vllm_omni_import_matches_checkout": import_matches_checkout,
        "server_git_commit_captured": server_commit_captured,
        "server_source_tree_clean": server_source_clean,
        "deploy_config_path_and_sha256_captured": deploy_config_captured,
        "client_provenance_captured": client_provenance_present,
        "client_git_commit_captured": client_commit_captured,
        "client_source_tree_clean": client_source_clean,
        "client_server_git_commit_match": client_server_commit_match,
        "client_diagnostics_off": client_diagnostics_off,
        "diagnostics_off": diagnostics_off,
        "input_stream_complete": input_stream_complete,
        "terminal_measurement_complete": measurement_complete,
        "zero_token_truncation": not truncations,
        "decode_prefix_cache_complete": prefix_complete,
        "physical_d_kv_transfer_evidence_complete": transfer_evidence_complete,
        "zero_runtime_fallback": not any(failure_counts.values())
        and (client_frame_audit is None or client_frame_audit["by_source"].get("formal_fallback") == 0)
        and (
            client_audio_sidecar_audit is None
            or client_audio_sidecar_audit["audio_fallback_units"] == 0
        ),
        "client_frame_consumption_complete": (client_frame_audit_complete if client_frame_audit is not None else True),
        "client_audio_sidecar_complete": (
            client_audio_sidecar_complete if audio_sidecar_audit_required else True
        ),
        "zero_observed_preemption_or_eviction": preemption_or_eviction == 0,
    }
    violations = [name for name, passed in checks.items() if not passed]
    return {
        "valid": not violations,
        "checks": checks,
        "violations": violations,
        "note": (
            "The log must contain exactly one fresh server lifetime. Formal runs "
            "disable detailed diagnostics; a separate instrumented rerun is needed "
            "for scheduler/runner decomposition."
        ),
        "provenance": provenance,
        "diagnostic_marker_counts": diagnostic_counts,
        "truncation_count": len(truncations),
        "truncation_examples": truncations[:5],
        "decode_prefix_cache": {
            "source": prefix_source,
            "records": prefix_records,
            "expected_records": expected_prefix_records,
            "expected_uncached_suffix_tokens": [1, 2],
            "expected_locally_computed_tokens": [1, 2],
            "mismatches": len(prefix_mismatches),
            "mismatch_examples": prefix_mismatches[:5],
            "instrumented_log_records": logged_prefix_records,
        },
        "physical_d_kv_transfer": transfer_evidence,
        "runtime_failure_counts": failure_counts,
        "client_frame_audit": client_frame_audit,
        "client_audio_sidecar_audit": (
            {
                **client_audio_sidecar_audit,
                "expected_units": expected_input_units,
                "required": audio_sidecar_audit_required,
                "complete": client_audio_sidecar_complete,
            }
            if client_audio_sidecar_audit is not None
            else None
        ),
        "preemption_or_eviction_count": preemption_or_eviction,
    }


def _periodic_pd_records(
    completions: list[dict[str, Any]],
    pd_slots: list[dict[str, Any]],
    input_units_per_session: int | None,
    measurement_ranges: dict[str, tuple[int, int]] | None = None,
    physical_measurement_keys: set[tuple[str, int]] | None = None,
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
        request_id = str(completion.get("request_id", ""))
        in_range = bool(
            isinstance(generation, int)
            and (
                (
                    physical_measurement_keys
                    and _physical_record_is_measured(
                        request_id,
                        generation,
                        physical_measurement_keys,
                    )
                )
                or (
                    not physical_measurement_keys
                    and measurement_ranges
                    and _sequence_in_measurement(
                        request_id,
                        generation,
                        measurement_ranges,
                    )
                )
            )
        )
        if isinstance(generation, int) and (
            in_range or (not measurement_ranges and 1 <= generation <= input_units_per_session)
        ):
            periodic_completions.append(completion)

    periodic_slots: list[dict[str, Any]] = []
    for slot in pd_slots:
        try:
            sequence = int(slot["seq"])
        except (TypeError, ValueError):
            continue
        request_id = str(slot.get("request_id", ""))
        in_range = bool(
            (
                physical_measurement_keys
                and _physical_record_is_measured(
                    request_id,
                    sequence,
                    physical_measurement_keys,
                )
            )
            or (
                not physical_measurement_keys
                and measurement_ranges
                and _sequence_in_measurement(
                    request_id,
                    sequence,
                    measurement_ranges,
                )
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


def _client_frame_audit(run: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate fixed-size stage-0 input counters carried by D witnesses."""
    top_witness = run.get("pd_completion_witness")
    if not isinstance(top_witness, dict) or top_witness.get("source") != (
        "client-visible physical D completion witness"
    ):
        return None
    units = 0
    frames_consumed = 0
    arrival_frames = 0
    fallback_frames = 0
    for user in run.get("users", []):
        witness = user.get("pd_completion_witness") if isinstance(user, dict) else None
        records = witness.get("records") if isinstance(witness, dict) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            input_frames = int(record.get("input_video_frames") or 0)
            frames_consumed += input_frames
            arrival_frames += int(record.get("arrival_video_frames") or 0)
            fallback_frames += int(record.get("vision_fallback_frames") or 0)
            units += input_frames > 0
    return {
        "source": "client_physical_completion_witness",
        "units": units,
        "frames_consumed": frames_consumed,
        "by_source": {
            "arrival": arrival_frames,
            "formal_fallback": fallback_frames,
        },
    }


def _client_audio_sidecar_audit(run: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate fail-closed audio sidecar counters from physical D witnesses."""
    top_witness = run.get("pd_completion_witness")
    if not isinstance(top_witness, dict) or top_witness.get("source") != (
        "client-visible physical D completion witness"
    ):
        return None
    records_seen = 0
    audited_records = 0
    malformed_records = 0
    arrival_audio_units = 0
    audio_fallback_units = 0
    for user in run.get("users", []):
        witness = user.get("pd_completion_witness") if isinstance(user, dict) else None
        records = witness.get("records") if isinstance(witness, dict) else None
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            records_seen += 1
            arrival = record.get("arrival_audio_units")
            fallback = record.get("audio_fallback_units")
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in (arrival, fallback)
            ):
                malformed_records += 1
                continue
            audited_records += 1
            arrival_audio_units += int(arrival)
            audio_fallback_units += int(fallback)
    return {
        "source": "client_physical_completion_witness",
        "records": records_seen,
        "audited_records": audited_records,
        "malformed_records": malformed_records,
        "arrival_audio_units": arrival_audio_units,
        "audio_fallback_units": audio_fallback_units,
    }


def _stage_summary(
    stage: int,
    completions: list[dict[str, Any]],
    *,
    periodic: bool,
) -> dict[str, Any]:
    records = [item for item in completions if item["stage"] == stage]
    service_ms = [item["service_ms"] for item in records]
    unit_rtf = [MODEL_UNIT_MS / value for value in service_ms if value > 0] if periodic else []
    audio_rtf = [item["audio_s"] * 1000.0 / item["service_ms"] for item in records if item["audio_s"] > 0]
    result: dict[str, Any] = {
        "requests": len(records),
        "service_ms": _latency_summary(service_ms),
        "accumulated_unit_rtf": round(len(service_ms) * MODEL_UNIT_MS / sum(service_ms), 4)
        if periodic and service_ms and sum(service_ms) > 0
        else None,
        "unit_rtf": _rtf_summary(unit_rtf),
        "unit_rtf_lt_1": _rtf_miss_count(unit_rtf),
        "unit_rtf_lt_1_rate": round(_rtf_miss_count(unit_rtf) / len(unit_rtf), 4) if unit_rtf else None,
        "ttft_ms": _latency_summary([item["ttft_ms"] for item in records if item["ttft_ms"] > 0]),
        "dispatch_to_core_ms": _latency_summary([item["dispatch_ms"] for item in records if "dispatch_ms" in item]),
        "core_admit_to_done_ms": _latency_summary(
            [item["core_to_done_ms"] for item in records if "core_to_done_ms" in item]
        ),
        "scheduler_queue_ms": _latency_summary(
            [item["scheduler_queue_ms"] for item in records if "scheduler_queue_ms" in item]
        ),
        "scheduled_to_done_ms": _latency_summary(
            [item["scheduled_to_done_ms"] for item in records if "scheduled_to_done_ms" in item]
        ),
        "runner_ms_sum": _latency_summary([item["runner_ms_sum"] for item in records if "runner_ms_sum" in item]),
        "first_runner_ms": _latency_summary([item["first_runner_ms"] for item in records if "first_runner_ms" in item]),
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
            [item["mixed_decode_runner_ms_per_step"] for item in records if "mixed_decode_runner_ms_per_step" in item]
        ),
        "decode_only_runner_ms_per_step": _latency_summary(
            [item["decode_only_runner_ms_per_step"] for item in records if "decode_only_runner_ms_per_step" in item]
        ),
        "mixed_decode_step_runner_ms": _latency_summary(
            [value for item in records for value in item.get("mixed_decode_step_runner_ms", [])]
        ),
        "decode_only_step_runner_ms": _latency_summary(
            [value for item in records for value in item.get("decode_only_step_runner_ms", [])]
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
    parser.add_argument(
        "--server-provenance-json",
        help="provenance emitted by clean_server.py; required for a valid formal result",
    )
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = json.loads(Path(args.run_json).read_text())
    server_provenance = (
        json.loads(Path(args.server_provenance_json).read_text()) if args.server_provenance_json else None
    )
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
    client_pd_slots = _client_pd_completion_slots(run)
    physical_measurement_keys = _client_physical_measurement_keys(client_pd_slots)
    completions, server_pd_slots = _periodic_pd_records(
        completions,
        pd_slots,
        input_units_per_session,
        measurement_ranges,
        physical_measurement_keys,
    )
    pd_slot_source = "server_cadence"
    # Capacity and completeness always use the fixed-size client witness. A
    # diagnostic cadence trace may contain autonomous physical sequences and
    # is used only for joined stage decomposition below.
    if client_pd_slots:
        pd_slots = client_pd_slots
        pd_slot_source = "client_physical_d_completion_witness"
    else:
        pd_slots = server_pd_slots
    stage_ids = sorted({item["stage"] for item in completions})
    is_pd = bool(pd_slots) or bool(client_pd_slots) or 3 in stage_ids
    periodic_stage_ids = (0, 1, 2) if is_pd else (0, 1)
    stages = {
        str(stage): _stage_summary(
            stage,
            completions,
            periodic=stage in periodic_stage_ids,
        )
        for stage in stage_ids
    }
    periodic_stages = {str(stage): stages[str(stage)] for stage in periodic_stage_ids if str(stage) in stages}
    periodic_stage_p05 = {
        stage: (summary.get("unit_rtf", {}).get("p05") or 0.0) for stage, summary in periodic_stages.items()
    }
    rtf_pass = bool(periodic_stages) and all(
        summary.get("unit_rtf_lt_1", 0) == 0 for summary in periodic_stages.values()
    )
    p95_pass = bool(periodic_stage_p05) and all(value >= 1.0 for value in periodic_stage_p05.values())
    # ``e2e_ms`` remains the public compatibility field, with an explicit
    # first-media/input-start origin.  ``ready_to_d_ms`` removes the deliberate
    # five-chunk input aggregation interval and isolates runnable-unit latency.
    pd_slot_e2e = [float(slot["e2e_ms"]) for slot in pd_slots]
    pd_slot_ready_to_d = [
        float(slot["ready_to_d_ms"]) for slot in pd_slots if isinstance(slot.get("ready_to_d_ms"), int | float)
    ]
    pd_slot_input_aggregation = [
        float(slot["input_aggregation_ms"])
        for slot in pd_slots
        if isinstance(slot.get("input_aggregation_ms"), int | float)
    ]
    pd_slot_d_service = [
        float(slot["d_service_ms"]) for slot in pd_slots if isinstance(slot.get("d_service_ms"), int | float)
    ]
    pd_slot_pre_d_upper_bound = [
        max(float(slot["ready_to_d_ms"]) - float(slot["d_service_ms"]), 0.0)
        for slot in pd_slots
        if isinstance(slot.get("ready_to_d_ms"), int | float) and isinstance(slot.get("d_service_ms"), int | float)
    ]
    pd_slot_summary = {
        "count": len(pd_slot_e2e),
        "dual_origin_count": sum(slot.get("timing_schema") == "dual_input_origin" for slot in pd_slots),
        "legacy_single_origin_count": sum(slot.get("timing_schema") == "legacy_single_origin" for slot in pd_slots),
        "timing_definition": {
            "e2e_ms": "first media arrival for the 1 s unit to D completion",
            "ready_to_d_ms": "fifth 200 ms audio chunk sent to D completion",
            "input_aggregation_ms": "first media arrival to fifth audio chunk sent",
            "d_service_ms": "physical D request submit to D completion",
            "pre_d_upper_bound_ms": (
                "ready-to-D minus D service; upper bound for P queue/compute, KV handoff, and D ingress before submit"
            ),
        },
        "e2e_ms": _latency_summary(pd_slot_e2e),
        "input_start_e2e_ms": _latency_summary(pd_slot_e2e),
        "ready_to_d_ms": _latency_summary(pd_slot_ready_to_d),
        "input_aggregation_ms": _latency_summary(pd_slot_input_aggregation),
        "d_service_ms": _latency_summary(pd_slot_d_service),
        "pre_d_upper_bound_ms": _latency_summary(pd_slot_pre_d_upper_bound),
        "e2e_over_1000ms": sum(value >= MODEL_UNIT_MS for value in pd_slot_e2e),
        "ready_to_d_over_1000ms": sum(value >= MODEL_UNIT_MS for value in pd_slot_ready_to_d),
        "wait_previous_d_ms": _latency_summary(
            [
                float(slot["wait_previous_d_ms"])
                for slot in pd_slots
                if isinstance(slot.get("wait_previous_d_ms"), int | float)
            ]
        ),
        "physical_kv_transfer": _physical_d_kv_transfer_summary(pd_slots),
    }
    expected_input_units = _expected_input_units(run, input_units_per_session)
    required_input_stages = (0, 1) if is_pd else (0,)
    server_cadence_completed_input_units = {
        str(stage): sum(item["stage"] == stage for item in completions) for stage in required_input_stages
    }
    cadence_measurement_complete = _measurement_is_complete(
        run,
        expected_input_units,
        server_cadence_completed_input_units,
        is_pd=is_pd,
    )
    client_measurement_complete = _client_pd_measurement_is_complete(
        run,
        expected_input_units,
    )
    completed_input_units = dict(server_cadence_completed_input_units)
    completed_input_units_source = "server_cadence"
    if client_measurement_complete:
        client_completed = int(run["pd_completion_witness"]["completed"])
        # A physical D completion causally proves that its matching P unit also
        # completed. This keeps the canonical count truthful when formal
        # cadence logging is intentionally disabled.
        completed_input_units = {str(stage): client_completed for stage in required_input_stages}
        completed_input_units_source = "client_physical_d_completion_witness"
    measurement_complete = cadence_measurement_complete or client_measurement_complete
    client_frame_audit = _client_frame_audit(run)
    client_audio_sidecar_audit = _client_audio_sidecar_audit(run)
    frame_audit = client_frame_audit or _parse_frame_audit(
        server_log,
        started,
        ended,
        measurement_ranges,
    )
    expected_frames = run.get("frames_sent")
    frame_audit_required = client_frame_audit is not None or run.get("config", {}).get("frame_audit_required") is True
    frame_audit_complete = bool(
        isinstance(expected_frames, int)
        and expected_frames >= 0
        and frame_audit["frames_consumed"] == expected_frames
        and frame_audit["units"] == expected_input_units
        and (client_frame_audit is None or frame_audit["by_source"].get("arrival") == expected_frames)
        and frame_audit["by_source"].get("formal_fallback", 0) == 0
    )
    if frame_audit_required:
        measurement_complete = measurement_complete and frame_audit_complete
    audio_sidecar_audit_required = bool(
        client_audio_sidecar_audit is not None
        or run.get("config", {}).get("audio_sidecar_audit_required") is True
    )
    audio_sidecar_audit_complete = bool(
        client_audio_sidecar_audit is not None
        and expected_input_units is not None
        and client_audio_sidecar_audit["audited_records"] == expected_input_units
        and client_audio_sidecar_audit["arrival_audio_units"] == expected_input_units
        and client_audio_sidecar_audit["audio_fallback_units"] == 0
        and client_audio_sidecar_audit["malformed_records"] == 0
    )
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
        and float(pd_long_horizon.get("per_session_stream_rtf", {}).get("min") or 0.0) >= 1.0
        and int(pd_long_horizon.get("complete_sessions", 0)) == expected_sessions
    )
    primary_capacity_pass = (
        long_horizon_rtf_pass and measurement_complete if is_pd else rtf_pass and measurement_complete
    )
    cleanliness = _benchmark_cleanliness(
        server_log,
        run,
        measurement_ranges,
        expected_input_units,
        measurement_complete,
        server_provenance,
    )
    workload_validity = _formal_capacity_workload_validity(run)
    benchmark_valid = bool(cleanliness["valid"] and workload_validity["valid"])
    benchmark_invalid_reasons = [
        *(f"reproducibility_or_runtime:{name}" for name in cleanliness["violations"]),
        *(f"workload:{name}" for name in workload_validity["violations"]),
    ]
    result = {
        "run_json": str(Path(args.run_json).resolve()),
        "definition": "RTF = 1 second model unit / stage service time; RTF >= 1 is real-time",
        "capacity_slo": (
            "the complete long-session input set finishes without errors and "
            "every session's stream RTF is >= 1 from first media arrival through "
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
        "strict_unit_rtf_capacity_pass": rtf_pass and measurement_complete and benchmark_valid,
        "long_horizon_rtf_capacity_pass": (long_horizon_rtf_pass and measurement_complete and benchmark_valid),
        "capacity_pass": primary_capacity_pass and benchmark_valid,
        "p95_capacity_pass": p95_pass and measurement_complete and benchmark_valid,
        "measurement_complete": measurement_complete,
        "measurement_completion_witness": (
            "client_physical_d_completion_witness"
            if client_measurement_complete
            else "server_cadence"
            if cadence_measurement_complete
            else None
        ),
        "benchmark_valid": benchmark_valid,
        "benchmark_invalid_reasons": benchmark_invalid_reasons,
        "formal_capacity_validity": workload_validity,
        "benchmark_cleanliness": cleanliness,
        "expected_input_units": expected_input_units,
        "completed_input_units": completed_input_units,
        "completed_input_units_source": completed_input_units_source,
        "server_cadence_completed_input_units": (server_cadence_completed_input_units),
        "completion_witness_stage": 1 if is_pd else 0,
        "failed_users": int(run.get("failed_users", 0)),
        "frame_audit": {
            **frame_audit,
            "expected_frames": expected_frames,
            "required": frame_audit_required,
            "complete": frame_audit_complete,
        },
        "audio_sidecar_audit": {
            **(client_audio_sidecar_audit or {}),
            "expected_units": expected_input_units,
            "required": audio_sidecar_audit_required,
            "complete": audio_sidecar_audit_complete,
        },
        "limiting_periodic_stage": min(periodic_stage_p05, key=periodic_stage_p05.get) if periodic_stage_p05 else None,
        "stages": stages,
    }
    if is_pd:
        result["pd_slot"] = pd_slot_summary
        result["pd_long_horizon"] = pd_long_horizon
        result["pd_session_recurrence"] = _pd_session_recurrence_summary(pd_slots)
        result["pd_chain"] = _pd_chain_summary(
            completions,
            server_pd_slots or pd_slots,
        )
        result["pd_context_cycles"] = _pd_context_cycle_summary(
            completions,
            pd_slots,
        )
        result["vision_arrival"] = vision_timing
        result["measurement_scope"] = {
            "input_units_per_session": input_units_per_session,
            "per_session_sequence_ranges": {
                session_id: {"start": bounds[0], "end": bounds[1]}
                for session_id, bounds in sorted(measurement_ranges.items())
            },
            "periodic_stage_requests": {
                str(stage): sum(item["stage"] == stage for item in completions) for stage in (0, 1)
            },
            "excluded_stage_requests": {
                str(stage): sum(item["stage"] == stage for item in all_completions)
                - sum(item["stage"] == stage for item in completions)
                for stage in (0, 1)
            },
            "periodic_pd_slots": len(pd_slots),
            "pd_slot_source": pd_slot_source,
            "diagnostic_server_pd_slots": len(server_pd_slots),
            "excluded_server_pd_slots": max(
                len(all_pd_slots) - len(server_pd_slots),
                0,
            ),
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
