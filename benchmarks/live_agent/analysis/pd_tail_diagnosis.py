#!/usr/bin/env python3
"""Diagnose Thinker-P startup tail from a continuous-AV P/D result.

The normal stage table is enough to report request-level P latency and the
actual stage-0 prefix-cache miss. A run with these diagnostic flags also
enables runner-batch attribution:

  VLLM_OMNI_LOG_SCHED_DIAG=1
  VLLM_OMNI_LOG_RUNNER_DIAG=1
  VLLM_OMNI_LOG_HANDOFF_DIAG=1

This tool deliberately keeps two token counts separate:

* stage-0 cache miss = prompt_tokens - stage-0 [prefix-cache] hit_tokens
* P->D transfer delta = prompt_tokens - [nixl-delta-push] prefix_tokens

The latter is block-aligned transfer work and must not be described as P
prefill compute.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
from collections import defaultdict, deque
from typing import Any

from stage_stats_v2 import derive, parse

PERCENTILES = (50, 95, 99)

PREFIX_CACHE = re.compile(
    r"\[prefix-cache\]\s+request=(?P<request>\S+)\s+"
    r"hit_tokens=(?P<hit>\d+)\s+prompt_tokens=(?P<prompt>\d+)"
)
SNAPSHOT_MODE = re.compile(
    r"\[Orchestrator\]\[PD snapshot\]\s+req=(?P<request>\S+)\s+"
    r"mode=(?P<mode>full|delta)\s+"
    r"(?:lineage=(?P<lineage>\S+)\s+parent_revision=(?P<parent_revision>\d+)\s+"
    r"(?:selected_parent_revision=(?P<selected_parent_revision>\d+)\s+)?"
    r"revision=(?P<revision>\d+)\s+)?prompt_rows=(?P<prompt>\d+)\s+"
    r"reusable_parent_rows=(?P<parent>\d+)"
)
SCHED_DIAG = re.compile(
    r"\[SCHED-DIAG\]\s+stage=0\s+mono=(?P<mono>[\d.]+)\s+"
    r"req=(?P<request>\S+)\s+queue_ms=(?P<queue>[\d.]+)\s+"
    r"scheduler_queue_ms=(?P<scheduler_queue>[\d.]+)\s+"
    r"prompt=(?P<prompt>\d+)\s+cached=(?P<cached>\d+)\s+"
    r"scheduled=(?P<scheduled>\d+)\s+waiting=(?P<waiting>\d+)\s+"
    r"running=(?P<running>\d+)"
)
RUNNER_DIAG = re.compile(
    r"\[RUNNER-DIAG\]\s+stage=0\s+mono=(?P<mono>[\d.]+)\s+"
    r"reqs=(?P<requests>\S+)\s+computed=(?P<computed>\S+)\s+"
    r"scheduled=(?P<scheduled>\S+)\s+prepare_ms=(?P<prepare>[\d.-]+)\s+"
    r"forward_wall_ms=(?P<forward_wall>[\d.-]+)\s+"
    r"forward_gpu_ms=(?P<forward_gpu>[\d.-]+)\s+"
    r"execute_post_ms=(?P<execute_post>[\d.-]+)\s+"
    r"sample_pre_snapshot_ms=(?P<sample>[\d.-]+)\s+"
    r"snapshot_ms=(?P<snapshot>[\d.-]+)\s+"
    r"output_wait_ms=(?P<output_wait>[\d.-]+)\s+"
    r"output_build_ms=(?P<output_build>[\d.-]+)\s+"
    r"total_ms=(?P<total>[\d.-]+)"
)
CORE_OUTPUT_READY = re.compile(
    r"\[HANDOFF-DIAG\]\s+event=core-output-ready\s+stage=0\s+"
    r"wall=(?P<wall>[\d.]+)\s+(?:mono=(?P<mono>[\d.]+)\s+)?reqs=(?P<requests>\S+)\s+"
    r"payload_mib=(?P<payload>[\d.]+)\s+scheduler_update_ms=(?P<update>[\d.]+)"
)
CORE_PREPROCESS = re.compile(
    r"\[INGRESS-DIAG\]\s+event=core-preprocess\s+stage=0\s+"
    r"wall=(?P<wall>[\d.]+)\s+req=(?P<request>\S+)\s+.*total_ms=(?P<total>[\d.]+)"
)
QUERY_SERIAL = re.compile(
    r"\[query-serial\]\s+session=(?P<session>\S+)\s+request=(?P<request>\S+)\s+"
    r"warmup_running=(?P<running>True|False)\s+wait_ms=(?P<wait>[\d.]+)"
)
QUERY_BREAKDOWN = re.compile(
    r"\[QUERY-BREAKDOWN\]\s+session=(?P<session>\S+)\s+request=(?P<request>\S+)\s+"
    r"render_ms=(?P<render>[\d.]+)\s+engine_to_first_text_ms=(?P<text>[\d.-]+)\s+"
    r"engine_to_first_audio_ms=(?P<audio>[\d.-]+)"
)
NIXL_DELTA_LOAD = re.compile(
    r"\[nixl-delta-load\]\s+request=(?P<request>\S+)\s+"
    r"transfer_load_ms=(?P<duration>[\d.]+)"
)
PD_CACHE_SYNC = re.compile(
    r"\[Orchestrator\]\[PD cache-sync\]\s+completed\s+req=(?P<request>\S+)\s+"
    r"lineage=(?P<lineage>\S+)\s+revision=(?P<revision>\d+)\s+d_ms=(?P<duration>[\d.]+)"
)
NIXL_PUSH_DIAG = re.compile(
    r"\[NIXL-PUSH-DIAG\]\s+request=(?P<request>\S+)\s+"
    r"source_blocks=(?P<source>\d+)\s+delta_blocks=(?P<delta>\d+)\s+"
    r"select_ms=(?P<select>[\d.]+)\s+submit_ms=(?P<submit>[\d.]+)\s+"
    r"total_ms=(?P<total>[\d.]+)"
)
NIXL_D_REG_ENQUEUED = re.compile(
    r"\[NIXL-D-TRACE\]\s+event=registration-enqueued\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_D_REG_SENT = re.compile(
    r"\[NIXL-D-TRACE\]\s+event=registration-sent\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_P_REG_RECEIVED = re.compile(
    r"\[NIXL-P-TRACE\]\s+event=registration-received\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_P_WRITE_SUBMITTED = re.compile(
    r"\[NIXL-P-TRACE\]\s+event=write-submitted\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_P_FINISHED_STAGED = re.compile(
    r"\[NIXL-P-TRACE\]\s+event=finished-blocks-staged\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_P_FINISHED_RECEIVED = re.compile(
    r"\[NIXL-P-TRACE\]\s+event=finished-metadata-received\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_D_COMPLETED = re.compile(
    r"\[NIXL-D-TRACE\]\s+event=completion-observed\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_D_COMPLETION_FORWARDED = re.compile(
    r"\[NIXL-D-TRACE\]\s+event=completion-forwarded\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
NIXL_D_COMPLETION_CORE_OBSERVED = re.compile(
    r"\[NIXL-D-TRACE\]\s+event=completion-core-observed\s+"
    r"request=(?P<request>\S+)\s+mono=(?P<mono>[\d.]+)"
)
PD_D_ACTIVATED = re.compile(
    r"\[PD-D-CONTROL\]\s+event=prepared-cache-activate\s+"
    r"request=(?P<request>\S+)\s+held_ms=(?P<held>[\d.]+)\s+"
    r"imported_tokens=(?P<tokens>\d+)"
)
PD_D_HELD = re.compile(
    r"\[PD-D-CONTROL\]\s+event=decode-held-for-import\s+"
    r"request=(?P<request>\S+)"
)


def nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


def percentile_map(values: list[float]) -> dict[str, float]:
    return {f"p{percentile}": nearest_rank(values, percentile) for percentile in PERCENTILES}


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_delta) * sum(value * value for value in right_delta))
    if denominator == 0:
        return float("nan")
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def split_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_diagnostics(log_path: pathlib.Path) -> dict[str, Any]:
    prefix_cache: dict[str, tuple[int, int]] = {}
    snapshot_modes: dict[str, dict[str, int | str]] = {}
    scheduler_admission: dict[str, dict[str, float | int]] = {}
    pending_batches: dict[tuple[str, ...], deque[dict[str, Any]]] = defaultdict(deque)
    runner_batches: list[dict[str, Any]] = []
    matched_batches: list[dict[str, Any]] = []
    query_serial: dict[str, dict[str, Any]] = {}
    query_breakdown: dict[str, dict[str, Any]] = {}
    nixl_delta_load_ms: dict[str, float] = {}
    pd_cache_sync_ms: dict[str, float] = {}
    nixl_push_diag: list[dict[str, float | int | str]] = []
    core_preprocess_wall: dict[str, float] = {}
    core_output_ready_mono: dict[str, float] = {}
    pd_control_times: dict[str, dict[str, float]] = defaultdict(dict)
    pd_activation: dict[str, dict[str, float | int]] = {}
    pd_decode_held: set[str] = set()
    stage0_wall_minus_mono: list[float] = []

    for line in log_path.open(errors="replace"):
        if "stage0_replica0" in line:
            match = PREFIX_CACHE.search(line)
            if match:
                prefix_cache[match["request"]] = (int(match["hit"]), int(match["prompt"]))

        match = SNAPSHOT_MODE.search(line)
        if match:
            snapshot = {
                "mode": match["mode"],
                "prompt_rows": int(match["prompt"]),
                "parent_rows": int(match["parent"]),
            }
            if match["lineage"] is not None:
                snapshot.update(
                    lineage=match["lineage"],
                    parent_revision=int(match["parent_revision"]),
                    revision=int(match["revision"]),
                )
                if match["selected_parent_revision"] is not None:
                    snapshot["selected_parent_revision"] = int(match["selected_parent_revision"])
            snapshot_modes[match["request"]] = snapshot

        match = SCHED_DIAG.search(line)
        if match and match["request"] not in scheduler_admission:
            scheduler_admission[match["request"]] = {
                "mono": float(match["mono"]),
                "queue_ms": float(match["queue"]),
                "scheduler_queue_ms": float(match["scheduler_queue"]),
                "prompt_tokens": int(match["prompt"]),
                "cached_tokens": int(match["cached"]),
                "scheduled_tokens": int(match["scheduled"]),
                "waiting": int(match["waiting"]),
                "running": int(match["running"]),
            }

        match = RUNNER_DIAG.search(line)
        if match:
            requests = tuple(match["requests"].split(","))
            computed = split_ints(match["computed"])
            scheduled = split_ints(match["scheduled"])
            batch = {
                "mono": float(match["mono"]),
                "requests": requests,
                "computed_tokens": computed,
                "scheduled_tokens": scheduled,
                "total_scheduled_tokens": sum(scheduled),
                "prepare_ms": float(match["prepare"]),
                "forward_wall_ms": float(match["forward_wall"]),
                "forward_gpu_ms": float(match["forward_gpu"]),
                "execute_post_ms": float(match["execute_post"]),
                "sample_pre_snapshot_ms": float(match["sample"]),
                "snapshot_ms": float(match["snapshot"]),
                "output_wait_ms": float(match["output_wait"]),
                "output_build_ms": float(match["output_build"]),
                "total_ms": float(match["total"]),
            }
            runner_batches.append(batch)
            pending_batches[requests].append(batch)

        match = CORE_OUTPUT_READY.search(line)
        if match:
            if match["mono"] is not None:
                stage0_wall_minus_mono.append(float(match["wall"]) - float(match["mono"]))
            requests = tuple(match["requests"].split(","))
            if match["mono"] is not None:
                for request in requests:
                    core_output_ready_mono[request] = float(match["mono"])
            if pending_batches[requests]:
                batch = pending_batches[requests].popleft()
                batch["payload_mib"] = float(match["payload"])
                batch["scheduler_update_ms"] = float(match["update"])
                if match["mono"] is not None:
                    batch["runner_to_core_output_ms"] = max(
                        0.0,
                        (float(match["mono"]) - float(batch["mono"])) * 1000.0 - float(batch["total_ms"]),
                    )
                matched_batches.append(batch)

        match = CORE_PREPROCESS.search(line)
        if match:
            core_preprocess_wall[match["request"]] = float(match["wall"])

        match = QUERY_SERIAL.search(line)
        if match:
            query_serial[match["request"]] = {
                "session": match["session"],
                "warmup_running": match["running"] == "True",
                "wait_ms": float(match["wait"]),
            }

        match = QUERY_BREAKDOWN.search(line)
        if match:
            query_breakdown[match["request"]] = {
                "session": match["session"],
                "render_ms": float(match["render"]),
                "engine_to_first_text_ms": float(match["text"]),
                "engine_to_first_audio_ms": float(match["audio"]),
            }

        match = NIXL_DELTA_LOAD.search(line)
        if match:
            nixl_delta_load_ms[match["request"]] = float(match["duration"])

        match = PD_CACHE_SYNC.search(line)
        if match:
            pd_cache_sync_ms[match["request"]] = float(match["duration"])

        match = NIXL_PUSH_DIAG.search(line)
        if match:
            nixl_push_diag.append(
                {
                    "request": match["request"],
                    "source_blocks": int(match["source"]),
                    "delta_blocks": int(match["delta"]),
                    "select_ms": float(match["select"]),
                    "submit_ms": float(match["submit"]),
                    "total_ms": float(match["total"]),
                }
            )

        for event, pattern in (
            ("d_registration_enqueued", NIXL_D_REG_ENQUEUED),
            ("d_registration_sent", NIXL_D_REG_SENT),
            ("p_registration_received", NIXL_P_REG_RECEIVED),
            ("p_finished_staged", NIXL_P_FINISHED_STAGED),
            ("p_finished_received", NIXL_P_FINISHED_RECEIVED),
            ("p_write_submitted", NIXL_P_WRITE_SUBMITTED),
            ("d_completion_forwarded", NIXL_D_COMPLETION_FORWARDED),
            ("d_completion_core_observed", NIXL_D_COMPLETION_CORE_OBSERVED),
            ("d_completion_observed", NIXL_D_COMPLETED),
        ):
            match = pattern.search(line)
            if match:
                pd_control_times[match["request"]][event] = float(match["mono"])

        match = PD_D_ACTIVATED.search(line)
        if match:
            pd_activation[match["request"]] = {
                "held_ms": float(match["held"]),
                "imported_tokens": int(match["tokens"]),
            }
        match = PD_D_HELD.search(line)
        if match:
            pd_decode_held.add(match["request"])

    for batch in runner_batches:
        modes = [snapshot_modes.get(request, {}).get("mode") for request in batch["requests"]]
        batch["full_snapshots"] = sum(mode == "full" for mode in modes)
        batch["delta_snapshots"] = sum(mode == "delta" for mode in modes)

    if stage0_wall_minus_mono:
        monotonic_to_wall = statistics.median(stage0_wall_minus_mono)
        for request, admission in scheduler_admission.items():
            preprocess_wall = core_preprocess_wall.get(request)
            if preprocess_wall is not None:
                admission["core_ingress_wait_ms"] = max(
                    0.0,
                    (float(admission["mono"]) + monotonic_to_wall - preprocess_wall) * 1000.0,
                )

    return {
        "prefix_cache": prefix_cache,
        "snapshot_modes": snapshot_modes,
        "scheduler_admission": scheduler_admission,
        "runner_batches": runner_batches,
        "matched_batches": matched_batches,
        "query_serial": query_serial,
        "query_breakdown": query_breakdown,
        "nixl_delta_load_ms": nixl_delta_load_ms,
        "pd_cache_sync_ms": pd_cache_sync_ms,
        "nixl_push_diag": nixl_push_diag,
        "core_output_ready_mono": core_output_ready_mono,
        "pd_control_times": dict(pd_control_times),
        "pd_activation": pd_activation,
        "pd_decode_held": sorted(pd_decode_held),
    }


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def base_request_id(request_id: str) -> str:
    return request_id.rsplit("-", 1)[0]


def select_result_rows(
    stage_rows: list[dict[str, Any]],
    turns: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the latest engine request for every client session/turn.

    A live server log may contain several benchmark runs that reuse client
    names such as ``r0u0``. The result's turns file identifies the expected
    keys, while the latest matching engine row belongs to the newest run.
    """
    if not turns:
        return stage_rows
    expected = {(session, turn - 1) for session, turn in turns}
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in stage_rows:
        session = row.get("session")
        turn = row.get("turn")
        if session is None or turn is None:
            continue
        key = (str(session), int(turn))
        if key in expected:
            latest[key] = row
    return list(latest.values())


def request_token_metric(
    rows: list[dict[str, Any]],
    per_request: dict[str, tuple[int, int]],
) -> list[float]:
    output = []
    for row in rows:
        pair = per_request.get(row["request_id"])
        if pair is not None:
            prefix, prompt = pair
            output.append(float(prompt - prefix))
    return output


def client_tail_rows(
    rows: list[dict[str, Any]],
    turns: dict[tuple[str, int], dict[str, Any]],
    query_serial: dict[str, dict[str, Any]],
    query_breakdown: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        session = row.get("session")
        turn = row.get("turn")
        if session is None or turn is None:
            continue
        client = turns.get((str(session), int(turn) + 1))
        if client is None or client.get("ttfa_ms") is None:
            continue
        request = base_request_id(str(row["request_id"]))
        serial_ms = float(query_serial.get(request, {}).get("wait_ms", 0.0))
        query_metrics = query_breakdown.get(request, {})
        render_ms = float(query_metrics.get("render_ms", 0.0))
        engine_ms = row.get("audio_ttfa_ms")
        if engine_ms is None and float(query_metrics.get("engine_to_first_audio_ms", -1.0)) >= 0:
            engine_ms = float(query_metrics["engine_to_first_audio_ms"])
        engine_ms = float(engine_ms or 0.0)
        client_ms = float(client["ttfa_ms"])
        output.append(
            {
                "request_id": row["request_id"],
                "session": session,
                "turn": int(turn),
                "client_ttfa_ms": client_ms,
                "session_serial_wait_ms": serial_ms,
                "render_ms": render_ms,
                "thinker_p_ms": row.get("prefill_ttft_ms"),
                "p_to_thinker_d_ms": row.get("thinker_add_ms"),
                "talker_ms": row.get("talker_add_ms"),
                "code2wav_ms": row.get("code2wav_add_ms"),
                "engine_audio_ttfa_ms": row.get("audio_ttfa_ms"),
                "engine_to_first_audio_ms": engine_ms,
                "stage_complete_pd": bool(row.get("is_pd")),
                "unattributed_ms": client_ms - serial_ms - render_ms - engine_ms,
            }
        )
    return sorted(output, key=lambda row: float(row["client_ttfa_ms"]), reverse=True)[:limit]


def format_triplet(percentiles: dict[str, float], *, digits: int = 0) -> str:
    values_text = []
    for percentile in PERCENTILES:
        value = percentiles[f"p{percentile}"]
        values_text.append(f"{value:.{digits}f}")
    return "/".join(values_text)


def snapshot_breakdown(
    rows: list[dict[str, Any]],
    snapshot_modes: dict[str, dict[str, int | str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in ("full", "delta"):
        selected = [row for row in rows if snapshot_modes.get(row["request_id"], {}).get("mode") == mode]
        if selected:
            output[mode] = {
                "n": len(selected),
                "p_time_to_output_ms": percentile_map(values(selected, "prefill_ttft_ms")),
                "p_scheduled_to_output_ms": percentile_map(values(selected, "prefill_execute_ms")),
                "audio_ttfa_ms": percentile_map(values(selected, "audio_ttfa_ms")),
            }
    return output


def batch_correlations(batches: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "payload_vs_output_build": pearson(
            [float(batch["payload_mib"]) for batch in batches],
            [float(batch["output_build_ms"]) for batch in batches],
        ),
        "full_snapshots_vs_payload": pearson(
            [float(batch["full_snapshots"]) for batch in batches],
            [float(batch["payload_mib"]) for batch in batches],
        ),
        "full_snapshots_vs_output_build": pearson(
            [float(batch["full_snapshots"]) for batch in batches],
            [float(batch["output_build_ms"]) for batch in batches],
        ),
        "scheduled_tokens_vs_forward_wall": pearson(
            [float(batch["total_scheduled_tokens"]) for batch in batches],
            [float(batch["forward_wall_ms"]) for batch in batches],
        ),
    }


def pd_control_breakdown(
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    request_ids = {str(row["request_id"]) for row in rows}
    control = diagnostics["pd_control_times"]
    p_ready = diagnostics["core_output_ready_mono"]
    activation = diagnostics["pd_activation"]
    decode_held = set(diagnostics.get("pd_decode_held", ()))

    def interval(start: str, end: str) -> list[float]:
        output = []
        for request_id in request_ids:
            times = control.get(request_id, {})
            start_value = p_ready.get(request_id) if start == "p_output_ready" else times.get(start)
            end_value = p_ready.get(request_id) if end == "p_output_ready" else times.get(end)
            if start_value is not None and end_value is not None:
                output.append((float(end_value) - float(start_value)) * 1000.0)
        return output

    intervals = {
        "registration_queue_ms": interval("d_registration_enqueued", "d_registration_sent"),
        "registration_notification_ms": interval("d_registration_sent", "p_registration_received"),
        "registration_lead_before_p_ready_ms": interval("d_registration_enqueued", "p_output_ready"),
        "p_ready_to_write_ms": interval("p_output_ready", "p_write_submitted"),
        "p_finished_to_worker_ms": interval(
            "p_finished_staged", "p_finished_received"
        ),
        "p_worker_to_write_ms": interval(
            "p_finished_received", "p_write_submitted"
        ),
        "write_to_d_notification_ms": interval(
            "p_write_submitted", "d_completion_forwarded"
        ),
        "d_notification_to_core_ms": interval(
            "d_completion_forwarded", "d_completion_core_observed"
        ),
        "write_to_d_completion_ms": interval("p_write_submitted", "d_completion_observed"),
        "p_ready_to_d_completion_ms": interval("p_output_ready", "d_completion_observed"),
    }
    activated = [activation[request_id] for request_id in request_ids if request_id in activation]
    if activated:
        intervals["d_ready_to_activation_hold_ms"] = [float(row["held_ms"]) for row in activated]
        intervals["pending_import_to_local_activation_ms"] = [
            float(activation[request_id]["held_ms"])
            for request_id in request_ids & decode_held
            if request_id in activation
        ]
        intervals["prepared_cache_wait_for_query_ms"] = [
            float(activation[request_id]["held_ms"])
            for request_id in request_ids - decode_held
            if request_id in activation
        ]

    output: dict[str, Any] = {
        name: {"observed": len(samples), "percentiles": percentile_map(samples)}
        for name, samples in intervals.items()
        if samples
    }

    prompt_tokens = {
        str(row["request_id"]): int(row["prompt_tokens"])
        for row in rows
        if row.get("prompt_tokens") is not None
    }
    imported_mismatches = 0
    for request_id in request_ids:
        activated_row = activation.get(request_id)
        prompt = prompt_tokens.get(request_id)
        if activated_row is not None and prompt is not None:
            imported_mismatches += int(activated_row["imported_tokens"] != prompt - 1)
    output["activation"] = {
        "observed": len(activated),
        "imported_tokens_not_prompt_minus_one": imported_mismatches,
    }
    return output


def tail_rows(
    rows: list[dict[str, Any]],
    runner_batches: list[dict[str, Any]],
    snapshot_modes: dict[str, dict[str, int | str]],
    prefix_cache: dict[str, tuple[int, int]],
    scheduler_admission: dict[str, dict[str, float | int]],
    limit: int,
) -> list[dict[str, Any]]:
    request_batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in runner_batches:
        for request in batch["requests"]:
            request_batches[request].append(batch)

    selected = sorted(
        (row for row in rows if row.get("prefill_ttft_ms") is not None),
        key=lambda row: float(row["prefill_ttft_ms"]),
        reverse=True,
    )[:limit]
    output = []
    for row in selected:
        request_id = row["request_id"]
        batches = request_batches.get(request_id, [])
        scheduled_tokens = 0
        for batch in batches:
            request_index = batch["requests"].index(request_id)
            scheduled_tokens += int(batch["scheduled_tokens"][request_index])
        hit_prompt = prefix_cache.get(request_id)
        output.append(
            {
                "request_id": request_id,
                "session": row.get("session"),
                "turn": row.get("turn"),
                "snapshot_mode": snapshot_modes.get(request_id, {}).get("mode"),
                "p_time_to_output_ms": row.get("prefill_ttft_ms"),
                "p_scheduled_to_output_ms": row.get("prefill_execute_ms"),
                "core_ingress_wait_ms": scheduler_admission.get(request_id, {}).get(
                    "core_ingress_wait_ms"
                ),
                "p_cache_miss_tokens": (hit_prompt[1] - hit_prompt[0]) if hit_prompt is not None else None,
                "runner_steps": len(batches),
                "request_scheduled_tokens": scheduled_tokens,
                "batch_forward_wall_ms": sum(float(batch["forward_wall_ms"]) for batch in batches),
                "batch_output_build_ms": sum(float(batch["output_build_ms"]) for batch in batches),
                "batch_output_exposure_ms": sum(float(batch.get("runner_to_core_output_ms", 0.0)) for batch in batches),
                "max_batch_payload_mib": max(
                    (float(batch["payload_mib"]) for batch in batches if batch.get("payload_mib") is not None),
                    default=None,
                ),
                "max_batch_full_snapshots": max(
                    (int(batch["full_snapshots"]) for batch in batches),
                    default=None,
                ),
                "max_batch_requests": max((len(batch["requests"]) for batch in batches), default=None),
            }
        )
    return output


def build_report(result_dir: pathlib.Path, top: int) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    log_path = result_dir / "engine.log"
    if not summary_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"{result_dir} must contain summary.json and engine.log")

    summary = json.loads(summary_path.read_text())
    turns_path = result_dir / "turns.jsonl"
    turns: dict[tuple[str, int], dict[str, Any]] = {}
    if turns_path.exists():
        for line in turns_path.open(errors="replace"):
            turn = json.loads(line)
            session = turn.get("session_id") or turn.get("user")
            turn_number = turn.get("turn")
            if session is not None and turn_number is not None:
                turns[(str(session), int(turn_number))] = turn
    warmup_turns = int(summary.get("warmup_turns") or 0)
    stage_rows = select_result_rows(derive(parse(log_path)), turns)
    scored_rows = [
        row
        for row in stage_rows
        if row.get("session") is not None and row.get("turn") is not None and int(row["turn"]) >= warmup_turns
    ]
    diagnostics = parse_diagnostics(log_path)

    prefix_cache = diagnostics["prefix_cache"]
    actual_miss = request_token_metric(scored_rows, prefix_cache)
    nixl_delta = [
        float(row["prompt_tokens"] - row["kv_prefix_tokens"])
        for row in scored_rows
        if row.get("prompt_tokens") is not None and row.get("kv_prefix_tokens") is not None
    ]
    scheduler_rows = [
        diagnostics["scheduler_admission"][row["request_id"]]
        for row in scored_rows
        if row["request_id"] in diagnostics["scheduler_admission"]
    ]
    core_ingress_wait = [
        float(row["core_ingress_wait_ms"])
        for row in scheduler_rows
        if row.get("core_ingress_wait_ms") is not None
    ]
    matched_batches = diagnostics["matched_batches"]
    scored_bases = {base_request_id(str(row["request_id"])) for row in scored_rows}
    serial_rows = [
        diagnostics["query_serial"][request] for request in scored_bases if request in diagnostics["query_serial"]
    ]
    render_rows = [
        diagnostics["query_breakdown"][request] for request in scored_bases if request in diagnostics["query_breakdown"]
    ]
    d_transfer_load = [
        diagnostics["nixl_delta_load_ms"][row["request_id"]]
        for row in scored_rows
        if row["request_id"] in diagnostics["nixl_delta_load_ms"]
    ]

    report: dict[str, Any] = {
        "result_dir": str(result_dir),
        "deploy_config": summary.get("deploy_config"),
        "users": summary.get("users"),
        "turns_per_user": summary.get("turns_per_user"),
        "warmup_turns": warmup_turns,
        "scored_requests": len(scored_rows),
        "stage_complete_pd_requests": sum(bool(row.get("is_pd")) for row in scored_rows),
        "summary_ttfa_ms": {
            "p50": summary.get("ttfa_p50_ms"),
            "p95": summary.get("ttfa_p95_ms"),
            "p99": summary.get("ttfa_p99_ms"),
        },
        "thinker_p": {
            "time_to_output_ms": percentile_map(values(scored_rows, "prefill_ttft_ms")),
            "scheduled_to_output_ms": percentile_map(values(scored_rows, "prefill_execute_ms")),
            "event_queue_ms": percentile_map(values(scored_rows, "prefill_queue_ms")),
            "actual_cache_miss_tokens": percentile_map(actual_miss),
            "nixl_transfer_delta_tokens": percentile_map(nixl_delta),
        },
        "engine_stages": {
            "audio_ttfa_ms": percentile_map(values(scored_rows, "audio_ttfa_ms")),
            "thinker_d_add_ms": percentile_map(values(scored_rows, "thinker_add_ms")),
            "thinker_d_queue_ms": percentile_map(values(scored_rows, "thinker_queue_ms")),
            "thinker_d_prefill_ms": percentile_map(values(scored_rows, "thinker_prefill_ms")),
            "thinker_d_transfer_load_ms": percentile_map(d_transfer_load),
            "talker_add_ms": percentile_map(values(scored_rows, "talker_add_ms")),
            "talker_queue_ms": percentile_map(values(scored_rows, "talker_queue_ms")),
            "talker_scheduled_to_output_ms": percentile_map(values(scored_rows, "talker_prefill_ms")),
            "code2wav_add_ms": percentile_map(values(scored_rows, "code2wav_add_ms")),
            "code2wav_queue_ms": percentile_map(values(scored_rows, "code2wav_queue_ms")),
            "code2wav_scheduled_to_output_ms": percentile_map(values(scored_rows, "code2wav_prefill_ms")),
            "parsed": {
                "audio_ttfa": len(values(scored_rows, "audio_ttfa_ms")),
                "thinker_d": len(values(scored_rows, "thinker_add_ms")),
                "thinker_d_transfer_load": len(d_transfer_load),
                "talker": len(values(scored_rows, "talker_add_ms")),
                "code2wav": len(values(scored_rows, "code2wav_add_ms")),
            },
        },
        "snapshot_modes": snapshot_breakdown(scored_rows, diagnostics["snapshot_modes"]),
    }

    if serial_rows or render_rows or diagnostics["pd_cache_sync_ms"]:
        waits = [float(row["wait_ms"]) for row in serial_rows]
        report["application"] = {
            "session_serial_wait_ms": percentile_map(waits),
            "session_serial_wait_observed": len(waits),
            "query_waited_for_arrival": sum(bool(row["warmup_running"]) for row in serial_rows),
            "render_ms": percentile_map([float(row["render_ms"]) for row in render_rows]),
            "render_observed": len(render_rows),
            "arrival_d_cache_sync_ms": percentile_map(list(diagnostics["pd_cache_sync_ms"].values())),
            "arrival_d_cache_sync_observed": len(diagnostics["pd_cache_sync_ms"]),
        }

    if turns:
        report["client_ttfa_tail"] = client_tail_rows(
            scored_rows,
            turns,
            diagnostics["query_serial"],
            diagnostics["query_breakdown"],
            top,
        )

    if scheduler_rows:
        report["thinker_p"]["scheduler_admission_ms"] = percentile_map(
            [float(row["scheduler_queue_ms"]) for row in scheduler_rows]
        )
        report["thinker_p"]["scheduler_admission_max_ms"] = max(
            float(row["scheduler_queue_ms"]) for row in scheduler_rows
        )
        if core_ingress_wait:
            report["thinker_p"]["core_ingress_to_scheduler_ms"] = percentile_map(core_ingress_wait)
            report["thinker_p"]["core_ingress_to_scheduler_max_ms"] = max(core_ingress_wait)

    if diagnostics["nixl_push_diag"]:
        pushes = diagnostics["nixl_push_diag"]
        report["thinker_p"]["nixl_push"] = {
            "observed": len(pushes),
            "select_ms": percentile_map(values(pushes, "select_ms")),
            "submit_ms": percentile_map(values(pushes, "submit_ms")),
            "total_ms": percentile_map(values(pushes, "total_ms")),
            "total_sum_ms": sum(values(pushes, "total_ms")),
        }

    if diagnostics["pd_activation"]:
        report["pd_control_path"] = pd_control_breakdown(scored_rows, diagnostics)

    if matched_batches:
        scored_request_ids = {str(row["request_id"]) for row in scored_rows}
        scored_batches = [
            batch for batch in matched_batches if scored_request_ids.intersection(batch["requests"])
        ]
        report["runner_batches"] = {
            "parsed": len(diagnostics["runner_batches"]),
            "matched_to_payload": len(matched_batches),
            "correlations": batch_correlations(matched_batches),
        }
        exposure = [
            float(batch["runner_to_core_output_ms"])
            for batch in matched_batches
            if batch.get("runner_to_core_output_ms") is not None
        ]
        if exposure:
            report["runner_batches"]["runner_to_core_output_ms"] = percentile_map(exposure)
        if scored_batches:
            scored_exposure = [
                float(batch["runner_to_core_output_ms"])
                for batch in scored_batches
                if batch.get("runner_to_core_output_ms") is not None
            ]
            report["runner_batches"]["scored_request_batches"] = {
                "n": len(scored_batches),
                "prepare_ms": percentile_map(values(scored_batches, "prepare_ms")),
                "forward_wall_ms": percentile_map(values(scored_batches, "forward_wall_ms")),
                "forward_gpu_ms": percentile_map(values(scored_batches, "forward_gpu_ms")),
                "output_wait_ms": percentile_map(values(scored_batches, "output_wait_ms")),
                "output_build_ms": percentile_map(values(scored_batches, "output_build_ms")),
                "runner_total_ms": percentile_map(values(scored_batches, "total_ms")),
                "payload_mib": percentile_map(values(scored_batches, "payload_mib")),
                "runner_to_core_output_ms": percentile_map(scored_exposure),
            }
        report["tail_requests"] = tail_rows(
            scored_rows,
            diagnostics["runner_batches"],
            diagnostics["snapshot_modes"],
            prefix_cache,
            diagnostics["scheduler_admission"],
            top,
        )

    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"result: {report['result_dir']}")
    print(
        f"workload: users={report['users']} turns={report['turns_per_user']} "
        f"warmup={report['warmup_turns']} scored={report['scored_requests']} "
        f"stage-complete-PD={report['stage_complete_pd_requests']}"
    )
    print(f"TTFA p50/p95/p99 ms: {format_triplet(report['summary_ttfa_ms'])}")

    application = report.get("application")
    if application is not None:
        print("\nApplication")
        print(
            "  session serial wait ms:  "
            f"{format_triplet(application['session_serial_wait_ms'])} "
            f"({application['query_waited_for_arrival']}/{application['session_serial_wait_observed']} waited)"
        )
        print(
            "  prompt render ms:         "
            f"{format_triplet(application['render_ms'])} ({application['render_observed']} observed)"
        )
        if application["arrival_d_cache_sync_observed"]:
            print(
                "  arrival D cache-sync ms: "
                f"{format_triplet(application['arrival_d_cache_sync_ms'])} "
                f"({application['arrival_d_cache_sync_observed']} observed)"
            )

    thinker_p = report["thinker_p"]
    print("\nThinker-P")
    print(f"  time-to-output ms:       {format_triplet(thinker_p['time_to_output_ms'])}")
    print(f"  scheduled->output ms:    {format_triplet(thinker_p['scheduled_to_output_ms'])}")
    print(f"  event queue ms:           {format_triplet(thinker_p['event_queue_ms'], digits=3)}")
    if "scheduler_admission_ms" in thinker_p:
        print(
            "  scheduler admission ms:  "
            f"{format_triplet(thinker_p['scheduler_admission_ms'], digits=3)} "
            f"(max {thinker_p['scheduler_admission_max_ms']:.3f})"
        )
    if "core_ingress_to_scheduler_ms" in thinker_p:
        print(
            "  core ingress->scheduler: "
            f"{format_triplet(thinker_p['core_ingress_to_scheduler_ms'])} "
            f"(max {thinker_p['core_ingress_to_scheduler_max_ms']:.3f})"
        )
    if "nixl_push" in thinker_p:
        push = thinker_p["nixl_push"]
        print(
            "  P NIXL push total ms:   "
            f"{format_triplet(push['total_ms'], digits=3)} "
            f"({push['observed']} observed, sum {push['total_sum_ms']:.1f} ms)"
        )
    print(f"  actual P miss tokens:     {format_triplet(thinker_p['actual_cache_miss_tokens'])}")
    print(f"  P->D transfer delta tok:  {format_triplet(thinker_p['nixl_transfer_delta_tokens'])}")

    engine_stages = report["engine_stages"]
    parsed = engine_stages["parsed"]
    print("\nEngine path")
    print(
        "  engine audio TTFA ms:    "
        f"{format_triplet(engine_stages['audio_ttfa_ms'])} ({parsed['audio_ttfa']} parsed)"
    )
    print(
        "  Thinker-D added ms:      "
        f"{format_triplet(engine_stages['thinker_d_add_ms'])} ({parsed['thinker_d']} parsed)"
    )
    print(f"    D event queue ms:      {format_triplet(engine_stages['thinker_d_queue_ms'])}")
    print(f"    D scheduled->output ms:{format_triplet(engine_stages['thinker_d_prefill_ms'])}")
    if parsed["thinker_d_transfer_load"]:
        print(
            "    D registration->installed: "
            f"{format_triplet(engine_stages['thinker_d_transfer_load_ms'])} "
            f"({parsed['thinker_d_transfer_load']} parsed)"
        )
    print(
        "  Talker added ms:         "
        f"{format_triplet(engine_stages['talker_add_ms'])} ({parsed['talker']} parsed)"
    )
    print(f"    Talker queue ms:       {format_triplet(engine_stages['talker_queue_ms'])}")
    print(
        "    Talker sched->output:  "
        f"{format_triplet(engine_stages['talker_scheduled_to_output_ms'])}"
    )
    print(
        "  Code2Wav added ms:       "
        f"{format_triplet(engine_stages['code2wav_add_ms'])} ({parsed['code2wav']} parsed)"
    )
    print(f"    Code2Wav queue ms:     {format_triplet(engine_stages['code2wav_queue_ms'])}")
    print(
        "    Code2Wav sched->output:"
        f"{format_triplet(engine_stages['code2wav_scheduled_to_output_ms'])}"
    )

    pd_control = report.get("pd_control_path")
    if pd_control is not None:
        print("\nEarly P/D control path")
        labels = (
            ("registration_queue_ms", "D registration queue"),
            ("registration_notification_ms", "D registration -> P"),
            ("registration_lead_before_p_ready_ms", "registration lead before P ready"),
            ("p_ready_to_write_ms", "P ready -> write submit"),
            ("p_finished_to_worker_ms", "P finished -> worker metadata"),
            ("p_worker_to_write_ms", "P worker -> write submit"),
            ("write_to_d_notification_ms", "write submit -> NIXL notification"),
            ("d_notification_to_core_ms", "NIXL notification -> D Core"),
            ("write_to_d_completion_ms", "write submit -> D ready"),
            ("p_ready_to_d_completion_ms", "P ready -> D ready"),
            ("d_ready_to_activation_hold_ms", "D ready -> query activation hold"),
            ("pending_import_to_local_activation_ms", "pending import -> local activation"),
            ("prepared_cache_wait_for_query_ms", "prepared cache waits for query"),
        )
        for key, label in labels:
            row = pd_control.get(key)
            if row is not None:
                print(
                    f"  {label:34s} "
                    f"{format_triplet(row['percentiles'])} ms ({row['observed']} observed)"
                )
        activation = pd_control["activation"]
        print(
            "  activation token invariant          "
            f"{activation['observed']} observed, "
            f"{activation['imported_tokens_not_prompt_minus_one']} mismatch"
        )

    if report["snapshot_modes"]:
        print("\nSnapshot mode")
        for mode in ("full", "delta"):
            row = report["snapshot_modes"].get(mode)
            if row is None:
                continue
            print(
                f"  {mode:5s} n={row['n']:3d}  "
                f"P time {format_triplet(row['p_time_to_output_ms'])} ms  "
                f"P execute {format_triplet(row['p_scheduled_to_output_ms'])} ms  "
                f"TTFA {format_triplet(row['audio_ttfa_ms'])} ms"
            )

    runner = report.get("runner_batches")
    if runner is not None:
        correlations = runner["correlations"]
        print(f"\nRunner batches: {runner['matched_to_payload']}/{runner['parsed']} matched to payload")
        print(f"  corr(payload MiB, output build ms):      {correlations['payload_vs_output_build']:.3f}")
        print(f"  corr(full snapshots, payload MiB):       {correlations['full_snapshots_vs_payload']:.3f}")
        print(f"  corr(full snapshots, output build ms):   {correlations['full_snapshots_vs_output_build']:.3f}")
        print(f"  corr(scheduled tokens, forward wall ms): {correlations['scheduled_tokens_vs_forward_wall']:.3f}")
        if "runner_to_core_output_ms" in runner:
            print(f"  runner -> core output ms:                {format_triplet(runner['runner_to_core_output_ms'])}")
        scored_batches = runner.get("scored_request_batches")
        if scored_batches:
            print(f"  scored-query batches:                    {scored_batches['n']}")
            print(
                "    prepare / forward-wall / forward-GPU p99: "
                f"{scored_batches['prepare_ms']['p99']:.0f}/"
                f"{scored_batches['forward_wall_ms']['p99']:.0f}/"
                f"{scored_batches['forward_gpu_ms']['p99']:.0f} ms"
            )
            print(
                "    output-wait / output-build p99:          "
                f"{scored_batches['output_wait_ms']['p99']:.0f}/"
                f"{scored_batches['output_build_ms']['p99']:.0f} ms"
            )
            print(
                "    runner total / core exposure p99:     "
                f"{scored_batches['runner_total_ms']['p99']:.0f}/"
                f"{scored_batches['runner_to_core_output_ms']['p99']:.0f} ms"
            )

    tail = report.get("tail_requests")
    if tail:
        print("\nHighest Thinker-P time-to-output requests")
        print(
            f"  {'request':34s} {'mode':>5s} {'P ms':>7s} {'ingress':>7s} {'exec':>7s} {'miss':>6s} "
            f"{'steps':>5s} {'sched':>6s} {'fwd':>7s} {'build':>7s} {'expose':>7s} {'MiB':>7s} {'full':>4s}"
        )
        for row in tail:
            request_id = str(row["request_id"])
            print(
                f"  {request_id[:34]:34s} {str(row['snapshot_mode'] or '-'):>5s} "
                f"{float(row['p_time_to_output_ms']):7.0f} "
                f"{float(row['core_ingress_wait_ms'] or 0):7.0f} "
                f"{float(row['p_scheduled_to_output_ms']):7.0f} "
                f"{int(row['p_cache_miss_tokens'] or 0):6d} "
                f"{int(row['runner_steps']):5d} "
                f"{int(row['request_scheduled_tokens']):6d} "
                f"{float(row['batch_forward_wall_ms']):7.0f} "
                f"{float(row['batch_output_build_ms']):7.0f} "
                f"{float(row['batch_output_exposure_ms']):7.0f} "
                f"{float(row['max_batch_payload_mib'] or 0):7.0f} "
                f"{int(row['max_batch_full_snapshots'] or 0):4d}"
            )

    client_tail = report.get("client_ttfa_tail")
    if client_tail:
        print("\nHighest client TTFA requests")
        print(
            f"  {'request':34s} {'TTFA':>7s} {'serial':>7s} {'render':>7s} {'engine':>7s} "
            f"{'P':>7s} {'P->D':>7s} {'Talker':>7s} {'C2W':>7s} {'other':>7s}"
        )

        def _metric(value: Any) -> str:
            return f"{float(value):7.0f}" if value is not None else f"{'-':>7s}"

        for row in client_tail:
            request_id = str(row["request_id"])
            print(
                f"  {request_id[:34]:34s} "
                f"{float(row['client_ttfa_ms'] or 0):7.0f} "
                f"{float(row['session_serial_wait_ms'] or 0):7.0f} "
                f"{float(row['render_ms'] or 0):7.0f} "
                f"{float(row['engine_to_first_audio_ms'] or 0):7.0f} "
                f"{_metric(row['thinker_p_ms'])} "
                f"{_metric(row['p_to_thinker_d_ms'])} "
                f"{_metric(row['talker_ms'])} "
                f"{_metric(row['code2wav_ms'])} "
                f"{float(row['unattributed_ms'] or 0):7.0f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=8, help="number of highest-P-latency requests to print")
    parser.add_argument("--out", type=pathlib.Path, help="optional JSON report path")
    args = parser.parse_args()

    report = build_report(args.result_dir, max(1, args.top))
    print_report(report)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
