"""Small, clock-explicit capacity checks; no model or GPU imports."""

from __future__ import annotations

import math
import re


def pd_input_backlog(run):
    """A previous real unit must finish by the next complete-unit arrival.

    Input assembly is not backlog. The last unit has a nominal one-second
    deadline. Missing clocks/identities are unknown evidence, never a pass.
    """
    sessions = []
    for user in run.get("users", []):
        timings = user.get("input_unit_timings", [])
        records = user.get("pd_completion_witness", {}).get("records", [])
        by_index = {}
        invalid = []
        for record in records:
            index = record.get("input_unit_index")
            if index in by_index:
                invalid.append("duplicate_completion")
            by_index[index] = record
        expected = [item.get("input_unit_index") for item in timings]
        if not expected or len(set(expected)) != len(expected):
            invalid.append("missing_or_duplicate_input_timings")
        if set(expected) != set(by_index):
            invalid.append("incomplete_completion_set")
        if user.get("input_stream_complete") is not True:
            invalid.append("incomplete_input_stream")
        events = []
        ordered = sorted(timings, key=lambda row: row.get("input_unit_index", -1))
        for position, timing in enumerate(ordered):
            index = timing.get("input_unit_index")
            ready = timing.get("model_unit_ready_at_s")
            following = ordered[position + 1] if position + 1 < len(ordered) else None
            done = by_index.get(index, {}).get("done_at_s")
            deadline = (
                following.get("model_unit_ready_at_s")
                if following
                else (ready + 1.0 if isinstance(ready, int | float) else None)
            )
            if not all(isinstance(t, int | float) and math.isfinite(t) for t in (ready, done, deadline)):
                invalid.append("missing_or_nonfinite_clock")
                continue
            if done < ready or deadline <= ready:
                invalid.append("inconsistent_clock")
                continue
            if done > deadline:
                events.append({"input_unit_index": index, "lateness_ms": (done - deadline) * 1000})
        sessions.append(
            {
                "session_id": user.get("session_id"),
                "valid": not invalid,
                "violations": sorted(set(invalid)),
                "no_backlog": not invalid and not events,
                "late_units": len(events),
                "worst_events": sorted(events, key=lambda row: row["lateness_ms"], reverse=True)[:5],
            }
        )
    valid = bool(sessions) and all(row["valid"] for row in sessions)
    return {
        "valid": valid,
        "no_backlog": valid and all(row["no_backlog"] for row in sessions),
        "late_units": sum(row["late_units"] for row in sessions),
        "definition": "D(i) completes by input-ready(i+1); final D completes within 1 s of readiness",
        "sessions": sessions,
    }


def sender_timing_audit(run):
    """Separate configured arrival jitter from unintended load-generator lag."""
    limit = run.get("config", {}).get("max_send_drift_ms")
    valid_limit = isinstance(limit, int | float) and math.isfinite(limit) and limit > 0
    users = run.get("users", [])
    violations = []
    for user in users:
        drift = user.get("send_drift_ms", {})
        expected = user.get("units_sent", 0) * 5
        maximum = user.get("send_drift_max_unrounded_ms", drift.get("max"))
        if not expected or drift.get("count") != expected or not isinstance(maximum, int | float):
            violations.append({"session_id": user.get("session_id"), "reason": "missing_samples"})
        elif not math.isfinite(maximum) or not valid_limit or maximum > limit:
            violations.append({"session_id": user.get("session_id"), "reason": "sender_late", "max_ms": maximum})
    return {
        "valid": bool(users) and valid_limit and not violations,
        "max_allowed_ms": limit,
        "violations": violations,
        "definition": "send completion minus jittered absolute send deadline; not engine latency",
    }


def pd_input_budget_rtf(run):
    """Compare completed input seconds with one continuous wall-clock span.

    Start when the first complete model unit is ready, not at its first
    200-ms media chunk. The numerator budgets one second for EVERY unit,
    including the first. Include all waits through the final D completion;
    do not sum overlapping per-unit latencies or average per-unit RTFs.
    This is the RTF-only check; bounded backlog is a separate capacity gate.
    """
    clocks = pd_input_backlog(run)
    sessions = []
    for user, clock in zip(run.get("users", []), clocks["sessions"], strict=True):
        timings = sorted(user.get("input_unit_timings", []), key=lambda x: x["input_unit_index"])
        records = sorted(user.get("pd_completion_witness", {}).get("records", []), key=lambda x: x["input_unit_index"])
        indices = [t["input_unit_index"] for t in timings]
        contiguous = bool(indices) and indices == list(range(indices[0], indices[-1] + 1))
        valid = clock["valid"] and contiguous
        start = timings[0].get("model_unit_ready_at_s") if valid else None
        finish = max(r["done_at_s"] for r in records) if valid else None
        wall_s = finish - start if valid else None
        valid = bool(valid and wall_s > 0)
        rtf = len(records) / wall_s if valid else None
        sessions.append({
            "session_id": user.get("session_id"),
            "valid": valid,
            "first_input_unit": indices[0] if indices else None,
            "last_input_unit": indices[-1] if indices else None,
            "completed_units": len(records),
            "input_budget_s": float(len(records)),
            "start_complete_input_ready_at_s": start,
            "last_d_done_at_s": finish,
            "wall_s": wall_s,
            "rtf": rtf,
            "budget_overrun_ms": max(0.0, wall_s - len(records)) * 1000 if valid else None,
            "capacity_pass": valid and rtf > 1.0,
        })
    valid = bool(sessions) and all(row["valid"] for row in sessions)
    return {
        "definition": "completed one-second input budget / wall time from first complete-input readiness "
        "to last D completion; every session must have unrounded RTF > 1; backlog is checked separately",
        "valid": valid,
        "sessions": sessions,
        "min_rtf_unrounded": min(row["rtf"] for row in sessions) if valid else None,
        "capacity_pass": valid and all(row["capacity_pass"] for row in sessions),
    }


def pd_inherited_backlog(run, *, max_backlog_ms=500.0, first_units=None):
    """Bound max(0, previous D completion - current complete-input readiness).

    Evaluate every user, not a percentile or an average. Keep the predecessor
    before the window boundary so the first sliding unit cannot hide backlog.
    This excludes current-unit execution and the input-assembly interval.
    """
    if not isinstance(max_backlog_ms, int | float) or not math.isfinite(max_backlog_ms) or max_backlog_ms < 0:
        raise ValueError("max_backlog_ms must be finite and nonnegative")
    clocks = pd_input_backlog(run)
    sessions = []
    for user, clock in zip(run.get("users", []), clocks["sessions"], strict=True):
        timings = sorted(user.get("input_unit_timings", []), key=lambda x: x["input_unit_index"])
        records = {x["input_unit_index"]: x for x in user.get("pd_completion_witness", {}).get("records", [])}
        indices = [x["input_unit_index"] for x in timings]
        cutoff = (
            first_units.get(user.get("session_id")) if first_units is not None else (indices[0] if indices else None)
        )
        valid = bool(clock["valid"] and indices and cutoff in indices)
        valid = valid and indices == list(range(indices[0], indices[-1] + 1))
        events = []
        if valid:
            previous_done = None
            for timing in timings:
                index = timing["input_unit_index"]
                done = records[index]["done_at_s"]
                if previous_done is not None and done < previous_done:
                    valid = False
                if index >= cutoff:
                    wait_ms = (
                        max(0.0, previous_done - timing["model_unit_ready_at_s"]) * 1000
                        if previous_done is not None else 0.0
                    )
                    events.append({"input_unit_index": index, "backlog_ms": wait_ms})
                previous_done = done
        maximum = max((x["backlog_ms"] for x in events), default=None)
        exceeded = sum(x["backlog_ms"] > max_backlog_ms for x in events)
        sessions.append({
            "session_id": user.get("session_id"),
            "valid": valid and bool(events),
            "first_input_unit": cutoff,
            "evaluated_units": len(events),
            "max_backlog_ms": maximum,
            "exceeded_units": exceeded,
            "worst_events": sorted(events, key=lambda x: x["backlog_ms"], reverse=True)[:5],
            "capacity_pass": valid and bool(events) and exceeded == 0,
        })
    valid = bool(sessions) and all(x["valid"] for x in sessions)
    return {
        "definition": "max(0, D(i-1) completion - complete input(i) readiness); "
        "every evaluated unit of every session must be <= limit_ms, without rounding; "
        "current-unit execution is excluded; later recovery does not erase a violation",
        "scope": "post_window" if first_units is not None else "whole_run",
        "limit_ms": max_backlog_ms,
        "valid": valid,
        "max_backlog_ms": max((x["max_backlog_ms"] for x in sessions), default=None) if valid else None,
        "exceeded_units": sum(x["exceeded_units"] for x in sessions),
        "sessions": sessions,
        "capacity_pass": valid and all(x["capacity_pass"] for x in sessions),
    }


def pd_sliding_window_backlog(run, *, window_tokens, min_post_window_units=120, max_backlog_ms=500.0):
    """Require per-user long RTF > 1 AND bounded inherited backlog after SWA.

    Logical prompt length keeps growing; it is NOT resident KV length. Actual
    engine/window configuration and physical residency must be checked too.
    Every session must replace at least one whole window after first filling it.
    """
    full = pd_input_backlog(run)
    steady_users, post_users, coverage = [], [], []
    for user in run.get("users", []):
        timings = sorted(user.get("input_unit_timings", []), key=lambda x: x["input_unit_index"])
        records = sorted(user.get("pd_completion_witness", {}).get("records", []), key=lambda x: x["input_unit_index"])
        first = timings[0]["input_unit_index"] if timings else 0
        post_records = [
            r for r in records if isinstance(r.get("prompt_tokens"), int) and r["prompt_tokens"] >= window_tokens + 16
        ]
        start = post_records[0]["input_unit_index"] if post_records else None
        max_prompt = max((r.get("prompt_tokens", 0) for r in records), default=0)
        monotonic = all(b.get("prompt_tokens", 0) >= a.get("prompt_tokens", 0) for a, b in zip(records, records[1:]))
        coverage.append(
            {
                "session_id": user.get("session_id"),
                "first_sliding_unit": start,
                "post_window_units": len(post_records),
                "max_logical_prompt_tokens": max_prompt,
                "no_prompt_recycling": monotonic,
                "sufficient": (
                    len(post_records) >= min_post_window_units and max_prompt >= 2 * window_tokens and monotonic
                ),
            }
        )
        for cutoff, target in ((first + 1, steady_users), (start, post_users)):
            kept_timings = [t for t in timings if cutoff is not None and t["input_unit_index"] >= cutoff]
            kept_indices = {t["input_unit_index"] for t in kept_timings}
            target.append(
                dict(
                    user,
                    input_unit_timings=kept_timings,
                    pd_completion_witness={"records": [r for r in records if r["input_unit_index"] in kept_indices]},
                )
            )
    steady = pd_input_backlog({"users": steady_users})
    post = pd_input_backlog({"users": post_users})
    long_horizon = pd_input_budget_rtf({"users": post_users})
    bounded = pd_inherited_backlog(
        run, max_backlog_ms=max_backlog_ms,
        first_units={row["session_id"]: row["first_sliding_unit"] for row in coverage},
    )
    sufficient = bool(coverage) and all(row["sufficient"] for row in coverage)
    return {
        "window_tokens": window_tokens,
        "min_post_window_units": min_post_window_units,
        "coverage_sufficient": sufficient,
        "sessions": coverage,
        "whole_run": full,
        "excluding_first_unit": steady,
        "after_window_full": post,
        "long_horizon_rtf": long_horizon,
        "bounded_backlog": bounded,
        "capacity_pass": full["valid"] and sufficient and long_horizon["capacity_pass"] and bounded["capacity_pass"],
        "definition": "every session crosses and replaces a full KV window, with >=min_post_window_units "
        "after filling; post-window completed input seconds / wall time from the first post-window "
        "complete input to last D completion must be > 1 (unrounded); "
        f"every post-window inherited backlog must be <= {max_backlog_ms:g} ms, even if later recovered",
    }


def sliding_window_engine_audit(log_text, deployment, *, window_tokens, users):
    """Check the deployed attention policy plus actual P/D live block counts."""
    stages = {s["stage_id"]: s for s in deployment.get("stages", [])}
    pin = stages.get(0, {}).get("hf_overrides", {}).get("vllm_omni_pinned_prefix_tokens", 0)
    config_matches = all(
        stages.get(i, {}).get("hf_overrides", {}).get("sliding_window") == window_tokens
        and stages.get(i, {}).get("hf_overrides", {}).get("use_sliding_window") is True
        and stages.get(i, {})
        .get("hf_overrides", {})
        .get("vllm_omni_minicpmo_pd_prefill" if i == 0 else "vllm_omni_minicpmo_pd_decode")
        is True
        and stages.get(i, {}).get("hf_overrides", {}).get("vllm_omni_minicpmo_sliding_window_tokens") == window_tokens
        and stages.get(i, {}).get("hf_overrides", {}).get("vllm_omni_pinned_prefix_tokens", 0) == pin
        for i in (0, 1)
    )
    pattern = re.compile(
        r"StageEngineCoreProc_stage([01])_replica\d+.*\[kv-window\] "
        r"request=(\S+) logical_tokens=(\d+) window_tokens=(\d+) resident_blocks=\[(\d+)\]"
        r"(?: pinned_prefix_tokens=(\d+) pinned_resident_blocks=\[(\d+)\])?"
    )
    records = []
    identities = {0: set(), 1: set()}
    for match in pattern.finditer(log_text):
        stage, request, logical, window, blocks, pinned, pinned_blocks = match.groups()
        stage, logical, window, blocks = int(stage), int(logical), int(window), int(blocks)
        identity = request if stage == 0 else request.rsplit("-", 1)[0]
        identities[stage].add(identity)
        records.append(
            {
                "stage": stage,
                "request": request,
                "logical_tokens": logical,
                "window_tokens": window,
                "resident_blocks": blocks,
                "pinned_prefix_tokens": int(pinned or 0),
                "pinned_resident_blocks": int(pinned_blocks or 0),
            }
        )
    bound = math.ceil(window_tokens / 16) + 2 + pin // 16
    bounded = bool(records) and all(
        r["window_tokens"] == window_tokens and r["resident_blocks"] <= bound
        and r["pinned_prefix_tokens"] == pin and r["pinned_resident_blocks"] == pin // 16
        for r in records
    )
    return {
        "valid": config_matches and bounded and all(len(ids) == users for ids in identities.values()),
        "config_matches": config_matches,
        "bounded_residency": bounded,
        "sessions_by_stage": {str(stage): len(ids) for stage, ids in identities.items()},
        "boundary_block_allowance": 2,
        "max_resident_blocks": max((r["resident_blocks"] for r in records), default=0),
        "samples": records,
    }
