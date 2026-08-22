#!/usr/bin/env python3
"""Verify that a continuous-AV result used the mechanisms it claims.

Usage:
    verify_run.py RESULT_DIR [RESULT_DIR ...]
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FLAGS_PY = REPO_ROOT / "vllm_omni/core/sched/runtime_flags.py"
TRACES = {
    "VLLM_OMNI_MAILBOX": (re.compile(r"shm_mailbox_connector|ShmMailboxConnector"), "SHM mailbox"),
}


def defaults() -> dict[str, str]:
    source = FLAGS_PY.read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "_DEFAULTS" for target in node.targets):
            return dict(ast.literal_eval(node.value))
    return {}


DEFAULTS = defaults()


def check(result_dir: Path) -> int:
    summary_path = result_dir / "summary.json"
    log_path = result_dir / "engine.log"
    if not summary_path.is_file() or not log_path.is_file():
        print(f"!! {result_dir}: expected summary.json and engine.log")
        return 1

    summary = json.loads(summary_path.read_text())
    log = log_path.read_text(errors="ignore")
    deploy = Path(summary.get("deploy_config") or "")
    print(
        f"== {result_dir.name}: users={summary.get('users')} turns={summary.get('turns_per_user')} deploy={deploy.name}"
    )

    bad = 0
    if deploy.name != "origin_deploy_3gpu.yaml":
        print(f"   !! unexpected deploy: {deploy}")
        bad += 1

    if summary.get("workload_schema") != 4:
        print(
            "   !! workload schema is not 4; this result does not use the "
            "fixed audio-ready-500 measurement contract"
        )
        bad += 1
    if summary.get("audio_ready_threshold_ms") != 500.0:
        print(
            "   !! audio-ready threshold is not the canonical 500 ms: "
            f"{summary.get('audio_ready_threshold_ms')}"
        )
        bad += 1

    for flag, (pattern, label) in TRACES.items():
        value = DEFAULTS.get(flag, "0")
        if value in ("0", "", "false", "False"):
            continue
        hits = pattern.findall(log)
        if not hits:
            print(f"   !! {label}: enabled by default but no log evidence")
            bad += 1
        else:
            print(f"   ok {label}: {len(hits)} log entries")

    stages = set(re.findall(r"StageEngineCoreProc_stage(\d+)_replica0 pid=\d+", log))
    print(f"   stage processes: {sorted(stages)}")
    if stages != {"0", "1", "2"}:
        print("   !! expected separate thinker, talker, and code2wav processes")
        bad += 1

    finite_ids = re.findall(r"\[finite-request\].* request=(video-[0-9a-f]+)", log)
    expected_requests = (
        int(summary.get("users") or 0)
        * int(summary.get("turns_per_user") or 0)
        * int(summary.get("repeat_sessions") or 1)
    )
    if len(finite_ids) != expected_requests or len(set(finite_ids)) != len(finite_ids):
        print(
            f"   !! finite requests: lines={len(finite_ids)} unique={len(set(finite_ids))} "
            f"expected={expected_requests}"
        )
        bad += 1
    else:
        print(f"   ok finite requests: {len(finite_ids)} turns, all request ids unique")

    warm_ids = re.findall(r"\[arrival-prefill\].* request=(video-warm-[0-9a-f]+)", log)
    warm_failures = len(re.findall(r"\[arrival-prefill\] failed", log))
    if not warm_ids:
        print("   !! no video arrival-prefill request was observed")
        bad += 1
    elif warm_failures:
        print(f"   !! arrival-prefill failures: {warm_failures}")
        bad += 1
    elif len(set(warm_ids)) != len(warm_ids):
        print(f"   !! arrival-prefill request ids are not unique: {len(warm_ids)} lines")
        bad += 1
    else:
        print(f"   ok arrival-prefill requests: {len(warm_ids)}, all finite and unique")

    final_frame_counts = [
        int(value)
        for value in re.findall(r"\[finite-request\].* frames=(\d+)", log)
    ]
    client_consumed = int(summary.get("frames_consumed") or 0)
    if len(final_frame_counts) != expected_requests or sum(final_frame_counts) != client_consumed:
        print(
            f"   !! final media ledger: engine_frames={sum(final_frame_counts)} "
            f"client_consumed={client_consumed} request_rows={len(final_frame_counts)}/{expected_requests}"
        )
        bad += 1
    else:
        print(f"   ok final media ledger: {client_consumed} accepted frame occurrences consumed")

    try:
        deploy_text = deploy.read_text()
        thinker = deploy_text.split("- stage_id: 1", 1)[0]
        prefix_enabled = bool(re.search(r"enable_prefix_caching:\s*true", thinker))
    except OSError:
        prefix_enabled = False
    if not prefix_enabled:
        print("   !! thinker prefix caching is not enabled")
        bad += 1
    else:
        print("   ok thinker prefix caching enabled")

    prefix_hits = [
        int(value)
        for value in re.findall(r"\[prefix-cache\].*hit_tokens=(\d+)", log)
    ]
    if expected_requests > int(summary.get("users") or 0) and not any(prefix_hits):
        print("   !! no non-zero prefix-cache hit was observed after first turns")
        bad += 1
    elif prefix_hits:
        nonzero = sum(value > 0 for value in prefix_hits)
        print(
            f"   ok prefix-cache observations: {nonzero}/{len(prefix_hits)} hit, "
            f"max={max(prefix_hits)} tokens"
        )

    if not summary.get("capacity_pass"):
        print("   -- capacity SLO did not pass")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    root = Path(os.environ.get("RESULTS_DIR", "."))
    paths = [Path(value) if Path(value).is_absolute() else root / value for value in sys.argv[1:]]
    raise SystemExit(1 if sum(check(path) for path in paths) else 0)
