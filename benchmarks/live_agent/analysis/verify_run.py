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
    "VLLM_OMNI_INLINE_RECV": (re.compile(r"stage=1 .*irecv=(\d+)/"), "inline receive"),
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

    for flag, (pattern, label) in TRACES.items():
        value = DEFAULTS.get(flag, "0")
        if value in ("0", "", "false", "False"):
            continue
        hits = pattern.findall(log)
        if not hits:
            print(f"   !! {label}: enabled by default but no log evidence")
            bad += 1
        elif flag == "VLLM_OMNI_INLINE_RECV":
            count = max(int(hit) for hit in hits)
            print(f"   {'ok' if count else '!!'} {label}: {count}")
            bad += int(count == 0)
        else:
            print(f"   ok {label}: {len(hits)} log entries")

    stages = set(re.findall(r"StageEngineCoreProc_stage(\d+)_replica0 pid=\d+", log))
    print(f"   stage processes: {sorted(stages)}")
    if stages != {"0", "1", "2"}:
        print("   !! expected separate thinker, talker, and code2wav processes")
        bad += 1

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
