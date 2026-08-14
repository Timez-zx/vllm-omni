#!/usr/bin/env python3
"""Did the cell run the configuration it claims to have run?

    verify_cell.py DIR [DIR ...]

This exists because of a specific, repeated failure: a treatment that never
engaged. The first inline-receive cell measured irecv=0/0 -- the code sat in a
branch this deployment does not take -- and reproduced the baseline exactly,
which reads as "the idea does not work" rather than "the idea did not run".

So every mechanism that claims an effect must leave a COUNTABLE trace in the
engine log, and this script asserts the trace is non-zero. Reading the raw env
is not enough on its own: unset does not mean off here (runtime_flags.py has ON
defaults), and a boot banner that prints the raw env shows 0 for knobs that are
actually on.

Requires meta.json (written by run_cell.sh) and engine_slice.log.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

FLAGS_PY = Path(__file__).resolve().parents[2] / "vllm_omni/core/sched/runtime_flags.py"

# flag -> (log pattern proving it ran, human name)
TRACES = {
    "VLLM_OMNI_INLINE_RECV": (re.compile(r"stage=1 .*irecv=(\d+)/"), "内联取件命中数"),
    "VLLM_OMNI_MAILBOX": (re.compile(r"shm_mailbox_connector|ShmMailboxConnector"), "常驻信箱在用"),
}


def defaults() -> dict:
    """Effective defaults, parsed (not imported) out of runtime_flags.py.

    Parsed because importing vllm_omni pulls in torch, and a configuration
    checker has to run anywhere the results are, including a laptop.
    """
    src = FLAGS_PY.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_DEFAULTS" for t in node.targets):
            return dict(ast.literal_eval(node.value))
    return {}


DEFAULTS = defaults()


def env_of(d: Path) -> tuple[dict, dict]:
    meta = json.loads((d / "meta.json").read_text())
    env: dict[str, str] = {}
    for blob in (meta.get("inherited_env", ""), meta.get("env", "")):
        for tok in (blob or "").split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                env[k] = v
    return meta, env


def check(d: Path) -> int:
    meta, env = env_of(d)
    log = d / "engine_slice.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    print(f"== {d.name}: users={meta.get('users')} turns={meta.get('turns')} "
          f"deploy={meta.get('deploy')}")
    bad = 0
    for key, (pat, name) in TRACES.items():
        # EFFECTIVE value: an arm that inherits an ON default is declaring the
        # treatment just as much as one that sets it explicitly.
        v = env.get(key, DEFAULTS.get(key))
        if v is None or v in ("0", "", "false", "False"):
            print(f"   -- {key}={v!r}:未启用,跳过")
            continue
        hits = pat.findall(text)
        if not hits:
            print(f"   !! {key}={v} 声明开启,但日志里没有任何证据({name})")
            bad += 1
            continue
        if key == "VLLM_OMNI_INLINE_RECV":
            n = max(int(x) for x in hits)
            print(f"   {'!!' if n == 0 else 'ok'} {name} = {n}"
                  + ("   ← 声明开启但一次都没命中" if n == 0 else ""))
            bad += 1 if n == 0 else 0
        else:
            print(f"   ok {name}({len(hits)} 处证据)")
    # The vocoder split is a deployment fact rather than an env value: read the
    # process count off the log. 3 = code2wav in its own process (what the
    # measurements assume), 2 = it is a thread inside the talker.
    pids = set(re.findall(r"StageEngineCoreProc_stage(\d)_replica0 pid=\d+", text))
    print(f"   引擎进程数={len(pids)}(3 = 声码器独立进程,2 = 与 talker 共进程)")
    if len(pids) and len(pids) != 3:
        bad += 1
    return bad


if __name__ == "__main__":
    args = [Path(x if x.startswith("/") else f"/home/ubuntu/data/results/{x}")
            for x in sys.argv[1:]]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    rc = sum(check(d) for d in args)
    print("\n" + ("通过:声明的机制都留下了证据"
                  if rc == 0 else f"发现 {rc} 处问题 —— 该格的结论需要重做"))
    raise SystemExit(1 if rc else 0)
