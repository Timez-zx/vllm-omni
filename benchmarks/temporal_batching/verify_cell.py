#!/usr/bin/env python3
"""Make the two failure modes of this study impossible to hit silently.

    verify_cell.py DIR              -- did every declared treatment actually engage?
    verify_cell.py DIR_A DIR_B      -- do these two cells differ ONLY in scheduling?

Both checks exist because both errors happened, repeatedly, and neither is
visible in the client metrics:

1. A treatment that never ran. The first inline-receive cell measured
   irecv=0/0 -- the code sat in a branch this deployment does not take -- and
   reproduced the baseline exactly, which reads as "the idea does not work".
   Fix: every knob that claims an effect must leave a countable trace, and this
   script asserts the trace is non-zero.

2. Two arms differing in more than the thing under test. Five separate times:
   the audio arrival prefill was on for one arm only; three "replicates" were
   three configurations; a guard made two fixes mutually exclusive; the windowed
   vocoder was on for one arm; inline receive was silently disabled on the other.
   Each was worth about 2x on the headline number and one of them reversed the
   conclusion. Fix: diff the recorded environments and name anything that
   differs outside the scheduling set.

Requires meta.json (written by run_cell.sh) and engine_slice.log.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Knobs that ARE the scheduling policy under test: these may legitimately differ
# between the arms being compared. Everything else must match, or the comparison
# is measuring something other than what it claims.
SCHEDULING = {
    "VLLM_OMNI_TEMPORAL_TICK_MS",
    "VLLM_OMNI_TEMPORAL_BARRIER",
    "VLLM_OMNI_TEMPORAL_ENGINE",
    "VLLM_OMNI_TEMPORAL_REPLAY",
    "VLLM_OMNI_TEMPORAL_SLACK_TOKENS",
    "VLLM_OMNI_TEMPORAL_VOCODE_PHASES",
    "VLLM_OMNI_TEMPORAL_PHASE0_MS",
    "VLLM_OMNI_TEMPORAL_PHASE1_MS",
    "VLLM_OMNI_TEMPORAL_PHASE2_MS",
    "VLLM_OMNI_TEMPORAL_TIME_SLACK",
    "VLLM_OMNI_TEMPORAL_DECODE_ZONE_MS",
    "VLLM_OMNI_TEMPORAL_GATE_EXEMPT",
    "VLLM_OMNI_TEMPORAL_LEAD_MS",
    "VLLM_OMNI_TEMPORAL_CATCHUP",
    "VLLM_OMNI_TEMPORAL_FRAME_TICK",     # the frame mailbox: clock alignment
    "VLLM_OMNI_FRAME_FLUSH_MS",          # is the periodic idea at the input edge
    "VLLM_OMNI_CELL_ARM",
}

# treatment -> (env predicate, log pattern that proves it ran, human name)
TRACES = [
    ("VLLM_OMNI_INLINE_RECV", lambda v: v not in ("0", ""),
     re.compile(r"stage=1 .*irecv=(\d+)/"), "内联取件命中数"),
    ("VLLM_OMNI_TEMPORAL_MAILBOX", lambda v: v not in ("0", ""),
     re.compile(r"tick_mailbox_connector|TickMailboxConnector"), "常驻信箱在用"),
    ("VLLM_OMNI_TEXT_COALESCE_TICKS", lambda v: v not in ("0", ""),
     re.compile(r"\[TEXT-CHUNK\]"), "文本合批(需人工核对件数)"),
]


# "unset" is NOT "off" in this codebase: unset knobs resolve against
# _LIVE_DEFAULTS (inline receive, windowed vocoder, inline send and the mailbox
# all default ON). Comparing raw env strings therefore reports differences that
# do not exist -- the same trap as the boot banner, which prints the raw env and
# shows 0 for knobs that are actually on. Resolve before diffing.
def _defaults() -> dict:
    # Parsed from source rather than imported: importing vllm_omni pulls in torch,
    # and a口径 checker must run anywhere the results are, including a laptop.
    import ast
    src = Path("/home/ubuntu/data/vllm-omni/vllm_omni/core/sched/temporal_pacing.py").read_text()
    lit = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_LIVE_DEFAULTS" for t in node.targets):
            lit = ast.literal_eval(node.value)
            break
    d = dict(lit)
    d.setdefault("VLLM_OMNI_INLINE_RECV_ASYNC", "0")
    d.setdefault("VLLM_OMNI_T2T_LEAN_DECODE", "1")
    d.setdefault("VLLM_OMNI_FRAME_FLUSH_MS", "")
    return d


DEFAULTS = _defaults()


def env_of(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    env = {}
    for blob in (meta.get("arm_env", ""), meta.get("inherited_env", ""), meta.get("env", "")):
        for tok in (blob or "").split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                env[k] = v
    return meta, env


def check_engagement(d: Path) -> int:
    meta, env = env_of(d)
    log = (d / "engine_slice.log")
    text = log.read_text(errors="ignore") if log.exists() else ""
    print(f"== {d.name}: arm={meta.get('arm')} users={meta.get('users')} deploy={meta.get('deploy')}")
    bad = 0
    for key, pred, pat, name in TRACES:
        # EFFECTIVE value, not the raw env: an arm that inherits a default is
        # just as much "declaring the treatment" as one that sets it. Reading
        # the raw env here is precisely the mistake this tool exists to catch --
        # it let a cell whose inline receive was silently skipped 31468 times
        # pass the check on its first version.
        v = env.get(key, DEFAULTS.get(key))
        if v is None or not pred(v):
            continue
        hits = pat.findall(text)
        if not hits:
            print(f"   !! {key}={v} 声明开启,但日志里没有任何证据({name})")
            bad += 1
            continue
        if key == "VLLM_OMNI_INLINE_RECV":
            n = max(int(x) for x in hits)
            flag = "!!" if n == 0 else "ok"
            print(f"   {flag} {name} = {n}" + ("   ← 声明开启但一次都没命中" if n == 0 else ""))
            bad += 1 if n == 0 else 0
        else:
            print(f"   ok {name}({len(hits)} 处证据)")
    # the vocoder split is a deployment fact, not an env: read it off the log
    pids = set(re.findall(r"StageEngineCoreProc_stage(\d)_replica0 pid=\d+", text))
    print(f"   引擎进程数={len(pids)}(3 = 声码器独立进程,2 = 与 talker 共进程)")
    return bad


def check_pair(a: Path, b: Path) -> int:
    (ma, ea), (mb, eb) = env_of(a), env_of(b)
    keys = (set(ea) | set(eb) | set(DEFAULTS)) - SCHEDULING
    def eff(env, k):
        return env.get(k, DEFAULTS.get(k, "<未设>"))
    diffs = [(k, eff(ea, k), eff(eb, k)) for k in sorted(keys)
             if eff(ea, k) != eff(eb, k)]
    print(f"\n== 对比 {a.name} vs {b.name}")
    if ma.get("deploy") != mb.get("deploy"):
        print(f"   !! 部署文件不同: {ma.get('deploy')} vs {mb.get('deploy')}")
    if ma.get("users") != mb.get("users") or ma.get("turns") != mb.get("turns"):
        print(f"   !! 负载不同: users {ma.get('users')}/{mb.get('users')} "
              f"turns {ma.get('turns')}/{mb.get('turns')}")
    if ma.get("session_cfg") != mb.get("session_cfg"):
        print("   !! 会话配置不同(压缩触发线/音频到达预填充等):")
        print(f"      A: {ma.get('session_cfg')}")
        print(f"      B: {mb.get('session_cfg')}")
    if not diffs:
        print("   ok 除调度类开关外,两臂环境一致")
        return 0
    print("   !! 以下非调度项在两臂间不同 —— 这个对比测的不只是调度策略:")
    for k, va, vb in diffs:
        print(f"      {k}: {va}  vs  {vb}")
    return len(diffs)


if __name__ == "__main__":
    args = [Path(x if x.startswith("/") else f"/home/ubuntu/data/results/{x}") for x in sys.argv[1:]]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    rc = sum(check_engagement(d) for d in args)
    if len(args) == 2:
        rc += check_pair(*args)
    print("\n" + ("通过:未发现口径问题" if rc == 0 else f"发现 {rc} 处问题 —— 该对比或该格的结论需要重做"))
    raise SystemExit(1 if rc else 0)
