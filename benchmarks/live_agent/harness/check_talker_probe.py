#!/usr/bin/env python3
"""Bring-up gate for the talker-delta probe. Run after a short smoke session.

WHY A GATE EXISTS AT ALL
------------------------
This study has already burned hours on arms that ran to completion and were only then
found never to have entered the new code path: an arm that set every switch off and
"agreed within 1%", and four sessions pointed at a directory containing no frames. Both
looked like clean null results. So each arm here must PROVE it is the arm it claims to
be, from server-side evidence, before any measurement time is spent.

WHAT IT CHECKS, and what each failure means

  1. Requests completed at all. Zero parsed requests means the smoke session never
     reached the server -- check the stimulus path and the websocket, not the patch.

  2. Stage-1 num_tokens_in is NON-ZERO. Upstream reports this only for stage 0
     (stage_pool.py, `if self.stage_id == 0`), so a zero here means the stage_pool
     patch is not live and the experiment has no x-axis. Nothing downstream is
     interpretable; abort.

  3. Arm identity, from the ratio of stage-1 to stage-0 prompt tokens:
       - control arm (PA_TALKER_LAST_BLOCK=0): the talker sums every user block, so
         the ratio should be near 1 and must RISE with the thinker's prompt.
       - treatment arm (PA_TALKER_LAST_BLOCK=1): the talker sees only the newest
         block, so the ratio must FALL well below the control's as the prompt grows.
     If the treatment's ratio looks like the control's, the switch did not take effect
     and the arm must be discarded rather than reported as "no effect".

  4. Audio was produced on every turn, with a plausible duration. The treatment
     deliberately withholds conditioning from the talker, so degraded audio is an
     expected outcome rather than a bug -- but it must be MEASURED and reported, not
     discovered later. A hard failure here is "no audio at all", which would make the
     latency numbers meaningless because there would be no first-audio event to time.

Exit code 0 = arm verified, spend the GPU time. Non-zero = do not.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
from stage_stats_v2 import parse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--arm", required=True, choices=["control", "treatment"])
    ap.add_argument("--min-requests", type=int, default=4)
    ap.add_argument("--boot", type=int, default=None,
                    help="only look at this boot index (logs are appended across boots)")
    args = ap.parse_args()

    recs = parse(pathlib.Path(args.log))
    if args.boot is not None:
        recs = [r for r in recs if r["boot"] == args.boot]
    else:
        # the most recent boot present
        if recs:
            last = max(r["boot"] for r in recs)
            recs = [r for r in recs if r["boot"] == last]

    fails: list[str] = []
    notes: list[str] = []

    print(f"[gate] arm={args.arm}  log={args.log}  requests parsed={len(recs)}")
    if len(recs) < args.min_requests:
        print(f"[gate] FAIL: only {len(recs)} requests, need {args.min_requests}. "
              f"The smoke session did not reach the server -- check the stimulus path.")
        return 2

    rows = []
    for r in recs:
        s0 = r["stages"].get(0) or {}
        s1 = r["stages"].get(1) or {}
        s2 = r["stages"].get(2) or {}
        rows.append({
            "p0": s0.get("num_tokens_in"),
            "p1": s1.get("num_tokens_in"),
            "t0": s0.get("vllm_ttft_ms"),
            "t1": s1.get("vllm_ttft_ms"),
            "out1": s1.get("num_tokens_out"),
            "aud": s2.get("audio_duration_s"),
        })

    print(f"[gate] {'#':>3s} {'stage0 tok':>10s} {'stage1 tok':>10s} {'ratio':>6s} "
          f"{'talker ms':>9s} {'codec out':>9s} {'audio s':>8s}")
    for i, x in enumerate(rows):
        ratio = (x["p1"] / x["p0"]) if (x["p0"] and x["p1"] is not None) else float("nan")
        tk = (x["t1"] - x["t0"]) if (x["t1"] is not None and x["t0"] is not None) else float("nan")
        print(f"[gate] {i:3d} {x['p0'] or 0:10.0f} "
              f"{(x['p1'] if x['p1'] is not None else -1):10.0f} {ratio:6.3f} "
              f"{tk:9.0f} {(x['out1'] or 0):9.0f} {(x['aud'] or 0):8.2f}")

    # 2. stage-1 prompt tokens must be reported
    p1s = [x["p1"] for x in rows if x["p1"] is not None]
    if not p1s or max(p1s) == 0:
        fails.append(
            "stage-1 num_tokens_in is 0 for every request. The stage_pool patch is NOT "
            "live, so the talker's prompt length is unrecorded and this experiment has "
            "no x-axis. Re-install the patch set."
        )
    else:
        notes.append(f"stage-1 prompt tokens observed: {min(p1s):.0f}..{max(p1s):.0f}")

    # 3. arm identity from the ratio trend over the smoke turns
    pairs = [(x["p0"], x["p1"]) for x in rows if x["p0"] and x["p1"] is not None]
    if len(pairs) >= 3:
        ratios = [b / a for a, b in pairs]
        late = st.median(ratios[len(ratios) // 2:])
        notes.append(f"stage1/stage0 prompt ratio, later turns: {late:.3f}")
        if args.arm == "control" and late < 0.5:
            fails.append(
                f"control arm shows ratio {late:.3f}: the talker is NOT summing every "
                f"user block, so PA_TALKER_LAST_BLOCK leaked into the control."
            )
        if args.arm == "treatment" and late > 0.5:
            fails.append(
                f"treatment arm shows ratio {late:.3f}: the talker prompt still tracks "
                f"the thinker's, so PA_TALKER_LAST_BLOCK did NOT take effect. Discard "
                f"this arm; do not report it as 'no effect'."
            )
    else:
        fails.append("not enough requests with both prompt lengths to identify the arm")

    # 4. audio actually produced
    auds = [x["aud"] for x in rows if x["aud"] is not None]
    silent = sum(1 for a in auds if a <= 0.01)
    if not auds or silent == len(auds):
        fails.append(
            "no audio on any turn. There is no first-audio event to time, so latency "
            "cannot be measured on this arm."
        )
    elif silent:
        notes.append(f"WARNING {silent}/{len(auds)} turns produced no audio")
    if auds:
        notes.append(f"audio duration s: median {st.median(auds):.2f}, "
                     f"range {min(auds):.2f}..{max(auds):.2f} "
                     f"(a COVARIATE for the treatment arm, report it)")

    print()
    for n in notes:
        print(f"[gate] note: {n}")
    if fails:
        print()
        for f in fails:
            print(f"[gate] FAIL: {f}")
        return 1
    print()
    print("[gate] PASS -- arm identity verified from server-side evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
