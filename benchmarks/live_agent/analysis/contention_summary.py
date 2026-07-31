#!/usr/bin/env python3
"""Cross-run summary of the contention sweep.

Answers three questions the per-run decomposition cannot answer on its own:

  1. Which SEGMENT of TTFA grows with contention? That localises the bottleneck
     without guessing:
        admission grows        -> queueing before the engine
        first-token grows      -> stage 0 (encoders + vision prefill + decode)
        first-audio grows      -> stage 1 talker / stage 2 code2wav
  2. Is the degradation FAIR across users, or is some user starved? Reported as
     the spread of per-user p50 TTFA within a run.
  3. Does the GPU actually saturate, or is the added latency queueing on a
     resource that is not compute-bound?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir", default="/data/zx/results")
    ap.add_argument("--users", default="1,2,4,8")
    args = ap.parse_args()
    R = pathlib.Path(args.resdir)
    us = [int(x) for x in args.users.split(",")]

    runs = {}
    for u in us:
        p = R / f"ttfa_c{u}.json"
        if p.exists():
            runs[u] = json.loads(p.read_text())
    if not runs:
        print("no contention results yet")
        return 1

    SEG = [("admit_s", "admission"),
           ("to_first_token_s", "-> 1st text token (stage 0)"),
           ("to_first_audio_s", "-> 1st audio (stage 1+2)"),
           ("ttfa_end_s", "TTFA total")]

    print("\n=== TTFA vs concurrent users (ms, p50 / p95) ===")
    print(f"{'users':>6}{'reps':>6}", end="")
    for _, lab in SEG:
        print(f"{lab:>30}", end="")
    print()
    for u in sorted(runs):
        a = runs[u]["aggregate"]
        n = len([r for r in runs[u]["rows"] if r.get("ttfa_end_s")])
        print(f"{u:>6}{n:>6}", end="")
        for k, _ in SEG:
            if k in a:
                print(f"{a[k]['p50']*1000:>16.0f} /{a[k]['p95']*1000:>11.0f}", end="")
            else:
                print(f"{'-':>30}", end="")
        print()

    base = runs[min(runs)]["aggregate"]
    print("\n=== growth relative to 1 user (p50) -- localises the bottleneck ===")
    print(f"{'users':>6}", end="")
    for _, lab in SEG:
        print(f"{lab:>30}", end="")
    print()
    for u in sorted(runs):
        a = runs[u]["aggregate"]
        print(f"{u:>6}", end="")
        for k, _ in SEG:
            if k in a and k in base and base[k]["p50"] > 0:
                print(f"{a[k]['p50']/base[k]['p50']:>29.2f}x", end="")
            else:
                print(f"{'-':>30}", end="")
        print()

    print("\n=== added milliseconds vs 1 user, by segment (p50) ===")
    print(f"{'users':>6}{'admission':>12}{'stage 0':>12}{'stage 1+2':>12}"
          f"{'total':>10}{'dominant contributor':>24}")
    for u in sorted(runs):
        if u == min(runs):
            continue
        a = runs[u]["aggregate"]
        d = {}
        for k in ("admit_s", "to_first_token_s", "to_first_audio_s"):
            d[k] = (a[k]["p50"] - base[k]["p50"]) * 1000 if k in a and k in base else 0.0
        tot = (a["ttfa_end_s"]["p50"] - base["ttfa_end_s"]["p50"]) * 1000
        dom = max(d, key=lambda k: d[k])
        name = {"admit_s": "queueing (admission)",
                "to_first_token_s": "stage 0 thinker",
                "to_first_audio_s": "stage 1+2 talker/code2wav"}[dom]
        print(f"{u:>6}{d['admit_s']:>12.0f}{d['to_first_token_s']:>12.0f}"
              f"{d['to_first_audio_s']:>12.0f}{tot:>10.0f}"
              f"{name:>24} ({d[dom]/tot*100 if tot else 0:.0f}%)")

    print("\n=== fairness: per-user p50 TTFA within each run (ms) ===")
    for u in sorted(runs):
        byu = {}
        for r in runs[u]["rows"]:
            v = r.get("ttfa_end_s")
            if isinstance(v, (int, float)):
                byu.setdefault(r["uid"], []).append(v * 1000)
        if not byu:
            continue
        p50s = {k: pct(v, 50) for k, v in sorted(byu.items())}
        lo, hi = min(p50s.values()), max(p50s.values())
        print(f"  users={u}: " + "  ".join(f"u{k}={v:.0f}" for k, v in p50s.items())
              + f"   spread {hi-lo:.0f} ms ({hi/lo:.2f}x)")

    print("\n=== does the GPU actually saturate? (median over reps) ===")
    print(f"{'users':>6}{'busy during speech':>21}{'busy during TTFA':>19}")
    for u in sorted(runs):
        sp = [r["gpu_busy_speech_frac"] for r in runs[u]["rows"]
              if isinstance(r.get("gpu_busy_speech_frac"), (int, float))]
        tw = [r["gpu_busy_ttfa_frac"] for r in runs[u]["rows"]
              if isinstance(r.get("gpu_busy_ttfa_frac"), (int, float))]
        print(f"{u:>6}{(st.median(sp)*100 if sp else float('nan')):>20.1f}%"
              f"{(st.median(tw)*100 if tw else float('nan')):>18.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
