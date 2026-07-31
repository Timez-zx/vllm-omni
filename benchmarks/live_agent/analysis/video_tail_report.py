#!/usr/bin/env python3
"""Tail latency of the realistic workload: p50/p90/p95/p99 up to 8 users.

Experiment A (frame count forced, filter off) is not repeated here -- it exists to
establish causality, not to describe a workload. This reports only the shipped
configuration: EVS on at 0.95, num_frames 16, max_frames 64, real content.

**On percentile honesty.** A percentile needs samples. With n scored turns, the
q-th percentile is interpolated between the two order statistics around q*(n-1),
so p99 at n=19 is simply the maximum wearing a different name. The rule of thumb
used here: an estimate is only labelled SUPPORTED when n >= 10/(1-q), i.e. 200
samples for p95 and 1000 for p99. Anything less is printed but marked, because
reporting p99 from 19 samples as if it were a percentile is how tail claims go
wrong.

Sample counts by design (60 turns, first dropped):

    8 users  8 concurrent sessions x 59  = 472
    4 users  4 concurrent sessions x 59  = 236
    1 user   4 SEQUENTIAL sessions x 59  = 236   <- see below

The 1-user arm is four sequential 60-turn sessions rather than one 237-turn
session, and that choice is load-bearing. TTFA degrades within a session as the
rolling frame buffer fills (low motion, 4 users: turn 1-10 p50 499 ms -> turn 51-60
1,024 ms, prompt 1,441 -> 12,957 tokens). One long session would put most of its
samples at turn indices the 8-user run never reached, so the 1-vs-8 comparison
would be reading session ageing as concurrency. Four short sessions hold turn index
to 2-60 on every arm, matching the 4- and 8-user runs exactly.

So p90 and p95 are supported on the 1-, 4- and 8-user arms, and **p99 is supported
nowhere** -- at n=236 it lands on the 2nd-worst turn, at n=472 on the 6th-worst. It
is printed but flagged, because reporting p99 from 236 samples as though it were a
percentile is how tail claims go wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

TAG = re.compile(r"^u(?P<u>\d+)_(?P<c>static|low|high)$")
CONTENT_ORDER = {"static": 0, "low": 1, "high": 2}
LABEL = {"static": "screencast (static)",
         "low": "talkinghead (low motion)",
         "high": "handheld (high motion)"}
SEGS = [("admit_s", "admission"),
        ("to_first_token_s", "encoders+prefill+1st tok"),
        ("to_first_audio_s", "talker+code2wav")]


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile on the order statistics."""
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def supported(n: int, q: float) -> bool:
    """Is n large enough for this percentile to mean anything?"""
    return n >= 10.0 / (1.0 - q)


def load(res: pathlib.Path, tag: str) -> dict | None:
    p = res / f"decomp_vt_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = [r for r in d.get("rows", []) if r.get("ttfa_end_s") is not None]
    if not rows:
        return None
    out: dict = {"users": d.get("users"), "n": len(rows)}
    for k, _ in SEGS + [("ttfa_end_s", "TTFA")]:
        v = [r[k] * 1000 for r in rows if r.get(k) is not None]
        out[k] = {"n": len(v), "p50": pct(v, .50), "p90": pct(v, .90),
                  "p95": pct(v, .95), "p99": pct(v, .99),
                  "max": max(v) if v else None,
                  "mean": st.fmean(v) if v else None}
    # how many turns blew past the comfort line, and past a second
    t = [r["ttfa_end_s"] * 1000 for r in rows]
    out["frac_over_500ms"] = sum(1 for x in t if x > 500) / len(t)
    out["frac_over_1s"] = sum(1 for x in t if x > 1000) / len(t)
    out["frac_over_2s"] = sum(1 for x in t if x > 2000) / len(t)
    # per-user spread: is the pain shared or concentrated on one unlucky user?
    by_u: dict = {}
    for r in rows:
        by_u.setdefault(r.get("uid"), []).append(r["ttfa_end_s"] * 1000)
    out["per_user_p50"] = {str(k): round(st.median(v), 0) for k, v in sorted(by_u.items())}
    return out


def ms(x) -> str:
    return "-" if x is None else f"{x:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--log", default="/data/zx/results/server_vt.log")
    ap.add_argument("--out", default="/data/zx/results/video_tail.json")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    arms: dict[tuple[int, str], dict] = {}
    for p in sorted(glob.glob(str(res / "decomp_vt_*.json"))):
        tag = pathlib.Path(p).stem.replace("decomp_vt_", "")
        m = TAG.match(tag)
        if not m:
            continue
        d = load(res, tag)
        if d:
            arms[(int(m["u"]), m["c"])] = d
    if not arms:
        print("no decomp_vt_* found")
        return 1

    keys = sorted(arms, key=lambda k: (CONTENT_ORDER.get(k[1], 9), k[0]))
    ns = {n for (_, _), d in arms.items() for n in [d["n"]]}

    print("\n" + "=" * 96)
    print("TTFA TAIL, shipped config (EVS on 0.95, num_frames 16), by content and user count")
    print("=" * 96)
    print(f"{'content':<26}{'users':>6}{'n':>5}{'p50':>8}{'p90':>8}{'p95':>8}"
          f"{'p99':>9}{'worst':>8}{'mean':>8}   {'>0.5s':>6}{'>1s':>6}{'>2s':>6}")
    for k in keys:
        d = arms[k]
        t = d["ttfa_end_s"]
        print(f"{LABEL[k[1]]:<26}{k[0]:>6}{d['n']:>5}"
              f"{ms(t['p50']):>8}{ms(t['p90']):>8}{ms(t['p95']):>8}"
              f"{ms(t['p99']):>9}{ms(t['max']):>8}{ms(t['mean']):>8}   "
              f"{d['frac_over_500ms']:>5.0%}{d['frac_over_1s']:>6.0%}{d['frac_over_2s']:>6.0%}")

    print("\n--- how much of each percentile the sample count actually supports ---")
    print(f"{'n (scored turns)':<20}{'p90':>10}{'p95':>10}{'p99':>10}")
    for n in sorted(ns):
        row = f"{n:<20}"
        for q in (.90, .95, .99):
            row += f"{('SUPPORTED' if supported(n, q) else 'thin'):>10}"
        print(row)
    print("  rule: n >= 10/(1-q). p99 needs 1000 samples; no single session here has")
    print("  that, so read the p99 column as 'worst couple of turns', not a percentile.")

    print("\n--- amplification with user count (per content) ---")
    for c in ("static", "low", "high"):
        us = sorted(u for u, cc in arms if cc == c)
        if len(us) < 2:
            continue
        base = arms[(us[0], c)]["ttfa_end_s"]
        print(f"  {LABEL[c]}")
        for q in ("p50", "p95"):
            row = f"    {q}: "
            for u in us:
                v = arms[(u, c)]["ttfa_end_s"][q]
                r = (v / base[q]) if (v and base[q]) else None
                row += f"u{u}={ms(v)}ms({r:.2f}x) " if r else f"u{u}=- "
            print(row)

    print("\n--- tail spread: p95 / p50 (how much worse a bad turn is) ---")
    print(f"{'content':<26}" + "".join(f"{('u'+str(u)):>9}"
                                       for u in sorted({u for u, _ in arms})))
    for c in ("static", "low", "high"):
        row = f"{LABEL[c]:<26}"
        for u in sorted({u for u, _ in arms}):
            d = arms.get((u, c))
            if not d:
                row += f"{'-':>9}"
                continue
            t = d["ttfa_end_s"]
            row += f"{(t['p95']/t['p50'] if t['p50'] else 0):>8.2f}x"
        print(row)

    print("\n--- is the pain shared, or concentrated on one user? (per-user p50) ---")
    for k in keys:
        d = arms[k]
        if d["users"] and d["users"] > 1:
            v = list(d["per_user_p50"].values())
            print(f"  {LABEL[k[1]]:<26} u{k[0]}: {d['per_user_p50']}"
                  f"   spread max/min = {max(v)/min(v):.2f}x" if min(v) else "")

    print("\n--- segment breakdown at p95 (where the bad turns lose their time) ---")
    print(f"{'content':<26}{'users':>6}" + "".join(f"{lab:>28}" for _, lab in SEGS))
    for k in keys:
        d = arms[k]
        row = f"{LABEL[k[1]]:<26}{k[0]:>6}"
        for kk, _ in SEGS:
            row += f"{ms(d[kk]['p95']):>27} "
        print(row)

    # engine pressure: did 8 users x high motion overflow stage-0 KV?
    log = pathlib.Path(args.log)
    if log.exists():
        text = log.read_text(errors="ignore")
        pats = {
            "preempted": r"[Pp]reempt",
            "recompute": r"recompute",
            "cache full / no blocks": r"[Cc]annot allocate|out of.*blocks|KV cache is full",
            "prompt too long": r"longer than the maximum",
        }
        print("\n--- engine pressure signals in the server log ---")
        any_hit = False
        for lab, pat in pats.items():
            n = len(re.findall(pat, text))
            if n:
                any_hit = True
                print(f"  {lab:<26} {n} lines")
        if not any_hit:
            print("  none -- no preemption, no KV exhaustion, no prompt-length rejections")
        print("  (8 users x high motion needs ~115,200 prompt tokens of KV against")
        print("   106,880 available on stage 0, so this is the thing to check.)")

    pathlib.Path(args.out).write_text(json.dumps(
        {f"u{u}_{c}": d for (u, c), d in arms.items()}, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
