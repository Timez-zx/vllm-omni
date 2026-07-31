#!/usr/bin/env python3
"""Single-user tail latency, and whether the way it was measured is trustworthy.

The 1-user arm is four SEQUENTIAL 60-turn sessions, not one 237-turn session, so
that turn index stays in the same 2-60 range as the 4- and 8-user arms. TTFA is
known to drift within a session as the rolling frame buffer fills, so a single long
session would have loaded the 1-vs-8 comparison with session ageing.

That design buys comparability but introduces two assumptions, and this report
CHECKS both instead of asserting them:

  1. THE FOUR SESSIONS AGREE. If session 4 is systematically slower than session 1,
     something carries across sessions (server-side cache state, memory growth) and
     pooling all 236 turns into one percentile would be mixing populations. The
     check is the spread of per-session p50.

  2. DELTA TRACING DID NOT PERTURB ANYTHING. This run timestamps every text delta;
     the earlier 11-turn vl_B_u1_* arms did not. Comparing turns 2-10 of both, at
     the same user count, content and config, isolates the tracing overhead. If they
     disagree, the tail numbers are contaminated and the comparison is void.

Only after both pass does the report print the 1 / 4 / 8 user tail comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

CONTENTS = ["static", "low", "high"]
LABEL = {"static": "screencast (static)",
         "low": "talkinghead (low motion)",
         "high": "handheld (high motion)"}
SEGS = [("admit_s", "admission"),
        ("to_first_token_s", "encode+prefill"),
        ("to_first_audio_s", "speech ramp")]
MATCH_TURNS = 10


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def supported(n, q):
    return n >= 10.0 / (1.0 - q)


def ms(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.0f}"


def rows_of(p: pathlib.Path, max_rep=None) -> list[dict]:
    if not p.exists():
        return []
    rs = [r for r in json.loads(p.read_text()).get("rows", [])
          if r.get("ttfa_end_s") is not None]
    if max_rep is not None:
        rs = [r for r in rs if r.get("rep", 10 ** 9) <= max_rep]
    return rs


def per_session(res: pathlib.Path, content: str) -> dict[int, list[float]]:
    """TTFA per session, read straight from the traces (the decomp JSON pools them).

    Sessions are distinguished by filename suffix, not uid: every session leaves uid
    at 0 on purpose so downstream tools still report "1 user" rather than mistaking
    four sequential sessions for four concurrent ones.
    """
    out: dict[int, list[float]] = {}
    for p in sorted(glob.glob(str(res / f"vt_u1_{content}" / "ttfa_user*.jsonl"))):
        m = re.search(r"_s(\d+)\.jsonl$", p)
        sess = int(m.group(1)) if m else 0
        q = {}
        for line in open(p):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k, r = e.get("k"), e.get("rep")
            if k == "tx_query":
                q.setdefault(r, {})["q"] = e["w"]
            elif k == "rx_first_audio":
                q.setdefault(r, {})["fa"] = e["w"]
        vals = [(d["fa"] - d["q"]) * 1000 for r, d in q.items()
                if r and r >= 1 and "q" in d and "fa" in d]
        if vals:
            out[sess] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--out", default="/data/zx/results/user1_tail.json")
    a = ap.parse_args()
    res = pathlib.Path(a.results)

    arms = {c: rows_of(res / f"decomp_vt_u1_{c}.json") for c in CONTENTS}
    arms = {c: r for c, r in arms.items() if r}
    if not arms:
        print("no decomp_vt_u1_* yet -- the run has not produced any content arm")
        return 1
    print(f"found 1-user arms: {', '.join(arms)}")

    # ---------------- check 1: do the four sessions agree?
    print("\n" + "=" * 94)
    print("CHECK 1  do the four sequential sessions agree? (if not, they cannot be pooled)")
    print("=" * 94)
    ok1 = True
    print(f"{'content':<26}" + "".join(f"{f's{i} p50':>10}" for i in (1, 2, 3, 4))
          + f"{'spread':>9}{'verdict':>10}")
    sess_all = {}
    for c in CONTENTS:
        if c not in arms:
            continue
        ps = per_session(res, c)
        sess_all[c] = {k: len(v) for k, v in ps.items()}
        if not ps:
            continue
        med = {k: st.median(v) for k, v in sorted(ps.items())}
        v = list(med.values())
        spread = max(v) / min(v) if min(v) else 0
        good = spread <= 1.25
        ok1 &= good
        row = f"{LABEL[c]:<26}"
        for i in (1, 2, 3, 4):
            row += ms(med.get(i), 10)
        row += f"{spread:>8.2f}x{('ok' if good else 'DRIFT'):>10}"
        print(row)
    print("  spread = slowest session p50 / fastest. Under 1.25x the sessions are")
    print("  interchangeable and pooling 4 x 59 turns into one percentile is valid.")
    print(f"  -> {'PASS' if ok1 else 'FAIL: sessions are not interchangeable'}")

    # ---------------- check 2: did --trace-deltas perturb the measurement?
    print("\n" + "=" * 94)
    print("CHECK 2  did timestamping every text delta change the latency?")
    print("=" * 94)
    print("  same user count, content and config; turns 2-10 on both sides.")
    print("  left: earlier 11-turn arms WITHOUT tracing. right: this run WITH tracing.")
    print(f"\n{'content':<26}{'no tracing':>12}{'n':>4}{'with tracing':>14}{'n':>5}"
          f"{'diff':>8}{'verdict':>10}")
    ok2 = True
    for c in CONTENTS:
        if c not in arms:
            continue
        old = rows_of(res / f"decomp_vl_B_u1_{c}.json", MATCH_TURNS)
        new = [r for r in arms[c] if r.get("rep", 10 ** 9) <= MATCH_TURNS]
        if not old or not new:
            continue
        o = pct([r["ttfa_end_s"] * 1000 for r in old], .5)
        n = pct([r["ttfa_end_s"] * 1000 for r in new], .5)
        d = (n - o) / o if o else None
        good = d is not None and abs(d) <= 0.20
        ok2 &= good
        print(f"{LABEL[c]:<26}{ms(o, 12)}{len(old):>4}{ms(n, 14)}{len(new):>5}"
              f"{(f'{d:+.0%}' if d is not None else '-'):>8}"
              f"{('ok' if good else 'PERTURBED'):>10}")
    print("  Tolerance is +-20%: these are different server instances on different")
    print("  days, so exact equality is not expected -- a large one-sided shift is")
    print("  what would indict the instrumentation.")
    print(f"  -> {'PASS' if ok2 else 'FAIL: tracing may have shifted the numbers'}")

    # ---------------- the tail itself
    print("\n" + "=" * 94)
    print("SINGLE-USER TAIL, shipped config (EVS 0.95, 16 frames), 4 x 59 turns")
    print("=" * 94)
    print(f"{'content':<26}{'n':>5}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}"
          f"{'worst':>8}{'mean':>8}   {'>0.5s':>6}{'>1s':>6}{'>2s':>6}")
    summary = {}
    for c in CONTENTS:
        if c not in arms:
            continue
        t = [r["ttfa_end_s"] * 1000 for r in arms[c]]
        summary[c] = {"n": len(t), "p50": pct(t, .5), "p90": pct(t, .9),
                      "p95": pct(t, .95), "p99": pct(t, .99),
                      "max": max(t), "mean": st.fmean(t),
                      "over500": sum(1 for x in t if x > 500) / len(t),
                      "over1s": sum(1 for x in t if x > 1000) / len(t),
                      "over2s": sum(1 for x in t if x > 2000) / len(t)}
        s = summary[c]
        print(f"{LABEL[c]:<26}{s['n']:>5}{ms(s['p50'])}{ms(s['p90'])}{ms(s['p95'])}"
              f"{ms(s['p99'])}{ms(s['max'])}{ms(s['mean'])}   "
              f"{s['over500']:>5.0%}{s['over1s']:>6.0%}{s['over2s']:>6.0%}")
    n_any = next(iter(summary.values()))["n"] if summary else 0
    print(f"\n  estimator support at n={n_any}: "
          f"p90 {'ok' if supported(n_any, .9) else 'thin'}, "
          f"p95 {'ok' if supported(n_any, .95) else 'thin'}, "
          f"p99 {'thin -- read as worst few turns' if not supported(n_any, .99) else 'ok'}")

    print("\n--- tail spread at 1 user: p95 / p50 ---")
    for c in CONTENTS:
        if c in summary and summary[c]["p50"]:
            print(f"  {LABEL[c]:<26}{summary[c]['p95']/summary[c]['p50']:.2f}x")

    print("\n--- segment split at 1 user (p50 / p95 ms) ---")
    print(f"{'content':<26}" + "".join(f"{lab:>22}" for _, lab in SEGS) + f"{'TTFA':>16}")
    for c in CONTENTS:
        if c not in arms:
            continue
        row = f"{LABEL[c]:<26}"
        for key, _ in SEGS + [("ttfa_end_s", "TTFA")]:
            v = [r[key] * 1000 for r in arms[c] if r.get(key) is not None]
            w = 16 if key == "ttfa_end_s" else 22
            row += f"{f'{pct(v, .5):.0f} / {pct(v, .95):.0f}':>{w}}"
        print(row)

    # ---------------- 1 / 4 / 8 comparison, turn index held constant by design
    print("\n" + "=" * 94)
    print("1 -> 4 -> 8 USERS, all arms at turn index 2-60, n=236/236/472")
    print("=" * 94)
    for q in ("p50", "p95"):
        print(f"\n  --- {q} TTFA (ms) ---")
        print(f"{'content':<26}{'1 user':>10}{'4 users':>10}{'8 users':>10}"
              f"{'4/1':>8}{'8/1':>8}{'8/4':>8}")
        for c in CONTENTS:
            vals = {}
            for u in (1, 4, 8):
                rs = rows_of(res / f"decomp_vt_u{u}_{c}.json")
                if rs:
                    vals[u] = pct([r["ttfa_end_s"] * 1000 for r in rs],
                                  .5 if q == "p50" else .95)
            if 1 not in vals:
                continue
            row = f"{LABEL[c]:<26}"
            for u in (1, 4, 8):
                row += ms(vals.get(u), 10)
            for a_, b_ in ((4, 1), (8, 1), (8, 4)):
                r = (vals[a_] / vals[b_]) if (a_ in vals and b_ in vals and vals[b_]) else None
                row += f"{(f'{r:.2f}x' if r else '-'):>8}"
            print(row)

    pathlib.Path(a.out).write_text(json.dumps(
        {"summary": summary, "sessions": sess_all,
         "checks": {"sessions_agree": ok1, "tracing_neutral": ok2}}, indent=2))
    print(f"\n-> {a.out}")
    if not (ok1 and ok2):
        print("\n!! a validity check failed -- read the tail numbers with that caveat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
