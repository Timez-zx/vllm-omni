#!/usr/bin/env python3
"""Did shortening the TALKER's prompt to the newest frames stop its cost from growing?

Written before the arms finished, on purpose: the estimator choices below should not be
made after seeing which way the answer went.

THE TWO ARMS (harness/run_talker_delta_probe.sh)
  E640  control    PA_TURN_BLOCKS=1 PA_TALKER_LAST_BLOCK=0 -- talker sums every user block
  F640  treatment  PA_TURN_BLOCKS=1 PA_TALKER_LAST_BLOCK=1 -- talker sizes from the last one
Identical otherwise: 640x352, append-only, EVS gap [8,16], frame cap 284, high-motion
handheld stimulus, 1 user, 4 sessions x 60 turns.

THE DECISIVE NUMBER is the slope of the talker's own added latency against the THINKER's
prompt length. Against the thinker's, not the talker's, because that is the quantity the
proposal is trying to decouple from: "the session has a lot of video in it" must stop
implying "speaking is slow".

  E640 slope ~= F640 slope   -> the talker's cost is NOT its placeholder length. The
                                session-scoped incremental rewrite would not have helped.
  F640 slope ~= 0            -> the talker's cost IS its placeholder length, and true
                                incremental talker prefill is worth building.

ESTIMATOR RULES, both of which were violated earlier in this study and produced fictional
findings:
  - per-stage deltas are computed WITHIN a request, then aggregated. Never
    median(stage1_ttft) - median(stage0_ttft).
  - shares are per-request, then aggregated. Never p50(part)/p50(whole).

THE CONFOUND THAT MUST BE REPORTED, NOT ASSUMED AWAY: the treatment withholds
conditioning from the talker, so it may simply say less, and a talker that says less is
trivially faster. Stage-1 num_tokens_out, stage-2 audio_duration_s and stage-0
num_tokens_out are therefore printed for every turn bin, and a per-request
latency-per-codec-token is reported so the comparison survives a length difference.
"""
from __future__ import annotations

import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage_stats_v2 import derive, parse  # noqa: E402

RES = pathlib.Path("/data/zx/results")

# label -> (log, boot, description). server_td.log APPENDS per boot, so boot 1 is the
# control and boot 2 the treatment, in the order run_talker_delta_probe.sh boots them.
ARMS = [
    ("E640 control", "server_td.log", 1, "turn blocks, talker sums ALL user blocks"),
    ("F640 treatment", "server_td.log", 2, "turn blocks, talker sizes from the LAST block"),
    ("T640 reference", "server_ft.log", 2, "single block, append-only (the existing arm)"),
    ("C640 reference", "server_ft.log", 1, "shipped re-pick, prompt pinned ~3.8k"),
]

BINS = ((1, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59))
_cache: dict[str, list[dict]] = {}


def rows(log: str, boot: int, limit: int = 240) -> list[dict]:
    if log not in _cache:
        p = RES / log
        _cache[log] = derive(parse(p)) if p.exists() else []
    rs = [r for r in _cache[log] if r["boot"] == boot][:limit]
    for i, r in enumerate(rs):
        r["idx"] = i
    return rs


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def ols(pts):
    if len(pts) < 10:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0:
        return None
    slope = sxy / sxx
    return slope, my - slope * mx, ((sxy ** 2) / (sxx * syy) if syy else float("nan"))


def stage1_tok(r):
    return (r.get("_s1_in") if "_s1_in" in r else None)


def main() -> int:
    # derive() does not carry stage-1 num_tokens_in, so pull it from the raw records.
    raw: dict[str, list[dict]] = {}
    for _, log, _, _ in ARMS:
        if log in raw:
            continue
        p = RES / log
        raw[log] = parse(p) if p.exists() else []

    data = {}
    for label, log, boot, desc in ARMS:
        rs = rows(log, boot)
        if not rs:
            continue
        # join stage-1 num_tokens_in by request_id
        s1 = {}
        for rec in raw[log]:
            v = (rec["stages"].get(1) or {}).get("num_tokens_in")
            s1[rec["request_id"]] = v
        for r in rs:
            r["_s1_in"] = s1.get(r["request_id"])
        data[label] = (rs, desc)

    if not data:
        print("no data yet -- the arms have not produced a server log")
        return 1

    print("=" * 90)
    print("ARM IDENTITY -- does the talker's prompt actually differ between the arms?")
    print("=" * 90)
    print("If the treatment's stage-1 prompt tracks stage 0's, the switch never took")
    print("effect and the arm must be DISCARDED, not reported as 'no effect'.")
    print()
    print(f"  {'arm':16s} {'n':>4s} {'stage0 tok (first->last bin)':>30s} "
          f"{'stage1 tok (first->last)':>26s} {'ratio last bin':>15s}")
    for label, (rs, _) in data.items():
        bs = [(b, [r for r in rs if b[0] <= r["idx"] <= b[1]]) for b in BINS]
        bs = [(b, g) for b, g in bs if g]
        if len(bs) < 2:
            continue
        f, l = bs[0][1], bs[-1][1]
        p0f, p0l = med(r["prompt_tokens"] for r in f), med(r["prompt_tokens"] for r in l)
        p1f, p1l = med(r["_s1_in"] for r in f), med(r["_s1_in"] for r in l)
        ratio = (p1l / p0l) if p0l and p1l == p1l else float("nan")
        print(f"  {label:16s} {len(rs):4d} {f'{p0f:.0f} -> {p0l:.0f}':>30s} "
              f"{f'{p1f:.0f} -> {p1l:.0f}':>26s} {ratio:15.3f}")
    print()
    print("  (stage-1 tok is NaN for the reference arms: they ran before the stage_pool")
    print("   patch that reports num_tokens_in for stages other than 0.)")

    print()
    print("=" * 90)
    print("PER-TURN BREAKDOWN")
    print("=" * 90)
    for label, (rs, desc) in data.items():
        print(f"\n  {label}  --  {desc}")
        print(f"    {'turns':>7s} {'s0 tok':>7s} {'s1 tok':>7s} {'thinker':>8s} "
              f"{'TALKER':>7s} {'c2wav':>6s} {'TTFA':>7s} | {'text out':>8s} "
              f"{'codec out':>9s} {'audio s':>8s} {'ms/codec':>8s}")
        for a, b in BINS:
            g = [r for r in rs if a <= r["idx"] <= b]
            if not g:
                continue
            # per-request ratio, then median -- never median/median
            per = [r["talker_add_ms"] / r["codec_tokens_out"]
                   for r in g
                   if r.get("talker_add_ms") is not None and (r.get("codec_tokens_out") or 0) > 0]
            print(f"    {f'{a}-{b}':>7s} {med(r['prompt_tokens'] for r in g):7.0f} "
                  f"{med(r['_s1_in'] for r in g):7.0f} "
                  f"{med(r['thinker_ttft_ms'] for r in g):8.0f} "
                  f"{med(r['talker_add_ms'] for r in g):7.0f} "
                  f"{med(r['code2wav_add_ms'] for r in g):6.0f} "
                  f"{med(r['audio_ttfa_ms'] for r in g):7.0f} | "
                  f"{med(r['text_tokens_out'] for r in g):8.0f} "
                  f"{med(r['codec_tokens_out'] for r in g):9.0f} "
                  f"{med(r['audio_s'] for r in g):8.2f} "
                  f"{(st.median(per) if per else float('nan')):8.2f}")

    print()
    print("=" * 90)
    print("THE DECISIVE REGRESSION -- talker added latency vs the THINKER's prompt length")
    print("=" * 90)
    print(f"  {'arm':16s} {'n':>4s} {'ms per 1k s0 tok':>17s} {'floor ms':>9s} {'R2':>7s} "
          f"{'s0 tok range':>18s}")
    slopes = {}
    for label, (rs, _) in data.items():
        pts = [(r["prompt_tokens"], r["talker_add_ms"]) for r in rs
               if r["prompt_tokens"] and r["talker_add_ms"] is not None]
        f = ols(pts)
        if not f:
            print(f"  {label:16s} {len(pts):4d} {'prompt does not vary':>17s}")
            continue
        slope, icpt, r2 = f
        slopes[label] = 1000 * slope
        print(f"  {label:16s} {len(pts):4d} {1000*slope:17.1f} {icpt:9.0f} {r2:7.4f} "
              f"{f'{min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f}':>18s}")

    if "E640 control" in slopes and "F640 treatment" in slopes:
        e, f = slopes["E640 control"], slopes["F640 treatment"]
        print()
        print(f"  control {e:.1f} ms/1k  ->  treatment {f:.1f} ms/1k   "
              f"({'reduced ' + format(100*(1-f/e), '.0f') + '%' if e else 'n/a'})")
        print()
        if e and f / e < 0.25:
            print("  VERDICT: the talker's cost IS its placeholder length. Shortening the")
            print("  talker's prompt to the newest frames largely removes the growth, so")
            print("  true incremental talker prefill has a real payoff and the")
            print("  session-scoped streaming rewrite is justified.")
        elif e and f / e > 0.75:
            print("  VERDICT: the talker's cost is NOT its placeholder length. Its growth")
            print("  survives a short prompt, so something else scales with the thinker's")
            print("  prompt -- the connector shipping per-position conditioning is the")
            print("  leading suspect and transfers=[0->1=0.00ms] in the OmniTiming line is")
            print("  unmeasured rather than zero. The incremental rewrite would NOT have")
            print("  fixed this. Do not claim the mechanism without measuring it.")
        else:
            print("  VERDICT: partial. The placeholder length explains some but not most of")
            print("  the growth. Report the fraction, and do not round it to a story.")

        print()
        print("  BEFORE QUOTING ANY OF THIS, check the covariate columns above: if the")
        print("  treatment's codec_out or audio_s fell, part of its speedup is that it")
        print("  said less. The ms/codec column is the length-normalised comparison.")

    print()
    print("=" * 90)
    print("SECONDARY -- what the prompt restructure alone cost (E640 vs T640)")
    print("=" * 90)
    print("  Both are append-only at 640x352 with gap [8,16]. E640 splits the frames into")
    print("  one user block per turn; T640 keeps them in a single block. Any difference is")
    print("  the price of the restructure, and it must be small for E640 to serve as a")
    print("  control for F640.")
    for label in ("E640 control", "T640 reference"):
        if label not in data:
            continue
        rs = data[label][0]
        bs = [(b, [r for r in rs if b[0] <= r["idx"] <= b[1]]) for b in BINS]
        bs = [(b, g) for b, g in bs if g]
        if not bs:
            continue
        l = bs[-1][1]
        print(f"    {label:16s} last bin: s0 tok {med(r['prompt_tokens'] for r in l):6.0f}  "
              f"thinker {med(r['thinker_ttft_ms'] for r in l):5.0f}  "
              f"talker {med(r['talker_add_ms'] for r in l):6.0f}  "
              f"TTFA {med(r['audio_ttfa_ms'] for r in l):6.0f}  "
              f"text out {med(r['text_tokens_out'] for r in l):4.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
