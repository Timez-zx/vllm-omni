#!/usr/bin/env python3
"""Per-request, per-stage server-side telemetry from a vllm-omni server log.

WHY THIS EXISTS
---------------
Every latency number in this study so far was split CLIENT-side: tx_query -> rx_first_text
was called "thinker" and rx_first_text -> rx_first_audio was called "talker+code2wav". That
split is defensible but indirect, and it cannot see the prompt token count at all -- prompt
lengths were being INFERRED by inverting a fitted ms/1k rate, which is circular when the
thing under test is that very rate.

The server already logs what we wanted. Each finished request emits a StageRequestStats
table with one column per stage:

  | num_tokens_in                   |  2,624 |     0 |    0 |
  | vllm_ttft_ms                    | 879.27 | 1086.32 | 1192.61 |
  | num_tokens_out                  |     52 |   190 |    0 |
  | inter_output_latency_ms         |  13.49 | 10.65 | 239.43 |

So we get, per turn and for free:
  - the REAL prompt token count (stage 0 num_tokens_in), not an inversion
  - time to first output for each of the three stages separately
  - hence the talker's own added latency as stage1_ttft - stage0_ttft, per request

ESTIMATOR RULES (both of these were violated earlier in this study and produced fictional
findings, so they are enforced here rather than left to the caller):
  - a per-stage delta is computed WITHIN a request and only then aggregated. Never
    median(stage1) - median(stage0).
  - a share is computed per request and only then aggregated. Never p50(part)/p50(whole).

MATCHING TURNS
--------------
The client trace carries no request_id, so blocks are matched to turns by order: with one
user the requests are strictly sequential, so the k-th block of a boot is the k-th turn. The
--expect flag asserts the count so a silent mismatch cannot pass. Blocks are also stamped
with the log wall-clock time, which lets a caller cross-check against the client trace.

CAVEAT recorded on purpose: stage 1 and stage 2 report num_tokens_in = 0. The talker's
placeholder prompt (prompt_token_ids=[0]*prompt_len, qwen3_omni.py:710) is evidently not
counted here, so stage-1 prompt length is NOT directly observable from this log. Stage 0's
count is real. Do not quietly substitute one for the other.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAGE_HDR = re.compile(r"(\[stats\.py:\d+\]).*\[StageRequestStats \[request_id=([^\]]+)\]\]")
TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)")
BOOT = re.compile(r"Initializing a V1 LLM engine")
ROW = re.compile(r"^\s*\|\s*([a-z0-9_]+)\s*\|(.+)\|\s*$")

# Fields worth keeping. Values that look like "13.490 (n=51)" are reduced to the number.
KEEP = {
    "num_tokens_in", "num_tokens_out", "vllm_ttft_ms", "vllm_tpot_ms",
    "serving_time_to_first_output_ms", "stage_gen_time_ms", "inter_output_latency_ms",
    "audio_duration_s", "batch_size", "output_unit_count",
}


def _num(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    s = s.split("(")[0].strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse(path: pathlib.Path) -> list[dict]:
    """Return one dict per finished request, in log order."""
    out: list[dict] = []
    cur: dict | None = None
    boot = 0
    for raw in path.open(errors="replace"):
        line = ANSI.sub("", raw).rstrip()
        if BOOT.search(line):
            # one boot emits three of these (one per stage); count stage-0 only
            if "stage0_replica0" in line:
                boot += 1
            continue
        m = STAGE_HDR.search(line)
        if m:
            t = TS.search(line)
            cur = {
                "request_id": m.group(2),
                "t": t.group(1) if t else None,
                "boot": boot,
                "stages": {},
                "log_marker": m.group(1),
            }
            out.append(cur)
            continue
        if cur is None:
            continue
        # rows belonging to the table we are inside
        marker = cur["log_marker"]
        if marker not in line:
            # A different stats callsite means this table is over.
            if "[stats.py" in line:
                cur = None
            continue
        body = line.split(marker, 1)[1]
        rm = ROW.match(body)
        if not rm:
            continue
        field = rm.group(1)
        if field not in KEEP:
            continue
        vals = [_num(c) for c in rm.group(2).split("|")]
        for i, v in enumerate(vals):
            cur["stages"].setdefault(i, {})[field] = v
    # drop anything that did not get the three stages
    return [r for r in out if len(r["stages"]) >= 2]


def derive(recs: list[dict]) -> list[dict]:
    """Per-request derived quantities. All deltas computed WITHIN the request."""
    rows = []
    for r in recs:
        s = r["stages"]

        def g(i, k):
            return (s.get(i) or {}).get(k)

        t0, t1, t2 = g(0, "vllm_ttft_ms"), g(1, "vllm_ttft_ms"), g(2, "vllm_ttft_ms")
        if t0 is None or t1 is None:
            continue
        row = {
            "request_id": r["request_id"], "t": r["t"], "boot": r["boot"],
            "prompt_tokens": g(0, "num_tokens_in"),
            "text_tokens_out": g(0, "num_tokens_out"),
            "codec_tokens_out": g(1, "num_tokens_out"),
            "thinker_ttft_ms": t0,
            "talker_add_ms": t1 - t0,                       # per-request delta
            "code2wav_add_ms": (t2 - t1) if t2 is not None else None,
            "audio_ttfa_ms": t2,
            "thinker_itl_ms": g(0, "inter_output_latency_ms"),
            "talker_itl_ms": g(1, "inter_output_latency_ms"),
            "audio_s": g(2, "audio_duration_s"),
        }
        if t2:
            row["thinker_share"] = t0 / t2                  # per-request share
            row["talker_share"] = (t1 - t0) / t2
            row["code2wav_share"] = (t2 - t1) / t2
        rows.append(row)
    return rows


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 2:
        return float("nan")
    i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[i]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, action="append",
                    help="server log; repeatable. Use LABEL=PATH to name it.")
    ap.add_argument("--skip", type=int, default=1,
                    help="drop the first N requests of each boot (turn 1 is cold)")
    ap.add_argument("--expect", type=int, default=None,
                    help="assert this many usable requests per label")
    ap.add_argument("--bins", default="1-9,10-19,20-29,30-39,40-49,50-59")
    ap.add_argument("--out", default=None, help="write per-request rows as jsonl")
    ap.add_argument("--split-boot", action="store_true",
                    help="one label per server boot. REQUIRED whenever a log holds more "
                         "than one arm, which is the normal case: run_fix_tuned.sh boots "
                         "twice into the same log file. Lumping two arms together makes "
                         "the by-turn table meaningless.")
    ap.add_argument("--split-gap-s", type=float, default=None,
                    help="also split a boot wherever consecutive requests are more than "
                         "this many seconds apart. Used to separate two arms that shared "
                         "one boot (run_fix_latency_rest.sh runs low then static on one "
                         "server, with the decompose step in between).")
    ap.add_argument("--tok-max", type=float, default=None,
                    help="restrict the regression to requests below this prompt-token "
                         "count. Needed because an arm that hits the context cap piles up "
                         "many points at one x value and flattens the fit.")
    ap.add_argument("--request-regex", default=None,
                    help="keep only request IDs matching this regex, e.g. "
                         "'^video-(?!warm-)' for foreground AV turns")
    args = ap.parse_args()

    bins = []
    for b in args.bins.split(","):
        a, _, z = b.partition("-")
        bins.append((int(a), int(z)))

    allrows = {}
    for spec in args.log:
        label, _, path = spec.partition("=")
        if not path:
            label, path = pathlib.Path(spec).stem, spec
        p = pathlib.Path(path)
        if not p.exists():
            print(f"!! missing {p}", file=sys.stderr)
            continue
        rows = derive(parse(p))
        if args.request_regex:
            request_pattern = re.compile(args.request_regex)
            rows = [row for row in rows if request_pattern.search(row["request_id"])]
        by_boot = {}
        for r in rows:
            by_boot.setdefault(r["boot"], []).append(r)

        # Optionally cut each boot into segments at large inter-request gaps, so two arms
        # that shared one server become two labels.
        def segments(rs):
            if args.split_gap_s is None:
                return [rs]
            import datetime as _dt
            segs, cur_seg, prev = [], [], None
            for r in rs:
                t = None
                if r["t"]:
                    t = _dt.datetime.strptime("2026-" + r["t"], "%Y-%m-%d %H:%M:%S")
                if prev is not None and t is not None and (t - prev).total_seconds() > args.split_gap_s:
                    segs.append(cur_seg)
                    cur_seg = []
                cur_seg.append(r)
                if t is not None:
                    prev = t
            if cur_seg:
                segs.append(cur_seg)
            return segs

        groups = []
        for boot, rs in sorted(by_boot.items()):
            for k, seg in enumerate(segments(rs)):
                groups.append((boot, k, seg))

        print(f"{label}: {len(rows)} requests parsed over {len(by_boot)} boot(s), "
              f"{len(groups)} group(s)")
        for boot, k, seg in groups:
            kept = []
            for i, r in enumerate(seg):
                if i < args.skip:
                    continue
                r["idx"] = i          # 0-based turn index within the group
                kept.append(r)
            if args.split_boot or args.split_gap_s is not None:
                sub = f"{label}#b{boot}" + (f".{k}" if args.split_gap_s is not None else "")
            else:
                sub = label
            allrows.setdefault(sub, []).extend(kept)
            print(f"    {sub}: {len(seg)} requests, {len(kept)} kept, "
                  f"{seg[0]['t']} .. {seg[-1]['t']}, "
                  f"prompt {min(r['prompt_tokens'] or 0 for r in seg):.0f}"
                  f"-{max(r['prompt_tokens'] or 0 for r in seg):.0f} tok")
            if args.expect is not None and len(kept) != args.expect:
                print(f"    !! expected {args.expect}, got {len(kept)} -- turn matching NOT safe")

    print()
    print("=== per-request medians (deltas computed within a request, then medianed) ===")
    hdr = (f"{'label':22s} {'n':>4s} {'prompt_tok':>10s} {'thinker':>8s} {'talker':>8s} "
           f"{'code2wav':>8s} {'TTFA':>7s} {'TTFA p95':>8s} {'talker%':>7s}")
    print(hdr)
    for label, rows in allrows.items():
        if not rows:
            continue
        print(f"{label:22s} {len(rows):4d} {med(r['prompt_tokens'] for r in rows):10.0f} "
              f"{med(r['thinker_ttft_ms'] for r in rows):8.0f} "
              f"{med(r['talker_add_ms'] for r in rows):8.0f} "
              f"{med(r['code2wav_add_ms'] for r in rows):8.0f} "
              f"{med(r['audio_ttfa_ms'] for r in rows):7.0f} "
              f"{q([r['audio_ttfa_ms'] for r in rows], 95):8.0f} "
              f"{100*med(r['talker_share'] for r in rows if 'talker_share' in r):6.1f}%")

    print()
    print("=== by turn index: prompt tokens and the talker's own added latency ===")
    for label, rows in allrows.items():
        if not rows:
            continue
        print(f"\n  {label}")
        print(f"    {'turns':>9s} {'n':>4s} {'prompt_tok':>10s} {'thinker':>8s} "
              f"{'talker':>8s} {'code2wav':>8s} {'TTFA':>7s} {'text_out':>8s}")
        for a, b in bins:
            rs = [r for r in rows if a <= r["idx"] <= b]
            if not rs:
                continue
            print(f"    {f'{a}-{b}':>9s} {len(rs):4d} "
                  f"{med(r['prompt_tokens'] for r in rs):10.0f} "
                  f"{med(r['thinker_ttft_ms'] for r in rs):8.0f} "
                  f"{med(r['talker_add_ms'] for r in rs):8.0f} "
                  f"{med(r['code2wav_add_ms'] for r in rs):8.0f} "
                  f"{med(r['audio_ttfa_ms'] for r in rs):7.0f} "
                  f"{med(r['text_tokens_out'] for r in rs):8.0f}")

    # The crux regression: talker's added latency against the REAL prompt token count.
    print()
    print("=== talker added latency vs prompt tokens (OLS, per-request points) ===")
    print("    slope is ms per 1,000 prompt tokens; intercept is the fixed floor")
    for label, rows in allrows.items():
        pts = [(r["prompt_tokens"], r["talker_add_ms"]) for r in rows
               if r["prompt_tokens"] and r["talker_add_ms"] is not None
               and (args.tok_max is None or r["prompt_tokens"] <= args.tok_max)]
        if len(pts) < 10:
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        if sxx == 0:
            print(f"    {label:22s} prompt length does not vary -- slope undefined")
            continue
        slope = sxy / sxx
        icpt = my - slope * mx
        syy = sum((p[1] - my) ** 2 for p in pts)
        r2 = (sxy ** 2) / (sxx * syy) if syy else float("nan")
        print(f"    {label:22s} n={n:4d}  {1000*slope:8.2f} ms/1k  "
              f"floor {icpt:8.1f} ms  R2 {r2:.4f}  "
              f"prompt {min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f} tok")

    if args.out:
        with open(args.out, "w") as fh:
            for label, rows in allrows.items():
                for r in rows:
                    fh.write(json.dumps({"label": label, **r}) + "\n")
        print(f"\nwrote per-request rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
