#!/usr/bin/env python3
"""Per-request, per-stage server-side telemetry from a vllm-omni server log.

WHY THIS EXISTS
---------------
Client-side timing cannot distinguish P, D, Talker, and Code2Wav or observe the
actual prompt length. This tool reads the server's per-request stage tables and
computes every stage delta within a request before aggregation.

Each finished request emits one column per stage. A P/D deployment uses:

  0 = Thinker P, 1 = Thinker D, 2 = Talker, 3 = Code2Wav

The output includes the real prompt token count, scheduler queue,
scheduled-to-core-output prefill latency, and time added by every stage.

ESTIMATOR RULES (both of these were violated earlier in this study and produced fictional
findings, so they are enforced here rather than left to the caller):
  - a per-stage delta is computed WITHIN a request and only then aggregated. Never
    median(stage1) - median(stage0).
  - a share is computed per request and only then aggregated. Never p50(part)/p50(whole).

MATCHING TURNS
--------------
The server's ``[finite-request]`` record maps each engine request to its
session. Logical turns are assigned by request arrival order within that
session. The logged ``turn`` field is deliberately not used: it reflects the
current retained-history depth and can decrease after context compaction.
Older logs without this record fall back to completion order. ``--expect``
asserts the count so a silent mismatch cannot pass.

CAVEAT recorded on purpose: stage 1 and stage 2 report num_tokens_in = 0. The talker's
placeholder prompt (prompt_token_ids=[0]*prompt_len, qwen3_omni.py:710) is evidently not
counted here, so stage-1 prompt length is NOT directly observable from this log. Stage 0's
count is real. Do not quietly substitute one for the other.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics as st
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAGE_HDR = re.compile(r"(\[stats\.py:\d+\]).*\[StageRequestStats \[request_id=([^\]]+)\]\]")
TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)")
BOOT = re.compile(r"Initializing a V1 LLM engine")
ROW = re.compile(r"^\s*\|\s*([a-z0-9_]+)\s*\|(.+)\|\s*$")
STAGE_COLUMNS = re.compile(r"^\s*\|\s*Field\s*\|(.+)\|\s*$")
FINITE_REQUEST = re.compile(r"\[finite-request\]\s+session=(\S+)\s+request=(\S+)\s+turn=(\d+)")
NIXL_DELTA = re.compile(
    r"\[nixl-delta-push\]\s+request=(\S+)\s+prefix_tokens=(\d+)"
    r"(?:\s+source_blocks=(\d+)\s+delta_blocks=(\d+)|\s+full_hit=true)"
)

# Fields worth keeping. Values that look like "13.490 (n=51)" are reduced to the number.
KEEP = {
    "num_tokens_in",
    "num_tokens_out",
    "vllm_ttft_ms",
    "vllm_queue_ms",
    "vllm_prefill_ms",
    "vllm_tpot_ms",
    "serving_time_to_first_output_ms",
    "stage_gen_time_ms",
    "inter_output_latency_ms",
    "audio_duration_s",
    "batch_size",
    "output_unit_count",
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
    finite_requests: dict[str, tuple[str, int]] = {}
    finite_request_counts: dict[str, int] = {}
    nixl_delta: dict[str, tuple[int, int | None, int]] = {}
    cur: dict | None = None
    boot = 0
    for raw in path.open(errors="replace"):
        line = ANSI.sub("", raw).rstrip()
        finite_match = FINITE_REQUEST.search(line)
        if finite_match:
            session = finite_match.group(1)
            logical_turn = finite_request_counts.get(session, 0)
            finite_request_counts[session] = logical_turn + 1
            finite_requests[finite_match.group(2)] = (session, logical_turn)
        delta_match = NIXL_DELTA.search(line)
        if delta_match:
            nixl_delta[delta_match.group(1)] = (
                int(delta_match.group(2)),
                int(delta_match.group(3)) if delta_match.group(3) is not None else None,
                int(delta_match.group(4)) if delta_match.group(4) is not None else 0,
            )
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
        columns_match = STAGE_COLUMNS.match(body)
        if columns_match:
            try:
                cur["stage_ids"] = [int(cell.strip()) for cell in columns_match.group(1).split("|")]
            except ValueError:
                cur["stage_ids"] = None
            continue
        rm = ROW.match(body)
        if not rm:
            continue
        field = rm.group(1)
        if field not in KEEP:
            continue
        vals = [_num(c) for c in rm.group(2).split("|")]
        stage_ids = cur.get("stage_ids")
        if not stage_ids or len(stage_ids) != len(vals):
            stage_ids = list(range(len(vals)))
        for stage_id, value in zip(stage_ids, vals):
            cur["stages"].setdefault(stage_id, {})[field] = value
    # Drop incomplete tables, then attach the explicit multi-user turn mapping.
    complete = [r for r in out if len(r["stages"]) >= 2]
    for record in complete:
        base_request_id = record["request_id"].rsplit("-", 1)[0]
        mapped = finite_requests.get(base_request_id)
        if mapped is not None:
            record["session"], record["turn"] = mapped
        delta = nixl_delta.get(record["request_id"])
        if delta is not None:
            record["kv_prefix_tokens"], record["kv_source_blocks"], record["kv_delta_blocks"] = delta
    return complete


def derive(recs: list[dict]) -> list[dict]:
    """Per-request derived quantities. All deltas computed WITHIN the request."""
    rows = []
    for r in recs:
        s = r["stages"]

        def g(i, k):
            return (s.get(i) or {}).get(k)

        t0, t1 = g(0, "vllm_ttft_ms"), g(1, "vllm_ttft_ms")
        if t0 is None or t1 is None:
            continue
        is_pd = 3 in s
        # serving_time_to_first_output_ms is measured from the original request
        # timestamp at every stage, so stage-to-stage differences are meaningful.
        # vllm_ttft_ms can be local to a restarted/resumable segment and must not
        # be subtracted across stages.
        c0 = g(0, "serving_time_to_first_output_ms")
        c1 = g(1, "serving_time_to_first_output_ms")
        c2 = g(2, "serving_time_to_first_output_ms")
        c3 = g(3, "serving_time_to_first_output_ms") if is_pd else None
        if c0 is None or c1 is None:
            continue
        row = {
            "request_id": r["request_id"],
            "t": r["t"],
            "boot": r["boot"],
            "session": r.get("session"),
            "turn": r.get("turn"),
            "is_pd": is_pd,
            "prompt_tokens": g(0, "num_tokens_in"),
            "kv_prefix_tokens": r.get("kv_prefix_tokens"),
            "kv_source_blocks": r.get("kv_source_blocks"),
            "kv_delta_blocks": r.get("kv_delta_blocks"),
            "text_tokens_out": g(1 if is_pd else 0, "num_tokens_out"),
            "codec_tokens_out": g(2 if is_pd else 1, "num_tokens_out"),
            "prefill_ttft_ms": c0 if is_pd else None,
            "prefill_queue_ms": g(0, "vllm_queue_ms") if is_pd else None,
            # vllm_prefill_ms is scheduled_ts -> first EngineCore output. It
            # includes model execution and runner-side output preparation; it
            # is not a CUDA-only compute timer.
            "prefill_execute_ms": g(0, "vllm_prefill_ms") if is_pd else None,
            "thinker_ttft_ms": c1 if is_pd else c0,
            "thinker_add_ms": (c1 - c0) if is_pd else c0,
            "thinker_queue_ms": g(1 if is_pd else 0, "vllm_queue_ms"),
            "thinker_prefill_ms": g(1 if is_pd else 0, "vllm_prefill_ms"),
            "talker_queue_ms": g(2 if is_pd else 1, "vllm_queue_ms"),
            "talker_prefill_ms": g(2 if is_pd else 1, "vllm_prefill_ms"),
            "code2wav_queue_ms": g(3 if is_pd else 2, "vllm_queue_ms"),
            "code2wav_prefill_ms": g(3 if is_pd else 2, "vllm_prefill_ms"),
            "talker_add_ms": ((c2 - c1) if is_pd and c2 is not None else (c1 - c0) if not is_pd else None),
            "code2wav_add_ms": (
                (c3 - c2) if is_pd and c3 is not None and c2 is not None else (c2 - c1) if c2 is not None else None
            ),
            "post_thinker_add_ms": (c3 - c1) if is_pd and c3 is not None else (c2 - c0) if c2 is not None else None,
            "audio_ttfa_ms": c3 if is_pd else c2,
            "thinker_itl_ms": g(1 if is_pd else 0, "inter_output_latency_ms"),
            "talker_itl_ms": g(2 if is_pd else 1, "inter_output_latency_ms"),
            "audio_s": g(3 if is_pd else 2, "audio_duration_s"),
        }
        if row["audio_ttfa_ms"]:
            total = row["audio_ttfa_ms"]
            row["thinker_share"] = row["thinker_add_ms"] / total
            if row["talker_add_ms"] is not None:
                row["talker_share"] = row["talker_add_ms"] / total
            if row["code2wav_add_ms"] is not None:
                row["code2wav_share"] = row["code2wav_add_ms"] / total
        rows.append(row)
    return rows


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    # Match the benchmark summary contract: nearest-rank percentile.
    i = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
    return xs[i]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, action="append", help="server log; repeatable. Use LABEL=PATH to name it.")
    ap.add_argument("--skip", type=int, default=1, help="drop the first N requests of each boot (turn 1 is cold)")
    ap.add_argument("--expect", type=int, default=None, help="assert this many usable requests per label")
    ap.add_argument("--bins", default="1-9,10-19,20-29,30-39,40-49,50-59")
    ap.add_argument("--out", default=None, help="write per-request rows as jsonl")
    ap.add_argument(
        "--split-boot",
        action="store_true",
        help="one label per server boot. REQUIRED whenever a log holds more "
        "than one arm, which is the normal case: run_fix_tuned.sh boots "
        "twice into the same log file. Lumping two arms together makes "
        "the by-turn table meaningless.",
    )
    ap.add_argument(
        "--split-gap-s",
        type=float,
        default=None,
        help="also split a boot wherever consecutive requests are more than "
        "this many seconds apart. Used to separate two arms that shared "
        "one boot (run_fix_latency_rest.sh runs low then static on one "
        "server, with the decompose step in between).",
    )
    ap.add_argument(
        "--tok-max",
        type=float,
        default=None,
        help="restrict the regression to requests below this prompt-token "
        "count. Needed because an arm that hits the context cap piles up "
        "many points at one x value and flattens the fit.",
    )
    ap.add_argument(
        "--request-regex",
        default=None,
        help="keep only request IDs matching this regex, e.g. '^video-(?!warm-)' for foreground AV turns",
    )
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

        print(f"{label}: {len(rows)} requests parsed over {len(by_boot)} boot(s), {len(groups)} group(s)")
        for boot, k, seg in groups:
            kept = []
            for i, r in enumerate(seg):
                turn = r.get("turn")
                if (turn is not None and turn < args.skip) or (turn is None and i < args.skip):
                    continue
                r["idx"] = turn if turn is not None else i
                kept.append(r)
            if args.split_boot or args.split_gap_s is not None:
                sub = f"{label}#b{boot}" + (f".{k}" if args.split_gap_s is not None else "")
            else:
                sub = label
            allrows.setdefault(sub, []).extend(kept)
            print(
                f"    {sub}: {len(seg)} requests, {len(kept)} kept, "
                f"{seg[0]['t']} .. {seg[-1]['t']}, "
                f"prompt {min(r['prompt_tokens'] or 0 for r in seg):.0f}"
                f"-{max(r['prompt_tokens'] or 0 for r in seg):.0f} tok"
            )
            if args.expect is not None and len(kept) != args.expect:
                print(f"    !! expected {args.expect}, got {len(kept)} -- turn matching NOT safe")

    print()
    print("=== per-request medians (deltas computed within a request, then medianed) ===")
    hdr = (
        f"{'label':22s} {'n':>4s} {'prompt_tok':>10s} {'P':>8s} {'thinker+':>8s} {'talker':>8s} "
        f"{'code2wav':>8s} {'TTFA':>7s} {'TTFA p95':>8s} {'talker%':>7s}"
    )
    print(hdr)
    for label, rows in allrows.items():
        if not rows:
            continue
        print(
            f"{label:22s} {len(rows):4d} {med(r['prompt_tokens'] for r in rows):10.0f} "
            f"{med(r['prefill_ttft_ms'] for r in rows):8.0f} "
            f"{med(r['thinker_add_ms'] for r in rows):8.0f} "
            f"{med(r['talker_add_ms'] for r in rows):8.0f} "
            f"{med(r['code2wav_add_ms'] for r in rows):8.0f} "
            f"{med(r['audio_ttfa_ms'] for r in rows):7.0f} "
            f"{q([r['audio_ttfa_ms'] for r in rows], 95):8.0f} "
            f"{100 * med(r['talker_share'] for r in rows if 'talker_share' in r):6.1f}%"
        )

    print()
    print("=== by turn index: prompt tokens and the talker's own added latency ===")
    for label, rows in allrows.items():
        if not rows:
            continue
        print(f"\n  {label}")
        print(
            f"    {'turns':>9s} {'n':>4s} {'prompt_tok':>10s} {'P':>8s} {'thinker+':>8s} "
            f"{'talker':>8s} {'code2wav':>8s} {'TTFA':>7s} {'text_out':>8s}"
        )
        for a, b in bins:
            rs = [r for r in rows if a <= r["idx"] <= b]
            if not rs:
                continue
            print(
                f"    {f'{a}-{b}':>9s} {len(rs):4d} "
                f"{med(r['prompt_tokens'] for r in rs):10.0f} "
                f"{med(r['prefill_ttft_ms'] for r in rs):8.0f} "
                f"{med(r['thinker_add_ms'] for r in rs):8.0f} "
                f"{med(r['talker_add_ms'] for r in rs):8.0f} "
                f"{med(r['code2wav_add_ms'] for r in rs):8.0f} "
                f"{med(r['audio_ttfa_ms'] for r in rs):7.0f} "
                f"{med(r['text_tokens_out'] for r in rs):8.0f}"
            )

    # The crux regression: talker's added latency against the REAL prompt token count.
    print()
    print("=== talker added latency vs prompt tokens (OLS, per-request points) ===")
    print("    slope is ms per 1,000 prompt tokens; intercept is the fixed floor")
    for label, rows in allrows.items():
        pts = [
            (r["prompt_tokens"], r["talker_add_ms"])
            for r in rows
            if r["prompt_tokens"]
            and r["talker_add_ms"] is not None
            and (args.tok_max is None or r["prompt_tokens"] <= args.tok_max)
        ]
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
        r2 = (sxy**2) / (sxx * syy) if syy else float("nan")
        print(
            f"    {label:22s} n={n:4d}  {1000 * slope:8.2f} ms/1k  "
            f"floor {icpt:8.1f} ms  R2 {r2:.4f}  "
            f"prompt {min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f} tok"
        )

    if args.out:
        with open(args.out, "w") as fh:
            for label, rows in allrows.items():
                for r in rows:
                    fh.write(json.dumps({"label": label, **r}) + "\n")
        print(f"\nwrote per-request rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
