#!/usr/bin/env python3
"""Per-stage pipeline timing, straight from the server's own StageRequestStats.

The engine already prints, per request, a table with one column per stage:

    | serving_time_to_first_output_ms |  1,102.500 |  1,474.814 |  1,710.467 |
    | inter_output_latencies_ms       | 21.060 (n) | 17.678 (n) | 432.669 (n)|
    | num_tokens_in / num_tokens_out  | ...        | ...        | ...        |

That is a decomposition the client cannot see. `serving_time_to_first_output_ms`
gives the moment each of the three stages produced its first output, so the
first-text -> first-sound gap splits into "talker start" and "code2wav start"
without any new instrumentation. `inter_output_latency_ms` on stage 0 is the
thinker's per-token decode cost, which is the quantity that should grow with a
video-laden prompt.

TWO CORRECTIONS to earlier readings of this table, both found by re-deriving from
the log rather than trusting the first parse:

1. `serving_time_to_first_output_ms` IS PER TURN, not per session. An earlier caveat
   here claimed it described only the session's first turn. It does not: at the
   1-decimal precision this report prints, stage 0 shows 431 / 455 / 479 distinct
   values among the 480 tables of an arm (479 / 480 / 480 at full precision), and
   stage-0 values span 24-1,128 ms on static and 974-11,418 ms on high motion. Each
   turn is its own request as far as this field is concerned.

2. STAGE LAGS MUST BE MEDIANS OF PER-TURN DIFFERENCES, never differences of
   medians. Subtracting p50(stage2) - p50(stage0) on the high-motion arm gives
   3,794 ms; the median of the per-turn differences gives 2,581 ms, and the client
   and the [TIMING] line independently say 2,530-2,623 ms. The table and [TIMING]
   are in fact the SAME clock -- paired per turn they agree to 3 ms on both
   boundaries. The 1,200 ms "discrepancy" was entirely the estimator: on high motion
   the per-turn gap is anti-correlated with prefill time (r = -0.60, and the gap
   distribution is bimodal, p25 1,462 / p75 4,937 ms), so marginal medians are not
   subtractable. Static (+0.55) and low (+0.36) have tight gap distributions, which
   is why both conventions agreed there and the error stayed hidden.

   This is the same trap already documented in video_tail_decompose8.py, where
   differencing two medians manufactured a "31% client/server disagreement". It was
   reintroduced here. Hence per_turn_lags() below.

Cumulative fields (num_tokens_out, audio_duration_s, stage_gen_time_ms) DO
accumulate over the session, so their absolute values are per-session, not per-turn.
`inter_output_latency_ms` is a session average, and on a stalling arm that mean is a
poor summary -- the median inter-token gap on 8-user high motion is 34 ms against a
mean of 92 ms.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGTS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")
HDR = re.compile(r"\[StageRequestStats \[request_id=([^\]]+)\]\]")
ROW = re.compile(r"\|\s*([a-z_]+)\s*\|(.+)\|\s*$")
NUM = re.compile(r"-?[\d,]+\.?\d*")

FIELDS = ["serving_time_to_first_output_ms", "inter_output_latency_ms",
          "num_tokens_in", "num_tokens_out", "batch_size", "stage_gen_time_ms",
          "vllm_ttft_ms", "vllm_tpot_ms"]
STAGE = ["stage0 thinker", "stage1 talker", "stage2 code2wav"]


def cells(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for part in raw.split("|"):
        m = NUM.search(part)
        out.append(float(m.group(0).replace(",", "")) if m else None)
    return out


def parse(log: pathlib.Path) -> list[dict]:
    """Every StageRequestStats table, with its wall clock and request id."""
    tables: list[dict] = []
    cur: dict | None = None
    ref_year = datetime.datetime.now().year
    t_now = None
    for ln in ANSI.sub("", log.read_text(errors="ignore")).splitlines():
        m = LOGTS.search(ln)
        if m:
            mo, da, hh, mi, ss = (int(x) for x in m.groups())
            try:
                t_now = datetime.datetime(ref_year, mo, da, hh, mi, ss).timestamp()
            except ValueError:
                pass
        h = HDR.search(ln)
        if h:
            if cur:
                tables.append(cur)
            cur = {"rid": h.group(1), "t": t_now, "f": {}}
            continue
        if cur is None:
            continue
        r = ROW.search(ln)
        if r and r.group(1) in FIELDS:
            cur["f"][r.group(1)] = cells(r.group(2))
    if cur:
        tables.append(cur)
    return tables


def window(res: pathlib.Path, sub: str) -> tuple[float, float] | None:
    ws = []
    for p in (res / sub).glob("ttfa_user*.jsonl"):
        for line in p.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    ws.append(json.loads(line)["w"])
                except Exception:
                    pass
    return (min(ws) - 3, max(ws) + 3) if ws else None


def col(tabs: list[dict], field: str, i: int) -> list[float]:
    out = []
    for t in tabs:
        v = t["f"].get(field)
        if v and i < len(v) and v[i] is not None and v[i] > 0:
            out.append(v[i])
    return out


def p50(xs):
    return st.median(xs) if xs else None


def per_turn_lags(tabs: list[dict], field: str) -> tuple[float | None, float | None]:
    """Median of the PER-TURN stage-to-stage differences.

    Not p50(stage1) - p50(stage0). Those differ by 1,200 ms on the high-motion arm
    because the per-turn gap is anti-correlated with prefill time, so the marginal
    medians come from different turns. See the module docstring.
    """
    l1, l2 = [], []
    for t in tabs:
        v = t["f"].get(field)
        if not v or len(v) < 3:
            continue
        s0, s1, s2 = v[0], v[1], v[2]
        if s0 and s1 and s1 > 0 and s0 > 0:
            l1.append(s1 - s0)
        if s1 and s2 and s2 > 0 and s1 > 0:
            l2.append(s2 - s1)
    return (p50(l1), p50(l2))


def ms(x, w=12):
    return f"{'-':>{w}}" if x is None else f"{x:>{w},.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/data/zx/results/server_vt.log")
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--prefix", default="vt")
    ap.add_argument("--arms", default="u8_static,u8_low,u8_high")
    ap.add_argument("--out", default="/data/zx/results/stage_stats.json")
    a = ap.parse_args()

    res = pathlib.Path(a.results)
    tabs = parse(pathlib.Path(a.log))
    print(f"parsed {len(tabs)} StageRequestStats tables from {a.log}")
    if not tabs:
        return 1

    arms = {}
    for arm in a.arms.split(","):
        w = window(res, f"{a.prefix}_{arm}")
        if not w:
            print(f"  (no client traces for {arm}, skipped)")
            continue
        sel = [t for t in tabs if t["t"] and w[0] <= t["t"] <= w[1]]
        if sel:
            arms[arm] = sel
    if not arms:
        print("no tables fell inside any arm window")
        return 1

    print("\n" + "=" * 100)
    print("WHEN EACH STAGE PRODUCED ITS FIRST OUTPUT, per turn")
    print("=" * 100)
    print(f"{'arm':<12}{'n':>5}" + "".join(f"{s:>16}" for s in STAGE)
          + f"{'talker lag':>12}{'c2w lag':>10}{'BAD: dp50':>11}")
    for arm, sel in arms.items():
        v = [p50(col(sel, "serving_time_to_first_output_ms", i)) for i in range(3)]
        lag1, lag2 = per_turn_lags(sel, "serving_time_to_first_output_ms")
        bad = (v[2] - v[0]) if (v[0] and v[2]) else None      # the wrong estimator
        good = (lag1 + lag2) if (lag1 and lag2) else None
        print(f"{arm:<12}{len(sel):>5}" + "".join(ms(x, 16) for x in v)
              + ms(lag1, 12) + ms(lag2, 10) + ms(bad, 11))
        _ = good
    print("  talker lag / c2w lag: MEDIAN OF PER-TURN DIFFERENCES, the correct estimator.")
    print("  'BAD: dp50' is p50(stage2) - p50(stage0), shown only to expose how far the")
    print("  difference-of-medians shortcut strays: it is right on static and low and")
    print("  overstates the high-motion arm by ~1.2 s. Never quote that column.")
    print("  NOTE the stages OVERLAP -- the talker starts while the thinker is still")
    print("  generating -- so these lags are not additive costs of serial work.")

    print("\n" + "=" * 92)
    print("PER-OUTPUT LATENCY, session average -- the thinker's decode cost is the")
    print("quantity a video-heavy prompt should inflate")
    print("=" * 92)
    print(f"{'arm':<12}{'prompt tok':>12}" + "".join(f"{s:>18}" for s in STAGE))
    for arm, sel in arms.items():
        pt = p50(col(sel, "num_tokens_in", 0))
        v = [p50(col(sel, "inter_output_latency_ms", i)) for i in range(3)]
        print(f"{arm:<12}{ms(pt, 12)}" + "".join(ms(x, 18) for x in v))

    print("\n--- thinker decode cost vs prompt length (the causal link, if it holds) ---")
    rows = []
    for arm, sel in arms.items():
        pt = p50(col(sel, "num_tokens_in", 0))
        it = p50(col(sel, "inter_output_latency_ms", 0))
        if pt and it:
            rows.append((arm, pt, it))
    if len(rows) >= 2:
        rows.sort(key=lambda r: r[1])
        base = rows[0]
        print(f"{'arm':<12}{'prompt tok':>12}{'ms/token':>11}{'vs smallest':>13}"
              f"{'us per 1k ctx':>15}")
        for arm, pt, it in rows:
            d = ((it - base[2]) / ((pt - base[1]) / 1000.0) * 1000.0) \
                if pt != base[1] else None
            print(f"{arm:<12}{pt:>12,.0f}{it:>11.2f}{it/base[2]:>12.2f}x"
                  + (f"{d:>15.1f}" if d is not None else f"{'-':>15}"))
        print("  last column: extra microseconds per generated token for every 1,000")
        print("  tokens of context. A flat value across arms means one mechanism.")

    print("\n--- batch size actually achieved per stage ---")
    print(f"{'arm':<12}" + "".join(f"{s:>18}" for s in STAGE))
    for arm, sel in arms.items():
        v = [p50(col(sel, "batch_size", i)) for i in range(3)]
        print(f"{arm:<12}" + "".join(ms(x, 18) for x in v))
    print("  stage-0 max_num_seqs is 16 and max_num_batched_tokens 16,384, so a")
    print("  14,500-token request can only share a batch with very small ones.")

    pathlib.Path(a.out).write_text(json.dumps(
        {arm: {f: [p50(col(sel, f, i)) for i in range(3)] for f in FIELDS}
         for arm, sel in arms.items()}, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
