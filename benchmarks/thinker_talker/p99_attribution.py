#!/usr/bin/env python3
"""Where does the p99 TTFA increase come from as audio-only users scale?

Input: cell directories from run_p99_ladder.sh (turns.jsonl per cell, with the
absolute per-turn stamps t_q/t_ft/t_fa/t_done that mu_bench.py records).

Four decompositions, each aimed at one hypothesis:

  1. STAGE SPLIT -- per turn, thinker side = ttft (query -> first text) and
     speech side = ttfa - ttft (first text -> first audio). For the turns in
     the tail (>= p95 / p99 of the cell), the excess over the cell median is
     split within each turn and only then averaged (never median-minus-median;
     the estimator rule from stage_stats_v2.py).

  2. ARRIVAL STATE -- at each turn's t_q, count OTHER turns that were mid-TTFA
     (t_q' <= t_q < t_fa': the engine owes them a first sound) and mid-DELIVERY
     (t_fa' <= t_q < t_done': the speech stages are streaming for them). TTFA
     bucketed by those counts separates "queued behind others" from
     "everything got slower". Valid because every user shares one driver
     process and therefore one monotonic clock.

  3. TURN INDEX -- session mode grows the context every turn, so a
     context-cost story predicts late turns slower than early ones at the
     same user count. Buckets of 10.

  4. HYGIENE -- timeouts, session rolls, bad probes per cell (from
     summary.json), so a pathological cell cannot masquerade as a scaling law.

    p99_attribution.py --cells '/data/zx/results/p99aud_none_u*'
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_cell(d: pathlib.Path, warmup_turns: int = 0) -> dict:
    recs = []
    with (d / "turns.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    m = re.search(r"_u(\d+)$", d.name)
    return {"dir": d, "users": int(m.group(1)) if m else summary.get("users", 0),
            "recs": recs, "summary": summary, "warmup_turns": warmup_turns}


def arrival_state(recs: list[dict]) -> None:
    """Annotate each ok/timeout record with the system state at its arrival."""
    stamped = [r for r in recs if r.get("t_q") is not None]
    for r in stamped:
        n_wait = n_stream = 0
        for o in stamped:
            if o is r or o["user"] == r["user"]:
                continue
            tq, tfa, tdone = o["t_q"], o.get("t_fa"), o.get("t_done")
            # a timed-out turn with no first audio occupied the engine until
            # its timeout; treat its whole window as mid-TTFA
            if tq <= r["t_q"] and (tfa is None or r["t_q"] < tfa):
                if o.get("status") == "ok" or tfa is None:
                    n_wait += 1
            elif tfa is not None and tfa <= r["t_q"] and (tdone is None or r["t_q"] < tdone):
                n_stream += 1
        r["n_wait"] = n_wait
        r["n_stream"] = n_stream


def cell_report(cell: dict) -> dict:
    # Warmup turns are excluded from the statistics but NOT from the contention
    # counts: a warmup turn still loads the engine for its neighbours, so
    # arrival_state() has already run over every record.
    warmup = cell.get("warmup_turns", 0)
    ok = [r for r in cell["recs"] if r.get("status") == "ok"
          and r.get("ttfa_ms") is not None and r.get("ttft_ms") is not None
          and r.get("turn", 0) > warmup]
    for r in ok:
        r["speech_ms"] = r["ttfa_ms"] - r["ttft_ms"]
    ttfa = [r["ttfa_ms"] for r in ok]
    out = {
        "users": cell["users"], "n_ok": len(ok),
        "n_timeout": cell["summary"].get("n_timeout"),
        "rolls": cell["summary"].get("session_rolls"),
        "ttfa_p50": pct(ttfa, 50), "ttfa_p95": pct(ttfa, 95),
        "ttfa_p99": pct(ttfa, 99), "ttfa_max": max(ttfa) if ttfa else float("nan"),
        "ttft_p50": pct([r["ttft_ms"] for r in ok], 50),
        "ttft_p99": pct([r["ttft_ms"] for r in ok], 99),
        "speech_p50": pct([r["speech_ms"] for r in ok], 50),
        "speech_p99": pct([r["speech_ms"] for r in ok], 99),
    }

    # 1. stage split of the tail's excess, within-turn then averaged
    med_ttfa, med_ttft = out["ttfa_p50"], out["ttft_p50"]
    med_speech = out["speech_p50"]
    for tail_name, thresh in (("p95", out["ttfa_p95"]), ("p99", out["ttfa_p99"])):
        tail = [r for r in ok if r["ttfa_ms"] >= thresh]
        if not tail:
            continue
        exc_tot = st.mean([r["ttfa_ms"] - med_ttfa for r in tail])
        exc_thk = st.mean([r["ttft_ms"] - med_ttft for r in tail])
        exc_spc = st.mean([r["speech_ms"] - med_speech for r in tail])
        out[f"tail_{tail_name}"] = {
            "n": len(tail), "excess_ms": round(exc_tot, 1),
            "thinker_share": round(exc_thk / exc_tot, 3) if exc_tot else None,
            "speech_share": round(exc_spc / exc_tot, 3) if exc_tot else None,
        }

    # 2. arrival state
    arrival_state(cell["recs"])
    by_wait: dict[int, list[float]] = {}
    for r in ok:
        by_wait.setdefault(r["n_wait"], []).append(r["ttfa_ms"])
    out["ttfa_by_n_wait"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)}
        for k, v in sorted(by_wait.items())}
    by_stream: dict[int, list[float]] = {}
    for r in ok:
        if r["n_wait"] == 0:  # isolate the streaming effect from the queueing one
            by_stream.setdefault(min(r["n_stream"], 8), []).append(r["ttfa_ms"])
    out["ttfa_by_n_stream_at_wait0"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)}
        for k, v in sorted(by_stream.items())}
    tail95 = [r for r in ok if r["ttfa_ms"] >= out["ttfa_p95"]]
    out["n_wait_mean_all"] = round(st.mean([r["n_wait"] for r in ok]), 2) if ok else None
    out["n_wait_mean_tail95"] = round(st.mean([r["n_wait"] for r in tail95]), 2) if tail95 else None
    out["n_stream_mean_all"] = round(st.mean([r["n_stream"] for r in ok]), 2) if ok else None
    out["n_stream_mean_tail95"] = round(st.mean([r["n_stream"] for r in tail95]), 2) if tail95 else None

    # 3. turn-index buckets (context growth inside the long session)
    buckets: dict[str, list[float]] = {}
    for r in ok:
        b = f"{((r['turn'] - 1) // 10) * 10 + 1}-{((r['turn'] - 1) // 10) * 10 + 10}"
        buckets.setdefault(b, []).append(r["ttfa_ms"])
    out["ttfa_by_turn_bucket"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)}
        for k, v in sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))}

    # 4. hygiene
    probes = cell["summary"].get("engine_probes", {})
    out["bad_probes"] = {k: v for k, v in probes.items() if v and k in
                         ("unowned_audio", "torch_cat_error", "counter_leak_clamped",
                          "zero_output_wedge", "negative_slice", "preempted_reqs")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="glob of cell directories")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--warmup-turns", type=int, default=2,
                    help="drop the first N turns of each session from the "
                         "statistics (default 2, matching analyze.py)")
    args = ap.parse_args()

    cells = [load_cell(pathlib.Path(p), args.warmup_turns)
             for p in sorted(glob.glob(args.cells))
             if (pathlib.Path(p) / "turns.jsonl").exists()]
    cells.sort(key=lambda c: c["users"])
    reports = [cell_report(c) for c in cells]

    hdr = (f"{'users':>5} {'n':>4} {'p50':>6} {'p95':>7} {'p99':>7} "
           f"{'thk p50/p99':>12} {'spc p50/p99':>12} "
           f"{'tail95 thk/spc':>14} {'wait all/tail':>13} {'to':>3} {'roll':>4}")
    print(hdr)
    for r in reports:
        t95 = r.get("tail_p95", {})
        share = (f"{t95.get('thinker_share', float('nan')):.2f}/"
                 f"{t95.get('speech_share', float('nan')):.2f}") if t95 else "-"
        print(f"{r['users']:>5} {r['n_ok']:>4} {r['ttfa_p50']:>6.0f} {r['ttfa_p95']:>7.0f} "
              f"{r['ttfa_p99']:>7.0f} "
              f"{r['ttft_p50']:>5.0f}/{r['ttft_p99']:>5.0f} "
              f"{r['speech_p50']:>5.0f}/{r['speech_p99']:>5.0f} "
              f"{share:>14} "
              f"{r['n_wait_mean_all']:>5.2f}/{r['n_wait_mean_tail95']:>5.2f} "
              f"{r['n_timeout'] or 0:>3} {r['rolls'] or 0:>4}")
        if r["bad_probes"]:
            print(f"      !! bad probes: {r['bad_probes']}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(reports, indent=1))
        print(f"\nfull detail -> {args.json_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
