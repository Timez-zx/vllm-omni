#!/usr/bin/env python3
"""Figures for the long-session memory study.

L1  recall vs policy          -- does it remember (the point of the change)
L2  context growth per turn    -- what memory costs in the window
L3  TTFA vs turn index         -- what memory costs on the critical path
L4  where the memory-note call lands relative to the turn it belongs to
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the report's parsing rather than re-deriving it. Two copies would drift,
# and the parts that matter here are exactly the parts that were subtle: the
# wall-clock windowing (without it the recall and latency phases concatenate and
# the context slope comes out negative) and the parity split with its
# monotonicity check.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from longmem_report import parse_log, split_series, trace_bounds  # noqa: E402

COLORS = {"shipped": "#888888", "full_text": "#1f77b4",
          "full_mm": "#d62728", "text_memory": "#2ca02c",
          "text_memory+hint": "#17a02c"}
ORDER = ["shipped", "full_text", "full_mm", "text_memory", "text_memory+hint"]


def arm_log(res: pathlib.Path, tag: str) -> dict:
    """Latency-phase view of one arm's server log (plus its notes from both)."""
    logp = res / f"server_{tag}.log"
    w_rec = trace_bounds(res / f"longmem_{tag}" / "recall.jsonl")
    w_lat = trace_bounds(res / f"longmem_lat_{tag}" / "ttfa_user0.jsonl")
    lat = parse_log(logp, w_lat) if w_lat else {}
    rec = parse_log(logp, w_rec) if w_rec else {}
    src = lat if lat.get("tokens_in") else rec
    return {
        "tokens": src.get("tokens_in", []),
        "n_notes_in_src": len(src.get("notes", [])),
        "ttfa": [t["fa"] * 1000 for t in src.get("timings", [])],
        "notes": [n["dur_ms"] for n in
                  (rec.get("notes", []) + lat.get("notes", []))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--outdir", default="/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/figures")
    ap.add_argument("--window", type=int, default=65536)
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    arms: dict[str, dict] = {}
    # Two directories must be excluded, for different reasons:
    #   longmem_lat_*   the latency bench's traces, not a recall score
    #   *smoke*         the first smoke run, which is CONFOUNDED and must never
    #                   be plotted: it ran with EVS on, so the rolling buffer
    #                   held four scenes at once and the model scored 3/3 by
    #                   reading them off the screen rather than remembering
    #                   (report section 7.1). The data is kept as evidence of
    #                   the confound, not as a result.
    for sc in sorted(glob.glob(str(res / "longmem_*" / "recall_score.json"))):
        if "/longmem_lat_" in sc or "smoke" in sc:
            continue
        d = json.loads(pathlib.Path(sc).read_text())
        pol = d.get("policy", "?")
        arms.setdefault(pol, {})["recall"] = d
    for lg in sorted(glob.glob(str(res / "server_lm_*.log"))):
        tag = pathlib.Path(lg).stem.replace("server_", "")
        pol = tag.replace("lm_", "")
        arms.setdefault(pol, {})["log"] = arm_log(res, tag)
    if not arms:
        print("no longmem arms found")
        return 1
    pols = [p for p in ORDER if p in arms] + [p for p in arms if p not in ORDER]

    # ---- L1: recall ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = range(len(pols))
    first, lst = [], []
    for p in pols:
        r = (arms[p].get("recall") or {}).get("score", {}) or {}
        pf, pl = r.get("probe_first") or {}, r.get("probe_listall") or {}
        first.append(1.0 if pf.get("correct") else 0.0)
        lst.append(pl.get("frac", 0.0))
    ax[0].bar(xs, first, color=[COLORS.get(p, "#666") for p in pols])
    ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(pols, rotation=15)
    ax[0].set_ylim(0, 1.15); ax[0].set_ylabel("correct")
    ax[0].set_title("probe: what word was on the FIRST screen?")
    for i, v in enumerate(first):
        ax[0].text(i, v + 0.03, "CORRECT" if v else "wrong", ha="center", fontsize=9)
    ax[1].bar(xs, lst, color=[COLORS.get(p, "#666") for p in pols])
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(pols, rotation=15)
    ax[1].set_ylim(0, 1.15); ax[1].set_ylabel("fraction of words recalled")
    ax[1].set_title("probe: list every word seen so far")
    for i, v in enumerate(lst):
        r = (arms[pols[i]].get("recall") or {}).get("score", {}) or {}
        pl = r.get("probe_listall") or {}
        ax[1].text(i, v + 0.03, f"{pl.get('n_recalled','-')}/{pl.get('n_total','-')}",
                   ha="center", fontsize=9)
    fig.suptitle("L1  Does the model remember earlier turns of the session?")
    fig.tight_layout(); fig.savefig(out / "figL1_recall.png", dpi=140)
    plt.close(fig)

    # ---- L2: context growth -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for p in pols:
        lg = arms[p].get("log") or {}
        g, _, _how = split_series(lg.get("tokens", []), lg.get("n_notes_in_src", 0))
        if not g:
            continue
        ax.plot(range(len(g)), g, "o-", label=p, color=COLORS.get(p, "#666"), ms=4)
    ax.axhline(args.window, color="k", ls="--", lw=1)
    ax.text(0.02, args.window * 0.965, f"model context window = {args.window:,}",
            transform=ax.get_yaxis_transform(), fontsize=8, va="top")
    ax.set_xlabel("turn index"); ax.set_ylabel("prompt tokens for the turn")
    ax.set_title("L2  What memory costs inside the context window")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "figL2_context_growth.png", dpi=140)
    plt.close(fig)

    # ---- L3: TTFA vs turn ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for p in pols:
        v = (arms[p].get("log") or {}).get("ttfa", [])
        v = [x for x in v if x > 0]
        if not v:
            continue
        ax.plot(range(len(v)), v, "o-", label=p, color=COLORS.get(p, "#666"), ms=4)
    ax.axhline(500, color="k", ls=":", lw=1)
    ax.text(0.02, 510, "~500 ms: lag becomes noticeable",
            transform=ax.get_yaxis_transform(), fontsize=8)
    ax.set_xlabel("turn index"); ax.set_ylabel("time to first audio (ms)")
    ax.set_title("L3  What memory costs on the critical path")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "figL3_ttfa_vs_turn.png", dpi=140)
    plt.close(fig)

    # ---- L4: the memory-note call ------------------------------------------
    notes = (arms.get("text_memory", {}).get("log") or {}).get("notes", [])
    if notes:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(range(len(notes)), notes, "o-", color=COLORS["text_memory"], ms=5)
        ax.axhline(st.median(notes), color="k", ls="--", lw=1,
                   label=f"p50 = {st.median(notes):.0f} ms")
        ax.set_xlabel("note index (one per turn)")
        ax.set_ylabel("memory-note generation (ms)")
        ax.set_title("L4  Cost of writing the note (thinker only, off the critical path)")
        ax.legend(); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
        fig.tight_layout(); fig.savefig(out / "figL4_note_cost.png", dpi=140)
        plt.close(fig)

    made = sorted(p.name for p in out.glob("figL*.png"))
    print("wrote:", ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
