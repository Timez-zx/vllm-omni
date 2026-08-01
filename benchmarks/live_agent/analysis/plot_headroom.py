#!/usr/bin/env python3
"""Fig E5-3: decompose the perception-allocation gap into its two mechanisms.

Both mechanisms are free in GPU terms -- the model is shown exactly K frames
either way -- so any gain here is quality-per-frame, not capacity.

  gate gain  shipped / uniform-no-EVS      what the fixed pixel gate costs
  sel gain   uniform-no-EVS / emb-medoid   what content-aware selection adds
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"screencast": "#3B6FD4", "talkinghead": "#1F9E8E",
          "handheld_walk_talk": "#D4713B"}
LABEL = {"screencast": "screencast (static)",
         "talkinghead": "talking head (low motion)",
         "handheld_walk_talk": "handheld walk+talk (high motion)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/data/zx/results/frame_selection_headroom.json")
    ap.add_argument("--outdir", default="/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/figures")
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.inp).read_text())
    regimes = list(d["regimes"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for r in regimes:
        e = d["regimes"][r]
        ks = sorted(int(k) for k in e["budgets"])
        gate, sel = [], []
        for k in ks:
            b = e["budgets"][str(k)]
            sh = b["shipped_evs_then_uniform"]["coverage_error"]
            un = b["uniform_no_evs"]["coverage_error"]
            em = b["embed_greedy_kmedoid"]["coverage_error"]
            gate.append(sh / un if un else np.nan)
            sel.append(un / em if em else np.nan)
        c = COLORS.get(r, "#888")
        axes[0].plot(ks, gate, "o-", lw=1.9, ms=5, color=c, label=LABEL.get(r, r))
        axes[1].plot(ks, sel, "o-", lw=1.9, ms=5, color=c, label=LABEL.get(r, r))

    for ax, title, sub in (
        (axes[0], "Cost of the fixed EVS pixel gate",
         "shipped (EVS + uniform stride)  /  uniform stride, no gate"),
        (axes[1], "Additional gain from content-aware selection",
         "uniform stride, no gate  /  greedy k-medoid in CLIP space"),
    ):
        ax.axhline(1.0, color="#666", ls=":", lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32])
        ax.set_xticklabels(["4", "8", "16", "32"])
        ax.set_xlabel("frame budget K shown to the model")
        ax.set_ylabel("coverage-error ratio  (>1 = worse)")
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.legend(fontsize=8, framealpha=0.9)

    axes[0].annotate("gate helps here\n(crude keyframe detector\nbeats stride at tiny K)",
                     xy=(4, 0.82), xytext=(6.0, 1.9), fontsize=8, color="#1F9E8E",
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color="#1F9E8E", lw=1,
                                     connectionstyle="arc3,rad=-0.25"))
    fig.suptitle("Fig E5-3  Perception-allocation headroom at IDENTICAL frame budget "
                 "(free in GPU terms)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = pathlib.Path(args.outdir) / "figE5-3_headroom_decomposition.png"
    fig.savefig(p, dpi=160)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
