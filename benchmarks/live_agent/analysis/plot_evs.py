#!/usr/bin/env python3
"""Figures for the EVS pre-filter probe.

Fig E5-1: drop rate vs similarity threshold, per motion regime.
Fig E5-2: distribution of consecutive-frame RMSE, with the shipped default
          threshold drawn on top -- shows where the default cut lands relative
          to the natural inter-frame difference of each regime.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Brand-neutral, colour-blind-safe categorical palette.
COLORS = {
    "screencast": "#3B6FD4",
    "talkinghead": "#1F9E8E",
    "handheld_walk_talk": "#D4713B",
}
LABEL = {
    "screencast": "screencast (static desktop)",
    "talkinghead": "talking head (low motion)",
    "handheld_walk_talk": "handheld walk+talk (high motion)",
}
DEFAULT_T = 0.95
RMSE_AT_DEFAULT = np.sqrt(1.0 - DEFAULT_T) * 255.0  # 57.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/data/zx/results/evs_probe.json")
    ap.add_argument("--outdir", default="/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/figures")
    ap.add_argument("--fps", default="4fps")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.inp).read_text())
    regimes = list(data["regimes"].keys())
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Fig E5-1: drop rate vs threshold ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, fk in zip(axes, ["1fps", "2fps", "4fps"]):
        for r in regimes:
            if fk not in data["regimes"][r]:
                continue
            d = data["regimes"][r][fk]["drop_rate_vs_threshold"]
            ts = sorted(float(k) for k in d)
            ys = [d[f"{t:g}"] * 100 for t in ts]
            # plot against 1-threshold on a log axis so the useful range is legible
            ax.plot([1 - t for t in ts], ys, "o-", ms=4, lw=1.8,
                    color=COLORS.get(r, "#888"), label=LABEL.get(r, r))
        ax.axvline(1 - DEFAULT_T, color="#C0392B", ls="--", lw=1.4)
        ax.text(1 - DEFAULT_T, 4, "  shipped\n  default\n  0.95", color="#C0392B",
                fontsize=8, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("1 - threshold   (left = keep more frames)")
        ax.set_title(f"client frame rate: {fk}")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("frames dropped by EVS (%)")
    axes[0].legend(fontsize=8, loc="center left", framealpha=0.9)
    fig.suptitle("Fig E5-1  EVS pixel-similarity pre-filter: drop rate vs threshold "
                 "(vllm-omni v0.24.0, verbatim port)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = outdir / "figE5-1_evs_droprate_vs_threshold.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---------------- Fig E5-2: where the default cut lands ----------------
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = np.arange(len(regimes))
    for i, r in enumerate(regimes):
        c = data["regimes"][r][args.fps]["consecutive"]
        lo, mid, hi = c["rmse_p50"], c["rmse_p90"], c["rmse_max"]
        col = COLORS.get(r, "#888")
        ax.plot([i, i], [lo, hi], color=col, lw=8, alpha=0.30,
                solid_capstyle="butt")
        ax.plot([i, i], [lo, mid], color=col, lw=8, alpha=0.65,
                solid_capstyle="butt")
        ax.plot(i, lo, "o", color=col, ms=9, zorder=3)
        # keep the two labels from colliding when p50 and p90 are close together
        span = max(c["rmse_max"], RMSE_AT_DEFAULT)
        dy_hi = 6 if (mid - lo) < 0.06 * span else -4
        ax.annotate(f"p50={lo:.1f}", (i, lo), textcoords="offset points",
                    xytext=(12, -10), fontsize=9, color=col)
        ax.annotate(f"p90={mid:.1f}", (i, mid), textcoords="offset points",
                    xytext=(12, dy_hi), fontsize=8, color=col, alpha=0.8)
    ax.axhline(RMSE_AT_DEFAULT, color="#C0392B", ls="--", lw=1.6)
    ax.text(len(regimes) - 0.45, RMSE_AT_DEFAULT + 1.5,
            f"shipped default threshold 0.95\n= drop everything below RMSE {RMSE_AT_DEFAULT:.0f}/255",
            color="#C0392B", fontsize=9, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(r, r).replace(" (", "\n(") for r in regimes], fontsize=9)
    ax.set_ylabel("consecutive-frame RMSE  (64x64 thumbnail, /255)")
    ax.set_title(f"Fig E5-2  One fixed pixel threshold cannot serve all regimes "
                 f"({args.fps} client rate)", fontsize=11)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.set_ylim(0, max(110, RMSE_AT_DEFAULT * 1.6))
    fig.tight_layout()
    p2 = outdir / "figE5-2_evs_threshold_vs_regime_spread.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    print(f"wrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
