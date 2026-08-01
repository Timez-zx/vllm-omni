#!/usr/bin/env python3
"""Figures for the multi-user memory-overhead study.

M1  latency segments, stacked, vs user count -- where TTFA goes and how the mix
    shifts under contention
M2  resource share per component, stacked, per arm
M3  TTFA p50/p95 vs users, both policies -- the overhead of memory
M4  GPU busy while the user is still speaking -- saturation point
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEGS = [
    ("admit_s", "admission", "#bbbbbb"),
    ("to_first_token_s", "encoders + prefill + 1st token", "#4c78a8"),
    ("to_first_audio_s", "talker spin-up + code2wav", "#e45756"),
]
COMPS = [
    ("vision_encoder", "vision encoder", "#54a24b"),
    ("audio_encoder", "audio encoder", "#88d27a"),
    ("memory_note", "memory-note call", "#ffbf79"),
    ("thinker_llm_residual", "thinker LLM (residual)", "#4c78a8"),
    ("talker", "talker", "#e45756"),
    ("code2wav", "code2wav", "#b279a2"),
]
POL_STYLE = {"shipped": ("#888888", "o--"), "text_memory": ("#2ca02c", "s-")}


def load_decomp(res: pathlib.Path, pol: str, u: int) -> dict | None:
    p = res / f"decomp_mu_{pol}_u{u}.json"
    return json.loads(p.read_text()) if p.exists() else None


def rows_of(d: dict) -> list[dict]:
    return [r for r in d.get("rows", []) if r.get("ttfa_end_s") is not None]


def p(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--outdir", default="/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/figures")
    ap.add_argument("--users", default="1,2,4")
    ap.add_argument("--policies", default="shipped,text_memory")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    users = [int(x) for x in args.users.split(",")]
    pols = args.policies.split(",")

    D: dict[tuple[str, int], dict] = {}
    for pol in pols:
        for u in users:
            d = load_decomp(res, pol, u)
            if d:
                D[(pol, u)] = d
    if not D:
        print("no decomp_mu_* found")
        return 1
    have_pols = [pl for pl in pols if any(k[0] == pl for k in D)]

    # ---- M1: stacked latency segments vs users ------------------------------
    fig, axes = plt.subplots(1, len(have_pols), figsize=(6.0 * len(have_pols), 4.6),
                             sharey=True, squeeze=False)
    for ax, pol in zip(axes[0], have_pols):
        us = [u for u in users if (pol, u) in D]
        bottoms = np.zeros(len(us))
        for key, lab, col in SEGS:
            vals = []
            for u in us:
                rs = rows_of(D[(pol, u)])
                v = p([r[key] for r in rs if r.get(key) is not None], .5)
                vals.append((v or 0) * 1000)
            ax.bar([str(u) for u in us], vals, bottom=bottoms, label=lab, color=col)
            bottoms += np.array(vals)
        for x, tot in enumerate(bottoms):
            ax.text(x, tot + 8, f"{tot:.0f} ms", ha="center", fontsize=9)
        ax.set_title(pol)
        ax.set_xlabel("concurrent users")
        ax.grid(axis="y", alpha=.3)
    axes[0][0].set_ylabel("TTFA, p50 (ms)")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("M1  Where TTFA goes, and how the mix shifts with contention")
    fig.tight_layout()
    fig.savefig(out / "figM1_latency_segments.png", dpi=140)
    plt.close(fig)

    # ---- M2: resource share per component ----------------------------------
    bp = res / "session_breakdown.json"
    if bp.exists():
        bd = json.loads(bp.read_text()).get("arms", {})
        if bd:
            tags = sorted(bd, key=lambda t: (bd[t].get("label", t)))
            fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(tags)), 4.8))
            bottoms = np.zeros(len(tags))
            for key, lab, col in COMPS:
                vals = np.array([bd[t].get("components_share_of_sum", {}).get(key, 0.0)
                                 for t in tags])
                ax.bar(range(len(tags)), vals, bottom=bottoms, label=lab, color=col)
                bottoms += vals
            ax.set_xticks(range(len(tags)))
            ax.set_xticklabels([bd[t].get("label", t) for t in tags],
                               rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("share of summed component GPU time")
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(axis="y", alpha=.3)
            ax.set_title("M2  Resource share per component over the whole session\n"
                         "(thinker LLM is a residual; see session_breakdown.py caveats)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "figM2_resource_share.png", dpi=140)
            plt.close(fig)

    # ---- M3: TTFA vs users, both policies ----------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for pol in have_pols:
        col, style = POL_STYLE.get(pol, ("#333", "^-"))
        us = [u for u in users if (pol, u) in D]
        for q, alpha, tag in ((.5, 1.0, "p50"), (.95, 0.45, "p95")):
            vals = [(p([r["ttfa_end_s"] for r in rows_of(D[(pol, u)])], q) or 0) * 1000
                    for u in us]
            ax.plot(us, vals, style, color=col, alpha=alpha, ms=6,
                    label=f"{pol} {tag}")
    ax.axhline(500, color="k", ls=":", lw=1)
    ax.text(users[0], 512, "~500 ms: lag becomes noticeable", fontsize=8)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("TTFA (ms)")
    ax.set_xticks(users)
    ax.set_title("M3  What memory costs as users contend")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "figM3_ttfa_vs_users.png", dpi=140)
    plt.close(fig)

    # ---- M4: saturation ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for pol in have_pols:
        col, style = POL_STYLE.get(pol, ("#333", "^-"))
        us = [u for u in users if (pol, u) in D]
        vals = []
        for u in us:
            rs = rows_of(D[(pol, u)])
            v = [r["gpu_busy_speech_frac"] for r in rs
                 if r.get("gpu_busy_speech_frac") is not None]
            vals.append(st.median(v) * 100 if v else 0)
        ax.plot(us, vals, style, color=col, ms=6, label=pol)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("GPU busy while the user is speaking (%)")
    ax.set_xticks(users)
    ax.set_ylim(-3, 103)
    ax.set_title("M4  Saturation: idle headroom during speech disappears")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "figM4_saturation.png", dpi=140)
    plt.close(fig)

    made = sorted(x.name for x in out.glob("figM*.png"))
    print("wrote:", ", ".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
