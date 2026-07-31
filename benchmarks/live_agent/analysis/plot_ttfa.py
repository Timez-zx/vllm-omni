#!/usr/bin/env python3
"""Figures for the TTFA study.

Fig T1  TTFA stacked breakdown vs concurrent users
Fig T2  TTFA p50/p95 vs users, with per-user spread
Fig T3  causal test: TTFA vs initial_codec_chunk_frames
Fig T4  where the GPU is busy: during the user's speech vs during the TTFA window
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEG = [
    ("admit_s", "admission", "#8C8C8C"),
    ("to_first_token_s", "encoders + vision prefill + 1st token", "#3B6FD4"),
    ("to_first_audio_s", "talker spin-up + code2wav", "#D4713B"),
]


def load(p: str) -> dict | None:
    q = pathlib.Path(p)
    return json.loads(q.read_text()) if q.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir", default="/data/zx/results")
    ap.add_argument("--outdir", default="/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/figures")
    ap.add_argument("--users", default="1,2,4,8")
    ap.add_argument("--icf", default="1,4,8")
    args = ap.parse_args()

    R = pathlib.Path(args.resdir)
    O = pathlib.Path(args.outdir)
    O.mkdir(parents=True, exist_ok=True)
    ulist = [int(x) for x in args.users.split(",")]

    cont = {u: load(str(R / f"ttfa_c{u}.json")) for u in ulist}
    cont = {u: v for u, v in cont.items() if v}

    # ---------------- Fig T1: stacked breakdown vs users ----------------
    if cont:
        us = sorted(cont)
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        bottom = np.zeros(len(us))
        for key, label, color in SEG:
            vals = np.array([cont[u]["aggregate"].get(key, {}).get("p50", 0) * 1000
                             for u in us])
            ax.bar([str(u) for u in us], vals, bottom=bottom, label=label,
                   color=color, width=0.62, edgecolor="white", linewidth=0.6)
            for x, (b, v) in enumerate(zip(bottom, vals)):
                if v > 22:
                    ax.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals
        # label with the MEASURED TTFA p50, not the sum of segment medians --
        # a median of sums is not the sum of medians, and the 1-5 ms gap would
        # otherwise look like an arithmetic error
        for x, u in enumerate(us):
            ttfa = cont[u]["aggregate"]["ttfa_end_s"]["p50"] * 1000
            ax.text(x, bottom[x] + 12, f"{ttfa:.0f} ms", ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_xlabel("concurrent users (identical workload each)")
        ax.set_ylabel("TTFA from end of user speech (ms, p50)")
        ax.set_title("Fig T1  TTFA decomposition vs contention\n"
                     "Qwen3-Omni-30B-A3B, 1x RTX PRO 6000 96GB, vllm-omni 0.24.0\n"
                     "(bars are per-segment medians; top label is the measured "
                     "TTFA median)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25, axis="y", lw=0.6)
        ax.set_ylim(0, bottom.max() * 1.22)
        fig.tight_layout()
        p = O / "figT1_ttfa_breakdown_vs_users.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"wrote {p}")

        # ---------------- Fig T2: p50/p95 + per-user spread ----------------
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        p50 = [cont[u]["aggregate"]["ttfa_end_s"]["p50"] * 1000 for u in us]
        p95 = [cont[u]["aggregate"]["ttfa_end_s"]["p95"] * 1000 for u in us]
        mx = [cont[u]["aggregate"]["ttfa_end_s"]["max"] * 1000 for u in us]
        ax.plot(us, p50, "o-", lw=2, ms=6, color="#3B6FD4", label="p50")
        ax.plot(us, p95, "s--", lw=1.6, ms=5, color="#D4713B", label="p95")
        ax.plot(us, mx, "^:", lw=1.2, ms=5, color="#C0392B", label="max")
        # per-repetition scatter, so per-user unfairness is visible
        for u in us:
            xs = [r["ttfa_end_s"] * 1000 for r in cont[u]["rows"]
                  if isinstance(r.get("ttfa_end_s"), (int, float))]
            ax.scatter([u] * len(xs), xs, s=9, alpha=0.28, color="#555",
                       zorder=1, linewidths=0)
        ax.axhline(500, color="#1F9E8E", ls=":", lw=1.4)
        ax.text(us[0], 515, "  500 ms: conversational latency starts to be felt",
                fontsize=8, color="#1F9E8E", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(us)
        ax.set_xticklabels([str(u) for u in us])
        ax.set_xlabel("concurrent users")
        ax.set_ylabel("TTFA from end of user speech (ms)")
        ax.set_title("Fig T2  TTFA under contention (dots = individual repetitions)",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        fig.tight_layout()
        p = O / "figT2_ttfa_vs_users.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"wrote {p}")

    # ---------------- Fig T3: causal codec-chunk test ----------------
    icf = {}
    for v in [int(x) for x in args.icf.split(",")]:
        tag = "u1" if v == 4 else f"icf{v}"
        d = load(str(R / f"ttfa_{tag}.json"))
        if d:
            icf[v] = d
    if len(icf) >= 2:
        ks = sorted(icf)
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        tal = [icf[k]["aggregate"]["to_first_audio_s"]["p50"] * 1000 for k in ks]
        tot = [icf[k]["aggregate"]["ttfa_end_s"]["p50"] * 1000 for k in ks]
        ax.plot(ks, tot, "o-", lw=2, ms=7, color="#3B6FD4", label="TTFA total")
        ax.plot(ks, tal, "s-", lw=2, ms=6, color="#D4713B",
                label="talker spin-up + code2wav segment")
        # linear fit on the segment -> ms of pure buffering per codec frame
        if len(ks) >= 2:
            m, b = np.polyfit(ks, tal, 1)
            xs = np.linspace(0, max(ks) + 0.5, 50)
            ax.plot(xs, m * xs + b, "--", lw=1.2, color="#D4713B", alpha=0.55)
            ax.text(0.05, 0.06,
                    f"segment slope = {m:.1f} ms per initial codec frame\n"
                    f"intercept at 0 frames = {b:.0f} ms  (irreducible compute)",
                    transform=ax.transAxes, fontsize=9, color="#D4713B",
                    va="bottom")
        ax.axvline(4, color="#666", ls=":", lw=1.2)
        ax.text(4.1, min(tal) * 0.85, "shipped default", fontsize=8, color="#666")
        ax.set_xlabel("initial_codec_chunk_frames  (SharedMemoryConnector)")
        ax.set_ylabel("latency (ms, p50)")
        ax.set_title("Fig T3  Causal test: first-audio latency is part buffering,\n"
                     "not all compute", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(0, max(ks) + 0.6)
        fig.tight_layout()
        p = O / "figT3_codec_chunk_causal.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"wrote {p}")

    # ---------------- Fig T4: GPU busy, speech window vs TTFA window ----------
    if cont:
        us = sorted(cont)
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        w = 0.36
        xs = np.arange(len(us))
        sp, tw = [], []
        for u in us:
            rows = cont[u]["rows"]
            s = [r.get("gpu_busy_speech_frac") for r in rows]
            t = [r.get("gpu_busy_ttfa_frac") for r in rows]
            s = [v for v in s if isinstance(v, (int, float))]
            t = [v for v in t if isinstance(v, (int, float))]
            sp.append(float(np.median(s)) * 100 if s else 0.0)
            tw.append(float(np.median(t)) * 100 if t else 0.0)
        ax.bar(xs - w / 2, sp, w, label="while the user is speaking", color="#8C8C8C")
        ax.bar(xs + w / 2, tw, w, label="during the TTFA window", color="#3B6FD4")
        for x, v in zip(xs - w / 2, sp):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
        for x, v in zip(xs + w / 2, tw):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(u) for u in us])
        ax.set_xlabel("concurrent users")
        ax.set_ylabel("device GPU-busy fraction (%)")
        ax.set_title("Fig T4  At 1 user the GPU is idle for the whole utterance.\n"
                     "Adding users fills that idle time -- and saturates it.",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y", lw=0.6)
        fig.tight_layout()
        p = O / "figT4_gpu_busy_windows.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
