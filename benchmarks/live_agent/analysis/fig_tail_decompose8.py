#!/usr/bin/env python3
"""Figures for the 8-user latency decomposition.

figV5  where the time goes: the three segments stacked, absolute and normalised
figV6  why: what the device was doing for OTHER users, and the effective
       concurrency the engine actually achieved
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/data/zx/results")
CONTENTS = ["static", "low", "high"]
NAME = {"static": "static screen\n(screencast)",
        "low": "person sitting\n(talking head)",
        "high": "person walking\n(handheld)"}
SEG = [("admit_s", "admission (queued)", "#b0b7c3"),
       ("to_first_token_s", "encode + prefill", "#d95f02"),
       ("to_first_audio_s", "speech ramp (talker+code2wav)", "#1b9e77")]

# measured in section 7 / 8 of video_tail_decompose8.py
COMPETE = {"static": (0.79, 3.80), "low": (2.81, 3.63), "high": (6.69, 3.58)}
EFFCONC = {"static": 0.63, "low": 1.19, "high": 2.06}


def main() -> int:
    d = json.loads((RES / "video_tail_decompose8.json").read_text())

    # ---------------------------------------------------------------- figV5
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    x = range(len(CONTENTS))
    for norm, a in zip((False, True), ax):
        bottom = [0.0] * len(CONTENTS)
        for key, lab, col in SEG:
            v = [d[c][key]["p50"] for c in CONTENTS]
            if norm:
                tot = [d[c]["ttfa_end_s"]["p50"] for c in CONTENTS]
                v = [100.0 * a_ / b for a_, b in zip(v, tot)]
            a.bar(x, v, 0.6, bottom=bottom, label=lab, color=col,
                  edgecolor="white", linewidth=0.7)
            for i, (val, bot) in enumerate(zip(v, bottom)):
                if val > (4 if norm else 180):
                    a.text(i, bot + val / 2,
                           f"{val:.0f}%" if norm else f"{val:.0f}",
                           ha="center", va="center", fontsize=9,
                           color="white", fontweight="bold")
            bottom = [b + val for b, val in zip(bottom, v)]
        a.set_xticks(list(x))
        a.set_xticklabels([NAME[c] for c in CONTENTS], fontsize=9)
        if norm:
            a.set_ylabel("share of TTFA (%)")
            a.set_title("same split, normalised — the bottleneck MOVES", fontsize=10)
            a.set_ylim(0, 100)
        else:
            a.set_ylabel("TTFA p50 (ms)")
            a.set_title("8 concurrent users, shipped config (EVS 0.95, 16 frames)",
                        fontsize=10)
            for i, c in enumerate(CONTENTS):
                a.text(i, bottom[i] + 130, f"{bottom[i]:.0f} ms",
                       ha="center", fontsize=9, fontweight="bold")
            a.set_ylim(0, max(bottom) * 1.16)
        a.grid(axis="y", alpha=.3)
        a.set_axisbelow(True)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Where 8-user latency goes: light content is speech-bound, "
                 "high motion is prefill-bound", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(RES / "figV5_decompose8.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- figV6
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4))
    a = ax[0]
    pre = [COMPETE[c][0] for c in CONTENTS]
    spk = [COMPETE[c][1] for c in CONTENTS]
    a.bar(x, pre, 0.6, label="others waiting for first sound (prefill)",
          color="#d95f02", edgecolor="white")
    a.bar(x, spk, 0.6, bottom=pre, label="others already speaking (talker+code2wav)",
          color="#1b9e77", edgecolor="white")
    a.axhline(7, ls="--", c="k", lw=1)
    a.text(2.42, 7.15, "7 = all other users", fontsize=8, ha="right")
    for i, (p, s) in enumerate(zip(pre, spk)):
        a.text(i, p / 2, f"{p:.1f}", ha="center", va="center", color="white",
               fontsize=9, fontweight="bold")
        a.text(i, p + s / 2, f"{s:.1f}", ha="center", va="center", color="white",
               fontsize=9, fontweight="bold")
    a.set_xticks(list(x))
    a.set_xticklabels([NAME[c] for c in CONTENTS], fontsize=9)
    a.set_ylabel("other users competing, mean per turn")
    a.set_title("what the device was doing for SOMEBODY ELSE\n"
                "during my wait", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    a.grid(axis="y", alpha=.3)
    a.set_axisbelow(True)

    a = ax[1]
    v = [EFFCONC[c] for c in CONTENTS]
    a.bar(x, v, 0.6, color="#7570b3", edgecolor="white")
    a.axhline(1, ls=":", c="k", lw=1.2)
    a.text(2.42, 1.06, "1.0 = no better than one at a time", fontsize=8, ha="right")
    a.axhline(7.69, ls="--", c="#444", lw=1)
    a.text(2.42, 7.4, "7.7 = perfect batching of what was in flight",
           fontsize=8, ha="right")
    for i, val in enumerate(v):
        a.text(i, val + 0.16, f"{val:.2f}", ha="center", fontsize=10,
               fontweight="bold")
    a.set_xticks(list(x))
    a.set_xticklabels([NAME[c] for c in CONTENTS], fontsize=9)
    a.set_ylabel("requests actually served at once")
    a.set_ylim(0, 8.5)
    a.set_title("effective concurrency the engine achieved\n"
                "(in flight / measured slowdown)", fontsize=10)
    a.grid(axis="y", alpha=.3)
    a.set_axisbelow(True)
    fig.suptitle("Why: the competing work changes with content, and batching never "
                 "gets above ~2 of 7.7", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(RES / "figV6_why8.png", dpi=140)
    plt.close(fig)

    print("-> figV5_decompose8.png, figV6_why8.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
