"""Plot a measured queued-kernel witness; never synthesize GPU metrics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--correlation-id", type=int, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    witness = next(row for row in report["largest_gaps"] if row["correlation_id"] == args.correlation_id)
    mapping = report["mapping"]
    db = sqlite3.connect(f"file:{report['source']}?mode=ro", uri=True)
    pids = dict(db.execute("SELECT globalPid,pid FROM PROCESSES"))
    start = witness["api_end_s"] - 0.001
    end = witness["kernel_start_s"] + 0.002
    rows = [[], [], [], []]
    for left, right, global_pid, stream in db.execute(
        "SELECT start,end,globalPid,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?",
        (int(start * 1e9), int(end * 1e9)),
    ):
        pid = pids[global_pid]
        if pid == mapping["thinker_pid"]:
            row = 0 if stream == mapping["thinker_stream"] else 1 if stream == mapping["encoder_stream"] else None
        else:
            row = 2 if pid == mapping["talker_pid"] else 3 if pid == mapping["code2wav_pid"] else None
        if row is not None:
            rows[row].append(((left / 1e9 - start) * 1000, (right - left) / 1e6))
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(left=0.19, right=0.975, top=0.73, bottom=0.25)
    labels = [
        f"Thinker / stream {mapping['thinker_stream']}",
        f"Encoder / stream {mapping['encoder_stream']}",
        "Talker / separate process",
        "Code2Wav / separate process",
    ]
    colors = ["#236a91", "#9356a6", "#d78521", "#199b79"]
    for index, spans in enumerate(rows):
        ax.broken_barh(spans, (3 - index - 0.3, 0.6), facecolors=colors[index], linewidth=0)
    left = (witness["queue_start_s"] - start) * 1000
    right = (witness["kernel_start_s"] - start) * 1000
    ax.add_patch(Rectangle((left, 2.65), right - left, 0.7, fill=False, edgecolor="#da3b2e", hatch="///", linewidth=2))
    ax.axvline(left, color="#da3b2e", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(right, color="#da3b2e", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.annotate(
        "Launch API returned",
        xy=(1, 3.3),
        xytext=(1, 3.9),
        ha="center",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    ax.annotate(
        f"Already queued: {witness['gap_ms']:.3f} ms",
        xy=((left + right) / 2, 3.3),
        xytext=((left + right) / 2, 3.9),
        ha="center",
        fontsize=10,
        color="#b42a22",
        arrowprops={"arrowstyle": "->", "color": "#b42a22"},
    )
    ax.set(yticks=[3, 2, 1, 0], yticklabels=labels, ylim=(-0.55, 4.15), xlim=(0, (end - start) * 1000))
    ax.set_xlabel(f"Time relative to {start:.6f} s in the Nsight trace (ms)", fontsize=10)
    ax.grid(axis="x", alpha=0.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Code2Wav runs while an already-submitted Thinker kernel waits", x=0.04, ha="left", fontsize=17, weight="bold"
    )
    fig.text(
        0.04,
        0.87,
        "Measured Nsight CUDA timeline | GPU 0 | 3-user all-on-one-GPU deployment | long context",
        fontsize=11,
        color="#4e5c68",
    )
    other = witness["other_kernel_overlap_ms"]
    fig.text(
        0.04,
        0.13,
        f"Inside the {witness['gap_ms']:.3f} ms queued gap: Code2Wav {other['Code2Wav']:.3f} ms; "
        f"Encoder {other['Encoder']:.3f} ms; Talker {other['Talker']:.3f} ms.",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.075,
        "Earlier same-stream kernels/copies have ended; no recorded event-wait overlaps this gap. "
        "Red hatch is waiting, not a kernel.",
        fontsize=9,
        color="#4e5c68",
    )
    fig.text(
        0.04,
        0.03,
        "Rows are CUDA streams, not individual SMs. Kernel spans may include preemption; "
        "this figure does not establish warp/register saturation.",
        fontsize=9,
        color="#4e5c68",
    )
    fig.savefig(args.out, dpi=180, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
