"""Audit colocated CUDA work and conservatively identify already-queued gaps.

Kernel intervals are observed wall-clock spans, NOT SM or register occupancy.
Do not sum overlapping intervals to claim device utilization. The witnesses
exclude CUDA-graph nodes, pending same-stream copies, and recorded event waits.
Overlap is temporal evidence; placement ablations establish the causal effect.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


class Intervals:
    def __init__(self, rows):
        merged = []
        for start, end in sorted(rows):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.rows = merged
        self.starts = [row[0] for row in merged]
        self.ends = [row[1] for row in merged]
        self.prefix = [0]
        for start, end in merged:
            self.prefix.append(self.prefix[-1] + end - start)

    def overlap(self, start, end):
        left = bisect.bisect_right(self.ends, start)
        right = bisect.bisect_left(self.starts, end)
        if left >= right:
            return 0
        return (
            self.prefix[right]
            - self.prefix[left]
            - max(0, start - self.starts[left])
            - max(0, self.ends[right - 1] - end)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thinker-pid", type=int, required=True)
    parser.add_argument("--thinker-stream", type=int, required=True)
    parser.add_argument("--encoder-stream", type=int, required=True)
    parser.add_argument("--talker-pid", type=int, required=True)
    parser.add_argument("--code2wav-pid", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    db = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    pids = dict(db.execute("SELECT globalPid,pid FROM PROCESSES"))
    thinker_global = next(key for key, value in pids.items() if value == args.thinker_pid)
    groups = defaultdict(list)
    thinker_ops = []
    for start, end, gpid, stream, corr, name, graph in db.execute(
        "SELECT start,end,globalPid,streamId,correlationId,shortName,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        pid = pids[gpid]
        if pid == args.thinker_pid:
            label = (
                "Thinker" if stream == args.thinker_stream else "Encoder" if stream == args.encoder_stream else "Other"
            )
        elif pid == args.talker_pid:
            label = "Talker"
        elif pid == args.code2wav_pid:
            label = "Code2Wav"
        else:
            label = "Other"
        groups[label].append((start, end))
        if label == "Thinker":
            thinker_ops.append((start, end, "kernel", corr, name, graph))
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        for start, end in db.execute(
            f"SELECT start,end FROM {table} WHERE globalPid=? AND streamId=?",
            (thinker_global, args.thinker_stream),
        ):
            thinker_ops.append((start, end, "memory", None, None, None))
    intervals = {key: Intervals(value) for key, value in groups.items()}
    event_waits = Intervals(
        db.execute(
            "SELECT start,end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE globalPid=? AND streamId=? AND syncType=2",
            (thinker_global, args.thinker_stream),
        )
    )
    launches = {}
    for corr, start, end, name in db.execute(
        "SELECT correlationId,start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE (globalTid & -16777216)=?",
        (thinker_global,),
    ):
        if "Launch" in strings[name]:
            launches[corr] = (start, end, strings[name])
    gaps = []
    direct_launches = 0
    direct_gaps_ns = []
    event_wait_exclusions = 0
    previous_end = 0
    for start, end, kind, corr, name, graph in sorted(thinker_ops):
        if kind == "kernel" and not graph and corr in launches:
            api_start, api_end, api_name = launches[corr]
            ready = max(previous_end, api_end)
            direct_launches += 1
            if event_waits.overlap(ready, start):
                event_wait_exclusions += 1
            else:
                direct_gaps_ns.append(max(0, start - ready))
            if start - ready >= 100_000 and not event_waits.overlap(ready, start):
                overlaps = {
                    key: item.overlap(ready, start) / 1e6 for key, item in intervals.items() if key != "Thinker"
                }
                gaps.append(
                    {
                        "queue_start_s": ready / 1e9,
                        "kernel_start_s": start / 1e9,
                        "kernel_end_s": end / 1e9,
                        "gap_ms": (start - ready) / 1e6,
                        "api_start_s": api_start / 1e9,
                        "api_end_s": api_end / 1e9,
                        "previous_stream_activity_end_s": previous_end / 1e9,
                        "api": api_name,
                        "correlation_id": corr,
                        "kernel": strings[name],
                        "other_kernel_overlap_ms": overlaps,
                    }
                )
        previous_end = max(previous_end, end)
    all_intervals = Intervals(row for value in groups.values() for row in value)
    start = all_intervals.starts[0]
    end = all_intervals.ends[-1]
    direct_gaps_ns.sort()

    def gap_percentile_ms(percent):
        if not direct_gaps_ns:
            return None
        index = min(len(direct_gaps_ns) - 1, int(len(direct_gaps_ns) * percent / 100))
        return direct_gaps_ns[index] / 1e6

    report = {
        "source": str(args.sqlite.resolve()),
        "session_start": db.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME").fetchone(),
        "mapping": vars(args) | {"sqlite": str(args.sqlite), "out": str(args.out)},
        "observed_window_s": (end - start) / 1e9,
        "definition": "Kernel interval union, not SM/FLOPS/occupancy; tracing is diagnostic only",
        "modules": {
            key: {
                "kernel_count": len(groups[key]),
                "kernel_sum_ms": sum(e - s for s, e in groups[key]) / 1e6,
                "kernel_union_ms": item.prefix[-1] / 1e6,
                "kernel_union_fraction": item.prefix[-1] / (end - start),
            }
            for key, item in intervals.items()
        },
        "gap_definition": (
            "Direct kernel launch API has returned and previous same-stream kernel/copy/memset ended; "
            "no recorded same-stream event-wait overlaps. CUDA-graph nodes excluded. "
            "Temporal overlap alone is not a resource-specific causal attribution."
        ),
        "qualifying_queued_gaps": len(gaps),
        "direct_launches_with_api": direct_launches,
        "event_wait_exclusions": event_wait_exclusions,
        "direct_queue_gap_ms": {
            "count": len(direct_gaps_ns),
            "p50": gap_percentile_ms(50),
            "p95": gap_percentile_ms(95),
            "p99": gap_percentile_ms(99),
            "max": direct_gaps_ns[-1] / 1e6 if direct_gaps_ns else None,
        },
        "qualifying_gaps_per_1000_direct_launches": 1000 * len(gaps) / direct_launches if direct_launches else None,
        "queued_gap_sum_ms": sum(row["gap_ms"] for row in gaps),
        "queued_gap_overlap_ms": {
            key: sum(row["other_kernel_overlap_ms"].get(key, 0) for row in gaps)
            for key in intervals
            if key != "Thinker"
        },
        "largest_gaps": sorted(gaps, key=lambda row: row["gap_ms"], reverse=True)[:20],
        "code2wav_dominated_gaps": sorted(
            (row for row in gaps if row["other_kernel_overlap_ms"].get("Code2Wav", 0) >= row["gap_ms"] * 0.9),
            key=lambda row: row["gap_ms"],
            reverse=True,
        )[:20],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("gaps")}, indent=2))
    print("largest Code2Wav-dominated gap:", report["code2wav_dominated_gaps"][:1])


if __name__ == "__main__":
    main()
