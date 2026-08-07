"""Parse duplex serve log: per-session stage-0 ADMIT cadence = unit service truth.

Each duplex append is re-admitted to stage 0 once per model unit. If the engine
keeps up with N realtime sessions, every session's inter-admit interval stays
~1 s; backlog shows up as stretched intervals. This is the server-side
deadline-miss signal, independent of client-side turn attribution.

Usage: python analyze_admits.py <serve_log> [window_start_epoch] [window_end]
Prints per-session admit counts and interval p50/p95/max, plus a global line.
"""

import re
import sys
from collections import defaultdict
from datetime import datetime

LINE = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*stage 0 ADMIT req=duplex-s\.([A-Za-z0-9+/=_-]+)\.i\.(\d+)\.e\.(\d+)"
)


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))], 2)


def main(path: str) -> None:
    admits = defaultdict(list)
    year = datetime.now().year
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        ts = datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").timestamp()
        session = f"{m.group(2)[:12]}.i{m.group(3)}"
        admits[session].append(ts)

    print(f"{'session':<18} {'admits':>6} {'span_s':>7} {'int_p50':>8} {'int_p95':>8} {'int_max':>8}")
    all_intervals = []
    for session, times in sorted(admits.items(), key=lambda kv: kv[1][0]):
        times.sort()
        intervals = [b - a for a, b in zip(times, times[1:]) if b - a < 60]
        all_intervals.extend(intervals)
        span = times[-1] - times[0]
        print(
            f"{session:<18} {len(times):>6} {span:>7.1f} "
            f"{pct(intervals, 0.5) or '-':>8} {pct(intervals, 0.95) or '-':>8} {pct(intervals, 1.0) or '-':>8}"
        )
    print(
        f"\nGLOBAL intervals n={len(all_intervals)} "
        f"p50={pct(all_intervals, 0.5)} p95={pct(all_intervals, 0.95)} max={pct(all_intervals, 1.0)} "
        f"over_1.5s={sum(1 for v in all_intervals if v > 1.5)} "
        f"({100 * sum(1 for v in all_intervals if v > 1.5) / max(1, len(all_intervals)):.1f}%)"
    )


if __name__ == "__main__":
    main(sys.argv[1])
