#!/usr/bin/env python3
"""Summarize [SCHED-STEP] lines from an instrumented engine log.

    sched_steps.py /home/ubuntu/data/logs/temporal_engine.log [--stage 1]

The question these lines answer: did tick alignment actually change the BATCH
COMPOSITION -- many small steps (greedy collisions) vs fewer, larger, periodic
steps (temporal batching)? Reports, per stage:

  * steps counted, wall span
  * batch-size (nreq) histogram and mean tokens/step
  * inter-step interval percentiles (for the paced arms this should cluster
    at the tick; for greedy it collapses toward zero during bursts)
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys

LINE = re.compile(
    r"\[SCHED-STEP\] stage=(\d+) mono=([0-9.]+) nreq=(\d+) ntok=(\d+) held=(\d+)")


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--stage", type=int, default=None)
    args = ap.parse_args()

    per_stage: dict[int, list[tuple[float, int, int, int]]] = {}
    for line in open(args.log, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        stage, mono, nreq, ntok, held = (int(m.group(1)), float(m.group(2)),
                                         int(m.group(3)), int(m.group(4)), int(m.group(5)))
        if args.stage is not None and stage != args.stage:
            continue
        per_stage.setdefault(stage, []).append((mono, nreq, ntok, held))

    if not per_stage:
        print("no [SCHED-STEP] lines found (was the run INSTRUMENTED=1?)")
        return 1

    for stage in sorted(per_stage):
        rows = sorted(per_stage[stage])
        monos = [r[0] for r in rows]
        nreqs = [r[1] for r in rows]
        ntoks = [r[2] for r in rows]
        helds = [r[3] for r in rows]
        gaps = [(b - a) * 1000 for a, b in zip(monos, monos[1:]) if b - a < 2.0]
        hist: dict[int, int] = {}
        for n in nreqs:
            hist[n] = hist.get(n, 0) + 1
        span = monos[-1] - monos[0] if len(monos) > 1 else 0.0
        print(f"stage {stage}: steps={len(rows)} span={span:.1f}s "
              f"steps/s={len(rows)/span:.1f}" if span else f"stage {stage}: steps={len(rows)}")
        print(f"  batch nreq: mean={statistics.fmean(nreqs):.2f} "
              f"p50={pctl(nreqs,0.5)} p95={pctl(nreqs,0.95)} max={max(nreqs)} "
              f"hist={dict(sorted(hist.items()))}")
        print(f"  tokens/step: mean={statistics.fmean(ntoks):.1f} max={max(ntoks)}")
        print(f"  held (paced-out) per step: mean={statistics.fmean(helds):.2f} max={max(helds)}")
        if gaps:
            print(f"  inter-step ms: p50={pctl(gaps,0.5):.1f} p90={pctl(gaps,0.9):.1f} "
                  f"p99={pctl(gaps,0.99):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
