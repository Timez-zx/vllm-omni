#!/usr/bin/env python3
"""Summarize [SCHED-STEP] lines from an instrumented engine log.

    sched_steps.py /home/ubuntu/data/logs/thinker_talker_engine.log [--stage 1]

The number that decides whether this deployment can serve N sessions is the
stage-1 INTER-STEP INTERVAL, not any throughput figure. Each session owes the
listener 12.5 codec frames per second, one frame per scheduler step, so the
stage must give every session a step every 80 ms. Two things eat that budget:
the pass interval itself, and the number of passes a session spends parked
waiting for its next payload (the "interval tax"). The inequality that has
predicted pass/fail on every cell measured so far:

    1 / (tax x pass_interval) >= 12.5 frames/s

Reports, per stage: steps counted and rate, batch-size (nreq) histogram, mean
tokens per step, and the inter-step interval percentiles that feed the left
side of that inequality. Run the engine with VLLM_OMNI_LOG_SCHED_STEPS=1.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys

LINE = re.compile(
    r"\[SCHED-STEP\] stage=(\d+) mono=([0-9.]+) nreq=(\d+) ntok=(\d+)")


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

    per_stage: dict[int, list[tuple[float, int, int]]] = {}
    for line in open(args.log, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        stage, mono, nreq, ntok = (int(m.group(1)), float(m.group(2)),
                                   int(m.group(3)), int(m.group(4)))
        if args.stage is not None and stage != args.stage:
            continue
        per_stage.setdefault(stage, []).append((mono, nreq, ntok))

    if not per_stage:
        print("no [SCHED-STEP] lines found (was VLLM_OMNI_LOG_SCHED_STEPS=1 set?)")
        return 1

    for stage in sorted(per_stage):
        rows = sorted(per_stage[stage])
        monos = [r[0] for r in rows]
        nreqs = [r[1] for r in rows]
        ntoks = [r[2] for r in rows]
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
        if gaps:
            print(f"  inter-step ms: p50={pctl(gaps,0.5):.1f} p90={pctl(gaps,0.9):.1f} "
                  f"p99={pctl(gaps,0.99):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
