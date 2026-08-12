#!/usr/bin/env python3
"""Compute the DESIGN.zh.md metrics from mu_bench turns.jsonl files.

    analyze.py RUN_DIR [RUN_DIR ...] [--warmup-turns 2]

Each RUN_DIR is one (condition, N) cell written by mu_bench.py --out. Prints
one table row per dir and writes metrics.json into each.

The playback model (per turn): the client starts playing at the first delta's
arrival. Chunk i is NEEDED the moment the previous chunks' audio runs out; a
chunk arriving later than that is a DEADLINE MISS and the gap is a STALL the
listener hears. Chunks arriving early are buffered (no penalty) -- so greedy
generation is not penalized for bursting ahead; it is only penalized when a
mid-turn gap outruns the buffer it built. That keeps the comparison honest in
both directions.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

SR = 24000.0


def pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def playback(deltas: list[list[float]]) -> dict:
    """Deadline misses + stall seconds for one turn's [t_rel, samples] list."""
    if len(deltas) < 2:
        return {"misses": 0, "chunks": len(deltas), "stall_s": 0.0, "gaps": [], "excess": []}
    misses = 0
    stall = 0.0
    # cursor = wall time at which playback of everything delivered so far ends
    cursor = deltas[0][0] + deltas[0][1] / SR
    gaps, excess = [], []
    for i in range(1, len(deltas)):
        t, samples = deltas[i]
        prev_t, prev_samples = deltas[i - 1]
        gaps.append(t - prev_t)
        excess.append((t - prev_t) - prev_samples / SR)
        if t > cursor + 1e-4:            # arrived after the audio ran out
            misses += 1
            stall += t - cursor
            cursor = t                    # re-anchor: playback resumes now
        cursor += samples / SR
    return {"misses": misses, "chunks": len(deltas), "stall_s": stall,
            "gaps": gaps, "excess": excess}


def analyze_dir(d: pathlib.Path, warmup_turns: int) -> dict | None:
    tf = d / "turns.jsonl"
    if not tf.exists():
        print(f"!! {d}: no turns.jsonl", file=sys.stderr)
        return None
    turns = [json.loads(line) for line in tf.open()]
    ok = [t for t in turns if t.get("status") == "ok" and t.get("turn", 0) > warmup_turns]
    if not ok:
        print(f"!! {d}: no ok turns after warmup filter", file=sys.stderr)
        return None

    ttfa = [t["ttfa_ms"] for t in ok if t.get("ttfa_ms") is not None]
    all_gaps: list[float] = []
    all_excess: list[float] = []
    miss_chunks = 0
    total_chunks = 0
    stall_s = 0.0
    stall_turns = 0
    audio_s = 0.0
    for t in ok:
        audio_s += t.get("audio_s") or 0.0
        pb = playback(t.get("deltas") or [])
        miss_chunks += pb["misses"]
        total_chunks += pb["chunks"]
        stall_s += pb["stall_s"]
        stall_turns += 1 if pb["misses"] else 0
        all_gaps.extend(pb["gaps"])
        all_excess.extend(pb["excess"])

    # Aggregate throughput over the measured window: audio seconds delivered
    # per wall second, using the span from first query to last done.
    t0 = min(t["t_q"] for t in ok)
    t1 = max(t["t_done"] for t in ok if t.get("t_done"))
    wall = max(1e-9, t1 - t0)

    gp50 = pctl(all_gaps, 0.5)
    gp99 = pctl(all_gaps, 0.99)
    m = {
        "dir": str(d),
        "n_ok": len(ok),
        "n_timeout": sum(1 for t in turns if t.get("status") == "timeout"),
        "n_users": len({t["user"] for t in turns}),
        # latency
        "ttfa_p50_ms": pctl(ttfa, 0.5),
        "ttfa_p95_ms": pctl(ttfa, 0.95),
        "ttfa_p99_ms": pctl(ttfa, 0.99),
        # predictability
        "gap_p50_ms": gp50 * 1000 if gp50 else None,
        "gap_p95_ms": (pctl(all_gaps, 0.95) or 0) * 1000 if all_gaps else None,
        "gap_p99_ms": gp99 * 1000 if gp99 else None,
        "gap_p99_over_p50": (gp99 / gp50) if gp50 and gp99 else None,
        "gap_stdev_ms": statistics.pstdev(all_gaps) * 1000 if len(all_gaps) > 1 else None,
        "excess_p99_ms": (pctl([max(0.0, e) for e in all_excess], 0.99) or 0) * 1000 if all_excess else None,
        "deadline_miss_pct": 100.0 * miss_chunks / total_chunks if total_chunks else None,
        "stall_turn_pct": 100.0 * stall_turns / len(ok),
        "stall_s_total": round(stall_s, 3),
        "stall_ms_per_turn": 1000.0 * stall_s / len(ok),
        # throughput
        "audio_s_total": round(audio_s, 1),
        "throughput_audio_s_per_wall_s": round(audio_s / wall, 4),
        "rtf_deliver_p50": pctl([t["rtf_deliver"] for t in ok if t.get("rtf_deliver")], 0.5),
        "audio_s_p50": pctl([t["audio_s"] for t in ok], 0.5),
        "wall_s": round(wall, 1),
    }
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


COLS = [
    ("dir", 34), ("n_ok", 5),
    ("ttfa_p50_ms", 9), ("ttfa_p99_ms", 9),
    ("gap_p50_ms", 8), ("gap_p99_ms", 8), ("gap_p99_over_p50", 8),
    ("deadline_miss_pct", 7), ("stall_ms_per_turn", 9),
    ("throughput_audio_s_per_wall_s", 8), ("rtf_deliver_p50", 7),
]


def fmt(v, w):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.1f}".rjust(w) if abs(v) >= 10 else f"{v:.2f}".rjust(w)
    return str(v)[-w:].rjust(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--warmup-turns", type=int, default=2)
    args = ap.parse_args()
    header = " ".join(name[:w].rjust(w) for name, w in COLS)
    print(header)
    print("-" * len(header))
    for d in args.dirs:
        m = analyze_dir(pathlib.Path(d), args.warmup_turns)
        if m:
            print(" ".join(fmt(m.get(name), w) for name, w in COLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
