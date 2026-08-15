#!/usr/bin/env python3
"""Compute the two SLO metrics from mu_bench turns.jsonl files (see workflow.md).

    analyze.py RUN_DIR [RUN_DIR ...] [--warmup-turns 2]

Each RUN_DIR is one (condition, N) cell written by mu_bench.py --out. Prints
one table row per dir and writes metrics.json into each.

A run passes on exactly two numbers:

    ttfa_p99_ms  < 1000    first audio, measured from the end of user speech
    stall_p99_ms <   50    silence at a chunk seam, no client-side prebuffer

Everything else printed is diagnostic. The old criterion was the fraction of
chunks that arrived late; it was dropped because lateness is binary there (1 ms
and 3 s count the same) and its denominator moves with chunk size. Measured: of
the 0.6% late chunks at 200 sessions, 77% were shorter than 50 ms and inaudible.

The playback model (per turn): the client starts playing at the first delta's
arrival. Chunk i is NEEDED the moment the previous chunks' audio runs out; a
chunk arriving later than that leaves the speaker silent for `t - cursor`, which
is what the listener hears. Chunks arriving early are buffered (no penalty) --
so greedy generation is not penalized for bursting ahead; it is only penalized
when a mid-turn gap outruns the buffer it built. Every seam contributes one
number (0 when it is not late), and the p99 is taken over all of them.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

SR = 24000.0

# The pass criteria. 1000 ms: gaps past ~700 ms are heard as hesitation
# (Kendrick & Torreira 2015) and 1 s is the limit for uninterrupted flow of
# thought (Miller 1968); the looser of the two is taken. Note the budget is
# meant to include endpointing, which t_q does not -- a real product spends
# another 200-700 ms there. 50 ms: a stop closure in natural speech is already
# 50-100 ms of near-silence, so a shorter seam is masked by the speech itself.
TTFA_P99_MS = 1000.0
STALL_P99_MS = 50.0


def pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def playback(deltas: list[list[float]]) -> dict:
    """Deadline misses + stall seconds for one turn's [t_rel, samples] list."""
    if len(deltas) < 2:
        return {"misses": 0, "chunks": len(deltas), "stall_s": 0.0,
                "gaps": [], "excess": [], "stalls": []}
    misses = 0
    stall = 0.0
    # cursor = wall time at which playback of everything delivered so far ends
    cursor = deltas[0][0] + deltas[0][1] / SR
    gaps, excess, stalls = [], [], []
    for i in range(1, len(deltas)):
        t, samples = deltas[i]
        prev_t, prev_samples = deltas[i - 1]
        gaps.append(t - prev_t)
        excess.append((t - prev_t) - prev_samples / SR)
        if t > cursor + 1e-4:            # arrived after the audio ran out
            misses += 1
            stall += t - cursor
            stalls.append(t - cursor)     # seconds of silence at this seam
            cursor = t                    # re-anchor: playback resumes now
        else:
            stalls.append(0.0)            # seamless seam; still one sample
        cursor += samples / SR
    return {"misses": misses, "chunks": len(deltas), "stall_s": stall,
            "gaps": gaps, "excess": excess, "stalls": stalls}


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
    all_stalls: list[float] = []
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
        all_stalls.extend(pb["stalls"])

    # Aggregate throughput over the measured window: audio seconds delivered
    # per wall second, using the span from first query to last done.
    t0 = min(t["t_q"] for t in ok)
    t1 = max(t["t_done"] for t in ok if t.get("t_done"))
    wall = max(1e-9, t1 - t0)

    gp50 = pctl(all_gaps, 0.5)
    gp99 = pctl(all_gaps, 0.99)
    ttfa_p99 = pctl(ttfa, 0.99)
    stall_p99 = (pctl(all_stalls, 0.99) or 0.0) * 1000 if all_stalls else None
    m = {
        "dir": str(d),
        "n_ok": len(ok),
        "n_timeout": sum(1 for t in turns if t.get("status") == "timeout"),
        "n_users": len({t["user"] for t in turns}),
        # THE TWO CRITERIA
        "ttfa_p99_ms": ttfa_p99,
        "stall_p99_ms": stall_p99,
        "pass": (ttfa_p99 is not None and ttfa_p99 < TTFA_P99_MS
                 and stall_p99 is not None and stall_p99 < STALL_P99_MS),
        # latency
        "ttfa_p50_ms": pctl(ttfa, 0.5),
        "ttfa_p95_ms": pctl(ttfa, 0.95),
        # predictability
        "stall_p50_ms": (pctl(all_stalls, 0.5) or 0.0) * 1000 if all_stalls else None,
        "stall_p999_ms": (pctl(all_stalls, 0.999) or 0.0) * 1000 if all_stalls else None,
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
    ("dir", 30), ("n_ok", 5),
    # the two criteria, then the verdict
    ("ttfa_p99_ms", 11), ("stall_p99_ms", 12), ("pass", 5),
    # diagnostics
    ("ttfa_p50_ms", 9), ("stall_ms_per_turn", 9), ("deadline_miss_pct", 7),
    ("throughput_audio_s_per_wall_s", 8), ("rtf_deliver_p50", 7),
]


def fmt(v, w):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, bool):
        return ("PASS" if v else "FAIL").rjust(w)
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
