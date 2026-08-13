"""Sag dissection: is the u56 miss a RATE deficit (drift) or jitter (spikes)?

Part A (client, turns.jsonl, re-anchored playback per mu_bench semantics):
  - per-turn: n_miss, total_gap, gap sizes -> drift vs spike classification
  - miss position within turn (index quartile)
  - arrival-interval mean by turn quartile (is the sag mid-turn or uniform?)
Part B (production, engine_slice.log CHUNK-EMIT):
  - per-session EMIT interval mean vs 320ms -> is rate deficit born at production?
  - EMIT interval by chunk-index quartile
"""
import collections
import json
import re
import statistics
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/data/results/burst12_u56"

# ---------- Part A: client side ----------
turns = [json.loads(l) for l in open(D + "/turns.jsonl")]
turns = [t for t in turns if t.get("status") == "ok" and t.get("deltas")]

drift_turns = spike_turns = mixed_turns = clean_turns = 0
all_gap_events = []          # (gap_ms, frac_into_turn)
q_intervals = [[], [], [], []]   # arrival intervals by within-turn quartile
turn_rate = []               # arrival span / audio span (>1 = slower than realtime)
miss_run_lens = []

for t in turns:
    ds = t["deltas"]
    # playback model: cursor = playout frontier (abs time)
    cur = None
    misses = []           # (idx, gap_s)
    for i, (trel, nbytes) in enumerate(ds):
        dur = nbytes / 24000.0    # field is SAMPLES at 24kHz (88 deltas x .32s = audio_s checks out)
        if cur is None:
            cur = trel + dur
            continue
        if trel <= cur + 1e-9:
            cur += dur
        else:
            misses.append((i, trel - cur))
            cur = trel + dur      # re-anchor
    n = len(ds)
    if n >= 8:
        # arrival intervals by quartile of chunk index
        for i in range(1, n):
            dt = (ds[i][0] - ds[i-1][0])
            if 0 < dt < 3.0:
                q_intervals[min(3, 4 * i // n)].append(dt * 1000)
        span = ds[-1][0] - ds[0][0]
        audio = sum(b for _, b in ds[1:]) / 24000.0
        if audio > 1:
            turn_rate.append(span / audio)
    for i, g in misses:
        all_gap_events.append((g * 1000, i / max(1, n - 1)))
    # classify
    total = sum(g for _, g in misses) * 1000
    big = [g for _, g in misses if g * 1000 >= 300]
    if total < 100:
        clean_turns += 1
    elif big and sum(big) * 1000 > 0.6 * total:
        spike_turns += 1
    elif len(misses) >= 5 and not big:
        drift_turns += 1
    else:
        mixed_turns += 1
    # miss runs (consecutive-ish misses: gap between miss indices <= 3)
    run = 0
    prev = -10
    for i, _ in misses:
        if i - prev <= 3:
            run += 1
        else:
            if run:
                miss_run_lens.append(run)
            run = 1
        prev = i
    if run:
        miss_run_lens.append(run)

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")

print(f"== A: client (re-anchored)  turns={len(turns)}")
print(f"turn class: clean(<100ms)={clean_turns} drift(many small)={drift_turns} "
      f"spike(few big)={spike_turns} mixed={mixed_turns}")
gaps = [g for g, _ in all_gap_events]
print(f"miss events n={len(gaps)}  gap p50={pct(gaps,50):.0f} p90={pct(gaps,90):.0f} "
      f"p99={pct(gaps,99):.0f}ms  mean={statistics.mean(gaps):.0f}ms" if gaps else "no misses")
pos = [p for _, p in all_gap_events]
if pos:
    hist = [0]*4
    for p in pos:
        hist[min(3, int(p*4))] += 1
    print(f"miss position in turn (quartile counts): {hist}")
print("arrival interval by turn quartile: " +
      " | ".join(f"Q{i+1} mean={statistics.mean(q):.0f} p90={pct(q,90):.0f}" for i, q in enumerate(q_intervals) if q))
print(f"turn delivery rate (span/audio): p50={pct(turn_rate,50):.3f} p90={pct(turn_rate,90):.3f} "
      f"(>1 = slower than realtime)")
print(f"miss run lengths: p50={pct(miss_run_lens,50):.0f} p90={pct(miss_run_lens,90):.0f} max={max(miss_run_lens) if miss_run_lens else 0}")

# ---------- Part B: production side (CHUNK-EMIT) ----------
emit = collections.defaultdict(list)
rx = re.compile(r"\[CHUNK-EMIT\] rid=(\S+) chunk_id=(\d+) frames=(\d+) mono=([0-9.]+)")
for line in open(D + "/engine_slice.log", errors="ignore"):
    m = rx.search(line)
    if m:
        emit[m[1]].append((int(m[2]), int(m[3]), float(m[4])))

seg_rates = []           # per-segment emit span / audio span
eq_intervals = [[], [], [], []]
emit_gaps = []
for rid, evs in emit.items():
    evs.sort(key=lambda x: x[2])
    # split into segments on >2s silence (turn boundaries share rid across turns? rid is per segment usually)
    seg = [evs[0]]
    segs = []
    for e in evs[1:]:
        if e[2] - seg[-1][2] > 2.0:
            segs.append(seg)
            seg = []
        seg.append(e)
    segs.append(seg)
    for s in segs:
        if len(s) < 8:
            continue
        n = len(s)
        for i in range(1, n):
            dt = s[i][2] - s[i-1][2]
            if 0 < dt < 3.0:
                eq_intervals[min(3, 4 * i // n)].append(dt * 1000)
                emit_gaps.append(dt * 1000)
        span = s[-1][2] - s[0][2]
        audio = sum(e[1] for e in s[1:]) * 0.08   # frames -> s? 1 frame = 80ms
        if audio > 1:
            seg_rates.append(span / audio)

print(f"\n== B: production (CHUNK-EMIT)  sessions={len(emit)} segments used={len(seg_rates)}")
print(f"EMIT interval: mean={statistics.mean(emit_gaps):.1f}ms p50={pct(emit_gaps,50):.0f} "
      f"p90={pct(emit_gaps,90):.0f} p99={pct(emit_gaps,99):.0f}  (nominal 320)")
print("EMIT interval by segment quartile: " +
      " | ".join(f"Q{i+1} mean={statistics.mean(q):.0f}" for i, q in enumerate(eq_intervals) if q))
print(f"segment production rate (span/audio): p50={pct(seg_rates,50):.3f} p90={pct(seg_rates,90):.3f}")
