#!/usr/bin/env python3
"""Per-event waterfall for late chunks (tail-conditioned decomposition).

    waterfall.py RESULT_DIR [--thresh-ms 480]

Method (the discipline this project learned the hard way): diagnose tail
problems by decomposing THE TAIL EVENTS themselves, hop by hop, per event --
never by comparing per-hop medians (p50 "all hops healthy" is vacuous), and
per-hop p99 still misses cross-hop accumulation. Requires the resident stamps:
[TEXT-CHUNK] (stage-1 text arrival), [CHUNK-EMIT] (stage-1 send),
[AUDIO-CHUNK] (orchestrator arrival). Client deltas live in turns.jsonl.
"""
import bisect
import collections
import re
import statistics
import sys

d = sys.argv[1]
TH = float(sys.argv[sys.argv.index("--thresh-ms") + 1]) if "--thresh-ms" in sys.argv else 480.0
txt = open(d + "/engine_slice.log", errors="ignore").read()
emit = collections.defaultdict(list)
for m in re.finditer(r"\[CHUNK-EMIT\] rid=(\S+) chunk_id=\d+ frames=\d+ mono=([0-9.]+)", txt):
    emit[m[1]].append(float(m[2]))
orch = collections.defaultdict(list)
for m in re.finditer(r"\[AUDIO-CHUNK\] stage=2 req=(\S+) ts=([0-9.]+)", txt):
    orch[m[1]].append(float(m[2]))
textc = collections.defaultdict(list)
for m in re.finditer(r"\[TEXT-CHUNK\] stage=1 rid=(\S+) mono=([0-9.]+)", txt):
    textc[m[1]].append(float(m[2]))
offs = [sorted(orch[r])[0] - sorted(emit[r])[0] for r in emit if r in orch and emit[r] and orch[r]]
if not offs:
    sys.exit("no paired stamps; run with VLLM_OMNI_LOG_AUDIO_CHUNKS=1")
OFF = statistics.median(offs)
buckets = collections.Counter()
prod_ex, tr_ex, tw = [], [], []
for rid in orch:
    o, e, tt = sorted(orch[rid]), sorted(emit.get(rid, [])), sorted(textc.get(rid, []))
    n = min(len(o), len(e))
    for i in range(1, n):
        gap = (o[i] - o[i - 1]) * 1000
        if not (TH <= gap < 2000):
            continue
        comp = {
            "production": (e[i] - e[i - 1]) * 1000 - 320,
            "transit": ((o[i] - OFF - e[i]) - (o[i - 1] - OFF - e[i - 1])) * 1000,
        }
        j = bisect.bisect_left(tt, e[i])
        if j > 0:
            tw.append((e[i] - tt[j - 1]) * 1000)
        top = max(comp, key=comp.get)
        buckets[top] += 1
        prod_ex.append(comp["production"])
        tr_ex.append(comp["transit"])
def q(xs, p):
    xs = sorted(xs)
    return xs[int(p * len(xs) / 100)] if xs else float("nan")
print(f"late chunks (>={TH:.0f}ms): {sum(buckets.values())}")
print(f"attribution: {dict(buckets)}")
print(f"production excess in-slip: p50={q(prod_ex,50):.0f} p90={q(prod_ex,90):.0f}ms")
print(f"transit delta in-slip    : p50={q(tr_ex,50):.0f} p90={q(tr_ex,90):.0f}ms")
