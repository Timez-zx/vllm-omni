#!/usr/bin/env python3
"""What does the TALKER charge per prompt token when its cache is working?

THE QUESTION THIS ANSWERS
-------------------------
The user's proposal is: prefill each newly-arrived frame as it passes the EVS filter, on
BOTH stages, so no turn ever re-prefills an old frame. Whether that rescues append-only
frame handling comes down to one number: with the talker's prefix cache working, what does
the talker still charge per 1,000 prompt tokens? For the thinker that number is 1.26 ms/1k
(measured earlier, 63x below the 79.2 ms/1k all-new rate). For the talker it was unmeasured.

It turns out the server already logged it. Every finished request emits a per-stage table
(stats.py:861) carrying vllm_ttft_ms for stages 0/1/2 and num_tokens_in for stage 0. So the
talker's own added latency is stage1_ttft - stage0_ttft, computed WITHIN a request, and the
real prompt length is measured rather than inverted out of a fitted rate.

THE CONFOUND, AND WHY THE ANSWER SURVIVES IT
--------------------------------------------
stage1_ttft - stage0_ttft is not purely "the talker's work". Stage 0's TTFT is its first TEXT
token, and the talker cannot start until the thinker has produced some text, so the delta
also contains "waiting for the thinker to decode k tokens". The thinker's per-token decode
DOES slow with context (measured: inter_output_latency 14.8 -> 20.2 ms as the prompt grows
4k -> 35.5k), so part of the talker's apparent growth is really the thinker's decode.

Two independent controls are applied here:

  (1) HARD UPPER BOUND on the contamination. The waiting term is at most
      (all text tokens the thinker emits) x (the growth in its per-token latency). Both are
      measured, so the bound needs no assumption about when the talker starts.

  (2) MATCHED-PROMPT COMPARISON. T640 and P640 are both append-only at 640x352 and differ
      only in stage-1 enable_prefix_caching. Comparing them at turn indices where the prompt
      length AND the thinker's per-token latency nearly coincide cancels the waiting term,
      because it is the same in both arms.

READ THE CAVEAT AT THE BOTTOM OF THE OUTPUT BEFORE QUOTING ANY NUMBER.
"""
from __future__ import annotations

import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage_stats_v2 import derive, parse  # noqa: E402

RES = pathlib.Path("/data/zx/results")

# label -> (log file, boot index, human description of the configuration)
# Boot identification: run_fix_tuned.sh boots twice into server_ft.log, C640 first then T640
# (its own log markers at lines 2 and 13146 confirm the order). run_fix_latency_high.sh
# truncates its log per boot, so server_fx.log retains only its LAST arm, P640.
ARMS = {
    "C640": ("server_ft.log", 1, "shipped stride re-pick, 640x352, stage-1 PC off. Prompt PINNED ~3.8k"),
    "T640": ("server_ft.log", 2, "append-only gap[8,16], 640x352, stage-1 PC off"),
    "P640": ("server_fx.log", 1, "append-only gap[4,10], 640x352, stage-1 PC **ON**"),
    "C640_low": ("server_fx2.log", 1, "shipped re-pick at 640, LOW motion (falls into the append branch)"),
}

_cache: dict[str, list[dict]] = {}


def rows(log: str, boot: int, limit: int = 240) -> list[dict]:
    if log not in _cache:
        _cache[log] = derive(parse(RES / log))
    rs = [r for r in _cache[log] if r["boot"] == boot][:limit]
    for i, r in enumerate(rs):
        r["idx"] = i
    return rs


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def binned(rs, bins=((1, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59))):
    out = []
    for a, b in bins:
        g = [r for r in rs if a <= r["idx"] <= b]
        if g:
            out.append(((a, b), g))
    return out


def ols(pts):
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0:
        return None
    slope = sxy / sxx
    return slope, my - slope * mx, (sxy ** 2) / (sxx * syy) if syy else float("nan")


def main() -> int:
    data = {k: rows(*v[:2]) for k, v in ARMS.items()}

    print("=" * 78)
    print("CONTROL 1  -- hard upper bound on the thinker-decode contamination")
    print("=" * 78)
    print("  The talker waits for the thinker to decode some text. That waiting term can")
    print("  grow by AT MOST (total text tokens emitted) x (growth in thinker ms/token).")
    print("  Everything on the right-hand side is measured, so this bound is assumption-free.")
    print()
    print(f"  {'arm':10s} {'prompt tok':>18s} {'thinker ms/tok':>16s} {'text out':>9s} "
          f"{'talker growth':>14s} {'max from wait':>14s} {'-> at most':>11s}")
    for name in ("T640", "P640", "C640"):
        rs = data[name]
        bs = binned(rs)
        if len(bs) < 2:
            continue
        first, last = bs[0][1], bs[-1][1]
        p0, p1 = med(r["prompt_tokens"] for r in first), med(r["prompt_tokens"] for r in last)
        i0, i1 = med(r["thinker_itl_ms"] for r in first), med(r["thinker_itl_ms"] for r in last)
        t0, t1 = med(r["talker_add_ms"] for r in first), med(r["talker_add_ms"] for r in last)
        ntext = max(med(r["text_tokens_out"] for r in first), med(r["text_tokens_out"] for r in last))
        grow = t1 - t0
        maxwait = ntext * (i1 - i0)
        frac = (maxwait / grow * 100) if grow > 1 else float("nan")
        print(f"  {name:10s} {f'{p0:.0f}->{p1:.0f}':>18s} {f'{i0:.2f}->{i1:.2f}':>16s} "
              f"{ntext:9.0f} {grow:12.0f}ms {maxwait:12.0f}ms {frac:10.0f}%")
    print()
    print("  So for T640 at least 100-that% of the talker's growth is the TALKER's own work.")

    print()
    print("=" * 78)
    print("CONTROL 2  -- does the TALKER's own decode slow with context? (it should, if the")
    print("             remaining cost were attention over the cached prefix)")
    print("=" * 78)
    print(f"  {'arm':10s} {'prompt tok':>18s} {'talker ms/token':>18s}   verdict")
    for name in ("C640", "T640", "P640", "C640_low"):
        rs = data[name]
        bs = binned(rs)
        if len(bs) < 2:
            continue
        first, last = bs[0][1], bs[-1][1]
        p0, p1 = med(r["prompt_tokens"] for r in first), med(r["prompt_tokens"] for r in last)
        k0, k1 = med(r["talker_itl_ms"] for r in first), med(r["talker_itl_ms"] for r in last)
        v = "FLAT / falls" if k1 <= k0 * 1.05 else f"grows {k1/k0:.2f}x"
        print(f"  {name:10s} {f'{p0:.0f}->{p1:.0f}':>18s} {f'{k0:.2f}->{k1:.2f}':>18s}   {v}")
    print()
    print("  The talker's per-token decode does NOT slow as the context grows. So whatever")
    print("  the residual cost is, it is NOT the talker attending over a long cached prefix.")
    print("  Mechanism therefore UNKNOWN -- do not invent one. (A previous mechanism story")
    print("  in this study, 'batch_size==1 proves nothing batches', was refuted.)")

    print()
    print("=" * 78)
    print("CONTROL 3  -- matched-prompt comparison: PC off vs PC on at the same prompt")
    print("             length and nearly the same thinker decode rate, so the waiting")
    print("             term cancels")
    print("=" * 78)
    t640, p640 = binned(data["T640"]), binned(data["P640"])

    def pick(bs, target):
        best, bd = None, None
        for (a, b), g in bs:
            p = med(r["prompt_tokens"] for r in g)
            d = abs(p - target)
            if bd is None or d < bd:
                best, bd = ((a, b), g, p), d
        return best

    print(f"  {'target':>8s} | {'T640 (PC off)':>34s} | {'P640 (PC ON)':>34s} | {'saved':>7s}")
    print(f"  {'':>8s} | {'turns':>7s} {'tok':>7s} {'itl':>5s} {'talker':>7s}      | "
          f"{'turns':>7s} {'tok':>7s} {'itl':>5s} {'talker':>7s}      | {'':>7s}")
    savings = []
    for target in (10000, 15000, 20000, 25000, 30000, 34000):
        (a1, b1), g1, p1 = pick(t640, target)
        (a2, b2), g2, p2 = pick(p640, target)
        if abs(p1 - target) > 6000 or abs(p2 - target) > 6000:
            continue
        y1, y2 = med(r["talker_add_ms"] for r in g1), med(r["talker_add_ms"] for r in g2)
        i1, i2 = med(r["thinker_itl_ms"] for r in g1), med(r["thinker_itl_ms"] for r in g2)
        y2n = y2 * p1 / p2                      # normalise P640 to T640's prompt length
        save = (1 - y2n / y1) * 100
        savings.append(save)
        print(f"  {target:8d} | {f'{a1}-{b1}':>7s} {p1:7.0f} {i1:5.1f} {y1:6.0f}ms      | "
              f"{f'{a2}-{b2}':>7s} {p2:7.0f} {i2:5.1f} {y2:6.0f}ms      | {save:6.0f}%")
    if savings:
        print()
        print(f"  Turning the talker's prefix cache ON saves a consistent "
              f"{min(savings):.0f}-{max(savings):.0f}% (median {st.median(savings):.0f}%)")
        print("  of the talker's added latency, at matched prompt length. It does NOT")
        print("  remove the dependence on prompt length.")

    print()
    print("=" * 78)
    print("THE NUMBER -- talker ms per 1,000 prompt tokens")
    print("=" * 78)
    print(f"  {'arm':10s} {'cache':>6s} {'n':>4s} {'prompt range':>16s} {'ms/1k':>9s} "
          f"{'floor':>9s} {'R2':>7s}")
    for name, pc, tokmax in (("T640", "off", None), ("P640", "ON", 50000),
                             ("C640_low", "off", None)):
        rs = data[name]
        pts = [(r["prompt_tokens"], r["talker_add_ms"]) for r in rs
               if r["prompt_tokens"] and r["talker_add_ms"] is not None
               and (tokmax is None or r["prompt_tokens"] <= tokmax)]
        if len(pts) < 10:
            continue
        f = ols(pts)
        if not f:
            continue
        slope, icpt, r2 = f
        print(f"  {name:10s} {pc:>6s} {len(pts):4d} "
              f"{f'{min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f}':>16s} "
              f"{1000*slope:8.1f} {icpt:8.0f} {r2:7.4f}")
    print()
    print("  For comparison, the THINKER's marginal cost for a token already in its prefix")
    print("  cache is 1.26 ms/1k (measured earlier, against 79.2 ms/1k for an all-new token).")

    print()
    print("=" * 78)
    print("WHAT THIS IMPLIES FOR THE PROPOSAL -- best case arithmetic")
    print("=" * 78)
    rs = data["T640"]
    last = binned(rs)[-1][1]
    P = med(r["prompt_tokens"] for r in last)
    meas_th = med(r["thinker_ttft_ms"] for r in last)
    meas_tk = med(r["talker_add_ms"] for r in last)
    meas_c2 = med(r["code2wav_add_ms"] for r in last)
    meas_tt = med(r["audio_ttfa_ms"] for r in last)
    # best case: thinker pays only its measured fixed floor; talker pays the PC-on rate
    pts = [(r["prompt_tokens"], r["talker_add_ms"]) for r in data["P640"]
           if r["prompt_tokens"] and r["talker_add_ms"] is not None and r["prompt_tokens"] <= 50000]
    slope, icpt, _ = ols(pts)
    best_th = 62.5
    best_tk = icpt + slope * P
    best_tt = best_th + best_tk + meas_c2
    c640 = binned(data["C640"])[-1][1]
    c640_tt = med(r["audio_ttfa_ms"] for r in c640)
    c640_p = med(r["prompt_tokens"] for r in c640)
    print(f"  T640 at its last 10 turns, prompt = {P:.0f} tokens")
    print(f"    {'':22s} {'measured':>10s}   {'best case':>10s}")
    print(f"    {'thinker':22s} {meas_th:9.0f}ms {best_th:9.0f}ms   (perfect incremental -> the 62.5 ms floor)")
    print(f"    {'talker':22s} {meas_tk:9.0f}ms {best_tk:9.0f}ms   (the PC-on rate applied to the whole prompt)")
    print(f"    {'code2wav':22s} {meas_c2:9.0f}ms {meas_c2:9.0f}ms   (unaffected)")
    print(f"    {'TTFA':22s} {meas_tt:9.0f}ms {best_tt:9.0f}ms")
    print()
    print(f"  C640, same content, prompt PINNED at {c640_p:.0f} tokens:  TTFA {c640_tt:.0f} ms measured")
    print()
    print(f"  => even a PERFECT incremental prefill on both stages leaves the append-only arm")
    print(f"     at ~{best_tt:.0f} ms, still {best_tt/c640_tt:.1f}x worse than simply keeping the prompt short.")
    print(f"     The talker charges ~{1000*slope:.0f} ms per 1,000 prompt tokens even with its cache on,")
    print(f"     and {P:.0f} tokens x that rate is {best_tk:.0f} ms on its own.")

    print()
    print("=" * 78)
    print("CAVEATS -- read before quoting")
    print("=" * 78)
    print("  1. P640's stage-1 prefix cache may be UNSOUND. The talker's prompt ids are")
    print("     [0]*prompt_len (qwen3_omni.py:710) with multi_modal_data=None, so two")
    print("     different requests of equal length present identical token ids to the block")
    print("     hasher. If blocks collide, P640 skipped work it should have done, which makes")
    print("     its rate an OPTIMISTIC lower bound -- the conclusion above only gets stronger.")
    print("     Not yet verified either way.")
    print("  2. T640 and P640 differ in the EVS gap bracket ([8,16] vs [4,10]) as well as in")
    print("     the cache. That changes how fast the prompt grows, not the cost per token, so")
    print("     comparing SLOPES and matched-prompt points is sound; comparing raw per-turn")
    print("     medians between the two arms is not.")
    print("  3. The residual mechanism is unknown. The talker's per-token decode is flat, so")
    print("     it is not attention over the cached prefix. transfers=[0->1=0.00ms] in the")
    print("     OmniTiming line is almost certainly UNMEASURED rather than genuinely zero, so")
    print("     the connector is not ruled out.")
    print("  4. This says what PREFIX CACHING leaves behind. The engine's true incremental")
    print("     path (add_streaming_update_async, resumable=True) is a DIFFERENT mechanism")
    print("     and has never been exercised by the video endpoint. It could do better.")
    print("  5. Response length differs across arms (text_out 52 for C640 vs 90 for T640),")
    print("     so TTFA comparisons across arms carry that confound. The within-arm slopes do")
    print("     not: text_out is constant within each arm across all turn bins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
