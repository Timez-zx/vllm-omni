#!/usr/bin/env python3
"""Why does camera motion stretch the first-text-to-first-sound gap?

Established already, from timestamping every text delta:

  * The gap is 99% "the thinker is still writing". Stage 1 (talker) plus stage 2
    (code2wav) contribute 16-20 ms, and that figure is nearly constant across
    content -- 16 ms static, 19 ms low motion, 20 ms high motion. The speech path is
    not the bottleneck in any condition.

  * High motion does NOT make every token uniformly more expensive. The median
    inter-token pause is 33.6 ms against static's 22.6 ms -- only 1.5x. But the MEAN
    is 91.7 ms. The damage is in discrete pauses, not a graded slowdown, and the
    "2.7x per token" figure the server's session-average inter_output_latency_ms
    reports is that mean, which misdescribes the mechanism.

  * The pause-size histogram is BIMODAL: at 8 users, high motion spends 43% of the
    wait in pauses of 200-500 ms (about 5 per turn) and 25% in pauses over 1 s
    (about 0.5 per turn), with the 500-1000 ms bucket EMPTY. A gap in the middle of
    a distribution is the signature of a discrete blocking event, not of gradual
    contention.

WHAT IS NOT ESTABLISHED, and what this script is for.

The natural explanation -- my decode step is stuck behind other users' prefills --
was tested against a null model (an equal-length interval placed at random inside
the same turn's own wait) and FAILED: 4.71 other users prefilling during a stall
versus 5.20 expected by chance, ratio 0.91. It failed because at 8 users about 5 of
the 7 others are prefilling essentially all the time, so "someone is prefilling"
cannot discriminate a stalled instant from a running one. Two arrival-based variants
also came back null or negative.

The one strong signal was positional: stalls sit at the very start of the text
stream (median position 0.00 of the span), i.e. right at the prefill-to-decode
handover.

So the discriminating experiment is not a better correlation on the 8-user data --
it is the 1-USER arm at the same content and the same 14.5k-token prompt:

    stalls VANISH at 1 user  -> they are caused by the other users
    stalls PERSIST at 1 user -> they are intrinsic to a long prompt, and every
                                multi-user story told about them is wrong

Both arms carry --trace-deltas, same server config, same stimulus, so this is a
clean single-variable comparison. Nothing else needs to be instrumented.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st

BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 1e9)]
BNAMES = ["<50ms", "50-100", "100-200", "200-500", "500-1k", ">1s"]
CONTENTS = ["static", "low", "high"]
LABEL = {"static": "static screen", "low": "low motion", "high": "high motion"}


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def turns(pattern: str) -> list[dict]:
    """Per-turn delta timeline. rep 0 dropped as warm-up, as everywhere else."""
    out = []
    for p in sorted(glob.glob(pattern)):
        by: dict = {}
        for line in open(p):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k, r = e.get("k"), e.get("rep")
            if k == "rx_text_delta":
                by.setdefault(r, {}).setdefault("t", []).append(e["w"])
            elif k == "rx_first_audio":
                by.setdefault(r, {})["fa"] = e["w"]
            elif k == "rx_first_text":
                by.setdefault(r, {})["ft"] = e["w"]
        for r, d in by.items():
            if r and r >= 1 and "fa" in d and "t" in d:
                before = sorted(w for w in d["t"] if w <= d["fa"])
                if len(before) >= 2:
                    out.append({"before": before,
                                "gaps": [(b - a) * 1000
                                         for a, b in zip(before, before[1:])],
                                "fa": d["fa"], "n_all": len(d["t"])})
    return out


def describe(ts: list[dict]) -> dict | None:
    if not ts:
        return None
    gaps = [g for t in ts for g in t["gaps"]]
    span = [sum(t["gaps"]) for t in ts]
    time_in, count_in = [0.0] * len(BUCKETS), [0] * len(BUCKETS)
    for g in gaps:
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= g < hi:
                time_in[i] += g
                count_in[i] += 1
                break
    tot = sum(time_in) or 1.0
    # where in the wait do the big pauses sit? 0 = at the first token, 1 = at the sound
    pos = []
    for t in ts:
        s = t["before"][-1] - t["before"][0]
        if s <= 0:
            continue
        acc = t["before"][0]
        for g in t["gaps"]:
            if g > 200:
                pos.append((acc - t["before"][0]) / s)
            acc += g / 1000.0
    return {"n_turns": len(ts), "n_gaps": len(gaps),
            "span_p50": pct(span, .5), "tokens_p50": pct([len(t["gaps"]) + 1 for t in ts], .5),
            "mean": st.fmean(gaps), "p50": pct(gaps, .5), "p90": pct(gaps, .9),
            "p99": pct(gaps, .99), "max": max(gaps),
            "time_share": [x / tot for x in time_in],
            "per_turn_count": [c / len(ts) for c in count_in],
            "big_pause_pos_p50": pct(pos, .5) if pos else None}


def ms(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--out", default="/data/zx/results/thinker_stalls.json")
    a = ap.parse_args()
    res = pathlib.Path(a.results)

    # 8-user arms come from the short ramp probe. For 1 user, prefer the long tail run
    # (4 sequential sessions, ~236 turns) and fall back to the short probe's own
    # 1-user arm (14 turns) so a preview is available before the long run lands.
    # Both use --trace-deltas and the same shipped config.
    arms: dict = {}
    src: dict = {}
    for c in CONTENTS:
        d8 = describe(turns(str(res / f"rp_u8_{c}" / "ttfa_user*.jsonl")))
        if d8:
            arms[(8, c)] = d8
            src[(8, c)] = "probe"
        long_ = describe(turns(str(res / f"vt_u1_{c}" / "ttfa_user*.jsonl")))
        short = describe(turns(str(res / f"rp_u1_{c}" / "ttfa_user*.jsonl")))
        if long_ and long_["n_turns"] >= 40:
            arms[(1, c)] = long_
            src[(1, c)] = "tail run"
        elif short:
            arms[(1, c)] = short
            src[(1, c)] = "PREVIEW (short probe, few turns)"
        elif long_:
            arms[(1, c)] = long_
            src[(1, c)] = "PARTIAL tail run"
    if not arms:
        print("no delta-traced arms found")
        return 1
    have = sorted(arms, key=lambda k: (k[0], CONTENTS.index(k[1])))
    print("arms with delta traces:")
    for u, c in have:
        print(f"  {u}u {LABEL[c]:<16} {arms[(u, c)]['n_turns']:>4} turns   "
              f"source: {src[(u, c)]}")

    print("\n" + "=" * 104)
    print("INTER-TOKEN PAUSES BEFORE THE FIRST SOUND")
    print("=" * 104)
    print(f"{'arm':<22}{'turns':>7}{'span':>8}{'tokens':>8}{'mean':>7}{'p50':>7}"
          f"{'p90':>7}{'p99':>8}{'max':>7}{'mean/p50':>10}")
    for k in have:
        d = arms[k]
        print(f"{f'{k[0]}u {LABEL[k[1]]}':<22}{d['n_turns']:>7}{ms(d['span_p50'])}"
              f"{d['tokens_p50']:>8.0f}{ms(d['mean'], 7)}{ms(d['p50'], 7)}"
              f"{ms(d['p90'], 7)}{ms(d['p99'], 8)}{ms(d['max'], 7)}"
              f"{d['mean']/d['p50']:>10.2f}")
    print("  mean/p50 near 1.0 = smooth generation. Well above 1.0 = punctuated by")
    print("  discrete stalls, and then the mean is a misleading summary.")

    print("\n--- share of the wait spent in pauses of each size ---")
    print(f"{'arm':<22}" + "".join(f"{n:>10}" for n in BNAMES))
    for k in have:
        print(f"{f'{k[0]}u {LABEL[k[1]]}':<22}"
              + "".join(f"{s:>9.0%}" for s in arms[k]["time_share"]))

    print("\n--- how many pauses of each size, per turn ---")
    print(f"{'arm':<22}" + "".join(f"{n:>10}" for n in BNAMES))
    for k in have:
        print(f"{f'{k[0]}u {LABEL[k[1]]}':<22}"
              + "".join(f"{c:>10.2f}" for c in arms[k]["per_turn_count"]))
    print("  An EMPTY bucket between two populated ones is the signature of a")
    print("  discrete blocking event rather than graded contention.")

    print("\n" + "=" * 104)
    print("THE DISCRIMINATING TEST: does the stall survive with nobody else on the GPU?")
    print("=" * 104)
    print("  Same content, same ~14.5k-token prompt, same config. Only the number of")
    print("  users differs, so this attributes the stalls or refutes the attribution.")
    print(f"\n{'content':<16}{'1u stalls/turn':>16}{'8u stalls/turn':>16}"
          f"{'1u share of wait':>18}{'8u share of wait':>18}{'verdict':>26}")
    for c in CONTENTS:
        one, eight = arms.get((1, c)), arms.get((8, c))
        if not (one and eight):
            print(f"{LABEL[c]:<16}{'(waiting for the 1-user arm)' if not one else '':>16}")
            continue
        # a "stall" is anything at or above the 200 ms bucket
        s1 = sum(one["per_turn_count"][3:])
        s8 = sum(eight["per_turn_count"][3:])
        f1 = sum(one["time_share"][3:])
        f8 = sum(eight["time_share"][3:])
        if s8 >= 0.5 and s1 <= 0.15 * s8:
            v = "CAUSED BY OTHER USERS"
        elif s8 >= 0.5 and s1 >= 0.6 * s8:
            v = "INTRINSIC to long prompt"
        elif s8 < 0.5:
            v = "no stalls to explain"
        else:
            v = "MIXED"
        print(f"{LABEL[c]:<16}{s1:>16.2f}{s8:>16.2f}{f1:>17.0%}{f8:>17.0%}{v:>26}")

    print("\n--- where the big pauses (>200 ms) sit in the wait (0 = first token) ---")
    for k in have:
        p = arms[k]["big_pause_pos_p50"]
        if p is not None:
            print(f"  {f'{k[0]}u {LABEL[k[1]]}':<22} median position {p:.2f}")
    print("  Near 0.00 means the stall is at the prefill-to-decode handover.")

    pathlib.Path(a.out).write_text(json.dumps(
        {f"u{u}_{c}": d for (u, c), d in arms.items()}, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
