#!/usr/bin/env python3
"""What happens between the first text token and the first sound?

The tail sweep left this unexplained. At 8 users the gap is 369 ms for a static
screen and 2,532 ms for a walking camera -- a 6.9x spread on a segment that, if the
three stages really pipeline, should barely move.

The gap CANNOT be split using response.text.done: the server sends that message
from inside the audio branch, immediately before the first audio chunk
(video_stream_base.py:658), so first_text -> text_done equals first_text ->
first_audio identically. Splitting on it returns exactly 0 ms of talker+code2wav in
every arm, which looks like a clean result and is in fact a tautology.

So the run this reads used --trace-deltas, timestamping every text delta. Text
deltas are emitted as the thinker produces them, which makes the thinker observable
token by token and splits the gap for real:

    first text delta ----[ thinker still emitting text ]----> last text delta
                                                              before the sound
                     ----[ nothing arriving: talker + code2wav ]----> first audio

Two hypotheses, opposite fixes:

  A  thinker-bound. Text keeps arriving right up to the sound, and the gap between
     consecutive text deltas grows with prompt size. Then video hurts because every
     thinker decode step attends over 16 frames of context, and the fix is fewer
     frames in the prompt.

  B  talker-bound. Text stops early and then there is a long silence. Then the fix
     is stage 1 / stage 2 capacity, and cutting frames would not help.

The 1-user arms are the no-contention reference: what survives there is the
architecture, and the 1 -> 8 user difference is contention.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st

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


def ms(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.0f}"


def turns(res: pathlib.Path, tag: str) -> list[dict]:
    """Reassemble per-turn delta timelines from the raw traces."""
    out = []
    for p in sorted(glob.glob(str(res / f"rp_{tag}" / "ttfa_user*.jsonl"))):
        by_rep: dict = {}
        for line in open(p):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k, r = e.get("k"), e.get("rep")
            if k == "tx_query":
                by_rep.setdefault(r, {})["q"] = e["w"]
            elif k == "rx_text_delta":
                by_rep.setdefault(r, {}).setdefault("txt", []).append((e["w"], e.get("nchars", 0)))
            elif k == "rx_first_audio":
                by_rep.setdefault(r, {})["fa"] = e["w"]
            elif k == "rx_audio_delta":
                by_rep.setdefault(r, {}).setdefault("aud", []).append(e["w"])
        for r, d in by_rep.items():
            if r is None or r == 0 or "fa" not in d or not d.get("txt"):
                continue                      # rep 0 is warm-up, as elsewhere
            d["txt"].sort()
            out.append({"rep": r, **d})
    return out


def analyse(ts: list[dict]) -> dict | None:
    if not ts:
        return None
    ramp, streaming, silence, frac_before, n_before, n_total = [], [], [], [], [], []
    gaps_all, last_gap = [], []
    for t in ts:
        txt, fa = t["txt"], t["fa"]
        t0 = txt[0][0]
        before = [w for w, _ in txt if w <= fa]
        if not before:
            continue
        ramp.append((fa - t0) * 1000)
        streaming.append((before[-1] - t0) * 1000)   # text was still arriving
        silence.append((fa - before[-1]) * 1000)     # nothing arriving: stage 1+2
        n_before.append(len(before))
        n_total.append(len(txt))
        frac_before.append(len(before) / len(txt))
        d = [(b - a) * 1000 for a, b in zip(before, before[1:])]
        if d:
            gaps_all.append(st.median(d))
            last_gap.append(d[-1])
    if not ramp:
        return None
    return {"n": len(ramp),
            "ramp": pct(ramp, .5), "streaming": pct(streaming, .5),
            "silence": pct(silence, .5), "silence_p95": pct(silence, .95),
            "n_before": st.median(n_before), "n_total": st.median(n_total),
            "frac_before": st.median(frac_before),
            "inter_delta": pct(gaps_all, .5), "last_gap": pct(last_gap, .5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--users", default="1,8")
    ap.add_argument("--out", default="/data/zx/results/ramp_probe.json")
    a = ap.parse_args()
    res = pathlib.Path(a.results)
    users = [int(x) for x in a.users.split(",")]

    arms: dict = {}
    for u in users:
        for c in CONTENTS:
            d = analyse(turns(res, f"u{u}_{c}"))
            if d:
                arms[(u, c)] = d
    if not arms:
        print("no rp_* traces with deltas found")
        return 1

    print("\n" + "=" * 100)
    print("FIRST TEXT -> FIRST SOUND, split by what was actually arriving")
    print("=" * 100)
    print(f"{'arm':<26}{'n':>4}{'text still coming':>19}{'then silence':>14}"
          f"{'= gap':>9}{'silence share':>15}")
    for u in users:
        for c in CONTENTS:
            d = arms.get((u, c))
            if not d:
                continue
            print(f"{f'{u}u {LABEL[c]}':<26}{d['n']:>4}{ms(d['streaming'], 19)}"
                  f"{ms(d['silence'], 14)}{ms(d['ramp'], 9)}"
                  f"{d['silence']/d['ramp']:>14.0%}")
    print("  'text still coming'  first text delta -> last text delta before the sound")
    print("  'then silence'       last text delta -> first audio: stage 1 + stage 2")

    print("\n--- how much of the answer was already written when the sound started? ---")
    print(f"{'arm':<26}{'deltas before sound':>21}{'deltas total':>14}{'share':>9}")
    for u in users:
        for c in CONTENTS:
            d = arms.get((u, c))
            if not d:
                continue
            print(f"{f'{u}u {LABEL[c]}':<26}{d['n_before']:>21.0f}"
                  f"{d['n_total']:>14.0f}{d['frac_before']:>8.0%}")
    print("  Near 100% means the talker waits for a finished answer -- the stages do")
    print("  not really overlap. Well under 100% means they do.")

    print("\n--- the thinker's own pace: time between consecutive text deltas ---")
    print(f"{'arm':<26}{'p50 gap':>10}{'last gap':>11}{'deltas':>9}"
          f"{'implied text time':>19}")
    for u in users:
        for c in CONTENTS:
            d = arms.get((u, c))
            if not d:
                continue
            print(f"{f'{u}u {LABEL[c]}':<26}{ms(d['inter_delta'], 10)}"
                  f"{ms(d['last_gap'], 11)}{d['n_before']:>9.0f}"
                  f"{ms(d['inter_delta'] * (d['n_before'] - 1), 19)}")
    print("  If p50 gap grows with prompt size, thinker decode is the thing video")
    print("  makes expensive, and it is paid once per token of the answer.")

    print("\n--- contention: what does 1 -> 8 users do to each part? ---")
    print(f"{'content':<20}{'text-streaming part':>21}{'silence part':>15}{'gap':>9}")
    for c in CONTENTS:
        lo, hi = arms.get((min(users), c)), arms.get((max(users), c))
        if not (lo and hi):
            continue
        r_str = f"{hi['streaming'] / lo['streaming']:.2f}x"
        r_sil = f"{hi['silence'] / lo['silence']:.2f}x" if lo["silence"] else "-"
        r_gap = f"{hi['ramp'] / lo['ramp']:.2f}x"
        print(f"{LABEL[c]:<20}{r_str:>21}{r_sil:>15}{r_gap:>9}")

    print("\n--- verdict ---")
    for u in users:
        for c in CONTENTS:
            d = arms.get((u, c))
            if not d:
                continue
            share = d["silence"] / d["ramp"]
            v = ("THINKER-BOUND: the answer is still being written"
                 if share < 0.35 else
                 "TALKER-BOUND: text finished, stages 1-2 are the wait"
                 if share > 0.65 else "MIXED")
            print(f"  {f'{u}u {LABEL[c]}':<20} silence {share:>4.0%} of the gap  ->  {v}")

    pathlib.Path(a.out).write_text(json.dumps(
        {f"u{u}_{c}": d for (u, c), d in arms.items()}, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
