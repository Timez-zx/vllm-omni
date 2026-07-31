#!/usr/bin/env python3
"""What does a session roll cost, and does it keep the conversation alive?

A session-scoped request dies when the talker's stored token array reaches stage 1's
max_model_len. Rolling retires the request while it is still healthy and opens a fresh one
seeded with the recent text transcript. That buys an unbounded session and charges for it in
two places, both measured here:

  1. ONE COLD TURN per roll. The new request prefills the seed from nothing, so the turn
     immediately after a roll should be slower than its neighbours. Reported as the post-roll
     TTFA against the steady-state distribution, because an average over all turns would hide
     exactly the cost being paid.
  2. The accumulated VISUAL context. Text crosses a roll; KV does not. Not measurable from a
     latency trace -- it needs the recall protocol.

The roll turns are read from the client trace (`rx_session_rolled`), not from the server log,
so the timing and the attribution come from the same clock.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

RES = pathlib.Path("/data/zx/results")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="vt_u1_ROLL")
    ap.add_argument("--log", default="server_roll.log")
    args = ap.parse_args()

    turns: dict[int, dict] = {}
    rolled_at: set[int] = set()
    for f in sorted(glob.glob(str(RES / args.trace / "ttfa_user*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rep = r.get("rep", -1)
            k = r["k"]
            if k == "rx_session_rolled":
                # The event fires while the turn that TRIGGERED the roll is being set up, so
                # that same turn is the one paying the cold prefill.
                if rep >= 0:
                    rolled_at.add(rep)
                continue
            if rep < 0:
                continue
            if k in ("tx_query", "rx_first_text", "rx_first_audio"):
                turns.setdefault(rep, {})[k] = r["w"]

    rows = []
    for rep, d in sorted(turns.items()):
        if not all(x in d for x in ("tx_query", "rx_first_text", "rx_first_audio")):
            continue
        rows.append({
            "rep": rep,
            "thinker": (d["rx_first_text"] - d["tx_query"]) * 1000,
            "talker": (d["rx_first_audio"] - d["rx_first_text"]) * 1000,
            "ttfa": (d["rx_first_audio"] - d["tx_query"]) * 1000,
            "rolled": rep in rolled_at,
        })
    if not rows:
        print(f"no scored turns in {args.trace}")
        return 1

    # Server-side: the roll lines and the talker estimate, to confirm the saw-tooth.
    p = RES / args.log
    rolls_srv, est = [], []
    if p.exists():
        lines = [ANSI.sub("", x) for x in p.open(errors="replace")]
        mk = [i for i, x in enumerate(lines) if x.startswith("===== ROLL")]
        seg = lines[mk[-1]:] if mk else lines
        for x in seg:
            m = re.search(r"ROLL #(\d+) at turn=(\d+): talker was at ~(\d+) tokens", x)
            if m:
                rolls_srv.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            m = re.search(r"talker_est=(\d+)", x)
            if m:
                est.append(int(m.group(1)))

    print("=" * 88)
    print("DID THE SESSION SURVIVE?")
    print("=" * 88)
    print(f"  turns scored (rep>=1) : {len(rows)}")
    print(f"  rolls seen by client  : {len(rolled_at)}   at turns {sorted(rolled_at)}")
    if rolls_srv:
        print("  rolls seen by server  :")
        for n, turn, tok in rolls_srv:
            print(f"     #{n} at turn {turn}, talker was at ~{tok} tokens")
    if est:
        print(f"  talker_est peak        : {max(est)}   (stage-1 max_model_len is 65,536)")
        print(f"  talker_est saw-tooth   : {est[:6]} ... {est[-4:]}")

    normal = [r for r in rows if not r["rolled"]]
    cold = [r for r in rows if r["rolled"]]

    print()
    print("=" * 88)
    print("WHAT A ROLL COSTS -- the turn that pays the cold prefill")
    print("=" * 88)
    print(f"  {'group':16s} {'n':>4s} {'thinker':>9s} {'talker':>9s} {'TTFA p50':>9s} "
          f"{'TTFA max':>9s}")
    for lbl, g in (("steady state", normal), ("post-roll turn", cold)):
        if not g:
            print(f"  {lbl:16s} {0:4d}        --        --        --        --")
            continue
        print(f"  {lbl:16s} {len(g):4d} "
              f"{st.median([x['thinker'] for x in g]):8.0f}ms "
              f"{st.median([x['talker'] for x in g]):8.0f}ms "
              f"{st.median([x['ttfa'] for x in g]):8.0f}ms "
              f"{max(x['ttfa'] for x in g):8.0f}ms")
    if normal and cold:
        ratio = st.median([x["ttfa"] for x in cold]) / st.median([x["ttfa"] for x in normal])
        print()
        print(f"  A roll turn costs {ratio:.2f}x the steady-state TTFA.")
        print(f"  Amortised over the roll interval it is "
              f"{(st.median([x['ttfa'] for x in cold]) - st.median([x['ttfa'] for x in normal])) / max(1, len(rows) / max(1, len(cold))):.0f} ms per turn.")
        print("  Read the cold turn as the price of the session continuing at all: without a")
        print("  roll the same settings ended the conversation outright at turn 27.")

    print()
    print("=" * 88)
    print("PER-TURN")
    print("=" * 88)
    print(f"  {'turn':>5s} {'thinker':>9s} {'talker':>9s} {'TTFA':>8s}   roll")
    for r in rows:
        print(f"  {r['rep']:5d} {r['thinker']:8.0f}ms {r['talker']:8.0f}ms {r['ttfa']:7.0f}ms"
              f"   {'<-- ROLLED' if r['rolled'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
