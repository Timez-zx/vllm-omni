#!/usr/bin/env python3
"""What did the engine actually emit, per turn, under session mode?

Written for one specific unexplained failure. In fork run A the session died at turn 28:
the client received first_text and then no audio, the turn boundary never arrived, and the
server logged no error, no traceback, no length violation and no preemption. Nothing in the
default log says whether the engine stopped emitting audio outputs or kept emitting them
without the per-segment finish_reason that the boundary is detected from -- and those two
have completely different causes.

So this reads the VLLM_OMNI_LOG_SESSION_OUTPUTS=1 trace, which logs one line per output, and
answers per turn:

  * how many text and audio outputs arrived, and from which stage
  * how many carried a finish_reason (the boundary signal; exactly one audio one is expected)
  * whether audio arrived at all

TURN ATTRIBUTION. Outputs are assigned to turns by the "[session] turn=N done" lines, which
are emitted by the very code path being investigated -- so a turn whose boundary was MISSED
has no done line and its outputs would otherwise be silently merged into the next turn's
bucket. That is precisely the failure being looked for, so outputs after the last done line
are reported separately as a trailing bucket rather than attributed to any turn.

The trace is per-output and therefore large. Nothing here loads the whole file into memory
twice; it makes one pass.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
OUT_RX = re.compile(
    r"\[session-out\] type=(\S+) stage=(\S+) out\.finished=(\S+) ro\.finished=(\S+) "
    r"finish_reason=(\S+) ntok=(\S+) audio_n=(\S+)"
)
DONE_RX = re.compile(r"\[session\] turn=(\d+) done first_text=")
DELTA_RX = re.compile(r"\[session\] turn=(\d+) queue delta: (\d+) new frames, (\d+) tokens, "
                      r"cum=(\d+), talker_placeholder=(-?\d+)")
NEVER_RX = re.compile(r"\[session\] turn=(\d+) boundary NEVER ARRIVED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/data/zx/results/server_fork.log")
    ap.add_argument("--marker", default="===== FORK")
    ap.add_argument("--boot", type=int, default=-1,
                    help="which boot to read, -1 = last (the arm)")
    args = ap.parse_args()

    p = pathlib.Path(args.log)
    if not p.exists():
        print(f"no such log: {p}")
        return 1
    lines = [ANSI.sub("", l) for l in p.open(errors="replace")]
    marks = [i for i, l in enumerate(lines) if l.startswith(args.marker)]
    if not marks:
        print(f"no boot marker {args.marker!r} in {p}")
        return 1
    lo = marks[args.boot]
    hi = marks[args.boot + 1] if args.boot != -1 and args.boot + 1 < len(marks) else len(lines)
    seg = lines[lo:hi]
    print(f"boot: {seg[0].strip()[:66]}   ({len(seg)} lines)")

    turns: list[dict] = []
    cur = {"turn": None, "text": 0, "audio": 0, "fr_text": 0, "fr_audio": 0,
           "stages": set(), "audio_units": 0}
    deltas: dict[int, tuple[int, int, int]] = {}
    never: list[int] = []

    def flush(turn_id) -> None:
        cur["turn"] = turn_id
        turns.append(dict(cur, stages="".join(sorted(cur["stages"]))))
        for k in ("text", "audio", "fr_text", "fr_audio", "audio_units"):
            cur[k] = 0
        cur["stages"] = set()

    for l in seg:
        m = OUT_RX.search(l)
        if m:
            kind, stage, _of, _rf, fr, _nt, an = m.groups()
            cur["stages"].add(stage)
            if kind == "audio":
                cur["audio"] += 1
                if fr != "None":
                    cur["fr_audio"] += 1
                try:
                    cur["audio_units"] += int(an)
                except ValueError:
                    pass
            else:
                cur["text"] += 1
                if fr != "None":
                    cur["fr_text"] += 1
            continue
        m = DONE_RX.search(l)
        if m:
            flush(int(m.group(1)))
            continue
        m = DELTA_RX.search(l)
        if m:
            deltas[int(m.group(1))] = (int(m.group(2)), int(m.group(4)), int(m.group(5)))
            continue
        m = NEVER_RX.search(l)
        if m:
            never.append(int(m.group(1)))

    trailing = dict(cur, stages="".join(sorted(cur["stages"])))

    # Distinguish "the trace is absent" from "the engine emitted nothing". Without this the
    # per-turn flags below fire on every row of a log that simply predates the env var, which
    # reads as a catastrophic failure of a run that was in fact fine.
    traced = any(OUT_RX.search(l) for l in seg)
    if not traced:
        print("\nNO [session-out] lines in this boot -- the trace was not enabled, so what the")
        print("engine emitted per turn is unknowable from this log. The turn table below is")
        print("still shown, WITHOUT per-turn output flags, since there is nothing to flag.")
        print("Re-run the server with VLLM_OMNI_LOG_SESSION_OUTPUTS=1 to get the diagnosis.")
        print(f"\n{'turn':>5s} {'frames':>7s} {'cum':>7s} {'plchold':>8s}")
        for t in turns:
            nf, cum, ph = deltas.get(t["turn"], (0, 0, 0))
            print(f"{t['turn']:5d} {nf:7d} {cum:7d} {ph:8d}")
        if never:
            print(f"\nwatchdog fired for turn(s): {never}")
        return 1

    print(f"\n{'turn':>5s} {'frames':>7s} {'cum':>7s} {'plchold':>8s} "
          f"{'text_out':>9s} {'audio_out':>10s} {'fr_audio':>9s} {'audio_units':>12s} {'stages':>7s}")
    for t in turns:
        nf, cum, ph = deltas.get(t["turn"], (0, 0, 0))
        flag = ""
        if t["audio"] == 0:
            flag = "  <- NO AUDIO OUTPUT AT ALL"
        elif t["fr_audio"] == 0:
            flag = "  <- audio but NO finish_reason: boundary unreachable"
        elif t["fr_audio"] > 1:
            flag = f"  <- {t['fr_audio']} finish_reasons: turn closed early"
        print(f"{t['turn']:5d} {nf:7d} {cum:7d} {ph:8d} {t['text']:9d} {t['audio']:10d} "
              f"{t['fr_audio']:9d} {t['audio_units']:12d} {t['stages']:>7s}{flag}")

    if trailing["text"] or trailing["audio"]:
        print(f"\nAFTER the last completed turn (this is the failure, if there was one):")
        print(f"      {'':7s} {'':7s} {'':8s} {trailing['text']:9d} {trailing['audio']:10d} "
              f"{trailing['fr_audio']:9d} {trailing['audio_units']:12d} "
              f"{trailing['stages']:>7s}")
        print("  Read it as: the engine produced this much for the turn that never closed.")
        print("  text>0 and audio=0  -> the thinker answered and the talker never spoke.")
        print("  audio>0, fr_audio=0 -> it spoke, and no output carried the finish_reason")
        print("                          the boundary is detected from.")

    if never:
        print(f"\nwatchdog fired for turn(s): {never}")

    # A turn submitted twice is the client's next query being labelled with a turn index that
    # never advanced, which is a downstream SYMPTOM of the missed boundary rather than a
    # separate bug -- worth naming so it is not chased on its own.
    subs = [t for t, c in
            [(t, sum(1 for l in seg if f"turn={t} queue delta" in l)) for t in sorted(deltas)]
            if c > 1]
    if subs:
        print(f"\nturn(s) submitted more than once: {subs}")
        print("  Downstream of the missed boundary: the turn index only advances when a turn")
        print("  completes, so the client's next query reuses it. Not a separate bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
