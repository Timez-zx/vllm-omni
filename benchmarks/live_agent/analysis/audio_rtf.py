#!/usr/bin/env python3
"""Recover the REAL generated audio duration and real-time factor.

The client cannot measure these: vllm-omni's streaming path delivers only the
first audio delta (~97% of generated audio is discarded before it reaches the
wire), so any client-side duration or RTF is meaningless.

But the engine's own periodic stats table carries cumulative counters:

    audio_duration_s         seconds of waveform produced so far
    audio_generated_frames   sample count  (= audio_duration_s * sample_rate)
    audio_sample_rate        24000 for Qwen3-Omni code2wav
    stage_gen_time_ms        cumulative generation time, per stage

Differencing consecutive snapshots gives per-interval audio produced and per-
interval stage generation time, from which two different real-time factors can
be formed -- and the distinction matters:

    wall RTF  = wall seconds elapsed / audio seconds produced
                <1 means the pipeline outruns playback. This is what decides
                whether the CURRENT user hears gaps.

    gen  RTF  = stage generation seconds / audio seconds produced
                how much engine time the audio actually costs. This is what
                decides how much it steals from OTHER users.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics as st

ROW = re.compile(r"\|\s*(?P<name>[a-z_0-9]+)\s*\|(?P<vals>[^\n]*)\|")
TIMING = re.compile(r"\[TIMING\].*total=(?P<total>[\d.]+)s")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def nums(s: str) -> list[float]:
    out = []
    for cell in s.split("|"):
        c = cell.strip().replace(",", "")
        try:
            out.append(float(c))
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    text = ANSI.sub("", pathlib.Path(args.log).read_text(errors="ignore"))
    snaps: list[dict] = []
    cur: dict = {}
    turn_totals: list[float] = []
    for line in text.splitlines():
        m = TIMING.search(line)
        if m:
            turn_totals.append(float(m["total"]))
        if "stats.py" not in line:
            continue
        r = ROW.search(line)
        if not r:
            continue
        name, vals = r["name"], nums(r["vals"])
        if not vals:
            continue
        if name in cur:            # a new snapshot began
            snaps.append(cur)
            cur = {}
        cur[name] = vals
    if cur:
        snaps.append(cur)

    keep = [s for s in snaps if "audio_duration_s" in s]
    if len(keep) < 2:
        print(f"need >=2 stats snapshots with audio_duration_s; got {len(keep)}")
        return 1

    sr = next((s["audio_sample_rate"][-1] for s in keep
               if s.get("audio_sample_rate") and s["audio_sample_rate"][-1] > 0), 24000.0)
    print(f"\n=== recovered audio production (sample_rate={sr:.0f} Hz) ===")
    print(f"{'interval':>9}{'audio s':>10}{'stage1 gen s':>14}{'stage2 gen s':>14}"
          f"{'gen RTF (s1+s2)':>17}")
    aud_d, gen_d = [], []
    for a, b in zip(keep, keep[1:]):
        da = b["audio_duration_s"][-1] - a["audio_duration_s"][-1]
        if da <= 0:
            continue
        g1 = g2 = 0.0
        if "stage_gen_time_ms" in a and "stage_gen_time_ms" in b:
            va, vb = a["stage_gen_time_ms"], b["stage_gen_time_ms"]
            if len(va) >= 3 and len(vb) >= 3:
                g1 = (vb[1] - va[1]) / 1000.0
                g2 = (vb[2] - va[2]) / 1000.0
        aud_d.append(da)
        gen_d.append(g1 + g2)
        print(f"{len(aud_d):>9}{da:>10.2f}{g1:>14.2f}{g2:>14.2f}"
              f"{((g1+g2)/da if da else float('nan')):>17.3f}")

    if not aud_d:
        print("no positive audio intervals")
        return 1

    print(f"\n  audio produced per interval (p50):   {st.median(aud_d):.2f} s")
    print(f"  stage1+2 generation time    (p50):   {st.median(gen_d):.2f} s")
    print(f"  --> gen RTF (p50):                   {st.median(gen_d)/st.median(aud_d):.3f}")
    if turn_totals:
        tt = st.median(turn_totals[1:] or turn_totals)
        print(f"\n  turn wall time (p50, server TIMING): {tt:.2f} s")
        print(f"  --> wall RTF vs audio produced:      {tt/st.median(aud_d):.3f}")
    print("\n  gen RTF << 1  -> the audio costs far less engine time than it plays for;")
    print("                   the surplus is what a pacing scheduler could give to others.")
    print("  gen RTF ~ 1    -> no surplus; pacing the response path buys nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
