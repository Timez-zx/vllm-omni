#!/usr/bin/env python3
"""Is video what drives multi-user latency, and through which link?

    content motion -> EVS retention -> frames in the prompt -> prefill work -> TTFA
                                    \\-> frames NEWLY encoded -> encoder work -> TTFA

Two candidate mechanisms, and they need different measures:

  CONTEXT   every frame in the prompt contributes ~880 tokens that prefill must
            attend over, whether or not it was re-encoded.
            Measure: per-request PROMPT tokens (`num_tokens_in`).

  ENCODING  only frames not already in the multimodal cache get run through the
            vision encoder. Measure: vision-encoder patches / 3520.

These are NOT the same number, which is easy to get wrong. Requesting
`num_frames=4` produced per-turn encoder counts of 1/2/3/4 frames on a clip with
zero duplicate frames: the rolling buffer's consecutive subsamples overlap, so
frames already encoded on an earlier turn hit the cache. Reading the encoder count
as "frames in the prompt" would have understated the context by up to 4x.

Two experiments:

  A  CAUSAL. EVS off, `num_frames` swept 1..16 on ONE stimulus, at 1 and 4 users.
     Content held constant, so anything that moves is caused by frame count.

  B  REALISTIC. EVS on at the shipped 0.95, three stimuli spanning 107x in
     retained frames (measured offline), at 1/2/4 users.

The decisive comparison is between SEGMENTS, not totals. The three segments sum
exactly to TTFA, and only the prefill segment should depend on frames. If the
speech-output segment also grows with frames, the mechanism is not what it looks
like and the attribution is wrong -- so that is reported either way.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import re
import statistics as st

SEGS = [
    ("admit_s", "admission"),
    ("to_first_token_s", "encoders+prefill+1st tok"),
    ("to_first_audio_s", "talker+code2wav"),
]

# Two different per-frame constants, and mixing them up costs a factor of 4.
#
# The stage-0 probe records `ntok` as shape[0] of the first tensor handed to the
# vision encoder's forward -- i.e. PRE-MERGE PATCHES. Qwen's vision stack then does
# a 2x2 spatial merge before the tokens enter the prompt, so:
#
#   1 frame = 3520 encoder patches = ~880 prompt tokens
#
# 3520/4 = 880 against the 895 measured independently from prompt-length steps
# (891 and 903 on two frame-count transitions) -- 1.7% apart, which is the merge.
#
# This also closes a gap TTFA_STUDY.md left open: it reported vision-encoder token
# counts of 35,200-56,320 and declined to convert them to frames because the
# arithmetic "did not quite work out". 35,200/3520 = 10 and 56,320/3520 = 16 exactly
# -- 16 being the num_frames cap. The conversion works; the constant was wrong.
PATCHES_PER_FRAME = 3520.0
PROMPT_TOK_PER_FRAME = 895.0

A_TAG = re.compile(r"^A_u(?P<u>\d+)_f(?P<f>\d+)$")
B_TAG = re.compile(r"^B_u(?P<u>\d+)_(?P<c>static|low|high)$")
CONTENT_ORDER = {"static": 0, "low": 1, "high": 2}
CONTENT_LABEL = {"static": "screencast (static)",
                 "low": "talkinghead (low motion)",
                 "high": "handheld (high motion)"}


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of y on x."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else None


ANSI = re.compile(r"\x1b\[[0-9;]*m")
TOKENS_IN = re.compile(r"num_tokens_in\s*\|([^\n]*)")
LOGTS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")


def run_window(res: pathlib.Path, tag: str) -> tuple[float, float] | None:
    ws: list[float] = []
    for t in (res / f"vl_{tag}").glob("ttfa_user*.jsonl"):
        for line in t.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    ws.append(json.loads(line)["w"])
                except Exception:
                    pass
    return (min(ws), max(ws)) if ws else None


def prompt_tokens(res: pathlib.Path, tag: str, log: pathlib.Path) -> dict:
    """Per-request PROMPT length inside this run's window.

    This is the measure that matters for the *context* mechanism: every frame in
    the prompt contributes tokens that prefill must attend over, whether or not
    the encoder had to re-encode it. The encoder's own token count measures
    something different -- see vision_tokens().

    The log carries all runs of the one shared server, so it is windowed by the
    run's client-trace wall clock (log stamps are second-resolution, hence the
    +-3 s slack).
    """
    w = run_window(res, tag)
    if w is None or not log.exists():
        return {}
    lo, hi = w[0] - 3.0, w[1] + 3.0
    ref = datetime.datetime.fromtimestamp(w[0])
    vals: list[int] = []
    keep = True
    for ln in ANSI.sub("", log.read_text(errors="ignore")).splitlines():
        m = LOGTS.search(ln)
        if m is not None:
            mo, da, hh, mi, ss = (int(x) for x in m.groups())
            try:
                t = datetime.datetime(ref.year, mo, da, hh, mi, ss).timestamp()
                keep = lo <= t <= hi
            except ValueError:
                pass
        if not keep:
            continue
        g = TOKENS_IN.search(ln)
        if g:
            for cell in g.group(1).split("|"):
                c = cell.strip().replace(",", "")
                if c.isdigit() and int(c) > 0:
                    vals.append(int(c))
                    break
    if not vals:
        return {}
    return {"prompt_tok_p50": int(st.median(vals)),
            "prompt_tok_max": max(vals),
            "prompt_n": len(vals)}


def vision_tokens(res: pathlib.Path, tag: str, events: pathlib.Path) -> dict:
    """Frames NEWLY ENCODED per turn during this run's window.

    Not the same as frames in the prompt. The multimodal encoder cache keys on
    uuid = md5(jpeg bytes), and the rolling buffer's consecutive subsamples
    overlap, so a frame already encoded on an earlier turn is not re-encoded.
    Measured: requesting num_frames=4 produced per-turn encoder counts of
    1/2/3/4 frames (x1/x2/x6/x2) with zero duplicate frames in the source clip --
    i.e. the variation is cache hits, not resizing. Every value was an exact
    multiple of 3520, so the per-frame patch cost itself is fixed.
    """
    w = run_window(res, tag)
    if w is None or not events.exists():
        return {}
    lo, hi = w
    vt: list[int] = []
    at: list[int] = []
    for line in events.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        ls = e.get("launch_start")
        if ls is None or not (lo <= ls <= hi):
            continue
        if e.get("module") == "vision_encoder":
            vt.append(int(e.get("ntok") or 0))
        elif e.get("module") == "audio_encoder":
            at.append(int(e.get("ntok") or 0))
    return {
        "vis_calls": len(vt),
        "vis_tok_total": sum(vt),
        "vis_tok_p50": int(st.median(vt)) if vt else 0,
        "new_frames_per_turn": (st.median(vt) / PATCHES_PER_FRAME) if vt else 0.0,
        "prompt_tok_per_call": (st.median(vt) / PATCHES_PER_FRAME * PROMPT_TOK_PER_FRAME) if vt else 0.0,
        "aud_calls": len(at),
        "aud_tok_p50": int(st.median(at)) if at else 0,
    }


def load(res: pathlib.Path, tag: str) -> dict | None:
    p = res / f"decomp_vl_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = [r for r in d.get("rows", []) if r.get("ttfa_end_s") is not None]
    if not rows:
        return None
    rec: dict = {"users": d.get("users"), "n_turns": len(rows)}
    for k, _ in SEGS + [("ttfa_end_s", "TTFA")]:
        v = [r[k] for r in rows if r.get(k) is not None]
        rec[k] = {"p50": pct(v, .5), "p95": pct(v, .95)}
    sp = [r["gpu_busy_speech_frac"] for r in rows
          if r.get("gpu_busy_speech_frac") is not None]
    rec["busy_speech"] = st.median(sp) if sp else None
    return rec


def ms(x) -> str:
    return "-" if x is None else f"{x * 1000:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--events", default="/data/zx/results/stage0_events_vl.jsonl")
    ap.add_argument("--log", default="/data/zx/results/server_vl.log")
    ap.add_argument("--out", default="/data/zx/results/video_latency.json")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    events = pathlib.Path(args.events)
    tags = sorted(pathlib.Path(p).stem.replace("decomp_vl_", "")
                  for p in glob.glob(str(res / "decomp_vl_*.json")))
    if not tags:
        print("no decomp_vl_* found")
        return 1

    arms: dict[str, dict] = {}
    for t in tags:
        d = load(res, t)
        if d is None:
            continue
        d.update(vision_tokens(res, t, events))
        d.update(prompt_tokens(res, t, pathlib.Path(args.log)))
        arms[t] = d

    summary = {"arms": arms}

    # ------------------------------------------------------------- experiment A
    A = {}
    for t, d in arms.items():
        m = A_TAG.match(t)
        if m:
            A[(int(m["u"]), int(m["f"]))] = d
    if A:
        print("\n" + "=" * 78)
        print("A  CAUSAL: frame count swept, content fixed (talkinghead), EVS OFF")
        print("=" * 78)
        print(f"{'users':>6}{'req frames':>11}{'vis tok/call':>13}{'frames seen':>12}"
              f"{'admission':>11}{'prefill':>10}{'speech-out':>12}{'TTFA p50':>10}"
              f"{'TTFA p95':>10}{'busy@speech':>12}")
        for (u, f) in sorted(A):
            d = A[(u, f)]
            print(f"{u:>6}{f:>11}{d.get('vis_tok_p50', 0):>13}"
                  f"{d.get('new_frames_per_turn', 0):>12.1f}"
                  f"{ms(d['admit_s']['p50']):>11}"
                  f"{ms(d['to_first_token_s']['p50']):>10}"
                  f"{ms(d['to_first_audio_s']['p50']):>12}"
                  f"{ms(d['ttfa_end_s']['p50']):>10}"
                  f"{ms(d['ttfa_end_s']['p95']):>10}"
                  f"{(d['busy_speech'] * 100 if d['busy_speech'] is not None else float('nan')):>11.1f}%")
        print("  'frames seen' = vision-encoder tokens per call / 895, i.e. what")
        print("  actually reached the model. With EVS off it should track 'req frames'.")

        print("\n--- per-frame cost, by segment (least-squares slope vs frame count) ---")
        print(f"{'users':>6}{'admission':>14}{'prefill':>14}{'speech-out':>14}{'TTFA':>14}")
        for u in sorted({u for u, _ in A}):
            fs = sorted(f for uu, f in A if uu == u)
            xs = [float(f) for f in fs]
            row = f"{u:>6}"
            slopes = {}
            for k, _ in SEGS + [("ttfa_end_s", "TTFA")]:
                ys = [A[(u, f)][k]["p50"] * 1000 for f in fs]
                s = slope(xs, ys)
                slopes[k] = s
                row += f"{(f'{s:+.1f} ms/frm' if s is not None else '-'):>14}"
            print(row)
            summary.setdefault("A_slopes", {})[str(u)] = {
                k: (round(v, 2) if v is not None else None) for k, v in slopes.items()
            }
        print("  If video is the mechanism, prefill has a clear positive slope and")
        print("  speech-out is ~flat. A positive speech-out slope would mean frames")
        print("  are also slowing the response path, i.e. a different mechanism.")

        print("\n--- how much of TTFA is attributable to frames ---")
        for u in sorted({u for u, _ in A}):
            fs = sorted(f for uu, f in A if uu == u)
            f_lo, f_hi = fs[0], fs[-1]
            lo, hi = A[(u, f_lo)], A[(u, f_hi)]
            d_ttfa = (hi["ttfa_end_s"]["p50"] - lo["ttfa_end_s"]["p50"]) * 1000
            d_pref = (hi["to_first_token_s"]["p50"] - lo["to_first_token_s"]["p50"]) * 1000
            d_spch = (hi["to_first_audio_s"]["p50"] - lo["to_first_audio_s"]["p50"]) * 1000
            base = lo["ttfa_end_s"]["p50"] * 1000
            print(f"  users={u}: {f_lo} -> {f_hi} frames")
            print(f"    TTFA        {base:.0f} -> {hi['ttfa_end_s']['p50']*1000:.0f} ms "
                  f"({d_ttfa:+.0f} ms, {d_ttfa/base*100:+.0f}%)")
            print(f"    of which prefill {d_pref:+.0f} ms "
                  f"({(d_pref/d_ttfa*100 if d_ttfa else 0):.0f}% of the change)")
            print(f"             speech-out {d_spch:+.0f} ms "
                  f"({(d_spch/d_ttfa*100 if d_ttfa else 0):.0f}% of the change)")
            summary.setdefault("A_attribution", {})[str(u)] = {
                "frames": [f_lo, f_hi],
                "ttfa_base_ms": round(base, 1),
                "ttfa_delta_ms": round(d_ttfa, 1),
                "prefill_delta_ms": round(d_pref, 1),
                "speech_delta_ms": round(d_spch, 1),
                "prefill_share_of_change": round(d_pref / d_ttfa, 3) if d_ttfa else None,
            }

    # ------------------------------------------------------------- experiment B
    B = {}
    for t, d in arms.items():
        m = B_TAG.match(t)
        if m:
            B[(int(m["u"]), m["c"])] = d
    if B:
        print("\n" + "=" * 78)
        print("B  REALISTIC: content swept by motion level, EVS ON (0.95)")
        print("=" * 78)
        print(f"{'users':>6}{'content':<26}{'vis tok/call':>13}{'frames seen':>12}"
              f"{'vis calls':>10}{'admission':>11}{'prefill':>10}{'speech-out':>12}"
              f"{'TTFA p50':>10}{'busy@speech':>12}")
        for (u, c) in sorted(B, key=lambda k: (k[0], CONTENT_ORDER.get(k[1], 9))):
            d = B[(u, c)]
            print(f"{u:>6}{CONTENT_LABEL.get(c, c):<26}{d.get('vis_tok_p50', 0):>13}"
                  f"{d.get('new_frames_per_turn', 0):>12.1f}"
                  f"{d.get('vis_calls', 0):>10}"
                  f"{ms(d['admit_s']['p50']):>11}"
                  f"{ms(d['to_first_token_s']['p50']):>10}"
                  f"{ms(d['to_first_audio_s']['p50']):>12}"
                  f"{ms(d['ttfa_end_s']['p50']):>10}"
                  f"{(d['busy_speech'] * 100 if d['busy_speech'] is not None else float('nan')):>11.1f}%")

        print("\n--- content effect at fixed user count (static -> high motion) ---")
        for u in sorted({u for u, _ in B}):
            present = [c for c in ("static", "low", "high") if (u, c) in B]
            if len(present) < 2:
                continue
            lo, hi = B[(u, present[0])], B[(u, present[-1])]
            d_ttfa = (hi["ttfa_end_s"]["p50"] - lo["ttfa_end_s"]["p50"]) * 1000
            base = lo["ttfa_end_s"]["p50"] * 1000
            fl = lo.get("new_frames_per_turn", 0)
            fh = hi.get("new_frames_per_turn", 0)
            print(f"  users={u}: {CONTENT_LABEL[present[0]]} -> {CONTENT_LABEL[present[-1]]}")
            print(f"    frames the model saw   {fl:.1f} -> {fh:.1f}")
            print(f"    TTFA                   {base:.0f} -> {hi['ttfa_end_s']['p50']*1000:.0f} ms "
                  f"({d_ttfa:+.0f} ms, {d_ttfa/base*100:+.0f}%)")
            print(f"    prefill                {ms(lo['to_first_token_s']['p50'])} -> "
                  f"{ms(hi['to_first_token_s']['p50'])} ms")
            print(f"    speech-out             {ms(lo['to_first_audio_s']['p50'])} -> "
                  f"{ms(hi['to_first_audio_s']['p50'])} ms")

        print("\n--- user-count effect at fixed content ---")
        for c in ("static", "low", "high"):
            us = sorted(u for u, cc in B if cc == c)
            if len(us) < 2:
                continue
            row = f"  {CONTENT_LABEL[c]:<26}"
            for u in us:
                row += f" u{u}={ms(B[(u,c)]['ttfa_end_s']['p50'])}ms"
            lo, hi = B[(us[0], c)], B[(us[-1], c)]
            r = hi["ttfa_end_s"]["p50"] / lo["ttfa_end_s"]["p50"]
            row += f"   ({r:.2f}x from u{us[0]} to u{us[-1]})"
            print(row)

    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
