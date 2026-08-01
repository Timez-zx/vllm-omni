#!/usr/bin/env python3
"""Where does the 8-user latency go, and why?

The tail sweep answered "how bad" (p50/p95/p99 by content). This answers "where".
Five independent measurement sources are combined, and they are kept separate so a
disagreement between them shows up instead of being averaged away:

  1. CLIENT TRACES  ttfa_user*.jsonl -- wall-clock timestamps around each turn.
     Gives three segments that sum EXACTLY to TTFA:
        admission        t_query      -> t_rx_start    (queued, not yet answered)
        to_first_token   t_rx_start   -> t_first_text  (encode + prefill + 1 decode)
        to_first_audio   t_first_text -> t_first_audio (talker + code2wav ramp)
     This is the ground truth for latency, because it is what the user feels.

  2. CUDA-EVENT PROBE  stage0_events_vt.jsonl -- exact device time of the vision and
     audio encoders (they never enter a CUDA graph, so forward hooks fire). Lets the
     encoder be SUBTRACTED from segment 2, isolating attention-over-context.

  3. NVML 50 Hz  gpu_vt_*.jsonl -- device busy fraction and per-stage share.
     Absolute totals come from the dense device counter; per-stage split comes from
     summed per-PID ratios only. Mixing those up inflates duty by ~10x (see
     MULTIUSER_MEMORY.md 2.1).

  4. SERVER LOG  [TIMING] first_text / first_audio -- an independent clock on the
     same two boundaries as segment 2 and 3. Used purely as a cross-check.

  5. SERVER LOG  num_tokens_in -- prompt length per request, i.e. how much context
     prefill had to attend over.

**The question that decides the story.** A request can be slow for two different
reasons that look identical from outside:

     WORKING   the device is running this request's own kernels
     WAITING   the device is busy, but with somebody else's request

Both leave the GPU near 100% utilised, so utilisation alone cannot separate them.
What separates them is comparing the SAME content at different user counts: if a
segment scales ~linearly with user count, the added time is queueing behind other
users; if it stays flat, the batcher absorbed them.

That comparison is turn-index matched. It has to be: TTFA degrades over a session
(the rolling frame buffer fills), and the 1/2/4-user baselines are 11-turn runs
while the tail arms are 60-turn runs. Comparing all 60 turns against 10 would
attribute a session effect to concurrency. Only turns 1-10 are used on both sides.

(An earlier version of this header said "turns 2-10". That was a mislabel, not a
different filter: `rows_of` applies `rep <= MATCH_TURNS` and `rep` already starts at
1 because rep 0 was dropped upstream by `--skip-reps 1`. The distinction is not
cosmetic on the high-motion arm -- excluding rep 1 moves 1-user own-work 873 -> 891 ms
and 8-user own-work 3,253 -> 3,443 ms, i.e. effective concurrency 2.06 -> 1.99.)
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGTS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")
TOKENS_IN = re.compile(r"num_tokens_in\s*\|([^\n]*)")
TIMING = re.compile(r"\[TIMING\].*?first_text=([\d.]+)s first_audio=([\d.]+)s"
                    r" audio_chunks=(\d+)")

CONTENTS = ["static", "low", "high"]
LABEL = {"static": "screencast (static screen)",
         "low": "talkinghead (low motion)",
         "high": "handheld (high motion)"}
# 1 frame = 3520 encoder patches (pre-merge) = ~880 prompt tokens (after 2x2 merge)
PATCHES_PER_FRAME = 3520.0

MATCH_TURNS = 10   # turn-index window shared by the 11-turn and 60-turn runs


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def ms(x, w=7):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.0f}"


# ---------------------------------------------------------------- client traces

def rows_of(path: pathlib.Path, max_rep: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    rs = [r for r in d.get("rows", []) if r.get("ttfa_end_s") is not None]
    if max_rep is not None:
        rs = [r for r in rs if r.get("rep", 10 ** 9) <= max_rep]
    return rs


def concurrency(rows: list[dict]) -> dict:
    """Two different kinds of competing work, measured separately.

    A turn can be delayed by two things that are easy to conflate:

      PREFILL  another user is also waiting for its first sound, so its prefill
               competes with mine for the same stage-0 batch.
               Window: their t_query -> t_first_audio.

      SPEECH   another user already got its first sound and is still streaming --
               their talker and code2wav keep running for the whole response, tens
               of seconds, long after their TTFA ended.
               Window: their t_first_audio -> t_end.

    These have to be counted apart, because they predict opposite things. Prefill
    competition scales with how many users are *waiting*; speech competition scales
    with how LONG each response is, and a short-TTFA workload can still have every
    other user streaming audio the entire time. Counting only the first makes light
    content look like it is being delayed by something that is not there.
    """
    pre = [(r["t_query"], r["t_first_audio"], r.get("uid")) for r in rows
           if r.get("t_first_audio")]
    spk = [(r["t_first_audio"], r["t_end"], r.get("uid")) for r in rows
           if r.get("t_first_audio") and r.get("t_end")]
    n_pre, n_spk = [], []
    for a0, a1, ua in pre:
        n_pre.append(float(sum(1 for b0, b1, ub in pre
                               if ub != ua and b0 < a1 and b1 > a0)))
        n_spk.append(float(sum(1 for b0, b1, ub in spk
                               if ub != ua and b0 < a1 and b1 > a0)))
    return {"prefill": n_pre, "speech": n_spk}


# ---------------------------------------------------------------- probe / NVML

def window(res: pathlib.Path, tag: str, prefix: str) -> tuple[float, float] | None:
    ws = []
    for t in (res / f"{prefix}_{tag}").glob("ttfa_user*.jsonl"):
        for line in t.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    ws.append(json.loads(line)["w"])
                except Exception:
                    pass
    return (min(ws), max(ws)) if ws else None


def encoders(events: pathlib.Path, w: tuple[float, float], n_turns: int) -> dict:
    """Exact device time spent in the vision and audio encoders in this window.

    Per-turn attribution is deliberately NOT attempted at 8 users: several users'
    encoder calls interleave inside one turn's window, so assigning a call to a turn
    would double-count. The arm total divided by the turn count is honest, and it is
    all that is needed to ask "is prefill dominated by encoding or by attention".
    """
    if not events.exists() or not w:
        return {}
    lo, hi = w
    vis_ms, aud_ms, vis_tok, calls_v, calls_a = 0.0, 0.0, 0, 0, 0
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
        if e["module"] == "vision_encoder":
            vis_ms += e.get("gpu_ms", 0.0); vis_tok += e.get("ntok", 0); calls_v += 1
        elif e["module"] == "audio_encoder":
            aud_ms += e.get("gpu_ms", 0.0); calls_a += 1
    return {"vision_gpu_ms": vis_ms, "audio_gpu_ms": aud_ms,
            "vision_calls": calls_v, "audio_calls": calls_a,
            "frames_encoded": vis_tok / PATCHES_PER_FRAME,
            "vision_ms_per_turn": vis_ms / n_turns if n_turns else None,
            "audio_ms_per_turn": aud_ms / n_turns if n_turns else None,
            "frames_per_turn": (vis_tok / PATCHES_PER_FRAME / n_turns)
            if n_turns else None}


def nvml(path: pathlib.Path, w: tuple[float, float] | None) -> dict:
    """Device duty and per-stage share inside the window.

    Absolute busy time comes from the dense device `sm` counter, which is present in
    ~100% of samples and can legitimately be integrated. The per-stage split comes
    from summed per-PID `proc_sm`, used ONLY as a ratio -- it appears in <10% of
    samples and each value is a MAX over NVML's own ~200 ms window, so integrating
    it directly manufactures busy time out of nothing.
    """
    if not path.exists():
        return {}
    stage_of: dict[str, str] = {}
    t0 = None
    dev, n_dev = [], 0
    per_stage: dict[str, float] = {}
    clocks, temps, throttled, n_thr = [], [], 0, 0
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            k = e.get("k")
            if k == "meta":
                t0 = e.get("wall_start")
                continue
            if k == "stage_map":
                for pid, stg in e["map"].items():
                    stage_of[str(pid)] = stg
                continue
            if k != "s" or t0 is None:
                continue
            t = t0 + e["t"]
            if w and not (w[0] <= t <= w[1]):
                continue
            sm = e.get("sm")
            if sm is not None:
                dev.append(sm); n_dev += 1
            if e.get("sm_clock"):
                clocks.append(e["sm_clock"])
            if e.get("temp_c"):
                temps.append(e["temp_c"])
            if e.get("throttle") is not None:
                n_thr += 1; throttled += 1 if e["throttle"] else 0
            ps = e.get("proc_sm") or {}
            for pid, v in ps.items():
                stg = stage_of.get(str(pid))
                if stg and v:
                    per_stage[stg] = per_stage.get(stg, 0.0) + float(v)
    tot = sum(per_stage.values())
    return {"device_duty": (st.fmean([1.0 if x else 0.0 for x in dev]) if dev else None),
            "device_sm_mean": (st.fmean(dev) if dev else None),
            "n_samples": n_dev,
            "stage_share": {k: v / tot for k, v in sorted(per_stage.items())} if tot else {},
            "sm_clock_p50": (st.median(clocks) if clocks else None),
            "temp_max": (max(temps) if temps else None),
            "throttled_frac": (throttled / n_thr if n_thr else None)}


def log_window(log: pathlib.Path, w: tuple[float, float] | None) -> dict:
    """Prompt length and the server's own first_text / first_audio, windowed.

    One server served every arm, so the log has to be cut by the arm's client
    wall clock. Log stamps are second-resolution, hence the +-3 s slack.
    """
    if not log.exists() or not w:
        return {}
    lo, hi = w[0] - 3.0, w[1] + 3.0
    ref = datetime.datetime.fromtimestamp(w[0])
    toks, ftext, faud, chunks = [], [], [], []
    keep = False
    for ln in ANSI.sub("", log.read_text(errors="ignore")).splitlines():
        m = LOGTS.search(ln)
        if m:
            mo, da, hh, mi, ss = (int(x) for x in m.groups())
            try:
                t = datetime.datetime(ref.year, mo, da, hh, mi, ss).timestamp()
            except ValueError:
                t = None
            if t is not None:
                keep = lo <= t <= hi
        if not keep:
            continue
        mt = TOKENS_IN.search(ln)
        if mt:
            cell = mt.group(1).split("|")[0].strip().replace(",", "")
            if cell.isdigit() and int(cell):
                toks.append(int(cell))
        mg = TIMING.search(ln)
        if mg:
            ftext.append(float(mg.group(1)) * 1000)
            faud.append(float(mg.group(2)) * 1000)
            chunks.append(int(mg.group(3)))
    # The speech-ramp cross-check must be the MEDIAN OF THE PER-TURN DIFFERENCES,
    # not the difference of the two medians. Those are not the same statistic, and
    # using the latter reported a spurious 31% client/server disagreement on the
    # high-motion arm -- the two clocks actually agree to 4%.
    gaps = [b - a for a, b in zip(ftext, faud)]
    return {"prompt_tok_p50": pct(toks, .5), "prompt_tok_p95": pct(toks, .95),
            "prompt_n": len(toks),
            "srv_first_text_p50": pct(ftext, .5),
            "srv_ramp_p50": pct(gaps, .5),
            "srv_n": len(ftext),
            "audio_chunks_p50": pct([float(c) for c in chunks], .5)}


# ---------------------------------------------------------------- report

def arm(res: pathlib.Path, log: pathlib.Path, events: pathlib.Path,
        users: int, content: str) -> dict | None:
    dec = res / f"decomp_vt_u{users}_{content}.json"
    rows = rows_of(dec)
    if not rows:
        return None
    w = window(res, f"u{users}_{content}", "vt")
    g = nvml(res / f"gpu_vt_u{users}_{content}.jsonl", w)
    out = {"users": users, "content": content, "n": len(rows),
           "rows_first10": rows_of(dec, MATCH_TURNS),
           "gpu": g, "log": log_window(log, w),
           "enc": encoders(events, w, len(rows)),
           "conc": concurrency(rows),
           "busy_ttfa": [r["gpu_busy_ttfa_frac"] for r in rows
                         if r.get("gpu_busy_ttfa_frac") is not None]}
    for key in ("admit_s", "to_first_token_s", "to_first_audio_s", "ttfa_end_s"):
        v = [r[key] * 1000 for r in rows if r.get(key) is not None]
        out[key] = {"p50": pct(v, .5), "p95": pct(v, .95), "mean": st.fmean(v)}
    return out


def baseline(res: pathlib.Path, users: int, content: str) -> dict | None:
    """Turn-matched 1/2/4-user reference from the earlier 11-turn experiment B."""
    rows = rows_of(res / f"decomp_vl_B_u{users}_{content}.json", MATCH_TURNS)
    if not rows:
        return None
    o = {"n": len(rows)}
    for key in ("admit_s", "to_first_token_s", "to_first_audio_s", "ttfa_end_s"):
        v = [r[key] * 1000 for r in rows if r.get(key) is not None]
        o[key] = pct(v, .5)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--out", default="/data/zx/results/video_tail_decompose8.json")
    a = ap.parse_args()
    res = pathlib.Path(a.results)
    log = res / "server_vt.log"
    events = res / "stage0_events_vt.jsonl"
    U = a.users

    arms = {c: arm(res, log, events, U, c) for c in CONTENTS}
    arms = {k: v for k, v in arms.items() if v}
    if not arms:
        print("no arms found")
        return 1

    W = 96
    print("\n" + "=" * W)
    print(f"WHERE THE LATENCY GOES AT {U} CONCURRENT USERS  (shipped config: EVS 0.95, 16 frames)")
    print("=" * W)

    # ---- 1. the three segments, absolute and as a share of TTFA
    print("\n--- 1. TTFA split into its three segments (client clock, sums exactly) ---")
    print(f"{'content':<28}{'n':>5}{'admission':>12}{'encode+prefill':>16}"
          f"{'speech ramp':>14}{'TTFA':>10}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        print(f"{LABEL[c]:<28}{d['n']:>5}"
              f"{ms(d['admit_s']['p50'], 12)}{ms(d['to_first_token_s']['p50'], 16)}"
              f"{ms(d['to_first_audio_s']['p50'], 14)}{ms(d['ttfa_end_s']['p50'], 10)}")
    # Shares must be MEDIANS OF PER-TURN SHARES, not p50(part)/p50(whole). The three
    # segments sum exactly to TTFA on every single turn, but their p50s do not sum to
    # the p50 of TTFA unless the segments are co-monotone. On the high-motion arm they
    # are not: the part-p50s add to 7,140 ms against a TTFA p50 of 8,297 ms, so 14% of
    # the total goes unaccounted for and BOTH dominant segments read too low
    # (encode+prefill 54.3% instead of 58.6%, speech ramp 31.6% instead of 39.7%).
    # Static and low agree to 1-3 points either way, which is how the error hid.
    print(f"\n{'  ... as % of TTFA (median of per-turn shares)':<44}{'admission':>12}"
          f"{'encode+prefill':>16}{'speech ramp':>14}{'resid':>8}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        keys = ("admit_s", "to_first_token_s", "to_first_audio_s")
        rows = rows_of(res / f"decomp_vt_u{U}_{c}.json")
        sh: dict = {k: [] for k in keys}
        for r in rows:
            t = r.get("ttfa_end_s")
            if not t:
                continue
            for k in keys:
                if r.get(k) is not None:
                    sh[k].append(r[k] / t)
        v = {k: pct(sh[k], .5) for k in keys}
        print(f"{LABEL[c]:<44}"
              f"{v['admit_s']:>11.1%}{v['to_first_token_s']:>16.1%}"
              f"{v['to_first_audio_s']:>14.1%}"
              f"{1 - sum(v.values()):>7.1%}")
    print("    (per-turn shares; the naive p50(part)/p50(whole) understates the two")
    print("     dominant segments on the high-motion arm by 4-8 points each)")
    print(f"\n{'  ... same split at p95':<28}{'':>5}{'admission':>12}"
          f"{'encode+prefill':>16}{'speech ramp':>14}{'TTFA':>10}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        print(f"{LABEL[c]:<28}{'':>5}"
              f"{ms(d['admit_s']['p95'], 12)}{ms(d['to_first_token_s']['p95'], 16)}"
              f"{ms(d['to_first_audio_s']['p95'], 14)}{ms(d['ttfa_end_s']['p95'], 10)}")

    # ---- 2. inside segment 2: encoder vs attention-over-context
    print("\n--- 2. inside 'encode+prefill': is it the encoder, or attention over context? ---")
    print(f"{'content':<28}{'prompt tok':>12}{'frames/turn':>12}"
          f"{'vision ms':>11}{'audio ms':>10}{'residual':>11}{'enc share':>11}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        e, lg = d["enc"], d["log"]
        seg = d["to_first_token_s"]["p50"]
        v = e.get("vision_ms_per_turn") or 0.0
        au = e.get("audio_ms_per_turn") or 0.0
        print(f"{LABEL[c]:<28}{ms(lg.get('prompt_tok_p50'), 12)}"
              f"{(e.get('frames_per_turn') or 0):>12.2f}"
              f"{v:>11.1f}{au:>10.1f}{seg - v - au:>11.0f}{(v + au)/seg:>10.1%}")
    print("  vision/audio ms = exact device time (CUDA events), arm total / turns.")
    print("  residual = segment 2 minus both encoders: attention over the prompt,")
    print("  the first decode step, and any time queued behind other users.")

    # ---- 3. is the device idle-waiting or busy-with-others?
    print("\n--- 3. was the device idle during these slow turns? ---")
    print(f"{'content':<28}{'busy during TTFA':>18}{'duty whole arm':>16}"
          f"{'mean SM%':>10}{'clock MHz':>11}{'throttled':>11}{'temp':>7}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        g = d["gpu"]
        bt = st.fmean(d["busy_ttfa"]) if d["busy_ttfa"] else None
        duty, thr, msm = g.get("device_duty"), g.get("throttled_frac"), g.get("device_sm_mean")
        s_bt = f"{bt:.1%}" if bt is not None else "-"
        s_duty = f"{duty:.1%}" if duty is not None else "-"
        s_thr = f"{thr:.1%}" if thr is not None else "-"
        s_msm = f"{msm:.1f}%" if msm is not None else "-"
        print(f"{LABEL[c]:<28}{s_bt:>18}{s_duty:>16}{s_msm:>10}"
              f"{ms(g.get('sm_clock_p50'), 11)}{s_thr:>11}{ms(g.get('temp_max'), 7)}")
    print("  duty = fraction of 50 Hz samples with any kernel resident;")
    print("  mean SM% = average utilisation over the same samples. Both come from the")
    print("  dense device counter, so both may be integrated.")
    print("  'busy during TTFA' covers only the wait the user actually experiences;")
    print("  'whole arm' also covers speech streaming, when stage 0 idles -- so the")
    print("  two differ by design, and the LEFT column decides starved vs queued.")

    # ---- 4. which stage burned the device
    print("\n--- 4. which of the three stages burned the device time ---")
    stages = sorted({s for d in arms.values() for s in d["gpu"].get("stage_share", {})})
    print(f"{'content':<28}" + "".join(f"{s:>14}" for s in stages))
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        sh = d["gpu"].get("stage_share", {})
        print(f"{LABEL[c]:<28}"
              + "".join(f"{(f'{sh[s]:.1%}' if s in sh else '-'):>14}" for s in stages))
    print("  stage0 = see/hear/think, stage1 = talker, stage2 = waveform.")
    print("  Ratio only (summed per-PID NVML); absolute totals are not valid here.")

    # ---- 5. server-side cross-check on the same two boundaries
    print("\n--- 5. cross-check: server's own clock vs the client's ---")
    print(f"{'content':<28}{'srv first_text':>15}{'cli seg2':>11}{'diff':>8}"
          f"{'srv f_audio-f_text':>20}{'cli seg3':>11}{'diff':>8}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        lg = d["log"]
        s2, s3 = d["to_first_token_s"]["p50"], d["to_first_audio_s"]["p50"]
        ft, gap = lg.get("srv_first_text_p50"), lg.get("srv_ramp_p50")
        print(f"{LABEL[c]:<28}{ms(ft, 15)}{ms(s2, 11)}"
              f"{(f'{(s2-ft)/ft:+.0%}' if ft else '-'):>8}"
              f"{ms(gap, 20)}{ms(s3, 11)}"
              f"{(f'{(s3-gap)/gap:+.0%}' if gap else '-'):>8}")
    print("  Two clocks on the same two boundaries. Agreement here is what licenses")
    print("  reading the client segments as engine behaviour rather than transport.")

    # ---- 6. the working-vs-waiting test: scaling, turn-index matched
    print(f"\n--- 6. WORKING or WAITING? same content, 1 -> {U} users, turns 1-{MATCH_TURNS} only ---")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        print(f"\n  {LABEL[c]}")
        print(f"{'    users':<12}{'n':>5}{'admission':>12}{'encode+prefill':>16}"
              f"{'speech ramp':>14}{'TTFA':>10}")
        ref = None
        for u in (1, 2, 4):
            b = baseline(res, u, c)
            if not b:
                continue
            if ref is None:
                ref = b
            print(f"{('    ' + str(u)):<12}{b['n']:>5}{ms(b['admit_s'], 12)}"
                  f"{ms(b['to_first_token_s'], 16)}{ms(b['to_first_audio_s'], 14)}"
                  f"{ms(b['ttfa_end_s'], 10)}")
        r10 = d["rows_first10"]
        cur = {k: pct([x[k] * 1000 for x in r10 if x.get(k) is not None], .5)
               for k in ("admit_s", "to_first_token_s", "to_first_audio_s", "ttfa_end_s")}
        print(f"{('    ' + str(U)):<12}{len(r10):>5}{ms(cur['admit_s'], 12)}"
              f"{ms(cur['to_first_token_s'], 16)}{ms(cur['to_first_audio_s'], 14)}"
              f"{ms(cur['ttfa_end_s'], 10)}")
        if ref:
            print(f"{('    x vs 1u'):<12}{'':>5}"
                  + "".join(
                      f"{(f'{cur[k]/ref[k]:.2f}x' if ref[k] else '-'):>{w}}"
                      for k, w in (("admit_s", 12), ("to_first_token_s", 16),
                                   ("to_first_audio_s", 14), ("ttfa_end_s", 10))))
            print(f"{('    ideal batch'):<12}{'':>5}{'1.00x':>12}{'1.00x':>16}"
                  f"{'1.00x':>14}{'1.00x':>10}")
            print(f"{('    full serial'):<12}{'':>5}{(str(U)+'.00x'):>12}"
                  f"{(str(U)+'.00x'):>16}{(str(U)+'.00x'):>14}{(str(U)+'.00x'):>10}")

    # ---- 7. measured concurrency: did 8 users actually overlap?
    print(f"\n--- 7. what was the device doing FOR OTHER USERS during my wait? ---")
    print(f"{'content':<24}{'others prefilling':>19}{'others speaking':>18}"
          f"{'total competitors':>19}   of {U-1}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d or not d["conc"]["prefill"]:
            continue
        p, s = d["conc"]["prefill"], d["conc"]["speech"]
        print(f"{LABEL[c][:22]:<24}{st.fmean(p):>19.2f}{st.fmean(s):>18.2f}"
              f"{st.fmean(p) + st.fmean(s):>19.2f}")
    print("  Mean over turns, counted from wall clock. 'Prefilling' = another user is")
    print("  also waiting for its first sound. 'Speaking' = another user already got")
    print("  its first sound and its talker/code2wav are still running. The second")
    print("  can exceed the first by a lot, because a response outlives its own TTFA.")
    print("  Note the two windows are disjoint per user, so the total can reach U-1.")

    # ---- 8. the arithmetic: does queueing account for the whole gap?
    print(f"\n--- 8. does queueing ALONE account for the {U}-user prefill? ---")
    print("  Hypothesis: a turn's prefill time = its own work + the work of every")
    print("  other request in flight ahead of it. If so, then")
    print("      prefill(U users) ~= (1 + others in flight) x prefill(1 user)")
    print("  Both sides measured; the 1-user side has its encoders removed too, so")
    print("  only attention-over-context is being scaled.")
    print(f"\n{'content':<24}{'own work':>10}{'in flight':>11}{'predicted':>11}"
          f"{'measured':>10}{'pred/meas':>11}{'verdict':>14}")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        b1 = baseline(res, 1, c)
        if not b1:
            continue
        w1 = window(res, f"B_u1_{c}", "vl")
        e1 = encoders(res / "stage0_events_vl.jsonl", w1, 10) if w1 else {}
        own = b1["to_first_token_s"] - (e1.get("vision_ms_per_turn") or 0.0) \
            - (e1.get("audio_ms_per_turn") or 0.0)
        r10 = d["rows_first10"]
        seg2 = pct([x["to_first_token_s"] * 1000 for x in r10], .5)
        enc = (d["enc"].get("vision_ms_per_turn") or 0.0) \
            + (d["enc"].get("audio_ms_per_turn") or 0.0)
        meas = seg2 - enc
        inflight = 1.0 + st.fmean(d["conc"]["prefill"])
        pred = own * inflight
        ratio = pred / meas if meas else None
        verdict = ("queueing" if ratio and 0.7 <= ratio <= 1.4
                   else "batched" if ratio and ratio > 1.4 else "OTHER CAUSE")
        print(f"{LABEL[c][:22]:<24}{own:>10.0f}{inflight:>11.2f}{pred:>11.0f}"
              f"{meas:>10.0f}{(f'{ratio:.2f}' if ratio else '-'):>11}{verdict:>14}")
    print("  'queueing'    prediction within +-40% of measurement: waiting behind")
    print("                other prefills explains the slowdown and nothing else.")
    print("  'batched'     the engine beat serial execution -- prediction too high.")
    print("  'OTHER CAUSE' measured is WORSE than serial prefill, so other prefills")
    print("                cannot be the cause. See row 7: for light content the")
    print("                competing work is other users' ongoing SPEECH, which this")
    print("                prediction does not include.")
    print("  Effective concurrency the engine actually achieved (in flight / slowdown):")
    for c in CONTENTS:
        d = arms.get(c)
        if not d:
            continue
        b1 = baseline(res, 1, c)
        if not b1:
            continue
        w1 = window(res, f"B_u1_{c}", "vl")
        e1 = encoders(res / "stage0_events_vl.jsonl", w1, 10) if w1 else {}
        own = b1["to_first_token_s"] - (e1.get("vision_ms_per_turn") or 0.0) \
            - (e1.get("audio_ms_per_turn") or 0.0)
        r10 = d["rows_first10"]
        seg2 = pct([x["to_first_token_s"] * 1000 for x in r10], .5)
        enc = (d["enc"].get("vision_ms_per_turn") or 0.0) \
            + (d["enc"].get("audio_ms_per_turn") or 0.0)
        meas = seg2 - enc
        inflight = 1.0 + st.fmean(d["conc"]["prefill"])
        slow = meas / own if own else None
        if slow:
            print(f"    {LABEL[c][:22]:<24} {inflight:.2f} in flight / {slow:.2f}x "
                  f"slower = {inflight/slow:.2f} served at once")

    pathlib.Path(a.out).write_text(json.dumps(
        {c: {k: v for k, v in d.items() if k not in ("rows_first10", "conc", "busy_ttfa")}
         for c, d in arms.items()}, indent=2, default=str))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
