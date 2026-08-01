#!/usr/bin/env python3
"""Decompose TTFA per repetition, aligning three independent data sources.

Sources, all on absolute wall clock so they can be overlaid:
  ttfa_user*.jsonl      client trace: speech onset, query, first text, first audio
  stage0_events.jsonl   in-server per-call GPU intervals for the thinker's
                        vision encoder / audio encoder (LLM is unreachable by
                        nn.Module hooks -- see stage0_probe docstring)
  gpu_*.jsonl           NVML device + per-PID sampling (covers ALL stages, so it
                        catches talker/code2wav work that the probe cannot see)

Segments reported per repetition:

  speech            query        - speech_start     user is talking
  admit             rx_start     - query            queueing / admission
  to_first_token    first_text   - rx_start         encoders + vision-token
                                                    prefill + first decode step
  to_first_audio    first_audio  - first_text       talker spin-up + code2wav

  TTFA_end          first_audio  - query            the SLO metric
  TTFA_start        first_audio  - speech_start     the diagnostic timeline

The question the diagnostic anchor exists to answer: how much GPU work happened
*while the user was still speaking*? If that is ~0, every millisecond of encode
and prefill is being paid after the user stops, and lands directly on TTFA.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st


def load(p: str) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def reps_from_trace(ev: list[dict]) -> tuple[dict, list[dict]]:
    meta = next((e for e in ev if e.get("k") == "meta"), {})
    reps: dict[int, dict] = {}

    def R(i):
        return reps.setdefault(i, {"rep": i, "audio_tx": 0, "speech_tx": 0,
                                   "slips": 0})

    for e in ev:
        k, w, r = e.get("k"), e.get("w"), e.get("rep")
        if k == "rep_begin":
            R(r)["t_begin"] = w
        elif k == "speech_start":
            R(r)["t_speech_start"] = w
        elif k == "speech_end":
            R(r)["t_speech_end"] = w
        elif k == "tx_query":
            R(r)["t_query"] = w
            R(r)["frames_sent"] = e.get("frames_sent")
        elif k == "rx_start":
            R(r).setdefault("t_rx_start", w)
        elif k == "rx_first_text":
            R(r).setdefault("t_first_text", w)
        elif k == "rx_first_audio":
            R(r).setdefault("t_first_audio", w)
        elif k == "rx_text_done":
            R(r)["chars"] = e.get("chars")
        elif k == "rep_end":
            R(r)["t_end"] = w
            R(r)["n_audio_deltas"] = e.get("n_audio_deltas")
        elif k == "tx_audio":
            if r is not None and r >= 0:
                R(r)["audio_tx"] += 1
                if e.get("speech"):
                    R(r)["speech_tx"] += 1
        elif k in ("frame_slip", "audio_slip"):
            if r is not None and r >= 0:
                R(r)["slips"] += 1
        elif k == "rep_timeout":
            R(r)["timeout"] = True

    rows = []
    for i in sorted(reps):
        d = reps[i]
        q, ss = d.get("t_query"), d.get("t_speech_start")
        fa, ft, rs = d.get("t_first_audio"), d.get("t_first_text"), d.get("t_rx_start")
        d["speech_s"] = (q - ss) if (q and ss) else None
        d["admit_s"] = (rs - q) if (rs and q) else None
        d["to_first_token_s"] = (ft - rs) if (ft and rs) else None
        d["to_first_audio_s"] = (fa - ft) if (fa and ft) else None
        d["ttfa_end_s"] = (fa - q) if (fa and q) else None
        d["ttfa_start_s"] = (fa - ss) if (fa and ss) else None
        rows.append(d)
    return meta, rows


def gpu_work_in(events: list[dict], t0: float, t1: float) -> dict:
    """Sum stage-0 module GPU ms whose launch falls in [t0, t1)."""
    out: dict[str, dict] = {}
    for e in events:
        ls = e.get("launch_start")
        if ls is None or not (t0 <= ls < t1):
            continue
        m = e["module"]
        s = out.setdefault(m, {"gpu_ms": 0.0, "calls": 0})
        s["gpu_ms"] += e.get("gpu_ms", 0.0)
        s["calls"] += 1
    return out


def device_busy_in(samples: list[dict], wall0: float, t0: float, t1: float) -> float:
    """Integrate device SM utilization over [t0, t1) -> GPU-busy seconds."""
    pts = [(wall0 + s["t"], s.get("sm", 0)) for s in samples if "sm" in s]
    pts = [(w, v) for w, v in pts if t0 <= w < t1]
    if len(pts) < 2:
        return 0.0
    busy = 0.0
    for (wa, va), (wb, _) in zip(pts, pts[1:]):
        busy += (wb - wa) * va / 100.0
    return busy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/data/zx/results/ttfa/ttfa_user*.jsonl")
    ap.add_argument("--events", default="/data/zx/results/stage0_events.jsonl")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--skip-reps", type=int, default=1,
                    help="drop the first N repetitions as warmup")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tracefiles = sorted(glob.glob(args.traces))
    if not tracefiles:
        print(f"no traces matching {args.traces}")
        return 1
    events = load(args.events) if pathlib.Path(args.events).exists() else []
    gsamples, gwall0 = [], None
    if args.gpu and pathlib.Path(args.gpu).exists():
        gev = load(args.gpu)
        gmeta = next((e for e in gev if e.get("k") == "meta"), {})
        gwall0 = gmeta.get("wall_start")
        gsamples = [e for e in gev if e.get("k") == "s"]

    all_rows = []
    per_user = {}
    for tf in tracefiles:
        meta, rows = reps_from_trace(load(tf))
        uid = meta.get("uid", 0)
        rows = [r for r in rows if r["rep"] >= args.skip_reps]
        per_user[uid] = (meta, rows)
        all_rows.extend([dict(r, uid=uid) for r in rows])

    users = len(per_user)
    print(f"\n=== TTFA decomposition: {users} user(s), "
          f"{len(all_rows)} measured repetitions "
          f"(first {args.skip_reps} dropped as warmup) ===")
    m0 = next(iter(per_user.values()))[0]
    print(f"utterance {m0.get('utterance_s', 0):.2f} s | fps {m0.get('fps')} | "
          f"chunk {m0.get('chunk_ms')} ms | num_frames {m0.get('num_frames')} | "
          f"EVS {m0.get('evs')}@{m0.get('evs_threshold')}")
    print(f"query: {m0.get('query')!r}")

    slips = sum(r.get("slips", 0) for r in all_rows)
    print(f"\nclient health: schedule slips = {slips}"
          f"{'  <-- CLIENT SATURATED, numbers unsafe' if slips else '  (ok)'}")

    hdr = (f"\n{'uid':>4}{'rep':>4}{'speech':>8}{'admit':>8}{'->1st tok':>10}"
           f"{'->1st aud':>10}{'TTFA_end':>10}{'TTFA_start':>11}{'chars':>7}")
    print(hdr)
    for r in all_rows:
        def g(k, f="{:.3f}"):
            v = r.get(k)
            return f.format(v) if isinstance(v, (int, float)) else "-"
        print(f"{r['uid']:>4}{r['rep']:>4}{g('speech_s'):>8}{g('admit_s'):>8}"
              f"{g('to_first_token_s'):>10}{g('to_first_audio_s'):>10}"
              f"{g('ttfa_end_s'):>10}{g('ttfa_start_s'):>11}"
              f"{r.get('chars','-'):>7}"
              f"{'  TIMEOUT' if r.get('timeout') else ''}")

    print(f"\n{'segment':<22}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}{'mean':>9}")
    agg = {}
    for key, label in (("admit_s", "admission"),
                       ("to_first_token_s", "-> first text token"),
                       ("to_first_audio_s", "-> first audio"),
                       ("ttfa_end_s", "TTFA (from sp. end)"),
                       ("ttfa_start_s", "TTFA (from sp. start)")):
        xs = [r[key] for r in all_rows if isinstance(r.get(key), (int, float))]
        if not xs:
            continue
        agg[key] = {"p50": pct(xs, 50), "p95": pct(xs, 95), "min": min(xs),
                    "max": max(xs), "mean": st.fmean(xs), "n": len(xs)}
        a = agg[key]
        print(f"{label:<22}{a['p50']:>9.3f}{a['p95']:>9.3f}{a['min']:>9.3f}"
              f"{a['max']:>9.3f}{a['mean']:>9.3f}")

    # share of TTFA_end taken by each segment
    if "ttfa_end_s" in agg:
        tt = agg["ttfa_end_s"]["p50"]
        print(f"\nshare of TTFA (p50, from speech end = {tt*1000:.0f} ms):")
        for key, label in (("admit_s", "admission"),
                           ("to_first_token_s", "encoders + vision prefill + 1st token"),
                           ("to_first_audio_s", "talker spin-up + code2wav")):
            if key in agg:
                print(f"  {label:<40}{agg[key]['p50']*1000:>8.0f} ms"
                      f"{agg[key]['p50']/tt*100:>8.1f}%")

    # ---- did anything happen on the GPU while the user was talking? ----
    if events:
        print(f"\n=== stage-0 GPU work: during user speech vs after the query ===")
        print(f"{'uid':>4}{'rep':>4}   {'module':<16}{'during speech':>15}"
              f"{'after query':>14}")
        tot_dur, tot_aft = {}, {}
        for r in all_rows:
            ss, q, fa = r.get("t_speech_start"), r.get("t_query"), r.get("t_first_audio")
            if not (ss and q and fa):
                continue
            during = gpu_work_in(events, ss, q)
            after = gpu_work_in(events, q, fa)
            for m in sorted(set(during) | set(after)):
                d = during.get(m, {}).get("gpu_ms", 0.0)
                a = after.get(m, {}).get("gpu_ms", 0.0)
                tot_dur[m] = tot_dur.get(m, 0.0) + d
                tot_aft[m] = tot_aft.get(m, 0.0) + a
                print(f"{r['uid']:>4}{r['rep']:>4}   {m:<16}"
                      f"{d:>13.1f}ms{a:>12.1f}ms")
        print(f"\n  {'TOTAL':<24}{'during speech':>15}{'after query':>14}")
        for m in sorted(set(tot_dur) | set(tot_aft)):
            print(f"  {m:<24}{tot_dur.get(m,0):>13.1f}ms{tot_aft.get(m,0):>12.1f}ms")

    if gsamples and gwall0:
        print(f"\n=== device GPU-busy (ALL stages, NVML) per repetition ===")
        print(f"{'uid':>4}{'rep':>4}{'speech win s':>14}{'busy in speech':>16}"
              f"{'util %':>8}   {'TTFA win s':>11}{'busy in TTFA':>14}{'util %':>8}")
        for r in all_rows:
            ss, q, fa = r.get("t_speech_start"), r.get("t_query"), r.get("t_first_audio")
            if not (ss and q and fa):
                continue
            w1, b1 = q - ss, device_busy_in(gsamples, gwall0, ss, q)
            w2, b2 = fa - q, device_busy_in(gsamples, gwall0, q, fa)
            r["gpu_busy_speech_frac"] = (b1 / w1) if w1 else None
            r["gpu_busy_ttfa_frac"] = (b2 / w2) if w2 else None
            print(f"{r['uid']:>4}{r['rep']:>4}{w1:>14.2f}{b1:>14.3f}s"
                  f"{(b1/w1*100 if w1 else 0):>8.1f}   {w2:>11.2f}{b2:>12.3f}s"
                  f"{(b2/w2*100 if w2 else 0):>8.1f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"users": users, "rows": all_rows, "aggregate": agg}, indent=2,
            default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
