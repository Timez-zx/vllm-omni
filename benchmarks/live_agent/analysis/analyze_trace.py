#!/usr/bin/env python3
"""Reduce a paced_client trace (+ optional GPU sample log) to metrics.

Latency definitions used here (stated explicitly because the literature is
inconsistent about them):

  TTFT   t(first response.text.delta)  - t(video.query sent)
  TTFA   t(first response.audio.delta) - t(video.query sent)
         TTFA is the metric that matters for a voice agent: it is when the
         user starts hearing something. video.query is the proxy for "user
         stopped speaking".
  text_complete / audio_complete
         t(response.*.done) - t(video.query sent)
  audio RTF
         generated audio seconds / wall seconds spent generating it.
         RTF < 1 means the pipeline can sustain real-time speech output;
         RTF > 1 means audio arrives slower than it plays and the user hears
         gaps no matter how good TTFA was.
  audio delta gap
         inter-arrival time between consecutive response.audio.delta events.
         The tail of this distribution, not the mean, is what causes audible
         stutter.

Client-health checks that must pass before any server number is believed:
  frame/audio send lag and schedule slips. If the client could not keep its
  own send schedule, the session was not actually paced and the latency
  numbers describe a different workload than the one intended.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path: pathlib.Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def analyze_trace(ev: list[dict], audio_hz: int, audio_bytes_per_sample: int) -> dict:
    meta = next((e for e in ev if e.get("k") == "meta"), {})
    turns: dict[int, dict] = {}

    def T(i: int) -> dict:
        return turns.setdefault(i, {"audio_deltas": [], "text_deltas": [],
                                    "audio_bytes": 0, "frames": 0})

    for e in ev:
        k, t = e.get("k"), e.get("t")
        tn = e.get("turn")
        if k == "tx_query":
            T(tn)["t_query"] = t
            T(tn)["frames_total_at_query"] = e.get("frames_sent_total")
        elif k == "tx_frame":
            T(tn)["frames"] += 1
        elif k == "rx_response_start":
            T(tn)["t_start"] = t
        elif k == "rx_first_text_delta":
            T(tn)["t_first_text"] = t
        elif k == "rx_first_audio_delta":
            T(tn)["t_first_audio"] = t
        elif k == "rx_text_delta":
            T(tn)["text_deltas"].append(t)
        elif k == "rx_audio_delta":
            T(tn)["audio_deltas"].append(t)
            T(tn)["audio_bytes"] += e.get("nbytes", 0)
        elif k == "rx_text_done":
            T(tn)["t_text_done"] = t
            T(tn)["text"] = e.get("text", "")
        elif k == "rx_audio_done":
            T(tn)["t_audio_done"] = t
        elif k == "turn_timeout":
            T(tn)["timeout"] = True

    # client health
    frame_lag = [e["lag_s"] for e in ev if e.get("k") == "tx_frame"]
    audio_lag = [e["lag_s"] for e in ev if e.get("k") == "tx_audio"]
    slips = sum(1 for e in ev if e.get("k") in ("frame_schedule_slip", "audio_schedule_slip"))
    errors = [e for e in ev if e.get("k") in ("rx_error", "rx_closed")]

    per_turn = []
    for i in sorted(turns):
        d = turns[i]
        tq = d.get("t_query")
        row = {"turn": i, "frames_sent_in_window": d["frames"],
               "n_audio_deltas": len(d["audio_deltas"]),
               "n_text_deltas": len(d["text_deltas"]),
               "audio_bytes": d["audio_bytes"],
               "timeout": d.get("timeout", False),
               "text_preview": (d.get("text") or "")[:160]}
        if tq is None:
            per_turn.append(row)
            continue
        for name, key in (("ttft_s", "t_first_text"), ("ttfa_s", "t_first_audio"),
                          ("response_start_s", "t_start"),
                          ("text_complete_s", "t_text_done"),
                          ("audio_complete_s", "t_audio_done")):
            row[name] = (d[key] - tq) if key in d else None
        # audio real-time factor
        gen_s = d["audio_bytes"] / (audio_hz * audio_bytes_per_sample) if d["audio_bytes"] else 0.0
        row["audio_generated_s"] = gen_s
        if "t_first_audio" in d and "t_audio_done" in d and gen_s > 0:
            wall = d["t_audio_done"] - d["t_first_audio"]
            row["audio_wall_s"] = wall
            row["audio_rtf"] = (wall / gen_s) if gen_s else None
        # audio delta jitter
        gaps = [b - a for a, b in zip(d["audio_deltas"], d["audio_deltas"][1:])]
        if gaps:
            row["audio_gap_p50_ms"] = pct(gaps, 50) * 1000
            row["audio_gap_p95_ms"] = pct(gaps, 95) * 1000
            row["audio_gap_max_ms"] = max(gaps) * 1000
        per_turn.append(row)

    def agg(field: str) -> dict:
        xs = [r[field] for r in per_turn if r.get(field) is not None]
        if not xs:
            return {}
        return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95),
                "min": min(xs), "max": max(xs),
                "mean": st.fmean(xs)}

    return {
        "meta": meta,
        "client_health": {
            "frame_send_lag_p50_ms": pct(frame_lag, 50) * 1000 if frame_lag else None,
            "frame_send_lag_p95_ms": pct(frame_lag, 95) * 1000 if frame_lag else None,
            "frame_send_lag_max_ms": max(frame_lag) * 1000 if frame_lag else None,
            "audio_send_lag_p95_ms": pct(audio_lag, 95) * 1000 if audio_lag else None,
            "schedule_slips": slips,
            "n_frames_sent": len(frame_lag),
            "n_audio_chunks_sent": len(audio_lag),
            "errors": [e.get("message") or e.get("reason") for e in errors][:5],
        },
        "per_turn": per_turn,
        "aggregate": {k: agg(k) for k in
                      ("ttft_s", "ttfa_s", "text_complete_s", "audio_complete_s",
                       "audio_rtf", "audio_generated_s",
                       "audio_gap_p95_ms", "audio_gap_max_ms")},
    }


def analyze_gpu(ev: list[dict]) -> dict:
    meta = next((e for e in ev if e.get("k") == "meta"), {})
    smap: dict[str, str] = {}
    for e in ev:
        if e.get("k") == "stage_map":
            smap.update(e.get("map", {}))

    samples = [e for e in ev if e.get("k") == "s"]
    if not samples:
        return {"meta": meta, "note": "no samples"}

    dur = samples[-1]["t"] - samples[0]["t"]
    dev_sm = [s["sm"] for s in samples if "sm" in s]
    power = [s["power_w"] for s in samples if "power_w" in s]
    clocks = [s["sm_clock"] for s in samples if "sm_clock" in s]
    mem = [s["mem_used"] for s in samples if "mem_used" in s]
    throttled = sum(1 for s in samples if s.get("throttle", 0) not in (0, None))

    # integrate per-PID SM% over time -> share of GPU busy time per stage
    acc: dict[str, float] = {}
    n_with = 0
    for s in samples:
        pu = s.get("proc_sm") or {}
        if pu:
            n_with += 1
        for pid, v in pu.items():
            acc[pid] = acc.get(pid, 0.0) + float(v)
    total = sum(acc.values()) or 1.0
    by_stage: dict[str, float] = {}
    for pid, v in acc.items():
        label = smap.get(pid, f"pid{pid}")
        by_stage[label] = by_stage.get(label, 0.0) + v

    peak_proc_mem: dict[str, int] = {}
    for s in samples:
        for pid, v in (s.get("proc_mem") or {}).items():
            label = smap.get(pid, f"pid{pid}")
            peak_proc_mem[label] = max(peak_proc_mem.get(label, 0), int(v))

    return {
        "meta": meta,
        "duration_s": dur,
        "n_samples": len(samples),
        "device": {
            "sm_util_mean": st.fmean(dev_sm) if dev_sm else None,
            "sm_util_p95": pct(dev_sm, 95) if dev_sm else None,
            "sm_busy_fraction": (st.fmean(dev_sm) / 100.0) if dev_sm else None,
            "power_mean_w": st.fmean(power) if power else None,
            "power_max_w": max(power) if power else None,
            "sm_clock_mean_mhz": st.fmean(clocks) if clocks else None,
            "sm_clock_min_mhz": min(clocks) if clocks else None,
            "mem_used_peak_gb": max(mem) / 2**30 if mem else None,
            "throttled_samples": throttled,
            "throttled_fraction": throttled / len(samples),
        },
        "per_stage_sm_share": {k: v / total for k, v in
                               sorted(by_stage.items(), key=lambda kv: -kv[1])},
        "per_stage_peak_mem_gb": {k: v / 2**30 for k, v in
                                  sorted(peak_proc_mem.items(), key=lambda kv: -kv[1])},
        "samples_with_proc_sm": n_with,
        "stage_map": smap,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--audio-hz", type=int, default=24000,
                    help="output waveform sample rate (Qwen3-Omni code2wav = 24 kHz)")
    ap.add_argument("--audio-bytes-per-sample", type=int, default=2)
    args = ap.parse_args()

    res = {"trace": analyze_trace(load(pathlib.Path(args.trace)),
                                  args.audio_hz, args.audio_bytes_per_sample)}
    if args.gpu and pathlib.Path(args.gpu).exists():
        res["gpu"] = analyze_gpu(load(pathlib.Path(args.gpu)))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2))

    t = res["trace"]
    ch = t["client_health"]
    print("\n=== client health (must pass before trusting server numbers) ===")
    print(f"frames sent            {ch['n_frames_sent']}")
    print(f"audio chunks sent      {ch['n_audio_chunks_sent']}")
    fl = ch["frame_send_lag_p50_ms"]
    print(f"frame send lag         p50 {fl:.1f} ms  p95 {ch['frame_send_lag_p95_ms']:.1f} ms  "
          f"max {ch['frame_send_lag_max_ms']:.1f} ms" if fl is not None else "frame send lag  n/a")
    print(f"schedule slips         {ch['schedule_slips']}"
          f"{'   <-- CLIENT WAS THE BOTTLENECK' if ch['schedule_slips'] else ''}")
    if ch["errors"]:
        print(f"errors                 {ch['errors']}")

    print("\n=== per turn ===")
    hdr = (f"{'turn':>4}{'frames':>7}{'TTFT s':>9}{'TTFA s':>9}{'txt done':>10}"
           f"{'aud done':>10}{'aud gen s':>10}{'RTF':>7}{'gap p95':>9}{'gap max':>9}")
    print(hdr)
    for r in t["per_turn"]:
        def g(k, f="{:.3f}"):
            v = r.get(k)
            return f.format(v) if isinstance(v, (int, float)) else "-"
        print(f"{r['turn']:>4}{r['frames_sent_in_window']:>7}"
              f"{g('ttft_s'):>9}{g('ttfa_s'):>9}{g('text_complete_s'):>10}"
              f"{g('audio_complete_s'):>10}{g('audio_generated_s','{:.2f}'):>10}"
              f"{g('audio_rtf','{:.2f}'):>7}"
              f"{g('audio_gap_p95_ms','{:.0f}'):>9}{g('audio_gap_max_ms','{:.0f}'):>9}"
              f"{'  TIMEOUT' if r.get('timeout') else ''}")

    print("\n=== aggregate ===")
    for k, v in t["aggregate"].items():
        if v:
            print(f"{k:<20} n={v['n']:<3} p50={v['p50']:.3f}  p95={v['p95']:.3f}  "
                  f"min={v['min']:.3f}  max={v['max']:.3f}")

    if "gpu" in res:
        g = res["gpu"]
        d = g.get("device", {})
        print(f"\n=== GPU ({g['meta'].get('gpu')}) over {g.get('duration_s',0):.1f} s, "
              f"{g.get('n_samples')} samples ===")
        print(f"device SM util   mean {d.get('sm_util_mean')}%  p95 {d.get('sm_util_p95')}%")
        print(f"power            mean {d.get('power_mean_w')} W  max {d.get('power_max_w')} W "
              f"(limit {g['meta'].get('power_limit_w')} W)")
        print(f"SM clock         mean {d.get('sm_clock_mean_mhz')} MHz  min {d.get('sm_clock_min_mhz')} MHz "
              f"(max {g['meta'].get('max_sm_clock_mhz')} MHz)")
        print(f"mem peak         {d.get('mem_used_peak_gb'):.1f} GiB")
        print(f"throttled        {d.get('throttled_fraction',0)*100:.1f}% of samples")
        print("\nper-stage share of sampled GPU busy time:")
        for k, v in g.get("per_stage_sm_share", {}).items():
            print(f"  {k:<14}{v*100:>6.1f}%")
        print("per-stage peak GPU memory:")
        for k, v in g.get("per_stage_peak_mem_gb", {}).items():
            print(f"  {k:<14}{v:>6.1f} GiB")
        print(f"\n(samples carrying per-process SM data: {g.get('samples_with_proc_sm')}"
              f"/{g.get('n_samples')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
