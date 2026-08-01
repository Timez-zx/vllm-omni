#!/usr/bin/env python3
"""Multi-user overhead: what memory costs when users contend, and where.

Consumes `decomp_mu_<policy>_u<N>.json` from ttfa_decompose (per-turn latency
segments) and `session_breakdown.json` (per-component GPU share), and answers:

  1. latency per segment, vs user count, vs policy
  2. latency SHARE per segment -- where the time goes, not just how much
  3. A/B overhead: text_memory minus shipped at equal user count. This is the
     honest way to price memory under contention, because inside stage 0 the
     memory-note call and other users' turns share one process and NVML cannot
     separate them (see session_breakdown.py).
  4. turn-index effect: does contention make the memory slope worse
  5. saturation: GPU busy while the user is still speaking

The three segments sum exactly to TTFA (verified: 0.0203 + 0.0956 + 0.2614 =
0.3773 = ttfa_end_s on a sample row), so the shares are exact, not normalised.

    multiuser_overhead.py [--users 1,2,4] [--policies shipped,text_memory]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

SEGS = [
    ("admit_s", "admission"),
    ("to_first_token_s", "encoders + prefill + 1st token"),
    ("to_first_audio_s", "talker spin-up + code2wav"),
]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def load(res: pathlib.Path, pol: str, u: int) -> dict | None:
    p = res / f"decomp_mu_{pol}_u{u}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rows_of(d: dict) -> list[dict]:
    return [r for r in d.get("rows", []) if r.get("ttfa_end_s") is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--users", default="1,2,4")
    ap.add_argument("--policies", default="shipped,text_memory")
    ap.add_argument("--breakdown", default="/data/zx/results/session_breakdown.json")
    ap.add_argument("--out", default="/data/zx/results/multiuser_overhead.json")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    users = [int(x) for x in args.users.split(",")]
    pols = args.policies.split(",")

    data: dict[tuple[str, int], dict] = {}
    for pol in pols:
        for u in users:
            d = load(res, pol, u)
            if d:
                data[(pol, u)] = d

    if not data:
        print("no decomp_mu_* files found")
        return 1

    out: dict = {"arms": {}}

    # ---------------------------------------------------- 1. absolute latency --
    print("\n=== LATENCY PER SEGMENT (ms, p50 / p95) ===")
    hdr = f"{'policy':<13}{'users':>6}{'turns':>7}"
    for _, lab in SEGS:
        hdr += f"{lab:>34}"
    hdr += f"{'TTFA':>20}"
    print(hdr)
    for (pol, u), d in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs = rows_of(d)
        line = f"{pol:<13}{u:>6}{len(rs):>7}"
        rec: dict = {"users": u, "policy": pol, "n_turns": len(rs)}
        for k, _ in SEGS + [("ttfa_end_s", "TTFA")]:
            v = [r[k] for r in rs if r.get(k) is not None]
            p50, p95 = pct(v, .50), pct(v, .95)
            rec[k] = {"p50_ms": round(p50 * 1000, 1) if p50 else None,
                      "p95_ms": round(p95 * 1000, 1) if p95 else None}
            w = 34 if k != "ttfa_end_s" else 20
            line += f"{(f'{p50*1000:.0f} / {p95*1000:.0f}' if p50 else '-'):>{w}}"
        print(line)
        out["arms"][f"{pol}_u{u}"] = rec

    # ------------------------------------------------------- 2. latency share --
    print("\n=== LATENCY SHARE OF TTFA (where the time goes) ===")
    print(f"{'policy':<13}{'users':>6}" + "".join(f"{lab:>34}" for _, lab in SEGS))
    for (pol, u), d in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs = rows_of(d)
        tot = sum(r["ttfa_end_s"] for r in rs)
        line = f"{pol:<13}{u:>6}"
        shares = {}
        for k, _ in SEGS:
            s = sum(r.get(k) or 0.0 for r in rs)
            frac = s / tot if tot else 0.0
            shares[k] = round(frac, 4)
            line += f"{frac:>33.1%} "
        print(line)
        out["arms"][f"{pol}_u{u}"]["ttfa_share"] = shares

    # -------------------------------------------------- 3. A/B memory overhead --
    if len(pols) >= 2:
        base, treat = pols[0], pols[1]
        print(f"\n=== MEMORY OVERHEAD: {treat} minus {base}, at equal user count ===")
        print(f"{'users':>6}{'TTFA base':>12}{'TTFA mem':>11}{'delta':>9}{'ratio':>8}"
              f"{'speech-out base':>17}{'speech-out mem':>16}{'delta':>9}")
        for u in users:
            b, t = data.get((base, u)), data.get((treat, u))
            if not b or not t:
                continue
            bb, tt = rows_of(b), rows_of(t)
            bt = pct([r["ttfa_end_s"] for r in bb], .5)
            tv = pct([r["ttfa_end_s"] for r in tt], .5)
            bs = pct([r["to_first_audio_s"] for r in bb if r.get("to_first_audio_s")], .5)
            ts = pct([r["to_first_audio_s"] for r in tt if r.get("to_first_audio_s")], .5)
            if not (bt and tv):
                continue
            print(f"{u:>6}{bt*1000:>11.0f}m{tv*1000:>10.0f}m{(tv-bt)*1000:>8.0f}m"
                  f"{tv/bt:>7.2f}x"
                  f"{(bs or 0)*1000:>16.0f}m{(ts or 0)*1000:>15.0f}m"
                  f"{((ts or 0)-(bs or 0))*1000:>8.0f}m")
            out.setdefault("overhead", {})[f"u{u}"] = {
                "ttfa_base_ms": round(bt * 1000, 1),
                "ttfa_mem_ms": round(tv * 1000, 1),
                "delta_ms": round((tv - bt) * 1000, 1),
                "ratio": round(tv / bt, 3),
            }
        print("  'speech-out' is the talker+code2wav segment -- the expensive path.")

    # ------------------------------------------------------- 4. turn-index fx --
    print("\n=== TTFA vs TURN INDEX (does contention worsen the memory slope?) ===")
    print(f"{'policy':<13}{'users':>6}{'turn1':>8}{'last':>8}{'slope ms/turn':>15}"
          f"{'  per-turn p50 series (ms)'}")
    for (pol, u), d in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs = rows_of(d)
        by_rep: dict[int, list[float]] = {}
        for r in rs:
            by_rep.setdefault(r["rep"], []).append(r["ttfa_end_s"] * 1000)
        reps = sorted(by_rep)
        series = [st.median(by_rep[k]) for k in reps]
        if len(series) < 2:
            continue
        n = len(series)
        mx, my = (n - 1) / 2, sum(series) / n
        den = sum((i - mx) ** 2 for i in range(n))
        slope = sum((i - mx) * (y - my) for i, y in enumerate(series)) / den if den else 0
        print(f"{pol:<13}{u:>6}{series[0]:>8.0f}{series[-1]:>8.0f}{slope:>15.1f}"
              f"  {[round(x) for x in series]}")
        out["arms"][f"{pol}_u{u}"]["ttfa_slope_ms_per_turn"] = round(slope, 1)
        out["arms"][f"{pol}_u{u}"]["ttfa_series_ms"] = [round(x) for x in series]

    # ---------------------------------------------------------- 5. saturation --
    print("\n=== SATURATION: GPU busy while the user is STILL SPEAKING ===")
    print(f"{'policy':<13}{'users':>6}{'busy during speech':>21}{'busy during TTFA':>19}")
    for (pol, u), d in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs = rows_of(d)
        sp = [r["gpu_busy_speech_frac"] for r in rs if r.get("gpu_busy_speech_frac") is not None]
        tf = [r["gpu_busy_ttfa_frac"] for r in rs if r.get("gpu_busy_ttfa_frac") is not None]
        print(f"{pol:<13}{u:>6}"
              f"{(st.median(sp)*100 if sp else float('nan')):>20.1f}%"
              f"{(st.median(tf)*100 if tf else float('nan')):>18.1f}%")
        if sp:
            out["arms"][f"{pol}_u{u}"]["busy_during_speech"] = round(st.median(sp), 4)
    print("  0% during speech means the device was idle while the user talked --")
    print("  headroom that a scheduler could use. High values mean saturation.")

    # ----------------------------------------- 6. resource share, if available --
    bp = pathlib.Path(args.breakdown)
    if bp.exists():
        bd = json.loads(bp.read_text()).get("arms", {})
        if bd:
            ORDER = ["vision_encoder", "audio_encoder", "memory_note",
                     "thinker_llm_residual", "talker", "code2wav"]
            PRETTY = {"vision_encoder": "vis enc", "audio_encoder": "aud enc",
                      "memory_note": "mem note", "thinker_llm_residual": "thinker LLM",
                      "talker": "talker", "code2wav": "code2wav"}
            print("\n=== RESOURCE SHARE PER COMPONENT (from session_breakdown) ===")
            print(f"{'arm':<22}" + "".join(f"{PRETTY[c]:>13}" for c in ORDER)
                  + f"{'dev duty':>10}")
            for tag, a in sorted(bd.items()):
                row = "".join(
                    f"{a.get('components_share_of_sum', {}).get(c, 0):>12.1%} "
                    for c in ORDER)
                print(f"{a.get('label', tag):<22}{row}{(a.get('device_duty') or 0):>9.1%}")
            out["resource_share"] = {t: a.get("components_share_of_sum")
                                     for t, a in bd.items()}
    else:
        print(f"\n(no {bp} yet -- run session_breakdown.py for the resource split)")

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
