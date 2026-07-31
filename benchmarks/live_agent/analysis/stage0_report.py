#!/usr/bin/env python3
"""Diff two stage0_probe snapshots and report the within-stage-0 breakdown.

The probe accumulates from process start, which includes vLLM's startup
profiling and CUDA-graph capture forwards. Those must not be counted as session
cost, so the measurement is taken as a difference between a snapshot captured
immediately before the session and one captured immediately after.

Reported per module:
  gpu_ms        GPU time attributed to that module during the session
  duty_cycle    gpu_ms / session wall time -- the absolute quantity that bounds
                capacity: a module at d% of wall clock admits at most ~100/d
                concurrent sessions before that module alone saturates the GPU
  share         split within stage 0

Also splits the LLM by forward size, so prefill-shaped steps (many tokens) are
distinguishable from decode-shaped steps (about one token per running seq).
"""

from __future__ import annotations

import argparse
import json
import pathlib


def load(p: str) -> dict:
    return json.loads(pathlib.Path(p).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--nvml-stage0-share", type=float, default=None,
                    help="stage-0 share of device GPU busy time from NVML, "
                         "used to project the within-stage-0 split to a global share")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b, a = load(args.before), load(args.after)
    wall = a["wall_s"] - b["wall_s"]

    names = sorted(set(a["modules"]) | set(b["modules"]))
    rows = []
    for n in names:
        am, bm = a["modules"].get(n, {}), b["modules"].get(n, {})
        dms = am.get("gpu_ms", 0.0) - bm.get("gpu_ms", 0.0)
        dc = am.get("calls", 0) - bm.get("calls", 0)
        dt = am.get("tokens", 0) - bm.get("tokens", 0)
        sizes = {}
        for k, v in (am.get("by_size") or {}).items():
            pv = (bm.get("by_size") or {}).get(k, {"gpu_ms": 0.0, "calls": 0})
            g = v["gpu_ms"] - pv["gpu_ms"]
            c = v["calls"] - pv["calls"]
            if c > 0 or g > 0.01:
                sizes[k] = {"gpu_ms": g, "calls": c}
        if dc > 0 or dms > 0.01:
            rows.append({"module": n, "gpu_ms": dms, "calls": dc, "tokens": dt,
                         "by_size": sizes})

    total = sum(r["gpu_ms"] for r in rows) or 1.0
    for r in rows:
        r["share_within_stage0"] = r["gpu_ms"] / total
        r["duty_cycle"] = r["gpu_ms"] / (wall * 1000.0) if wall > 0 else None
        r["ms_per_call"] = r["gpu_ms"] / r["calls"] if r["calls"] else None
        if args.nvml_stage0_share is not None:
            r["projected_global_share"] = r["share_within_stage0"] * args.nvml_stage0_share

    rows.sort(key=lambda r: -r["gpu_ms"])

    print(f"\nstage-0 internal breakdown over {wall:.1f} s of session wall time")
    print(f"(instrumented GPU time in stage 0: {total:.1f} ms "
          f"= {total/(wall*1000)*100:.2f}% of wall clock)\n")
    print(f"{'module':<16}{'GPU ms':>10}{'calls':>8}{'ms/call':>9}"
          f"{'duty %':>9}{'share in stage0':>17}"
          + (f"{'global %':>10}" if args.nvml_stage0_share is not None else ""))
    for r in rows:
        line = (f"{r['module']:<16}{r['gpu_ms']:>10.1f}{r['calls']:>8}"
                f"{(r['ms_per_call'] or 0):>9.2f}{r['duty_cycle']*100:>9.2f}"
                f"{r['share_within_stage0']*100:>16.1f}%")
        if args.nvml_stage0_share is not None:
            line += f"{r['projected_global_share']*100:>9.1f}%"
        print(line)

    # perception vs thinking, the number the coarse split could not give
    perc = sum(r["gpu_ms"] for r in rows
               if r["module"] in ("vision_encoder", "audio_encoder"))
    llm = sum(r["gpu_ms"] for r in rows if r["module"] == "language_model")
    print(f"\nperception (vision + audio encoders): {perc:.1f} ms "
          f"= {perc/total*100:.1f}% of stage 0, duty {perc/(wall*1000)*100:.3f}% of wall")
    print(f"LLM (thinking / text gen):            {llm:.1f} ms "
          f"= {llm/total*100:.1f}% of stage 0, duty {llm/(wall*1000)*100:.3f}% of wall")
    if args.nvml_stage0_share is not None:
        print(f"\nprojected to the whole device (stage 0 = "
              f"{args.nvml_stage0_share*100:.1f}% of device GPU busy time):")
        print(f"  perception = {perc/total*args.nvml_stage0_share*100:.2f}% of all GPU time")
        print(f"  LLM        = {llm/total*args.nvml_stage0_share*100:.2f}% of all GPU time")

    for r in rows:
        if r["module"] == "language_model" and r["by_size"]:
            print("\nlanguage_model forwards by token count "
                  "(large = prefill-shaped, small = decode-shaped):")
            print(f"  {'<=tokens':>10}{'calls':>8}{'GPU ms':>10}{'ms/call':>9}{'share':>8}")
            def keyf(k):
                try:
                    return int(k)
                except ValueError:
                    return 10 ** 9
            for k in sorted(r["by_size"], key=keyf):
                v = r["by_size"][k]
                mc = v["gpu_ms"] / v["calls"] if v["calls"] else 0
                print(f"  {k:>10}{v['calls']:>8}{v['gpu_ms']:>10.1f}{mc:>9.2f}"
                      f"{v['gpu_ms']/r['gpu_ms']*100:>7.1f}%")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"wall_s": wall, "instrumented_gpu_ms": total, "modules": rows}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
