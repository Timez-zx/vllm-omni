#!/usr/bin/env python3
"""Does a multi-turn session actually reuse KV context?

The decisive signal is the SLOPE of per-turn latency against turn index.

  If context accumulates and KV is NOT reused, turn N must re-prefill the whole
  accumulated context, so the stage-0 segment grows roughly linearly in N.

  If KV is reused (prefix caching hitting the accumulated prefix), turn N only
  prefills what is new, so the stage-0 segment stays flat even though the
  context keeps growing.

Comparing three context policies isolates that:
  L0_shipped   last 1 turn, text-only, prefix caching off  (vllm-omni default)
  L1_fulltext  all history, text-only, prefix caching on
  L2_fullmm    all history, frames kept, prefix caching on (accumulating vision)

A least-squares slope in ms/turn is reported per policy, with the first turn
excluded as warmup.
"""

from __future__ import annotations

import argparse
import json
import pathlib

POLICIES = [
    ("L0_shipped", "shipped: last 1 turn, text-only, PC off"),
    ("L1_fulltext", "full history, text-only, PC on"),
    ("L2_fullmm", "full history, frames kept, PC on"),
]
SEGS = [("to_first_token_s", "stage 0 (encode+prefill+1st tok)"),
        ("to_first_audio_s", "stage 1+2"),
        ("ttfa_end_s", "TTFA")]


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), my
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return m, my - m * mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdir", default="/data/zx/results")
    ap.add_argument("--skip", type=int, default=1, help="drop first N turns as warmup")
    args = ap.parse_args()
    R = pathlib.Path(args.resdir)

    loaded = {}
    for tag, label in POLICIES:
        p = R / f"ttfa_{tag}.json"
        if p.exists():
            loaded[tag] = (label, json.loads(p.read_text()))

    if not loaded:
        print("no long-session results yet")
        return 1

    for tag, (label, d) in loaded.items():
        rows = [r for r in d["rows"] if r["rep"] >= args.skip
                and isinstance(r.get("ttfa_end_s"), (int, float))]
        if not rows:
            print(f"\n### {tag}: no usable turns")
            continue
        print(f"\n### {tag} — {label}   ({len(rows)} turns)")
        print(f"{'turn':>5}{'stage0 ms':>11}{'stage1+2 ms':>13}{'TTFA ms':>10}{'chars':>7}")
        for r in rows:
            print(f"{r['rep']:>5}"
                  f"{r.get('to_first_token_s', float('nan'))*1000:>11.0f}"
                  f"{r.get('to_first_audio_s', float('nan'))*1000:>13.0f}"
                  f"{r['ttfa_end_s']*1000:>10.0f}"
                  f"{str(r.get('chars','-')):>7}")
        xs = [float(r["rep"]) for r in rows]
        print(f"  {'segment':<34}{'slope ms/turn':>15}{'turn1':>9}{'turnN':>9}")
        for k, lab in SEGS:
            ys = [r[k] * 1000 for r in rows if isinstance(r.get(k), (int, float))]
            xr = [float(r["rep"]) for r in rows if isinstance(r.get(k), (int, float))]
            if len(ys) < 2:
                continue
            m, b = fit(xr, ys)
            print(f"  {lab:<34}{m:>+15.1f}{ys[0]:>9.0f}{ys[-1]:>9.0f}")

    print("\n" + "=" * 74)
    print("斜率读法 / how to read the slope:")
    print("  ~0 ms/turn   上下文没有累积，或 KV 被成功复用")
    print("  >0 ms/turn   上下文在累积且每轮重算 -> 长会话延迟会持续恶化")
    print("=" * 74)

    # side-by-side TTFA and stage-0 slope comparison
    if len(loaded) > 1:
        print(f"\n{'policy':<14}{'TTFA turn1':>12}{'TTFA turnN':>12}"
              f"{'TTFA slope':>12}{'stage0 slope':>14}")
        for tag, (label, d) in loaded.items():
            rows = [r for r in d["rows"] if r["rep"] >= args.skip
                    and isinstance(r.get("ttfa_end_s"), (int, float))]
            if len(rows) < 2:
                continue
            xr = [float(r["rep"]) for r in rows]
            yt = [r["ttfa_end_s"] * 1000 for r in rows]
            mt, _ = fit(xr, yt)
            s0 = [(float(r["rep"]), r["to_first_token_s"] * 1000) for r in rows
                  if isinstance(r.get("to_first_token_s"), (int, float))]
            ms, _ = fit([a for a, _ in s0], [b for _, b in s0]) if len(s0) > 1 else (float("nan"), 0)
            print(f"{tag:<14}{yt[0]:>12.0f}{yt[-1]:>12.0f}{mt:>+12.1f}{ms:>+14.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
