#!/usr/bin/env python3
"""How much does each stimulus move, and what does EVS do to it?

Runs before any GPU time is spent, because it decides what the latency sweep can
actually vary. The chain under test is

    content motion -> EVS retention -> frames in the prompt -> vision tokens
                   -> prefill work -> TTFA

and the first two links can be measured offline by replaying the shipped filter.
If two stimuli turn out to retain the same number of frames, running both on the
GPU would produce the same latency and prove nothing -- better to find that out
here.

The filter is reproduced exactly: 64x64 thumbnail, compared against the last
*retained* frame, dropped when similarity >= threshold where
similarity = 1 - MSE/255**2.

Also reports raw inter-frame difference independent of the filter, so "how much
does this clip move" is separable from "what does this threshold do to it".
"""

from __future__ import annotations

import argparse
import pathlib
import statistics as st

import numpy as np
from PIL import Image

MAXVAL = 255.0
THUMB = 64


def thumbs(paths: list[pathlib.Path], stride: int) -> list[np.ndarray]:
    out = []
    for p in paths[::stride]:
        try:
            im = Image.open(p).convert("RGB").resize((THUMB, THUMB))
        except Exception:
            continue
        out.append(np.asarray(im, dtype=np.float64))
    return out


def replay_evs(ths: list[np.ndarray], threshold: float) -> tuple[int, int]:
    """(retained, dropped) under the shipped rule."""
    cut = (1.0 - threshold) * MAXVAL * MAXVAL
    last = None
    kept = dropped = 0
    for c in ths:
        if last is None:
            last, kept = c, kept + 1
            continue
        if float(np.mean((last - c) ** 2)) <= cut:
            dropped += 1
        else:
            last = c
            kept += 1
    return kept, dropped


def consecutive_similarity(ths: list[np.ndarray]) -> list[float]:
    """1 - MSE/255^2 between CONSECUTIVE frames -- motion, independent of EVS."""
    out = []
    for a, b in zip(ths, ths[1:]):
        mse = float(np.mean((a - b) ** 2))
        out.append(1.0 - mse / (MAXVAL * MAXVAL))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="frame directories (or a parent containing scene_* dirs)")
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample frames before analysis (speed)")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="rate the client will send at, for the frames/second column")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.99, 0.98, 0.95, 0.90, 0.80])
    ap.add_argument("--tokens-per-frame", type=float, default=895.0,
                    help="measured: 891 and 903 from two independent step tests")
    args = ap.parse_args()

    dirs: list[pathlib.Path] = []
    for d in args.dirs:
        p = pathlib.Path(d)
        if not p.exists():
            print(f"  missing {p}")
            continue
        if any(p.glob("*.jpg")):
            dirs.append(p)
        else:
            subs = sorted(x for x in p.iterdir() if x.is_dir() and any(x.glob("*.jpg")))
            dirs.extend(subs)
    if not dirs:
        print("no frame directories found")
        return 1

    print("\n=== MOTION (consecutive-frame similarity; 1.0 = identical) ===")
    print(f"{'stimulus':<34}{'frames':>8}{'sim p50':>10}{'sim p05':>10}{'sim min':>10}")
    cache: dict[str, list[np.ndarray]] = {}
    for d in dirs:
        paths = sorted(d.glob("*.jpg"))
        t = thumbs(paths, args.stride)
        cache[str(d)] = t
        if len(t) < 2:
            print(f"{d.name:<34}{len(t):>8}{'-':>10}{'-':>10}{'-':>10}")
            continue
        sims = consecutive_similarity(t)
        sims_sorted = sorted(sims)
        print(f"{d.name:<34}{len(t):>8}{st.median(sims):>10.4f}"
              f"{sims_sorted[int(0.05 * len(sims_sorted))]:>10.4f}{min(sims):>10.4f}")
    print("  Higher similarity = less motion. EVS drops a frame when its similarity")
    print("  to the last RETAINED frame is >= threshold, so high-similarity clips")
    print("  collapse to very few frames.")

    print(f"\n=== EVS RETENTION vs THRESHOLD (default shipped threshold = 0.95) ===")
    hdr = f"{'stimulus':<34}{'frames':>8}"
    for th in args.thresholds:
        hdr += f"{('t=' + str(th)):>12}"
    print(hdr)
    retention: dict[str, dict[float, int]] = {}
    for d in dirs:
        t = cache[str(d)]
        if len(t) < 2:
            continue
        row = f"{d.name:<34}{len(t):>8}"
        retention[str(d)] = {}
        for th in args.thresholds:
            kept, _ = replay_evs(t, th)
            retention[str(d)][th] = kept
            row += f"{kept:>12}"
        print(row)

    print(f"\n=== WHAT THAT COSTS (at {args.fps} fps, ~{args.tokens_per_frame:.0f} tok/frame) ===")
    print(f"{'stimulus':<34}{'clip s':>9}{'kept @0.95':>12}{'kept/s':>9}"
          f"{'tok/s of video':>16}")
    for d in dirs:
        t = cache[str(d)]
        if len(t) < 2 or str(d) not in retention:
            continue
        n = len(t)
        secs = n * args.stride / args.fps
        kept = retention[str(d)][0.95] if 0.95 in retention[str(d)] else None
        if kept is None:
            continue
        kps = kept / secs
        print(f"{d.name:<34}{secs:>9.1f}{kept:>12}{kps:>9.3f}"
              f"{kps * args.tokens_per_frame:>16.1f}")
    print("  'tok/s of video' is the rate at which retained frames would add context")
    print("  if every retained frame were kept. Compare: audio is ~24.6 tok/s")
    print("  measured, and Gemini Live's documented video rate is ~258 tok/s.")

    print("\n=== WHAT THE LATENCY SWEEP CAN VARY ===")
    at95 = {pathlib.Path(k).name: v[0.95] for k, v in retention.items() if 0.95 in v}
    if at95:
        lo = min(at95.values())
        hi = max(at95.values())
        print(f"  retained frames at the shipped threshold span {lo} .. {hi} "
              f"({hi / max(lo, 1):.1f}x)")
        for name, v in sorted(at95.items(), key=lambda kv: kv[1]):
            print(f"    {name:<32} {v:>5}")
        if hi / max(lo, 1) < 2:
            print("  WARNING: the stimuli barely differ in retained frames, so an")
            print("  EVS-on content sweep would not separate them. Vary num_frames")
            print("  directly with EVS off instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
