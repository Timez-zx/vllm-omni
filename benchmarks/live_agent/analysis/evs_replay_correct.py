#!/usr/bin/env python3
"""Re-derive every EVS retention number using the SHIPPED filter, byte-for-byte.

WHY THIS FILE EXISTS. An earlier offline replay (`analysis/evs_embed_probe.py`)
approximated the shipped filter as a 64x64 GRAYSCALE thumbnail resized with PIL's
default resampler (BICUBIC). The shipped filter is neither:

    video_frame_filter.py::_decode_and_resize
        img.resize((n, n), Image.Resampling.BILINEAR).convert("RGB")
        np.asarray(img, dtype=np.uint8)                       # 64 x 64 x 3
    video_frame_filter.py::_compute_similarity
        mse = mean((a.astype(f32) - b.astype(f32)) ** 2)      # over RGB
        return 1.0 - mse / (255.0 * 255.0)

Note the order: resize FIRST, convert to RGB after, and the MSE is over three
channels rather than one. That single substitution inflated a whole family of
published numbers, all in the same direction:

    handheld retention at 0.95      156/253 (61.7%)  ->  143/253 (56.5%)
    cross-content spread            620x             ->  490x
    "84% retained, so ~17 new frames per turn"       ->  56.5%, ~11.8 frames

so the corrected figures are re-derived here from the installed implementation
itself, imported by file path rather than re-implemented, so it cannot drift again.

SAMPLING MODEL. The stimulus directories are already a 2 fps extraction of the source
video, and `ttfa_bench.py` sends `frames[frame_i % len(frames)]` on a 2 fps clock --
every file, in order, at 2 fps. So the operative retention is over the FULL file list.
An additional every-other-file subsample (which an earlier analysis used, and which is
where "84%" came from) corresponds to 1 fps playback and is not what any arm ran; it is
reported here only to show the gap.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib

STIMS = ("screencast", "talkinghead", "handheld_walk_talk")
LABEL = {"screencast": "screencast (static screen)",
         "talkinghead": "talkinghead (person sitting)",
         "handheld_walk_talk": "handheld (person walking)"}


def shipped_filter_cls():
    """Import FrameSimilarityFilter from the INSTALLED package, by file path.

    Imported rather than re-implemented on purpose: a hand port is exactly how the
    grayscale/bicubic error entered, and importing makes the analysis track whatever
    the deployment actually runs.
    """
    import vllm_omni
    p = (pathlib.Path(vllm_omni.__file__).parent
         / "entrypoints" / "openai" / "video_frame_filter.py")
    spec = importlib.util.spec_from_file_location("_vff", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FrameSimilarityFilter, p


def retention(Filter, blobs, threshold, min_gap=0, max_gap=0):
    """Retained count under the shipped filter, optionally gap-bracketed.

    The bracket semantics match harness/patches/video_stream_base.APPEND.py:
      * MIN_GAP drops without consulting the filter AND without moving its reference
        frame, so rate limiting cannot smuggle a near-duplicate through;
      * MAX_GAP clears the reference so the next frame is retained unconditionally.
    """
    f = Filter(threshold=threshold)
    kept, gap = 0, 10 ** 9
    for b in blobs:
        gap += 1
        if min_gap and gap < min_gap:
            continue
        if max_gap and gap >= max_gap:
            f._last_retained = None
        if f.should_retain(b):
            kept += 1
            gap = 0
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--min-gap", type=int, default=4)
    ap.add_argument("--max-gap", type=int, default=10)
    ap.add_argument("--out", default="/data/zx/results/evs_replay_correct.json")
    a = ap.parse_args()

    Filter, path = shipped_filter_cls()
    print(f"imported the installed filter from\n  {path}\n")

    data = {}
    for s in STIMS:
        fs = sorted(glob.glob(f"/data/zx/stimuli/frames/{s}/*"))
        data[s] = [open(p, "rb").read() for p in fs]
        print(f"  {s:<20} {len(fs)} files")

    print("\n" + "=" * 92)
    print(f"RETENTION AT THE SHIPPED THRESHOLD {a.threshold}, and gap-bracketed "
          f"[{a.min_gap},{a.max_gap}]")
    print("=" * 92)
    print(f"{'content':<28}{'shipped':>18}{'bracketed':>18}{'per-turn frames*':>18}")
    out = {}
    for s in STIMS:
        n = len(data[s])
        base = retention(Filter, data[s], a.threshold)
        brac = retention(Filter, data[s], a.threshold, a.min_gap, a.max_gap)
        # frames sent per turn, measured: ttfa_bench sends every file at 2 fps, and the
        # 1-user high-motion turn cycle was 9.16 s, so ~18-21 files arrive per turn.
        per_turn = 20.8 * brac / n
        out[s] = {"n": n, "shipped": base, "shipped_pct": 100 * base / n,
                  "bracketed": brac, "bracketed_pct": 100 * brac / n,
                  "retained_per_turn": per_turn}
        print(f"{LABEL[s]:<28}{f'{base}/{n} ({100*base/n:.1f}%)':>18}"
              f"{f'{brac}/{n} ({100*brac/n:.1f}%)':>18}{per_turn:>18.1f}")
    print("  * retained frames per turn at the measured 20.8 files arriving per turn")

    sp_base = max(o["shipped_pct"] for o in out.values()) / min(o["shipped_pct"] for o in out.values())
    sp_brac = max(o["bracketed_pct"] for o in out.values()) / min(o["bracketed_pct"] for o in out.values())
    print(f"\n  cross-content spread: shipped {sp_base:.1f}x  ->  bracketed {sp_brac:.2f}x")

    print("\n--- the substitution that inflated the earlier figures ---")
    print("  For reference only: the earlier analysis used a GRAYSCALE thumbnail with")
    print("  PIL's default BICUBIC resampler, and additionally replayed every OTHER")
    print("  file (a 1 fps model no arm ran). Both raised the handheld number.")
    import io
    import numpy as np
    from PIL import Image

    def wrong_thumb(b, n=64):
        return np.asarray(Image.open(io.BytesIO(b)).convert("L").resize((n, n)),
                          dtype=np.float32)

    def wrong_retention(blobs, th):
        last, kept = None, 0
        for b in blobs:
            cur = wrong_thumb(b)
            if last is None or 1.0 - float(np.mean((last - cur) ** 2)) / 255 ** 2 < th:
                last, kept = cur, kept + 1
        return kept

    print(f"\n{'content':<28}{'shipped (correct)':>20}{'gray/bicubic':>16}"
          f"{'gray/bicubic @1fps':>21}")
    for s in STIMS:
        n = len(data[s])
        w_all = wrong_retention(data[s], a.threshold)
        half = data[s][::2]
        w_half = wrong_retention(half, a.threshold)
        sh = out[s]['shipped']
        print(f"{LABEL[s]:<28}{f'{sh}/{n}':>20}"
              f"{f'{w_all}/{n}':>16}{f'{w_half}/{len(half)} ({100*w_half/len(half):.0f}%)':>21}")
        out[s]["gray_bicubic"] = w_all
        out[s]["gray_bicubic_1fps"] = f"{w_half}/{len(half)}"

    pathlib.Path(a.out).write_text(json.dumps(
        {"threshold": a.threshold, "min_gap": a.min_gap, "max_gap": a.max_gap,
         "spread_shipped": sp_base, "spread_bracketed": sp_brac,
         "filter_source": str(path), "per_content": out}, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
