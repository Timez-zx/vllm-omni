#!/usr/bin/env python3
"""EVS frame pre-filter probe.

Faithfully reimplements vllm_omni.entrypoints.openai.video_frame_filter.
FrameSimilarityFilter (as of vllm-omni v0.24.0) and measures its behaviour on
real video across motion regimes.

The shipped filter compares each incoming frame against the last *retained*
frame using a 64x64 RGB thumbnail and

    similarity = 1 - MSE / 255**2

dropping the frame when similarity >= threshold (default 0.95). Note that
threshold 0.95 corresponds to dropping whenever MSE <= 0.05 * 255**2 = 3251,
i.e. whenever RMSE <= 57 / 255 -- a very large pixel difference.

Outputs a JSON blob plus a human-readable table.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from PIL import Image

THUMB = 64
MAXVAL = 255.0


# ---------------------------------------------------------------------------
# Verbatim port of the shipped filter
# ---------------------------------------------------------------------------
class FrameSimilarityFilter:
    def __init__(self, threshold: float = 0.95, thumbnail_size: int = THUMB) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold
        self._thumbnail_size = thumbnail_size
        self._last_retained: np.ndarray | None = None
        self._retained_count = 0
        self._dropped_count = 0

    def should_retain(self, frame_jpeg: bytes) -> bool:
        current = self._decode_and_resize(frame_jpeg)
        if self._last_retained is None:
            self._last_retained = current
            self._retained_count += 1
            return True
        similarity = self._compute_similarity(self._last_retained, current)
        if similarity >= self._threshold:
            self._dropped_count += 1
            return False
        self._last_retained = current
        self._retained_count += 1
        return True

    @property
    def stats(self) -> dict:
        total = self._retained_count + self._dropped_count
        return {
            "retained_count": self._retained_count,
            "dropped_count": self._dropped_count,
            "total_count": total,
            "drop_rate": self._dropped_count / total if total else 0.0,
        }

    @staticmethod
    def _compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        return 1.0 - mse / (MAXVAL * MAXVAL)

    def _decode_and_resize(self, jpeg_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.resize(
            (self._thumbnail_size, self._thumbnail_size),
            Image.Resampling.BILINEAR,
        ).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
def _decode_one(path: str) -> np.ndarray:
    """Same pixel pipeline as FrameSimilarityFilter._decode_and_resize."""
    img = Image.open(path)
    img = img.resize((THUMB, THUMB), Image.Resampling.BILINEAR).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_thumbs_all(frame_dir: pathlib.Path, workers: int) -> np.ndarray:
    """Decode every frame once, at full rate, into a thumbnail stack.

    The shipped filter only ever looks at the 64x64 thumbnail, so sweeping
    thresholds over this stack is identical to re-running it on the JPEGs --
    just without paying for redundant JPEG decodes.
    """
    paths = [str(p) for p in sorted(frame_dir.glob("*.jpg"))]
    with mp.Pool(workers) as pool:
        thumbs = pool.map(_decode_one, paths, chunksize=8)
    return np.stack(thumbs)


def filter_on_thumbs(thumbs: np.ndarray, threshold: float) -> tuple[int, float]:
    """Run the shipped retain/drop logic over a thumbnail stack.

    Comparison is against the last RETAINED frame, so this is inherently
    sequential and cannot be vectorised away.
    """
    retained = 0
    dropped = 0
    last = None
    cut = (1.0 - threshold) * MAXVAL * MAXVAL  # drop iff mse <= cut
    for cur in thumbs:
        if last is None:
            last = cur.astype(np.float32)
            retained += 1
            continue
        c = cur.astype(np.float32)
        mse = float(np.mean((last - c) ** 2))
        if mse <= cut:  # similarity >= threshold
            dropped += 1
        else:
            last = c
            retained += 1
    total = retained + dropped
    return retained, (dropped / total if total else 0.0)


def consecutive_stats(thumbs: np.ndarray) -> dict:
    """MSE/RMSE between temporally adjacent frames (independent of filtering)."""
    a = thumbs[:-1].astype(np.float32)
    b = thumbs[1:].astype(np.float32)
    mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
    sim = 1.0 - mse / (MAXVAL * MAXVAL)
    return {
        "n_pairs": int(mse.size),
        "rmse_p50": float(np.percentile(np.sqrt(mse), 50)),
        "rmse_p90": float(np.percentile(np.sqrt(mse), 90)),
        "rmse_max": float(np.sqrt(mse).max()),
        "sim_p10": float(np.percentile(sim, 10)),
        "sim_p50": float(np.percentile(sim, 50)),
        "sim_min": float(sim.min()),
    }


def sweep(thumbs: np.ndarray, thresholds: list[float]) -> dict[float, float]:
    return {t: filter_on_thumbs(thumbs, t)[1] for t in thresholds}


def budget_analysis(n_retained: int, fps: float, max_frames: int, num_frames: int) -> dict:
    """What temporal coverage does the model actually receive?

    The shipped handler keeps at most `max_frames` in the buffer and uniformly
    strides that buffer down to `num_frames` when building the prompt.
    """
    buffered = min(n_retained, max_frames)
    sampled = min(buffered, num_frames)
    # wall-clock span the buffer covers, and the gap between frames the model sees
    span_s = buffered / fps if fps else float("nan")
    gap_s = span_s / sampled if sampled else float("nan")
    return {
        "retained_frames": n_retained,
        "buffered_frames": buffered,
        "frames_shown_to_model": sampled,
        "buffer_span_s": span_s,
        "effective_gap_between_shown_frames_s": gap_s,
        "effective_fps_seen_by_model": 1.0 / gap_s if gap_s else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", default="/data/zx/stimuli/frames")
    ap.add_argument("--base-fps", type=float, default=4.0)
    ap.add_argument("--out", default="/data/zx/results/evs_probe.json")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    root = pathlib.Path(args.stimuli)
    regimes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not regimes:
        print(f"no regime dirs under {root}", file=sys.stderr)
        return 1

    thresholds = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    # stride 1 -> 4 fps, 2 -> 2 fps, 4 -> 1 fps
    fps_strides = {4.0: 1, 2.0: 2, 1.0: 4}

    results: dict = {
        "meta": {
            "filter": "vllm-omni v0.24.0 FrameSimilarityFilter (verbatim port)",
            "similarity": "1 - MSE/255^2 on 64x64 RGB thumbnail vs last RETAINED frame",
            "shipped_default_threshold": 0.95,
            "shipped_default_num_frames": 4,
            "shipped_default_max_frames": 50,
            "base_fps": args.base_fps,
            "jpeg": "1280x720, ffmpeg -q:v 3",
        },
        "regimes": {},
    }

    for r in regimes:
        d = root / r
        results["regimes"][r] = {}
        all_thumbs = load_thumbs_all(d, args.workers)
        print(f"decoded {len(all_thumbs)} frames for {r}", file=sys.stderr)
        for fps, stride in fps_strides.items():
            thumbs = all_thumbs[::stride]
            if len(thumbs) < 3:
                continue
            drops = sweep(thumbs, thresholds)
            cons = consecutive_stats(thumbs)
            n_ret, _ = filter_on_thumbs(thumbs, 0.95)
            results["regimes"][r][f"{fps:g}fps"] = {
                "n_frames_in": int(len(thumbs)),
                "duration_s": len(thumbs) / fps,
                "consecutive": cons,
                "drop_rate_vs_threshold": {f"{t:g}": v for t, v in drops.items()},
                # server-side pydantic defaults
                "at_default_threshold_0.95": budget_analysis(n_ret, fps, 50, 4),
                # what the shipped reference client actually sends
                "at_reference_client_defaults": budget_analysis(n_ret, fps, 64, 16),
            }

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))

    # ---------------- human readable ----------------
    print(f"\nEVS pre-filter probe  (shipped default threshold = 0.95)")
    print(f"similarity = 1 - MSE/255^2 on 64x64 thumbnails; drop if sim >= threshold")
    print(f"threshold 0.95  <=>  drop whenever RMSE <= {np.sqrt(0.05)*255:.1f}/255\n")

    print("--- consecutive-frame difference of the RAW stream (before filtering) ---")
    print(f"{'regime':<22}{'fps':>5}{'pairs':>7}{'RMSE p50':>10}{'RMSE p90':>10}{'sim p50':>10}{'sim min':>10}")
    for r, per in results["regimes"].items():
        for fk, v in per.items():
            c = v["consecutive"]
            print(f"{r:<22}{fk:>5}{c['n_pairs']:>7}{c['rmse_p50']:>10.1f}"
                  f"{c['rmse_p90']:>10.1f}{c['sim_p50']:>10.4f}{c['sim_min']:>10.4f}")

    print("\n--- drop rate vs threshold ---")
    hdr = f"{'regime':<22}{'fps':>5}" + "".join(f"{t:>9g}" for t in thresholds)
    print(hdr)
    for r, per in results["regimes"].items():
        for fk, v in per.items():
            row = f"{r:<22}{fk:>5}"
            for t in thresholds:
                row += f"{v['drop_rate_vs_threshold'][f'{t:g}']*100:>8.1f}%"
            print(row)

    print("\n--- what the model actually sees at the shipped defaults ---")
    print(f"(enable_frame_filter=True, threshold=0.95, max_frames=50, num_frames=4)")
    print(f"{'regime':<22}{'fps':>5}{'in':>6}{'kept':>6}{'buf':>5}{'shown':>7}"
          f"{'span s':>8}{'gap s':>8}{'eff fps':>9}")
    for r, per in results["regimes"].items():
        for fk, v in per.items():
            b = v["at_default_threshold_0.95"]
            print(f"{r:<22}{fk:>5}{v['n_frames_in']:>6}{b['retained_frames']:>6}"
                  f"{b['buffered_frames']:>5}{b['frames_shown_to_model']:>7}"
                  f"{b['buffer_span_s']:>8.1f}{b['effective_gap_between_shown_frames_s']:>8.2f}"
                  f"{b['effective_fps_seen_by_model']:>9.3f}")

    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
