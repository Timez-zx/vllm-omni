#!/usr/bin/env python3
"""Can the model's OWN vision embedding replace the 64x64 grayscale MSE frame filter?

The shipped filter (`FrameSimilarityFilter`) resizes each frame to a 64x64 grayscale
thumbnail and drops it when 1 - MSE/255^2 >= threshold against the last retained
frame. Measured on this study's stimuli, that metric fails in two opposite
directions:

  * handheld video (whole-frame motion) retains 156 of 253 frames at the shipped
    0.95 -- far too many, and this is the direct cause of high-motion latency;
  * a 7.2-minute screencast retains 1 of 867, even though 21.5% of its pixels differ
    between the first and last frame at full resolution. Downsampling to 64x64 and
    averaging destroys exactly the localized, high-frequency change (text, slides)
    that carries the information. The synthetic recall set is the extreme case: nine
    completely different rendered words retain ONE frame in total.

So the filter is blind to the changes that matter for a screen-sharing workload and
oversensitive to the ones that do not matter for a walking-camera workload. Tuning
the threshold cannot fix both, because the defect is in the metric.

This script measures the alternative: cosine distance between the model's own
vision-encoder embeddings. That is the semantically right quantity -- it measures
change in the space the model actually reasons in -- and the question is whether it
is affordable and whether it separates informative from redundant frames better.

Affordability, from this study's own probe rather than a guess: the vision encoder
ran 71,514 ms of device time over 8,209,140 patches = 2,332 frames at 1280x720, i.e.
**30.7 ms per frame**. At 640x352 a frame is a quarter of the patches, so ~7.7 ms.
Filtering means encoding EVERY arriving frame rather than only retained ones, which
at 2 fps costs ~15 ms/s of device time -- and it lands in the idle window, which was
measured at 5.20 s of every 9.16 s turn (57%) with nothing running for that user.
The tower itself is 538.6 M parameters, ~1.08 GB in bf16, and lives in a single
checkpoint shard, so it can be loaded standalone without the 63 GB thinker.

GROUND TRUTH. The recall stimuli are the only labelled set available: nine scenes,
each a distinct rendered word, 40 frames apiece. A scene boundary is by construction
an informative change; a within-scene step is by construction redundant. Any
candidate metric can therefore be scored as a detector, not just described.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image

M = ("/data/zx/hf/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/"
     "26291f793822fb6be9555850f06dfe95f2d7e695")


# ---------------------------------------------------------------- pixel baselines

def thumb_gray(b: bytes, n: int = 64) -> np.ndarray:
    im = Image.open(io.BytesIO(b)).convert("L").resize((n, n))
    return np.asarray(im, dtype=np.float32)


def shipped_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Exactly the shipped metric: 1 - MSE/255^2 on a 64x64 grayscale thumbnail."""
    return 1.0 - float(np.mean((a - b) ** 2)) / (255.0 ** 2)


def frac_changed(a: np.ndarray, b: np.ndarray, tol: float = 16.0) -> float:
    """Fraction of pixels that moved by more than `tol`. Not diluted by area."""
    return float(np.mean(np.abs(a - b) > tol))


def blockmax_mse(a: np.ndarray, b: np.ndarray, blocks: int = 8) -> float:
    """Worst block's MSE rather than the whole-frame mean."""
    n = a.shape[0] // blocks
    worst = 0.0
    for i in range(blocks):
        for j in range(blocks):
            sa = a[i * n:(i + 1) * n, j * n:(j + 1) * n]
            sb = b[i * n:(i + 1) * n, j * n:(j + 1) * n]
            worst = max(worst, float(np.mean((sa - sb) ** 2)))
    return worst / (255.0 ** 2)


# ---------------------------------------------------------------- the model tower

def load_vision_tower(device: str = "cuda", dtype=torch.bfloat16):
    """Instantiate ONLY thinker.visual and load its weights from the one shard.

    Loading the full checkpoint would pull 63 GB for a 1.08 GB module. The index
    says every `thinker.visual.*` tensor lives in shard 1, so only that file is read.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    cfg = AutoConfig.from_pretrained(M, trust_remote_code=True)
    vcfg = cfg.thinker_config.vision_config
    # NOT on the meta device. Non-persistent buffers (rotary inv_freq and friends) are
    # absent from the checkpoint, so a meta-instantiated module leaves them dataless and
    # `.to(cuda)` then fails with "cannot copy out of meta tensor". Instantiating for
    # real costs a few seconds of random init that load_state_dict immediately
    # overwrites, which is cheaper than special-casing every buffer.
    tower = Qwen3OmniMoeVisionEncoder(vcfg)

    idx = json.loads((pathlib.Path(M) / "model.safetensors.index.json").read_text())
    shards = {v for k, v in idx["weight_map"].items() if k.startswith("thinker.visual.")}
    sd = {}
    for s in shards:
        blob = load_file(str(pathlib.Path(M) / s))
        for k, v in blob.items():
            if k.startswith("thinker.visual."):
                sd[k[len("thinker.visual."):]] = v
        del blob
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    tower = tower.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in tower.parameters())
    print(f"  vision tower loaded: {n/1e6:.1f} M params, {n*2/1e9:.2f} GB bf16, "
          f"{len(shards)} shard(s) read")
    if missing:
        print(f"  NOTE {len(missing)} missing keys (first: {missing[:2]})")
    if unexpected:
        print(f"  NOTE {len(unexpected)} unexpected keys (first: {unexpected[:2]})")
    return tower, vcfg


def embed_pairwise(tower, imgs: list[Image.Image], device="cuda",
                   dtype=torch.bfloat16) -> tuple[dict, float]:
    """Consecutive-frame embedding metrics, computed streaming.

    MEAN POOLING IS NOT A FAIR TEST OF THE EMBEDDING IDEA and measuring only that
    would wrongly kill it. Averaging 3,520 patch vectors into one commits exactly the
    dilution error that makes whole-frame MSE blind: a changed word moves a few dozen
    patches, and the mean barely notices (measured separation margin 0.0002, the worst
    of every candidate). So the non-pooled readings are computed too:

        mean cosine      the diluted baseline, for reference
        min patch cosine the single most-changed patch -- the analogue of block-max
        frac patches     fraction of patches whose own cosine dropped below a cut
        mean L2          magnitude rather than angle, also pooled

    Grids are compared streaming (only the previous frame is held) because keeping 45
    frames x 3,520 patches x hidden would be ~0.8 GB for no reason.
    """
    from transformers import AutoProcessor
    if not hasattr(embed_pairwise, "_proc"):
        embed_pairwise._proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
    proc = embed_pairwise._proc

    def grid_of(im):
        px = proc.image_processor(images=[im], return_tensors="pt")
        pv = px["pixel_values"].to(device=device, dtype=dtype)
        gt = px["image_grid_thw"].to(device)
        with torch.no_grad():
            h = tower(pv, grid_thw=gt)
        if hasattr(h, "last_hidden_state"):
            h = h.last_hidden_state
        elif isinstance(h, (tuple, list)):
            h = h[0]
        return h.float()                          # [n_patches, hidden]

    out = {"mean cosine": [], "min patch cosine": [],
           "frac patches moved": [], "mean L2 (sim)": []}
    t0 = time.perf_counter()
    prev = None
    for im in imgs:
        cur = grid_of(im)
        if prev is not None:
            n = min(prev.shape[0], cur.shape[0])
            a, b = prev[:n], cur[:n]
            pc = torch.nn.functional.cosine_similarity(a, b, dim=-1)   # per patch
            out["mean cosine"].append(float(pc.mean()))
            out["min patch cosine"].append(float(pc.min()))
            # "moved" = this patch's own cosine fell below 0.9; report as a similarity
            # so every metric in the table shares the convention "keep when low".
            out["frac patches moved"].append(1.0 - float((pc < 0.9).float().mean()))
            am, bm = a.mean(0), b.mean(0)
            out["mean L2 (sim)"].append(
                1.0 / (1.0 + float(torch.linalg.vector_norm(am - bm))))
        prev = cur
    torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(imgs))
    return out, ms


# ---------------------------------------------------------------- evaluation

def load_recall(limit_scenes: int, every: int) -> tuple[list[bytes], list[str]]:
    """The labelled set: frames plus their scene name, in streaming order."""
    blobs, labels = [], []
    for d in sorted(glob.glob("/data/zx/stimuli/recall/scene_*"))[:limit_scenes]:
        name = pathlib.Path(d).name
        for p in sorted(glob.glob(d + "/*"))[::every]:
            blobs.append(open(p, "rb").read())
            labels.append(name)
    return blobs, labels


def detector_scores(sims: list[float], is_boundary: list[bool]) -> dict:
    """Best achievable separation, swept over every threshold the data allows.

    A metric is useful only if SOME threshold both catches the boundaries and drops
    the within-scene steps. Reporting one hand-picked threshold would hide a metric
    that cannot separate at all.
    """
    best = None
    for th in sorted(set(sims)):
        # retained when similarity < threshold
        tp = sum(1 for s, b in zip(sims, is_boundary) if b and s < th)
        fp = sum(1 for s, b in zip(sims, is_boundary) if not b and s < th)
        fn = sum(1 for s, b in zip(sims, is_boundary) if b and s >= th)
        nb = sum(is_boundary)
        rec = tp / nb if nb else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": th, "recall": rec, "precision": prec, "f1": f1,
                    "kept": tp + fp, "missed_boundaries": fn}
    # separation margin: worst boundary vs best non-boundary
    bs = [s for s, b in zip(sims, is_boundary) if b]
    ns = [s for s, b in zip(sims, is_boundary) if not b]
    best["boundary_sim_max"] = max(bs) if bs else None
    best["within_sim_min"] = min(ns) if ns else None
    best["separable"] = bool(bs and ns and max(bs) < min(ns))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=9)
    ap.add_argument("--every", type=int, default=4,
                    help="subsample frames within a scene to keep the run short")
    ap.add_argument("--out", default="/data/zx/results/evs_embed.json")
    a = ap.parse_args()

    print("loading the vision tower standalone (no thinker, no engine)...")
    tower, vcfg = load_vision_tower()

    blobs, labels = load_recall(a.scenes, a.every)
    print(f"\nlabelled set: {len(blobs)} frames over "
          f"{len(set(labels))} scenes (every {a.every}th frame)")
    is_boundary = [False] + [labels[i] != labels[i - 1] for i in range(1, len(labels))]
    print(f"  scene boundaries (informative by construction): {sum(is_boundary)}")
    print(f"  within-scene steps (redundant by construction): "
          f"{len(is_boundary) - 1 - sum(is_boundary)}")

    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
    print("\nencoding every frame through the model's own vision tower...")
    emb_metrics, ms_per_frame = embed_pairwise(tower, imgs)
    print(f"  {ms_per_frame:.1f} ms per frame measured "
          f"({imgs[0].size[0]}x{imgs[0].size[1]})")
    print(f"  at 2 fps that is {2*ms_per_frame:.0f} ms/s = "
          f"{0.2*ms_per_frame:.1f}% of one GPU")
    print(f"  NOTE this is one-frame-at-a-time. The in-engine probe measured 30.7 ms"
          f"/frame because it batches; a per-arrival filter cannot batch, so"
          f" {ms_per_frame:.0f} ms is the honest figure for this use.")

    g64 = [thumb_gray(b, 64) for b in blobs]
    g256 = [thumb_gray(b, 256) for b in blobs]

    metrics = {
        "shipped 64x64 gray MSE": [shipped_similarity(g64[i - 1], g64[i])
                                   for i in range(1, len(g64))],
        "256x256 gray MSE": [shipped_similarity(g256[i - 1], g256[i])
                             for i in range(1, len(g256))],
        "frac pixels changed (64)": [1.0 - frac_changed(g64[i - 1], g64[i])
                                     for i in range(1, len(g64))],
        "frac pixels changed (256)": [1.0 - frac_changed(g256[i - 1], g256[i])
                                      for i in range(1, len(g256))],
        "block-max MSE (64, 8x8)": [1.0 - blockmax_mse(g64[i - 1], g64[i])
                                    for i in range(1, len(g64))],
    }
    for k, v in emb_metrics.items():
        metrics["EMB " + k] = v
    bnd = is_boundary[1:]

    print("\n" + "=" * 100)
    print("CAN THE METRIC TELL A SCENE CHANGE FROM A DUPLICATE?")
    print("=" * 100)
    print(f"{'metric':<28}{'best F1':>9}{'recall':>9}{'precision':>11}"
          f"{'kept':>7}{'missed':>8}{'cleanly separable':>19}")
    res = {}
    for name, sims in metrics.items():
        d = detector_scores(sims, bnd)
        res[name] = d
        print(f"{name:<28}{d['f1']:>9.3f}{d['recall']:>9.0%}{d['precision']:>11.0%}"
              f"{d['kept']:>7}{d['missed_boundaries']:>8}"
              f"{('YES' if d['separable'] else 'no'):>19}")
    print("  'cleanly separable' = every scene boundary scores less similar than every")
    print("  within-scene step, i.e. a single threshold works with zero errors.")

    print("\n--- the separation margin, which is what makes a threshold portable ---")
    print(f"{'metric':<28}{'worst boundary':>16}{'closest duplicate':>19}{'gap':>10}")
    for name, d in res.items():
        bm, wm = d["boundary_sim_max"], d["within_sim_min"]
        gap = (wm - bm) if (bm is not None and wm is not None) else None
        print(f"{name:<28}{bm:>16.4f}{wm:>19.4f}"
              + (f"{gap:>10.4f}" if gap is not None else f"{'-':>10}"))
    print("  A positive gap means the two populations do not overlap at all. The bigger")
    print("  it is, the less the threshold has to be tuned per content type.")

    pathlib.Path(a.out).write_text(json.dumps(
        {"ms_per_frame": ms_per_frame, "n_frames": len(blobs),
         "n_boundaries": sum(bnd), "results": res}, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
