#!/usr/bin/env python3
"""How much is lost by choosing frames blindly?

The shipped pipeline decides which frames the model ever sees using two
semantics-blind rules composed in sequence:

  1. EVS pixel gate      drop a frame if 1 - MSE/255^2 >= threshold, measured
                         on a 64x64 thumbnail against the last retained frame
  2. uniform stride      subsample whatever survived down to `num_frames`

Both are unaware of image content. This script asks the question that matters
for a perception budget: given that the model will only ever be shown K
frames out of N, how good is the shipped choice of K compared to the best
achievable choice of K?

Method
------
Embed every frame with a frozen image encoder (CLIP ViT-B/32 by default), then
score a selection S of size K by its *coverage error*:

    coverage_error(S) = mean over all frames f of  min_{s in S} d(f, s)

with d = cosine distance in embedding space. This is the k-center / k-medoid
objective: it measures how well the K retained frames represent the whole
stream. Lower is better. It is a proxy for semantic information retained, not
a task accuracy -- stated plainly because that distinction is the difference
between this being evidence and being overclaiming.

Selections compared at equal budget K:
    uniform         uniform stride over all N frames (what the handler does
                    once EVS has passed frames through)
    evs_then_uniform  the shipped composition: EVS gate, then uniform stride
    pixel_greedy    greedy k-center using pixel-thumbnail distance
    embed_greedy    greedy k-center in embedding space  <- the informed choice
    random          mean over several random draws (a floor)

The gap between `evs_then_uniform` and `embed_greedy` is the headroom a
content-aware frame allocator could capture at identical cost.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image

THUMB = 64
MAXVAL = 255.0


# --------------------------------------------------------------------------
def evs_keep_indices(thumbs: np.ndarray, threshold: float) -> list[int]:
    """Indices retained by the shipped EVS gate."""
    keep: list[int] = []
    last = None
    cut = (1.0 - threshold) * MAXVAL * MAXVAL
    for i, cur in enumerate(thumbs):
        c = cur.astype(np.float32)
        if last is None:
            keep.append(i)
            last = c
            continue
        if float(np.mean((last - c) ** 2)) > cut:
            keep.append(i)
            last = c
    return keep


def uniform_pick(pool: list[int], k: int) -> list[int]:
    """Exactly the handler's stride rule, applied to a candidate pool."""
    n = len(pool)
    if n <= k:
        return list(pool)
    stride = max(1, n // k)
    idx = [i * stride for i in range(k - 1)] + [n - 1]
    return [pool[i] for i in idx if i < n]


def greedy_kcenter(D: np.ndarray, k: int, seed_idx: int = 0) -> list[int]:
    """Farthest-point traversal: near-optimal for the k-center objective."""
    n = D.shape[0]
    k = min(k, n)
    sel = [seed_idx]
    mind = D[seed_idx].copy()
    while len(sel) < k:
        nxt = int(np.argmax(mind))
        if nxt in sel:
            break
        sel.append(nxt)
        mind = np.minimum(mind, D[nxt])
    return sorted(sel)


def greedy_kmedoid(D: np.ndarray, k: int) -> list[int]:
    """Greedy facility location: repeatedly add the frame that most reduces
    mean-min-distance.

    This is the oracle that MATCHES the coverage_error objective below. Greedy
    k-center (farthest-point traversal) optimizes the worst case instead and
    deliberately selects outliers, which makes it a poor -- and misleading --
    upper bound for a mean-based score. Greedy facility location is
    (1 - 1/e)-optimal for this objective, so it is a legitimate near-oracle.
    """
    n = D.shape[0]
    k = min(k, n)
    sel: list[int] = []
    mind = np.full(n, np.inf)
    for _ in range(k):
        # cost of adding each candidate j: mean of elementwise min(mind, D[j])
        cand = np.minimum(mind[None, :], D)          # (n_candidates, n_points)
        costs = cand.mean(axis=1)
        j = int(np.argmin(costs))
        if sel and costs[j] >= mind.mean() - 1e-12:
            break                                    # no further improvement
        sel.append(j)
        mind = cand[j]
    return sorted(sel)


def coverage_error(D: np.ndarray, sel: list[int]) -> float:
    if not sel:
        return float("nan")
    return float(D[:, sel].min(axis=1).mean())


# --------------------------------------------------------------------------
def embed_frames(paths: list[pathlib.Path], model_id: str, device: str,
                 batch: int) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
            f = None
            if hasattr(model, "get_image_features"):
                f = model.get_image_features(pixel_values=px)
            if f is None:
                f = model(pixel_values=px)
            # transformers versions differ on whether these return a bare
            # tensor or a ModelOutput; accept either.
            if not torch.is_tensor(f):
                for attr in ("image_embeds", "pooler_output"):
                    cand = getattr(f, attr, None)
                    if torch.is_tensor(cand):
                        f = cand
                        break
                else:
                    f = f.last_hidden_state.mean(dim=1)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def thumbs_of(paths: list[pathlib.Path]) -> np.ndarray:
    out = []
    for p in paths:
        img = Image.open(p).resize((THUMB, THUMB), Image.Resampling.BILINEAR).convert("RGB")
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", default="/data/zx/stimuli/frames")
    ap.add_argument("--out", default="/data/zx/results/frame_selection_headroom.json")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budgets", default="4,8,16,32")
    ap.add_argument("--evs-threshold", type=float, default=0.95)
    ap.add_argument("--max-frames-buffer", type=int, default=64,
                    help="handler buffer cap applied before uniform stride")
    ap.add_argument("--random-trials", type=int, default=20)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    root = pathlib.Path(args.stimuli)
    regimes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = np.random.default_rng(0)

    results: dict = {
        "meta": {
            "encoder": args.model,
            "metric": "coverage_error = mean_f min_{s in S} cosine_distance(f, s)",
            "note": "proxy for retained semantic information, NOT task accuracy",
            "evs_threshold": args.evs_threshold,
            "handler_buffer_cap": args.max_frames_buffer,
        },
        "regimes": {},
    }

    for r in regimes:
        paths = sorted((root / r).glob("*.jpg"))
        if len(paths) < max(budgets) + 2:
            continue
        emb = embed_frames(paths, args.model, args.device, args.batch)
        D = 1.0 - emb @ emb.T
        np.fill_diagonal(D, 0.0)
        th = thumbs_of(paths)
        flat = th.reshape(len(th), -1).astype(np.float32)
        # pixel distance matrix, same units the EVS gate uses (MSE on thumbnails)
        sq = (flat ** 2).sum(1)
        Dpix = np.maximum(sq[:, None] + sq[None, :] - 2 * flat @ flat.T, 0.0) / flat.shape[1]

        kept = evs_keep_indices(th, args.evs_threshold)
        n = len(paths)
        entry: dict = {
            "n_frames": n,
            "evs_kept": len(kept),
            "evs_drop_rate": 1.0 - len(kept) / n,
            "budgets": {},
        }

        for k in budgets:
            # shipped path: EVS gate -> buffer cap (keep most recent) -> uniform stride
            pool = kept[-args.max_frames_buffer:] if len(kept) > args.max_frames_buffer else kept
            sel_shipped = uniform_pick(pool, k)
            sel_uniform = uniform_pick(list(range(n)), k)
            sel_pix = greedy_kcenter(Dpix, k)
            sel_emb = greedy_kcenter(D, k)
            # matched oracle for the mean-based coverage objective
            sel_emb_med = greedy_kmedoid(D, k)
            sel_pix_med = greedy_kmedoid(Dpix, k)
            rand_errs = [
                coverage_error(D, sorted(rng.choice(n, size=min(k, n), replace=False).tolist()))
                for _ in range(args.random_trials)
            ]

            entry["budgets"][str(k)] = {
                "shipped_evs_then_uniform": {
                    "n_selected": len(sel_shipped),
                    "coverage_error": coverage_error(D, sel_shipped),
                },
                "uniform_no_evs": {
                    "n_selected": len(sel_uniform),
                    "coverage_error": coverage_error(D, sel_uniform),
                },
                "pixel_greedy_kcenter": {
                    "n_selected": len(sel_pix),
                    "coverage_error": coverage_error(D, sel_pix),
                },
                "embed_greedy_kcenter": {
                    "n_selected": len(sel_emb),
                    "coverage_error": coverage_error(D, sel_emb),
                },
                "embed_greedy_kmedoid": {
                    "n_selected": len(sel_emb_med),
                    "coverage_error": coverage_error(D, sel_emb_med),
                },
                "pixel_greedy_kmedoid": {
                    "n_selected": len(sel_pix_med),
                    "coverage_error": coverage_error(D, sel_pix_med),
                },
                "random_mean": float(np.mean(rand_errs)),
            }
        results["regimes"][r] = entry

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))

    print(f"\ncoverage error (lower = the K shown frames represent the stream better)")
    print(f"encoder: {args.model};  EVS threshold {args.evs_threshold};  "
          f"buffer cap {args.max_frames_buffer}\n")
    print(f"{'regime':<20}{'K':>4}{'shipped':>10}{'unif-noEVS':>12}{'pix-med':>10}"
          f"{'emb-med':>10}{'random':>9}{'gate gain':>11}{'sel gain':>10}")
    for r, e in results["regimes"].items():
        for k, b in e["budgets"].items():
            sh = b["shipped_evs_then_uniform"]["coverage_error"]
            un = b["uniform_no_evs"]["coverage_error"]
            pm = b["pixel_greedy_kmedoid"]["coverage_error"]
            em = b["embed_greedy_kmedoid"]["coverage_error"]
            rd = b["random_mean"]
            # decompose the total gap into the two mechanisms
            gate = (sh / un) if un else float("nan")   # what the EVS gate costs
            sel = (un / em) if em else float("nan")    # what smarter selection adds
            print(f"{r:<20}{k:>4}{sh:>10.4f}{un:>12.4f}{pm:>10.4f}"
                  f"{em:>10.4f}{rd:>9.4f}{gate:>10.2f}x{sel:>9.2f}x")
    print(f"\ngate gain = shipped / uniform-no-EVS : cost of the EVS pixel gate alone")
    print(f"sel gain  = uniform-no-EVS / emb-medoid : additional gain from content-aware")
    print(f"            selection, once the gate is out of the way.")
    print(f"Both at IDENTICAL frame budget K, so both are free in GPU terms.")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
