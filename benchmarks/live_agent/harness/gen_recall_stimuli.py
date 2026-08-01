#!/usr/bin/env python3
"""Build stimuli for the long-session recall test.

The latency stimuli (talking-head, handheld) show one continuous scene, so they
cannot test whether the model remembers anything: every turn looks the same, and
a model with no memory scores identically to one with perfect memory.

This makes a sequence of visually distinct, *machine-scorable* scenes -- one per
turn. Each scene is a solid background with a large uncommon word rendered in
it, plus a coloured shape as a second cue. Rendered words are used deliberately:
Qwen3-Omni reads text reliably, and an exact string match is an unambiguous
score, whereas "did it describe the couch correctly" is not.

Words are uncommon enough that a model cannot score by guessing the base rate of
English nouns, and semantically unrelated so recalling one gives no hint about
the others.

Also synthesizes the spoken prompts, because the workload asks by voice:
    q_describe.wav   "Describe what you can see in the camera right now."
    q_first.wav      "What word was shown on the very first screen you saw?"
    q_listall.wav    "List every word you have seen so far, in order."

The last one is the useful measurement: it yields a *graded* recall score (how
many of N words survive) instead of one pass/fail bit.

    gen_recall_stimuli.py --out /data/zx/stimuli/recall [--frames-per-scene 40]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (word, background RGB, shape name) -- shape gives a non-OCR cue so a scene is
# still identifiable if the text is missed.
SCENES: list[tuple[str, tuple[int, int, int], str]] = [
    ("ZEBRA",    (176,  38,  38), "circle"),
    ("PIANO",    ( 22,  74, 156), "square"),
    ("CACTUS",   ( 27, 122,  55), "triangle"),
    ("ANCHOR",   (128,  62, 150), "circle"),
    ("VIOLIN",   (196, 116,  16), "square"),
    ("ROCKET",   ( 20, 116, 130), "triangle"),
    ("WALNUT",   (140,  84,  36), "circle"),
    ("MAGNET",   ( 62,  62,  70), "square"),
    ("PYRAMID",  (168,  32, 108), "triangle"),
    ("TROMBONE", ( 46,  90,  30), "circle"),
]

# A small two-digit number in the bottom-right corner, drawn at a fraction of the
# word's size. This is the "detail the text note cannot carry": the note is one
# sentence naming the salient objects, so a tiny corner number is exactly the kind
# of thing it will not mention. A policy that keeps the pixels can still answer
# about it; a policy that keeps only the note cannot. That contrast is the point.
DETAILS = ["47", "83", "12", "65", "39", "74", "26", "58", "91", "33"]

PROMPTS = {
    "q_describe": "Describe what you can see in the camera right now.",
    "q_first": "What word was shown on the very first screen you saw?",
    "q_listall": "List every word you have seen so far, in order.",
    "q_shape": "What shape do you see on the left side?",
    "q_detail": "What small number was in the bottom right corner of the very first screen?",
}

W, H = 1280, 720


def draw_shape(d: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    fill = (245, 245, 245)
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    else:  # triangle
        d.polygon([(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=fill)


def make_scene(word: str, bg: tuple[int, int, int], shape: str, jitter: int,
               detail: str | None = None) -> Image.Image:
    """One frame of a scene. `jitter` shifts content a few px between frames.

    The jitter matters: the EVS frame filter drops a frame when it is >=95%
    similar to the last retained one, so byte-identical frames would collapse to
    a single retained frame per scene and the buffer behaviour would stop
    resembling a real camera. A few px of movement keeps frames distinct enough
    to be admitted while leaving the scene semantically identical.
    """
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    dx = (jitter % 7) - 3
    dy = ((jitter // 7) % 7) - 3
    draw_shape(d, shape, (120 + dx, 200 + dy, 380 + dx, 460 + dy))
    font = ImageFont.truetype(FONT, 150)
    bbox = d.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((W - tw) // 2 + 120 + dx, (H - th) // 2 + dy), word,
           font=font, fill=(250, 250, 250))
    small = ImageFont.truetype(FONT, 40)
    d.text((40, 40), f"scene: {word}", font=small, fill=(235, 235, 235))
    if detail:
        # Bottom-right, ~1/3 the caption size and ~1/5 the word size. Legible to
        # the vision encoder, but not something a one-sentence scene description
        # would bother to include.
        tiny = ImageFont.truetype(FONT, 34)
        db = d.textbbox((0, 0), detail, font=tiny)
        d.text((W - (db[2] - db[0]) - 48 + dx, H - (db[3] - db[1]) - 56 + dy),
               detail, font=tiny, fill=(240, 240, 240))
    return img


def synth(text: str, out: pathlib.Path, piper: str, voice: str) -> bool:
    """piper -> 16 kHz mono PCM16 wav (native rate; no resampling needed)."""
    try:
        proc = subprocess.run(
            [piper, "--model", voice, "--output_file", str(out)],
            input=text.encode(), capture_output=True, timeout=180,
        )
    except FileNotFoundError:
        print(f"  piper not found at {piper}", file=sys.stderr)
        return False
    if proc.returncode != 0 or not out.exists():
        print(f"  piper failed for {out.name}: {proc.stderr.decode()[:300]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/zx/stimuli/recall")
    ap.add_argument("--frames-per-scene", type=int, default=40)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--piper", default="/home/zx/miniconda3/envs/pa0/bin/piper")
    ap.add_argument("--voice", default="/data/zx/stimuli/en_US-amy-low.onnx")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--detail", action="store_true",
                    help="draw a small two-digit number in the bottom-right corner: "
                         "a visual detail a one-sentence note will not carry")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== scenes -> {out} ({args.frames_per_scene} frames each"
          f"{', with corner detail' if args.detail else ''}) ===")
    for i, (word, bg, shape) in enumerate(SCENES):
        d = out / f"scene_{i:02d}_{word.lower()}"
        d.mkdir(parents=True, exist_ok=True)
        det = DETAILS[i] if args.detail else None
        for j in range(args.frames_per_scene):
            make_scene(word, bg, shape, j, det).save(
                d / f"f{j:05d}.jpg", quality=args.quality
            )
        print(f"  scene_{i:02d} {word:9} {shape:8} detail={det or '-':>3} -> {d.name}")

    print(f"\n=== spoken prompts -> {out} ===")
    ok = True
    for name, text in PROMPTS.items():
        p = out / f"{name}.wav"
        if synth(text, p, args.piper, args.voice):
            import wave
            with wave.open(str(p)) as w:
                assert w.getnchannels() == 1 and w.getframerate() == 16000, \
                    f"{p}: expected mono/16k, got {w.getnchannels()}ch/{w.getframerate()}Hz"
                print(f"  {name:12} {w.getnframes()/w.getframerate():5.2f}s  \"{text}\"")
        else:
            ok = False

    # A manifest so the bench and the report agree on ground truth. Fourth column
    # is the corner detail, or "-" when this set was generated without it, so a
    # bench reading an older manifest still parses.
    (out / "manifest.txt").write_text(
        "\n".join(
            f"{i}\t{w}\t{s}\t{DETAILS[i] if args.detail else '-'}"
            for i, (w, _, s) in enumerate(SCENES)
        ) + "\n"
    )
    print(f"\n  manifest -> {out / 'manifest.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
