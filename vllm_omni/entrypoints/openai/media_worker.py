# SPDX-License-Identifier: Apache-2.0
"""[live-vllm CPU-plane] Standalone media worker.

Launched BY FILE PATH (never imported, never via multiprocessing spawn) so it
imports only PIL/numpy/stdlib -- no vllm, no torch, no CUDA, ~300 ms startup.
Protocol: length-prefixed pickle over stdin/stdout, one request at a time.

Request:  (raw_jpeg: bytes, max_w: int, max_h: int, quality: int, thumb: int)
Response: ("ok", shrunk_jpeg|None, thumb_bytes, (w, h), rgb_bytes, md5_hex)
        | ("err", repr(exception))
"""
import hashlib
import io
import pickle
import struct
import sys

import numpy as np
from PIL import Image


def _process(raw_bytes, max_width, max_height, jpeg_quality, thumb_size):
    # Downscale semantics = video_stream_base._downscale_frame_bytes:
    # fit-within, aspect preserved, BILINEAR, never upscale, None if it fits.
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    shrunk = None
    if max_width and max_height and not (w <= max_width and h <= max_height):
        scale = min(max_width / float(w), max_height / float(h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        shrunk = buf.getvalue()
    final_jpeg = shrunk if shrunk is not None else raw_bytes
    thumb = np.asarray(
        img.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    md5 = hashlib.md5(final_jpeg, usedforsecurity=False).hexdigest()
    return shrunk, thumb.tobytes(), img.size, img.tobytes(), md5


def _read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        head = _read_exact(stdin, 4)
        if head is None:
            return  # parent closed the pipe: exit quietly
        (n,) = struct.unpack("<I", head)
        body = _read_exact(stdin, n)
        if body is None:
            return
        try:
            args = pickle.loads(body)
            resp = ("ok", *_process(*args))
        except Exception as e:  # report, never die on a bad frame
            resp = ("err", repr(e))
        blob = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        stdout.write(struct.pack("<I", len(blob)))
        stdout.write(blob)
        stdout.flush()


if __name__ == "__main__":
    main()
