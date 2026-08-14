# SPDX-License-Identifier: Apache-2.0
"""[live-vllm CPU-plane] Subprocess frame pipeline (pool manager).

The serving process's event loop must stay clear for the DELIVERY path
(websocket audio sends): measured at 56 users, inline PIL work (decode x3 per
frame: downscale, similarity filter, prewarm) plus its GIL time stretched the
chunk-delivery tail past the lead buffer (gap p99 377->569 ms from u36 to u56
while p50 held at 319 ms and both GPUs sat at ~40%).

This pool does ONE decode per frame, in a worker SUBPROCESS -- outside this
process's GIL -- and returns everything every downstream consumer needs: the
downscaled JPEG, the similarity-filter thumbnail, the final image's raw RGB
(rebuilt into PIL via frombytes: a memcpy, not a decode) and the mm-cache md5.

Why hand-rolled subprocesses instead of ProcessPoolExecutor: any
multiprocessing spawn re-imports the parent's ``__main__`` -- for the
vllm-omni server that means re-importing torch/vllm per worker (tens of
seconds, GBs of RSS). The workers here are launched BY FILE PATH
(media_worker.py, PIL/numpy/stdlib only, ~300 ms startup) over a
length-prefixed pickle pipe. Fork is also out: this process holds CUDA-adjacent
threads and logging locks.

Sizing: VLLM_OMNI_MEDIA_WORKERS (default ``min(8, ncpu)``; 0 disables and the
legacy in-process path takes over). 8 workers sustain ~2000 frames/s.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

_THUMBNAIL_SIZE = 64  # lockstep with video_frame_filter._DEFAULT_THUMBNAIL_SIZE
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media_worker.py")


def _workers_configured() -> int:
    raw = os.environ.get("VLLM_OMNI_MEDIA_WORKERS", "")
    if raw.strip() == "":
        return min(8, os.cpu_count() or 8)
    try:
        return max(0, int(raw))
    except ValueError:
        return min(8, os.cpu_count() or 8)


_N_WORKERS = _workers_configured()


def enabled() -> bool:
    return _N_WORKERS > 0


@dataclass
class FrameResult:
    shrunk_jpeg: bytes | None  # None = original already fits (keep its bytes)
    thumb: np.ndarray          # 64x64x3 uint8 of the FINAL (post-downscale) image
    size: tuple[int, int]      # final image (w, h)
    rgb: bytes                 # final image raw RGB for Image.frombytes
    md5: str                   # md5 of the FINAL jpeg bytes (mm-cache uuid)


class _Worker:
    """One subprocess; one in-flight request at a time (pool serializes)."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, _WORKER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr inherited: a crashing worker's traceback lands in our log.
        )

    def request(self, payload: tuple) -> tuple:
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(struct.pack("<I", len(blob)))
        self.proc.stdin.write(blob)
        self.proc.stdin.flush()
        head = self.proc.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError("media worker died")
        (n,) = struct.unpack("<I", head)
        body = self.proc.stdout.read(n)
        if len(body) < n:
            raise RuntimeError("media worker died mid-response")
        return pickle.loads(body)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


_pool_lock = threading.Lock()
_spawned: list[_Worker] = []   # built by any thread (prewarm), no loop needed
_free: asyncio.Queue[_Worker] | None = None  # built ON the serving loop


def _spawn_workers() -> None:
    with _pool_lock:
        while len(_spawned) < _N_WORKERS:
            _spawned.append(_Worker())


def _ensure_pool() -> asyncio.Queue[_Worker]:
    """Bind the pool to the running loop on the first frame; the subprocesses
    themselves may have been prewarmed from any thread."""
    global _free
    if _free is None:
        _spawn_workers()
        _free = asyncio.Queue()
        with _pool_lock:
            for w in _spawned:
                _free.put_nowait(w)
        logger.info("[media-pipeline] %d subprocess workers online (%s)",
                    _N_WORKERS, _WORKER_PATH)
    return _free


async def process_frame(
    raw_bytes: bytes,
    max_width: int,
    max_height: int,
    jpeg_quality: int,
) -> FrameResult:
    free = _ensure_pool()
    worker = await free.get()
    try:
        if not worker.alive():
            worker = _Worker()  # transparent respawn
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, worker.request,
            (raw_bytes, max_width, max_height, jpeg_quality, _THUMBNAIL_SIZE),
        )
    except Exception:
        worker.close()
        worker = _Worker()  # never return a dead worker to the pool
        raise
    finally:
        free.put_nowait(worker)
    if resp[0] != "ok":
        raise ValueError(f"media worker: {resp[1]}")
    _, shrunk, thumb_bytes, size, rgb, md5 = resp
    thumb = np.frombuffer(thumb_bytes, dtype=np.uint8).reshape(
        _THUMBNAIL_SIZE, _THUMBNAIL_SIZE, 3)
    return FrameResult(shrunk_jpeg=shrunk, thumb=thumb, size=size, rgb=rgb, md5=md5)


def prewarm_pool() -> None:
    """Start the worker subprocesses ahead of the first frame (any thread)."""
    if enabled():
        _spawn_workers()
        logger.info("[media-pipeline] %d subprocess workers prewarmed", _N_WORKERS)
