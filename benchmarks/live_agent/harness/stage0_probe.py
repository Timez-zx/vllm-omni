#!/usr/bin/env python3
"""In-process GPU-time attribution inside vllm-omni stage 0 (the thinker).

Why this exists
---------------
NVML per-PID accounting can split GPU time between vllm-omni's stages because
each stage is its own OS process. It cannot look *inside* stage 0, which bundles
three very different jobs into one process:

    self.visual          vision encoder  (perception)
    self.audio_tower     audio encoder   (perception)
    self.language_model  the LLM         (thinking / text generation)

Reporting "stage 0 = 16-23% of GPU time" therefore only bounds perception from
above. This probe closes that gap by timing the three submodules directly.

Method
------
`register_forward_pre_hook` / `register_forward_hook` on each of the three
modules, with a pair of `torch.cuda.Event`s recorded on the current stream.
Because the hooks record events rather than reading them, no synchronization is
forced on the hot path: the event pairs are queued and drained later, and only
pairs whose end event has already completed (`Event.query()`) are reduced. A
wall-clock timer is recorded alongside so that GPU-busy time can be expressed
as a duty cycle rather than only as a share.

For the LLM the token count of each forward is recorded too, so prefill-shaped
steps (many tokens) can be separated from decode-shaped steps (roughly one
token per running sequence). vLLM v1 can mix both in one step, so this is a
histogram over forward sizes, not a clean partition -- reported as such.

Caveats, by construction:
  * Times are per-module GPU intervals on one stream. If work were overlapped
    across streams these would double-count; vLLM's model forward is
    single-stream, so on this path they do not.
  * Hook overhead is two event records per forward (~microseconds), but event
    records do serialize with respect to the stream, so measured totals are a
    slight over-estimate of the un-instrumented cost.
  * CUDA graph replay hides internal structure. Qwen3-Omni stage 0 runs with
    cudagraph enabled for the LLM backbone; a graph replay is timed as one
    opaque interval attributed to `language_model`, which is what we want, but
    it means the split *within* the LLM is not visible here.

Enabled only when PA_STAGE0_PROBE is set, so the default engine path is
untouched.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time

import torch

_OUT = os.environ.get("PA_STAGE0_PROBE_OUT", "/data/zx/results/stage0_probe.json")
# Per-call event log. Written with absolute wall-clock timestamps (time.time())
# so that records from this server process can be aligned against the client
# trace and the NVML sampler, which live in other processes. monotonic clocks
# are not comparable across processes; wall clock on one host is.
_EVENTS = os.environ.get("PA_STAGE0_PROBE_EVENTS", "")
_DUMP_EVERY_S = float(os.environ.get("PA_STAGE0_PROBE_DUMP_S", "2.0"))
_MAX_PENDING = 4096

_evf = None
if _EVENTS:
    try:
        _evf = open(_EVENTS, "a", buffering=1)  # line buffered
    except OSError:
        _evf = None

_lock = threading.Lock()
_t0 = time.monotonic()

# name -> accumulated ms, count, token histogram
_acc: dict[str, dict] = {}
_pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int]] = []
_installed = False


def _bucket(n: int) -> str:
    if n <= 0:
        return "0"
    for hi in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        if n <= hi:
            return str(hi)
    return ">16384"


def _entry(name: str) -> dict:
    return _acc.setdefault(
        name, {"gpu_ms": 0.0, "calls": 0, "tokens": 0, "by_size": {}}
    )


def _drain(force: bool = False) -> None:
    """Reduce completed event pairs. Never blocks unless force=True."""
    global _pending
    keep = []
    with _lock:
        pend, _pending = _pending, []
    for rec in pend:
        name, ev0, ev1, ntok, w0, w1 = rec
        if not force and not ev1.query():
            keep.append(rec)
            continue
        try:
            ms = ev0.elapsed_time(ev1)
        except Exception:
            continue
        e = _entry(name)
        e["gpu_ms"] += ms
        e["calls"] += 1
        e["tokens"] += ntok
        b = _bucket(ntok)
        s = e["by_size"].setdefault(b, {"gpu_ms": 0.0, "calls": 0})
        s["gpu_ms"] += ms
        s["calls"] += 1
        if _evf is not None:
            # w0/w1 bracket the Python-side launch of this module. GPU execution
            # happens at or after w0; at the 100 ms-1 s scale being studied the
            # launch time is an adequate timeline anchor, and gpu_ms is exact.
            _evf.write(json.dumps({
                "module": name, "launch_start": w0, "launch_end": w1,
                "gpu_ms": ms, "ntok": ntok, "pid": os.getpid(),
            }) + "\n")
    if keep:
        with _lock:
            _pending.extend(keep)


def _ntokens(args, kwargs) -> int:
    """Best-effort leading-dimension size of the first tensor argument."""
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
        if isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, torch.Tensor) and x.dim() >= 1:
                    return int(x.shape[0])
    return 0


def _attach(module: torch.nn.Module, name: str) -> None:
    state: dict = {}

    def pre(mod, args, kwargs):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        state[threading.get_ident()] = (ev0, _ntokens(args, kwargs), time.time())
        return None

    def post(mod, args, kwargs, output):
        got = state.pop(threading.get_ident(), None)
        if got is None:
            return output
        ev0, ntok, w0 = got
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        with _lock:
            if len(_pending) < _MAX_PENDING:
                _pending.append((name, ev0, ev1, ntok, w0, time.time()))
        return output

    module.register_forward_pre_hook(pre, with_kwargs=True)
    module.register_forward_hook(post, with_kwargs=True)


def dump() -> None:
    _drain()
    now = time.monotonic()
    wall = now - _t0
    with _lock:
        snap = {k: {kk: (dict(vv) if isinstance(vv, dict) else vv)
                    for kk, vv in v.items()} for k, v in _acc.items()}
    total = sum(v["gpu_ms"] for v in snap.values()) or 1.0
    out = {
        "pid": os.getpid(),
        "wall_s": wall,
        "modules": {},
        "note": "gpu_ms are per-module CUDA event intervals on the model stream; "
                "share_of_instrumented is the split WITHIN stage 0 only",
    }
    for k, v in sorted(snap.items(), key=lambda kv: -kv[1]["gpu_ms"]):
        out["modules"][k] = {
            **v,
            "share_of_instrumented": v["gpu_ms"] / total,
            "duty_cycle": v["gpu_ms"] / (wall * 1000.0) if wall > 0 else None,
        }
    out["instrumented_total_gpu_ms"] = total
    out["instrumented_duty_cycle"] = total / (wall * 1000.0) if wall > 0 else None
    tmp = _OUT + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, _OUT)
    except OSError:
        pass


def _dumper() -> None:
    while True:
        time.sleep(_DUMP_EVERY_S)
        try:
            dump()
        except Exception:
            pass


def install(model) -> None:
    """Attach hooks to the thinker's three top-level submodules."""
    global _installed
    if _installed or not os.environ.get("PA_STAGE0_PROBE"):
        return
    lm = getattr(model, "language_model", None)
    targets = (
        ("vision_encoder", getattr(model, "visual", None)),
        ("audio_encoder", getattr(model, "audio_tower", None)),
        # The thinker calls self.language_model.model(...) -- the inner
        # Qwen3MoeLLMModel backbone -- never the outer ForCausalLM wrapper, so
        # a hook on the wrapper never fires. Hook the backbone.
        ("llm_backbone", getattr(lm, "model", None) if lm is not None else None),
        # kept for completeness; expected to stay at zero calls
        ("llm_wrapper", lm),
    )
    attached = []
    for name, mod in targets:
        if isinstance(mod, torch.nn.Module):
            _attach(mod, name)
            attached.append(name)
    if not attached:
        return
    _installed = True
    threading.Thread(target=_dumper, daemon=True).start()
    atexit.register(lambda: (_drain(force=True), dump()))
    try:
        import logging

        logging.getLogger(__name__).warning(
            "[stage0_probe] instrumented: %s -> %s", ", ".join(attached), _OUT
        )
    except Exception:
        pass
