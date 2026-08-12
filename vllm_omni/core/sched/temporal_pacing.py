# SPDX-License-Identifier: Apache-2.0
"""Temporal pacing: hold ahead-of-realtime requests out of scheduler passes.

The temporal-batching experiment (benchmarks/temporal_batching/DESIGN.zh.md).
Audio has a natural 12.5 Hz rhythm -- one talker decode step samples one codec
frame, and one codec frame is 1920 samples @ 24 kHz = 80 ms of playback -- so a
session that has already generated more audio than playback needs (plus a lead
margin) gains nothing from generating NOW; it only contends with everyone
else's steps. The thinker obeys the SAME global clock with a token/s cap: its
per-tick work is leveled instead of bursty, while staying far above the
talker's text consumption rate so it can never starve the audio.

The pacer parks ahead-of-schedule requests for the current scheduler pass and
computes their release on a GLOBAL tick grid (all requests in the stage share
the grid), so every session's next unit becomes schedulable at the same
instant and upstream continuous batching naturally forms one large periodic
batch per tick.

Mechanics copy the chunk adapter's ``_held_non_active`` precedent: requests
are removed from ``self.running`` immediately before ``super().schedule()``
and put back before ``schedule()`` returns -- no status change, no counter
updates; ``has_requests()`` never sees the park, so the engine busy loop keeps
ticking. A request parked this pass is simply re-examined next pass.

Everything is read from env ONCE per engine-core process:

  VLLM_OMNI_TEMPORAL_TICK_MS          0 = off (default, greedy baseline); 80 / 160.
  VLLM_OMNI_TEMPORAL_LEAD_MS          allowed lead over realtime playback, default 240.
  VLLM_OMNI_TEMPORAL_INITIAL_FRAMES   talker per-turn exempt burst, default 4
                                      (= initial_codec_chunk_frames: the first
                                      audible chunk is never paced -- TTFA does
                                      not pay for pacing).
  VLLM_OMNI_TEMPORAL_THINKER_TPS      thinker rate cap in tokens/s, default 25
                                      (~5x speech consumption; 0 = thinker unpaced).
  VLLM_OMNI_TEMPORAL_THINKER_BURST    thinker per-turn exempt burst, default 16.
  VLLM_OMNI_TEMPORAL_NO_QUANT         1 = rate-limit only, no grid alignment
                                      (the ablation separating "slower" from
                                      "together").
  VLLM_OMNI_TEMPORAL_BARRIER          1 = stage-wide TICK BARRIER instead of
                                      per-request release times: at each tick
                                      boundary every active request gets its
                                      per-tick budget (rate*tick units); a
                                      request that used its budget is held to
                                      the NEXT boundary -- including requests
                                      that are BEHIND schedule (they get a
                                      catch-up multiplier, not a free run).
                                      This is what actually consolidates
                                      batches: the v1 per-request gate let any
                                      delayed request fall off the grid and
                                      free-run (release<=now passed it through
                                      every pass), measured as batch p50=2
                                      under pacing vs 7 under greedy at 16
                                      users. Burst exemption still bypasses
                                      the barrier so TTFA is untouched.
  VLLM_OMNI_TEMPORAL_CATCHUP          barrier mode: budget multiplier for
                                      requests behind schedule (default 2).
  VLLM_OMNI_LOG_SCHED_STEPS           log one [SCHED-STEP] line per non-idle
                                      scheduler pass ('' = off; '1'/'all' or a
                                      comma list of stage ids).

WARNING: parking + re-admitting a live request under async scheduling is the
documented -1-sentinel crash path (see deploy_web_demo.yaml's stage-1 note).
Paced stages must run with ``async_scheduling: false``; the experiment deploy
config (deploy_temporal_2gpu.yaml) sets it on both AR stages, baseline
included, so all arms compare like with like.
"""
from __future__ import annotations

import math
import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Qwen3 codec: valid codebook-0 ids are [0, 2048); pad/bos/eos live at 4196+
# and produce no audio frame, so they must not advance the frame count.
_CODEC_VALID_MAX = 2048

# [WP1] VLLM_OMNI_TEMPORAL_ENGINE=1 turns the engine-core busy loop into a
# tick loop: between pacing events it SLEEPS on the input queue (waking
# instantly for client requests) instead of hot-spinning through empty
# scheduler passes. Read once per engine-core process.
TICK_ENGINE_LOOP = (
    os.environ.get("VLLM_OMNI_TEMPORAL_ENGINE", "0") not in ("0", "", "false", "False")
    and float(os.environ.get("VLLM_OMNI_TEMPORAL_TICK_MS", "0") or 0) > 0
)
TICK_S = float(os.environ.get("VLLM_OMNI_TEMPORAL_TICK_MS", "0") or 0) / 1000.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _parse_log_steps(stage_id: Any) -> bool:
    raw = os.environ.get("VLLM_OMNI_LOG_SCHED_STEPS", "").strip()
    if not raw or raw in ("0", "false", "False"):
        return False
    if raw in ("1", "all"):
        return True
    return str(stage_id) in {s.strip() for s in raw.split(",")}


class TemporalPacer:
    """Per-stage pacing gate. Constructed by OmniARScheduler.__init__.

    ``self.enabled`` is False unless the env selects a tick AND this stage has
    a defined rate (thinker/talker); everything else in the scheduler stays on
    the exact baseline code path.
    """

    def __init__(self, model_config: Any):
        tick_ms = _env_float("VLLM_OMNI_TEMPORAL_TICK_MS", 0.0)
        self.tick_s = max(0.0, tick_ms) / 1000.0
        self.no_quant = os.environ.get("VLLM_OMNI_TEMPORAL_NO_QUANT", "0") not in ("0", "", "false", "False")
        self.lead_s = _env_float("VLLM_OMNI_TEMPORAL_LEAD_MS", 240.0) / 1000.0
        self.stage_id = getattr(model_config, "stage_id", -1)
        stage = str(getattr(model_config, "model_stage", "") or "")

        if "talker" in stage:
            # 1 sampled token = 1 codec frame = 80 ms: rate is the playback rate.
            self.rate = 12.5
            self.burst = int(_env_float("VLLM_OMNI_TEMPORAL_INITIAL_FRAMES", 4.0))
            self.codec_only = True
        elif "thinker" in stage:
            self.rate = _env_float("VLLM_OMNI_TEMPORAL_THINKER_TPS", 25.0)
            self.burst = int(_env_float("VLLM_OMNI_TEMPORAL_THINKER_BURST", 16.0))
            self.codec_only = False
        else:
            self.rate = 0.0
            self.burst = 0
            self.codec_only = False

        self.barrier = os.environ.get("VLLM_OMNI_TEMPORAL_BARRIER", "0") not in ("0", "", "false", "False")
        self.catchup = max(1.0, _env_float("VLLM_OMNI_TEMPORAL_CATCHUP", 2.0))
        # Per-tick budget in output units (thinker: tokens, talker: frames).
        self.tick_budget = max(1, round(self.rate * self.tick_s)) if self.rate > 0 else 0
        # Intra-tick SUB-SLOT grid: the k-th unit of a window releases at
        # window_open + k*sub_slot, so catch-up units also step TOGETHER
        # instead of trickling in whenever each request's data lands. Without
        # this, round 1 is one big batch but every later unit runs solo at
        # its own chunk-arrival moment (measured: 2481 nreq=1 steps, frame
        # efficiency 3.9 vs 6.1). Slots cover the worst-case catchup budget.
        max_units = max(1, int(math.ceil(self.tick_budget * self.catchup))) if self.tick_budget else 1
        self.sub_slot = self.tick_s / max_units if self.tick_s > 0 else 0.0
        # Barrier state: the boundary that opened the CURRENT tick window, and
        # each request's unit count snapshotted at that boundary.
        self._tick_open: float = 0.0
        self._units_at_tick: dict[str, int] = {}

        self.enabled = self.tick_s > 0 and self.rate > 0
        self.log_steps = _parse_log_steps(self.stage_id)
        # Earliest pending release among held requests, refreshed by every
        # split_running call. The tick engine loop (WP1) sleeps until this.
        self.next_wake: float | None = None
        # rid -> [output_len_seen, units_this_turn, anchor_t, holds]
        self._st: dict[str, list] = {}

        if self.enabled:
            logger.info(
                "[TemporalPacer] stage=%s(%s) ON: tick=%.0fms rate=%.1f/s lead=%.0fms "
                "burst=%d quantize=%s barrier=%s budget/tick=%d catchup=%.1fx",
                self.stage_id, stage or "?", self.tick_s * 1000, self.rate,
                self.lead_s * 1000, self.burst, not self.no_quant,
                self.barrier, self.tick_budget, self.catchup,
            )

    # ------------------------------------------------------------------ #

    def split_running(self, running: list, now: float) -> tuple[list, list, float | None]:
        """Partition ``running`` into (schedulable, held); also return the
        earliest release time among held requests (for the idle micro-sleep)."""
        if self.barrier:
            kept, held, next_release = self._split_barrier(running, now)
            self.next_wake = next_release
            return kept, held, next_release
        kept: list = []
        held: list = []
        next_release: float | None = None
        for req in running:
            release = self._release_time(req, now)
            if release is not None:
                held.append(req)
                if next_release is None or release < next_release:
                    next_release = release
            else:
                kept.append(req)
        self.next_wake = next_release
        return kept, held, next_release

    def _split_barrier(self, running: list, now: float) -> tuple[list, list, float | None]:
        """Stage-wide tick barrier.

        A tick WINDOW opens at each grid boundary. Every request present at
        the opening gets its per-tick budget (catch-up multiplier when behind
        playback schedule); budget spent -> held to the next boundary. A
        request that joins mid-window (its thinker chunk just arrived, or it
        was chunk-parked at the opening) is held to the next boundary too --
        that is exactly the re-alignment the v1 per-request gate lacked.
        Requests still inside their per-turn burst bypass everything, so TTFA
        never pays for the barrier.
        """
        boundary = math.floor(now / self.tick_s) * self.tick_s
        fresh = boundary > self._tick_open
        if fresh:
            self._tick_open = boundary
            self._units_at_tick = {}
        next_boundary = self._tick_open + self.tick_s
        kept: list = []
        held: list = []
        for req in running:
            units, anchor = self._observe(req, now)
            rid = req.request_id
            if anchor is None or units <= self.burst:
                kept.append(req)
                continue
            if fresh:
                self._units_at_tick[rid] = units
            base = self._units_at_tick.get(rid)
            if base is None:
                # Joined mid-window: wait for the boundary, where everyone
                # steps together.
                held.append(req)
                continue
            ahead = (units - self.burst) / self.rate - (now - anchor)
            if ahead > self.lead_s:
                held.append(req)
                continue
            # Catch up TOWARD THE LEAD BUFFER, not merely back to zero
            # deficit. Budget = exactly realtime leaves the client no slack:
            # measured 5.26% deadline misses at 16 users because one hiccup
            # was one audible stall and 2x-only-when-behind recovered too
            # slowly. Running at catchup-x while below the lead target builds
            # and holds ~lead_s of client buffer; the ahead-cap above stops
            # it from growing past that.
            budget = self.tick_budget * (self.catchup if ahead < self.lead_s * 0.75 else 1.0)
            used = units - base
            if used >= budget:
                held.append(req)
                continue
            # Sub-slot barrier: the k-th unit of this window releases at
            # open + k*sub_slot -- catch-up units gather and step together.
            slot_open = self._tick_open + used * self.sub_slot
            if now < slot_open:
                held.append(req)
                if slot_open < next_boundary:
                    next_boundary = slot_open  # earliest wake for the micro-sleep
            else:
                kept.append(req)
        return kept, held, (next_boundary if held else None)

    def _observe(self, req: Any, now: float) -> tuple[int, float | None]:
        """Update per-turn unit counters for one request; return (units, anchor)."""
        toks = req.output_token_ids
        n = len(toks)
        st = self._st.get(req.request_id)
        if st is None:
            st = [0, 0, None, 0]
            self._st[req.request_id] = st
        if n < st[0]:
            # Outputs were cleared at a segment stop: a NEW TURN began.
            # Everything is per-turn, so reset (including the anchor).
            st[0] = 0
            st[1] = 0
            st[2] = None
        if n > st[0]:
            if self.codec_only:
                add = 0
                for i in range(st[0], n):
                    if toks[i] < _CODEC_VALID_MAX:
                        add += 1
                st[1] += add
            else:
                st[1] = n
            st[0] = n
            if st[2] is None and st[1] > 0:
                # Anchor at the first output observed this turn, i.e. right
                # after the first decode step. Prefill runs BEFORE any output
                # exists, so prefill is naturally exempt from pacing.
                st[2] = now
        return st[1], st[2]

    def _release_time(self, req: Any, now: float) -> float | None:
        """None = schedule freely this pass; else the (grid-aligned) time at
        which this request's next unit is due.

        KNOWN LIMITATION (measured, kept for the ablation record): a request
        whose release time is already in the past is passed through every
        pass, so anything ever delayed (a big co-scheduled prefill, a chunk
        wait) falls off the grid and free-runs -- batch consolidation never
        happens. Use VLLM_OMNI_TEMPORAL_BARRIER=1 for the version that holds
        the grid.
        """
        units, anchor = self._observe(req, now)
        st = self._st[req.request_id]
        if anchor is None or units <= self.burst:
            return None
        # On schedule when (units - burst) / rate <= elapsed + lead.
        release = anchor + (units - self.burst) / self.rate - self.lead_s
        if release <= now:
            return None
        if not self.no_quant:
            # GLOBAL grid: monotonic time quantized to tick multiples. Same
            # grid for every request in this process, so releases coincide and
            # the batcher forms one periodic batch. (The two stage processes
            # have independent grid phases; within a stage -- which is where
            # batches form -- the grid is shared.)
            release = math.ceil(release / self.tick_s) * self.tick_s
        st[3] += 1
        return release

    def on_request_freed(self, request_id: str) -> None:
        self._st.pop(request_id, None)
        self._units_at_tick.pop(request_id, None)


class ChunkTickGate:
    """80 ms timetable for a CHUNK-CONSUMING stage (code2wav / LLM_GENERATION).

    The stage's work items are whole upstream chunks whose arrival times are
    scattered by the async transport (save thread -> shm -> poll). Greedy
    consumption runs one vocoder call per arrival, at arrival phase; this gate
    holds fresh chunk work until the next tick boundary so every session's
    ready chunk is vocoded in ONE batched step -- the same barrier idea as the
    AR stages, applied at this consumption point.

    The FIRST chunk of each segment bypasses the gate: it carries the reply's
    opening audio and holding it would bill TTFA for the timetable. Later
    chunks refill a client buffer that is already >= lead_s deep, so +<=1 tick
    of delivery delay is inaudible.

    Enabled by the same env pair as the pacer: VLLM_OMNI_TEMPORAL_TICK_MS > 0
    and VLLM_OMNI_TEMPORAL_BARRIER=1.
    """

    def __init__(self, model_config: Any):
        tick_ms = _env_float("VLLM_OMNI_TEMPORAL_TICK_MS", 0.0)
        barrier = os.environ.get("VLLM_OMNI_TEMPORAL_BARRIER", "0") not in ("0", "", "false", "False")
        self.tick_s = max(0.0, tick_ms) / 1000.0
        self.enabled = self.tick_s > 0 and barrier
        self.stage_id = getattr(model_config, "stage_id", -1)
        # Earliest pending release; the tick engine loop sleeps until this.
        self.next_wake: float | None = None
        # rid -> [chunks_seen_this_segment, release_t | None]
        # release None = no pending gated work; 0.0 = released/pass-through.
        self._st: dict[str, list] = {}
        if self.enabled:
            logger.info("[ChunkTickGate] stage=%s ON: tick=%.0fms (first chunk per segment exempt)",
                        self.stage_id, self.tick_s * 1000)

    def should_hold(self, request_id: str, now: float) -> bool:
        st = self._st.setdefault(request_id, [0, None])
        if st[1] is None:
            # New chunk work just became visible.
            st[0] += 1
            if st[0] <= 1:
                st[1] = 0.0  # segment's first chunk: TTFA bypass
                return False
            st[1] = math.ceil(now / self.tick_s) * self.tick_s
        if now < st[1]:
            if self.next_wake is None or self.next_wake <= now or st[1] < self.next_wake:
                self.next_wake = st[1]
            return True
        return False

    def on_work_consumed(self, request_id: str) -> None:
        st = self._st.get(request_id)
        if st is not None:
            st[1] = None

    def on_segment_end(self, request_id: str) -> None:
        self._st[request_id] = [0, None]

    def on_request_freed(self, request_id: str) -> None:
        self._st.pop(request_id, None)
