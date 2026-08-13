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

# resolved after _LIVE_DEFAULTS below; placeholder for import order
_GATE_EXEMPT_CHUNKS = 1

# [live-vllm P1] On this branch the tick engine is the ARCHITECTURE, not an
# option: every temporal gate defaults ON and the request-driven path is the
# thing you must opt INTO (set the env to 0 explicitly) to build a control
# arm. Single source of truth for those defaults -- factory.py,
# chunk_transfer_adapter.py and video_stream_base.py import the helpers so a
# default can never fork between modules.
_LIVE_DEFAULTS = {
    "VLLM_OMNI_TEMPORAL_TICK_MS": "80",
    "VLLM_OMNI_TEMPORAL_BARRIER": "1",
    "VLLM_OMNI_TEMPORAL_ENGINE": "1",
    "VLLM_OMNI_TEMPORAL_REPLAY": "1",
    "VLLM_OMNI_TEMPORAL_INLINE_SEND": "1",
    "VLLM_OMNI_TEMPORAL_MAILBOX": "1",
    "VLLM_OMNI_TEMPORAL_FRAME_TICK": "1",
    # [P2] per-pass cap on APERIODIC prefill tokens (the slack slot). The
    # tick's decode heartbeat is exempt (budgeted on top); 0 disables.
    # Sizing: one pass's prefill slice must finish inside the tick --
    # 6144 tokens ~ 40-60 ms of stage-0 prefill on this hardware, leaving
    # the decode step comfortable margin in an 80 ms tick.
    "VLLM_OMNI_TEMPORAL_SLACK_TOKENS": "6144",
    # [P4] vocode phase groups: each session vocodes only on tick edges of
    # its own group (round-robin assigned), so per-tick vocoder load is
    # N/K BY CONSTRUCTION instead of on average. A session's chunk cadence
    # is 1 per 4 ticks (codec_chunk_frames=4), so K=4 adds <=3 ticks of
    # delivery delay -- inside the lead buffer, same argument as the gate's
    # own +<=1 tick. 0/1 disables (plain next-edge release).
    "VLLM_OMNI_TEMPORAL_VOCODE_PHASES": "4",
    # [anti-shoulder] chunks exempt from the vocode gate at each segment
    # start: this IS the client's buffer depth in chunks (320 ms each).
    "VLLM_OMNI_TEMPORAL_GATE_EXEMPT": "3",
    # [P7 phase-locked pipeline] Per-stage phase offsets on the SHARED tick
    # grid (all processes quantize the same host CLOCK_MONOTONIC, so the
    # edges coincide numerically across stages). The tick becomes a fixed
    # program: T+0 thinker decode -> (transfer, phase-locked mailbox) ->
    # T+PHASE1 talker step (its inputs are READY BY CONSTRUCTION: produced at
    # T+0, shipped inline) -> T+PHASE2 vocode gate edge (the 4th frame enters
    # its box in the SAME tick). This removes the straggler CONCEPT rather
    # than compensating for it: a session's input window closes before its
    # own departure slot. Measured disease it targets: u56 production-gap
    # tail (p99 586ms vs 402 at u48) from per-session frames completing at
    # load-coupled moments.
    "VLLM_OMNI_TEMPORAL_PHASE0_MS": "0",
    "VLLM_OMNI_TEMPORAL_PHASE1_MS": "30",
    "VLLM_OMNI_TEMPORAL_PHASE2_MS": "64",
    # [P7] time-budgeted slack (freight limiter): a prefill slice may only be
    # as large as the tick's REMAINING time can absorb at the measured
    # prefill throughput; right after the stage's phase edge there is a
    # decode-only zone where freight is barred entirely. 0 disables (fixed
    # SLACK_TOKENS cap only).
    "VLLM_OMNI_TEMPORAL_TIME_SLACK": "1",
    "VLLM_OMNI_TEMPORAL_DECODE_ZONE_MS": "18",
    # [T2T coalesce] thinker->talker text transport granularity. One payload
    # per thinker TOKEN (the original protocol) makes the talker pause once
    # per token to take delivery (park -> connector load -> resume, >= 1
    # scheduler pass each). At u56 that is ~4 deliveries per audio chunk on a
    # stage whose pass cadence (46 ms) only affords ~1.7 passes per tick:
    # measured chunk cadence 377 ms vs the 320 ms contract for exactly the
    # first ~30% of every turn (while text still streams) -- which is where
    # 96% of all client misses live. Rows after the first EXEMPT flushes of a
    # segment accumulate to TOKENS rows per payload; segment end always
    # flushes. 1 disables (per-token protocol, the control arm).
    # The PRIMARY trigger is the tick DEADLINE (TICKS), not the row count:
    # one flush per TICKS ticks on the shared grid, so text lag is bounded
    # regardless of the thinker's rate and every session's delivery lands on
    # the same edge (one pass serves all). TOKENS is then a ceiling on
    # payload size -- a faster thinker makes bigger batches, not more
    # deliveries. Budget: stage 1 affords ~1.7 passes/tick with 1 owed to the
    # frame, so ~0.7/tick is available for signatures. One flush per tick
    # would spend 1.0/tick: over budget, and ~= the 11/s the per-token
    # protocol already self-throttles to (measured), i.e. no change at all.
    # TICKS=4 spends 0.25/tick (3x margin) and equals one audio chunk's
    # period. TICKS=0 = count-only with the geometric ramp (the arm that
    # first measured this mechanism); TOKENS=1 = per-token control arm.
    # VERDICT (measured, u56, paired with burst12): batching alone took client
    # miss 20.1% -> 1.5%, but ONLY by shortening the text window 3.4x (83
    # deliveries -> 16); total stall barely moved (1489 -> 1292 ms/turn) and
    # every remaining miss is in the turn opening. The reason is receiver-side:
    # a consumer that needs a payload leaves `running` for a pass and advances
    # exactly one step per park/resume round-trip, so its rate is capped at
    # pass_rate/2 (~10 frames/s at u56, below the 12.5 contract) and BANKED
    # rows cannot be drained without another payload event. So the cure is
    # VLLM_OMNI_INLINE_RECV (below), and with it a per-token payload costs
    # nothing -- batching is kept OFF by default, as an ablation arm, since it
    # only adds a cross-segment bank to get wrong.
    "VLLM_OMNI_TEXT_COALESCE_TICKS": "0",
    "VLLM_OMNI_TEXT_COALESCE_TOKENS": "1",
    "VLLM_OMNI_TEXT_COALESCE_EXEMPT": "4",
    # [P8] Take delivery of an upstream payload on the scheduler thread when it
    # is already in shared memory, instead of parking the consumer for a
    # round-trip through the recv thread. This is the fix for the ceiling
    # described above: the talker stays in `running` and keeps its decode slot,
    # so frame production is limited by the pacer (as designed) rather than by
    # the text-delivery protocol. 0 = parked-only path (control arm).
    "VLLM_OMNI_INLINE_RECV": "1",
    # [P5] streaming vocoder conv window: the conv/upsample stack's measured
    # left receptive field is 10 codec frames (autograd probe; the 25-frame
    # left_context is a heuristic sized for the pre-transformer's attention,
    # not the convs). With this ON, the pre-transformer still sees the full
    # 25+new window (attention semantics unchanged, quality-approved) but the
    # conv stack -- the FLOPs-dominant half, running at 1920x upsampled
    # resolution -- only processes the last (new + 11) frames. Emitted
    # samples are mathematically identical. 0 = full-window convs.
    "VLLM_OMNI_STREAM_VOCODER": "1",
}

# [P2] Priority-class marker for aperiodic work, riding SamplingParams
# .extra_args (the established marker channel -- see _PREFILL_ONLY_KEY in
# video_stream_base). "background" = invisible-latency work (compression
# shadow seeds): it yields the slack slot to anything a user is waiting on.
SLACK_CLASS_KEY = "vllm_omni_slack_class"


def live_env(name: str) -> str:
    # Empty string counts as unset, matching _env_float: only an explicit
    # value (e.g. "0") opts out of a live default.
    v = os.environ.get(name)
    if v is None or v == "":
        return _LIVE_DEFAULTS.get(name, "0")
    return v


def live_env_on(name: str) -> bool:
    return live_env(name) not in ("0", "", "false", "False")


try:
    _GATE_EXEMPT_CHUNKS = max(1, int(float(live_env("VLLM_OMNI_TEMPORAL_GATE_EXEMPT") or 1)))
except ValueError:
    _GATE_EXEMPT_CHUNKS = 1

# [WP1] VLLM_OMNI_TEMPORAL_ENGINE=1 turns the engine-core busy loop into a
# tick loop: between pacing events it SLEEPS on the input queue (waking
# instantly for client requests) instead of hot-spinning through empty
# scheduler passes. Read once per engine-core process.
TICK_ENGINE_LOOP = (
    live_env_on("VLLM_OMNI_TEMPORAL_ENGINE")
    and float(live_env("VLLM_OMNI_TEMPORAL_TICK_MS") or 0) > 0
)
TICK_S = float(live_env("VLLM_OMNI_TEMPORAL_TICK_MS") or 0) / 1000.0

# [WP2] VLLM_OMNI_TEMPORAL_REPLAY=1: pure-decode steps of an unchanged cohort
# are emitted by the scheduler's cohort-replay fast path instead of the full
# upstream scheduling pass. See OmniARScheduler._try_replay_schedule.
TICK_REPLAY = live_env_on("VLLM_OMNI_TEMPORAL_REPLAY")


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
        tick_ms = _env_float("VLLM_OMNI_TEMPORAL_TICK_MS", 80.0)  # live-vllm: default ON
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

        self.barrier = live_env_on("VLLM_OMNI_TEMPORAL_BARRIER")
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
        # [P7] Phase offset: this stage's grid edges sit at k*tick + phase.
        # thinker=PHASE0 (0), talker=PHASE1 (after the thinker step + inline
        # transfer) -- the pipeline schedule that makes the talker's inputs
        # ready-by-construction at its own edge.
        if "talker" in stage:
            self.phase_s = _env_float("VLLM_OMNI_TEMPORAL_PHASE1_MS",
                                      float(live_env("VLLM_OMNI_TEMPORAL_PHASE1_MS"))) / 1000.0
        else:
            self.phase_s = _env_float("VLLM_OMNI_TEMPORAL_PHASE0_MS",
                                      float(live_env("VLLM_OMNI_TEMPORAL_PHASE0_MS"))) / 1000.0
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
        # [diagnosis] hold-reason census: which barrier branch dominates
        # storm windows. Logged+reset every ~2500 split calls when
        # VLLM_OMNI_LOG_SCHED_STEPS selects this stage.
        self._hold_census = {"midjoin": 0, "ahead": 0, "budget": 0, "subslot": 0, "kept": 0, "waste": 0}
        self._census_calls = 0

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
        # [P7] the grid is phase-shifted: edges at k*tick + phase. Same host
        # CLOCK_MONOTONIC across stage processes, so thinker edges (phase 0)
        # and talker edges (phase 30ms) interleave into one pipeline.
        boundary = math.floor((now - self.phase_s) / self.tick_s) * self.tick_s + self.phase_s
        fresh = boundary > self._tick_open
        if fresh:
            self._tick_open = boundary
            # [live-vllm anti-wave #5] Enroll EVERY tracked session at the
            # boundary, not just the ones momentarily in `running`: a talker
            # flaps out of running for every text-chunk load, and a boundary
            # falling inside that flap used to erase its enrollment -- the
            # midjoin fine (measured 70-130/s at u56; #4 shrank the fine
            # from a full tick to a sub-slot, this removes the ticket
            # entirely). A session parked long (between turns) re-enrolls
            # with its unit count unchanged, so the budget cap still binds.
            self._units_at_tick = {rid: st[1] for rid, st in self._st.items()}
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
                # [live-vllm anti-wave #4] Joined mid-window: REGISTER now and
                # release on the sub-slot grid, instead of forfeiting the
                # whole tick. The forfeit rule was the v1-regression guard,
                # but the census convicted it as the dominant hold under
                # load (70-130 midjoin-holds/s at u56 = every session fined
                # ~1 tick every 3-8 ticks): a talker briefly leaves running
                # for every text-chunk load, and any resume landing after
                # the window opening paid a full tick. Budget accounting and
                # sub-slot quantization below still apply, so this is the
                # 40 ms catch-up bus, not a free run.
                self._hold_census["midjoin"] += 1
                self._units_at_tick[rid] = units
                base = units
            ahead = (units - self.burst) / self.rate - (now - anchor)
            if ahead > self.lead_s:
                self._hold_census["ahead"] += 1
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
                self._hold_census["budget"] += 1
                held.append(req)
                continue
            # Sub-slot barrier: the k-th unit of this window releases at
            # open + k*sub_slot -- catch-up units gather and step together.
            slot_open = self._tick_open + used * self.sub_slot
            if now < slot_open:
                self._hold_census["subslot"] += 1
                held.append(req)
                if slot_open < next_boundary:
                    next_boundary = slot_open  # earliest wake for the micro-sleep
            else:
                self._hold_census["kept"] += 1
                kept.append(req)
        self._census_calls += 1
        if self.log_steps and self._census_calls >= 2500:
            c = self._hold_census
            logger.info(
                "[pacer-census] stage=%s mono=%.3f kept=%d midjoin=%d ahead=%d budget=%d subslot=%d waste=%d",
                self.stage_id, now, c["kept"], c["midjoin"], c["ahead"], c["budget"], c["subslot"], c["waste"])
            for k in c:
                c[k] = 0
            self._census_calls = 0
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
                # [diagnosis] invalid-token census: box-level stutter hides
                # in coarse global accounting.
                self._hold_census["waste"] += (n - st[0]) - add
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
        tick_ms = _env_float("VLLM_OMNI_TEMPORAL_TICK_MS", 80.0)  # live-vllm: default ON
        barrier = live_env_on("VLLM_OMNI_TEMPORAL_BARRIER")
        self.tick_s = max(0.0, tick_ms) / 1000.0
        self.enabled = self.tick_s > 0 and barrier
        self.stage_id = getattr(model_config, "stage_id", -1)
        # Same env as the AR stages' [SCHED-STEP] line: the generation
        # scheduler consults this to emit per-pass batch evidence, which is
        # how tick-aligned vocode batching is verified rather than assumed.
        self.log_steps = _parse_log_steps(self.stage_id)
        # [live-vllm P4] Phase groups: session s releases only on tick edges
        # where edge_index % K == group(s). Round-robin assignment at first
        # sight keeps groups exactly level (a hash could clump).
        try:
            self.phase_groups = max(0, int(float(live_env("VLLM_OMNI_TEMPORAL_VOCODE_PHASES") or 0)))
        except ValueError:
            self.phase_groups = 0
        # [P7] gate edges at k*tick + PHASE2: placed after the talker's step
        # (PHASE1 + step time) so a chunk completed this tick vocodes THIS
        # tick instead of waiting for the next plain edge.
        self.phase_s = _env_float("VLLM_OMNI_TEMPORAL_PHASE2_MS",
                                  float(live_env("VLLM_OMNI_TEMPORAL_PHASE2_MS"))) / 1000.0
        self._grp: dict[str, int] = {}
        self._next_grp = 0
        if self.enabled and self.phase_groups > 1:
            logger.info("[ChunkTickGate] stage=%s phase groups: K=%d (per-tick vocode load = N/K by construction)",
                        self.stage_id, self.phase_groups)
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
            # [live-vllm anti-shoulder] Exempt the first E chunks per segment
            # (was 1). The single-chunk exemption silently capped the CLIENT
            # buffer at ~one chunk (320 ms) forever: the gate meters every
            # later chunk to playback cadence, so whatever lead the talker
            # builds NEVER reaches the listener -- measured as lead 240->400
            # having ZERO effect on misses while a 15% shoulder of +40-160 ms
            # boxes kept draining the paper-thin buffer. With E=3 the client
            # opens each segment ~960 ms deep (production covers it via the
            # per-turn burst + catch-up budget) and the shoulder is absorbed.
            if st[0] <= _GATE_EXEMPT_CHUNKS:
                st[1] = 0.0  # TTFA/buffer-building bypass
                return False
            # [P4] release on the next edge OF THIS SESSION'S PHASE GROUP,
            # not just the next edge: per-tick vocode load becomes N/K by
            # construction. Groups are round-robin at first sight. The
            # alignment costs up to K-1 ticks ONCE per segment (the 4-tick
            # chunk cadence preserves group phase afterwards), so it starts
            # at chunk 3: chunk 2 lands while the client buffer is thinnest
            # and keeps today's plain next-edge release.
            # [P7] edges live on the phase-shifted grid (k*tick + PHASE2).
            edge = math.ceil((now - self.phase_s) / self.tick_s)
            if self.phase_groups > 1 and st[0] > 2:
                grp = self._grp.get(request_id)
                if grp is None:
                    grp = self._grp[request_id] = self._next_grp
                    self._next_grp = (self._next_grp + 1) % self.phase_groups
                while edge % self.phase_groups != grp:
                    edge += 1
            st[1] = edge * self.tick_s + self.phase_s
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
        # Phase group is released with the request; the slot recycles to the
        # round-robin naturally as new sessions arrive.
        self._grp.pop(request_id, None)
