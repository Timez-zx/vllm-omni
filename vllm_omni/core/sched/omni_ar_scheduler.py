from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from time import monotonic as _monotonic
from time import sleep as _sleep
from time import time
from typing import Any

import numpy as np
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.scheduler import PauseState
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.model_executor.stage_input_processors.tts_utils import PREFILL_ONLY_KEY


def _update_is_prefill_only(update: Any) -> bool:
    """Does THIS streaming update carry the prefill-only marker?

    Reads the update's own additional_information and nothing else. Two
    tempting shortcuts are wrong in ways that look like success:

    * request.additional_information keeps the FIRST chunk's payload for the
      whole session (nothing in the stage-0 update path replaces it), so a
      request-level read classifies every segment by the opening chunk --
      either turns park unsampled forever or appends never park.
    * tts_utils.prefill_only_channel() checks extra_args FIRST, and arrival
      appends are dual-marked (extra_args stamped by the chunk stream,
      additional_information by the append builder), so any channel-equality
      test routed through that helper matches nothing: an inert guard.
    """
    info = getattr(update, "additional_information", None)
    entries = getattr(info, "entries", None)
    if isinstance(entries, dict) and PREFILL_ONLY_KEY in entries:
        entry = entries[PREFILL_ONLY_KEY]
        list_data = getattr(entry, "list_data", None)
        if isinstance(list_data, list) and list_data:
            entry = list_data[0]
        if isinstance(entry, list) and entry:
            entry = entry[0]
        if str(entry).strip().lower() in ("1", "true", "yes"):
            return True
    # Defense in depth: the update's OWN sampling_params.extra_args -- the
    # channel that demonstrably survives every transport (it is how the chunk
    # adapter has been seeing the marker all along, while the payload struct
    # was being dropped by a dict-only filter upstream). Still a per-update
    # value: a turn chunk's update carries the turn's params, never a stale
    # append's.
    params = getattr(update, "sampling_params", None)
    extra = getattr(params, "extra_args", None)
    return (isinstance(extra, dict)
            and str(extra.get(PREFILL_ONLY_KEY, "")).strip().lower()
            in ("1", "true", "yes"))
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.core.sched.temporal_pacing import SLACK_CLASS_KEY as _SLACK_CLASS_KEY
from vllm_omni.core.sched.temporal_pacing import TICK_ENGINE_LOOP as _TICK_ENGINE_LOOP
from vllm_omni.core.sched.temporal_pacing import TICK_REPLAY as _TICK_REPLAY
from vllm_omni.core.sched.temporal_pacing import TemporalPacer, live_env, live_env_on

# [live-vllm P2] Per-pass aperiodic-prefill cap (the slack slot); decode
# heartbeat tokens are budgeted ON TOP of this, so a prefill slice can never
# stretch a pass past the tick edge. Read once per engine-core process.
_SLACK_TOKENS = int(float(live_env("VLLM_OMNI_TEMPORAL_SLACK_TOKENS") or 0))

# [live-vllm P7] Freight limiter: size each prefill slice by TIME, not by a
# fixed token count -- the slice may only be as large as the tick's remaining
# window absorbs at the MEASURED prefill throughput, and the first
# DECODE_ZONE ms after the stage's phase edge carry no freight at all. This
# is what makes "the thinker's step completes before the talker's phase edge"
# a guarantee instead of a hope (measured: stage-0 pass tail p99 190ms from
# 6144-token slices riding decode passes).
_TIME_SLACK = live_env_on("VLLM_OMNI_TEMPORAL_TIME_SLACK")
_DECODE_ZONE_S = float(live_env("VLLM_OMNI_TEMPORAL_DECODE_ZONE_MS") or 0) / 1000.0

# [diagnosis] VLLM_OMNI_LOG_SEG_CYCLES=1: one line per streaming-segment
# lifecycle event (park = segment done and the NEXT one has not arrived;
# cont = next segment was already queued, no park; wake = update arrived for
# a parked session). The instrument that convicts or acquits the park/wake
# duty-cycle hypothesis for the u56 production slips.
_LOG_SEG_CYCLES = live_env_on("VLLM_OMNI_LOG_SEG_CYCLES")


def _slack_is_background(request: Any) -> bool:
    sp = getattr(request, "sampling_params", None)
    ea = getattr(sp, "extra_args", None) if sp is not None else None
    return bool(ea) and ea.get(_SLACK_CLASS_KEY) == "background"
from vllm_omni.core.sched.utils import omni_routed_experts_for_request
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.outputs import OmniConnectorOutput

logger = init_logger(__name__)


class SampledLogprobContractError(RuntimeError):
    """The model runner returned unusable sampled-token logprobs."""


def _slice_sampled_logprobs(logprobs: Any, req_index: int, sampled_token_ids: list[int]) -> Any:
    """Slice and validate the sampled-token logprobs for one AR request."""
    if logprobs is None:
        raise SampledLogprobContractError("AR logprobs were requested, but the model runner returned none")

    sliced = logprobs.slice_request(req_index, len(sampled_token_ids))
    token_rows = np.asarray(sliced.logprob_token_ids)
    value_rows = np.asarray(sliced.logprobs)
    expected_rows = len(sampled_token_ids)

    if token_rows.ndim != 2 or value_rows.ndim != 2:
        raise SampledLogprobContractError(
            "AR sampled-token logprobs must be rank-2 arrays, "
            f"got token_ids={token_rows.shape} logprobs={value_rows.shape}"
        )
    if token_rows.shape[0] != expected_rows or value_rows.shape[0] != expected_rows:
        raise SampledLogprobContractError(
            "AR sampled-token logprob row count does not match generated tokens: "
            f"tokens={expected_rows} token_id_rows={token_rows.shape[0]} "
            f"logprob_rows={value_rows.shape[0]}"
        )
    if expected_rows == 0:
        return sliced
    if token_rows.shape[1] == 0 or value_rows.shape[1] == 0:
        raise SampledLogprobContractError("AR sampled-token logprob rows are empty")

    sampled = np.asarray(sampled_token_ids)
    if not np.array_equal(token_rows[:, 0], sampled):
        mismatch = np.flatnonzero(token_rows[:, 0] != sampled)
        first = int(mismatch[0])
        raise SampledLogprobContractError(
            "AR sampled-token logprobs are misaligned: "
            f"row={first} generated_token={int(sampled[first])} "
            f"logprob_token={int(token_rows[first, 0])}"
        )
    if not np.isfinite(value_rows[:, 0]).all():
        bad_rows = np.flatnonzero(~np.isfinite(value_rows[:, 0])).tolist()
        raise SampledLogprobContractError(f"AR sampled-token logprobs contain non-finite values at rows {bad_rows}")
    return sliced


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    # How long a stage may schedule nothing, while still tracking requests, before it is
    # reported as wedged (see _check_for_wedged_requests). Generous on purpose: a streaming
    # session is legitimately idle between turns, and the client's own turn timeout is 240s,
    # so this has to be well inside that to be useful but far outside normal think time.
    _WEDGE_REPORT_AFTER_S = 45.0

    # Set once a stage has tracked at least one request, so that dropping back to zero can be
    # told apart from never having started. See _check_for_wedged_requests.
    _had_requests = 0
    _empty_reported_t = 0.0
    _empty_since = None

    # Heartbeat cadence for the presence check at the top of schedule().
    _HEARTBEAT_EVERY_S = 10.0
    _last_heartbeat_t = 0.0

    # Cadence for the starved-loop report in has_requests(), which is called every loop
    # iteration and so must be rate-limited hard.
    _STARVED_REPORT_EVERY_S = 5.0
    _starved_reported_t = 0.0
    _counter_repaired_t = 0.0
    _counter_clamped_t = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        self.finished_req_ids_dict = defaultdict(set)

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        self.input_coordinator: OmniSchedulingCoordinator | None = None
        if uses_full_payload_input_coordinator(model_config):
            self.input_coordinator = OmniSchedulingCoordinator(
                stage_id=getattr(model_config, "stage_id", 0),
            )
        # [Temporal batching experiment] pacing gate; a no-op unless
        # VLLM_OMNI_TEMPORAL_TICK_MS selects a tick AND this stage is
        # thinker/talker. See core/sched/temporal_pacing.py.
        self.temporal_pacer = TemporalPacer(model_config)
        # When each currently-tracked request was first seen with no sampled output, and which
        # have already been reported as producing nothing. See _check_for_wedged_requests.
        self._req_seen_t: dict[str, float] = {}
        self._mute_reported: set[str] = set()
        self._latest_omni_connector_output: OmniConnectorOutput | None = None
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when orchestrator will not run stage 1+ for this request (e.g. text-only).

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _in_decode_zone(self, now: float | None = None) -> bool:
        """[live-vllm W1/W2] True inside the decode-only window right after
        this stage's phase edge."""
        if not (_TIME_SLACK and _DECODE_ZONE_S > 0 and self.temporal_pacer.enabled
                and self.temporal_pacer.barrier and self.temporal_pacer.tick_s > 0):
            return False
        if now is None:
            now = _monotonic()
        in_win = (now - self.temporal_pacer.phase_s) % self.temporal_pacer.tick_s
        return in_win < _DECODE_ZONE_S

    def _should_defer_waiting_admission(self) -> bool:
        # [live-vllm W1] Membership events must never break the heartbeat:
        # inside the decode zone, ALL waiting admissions defer to the slack
        # window later in the same tick, so the decode pass sees an
        # unchanged cohort and the replay fast path stays eligible even
        # while a compression wave is being born. Measured basis: within
        # the wave window, seconds containing a warmup launch carried 10.7%
        # slip rate vs 2.5% for event-free seconds -- the tax is admission
        # work displacing the heartbeat, not compute volume.
        return bool(self.waiting) and self._in_decode_zone()

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self.transfer_triggered_requests.add(request.request_id)

                self._mark_request_for_kv_transfer(request.request_id, confirmed_computed)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    # Defer the stop until KV extraction completes so that
                    # the kv_ready signal can be emitted while the request
                    # is still alive.  The request will be stopped on the
                    # next scheduler step after extraction ack arrives.
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                self.transfer_triggered_requests.add(request.request_id)

                try:
                    idx = new_token_ids.index(target_token_id)
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = confirmed_computed - tokens_to_exclude
                except ValueError:
                    snapshot_len = confirmed_computed

                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        return False

    def add_request(self, request: Request) -> None:
        """Log every admission, then admit.

        There is no other way to tell "the request never reached this stage" from "it reached
        it and was removed again", and those have completely different causes. The question is
        live: rolling a session mints a new engine request, the API server logs sending it to
        all three stages, stage 0 accepts it and streams 370 payloads at stage 1 -- and stage 1
        behaves exactly as though it has no request at all. Upstream's `EngineCore.add_request`
        calls straight through to here, so a line here means the message arrived; its absence
        means it did not. Note the `abort_immediately` flag, which upstream honours by aborting
        the request on the very next line after this call.
        """
        logger.info(
            "[OmniARScheduler] stage %s ADMIT req=%s resumable=%s prompt_tokens=%s "
            "abort_immediately=%s (tracked before this: %d)",
            self.vllm_config.model_config.stage_id,
            getattr(request, "request_id", "?"),
            getattr(request, "resumable", None),
            getattr(request, "num_prompt_tokens", "?"),
            getattr(request, "abort_immediately", None),
            len(self.requests),
        )
        super().add_request(request)

    def _log_request_table(self, why: str) -> None:
        """Dump every tracked request's scheduling state.

        Upstream's ``assert num_new_tokens > 0`` carries no context and kills the whole
        engine-core process, so a bare traceback says only that SOME request had nothing to
        compute -- not which, not in which queue, not with what token counts. Two rounds of
        fixing this by deduction produced a guard that never fired (first over ``waiting``,
        then over ``waiting`` and ``skipped_waiting``), which is a sign that the request is
        not where reading the code says it should be. This prints the state instead.

        Note the queues are printed as they are AT THE MOMENT OF THE FAILURE, which is inside
        ``super().schedule()``: the chunk transfer adapter removes requests waiting on a chunk
        from both queues before that call and restores them after, and waiting admission may
        have been swapped out for an empty queue, so a request can be tracked in
        ``self.requests`` while appearing in none of the queues here. That is a finding, not a
        gap in the dump -- it is exactly the case a queue sweep cannot see.
        """
        def describe(request: Any) -> str:
            sq = getattr(request, "streaming_queue", None)
            return (
                f"id={getattr(request, 'request_id', '?')} "
                f"status={getattr(getattr(request, 'status', None), 'name', '?')} "
                f"num_tokens={getattr(request, 'num_tokens', '?')} "
                f"computed={getattr(request, 'num_computed_tokens', '?')} "
                f"prompt={getattr(request, 'num_prompt_tokens', '?')} "
                f"output={len(getattr(request, 'output_token_ids', ()) or ())} "
                f"resumable={getattr(request, 'resumable', None)} "
                f"streaming_queue={len(sq) if sq is not None else None} "
                f"preemptions={getattr(request, 'num_preemptions', '?')}"
            )

        logger.error("[OmniARScheduler] %s (stage %s)", why,
                     self.vllm_config.model_config.stage_id)
        queued: set[int] = set()
        for name, queue in (("waiting", self.waiting),
                            ("skipped_waiting", self.skipped_waiting),
                            ("running", self.running)):
            items = list(queue)
            logger.error("[OmniARScheduler]   %s: %d", name, len(items))
            for request in items:
                queued.add(id(request))
                logger.error("[OmniARScheduler]     %s", describe(request))
        # Anything tracked but in no queue -- see the docstring; this is the interesting row.
        orphans = [r for r in self.requests.values() if id(r) not in queued]
        logger.error("[OmniARScheduler]   tracked but in NO queue: %d", len(orphans))
        for request in orphans:
            logger.error("[OmniARScheduler]     %s", describe(request))

        # The chunk transfer adapter's state, because half the ways a downstream stage can
        # stop making progress live in there rather than in the queues. In particular
        # `requests_with_ready_chunks` is only ever cleared by `_clear_chunk_ready`, which
        # keys off requests appearing in a scheduler_output -- so a request that is in that
        # set and never gets scheduled is skipped by `_process_chunk_queue` on every
        # subsequent pass (it `continue`s before load_async), and no further chunk is ever
        # loaded for it. That is indistinguishable from "idle" without printing the set.
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None:
            return
        def ids(value: Any) -> str:
            if value is None:
                return "-"
            try:
                items = [getattr(r, "request_id", r) for r in value]
            except TypeError:
                return repr(value)[:120]
            return f"{len(items)}{items[:4]}"

        logger.error("[OmniARScheduler]   chunk adapter state:")
        for name in ("requests_with_ready_chunks", "waiting_for_chunk_waiting_requests",
                     "waiting_for_chunk_running_requests", "_finished_load_reqs",
                     "_active_streams", "_held_non_active", "finished_requests",
                     # `segment_finished_requests` gates `is_done_receiving_chunks`, which the
                     # RECEIVER consults as well as the scheduler. While a request sits in it,
                     # stage 1 neither pulls a chunk nor schedules, and only a streaming update
                     # clears it -- so if the update lands before the previous segment's final
                     # payload sets the flag, the wakeup is lost and the stage is silent for
                     # good while stage 0 keeps pushing payloads onto the edge. That is the
                     # shape of the observed wedge, and it is invisible without this row.
                     "segment_finished_requests",
                     "requests_origin_status"):
            logger.error("[OmniARScheduler]     %-36s %s", name,
                         ids(getattr(adapter, name, None)))
        logger.error("[OmniARScheduler]     %-36s %s", "_active_window",
                     getattr(adapter, "_active_window", "?"))

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Heartbeat, rate-limited. Every other diagnostic in this class lives at the END of
        # this method, so all of them go quiet together if the engine's busy loop stops calling
        # it -- and their silence then gets read as "nothing is wrong with the request", which
        # is the opposite of the truth. Upstream's loop blocks while
        # `not self.scheduler.has_requests()`, and has_requests() counts `waiting`/`running`,
        # NOT `self.requests`; the chunk transfer adapter takes a request OUT of those queues
        # while its payload load is pending. So a request can be tracked, unschedulable, and
        # invisible to every check below, with the loop parked. This line is the only way to
        # tell that state from a healthy idle stage, because it reports presence rather than
        # absence.
        # [Tick engine WP1] safe default; the tail sets the real value.
        self.omni_tick_idle = False
        _hb_now = time()
        if _hb_now - self._last_heartbeat_t >= self._HEARTBEAT_EVERY_S:
            self._last_heartbeat_t = _hb_now
            logger.info(
                "[OmniARScheduler] stage %s heartbeat: tracked=%d waiting=%d skipped=%d "
                "running=%d",
                self.vllm_config.model_config.stage_id,
                len(self.requests), len(self.waiting), len(self.skipped_waiting),
                len(self.running),
            )

        # Encoder-cache deadlock breaker. Upstream frees a request's passed
        # encoder inputs only in update_from_output -- i.e. only for requests
        # that STEPPED. Under a collective multimodal arrival wave (24 users x
        # ~10 video frames x ~222 embeds > the 62,720-embed cache) every
        # request ends up truncated at its next frame (num_new_tokens=0, cache
        # full), so nobody steps, so nobody frees the frames they already
        # computed, so the cache stays full: a self-sustaining stall that only
        # a client-timeout abort used to break (measured: all 24 sessions
        # frozen for ~170-180 s in the u24-mixed and u32-short cells, all
        # arms). Sweeping the frees at schedule() time instead of step time
        # removes the step->free dependency and with it the deadlock: a
        # request's already-passed frames release as soon as the scheduler
        # runs, whether or not the request itself can move. The condition
        # inside _free_encoder_inputs (positions fully computed and past
        # placeholders) is what makes this safe to call at any moment; cost is
        # O(tracked x cached-ids) over small sets.
        for _req in self.requests.values():
            if _req.has_encoder_inputs:
                self._free_encoder_inputs(_req)

        # Remove FINISHED_ABORTED requests before the upstream scheduler sees
        # them. Upstream vllm raises RuntimeError on this status; omni allows
        # async abort (e.g. client disconnect during TTS streaming) to leave
        # requests in the waiting/running queues temporarily.
        #
        # `skipped_waiting` is swept too, and leaving it out is what killed a stage whenever a
        # streaming session ended. An aborted request parked there was never dropped, and
        # upstream's waiting loop draws from `skipped_waiting` BEFORE `waiting` under FCFS
        # (`_select_waiting_queue_for_scheduling`), so it picked the aborted session up, found
        # num_tokens == num_computed_tokens, and tripped `assert num_new_tokens > 0` -- which
        # takes down the whole engine-core process. Measured, from the state dump below:
        #   skipped_waiting: 1
        #     status=FINISHED_ABORTED num_tokens=4592 computed=4592 resumable=True
        #
        # Via `remove_requests` rather than `remove`, which is also a latent fix: `remove` is
        # available only because FCFSRequestQueue happens to subclass deque. Under the
        # PRIORITY policy `self.waiting` is a PriorityRequestQueue, which has no `remove` at
        # all, so the original line would have raised AttributeError on the first async abort.
        # `remove_requests` is the RequestQueue interface and works for both.
        for req in [r for r in self.running
                    if getattr(r, "status", None) == RequestStatus.FINISHED_ABORTED]:
            self.running.remove(req)
        for queue in (self.waiting, self.skipped_waiting):
            doomed = [r for r in queue
                      if getattr(r, "status", None) == RequestStatus.FINISHED_ABORTED]
            if doomed:
                queue.remove_requests(doomed)
        self._recover_orphaned_requests()
        self._consume_pending_connector_output(model_mode="ar")
        self._process_pending_input_timeouts()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting, self.running, scheduler_requests=self.requests
            )


        self._clamp_streaming_parked_counter()

        # [Temporal pacing] hold ahead-of-realtime requests out of THIS pass.
        # Same shape as the adapter's _held_non_active: removed from running
        # here, restored in the finally below -- within one schedule() call, so
        # status, counters, has_requests() and orphan recovery never see it.
        _paced_held: list[Request] = []
        _paced_next_release: float | None = None
        if self.temporal_pacer.enabled and self.running:
            _kept, _paced_held, _paced_next_release = self.temporal_pacer.split_running(
                self.running, _monotonic())
            if _paced_held:
                self.running[:] = _kept

        original_waiting = None
        if self._should_defer_waiting_admission():
            original_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)

        # [live-vllm P2] Slack slot + express lane. Two rules, both scoped to
        # this one pass and unwound in the finally:
        #   (a) pass budget = decode needs (2 tok x running) + slack cap --
        #       an aperiodic prefill slice can no longer consume the whole
        #       max_num_batched_tokens and stretch the pass past the tick.
        #   (b) BACKGROUND waiting work (compression shadow seeds, marked via
        #       SLACK_CLASS_KEY) is parked whenever any FOREGROUND request is
        #       waiting: user-visible opens never queue behind invisible
        #       warm-ups. Parked = same shape as the pacer park (removed for
        #       one pass, restored before anything can look).
        _slack_bg_held: list[Request] = []
        _slack_budget_saved: int | None = None
        if _SLACK_TOKENS > 0 and self.temporal_pacer.enabled:
            waiting_now = list(self.waiting)
            bg = [r for r in waiting_now if _slack_is_background(r)]
            if bg and len(bg) < len(waiting_now):
                self.waiting.remove_requests(bg)
                _slack_bg_held = bg
                self._slack_bg_parks = getattr(self, "_slack_bg_parks", 0) + len(bg)
            _now_mono = _monotonic()
            # [P7] settle the previous pass's prefill-throughput sample: the
            # gap to the next schedule() call approximates its execute time.
            _prev = getattr(self, "_ts_prev", None)
            if _prev is not None:
                _dt = _now_mono - _prev[0]
                if 0 < _dt < 2 * max(self.temporal_pacer.tick_s, 0.04) and _prev[1] > 256:
                    _sample = _prev[1] / _dt
                    _old = getattr(self, "_prefill_tps", 100_000.0)
                    self._prefill_tps = min(500_000.0, max(20_000.0, 0.8 * _old + 0.2 * _sample))
                self._ts_prev = None
            _slack_cap = _SLACK_TOKENS
            if (_TIME_SLACK and self.temporal_pacer.barrier
                    and self.temporal_pacer.tick_s > 0):
                _in_win = (_now_mono - self.temporal_pacer.phase_s) % self.temporal_pacer.tick_s
                # [W2] slack window: drain teardown block-frees parked by
                # the decode zone (bounded work, pool accounting catches up
                # within the tick).
                if _in_win >= _DECODE_ZONE_S:
                    _dbf = getattr(self, "_deferred_block_frees", None)
                    while _dbf:
                        self._free_blocks(_dbf.pop())
                if _in_win < _DECODE_ZONE_S:
                    # decode-only zone right after this stage's phase edge:
                    # the heartbeat step never queues behind freight.
                    _slack_cap = 0
                else:
                    _remaining = self.temporal_pacer.tick_s - _in_win
                    _tps = getattr(self, "_prefill_tps", 100_000.0)
                    _slack_cap = min(_SLACK_TOKENS, int(_tps * _remaining * 0.7))
            _slack_budget_saved = self.max_num_scheduled_tokens
            self.max_num_scheduled_tokens = min(
                _slack_budget_saved,
                _slack_cap + 2 * len(self.running),
            )

        try:
            # [WP2 cohort replay] a pure-decode step for an unchanged cohort
            # needs none of the full pass's queue scans / budget arithmetic /
            # encoder logic -- emit it directly. Returns None (-> full path)
            # whenever ANY validity gate fails.
            scheduler_output = self._try_replay_schedule()
            # [live-vllm P1] replay is the intended STEADY-STATE path, not an
            # opportunistic fast path -- measure its coverage so "steady state
            # = replay" is a number, not a hope. Non-idle passes only.
            if scheduler_output is not None:
                self._replay_hits = getattr(self, "_replay_hits", 0) + 1
            else:
                scheduler_output = super().schedule(throttle_prefills)
                if scheduler_output.total_num_scheduled_tokens:
                    self._replay_misses = getattr(self, "_replay_misses", 0) + 1
            # [P7] stamp this pass's freight so the NEXT call can settle a
            # prefill-throughput sample (call gap ~ this pass's execute time).
            _pf_toks = (scheduler_output.total_num_scheduled_tokens
                        - len(scheduler_output.num_scheduled_tokens))
            if _pf_toks > 256:
                self._ts_prev = (_monotonic(), _pf_toks)
            _hits = getattr(self, "_replay_hits", 0)
            _misses = getattr(self, "_replay_misses", 0)
            if (_hits + _misses) and (_hits + _misses) % 5000 == 0:
                logger.info(
                    "[replay-coverage] stage=%s hits=%d full=%d (%.1f%% of non-idle passes replayed) slack_bg_parks=%d",
                    self.vllm_config.model_config.stage_id, _hits, _misses,
                    100.0 * _hits / (_hits + _misses),
                    getattr(self, "_slack_bg_parks", 0),
                )
        except AssertionError:
            # Upstream asserts kill the engine-core process. Dump the state before it dies,
            # or the only evidence is a traceback with no request in it.
            self._log_request_table("upstream schedule() raised AssertionError")
            raise
        finally:
            if _slack_budget_saved is not None:
                self.max_num_scheduled_tokens = _slack_budget_saved
            if _slack_bg_held:
                # Background work re-queues at the front: it keeps its arrival
                # position for the next pass where no foreground is waiting.
                self.waiting.prepend_requests(_slack_bg_held)
            if _paced_held:
                # Back into running before anything else can look: a paced
                # request is RUNNING in every observable sense, it just sat
                # out this one pass.
                self.running.extend(_paced_held)
            if original_waiting is not None:
                deferred_waiting = list(self.waiting)
                if deferred_waiting:
                    original_waiting.prepend_requests(deferred_waiting)
                self.waiting = original_waiting
                # [W1] wake at the zone end so deferred admissions run in
                # THIS tick's slack window, not the next grid edge. Set in
                # the finally because split_running overwrites next_wake
                # during the pass.
                if self.waiting and self._in_decode_zone():
                    _nw_now = _monotonic()
                    _zone_end = (_nw_now
                                 - ((_nw_now - self.temporal_pacer.phase_s)
                                    % self.temporal_pacer.tick_s)
                                 + _DECODE_ZONE_S)
                    _nw = self.temporal_pacer.next_wake
                    if _nw is None or _zone_end < _nw:
                        self.temporal_pacer.next_wake = _zone_end
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(
                    self.waiting,
                    self.running,
                    scheduler_requests=self.requests,
                )
            if self.input_coordinator:
                self.input_coordinator.restore_queues(self.waiting)
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    prompt_is_token_ids=nr.prompt_is_token_ids,
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                    model_intermediate_buffer=(
                        getattr(request, "model_intermediate_buffer", None) if request else None
                    ),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        self._check_for_wedged_requests(scheduler_output)

        # [Temporal pacing] one line per NON-IDLE pass: the raw material for
        # the batch-size timeline that verifies (or refutes) tick-aligned
        # batch formation. Idle passes are skipped so greedy baselines log
        # nothing extra between turns.
        if self.temporal_pacer.log_steps and scheduler_output.total_num_scheduled_tokens:
            logger.info(
                "[SCHED-STEP] stage=%s mono=%.6f nreq=%d ntok=%d held=%d run=%d wait=%d "
                "irecv=%d/%d/%d/%d",
                self.vllm_config.model_config.stage_id,
                _monotonic(),
                len(scheduler_output.num_scheduled_tokens),
                scheduler_output.total_num_scheduled_tokens,
                len(_paced_held),
                len(self.running),
                len(self.waiting),
                # [P8] cumulative hits/misses/skips/dupes for inline receive.
                # hit = delivery that cost the consumer no park; miss = payload
                # genuinely not there yet; skip = something already owned the
                # fetch or a payload was already in hand; dupe = a second
                # load_async for a request that already had one queued. hits
                # near total means the park is off the text path. All four zero
                # means the code never ran -- which is how the first attempt's
                # null result was diagnosed, so the distinction is load-bearing.
                getattr(self.chunk_transfer_adapter, "_inline_recv_hits", 0)
                if self.chunk_transfer_adapter else 0,
                getattr(self.chunk_transfer_adapter, "_inline_recv_misses", 0)
                if self.chunk_transfer_adapter else 0,
                getattr(self.chunk_transfer_adapter, "_inline_recv_skips", 0)
                if self.chunk_transfer_adapter else 0,
                getattr(self.chunk_transfer_adapter, "_async_load_dupes", 0)
                if self.chunk_transfer_adapter else 0,
            )

        # [Tick engine WP1] tell the engine loop whether this pass found any
        # schedulable work. True -> the loop may sleep until the pacer's
        # next_wake (handling client input the moment it arrives); False ->
        # step again immediately (prefill/slack work never waits for a tick).
        self.omni_tick_idle = (scheduler_output.total_num_scheduled_tokens == 0
                               and not self.waiting)

        # [Temporal pacing] when EVERYTHING schedulable is parked and this
        # pass produced no work, the busy loop would spin hot until the next
        # release. Yield for up to 1 ms (bounded, so a new turn's streaming
        # update is delayed by at most that much). With the tick engine loop
        # (VLLM_OMNI_TEMPORAL_ENGINE=1) the loop itself owns the wait, so the
        # bounded yield here is skipped.
        if (not _TICK_ENGINE_LOOP and _paced_held
                and scheduler_output.total_num_scheduled_tokens == 0
                and not self.waiting):
            _now2 = _monotonic()
            if _paced_next_release is not None and _paced_next_release > _now2:
                _sleep(min(_paced_next_release - _now2, 0.001))

        # Wrap in omni scheduler output to carry transfer metadata.
        return self._wrap_omni_scheduler_output(
            scheduler_output,
            finished_requests_needing_kv_transfer=finished_reqs,
        )

    def _try_replay_schedule(self) -> SchedulerOutput | None:
        """[Tick engine WP2] cohort-replay fast path.

        For a step where every schedulable request is a decode-ready member
        of an unchanged cohort (1 new token each), the full upstream pass
        computes nothing this method does not: the per-step diff is exactly
        {num_computed_tokens, num_output_tokens, new_block_ids-on-crossing}
        (verified against vllm 0.26 scheduler.py line by line -- see
        TICK_ENGINE_PLAN.zh.md). Everything stateful is REUSED from upstream
        (allocate_slots per request -- proven side-effect-equivalent for
        running decode, incl. the prefix-cache fill commit; \
        _make_cached_request_data -- which also handles the MRV1
        all_token_ids re-send for requests the pacer held out of the
        previous step; _update_after_schedule -- computed/in-flight advance
        and finished/preempted set rebind).

        Returns None -> caller runs the full path. Never partially commits:
        the only mutations before the last possible bail are current_step,
        new_step_starts() and allocate_slots(), each of which the full path
        tolerates (counter drift is benign, new_step_starts is idempotent
        for these managers, and re-running allocate_slots after a partial
        allocation needs 0 new blocks).
        """
        if not (_TICK_REPLAY and self.temporal_pacer.enabled and self.temporal_pacer.barrier):
            return None
        if (self._pause_state != PauseState.UNPAUSED or self.waiting
                or not self.running):
            return None
        # Config gates: features whose per-step logic the replay does not
        # reproduce. All are False/None in the target deploy; any of them
        # being active simply disables the fast path.
        if (self.kv_transfer_criteria is not None
                or self.input_coordinator is not None
                or self.connector is not None
                or getattr(self, "ec_connector", None) is not None
                or self.num_spec_tokens
                or getattr(self, "dynamic_sd_lookup", None) is not None):
            return None
        for r in self.running:
            # decode-ready: exactly one uncomputed token (the one sampled by
            # the previous step), no async placeholders, none of the
            # per-request features the full pass special-cases.
            if (r.num_tokens - r.num_computed_tokens != 1
                    or r.num_output_placeholders
                    or r.use_structured_output
                    or r.lora_request is not None
                    or r.pooling_params is not None
                    or self.current_step + 1 < getattr(r, "next_decode_eligible_step", 0)):
                return None

        # ---- commit ----
        self.current_step += 1
        scheduled_timestamp = _monotonic()
        self.kv_cache_manager.new_step_starts()

        kept = list(self.running)
        num_scheduled_tokens: dict[str, int] = {}
        req_to_new_blocks = {}
        for r in kept:
            new_blocks = self.kv_cache_manager.allocate_slots(
                r, 1, num_lookahead_tokens=self.num_lookahead_tokens)
            if new_blocks is None:
                # Block-pool pressure: the full path owns preemption.
                return None
            req_to_new_blocks[r.request_id] = new_blocks
            num_scheduled_tokens[r.request_id] = 1
            if self.log_stats:
                r.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

        cached_reqs_data = self._make_cached_request_data(
            kept, [], num_scheduled_tokens, {}, req_to_new_blocks)
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies())
        if kv_cache_block_copies:
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)

        num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
            kept[0].request_id)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=len(kept),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            scheduled_encoder_input_stats=None,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            kv_cache_block_copies=kv_cache_block_copies or None,
            num_spec_tokens_to_schedule=self.num_spec_tokens,
        )
        if self.defer_block_free:
            self.sched_step_seq += 1
        self._update_after_schedule(scheduler_output)
        # Evidence counter: without this line there is no way to verify the
        # fast path is actually taken in a live run.
        self._replay_steps = getattr(self, "_replay_steps", 0) + 1
        if self._replay_steps % 500 == 1:
            logger.info("[OmniARScheduler] stage %s cohort-replay steps=%d (batch=%d)",
                        self.vllm_config.model_config.stage_id, self._replay_steps, len(kept))
        return scheduler_output

    def _check_for_wedged_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Dump state once if a stage stops making PROGRESS while requests are tracked.

        A stage that CRASHES leaves a traceback. A stage that quietly stops leaves nothing at
        all, and that is the observed failure mode of a long streaming session: the client
        receives a turn's text and then no audio, forever, while stage 0 keeps generating.

        PROGRESS, not "anything scheduled". The first version of this keyed on
        `total_num_scheduled_tokens == 0` and did not fire on a reproduction that stalled for
        126s -- comfortably past its 45s threshold -- which says the stalled stage was still
        being handed work every pass and simply never produced anything from it. Silence in the
        log is not evidence either way, because a healthy stage 1 logs nothing per step; that
        only looked like evidence in an earlier run where transfer logging happened to be on.

        So progress is defined per request as the pair (num_computed_tokens, output length),
        and the stage is considered stuck when NO tracked request has moved either number for
        `_WEDGE_REPORT_AFTER_S`. That covers both shapes: nothing scheduled at all, and
        scheduled repeatedly without advancing. Being idle between turns is still fine -- with
        no requests tracked the timer is simply held at the current time.
        """
        now = time()
        if not self.requests:
            # "No tracked requests" is normal at startup and between clients, but a stage that
            # HAD requests and now has none while work is still arriving is its own failure --
            # and it is invisible, because every other branch here needs a request to describe.
            # It came up rolling a session: stage 0 shipped 133 payloads to stage 1 over the
            # 0->1 edge and stage 1 emitted nothing, and no diagnostic could say whether the
            # request was stuck or simply absent. Reported once per transition.
            # REPEATS, rate-limited, rather than reporting once. A latching version of this
            # hid the very answer it was added to find: at a session roll stage 1 went empty,
            # reported once, and then stayed silent for two minutes while 370 payloads piled up
            # on the 0->1 edge -- so "stage 1 emits nothing" looked like a stage with no
            # diagnostics rather than a stage with no request. Staying empty is the finding, so
            # it has to keep saying so.
            if self._had_requests and now - self._empty_reported_t >= self._WEDGE_REPORT_AFTER_S:
                self._empty_reported_t = now
                logger.error(
                    "[OmniARScheduler] stage %s tracks ZERO requests and has for %.0fs, "
                    "having tracked %d before. If payloads are still arriving for this stage "
                    "the request was DROPPED, not stalled -- look upstream of the scheduler.",
                    self.vllm_config.model_config.stage_id,
                    now - self._empty_since if self._empty_since else 0.0,
                    self._had_requests,
                )
            if self._empty_since is None:
                self._empty_since = now
            self._last_progress_t = now
            self._wedge_reported = False
            self._progress_fingerprint = None
            return
        self._had_requests = len(self.requests)
        self._empty_since = None

        # A request that is being scheduled but has never SAMPLED anything is its own failure,
        # distinct from a stalled one, and the fingerprint below cannot tell them apart: it
        # counts num_computed_tokens, which advances during prefill, so a request that prefills
        # over and over looks exactly like one that is decoding. That distinction is the open
        # question after a session roll -- stage 1 holds the request, the fingerprint keeps
        # moving, and nothing is ever put on the 1->2 edge.
        for rid, r in self.requests.items():
            if len(getattr(r, "output_token_ids", ()) or ()):
                self._mute_reported.discard(rid)
                self._req_seen_t.pop(rid, None)
                continue
            if getattr(r, "status", None) == RequestStatus.WAITING_FOR_STREAMING_REQ:
                # Parked between segments is a session request's healthy
                # resting state, and with zero-output appends (section 25) a
                # duplex-fed session legitimately shows no sampled output for
                # the entire inter-turn window. Only a request that is
                # RUNNABLE and silent is wedge-suspect.
                self._mute_reported.discard(rid)
                self._req_seen_t.pop(rid, None)
                continue
            t0 = self._req_seen_t.setdefault(rid, now)
            if now - t0 >= self._WEDGE_REPORT_AFTER_S and rid not in self._mute_reported:
                self._mute_reported.add(rid)
                self._log_request_table(
                    f"req={rid} has been tracked for {now - t0:.0f}s and has sampled ZERO "
                    f"output tokens -- being scheduled but producing nothing"
                )

        fingerprint = tuple(
            sorted(
                (rid, r.num_computed_tokens, len(getattr(r, "output_token_ids", ()) or ()))
                for rid, r in self.requests.items()
            )
        )
        if fingerprint != getattr(self, "_progress_fingerprint", None):
            self._progress_fingerprint = fingerprint
            self._last_progress_t = now
            self._wedge_reported = False
            return
        if getattr(self, "_wedge_reported", False):
            return
        since = now - getattr(self, "_last_progress_t", now)
        if since < self._WEDGE_REPORT_AFTER_S:
            if not hasattr(self, "_last_progress_t"):
                self._last_progress_t = now
            return
        self._wedge_reported = True
        scheduled = getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        self._log_request_table(
            f"no request advanced for {since:.0f}s while {len(self.requests)} are tracked "
            f"(this pass scheduled {scheduled} tokens) -- this stage looks WEDGED, not idle"
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        inter_stage_outputs = getattr(model_runner_output, "inter_stage_outputs", None)
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Pre-process KV extraction acks so that the per-request loop below
        # can see up-to-date active_kv_transfers state and emit kv_ready
        # signals while requests are still alive (before any deferred stop).
        kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    self.active_kv_transfers.discard(req_id)
                    req = self.requests.get(req_id)
                    if req is not None and not req.is_finished():
                        outputs[req.client_index].append(
                            OmniEngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=[],
                                kv_transfer_params={"kv_ready": True},
                            )
                        )
                except Exception:
                    init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is not None:
                # vLLM 0.26: settle the in-flight tokens counted in schedule().
                # Must happen before the skips below — failed-KV-load and
                # already-finished requests were incremented too, and the two
                # readers (allocate_slots, _connector_finished) clamp with
                # max(0, computed - in_flight), so a leaked counter silently
                # freezes sliding-window block freeing.
                request.num_in_flight_tokens -= num_tokens_scheduled
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
            status_before_stop = request.status
            new_logprobs = None
            logprob_validation_failed = False

            # Validate before mutating request token state. A bad runner output
            # is request-local: terminate only this request and keep processing
            # the rest of the batch.
            if (
                generated_token_ids
                and request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
            ):
                try:
                    new_logprobs = _slice_sampled_logprobs(logprobs, req_index, generated_token_ids)
                except SampledLogprobContractError as exc:
                    logger.error("Invalid AR sampled-token logprobs for request %s: %s", req_id, exc)
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = str(exc)
                    request.resumable = False
                    generated_token_ids = []
                    logprob_validation_failed = True

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = logprob_validation_failed
            is_segment_finished = False
            finished = False
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            mm_output = mm_outputs[req_index] if mm_outputs else None
            inter_stage_output = inter_stage_outputs[req_index] if inter_stage_outputs else None
            kv_transfer_params = None
            finish_reason = None
            routed_experts = None

            # THE ZERO-OUTPUT APPEND (section 25). A prefill-only arrival
            # append exists to put its tokens into this stage's KV, nothing
            # else -- but the runner samples one throwaway token in the same
            # forward that completes the prefill (the marker cannot reach the
            # runner: streaming extensions travel as CachedRequestData, which
            # carries no additional_information). So the discard happens HERE:
            # drop the sampled token, pre-set the exact status today's
            # max_tokens=1 append reaches via check_stop, and let the stock
            # stopped-path park the request. Downstream of this branch the
            # segment never existed: no EngineCoreOutput (the API server gets
            # no junk to attribute), no save_async (no boundary to stage 1, no
            # vocoder flush -- the 19-52% GPU splash measured at 128 users).
            # The park itself is deliberately NOT re-implemented: only
            # _handle_stopped_request keeps the streaming-queue drain, the
            # parked-counter increment and the skipped_waiting enqueue atomic.
            prefill_only_parked = False
            if (new_token_ids and not stopped and request.resumable
                    and getattr(request, "omni_prefill_only_segment", False)):
                prefill_only_parked = True
                request.omni_prefill_only_segment = False
                # Async scheduling already added placeholder(s) for the
                # token(s) being dropped; without this decrement the rollback
                # at the stopped-path below would re-prefill the last prompt
                # token (re-sampling the throwaway) and swallow the next
                # segment's first real output.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders = max(
                        0, request.num_output_placeholders - len(new_token_ids))
                # Presence probe (the inert-guard lesson): the park announces
                # itself, so its absence under a duplex load is diagnosable.
                logger.info(
                    "[prefill-only] parked req=%s: %d sampled token(s) discarded, "
                    "segment invisible downstream (computed=%d)",
                    req_id, len(new_token_ids), request.num_computed_tokens,
                )
                new_token_ids = []
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                stopped = True

            # Check for stop and update request status.
            if new_token_ids:
                num_sampled_tokens = len(new_token_ids)
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
                if new_logprobs is not None and len(new_token_ids) < num_sampled_tokens:
                    # A mid-step stop (e.g. spec-decode tokens sampled past
                    # EOS) trims new_token_ids after the validation slice
                    # above; re-slice so the emitted logprob rows stay 1:1
                    # with the emitted tokens, as upstream vLLM does by
                    # slicing after the trim.
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # If criteria returns True, it means we must STOP the request.
            # If criteria returns False, it might have triggered a background
            # transfer (e.g. prefill finished / special token) but continues decoding.
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            if stopped:
                if model_runner_output.routed_experts is not None:
                    routed_experts = omni_routed_experts_for_request(model_runner_output.routed_experts, request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                is_segment_finished = not finished
                if finished:
                    request.resumable = False
                if not finished:
                    # for streaming input request only
                    if self.chunk_transfer_adapter:
                        if self.vllm_config.model_config.stage_id != 0:
                            # Downstream async-chunk stages receive real payloads from the
                            # connector. This update only resumes polling for the next segment.
                            self.chunk_transfer_adapter.segment_finished_requests.discard(request.request_id)
                    outstanding_async_tokens = request.num_output_placeholders
                    if outstanding_async_tokens > 0:
                        # Discard only outputs that are already in flight and
                        # roll back their optimistic computed-token accounting.
                        request.async_tokens_to_discard = outstanding_async_tokens
                        request.num_computed_tokens -= outstanding_async_tokens
                        request.num_output_placeholders = 0
                    request.spec_token_ids = []
                    request._output_token_ids.clear()
                if finished:
                    kv_transfer_params, _ = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if prefill_only_parked:
                # A parked prefill-only segment emits NOTHING and ships
                # NOTHING: skipping the two blocks below IS the fix. The
                # `stopped` flag would otherwise emit a junk EngineCoreOutput
                # (the receipt the API server used to have to attribute and
                # swallow) and save_async would ship the segment boundary
                # that made stage 1 wake and stage 2 flush per append.
                pass
            elif new_token_ids or mm_output is not None or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        multimodal_output=mm_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        is_segment_finished=is_segment_finished,
                        new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id, None),
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if (self.chunk_transfer_adapter is not None
                    and not prefill_only_parked
                    and (inter_stage_output is not None or is_segment_finished or finished)):
                self.chunk_transfer_adapter.save_async(
                    inter_stage_output,
                    request,
                    is_segment_finished,
                )

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup_receiver(
                        request.request_id,
                    )

        # [Omni] Cleanup state for finished requests
        for req in stopped_running_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # Same for preempted
        for req in stopped_preempted_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if scheduler_kv_connector_stats is not None and not scheduler_kv_connector_stats.is_empty():
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        # FIXME: finished_req_ids_dict is unconditionally initialized as
        # defaultdict(set) in __init__ (not gated by include_finished_set).
        # This branch is therefore always eligible once any client_index is
        # populated; revisit when wiring streaming-only / upstream semantics.
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                eco = engine_core_outputs.get(client_index)
                if eco is None:
                    eco = EngineCoreOutputs()
                    engine_core_outputs[client_index] = eco
                emitted = {o.request_id for o in eco.outputs}
                for req_id in finished_set:
                    if req_id not in emitted:
                        eco.outputs.append(EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT))
                eco.finished_requests = finished_set
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        self._capture_omni_connector_output(model_runner_output)

        # Free blocks that were held for transfer (kv_ready and
        # active_kv_transfers updates already done before the per-request loop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            self.active_kv_transfers.discard(req_id)
                            self.pending_stop_after_extraction.discard(req_id)
                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
                except Exception:
                    init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        return engine_core_outputs

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            The Request objects that were aborted. Will not include any that
            were already finished.
        """
        # TODO(yrr): chunk transfer adapter & input_coordinator unified to one
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        # Realign stale ``request.status`` (chunk-transfer-adapter's
        # ``requests_origin_status`` table doesn't follow the
        # ``waiting → running`` admit transition; without this, an abort
        # arriving between admit and the next deque round-trip leaves
        # the request in ``self.running`` with ``status=WAITING`` and
        # upstream ``Scheduler.finish_requests`` silently fails to
        # release the worker's ``input_batch`` slot -- after
        # ``max_num_seqs`` such aborts new requests hang at
        # ``chunks=0``). Only the ``async_chunk`` path triggers the
        # staleness; with ``async_chunk`` disabled this is a cheap O(n)
        # no-op over an already-aligned set, kept unconditional so the
        # abort path stays uniform across configurations. See
        # ``OmniSchedulerMixin._realign_request_status_to_queues`` and
        # #3774 discussion.
        self._realign_request_status_to_queues(request_ids)

        finished = super().finish_requests(request_ids, finished_status)

        # Defensive post-finish purge: belt-and-suspenders to the
        # realignment above. Even after realign + ``super()``, corner
        # cases (mid-transition status, connector cleanups that pop
        # from ``self.requests`` without unwinding ``self.running``)
        # can leave already-finished or untracked entries in
        # ``self.running``. Sweep them now so the worker's
        # ``input_batch`` slot never pins a freed request and starves
        # new admissions. See ``OmniSchedulerMixin._purge_finished_from_running``.
        self._purge_finished_from_running()

        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            for request in finished:
                self._free_input_coordinator_request(request.request_id)
        return finished

    def _handle_stopped_request(self, request: Request) -> bool:
        had_queued = bool(getattr(request, "streaming_queue", None))
        finished = super()._handle_stopped_request(request)
        if _LOG_SEG_CYCLES and not finished:
            logger.info(
                "[SEG-CYCLE] stage=%s rid=%s ev=%s mono=%.6f out=%d",
                self.vllm_config.model_config.stage_id, request.request_id,
                "cont" if had_queued else "park", _monotonic(),
                len(request.output_token_ids),
            )
        return finished

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        if outstanding_async_tokens > 0:
            # Async scheduling may already have sampled the previous
            # segment's next token. Drop that late token instead of
            # appending it to the new streaming segment.
            session.async_tokens_to_discard = 1
            session.num_computed_tokens -= session.num_output_placeholders
            session.num_output_placeholders = 0
            session.spec_token_ids = []
        stage_id = self.vllm_config.model_config.stage_id
        if self.chunk_transfer_adapter and self.chunk_transfer_adapter.receives_chunks:
            self.chunk_transfer_adapter.requests_num_chunks_sent.pop(session.external_req_id, None)
            if stage_id != 0:
                # Downstream async-chunk stages receive real payloads from the
                # connector. This update only resumes polling for the next segment.
                self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
                # Do not replace prompt/additional_information here; the next
                # upstream chunk will populate them in chunk transfer adapter.
                session.arrival_time = update.arrival_time
                session.sampling_params = update.sampling_params
                if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                session.status = RequestStatus.WAITING
                if session in self.skipped_waiting:
                    self.skipped_waiting.remove_requests((session,))
                    self._enqueue_waiting_request(session)

                if self.log_stats:
                    session.record_event(EngineCoreEventType.QUEUED)
                return
        update_infos = (
            getattr(update, "model_intermediate_buffer", None),
            getattr(update, "additional_information", None),
        )
        replace_streaming_prompt = any(
            isinstance(info, dict)
            and isinstance(info.get("meta"), dict)
            and info["meta"].get("replace_streaming_prompt") is True
            for info in update_infos
        )
        if replace_streaming_prompt:
            self._replace_streaming_session(session, update)
            return
        super()._update_request_as_session(session, update)
        # [skipped-waiting requeue] Upstream just flipped the parked session's
        # status to WAITING -- but when this call came from add_request (a new
        # segment arriving for a PARKED session), the request object is still
        # sitting in skipped_waiting among every other session's blocked
        # parked requests, and the scheduler reaches it there by luck. Probe
        # measurement: a 15-token segment sat 12 s between admission and
        # execution while the stage was otherwise fresh -- the entire residual
        # p99/max outlier family (5-13 s turn starts) of the burst study. The
        # downstream branch above has always re-enqueued on wake; the stage-0
        # path was missing the same dance. _enqueue_waiting_request routes by
        # status, so a genuinely still-blocked request lands back in
        # skipped_waiting and nothing changes for it. On the
        # _handle_stopped_request path the session is in neither queue and
        # this is a no-op (the caller enqueues right after).
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
            self._enqueue_waiting_request(session)
        # Apply the update's max_tokens. Upstream carries it on every StreamingUpdate and
        # never applies it -- `Request.max_tokens` keeps the FIRST chunk's value for the
        # whole session. The stop check compares per-segment output counts (upstream
        # clears `_output_token_ids` in the update above) against that stale cap, so a
        # per-chunk max_tokens is silently ignored. Measured consequence: a prefill-only
        # append submitted with max_tokens=1 generated ~20 tokens -- a whole unasked reply,
        # withheld from the talker but burned on the thinker and left in its context.
        # For ordinary turns every chunk carries the same value, so this is a no-op there.
        update_max_tokens = getattr(update, "max_tokens", None)
        if isinstance(update_max_tokens, int) and update_max_tokens > 0:
            session.max_tokens = update_max_tokens

        # [live-vllm P2] A new streaming chunk means a live user is attached
        # to this request: whatever priority class the FIRST chunk carried
        # (a shadow's seed is "background") no longer describes it. Clear the
        # marker so the express lane can never park a request a user is
        # actually waiting on -- the swap turn is exactly this transition.
        _sp = getattr(session, "sampling_params", None)
        _ea = getattr(_sp, "extra_args", None) if _sp is not None else None
        if _ea and _SLACK_CLASS_KEY in _ea:
            _ea.pop(_SLACK_CLASS_KEY, None)

        if _LOG_SEG_CYCLES:
            logger.info(
                "[SEG-CYCLE] stage=%s rid=%s ev=wake mono=%.6f new_toks=%d",
                self.vllm_config.model_config.stage_id, req_id, _monotonic(),
                len(update.prompt_token_ids),
            )
        # Per-UPDATE prefill-only capture (the zero-output append, section 25).
        # Marked segments are discarded at sampling time in update_from_output;
        # the flag is one-shot per segment: set here for the chunk that carried
        # the marker, overwritten here by every later chunk, and cleared by the
        # park itself. Assigned unconditionally so a turn following an append
        # can never inherit a stale True.
        session.omni_prefill_only_segment = _update_is_prefill_only(update)

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        # Say WHY, once per request. A request leaving the scheduler is the moment that decides
        # whether a stage goes quiet, and until this line existed the reason was unrecoverable
        # after the fact: rolling a session admitted the new request on stage 1 and then dropped
        # it within a pass or two, and the only visible trace was the stage reporting zero
        # tracked requests for the next 126s while payloads piled up on the 0->1 edge. The
        # status distinguishes an abort from a stop from a length cap, and those have nothing to
        # do with each other.
        logger.info(
            "[OmniARScheduler] stage %s FREE req=%s status=%s prompt_tokens=%s output=%d "
            "computed=%s",
            self.vllm_config.model_config.stage_id,
            request.request_id,
            getattr(getattr(request, "status", None), "name", "?"),
            getattr(request, "num_prompt_tokens", "?"),
            len(getattr(request, "output_token_ids", ()) or ()),
            getattr(request, "num_computed_tokens", "?"),
        )

        self._omits_kv_transfer_cache.pop(request.request_id, None)
        self.temporal_pacer.on_request_freed(request.request_id)

        # [Upstream compat] Discard request from in-flight prefills set added
        # upstream for routed-experts in-flight reservation tracking.
        # Use getattr for safety with test __new__ code paths.
        getattr(self, "_inflight_prefills", set()).discard(request)

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # Mirror the generation scheduler's try/finally pattern so the
        # input_coordinator entry is always pruned along every return path,
        # including the early returns for in-flight / waiting KV transfers
        # below. _free_input_coordinator_request is a no-op when the
        # coordinator is None, so the unconditional finally is safe.
        try:
            # 2. Omni Specific: Check if we need to transfer KV
            if self._should_transfer_kv_for_request(request_id):
                already_triggered = request_id in self.transfer_triggered_requests
                is_active = request_id in self.active_kv_transfers

                if already_triggered:
                    if is_active:
                        # It triggered but hasn't finished yet. We MUST wait.
                        logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                        self.waiting_for_transfer_free.add(request_id)
                        kv_xfer_params = None
                        return kv_xfer_params, None
                    elif request_id in self.waiting_for_transfer_free:
                        # Blocks held until KV extraction completes in a future step.
                        return None, None
                    else:
                        logger.debug(
                            f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                            "Freeing immediately."
                        )
                else:
                    self.waiting_for_transfer_free.add(request_id)
                    confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                    self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                    # Return KV transfer metadata so it propagates to RequestOutput
                    if request_id in self.requests_needing_kv_transfer:
                        transfer_data = self.requests_needing_kv_transfer[request_id]
                        kv_xfer_params = {
                            "past_key_values": transfer_data["block_ids"],
                            "kv_metadata": {
                                "seq_len": transfer_data["seq_len"],
                                "block_ids": transfer_data["block_ids"],
                            },
                        }
                        # Also update request.additional_information for good measure
                        add_info = getattr(request, "additional_information", None)
                        # If additional_information is an AdditionalInformationPayload-like object,
                        # unpack it into a plain dict.
                        if (
                            add_info is not None
                            and hasattr(add_info, "entries")
                            and isinstance(getattr(add_info, "entries"), dict)
                        ):
                            request.additional_information = deserialize_additional_information(add_info)
                            add_info = request.additional_information
                        if add_info is None:
                            request.additional_information = {}
                            add_info = request.additional_information
                        if isinstance(add_info, dict):
                            add_info.update(kv_xfer_params)

                    return kv_xfer_params, None

            # 3. Standard Freeing
            delay_free_blocks |= connector_delay_free_blocks
            # [live-vllm W2] a teardown landing inside the decode zone must
            # not spend the heartbeat's time walking block tables (measured:
            # teardown seconds carried 1.8x the slip rate of event-free wave
            # seconds). Park the request; the slack window drains it within
            # the same tick.
            if not delay_free_blocks and self._in_decode_zone():
                if not hasattr(self, "_deferred_block_frees"):
                    self._deferred_block_frees = []
                self._deferred_block_frees.append(request)
                delay_free_blocks = True
            if not delay_free_blocks:
                self._free_blocks(request)

            return kv_xfer_params, None
        finally:
            self._free_input_coordinator_request(request_id)
            # Normal completion runs through here, not finish_requests()
            # (the abort path) -- see vllm-project/vllm-omni#5349.
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.cleanup_receiver(request_id)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        if not need_send:
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def _has_requests_awaiting_chunk(self) -> bool:
        """True while the chunk transfer adapter is holding a request out of the queues.

        This has to count as work, or the engine deadlocks. The adapter takes a request OUT of
        both `waiting` and `running` while its payload load is in flight, and upstream's
        `has_requests()` only looks at those two queues -- so the loop concludes it has nothing
        to do and blocks. But noticing that the load has COMPLETED happens in
        `process_pending_chunks`, which only runs from `schedule()`, which only runs if the loop
        does not block. Nothing breaks the cycle, because the only thing that wakes the loop is
        new client input.

        Measured, rolling a session: after the new request was admitted, stage 1 logged not one
        scheduler heartbeat for 125 seconds -- while 274 payloads arrived for it on the 0->1
        edge -- and was finally freed with num_computed_tokens still 0. Ordinary turns escape
        this only because each turn's own `add_request` happens to wake the loop; the first turn
        of a rolled request has no such event, since the client is waiting on that very turn.
        """
        adapter = self.chunk_transfer_adapter
        if adapter is None:
            return False
        # `_finished_load_reqs` is the load-bearing one and it is NOT interchangeable with the
        # deques. A COMPLETED load lives there, and only `_process_chunk_queue*` -- reached from
        # `schedule()` -- moves the request back into a queue and consumes it. If the loop parks
        # while that set is non-empty, the payload is already in hand and nothing will ever pick
        # it up.
        #
        # It is also the field that separates a deadlock from healthy idling, which matters
        # because the two look nearly identical. Between turns a session legitimately sits with
        # `origin_status=1` and the queues empty, and the loop SHOULD park until the client
        # speaks again -- keying on `origin_status` would spin a core forever. Measured, stage 1,
        # same run: at 11:45:30 (healthy, parked between turns) finished_load=0; at 11:55:29
        # (deadlocked, turn never completed) finished_load=1 with every queue empty.
        return bool(
            getattr(adapter, "_finished_load_reqs", None)
            or getattr(adapter, "waiting_for_chunk_waiting_requests", None)
            or getattr(adapter, "waiting_for_chunk_running_requests", None)
        )

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        # ... and for requests the chunk transfer adapter is holding. Same reasoning as the KV
        # transfer check above: work is pending, so the loop must not be allowed to quiesce.
        if self._has_requests_awaiting_chunk():
            return True
        result = super().has_requests()
        if not result and self.requests:
            self._report_starved_loop()
        return result

    def _recover_orphaned_requests(self) -> None:
        """Re-enqueue a tracked, unfinished request that belongs to no queue and no holder.

        A request in `self.requests` that is not finished must be in `waiting`,
        `skipped_waiting`, `running`, or held deliberately by the chunk transfer adapter. Being
        in none of them means nothing will ever schedule it and nothing will ever restore it: the
        session stops mid-turn and the client waits out its timeout.

        Measured, stage 1, turn 17 of a 40-turn session, with the engine loop still stepping
        (heartbeats continued, so this is not the parked-loop failure):

            tracked but in NO queue: 1
              status=WAITING num_tokens=40670 computed=40669   (one token short)
            adapter: _finished_load_reqs 1[...]  requests_origin_status 1[...]
                     waiting_for_chunk_waiting_requests 0   waiting_for_chunk_running_requests 0

        `restore_queues` had already returned it to `waiting` and cleared its deques -- the
        orphaned `requests_origin_status` entry is what proves it passed through -- and something
        then removed it from `waiting` without placing it anywhere. Which path does that is still
        unknown, so this repairs the INVARIANT rather than the cause, and says so loudly. The
        guard is narrow on purpose: a request the adapter is genuinely holding sits in one of its
        deques or in `_held_non_active`, and is left alone.
        """
        if not self.requests:
            return
        queued: set[int] = set()
        for container in (self.waiting, self.skipped_waiting, self.running):
            for request in container:
                queued.add(id(request))
        adapter = self.chunk_transfer_adapter
        if adapter is not None:
            for name in ("waiting_for_chunk_waiting_requests",
                         "waiting_for_chunk_running_requests", "_held_non_active"):
                for request in getattr(adapter, name, ()) or ():
                    queued.add(id(request))

        for request in list(self.requests.values()):
            if id(request) in queued:
                continue
            if getattr(request, "is_finished", None) and request.is_finished():
                continue
            logger.error(
                "[OmniARScheduler] stage %s recovering ORPHANED req=%s status=%s "
                "num_tokens=%s computed=%s -- tracked but in no queue and held by nothing, so "
                "nothing would ever schedule it. Re-enqueueing to waiting.",
                self.vllm_config.model_config.stage_id, request.request_id,
                getattr(getattr(request, "status", None), "name", "?"),
                getattr(request, "num_tokens", "?"),
                getattr(request, "num_computed_tokens", "?"),
            )
            if adapter is not None:
                # Drop the stale bookkeeping that says the adapter is holding it, or the next
                # pass will believe the request is parked when it is not.
                getattr(adapter, "requests_origin_status", {}).pop(request.request_id, None)
            if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                # A PARKED session found in no queue is a real leak, but
                # forcing it to WAITING is an engine-killer: it is fully
                # computed, and the waiting path asserts num_new_tokens > 0.
                # Re-enqueue it AS parked (the enqueue routes blocked statuses
                # to skipped_waiting) and leave the counter alone -- it was
                # incremented at the park and never decremented.
                self._enqueue_waiting_request(request)
                continue
            request.status = RequestStatus.WAITING
            self._enqueue_waiting_request(request)

    def _clamp_streaming_parked_counter(self) -> None:
        """Stop a leaked `num_waiting_for_streaming_input` from closing admission for good.

        The same leak `get_num_unfinished_requests` works around has a SECOND consumer, and
        deriving the count there does nothing for this one. Upstream's waiting loop gates
        admission on

            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs: break

        so once the leaked counter reaches `max_num_seqs` the loop breaks on its first pass
        forever. Nothing is admitted, nothing runs, and -- unlike the parked-loop failure --
        `schedule()` keeps being called, so the heartbeat keeps printing and the stage looks
        alive. Measured on stage 1 with `max_num_seqs: 4`: the counter climbed 1, 2, 3, 4 across
        four browser sessions and the FIFTH session got text from stage 0 and never one audio
        token, ending in `has been tracked for 45s and has sampled ZERO output tokens`. The
        engine survived exactly `max_num_seqs` sessions. Read as a client bug it is invisible:
        the reply arrives, only the voice is missing.

        The check is against `self.requests`, deliberately, not against the two queues. Clamping
        to a queue-derived count was tried in `get_num_unfinished_requests` and made things
        worse -- the chunk transfer adapter holds a parked request OUT of both queues, so the
        queue-derived count is legitimately 0 there, and zeroing the counter left upstream to
        decrement it to -1. Every tracked request is in `self.requests` whoever is holding it,
        which makes this bound the true one.

        Only ever clamps DOWN. Leaking is the failure that has been observed; a counter that is
        too LOW would mean a park this scheduler never saw, and inventing slots to cover that
        would hide it.
        """
        parked = sum(
            1 for request in self.requests.values()
            if getattr(request, "status", None) == RequestStatus.WAITING_FOR_STREAMING_REQ
        )
        if self.num_waiting_for_streaming_input <= parked:
            return
        leaked = self.num_waiting_for_streaming_input - parked
        self.num_waiting_for_streaming_input = parked
        now = time()
        if now - self._counter_clamped_t >= self._STARVED_REPORT_EVERY_S:
            self._counter_clamped_t = now
            logger.warning(
                "[OmniARScheduler] stage %s streaming-parked counter had leaked %d slot(s) "
                "(was %d, actually parked %d of %d tracked); clamped. At max_num_seqs=%d a leak "
                "of that size closes admission permanently.",
                self.vllm_config.model_config.stage_id,
                leaked, parked + leaked, parked, len(self.requests),
                self.max_num_running_reqs,
            )

    def get_num_unfinished_requests(self) -> int:
        """Derive the streaming-parked count from the queues instead of trusting a counter.

        Upstream computes

            num_waiting = len(waiting) + len(skipped_waiting) - num_waiting_for_streaming_input

        and `num_waiting_for_streaming_input` is a hand-maintained counter: incremented when a
        request parks for streaming input, decremented when a parked request receives an update
        or is finished while still parked. Every decrement is guarded by
        `status == WAITING_FOR_STREAMING_REQ`, so any path that changes the status FIRST and
        removes the request afterwards leaks the counter permanently.

        Retiring a session request is exactly such a path, and the leak is fatal rather than
        cosmetic. One leaked unit cancels one real request: with the counter stuck at 1 a
        freshly admitted request sitting in `waiting` gives 1 + 0 - 1 = 0, the engine core
        concludes it has no work, and its busy loop parks. Nothing then runs `schedule()`, so the
        payloads arriving for that request are never consumed and it is never prefilled --
        measured as 0 scheduler heartbeats for 125s while 274 payloads arrived on the 0->1 edge,
        and the request finally freed with num_computed_tokens == 0.

        Counting the parked requests directly cannot leak, so the disagreement is repaired here
        rather than chased through every path that might cause it. The repair is logged, because
        a silent correction would hide a real upstream defect.
        """
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)

        num_waiting = 0
        parked = 0
        for queue in (self.waiting, self.skipped_waiting):
            for request in queue:
                if getattr(request, "status", None) == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    parked += 1
                else:
                    num_waiting += 1

        # The counter is NOT written back. An earlier version of this repaired it to the derived
        # value, and that made things worse: while the chunk adapter is holding a parked request
        # OUT of both queues the derived count is legitimately 0, so the repair zeroed a counter
        # that upstream then decremented anyway, leaving it at -1. Reporting the disagreement is
        # useful; mutating another component's bookkeeping from a read-only accessor is not.
        if parked != self.num_waiting_for_streaming_input:
            now = time()
            if now - self._counter_repaired_t >= self._STARVED_REPORT_EVERY_S:
                self._counter_repaired_t = now
                logger.warning(
                    "[OmniARScheduler] stage %s num_waiting_for_streaming_input is %d but %d "
                    "request(s) in the queues are actually parked for streaming input. Using the "
                    "derived count. waiting=%d skipped_waiting=%d running=%d tracked=%d",
                    self.vllm_config.model_config.stage_id,
                    self.num_waiting_for_streaming_input, parked,
                    len(self.waiting), len(self.skipped_waiting), len(self.running),
                    len(self.requests),
                )
        return num_waiting + len(self.running)

    def _report_starved_loop(self) -> None:
        """Say so when this scheduler tells the engine loop there is nothing to do while it is
        still tracking requests, and name every container that could be holding one.

        THE ONLY USEFUL OBSERVATION POINT for this failure. `has_work()` in the engine core is
        `engines_running or scheduler.has_requests() or batch_queue`, so this method is the
        decision that parks the loop -- and once parked, nothing inside `schedule()` runs, which
        is where every other diagnostic in this class lives. A heartbeat at the top of
        `schedule()` proved the loop parks (0 beats in 125s while 274 payloads arrived) but by
        construction could not show WHERE the request was, because it cannot run either.

        Rate-limited: this is called on every loop iteration.
        """
        now = time()
        if now - self._starved_reported_t < self._STARVED_REPORT_EVERY_S:
            return
        self._starved_reported_t = now
        adapter = self.chunk_transfer_adapter

        def n(obj: Any) -> Any:
            try:
                return len(obj)
            except TypeError:
                return "?"

        logger.error(
            "[OmniARScheduler] stage %s is telling the engine loop THERE IS NO WORK while "
            "tracking %d request(s) -- the loop will park. waiting=%s skipped_waiting=%s "
            "running=%s | adapter: wait_chunk_waiting=%s wait_chunk_running=%s ready_chunks=%s "
            "finished_load=%s held_non_active=%s active_streams=%s segment_finished=%s "
            "origin_status=%s | streaming_parked_counter=%s | statuses=%s",
            self.vllm_config.model_config.stage_id, len(self.requests),
            n(self.waiting), n(self.skipped_waiting), n(self.running),
            n(getattr(adapter, "waiting_for_chunk_waiting_requests", None)) if adapter else "-",
            n(getattr(adapter, "waiting_for_chunk_running_requests", None)) if adapter else "-",
            n(getattr(adapter, "requests_with_ready_chunks", None)) if adapter else "-",
            n(getattr(adapter, "_finished_load_reqs", None)) if adapter else "-",
            n(getattr(adapter, "_held_non_active", None)) if adapter else "-",
            n(getattr(adapter, "_active_streams", None)) if adapter else "-",
            n(getattr(adapter, "segment_finished_requests", None)) if adapter else "-",
            n(getattr(adapter, "requests_origin_status", None)) if adapter else "-",
            self.num_waiting_for_streaming_input,
            [getattr(getattr(r, "status", None), "name", "?") for r in self.requests.values()],
        )

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        # A request awaiting a chunk payload is unfinished by any reading, and the loop must keep
        # stepping or it will never observe the payload arriving. See _has_requests_awaiting_chunk.
        if self._has_requests_awaiting_chunk():
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""
