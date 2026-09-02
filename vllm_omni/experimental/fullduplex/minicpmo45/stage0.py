from __future__ import annotations

import base64
import copy
import hashlib
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy

logger = init_logger(__name__)

_MINICPMO45_SPECIAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS
_MINICPMO45_OPTIONAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS
_MINICPMO45_PROCESSOR_LOAD_LOCK = Lock()
_MINICPMO45_LOG_PREP_DIAG = os.environ.get("MINICPMO45_LOG_PREP_DIAG", "0") not in ("0", "", "false", "False")
_DEFAULT_MAX_ARRIVAL_VISION_CACHE_ENTRIES = 0
_MINICPMO45_MAX_RETIRED_VISION_KEYS = 256
_MINICPMO45_MAX_RETIRED_VISION_SESSIONS = 4_096
_DEFAULT_MAX_ARRIVAL_AUDIO_CACHE_ENTRIES = 0
_MINICPMO45_MAX_RETIRED_AUDIO_KEYS = 512
_MINICPMO45_MAX_RETIRED_AUDIO_SESSIONS = 4_096


def _max_arrival_vision_cache_entries() -> int | None:
    """Return the embedding-cache limit, or ``None`` for capacity runs."""
    raw = os.environ.get(
        "MINICPMO45_MAX_ARRIVAL_VISION_CACHE_ENTRIES",
        str(_DEFAULT_MAX_ARRIVAL_VISION_CACHE_ENTRIES),
    )
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def _max_arrival_audio_cache_entries() -> int | None:
    """Return the encoded-audio result limit, or ``None`` when unbounded."""
    raw = os.environ.get(
        "MINICPMO45_MAX_ARRIVAL_AUDIO_CACHE_ENTRIES",
        str(_DEFAULT_MAX_ARRIVAL_AUDIO_CACHE_ENTRIES),
    )
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


@dataclass
class _MiniCPMO45Stage0SessionState:
    session_id: str
    streaming_processor: Any | None = None
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_chunk_idx: int = 0
    context_embeds: list[Any] = field(default_factory=list)
    context_token_ids: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    prepared_append_identity: tuple[int | None, int] | None = None
    prepared_inputs_embeds: Any | None = None
    prepared_input_token_ids: list[int] = field(default_factory=list)
    # Bounded references used only if vLLM preempts the live P request.  The
    # steady path must not concatenate/materialize a second prompt tensor.
    prepared_rebase_prefix_embeds: tuple[Any, ...] = ()
    prepared_rebase_prefix_token_ids: tuple[int, ...] = ()
    prepared_result: dict[str, Any] = field(default_factory=dict)
    audio_past_key_values: Any | None = None
    # Once a formal append consumes an arrival-side audio result, the
    # auxiliary GPU owns the Whisper/Mel lineage. A later cache miss must fail
    # closed instead of silently continuing from this intentionally incomplete
    # request-local audio state.
    audio_sidecar_active: bool = False
    pending_terminator_token: int | None = None
    last_terminator_token: int | None = None
    pending_speech_context: bool = False
    pending_speech_append_identity: tuple[int | None, int] | None = None
    pending_speech_response_open: bool = False
    generated_tokens: list[int] = field(default_factory=list)
    current_segment_output_tokens: list[int] = field(default_factory=list)
    pd_feedback_token_ids: list[int] = field(default_factory=list)
    pd_feedback_append_identity: tuple[int | None, int] | None = None
    last_unit_inputs_embeds: Any | None = None
    last_unit_input_token_ids: list[int] = field(default_factory=list)
    context_rollovers: int = 0


@dataclass
class _MiniCPMO45PreparedAudioUnit:
    """One processor-exact audio unit prepared ahead of per-request input assembly."""

    chunk_idx: int
    batch_feature: Any
    consumed_samples: int
    audio_embeds: Any | None = None


@dataclass
class _MiniCPMO45PreparedAudioAppend:
    """Transactional result of parallel CPU audio preparation.

    The live session processor is restored immediately after preparation.  Its
    post-append snapshot and the session fields are committed only when the
    normal per-request preprocess path consumes this object successfully.
    """

    start_chunk_idx: int
    start_buffer_len: int
    units: list[_MiniCPMO45PreparedAudioUnit]
    remaining_audio_buffer: np.ndarray
    mel_snapshot_after: Any
    feature_ms: float = 0.0
    audio_past_key_values: Any | None = None
    encoded: bool = False
    arrival_preencoded: bool = False
    arrival_preencode_seq: int | None = None
    arrival_preencode_id: str | None = None


@dataclass
class _MiniCPMO45ArrivalAudioLineage:
    """GPU2-owned streaming-audio state independent of formal P progress."""

    epoch: int
    next_seq: int
    state: _MiniCPMO45Stage0SessionState


class MiniCPMO45Stage0DuplexRuntime:
    """Build scheduler-owned MiniCPM-o 4.5 Stage0 duplex inputs."""

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str = "cuda") -> None:
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.sessions: dict[tuple[str, int], _MiniCPMO45Stage0SessionState] = {}
        # Session instructions and reference voice are commonly identical for
        # every connection. Their token/voice embeddings are immutable model
        # outputs, so retaining a small process-local LRU avoids re-running the
        # same reference-audio encoder and many one-token embedding launches at
        # the first unit of every session.
        self._session_context_cache: OrderedDict[
            tuple[str | None, tuple[int, ...] | None, bytes | None],
            tuple[tuple[Any, ...], tuple[int, ...]],
        ] = OrderedDict()
        self._arrival_vision_cache: OrderedDict[tuple[str, int, int, str], list[Any]] = OrderedDict()
        self._arrival_vision_cache_pending: dict[tuple[str, int, int, str], object] = {}
        self._arrival_vision_retired: OrderedDict[tuple[str, int, int, str], None] = OrderedDict()
        self._arrival_vision_retired_sessions: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._arrival_vision_cache_lock = Lock()
        self._vision_execution_lock = RLock()
        self._arrival_audio_cache: OrderedDict[
            tuple[str, int, int, int, str],
            _MiniCPMO45PreparedAudioAppend,
        ] = OrderedDict()
        self._arrival_audio_lineages: dict[
            tuple[str, int],
            _MiniCPMO45ArrivalAudioLineage,
        ] = {}
        self._arrival_audio_retired: OrderedDict[
            tuple[str, int, int, int, str],
            None,
        ] = OrderedDict()
        self._arrival_audio_retired_sessions: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._arrival_audio_cache_lock = Lock()
        self._audio_execution_lock = RLock()
        self._audio_prepare_executor: ThreadPoolExecutor | None = None
        self.thinker = getattr(stage_model, "thinker", None) or getattr(stage_model, "model", None) or stage_model
        self.processor = (
            getattr(stage_model, "processor", None)
            or getattr(self.thinker, "processor", None)
            or self._load_processor_from_path(model_path)
        )
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None)
            if self.processor is not None
            else getattr(stage_model, "tokenizer", None)
        )
        self._init_token_ids()
        if self.tokenizer is not None:
            self._require_special_token_ids()

    def _stage_runtime_ready(self) -> bool:
        return self.processor is not None and self.tokenizer is not None and self.thinker is not None

    def _configure_streaming_processor(
        self,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        processor = self.processor
        if processor is None:
            return None
        if state is not None:
            if state.streaming_processor is not None:
                return state.streaming_processor
            processor = copy.copy(processor)
            shared_mel = getattr(self.processor, "_streaming_mel_processor", None)
            if shared_mel is not None:
                processor._streaming_mel_processor = copy.deepcopy(shared_mel)
            state.streaming_processor = processor
        if processor is None:
            return
        set_streaming_mode = getattr(processor, "set_streaming_mode", None)
        if callable(set_streaming_mode):
            set_streaming_mode(
                mode="exact",
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                first_chunk_ms=int(self._stage_param("first_chunk_ms", 1035)),
                cnn_redundancy_ms=int(self._stage_param("cnn_redundancy_ms", 20)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            # Match official init_streaming_processor: reset the streaming mel-processor
            # buffers at session init (modeling_minicpmo_unified.py:207).
            reset_streaming = getattr(processor, "reset_streaming", None)
            if callable(reset_streaming):
                reset_streaming()
            return processor
        configure_streaming = getattr(processor, "configure_streaming", None)
        if callable(configure_streaming):
            configure_streaming(
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
        return processor

    def _prepare_session_context(
        self,
        state: _MiniCPMO45Stage0SessionState,
        session_config: dict[str, Any],
        *,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        if not self._stage_runtime_ready():
            return
        self._require_special_token_ids()
        ref_audio = self._decode_ref_audio_from_session_config(runtime_config or {})
        context_cache_key = self._session_context_cache_key(
            session_config.get("instructions"),
            ref_audio,
        )
        if context_cache_key is not None:
            cached = self._session_context_cache.get(context_cache_key)
            if cached is not None:
                cached_embeds, cached_token_ids = cached
                state.context_embeds.extend(cached_embeds)
                state.context_token_ids.extend(cached_token_ids)
                self._session_context_cache.move_to_end(context_cache_key)
                return
        # Matches MiniCPMODuplex.prepare() in the released checkpoint's
        # modeling_minicpmo.py: the <|audio_start|>/<|audio_end|> markers are
        # only present when reference audio is embedded between them. The
        # template is shared with the serving adapter so the first-append
        # scheduler reserve can count these tokens exactly.
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            session_config.get("instructions"),
            ref_audio is not None,
        )
        for token_id in self._encode_text(prefix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        if ref_audio is not None:
            ref_audio_embeds = self._stage_ref_audio_embeddings(ref_audio, state=state)
            if ref_audio_embeds is not None:
                ref_audio_embeds = self._as_2d_tensor(ref_audio_embeds)
                state.context_embeds.append(ref_audio_embeds)
                state.context_token_ids.extend([self.unit_token_id] * int(ref_audio_embeds.shape[0]))
        for token_id in self._encode_text(suffix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        if context_cache_key is not None:
            self._session_context_cache[context_cache_key] = (
                tuple(state.context_embeds),
                tuple(state.context_token_ids),
            )
            self._session_context_cache.move_to_end(context_cache_key)
            while len(self._session_context_cache) > 16:
                self._session_context_cache.popitem(last=False)

    def _session_context_cache_key(
        self,
        instructions: Any,
        ref_audio: Any | None,
    ) -> tuple[str | None, tuple[int, ...] | None, bytes | None] | None:
        instruction_key = None if instructions is None else str(instructions)
        if ref_audio is None:
            return instruction_key, None, None

        # A streaming-only fallback can mutate a session's Mel/cache lineage;
        # that result is not shareable. The released MiniCPM-o checkpoint has
        # a stateless reference-audio encoder, which is safe to cache.
        process_audio = getattr(self.processor, "process_audio", None)
        has_stateless_encoder = any(
            callable(getattr(target, name, None))
            for target in (self.stage_model, self.thinker)
            for name in ("get_audio_embedding", "get_audio_hidden_states")
        )
        if not callable(process_audio) or not has_stateless_encoder:
            return None
        waveform = np.ascontiguousarray(ref_audio, dtype=np.float32)
        digest = hashlib.blake2b(
            memoryview(waveform).cast("B"),
            digest_size=16,
        ).digest()
        return instruction_key, tuple(int(dim) for dim in waveform.shape), digest

    def get_or_create_session_state(
        self,
        session_id: str,
        incarnation: int,
        *,
        session_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> _MiniCPMO45Stage0SessionState:
        """Initialize one session before request-local preprocessing.

        ``preprocess_batch`` calls this early so first-unit audio can join the
        same cross-session batch as later units. The per-request hook calls it
        again as an idempotent fallback.
        """
        session_key = (session_id, int(incarnation))
        state = self.sessions.get(session_key)
        if state is not None:
            return state
        state = _MiniCPMO45Stage0SessionState(session_id=session_id)
        self.sessions[session_key] = state
        if hasattr(self.thinker, "audio_past_key_values"):
            self.thinker.audio_past_key_values = None
        self._configure_streaming_processor(state)
        self._prepare_session_context(
            state,
            dict(session_config or {}),
            runtime_config=dict(runtime_config or {}),
        )
        return state

    def cache_arrival_vision_embeddings(
        self,
        *,
        session_id: str,
        incarnation: int,
        epoch: int,
        preencode_ids: list[str],
        frame_blocks: list[list[Any]],
    ) -> int:
        """Cache frame embeddings without advancing the session's LLM state.

        The encoder returns blocks on the Thinker device. Arrival work can run
        ahead of formal P consumption, so retaining those blocks on GPU would
        turn an unbounded input queue into unbounded GPU memory. Store the
        compact blocks on CPU and move only the consumed frame back to the
        Thinker device on the formal request path.
        """
        if len(preencode_ids) != len(frame_blocks):
            return 0
        cache = getattr(self, "_arrival_vision_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._arrival_vision_cache = cache
        lock = getattr(self, "_arrival_vision_cache_lock", None)
        if lock is None:
            lock = Lock()
            self._arrival_vision_cache_lock = lock
        session_prefix = (str(session_id), int(incarnation))
        pending = getattr(self, "_arrival_vision_cache_pending", None)
        if not isinstance(pending, dict):
            pending = {}
            self._arrival_vision_cache_pending = pending
        retired_sessions = getattr(
            self,
            "_arrival_vision_retired_sessions",
            None,
        )
        if not isinstance(retired_sessions, OrderedDict):
            retired_sessions = OrderedDict()
            self._arrival_vision_retired_sessions = retired_sessions
        retired = getattr(self, "_arrival_vision_retired", None)
        if not isinstance(retired, OrderedDict):
            retired = OrderedDict()
            self._arrival_vision_retired = retired

        reservations: list[tuple[tuple[str, int, int, str], object, list[Any]]] = []
        with lock:
            if session_prefix in retired_sessions:
                return 0
            stale = [key for key in cache if key[:2] == session_prefix and key[2] != int(epoch)]
            for key in stale:
                cache.pop(key, None)
            stale_pending = [key for key in pending if key[:2] == session_prefix and key[2] != int(epoch)]
            for key in stale_pending:
                pending.pop(key, None)
            for preencode_id, blocks in zip(
                preencode_ids,
                frame_blocks,
                strict=True,
            ):
                if not isinstance(preencode_id, str) or not preencode_id:
                    continue
                key = (*session_prefix, int(epoch), preencode_id)
                if key in retired:
                    continue
                reservation = object()
                pending[key] = reservation
                reservations.append((key, reservation, blocks))

        # GPU -> CPU copies may synchronize the producing stream. They must not
        # hold the global cache metadata lock, otherwise one slow copy blocks
        # every session's take/retire operation. Retirement is rechecked after
        # the copies, under the lock, so a late speculative write still cannot
        # resurrect a frame that the formal path has already fenced.
        prepared: list[tuple[tuple[str, int, int, str], object, list[Any]]] = []
        try:
            for key, reservation, blocks in reservations:
                cpu_blocks: list[Any] = []
                for block in blocks:
                    detach = getattr(block, "detach", None)
                    cached = detach() if callable(detach) else block
                    to = getattr(cached, "to", None)
                    if callable(to):
                        cached = to(device="cpu")
                    cpu_blocks.append(cached)
                prepared.append((key, reservation, cpu_blocks))
        except BaseException:
            with lock:
                for key, reservation, _blocks in reservations:
                    if pending.get(key) is reservation:
                        pending.pop(key, None)
            raise

        with lock:
            encoded = 0
            for key, reservation, cpu_blocks in prepared:
                if pending.get(key) is not reservation:
                    continue
                pending.pop(key, None)
                cache[key] = cpu_blocks
                cache.move_to_end(key)
                encoded += 1
            limit = _max_arrival_vision_cache_entries()
            if limit is not None:
                while len(cache) > limit:
                    cache.popitem(last=False)
        return encoded

    def arrival_vision_cache_size(self) -> int:
        cache = getattr(self, "_arrival_vision_cache", None)
        lock = getattr(self, "_arrival_vision_cache_lock", None)
        if not isinstance(cache, OrderedDict) or lock is None:
            return 0
        with lock:
            return len(cache)

    def retire_arrival_vision_embeddings(
        self,
        *,
        session_id: str,
        incarnation: int,
        epoch: int | None,
        preencode_ids: list[str],
    ) -> None:
        """Fence a formal fallback from a late speculative cache write."""
        if epoch is None or not preencode_ids:
            return
        lock = getattr(self, "_arrival_vision_cache_lock", None)
        if lock is None:
            lock = Lock()
            self._arrival_vision_cache_lock = lock
        retired = getattr(self, "_arrival_vision_retired", None)
        if not isinstance(retired, OrderedDict):
            retired = OrderedDict()
            self._arrival_vision_retired = retired
        cache = getattr(self, "_arrival_vision_cache", None)
        pending = getattr(self, "_arrival_vision_cache_pending", None)
        keys = [
            (str(session_id), int(incarnation), int(epoch), preencode_id)
            for preencode_id in preencode_ids
            if isinstance(preencode_id, str) and preencode_id
        ]
        with lock:
            for key in keys:
                if isinstance(cache, OrderedDict):
                    cache.pop(key, None)
                if isinstance(pending, dict):
                    pending.pop(key, None)
                retired[key] = None
                retired.move_to_end(key)
            while len(retired) > _MINICPMO45_MAX_RETIRED_VISION_KEYS:
                retired.popitem(last=False)

    def take_arrival_vision_embeddings(
        self,
        *,
        session_id: str,
        incarnation: int,
        epoch: int | None,
        preencode_ids: list[str],
    ) -> list[list[Any]] | None:
        """Atomically consume a complete arrival-preencoded frame set."""
        if epoch is None or not preencode_ids:
            return None
        cache = getattr(self, "_arrival_vision_cache", None)
        lock = getattr(self, "_arrival_vision_cache_lock", None)
        if not isinstance(cache, OrderedDict) or lock is None:
            return None
        keys = [(str(session_id), int(incarnation), int(epoch), preencode_id) for preencode_id in preencode_ids]
        with lock:
            if any(key not in cache for key in keys):
                return None
            return [list(cache.pop(key)) for key in keys]

    def discard_arrival_vision_session(
        self,
        session_id: str,
        incarnation: int,
    ) -> None:
        cache = getattr(self, "_arrival_vision_cache", None)
        lock = getattr(self, "_arrival_vision_cache_lock", None)
        pending = getattr(self, "_arrival_vision_cache_pending", None)
        retired = getattr(self, "_arrival_vision_retired", None)
        retired_sessions = getattr(
            self,
            "_arrival_vision_retired_sessions",
            None,
        )
        if lock is None:
            return
        prefix = (str(session_id), int(incarnation))
        with lock:
            if isinstance(cache, OrderedDict):
                for key in [key for key in cache if key[:2] == prefix]:
                    cache.pop(key, None)
            if isinstance(pending, dict):
                for key in [key for key in pending if key[:2] == prefix]:
                    pending.pop(key, None)
            if isinstance(retired, OrderedDict):
                for key in [key for key in retired if key[:2] == prefix]:
                    retired.pop(key, None)
            if not isinstance(retired_sessions, OrderedDict):
                retired_sessions = OrderedDict()
                self._arrival_vision_retired_sessions = retired_sessions
            retired_sessions[prefix] = None
            retired_sessions.move_to_end(prefix)
            while len(retired_sessions) > _MINICPMO45_MAX_RETIRED_VISION_SESSIONS:
                retired_sessions.popitem(last=False)

    @staticmethod
    def _arrival_audio_job_identity(
        job: dict[str, object],
    ) -> tuple[str, int, int, int, str] | None:
        session_id = job.get("session_id")
        preencode_id = job.get("audio_preencode_id")
        if not isinstance(session_id, str) or not session_id:
            return None
        if not isinstance(preencode_id, str) or not preencode_id:
            return None
        try:
            incarnation = int(job.get("incarnation", 0))
            epoch = int(job.get("epoch", 0))
            seq = int(job["audio_preencode_seq"])
        except (KeyError, TypeError, ValueError):
            return None
        if seq < 0:
            return None
        return session_id, incarnation, epoch, seq, preencode_id

    def _new_arrival_audio_lineage(
        self,
        *,
        session_id: str,
        epoch: int,
        first_seq: int,
    ) -> _MiniCPMO45ArrivalAudioLineage:
        state = _MiniCPMO45Stage0SessionState(session_id=session_id)
        self._configure_streaming_processor(state)
        return _MiniCPMO45ArrivalAudioLineage(
            epoch=int(epoch),
            next_seq=int(first_seq),
            state=state,
        )

    def _encode_prepared_audio_explicit(
        self,
        plan: _MiniCPMO45PreparedAudioAppend,
        state: _MiniCPMO45Stage0SessionState,
    ) -> bool:
        """Encode one plan without swapping a process-global audio KV field."""
        target = next(
            (
                candidate
                for candidate in (self.stage_model, self.thinker)
                if callable(
                    getattr(
                        candidate,
                        "get_audio_embedding_streaming_batch",
                        None,
                    )
                )
            ),
            None,
        )
        if target is None or not plan.units:
            return False
        cache = state.audio_past_key_values
        try:
            for unit in plan.units:
                outputs, cache = target.get_audio_embedding_streaming_batch(
                    unit.batch_feature,
                    past_key_values=cache,
                    use_extra_context=True,
                    prefix_extra_frames=0 if unit.chunk_idx == 0 else 2,
                    suffix_extra_frames=2,
                    return_on_audio_device=True,
                )
                if len(outputs) != 1:
                    raise ValueError("streaming audio sidecar returned wrong batch size")
                audio_embeds = self._cat_nested_tensors(outputs[0])
                if audio_embeds is None:
                    raise ValueError("streaming audio sidecar returned empty embeddings")
                unit.audio_embeds = audio_embeds
            plan.audio_past_key_values = cache
            plan.encoded = True
            return True
        except Exception:
            logger.exception(
                "MiniCPM-o arrival audio explicit encoding failed for session %s chunk %s",
                state.session_id,
                plan.start_chunk_idx,
            )
            plan.audio_past_key_values = None
            plan.encoded = False
            for unit in plan.units:
                unit.audio_embeds = None
            return False

    def _commit_arrival_audio_plan(
        self,
        *,
        identity: tuple[str, int, int, int, str],
        lineage: _MiniCPMO45ArrivalAudioLineage,
        plan: _MiniCPMO45PreparedAudioAppend,
    ) -> bool:
        """Advance GPU2's lineage and retain only the compact P input rows."""
        if not plan.encoded or not plan.units:
            return False
        next_cache = plan.audio_past_key_values
        if next_cache is None:
            return False

        # A Whisper KV snapshot can be O(100 MiB) per session.  It remains only
        # in the latest GPU2 lineage; each unconsumed slot stores its ~80 KiB
        # pooled embedding on CPU, never a versioned KV copy.
        key = identity
        prefix = key[:2]
        with self._arrival_audio_cache_lock:
            if prefix in self._arrival_audio_retired_sessions:
                return False
            if key in self._arrival_audio_retired:
                return False
        try:
            next_audio_buffer = np.asarray(
                plan.remaining_audio_buffer,
                dtype=np.float32,
            ).copy()
            cpu_embeds = []
            for unit in plan.units:
                block = self._as_2d_tensor(unit.audio_embeds)
                cpu_embeds.append(block.detach().to(device="cpu"))
        except Exception:
            logger.exception(
                "MiniCPM-o arrival audio CPU materialization failed for session %s seq %s",
                identity[0],
                identity[3],
            )
            return False

        state = lineage.state
        processor = self._configure_streaming_processor(state)
        mel_processor = getattr(processor, "_streaming_mel_processor", None)
        get_mel_snapshot = getattr(mel_processor, "get_snapshot", None)
        previous_mel_snapshot = None
        if callable(get_mel_snapshot):
            try:
                previous_mel_snapshot = get_mel_snapshot()
            except Exception:
                logger.exception(
                    "MiniCPM-o arrival audio failed to snapshot Mel state for session %s seq %s",
                    identity[0],
                    identity[3],
                )
                return False

        # Retirement may race the CPU copies above. Hold the metadata lock from
        # the final fence through the lineage/cache commit so a retired job can
        # never advance authoritative streaming state.
        with self._arrival_audio_cache_lock:
            if prefix in self._arrival_audio_retired_sessions:
                return False
            if key in self._arrival_audio_retired:
                return False
            previous_audio_buffer = state.audio_buffer
            previous_audio_chunk_idx = state.audio_chunk_idx
            previous_audio_cache = state.audio_past_key_values
            previous_lineage_seq = lineage.next_seq
            previous_plan_values = (
                plan.audio_past_key_values,
                plan.mel_snapshot_after,
                plan.remaining_audio_buffer,
                plan.arrival_preencoded,
                plan.arrival_preencode_seq,
                plan.arrival_preencode_id,
            )
            previous_unit_values = [
                (unit.audio_embeds, unit.batch_feature)
                for unit in plan.units
            ]
            try:
                if not self._restore_streaming_mel_snapshot(
                    processor,
                    plan.mel_snapshot_after,
                ):
                    return False
                state.audio_buffer = next_audio_buffer
                state.audio_chunk_idx = plan.start_chunk_idx + len(plan.units)
                state.audio_past_key_values = next_cache
                for unit, cpu_embed in zip(plan.units, cpu_embeds, strict=True):
                    unit.audio_embeds = cpu_embed
                    unit.batch_feature = None
                plan.audio_past_key_values = None
                plan.mel_snapshot_after = None
                plan.remaining_audio_buffer = next_audio_buffer.copy()
                plan.arrival_preencoded = True
                plan.arrival_preencode_seq = identity[3]
                plan.arrival_preencode_id = identity[4]
                self._arrival_audio_cache[key] = plan
                self._arrival_audio_cache.move_to_end(key)
                limit = _max_arrival_audio_cache_entries()
                if limit is not None:
                    while len(self._arrival_audio_cache) > limit:
                        self._arrival_audio_cache.popitem(last=False)
                lineage.next_seq += 1
                return True
            except Exception:
                state.audio_buffer = previous_audio_buffer
                state.audio_chunk_idx = previous_audio_chunk_idx
                state.audio_past_key_values = previous_audio_cache
                lineage.next_seq = previous_lineage_seq
                (
                    plan.audio_past_key_values,
                    plan.mel_snapshot_after,
                    plan.remaining_audio_buffer,
                    plan.arrival_preencoded,
                    plan.arrival_preencode_seq,
                    plan.arrival_preencode_id,
                ) = previous_plan_values
                for unit, (audio_embeds, batch_feature) in zip(
                    plan.units,
                    previous_unit_values,
                    strict=True,
                ):
                    unit.audio_embeds = audio_embeds
                    unit.batch_feature = batch_feature
                self._arrival_audio_cache.pop(key, None)
                if previous_mel_snapshot is not None:
                    try:
                        self._restore_streaming_mel_snapshot(
                            processor,
                            previous_mel_snapshot,
                        )
                    except Exception:
                        logger.exception(
                            "MiniCPM-o arrival audio Mel rollback failed for session %s seq %s",
                            identity[0],
                            identity[3],
                        )
                logger.exception(
                    "MiniCPM-o arrival audio commit failed for session %s seq %s",
                    identity[0],
                    identity[3],
                )
                return False

    def preencode_arrival_audio(
        self,
        jobs: list[dict[str, object]],
    ) -> dict[str, object]:
        """Advance ordered streaming-audio lineages on the auxiliary GPU.

        Jobs from different sessions are encoded together when their Whisper
        cache shapes match. Multiple jobs from one session are processed in
        successive waves so a later unit always observes its parent's KV.
        """
        parsed: dict[
            tuple[str, int],
            list[tuple[tuple[str, int, int, int, str], dict[str, object]]],
        ] = {}
        results: dict[str, bool] = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            identity = self._arrival_audio_job_identity(job)
            if identity is None:
                continue
            parsed.setdefault(identity[:2], []).append((identity, job))
            results[identity[4]] = False
        for records in parsed.values():
            records.sort(key=lambda item: item[0][3])

        with self._audio_execution_lock:
            queues: dict[
                tuple[str, int],
                list[tuple[tuple[str, int, int, int, str], dict[str, object]]],
            ] = {}
            for prefix, records in parsed.items():
                first_identity = records[0][0]
                with self._arrival_audio_cache_lock:
                    retired_session = prefix in self._arrival_audio_retired_sessions
                if retired_session:
                    continue
                lineage = self._arrival_audio_lineages.get(prefix)
                if lineage is None or lineage.epoch != first_identity[2]:
                    # An epoch change is an explicit lineage boundary. Old
                    # embeddings and Whisper state cannot update the new one.
                    lineage = self._new_arrival_audio_lineage(
                        session_id=first_identity[0],
                        epoch=first_identity[2],
                        first_seq=first_identity[3],
                    )
                    self._arrival_audio_lineages[prefix] = lineage
                    with self._arrival_audio_cache_lock:
                        stale = [key for key in self._arrival_audio_cache if key[:2] == prefix]
                        for key in stale:
                            self._arrival_audio_cache.pop(key, None)
                queues[prefix] = records

            while queues:
                wave: list[
                    tuple[
                        tuple[str, int, int, int, str],
                        dict[str, object],
                        _MiniCPMO45ArrivalAudioLineage,
                    ]
                ] = []
                for prefix, records in list(queues.items()):
                    if not records:
                        queues.pop(prefix, None)
                        continue
                    identity, job = records[0]
                    lineage = self._arrival_audio_lineages[prefix]
                    # The worker may have committed this exact identity even
                    # when the RPC response was lost or timed out upstream.
                    # Treat the retained plan as the durable acknowledgement
                    # before applying the next-sequence gate.  The original
                    # commit already advanced ``next_seq``; replay must not do
                    # so a second time.
                    with self._arrival_audio_cache_lock:
                        already_cached = identity in self._arrival_audio_cache
                    if already_cached:
                        results[identity[4]] = True
                        records.pop(0)
                        if not records:
                            queues.pop(prefix, None)
                        continue
                    if identity[2] != lineage.epoch or identity[3] != lineage.next_seq:
                        # A gap cannot be reconstructed from later PCM. Leave
                        # this session failed without advancing its lineage.
                        queues.pop(prefix, None)
                        continue
                    wave.append((identity, job, lineage))

                if not wave:
                    break

                def prepare(
                    item: tuple[
                        tuple[str, int, int, int, str],
                        dict[str, object],
                        _MiniCPMO45ArrivalAudioLineage,
                    ],
                ) -> tuple[
                    tuple[str, int, int, int, str],
                    _MiniCPMO45ArrivalAudioLineage,
                    _MiniCPMO45PreparedAudioAppend | None,
                ]:
                    identity, job, lineage = item
                    payload = job.get("payload")
                    if not isinstance(payload, dict):
                        payload = job
                    try:
                        waveform = self._decode_audio_payload(payload)
                        plan = self._prepare_streaming_audio_append(
                            lineage.state,
                            waveform,
                        )
                    except Exception:
                        plan = None
                    return identity, lineage, plan

                executor = self._audio_prepare_executor
                if executor is None:
                    executor = ThreadPoolExecutor(
                        max_workers=8,
                        thread_name_prefix="minicpmo-audio-sidecar-prepare",
                    )
                    self._audio_prepare_executor = executor
                prepared_all = list(executor.map(prepare, wave))
                prepared = [item for item in prepared_all if item[2] is not None]
                for identity, _lineage, plan in prepared_all:
                    if plan is None:
                        queues.pop(identity[:2], None)
                batchable = [(plan, lineage.state) for _identity, lineage, plan in prepared if plan is not None]
                if len(batchable) >= 2:
                    self._stage_audio_embeddings_batch(
                        batchable,
                        return_on_audio_device=True,
                    )
                for identity, lineage, plan in prepared:
                    assert plan is not None
                    if not plan.encoded:
                        self._encode_prepared_audio_explicit(plan, lineage.state)
                    success = self._commit_arrival_audio_plan(
                        identity=identity,
                        lineage=lineage,
                        plan=plan,
                    )
                    results[identity[4]] = success
                    prefix = identity[:2]
                    records = queues.get(prefix)
                    if success and records and records[0][0] == identity:
                        records.pop(0)
                        if not records:
                            queues.pop(prefix, None)
                    elif not success:
                        queues.pop(prefix, None)

        encoded_jobs = sum(1 for success in results.values() if success)
        return {
            "supported": True,
            "encoded_jobs": encoded_jobs,
            "job_results": results,
        }

    def take_arrival_audio_append(
        self,
        *,
        session_id: str,
        incarnation: int,
        epoch: int | None,
        audio_preencode_seq: int | None,
        audio_preencode_id: str,
    ) -> _MiniCPMO45PreparedAudioAppend | None:
        if epoch is None or audio_preencode_seq is None or not audio_preencode_id:
            return None
        key = (
            str(session_id),
            int(incarnation),
            int(epoch),
            int(audio_preencode_seq),
            str(audio_preencode_id),
        )
        with self._arrival_audio_cache_lock:
            return self._arrival_audio_cache.pop(key, None)

    def retire_arrival_audio_append(
        self,
        *,
        session_id: str,
        incarnation: int,
        epoch: int,
        audio_preencode_seq: int,
        audio_preencode_id: str,
    ) -> None:
        key = (
            str(session_id),
            int(incarnation),
            int(epoch),
            int(audio_preencode_seq),
            str(audio_preencode_id),
        )
        with self._arrival_audio_cache_lock:
            self._arrival_audio_cache.pop(key, None)
            self._arrival_audio_retired[key] = None
            self._arrival_audio_retired.move_to_end(key)
            while len(self._arrival_audio_retired) > _MINICPMO45_MAX_RETIRED_AUDIO_KEYS:
                self._arrival_audio_retired.popitem(last=False)

    def discard_arrival_audio_session(
        self,
        session_id: str,
        incarnation: int,
    ) -> None:
        prefix = (str(session_id), int(incarnation))
        # Serialize teardown with the stateful encoder. This is infrequent and
        # prevents a late completion from resurrecting a closed session.
        with self._audio_execution_lock:
            self._arrival_audio_lineages.pop(prefix, None)
            with self._arrival_audio_cache_lock:
                for key in [key for key in self._arrival_audio_cache if key[:2] == prefix]:
                    self._arrival_audio_cache.pop(key, None)
                for key in [key for key in self._arrival_audio_retired if key[:2] == prefix]:
                    self._arrival_audio_retired.pop(key, None)
                self._arrival_audio_retired_sessions[prefix] = None
                self._arrival_audio_retired_sessions.move_to_end(prefix)
                while len(self._arrival_audio_retired_sessions) > _MINICPMO45_MAX_RETIRED_AUDIO_SESSIONS:
                    self._arrival_audio_retired_sessions.popitem(last=False)

    def arrival_audio_cache_size(self) -> int:
        with self._arrival_audio_cache_lock:
            return len(self._arrival_audio_cache)

    def _stage_prefill_embeddings_only(
        self,
        state: _MiniCPMO45Stage0SessionState,
        audio_waveform: Any,
        *,
        video_frames: list[Any] | None = None,
        max_slice_nums: int | list[int] = 1,
        preprocessed_vision: Any | None = None,
        preencoded_vision: list[list[Any]] | None = None,
        preprocessed_audio: _MiniCPMO45PreparedAudioAppend | None = None,
        epoch: int | None = None,
        seq: int | None = None,
        is_speech: bool = False,
        final: bool = False,
        context_rollover: bool = False,
        engine_rebase: bool = False,
        vision_input_source: str = "request_fallback",
    ) -> dict[str, Any]:
        """Build scheduler-owned Stage0 input embeddings for one audio append.

        Unlike the legacy worker-control path, this method never calls an eager
        model forward. The normal vLLM runner consumes the returned embeddings
        and owns attention metadata, block tables, KV cache, and sampling.
        """
        start_time = time.time()
        prep_diag: dict[str, float] | None = {} if _MINICPMO45_LOG_PREP_DIAG else None

        def sync_device() -> None:
            if prep_diag is None:
                return
            try:
                import torch

                if torch.cuda.is_available():
                    torch.accelerator.synchronize(self._model_device())
            except Exception:
                pass

        processor = self._configure_streaming_processor(state)
        append_identity = (epoch, seq) if seq is not None else None
        if (
            append_identity is not None
            and state.prepared_append_identity == append_identity
            and state.prepared_inputs_embeds is not None
        ):
            result = dict(state.prepared_result)
            if engine_rebase:
                if not state.prepared_rebase_prefix_embeds and not bool(result.get("rebase_prompt", False)):
                    return self._stage_prefill_result(
                        False,
                        start_time,
                        "engine rebase has no exact prepared compact prompt",
                    )
                import torch

                rebase_parts = [
                    *state.prepared_rebase_prefix_embeds,
                    state.prepared_inputs_embeds,
                ]
                result["inputs_embeds"] = torch.cat(
                    [self._as_2d_tensor(embed) for embed in rebase_parts],
                    dim=0,
                )
                result["input_token_ids"] = [
                    *state.prepared_rebase_prefix_token_ids,
                    *state.prepared_input_token_ids,
                ]
                result["rebase_prompt"] = True
                result["engine_rebase_prompt"] = True
                result["delta_num_tokens"] = len(result["input_token_ids"])
            else:
                result["inputs_embeds"] = state.prepared_inputs_embeds
                result["input_token_ids"] = list(state.prepared_input_token_ids)
            return result
        if engine_rebase:
            return self._stage_prefill_result(
                False,
                start_time,
                "engine rebase append identity does not match the prepared prompt",
            )
        self._require_special_token_ids()
        prepared_audio = preprocessed_audio
        # A first append and an explicit context rollover own every embedding
        # row in the scheduler prompt.  A steady append owns only its suffix;
        # the prefix remains authoritative in the engine KV cache.
        append_start_chunk_idx = (
            prepared_audio.start_chunk_idx
            if prepared_audio is not None
            else state.audio_chunk_idx
        )
        rebase_prompt = bool(context_rollover or append_start_chunk_idx == 0)
        if prepared_audio is None and state.audio_sidecar_active:
            return self._stage_prefill_result(
                False,
                start_time,
                "arrival audio lineage is active but this append has no encoded result",
            )
        if prepared_audio is None and (audio_waveform is None or len(audio_waveform) == 0):
            return self._stage_prefill_result(False, start_time, "empty audio")
        # Omni duplex: encode this append's camera frames up front so the unit
        # loop can interleave one <image> block per unit, mirroring official
        # streaming_prefill (feed <unit>, then image embeds, then audio).
        frame_blocks: list[list[Any]] = []
        sync_device()
        vision_start = time.perf_counter()
        if video_frames:
            if preencoded_vision is not None:
                if not all(isinstance(blocks, list) for blocks in preencoded_vision):
                    return self._stage_prefill_result(
                        False,
                        start_time,
                        "invalid preencoded streaming vision embeddings",
                    )
                # The unit-building loop consumes this list. Keep the batched
                # cache immutable so retries cannot observe a partially popped
                # result.
                model_device = self._model_device()
                frame_blocks = []
                for blocks in preencoded_vision:
                    materialized: list[Any] = []
                    for block in blocks:
                        to = getattr(block, "to", None)
                        materialized.append(to(device=model_device, non_blocking=True) if callable(to) else block)
                    frame_blocks.append(materialized)
            else:
                frame_blocks = self._stage_vision_embeddings(
                    video_frames,
                    max_slice_nums=max_slice_nums,
                    preprocessed=preprocessed_vision,
                )
            if frame_blocks is None or len(frame_blocks) != len(video_frames):
                return self._stage_prefill_result(False, start_time, "streaming vision embedding failed")
            self._require_vision_token_ids(
                include_slices=any(len(blocks) > 1 for blocks in frame_blocks),
            )
        sync_device()
        if prep_diag is not None:
            prep_diag["vision_consume_ms"] = (time.perf_counter() - vision_start) * 1000.0
        chunk_size = self._streaming_chunk_size(processor)
        if prepared_audio is not None and (
            prepared_audio.start_chunk_idx != state.audio_chunk_idx
            or prepared_audio.start_buffer_len != len(state.audio_buffer)
            or not prepared_audio.units
        ):
            # A stale speculative preparation must never advance the session.
            # Once GPU2 owns the lineage, request-local state deliberately has
            # no Whisper KV and cannot be an exact fallback.
            if prepared_audio.arrival_preencoded or state.audio_sidecar_active:
                return self._stage_prefill_result(
                    False,
                    start_time,
                    "arrival audio result does not match formal consumption order",
                )
            # Runner-local speculation can still use the exact fallback.
            prepared_audio = None
        if prepared_audio is None:
            state.audio_buffer = np.concatenate([state.audio_buffer, np.asarray(audio_waveform, dtype=np.float32)])
            self._pad_first_audio_chunk_if_needed(state, processor)
            if len(state.audio_buffer) < chunk_size:
                return self._stage_prefill_result(
                    False,
                    start_time,
                    f"audio not enough: need {chunk_size} samples, only {len(state.audio_buffer)}",
                )
        prepared_units = list(prepared_audio.units) if prepared_audio is not None else []
        prepared_unit_index = 0
        current_audio_chunk_idx = state.audio_chunk_idx

        retained_unit_embeds = state.last_unit_inputs_embeds
        retained_unit_token_ids = list(state.last_unit_input_token_ids)
        retained_output_token_ids = (
            list(state.current_segment_output_tokens[:-1]) if state.current_segment_output_tokens else []
        )
        if context_rollover and retained_unit_embeds is None:
            return self._stage_prefill_result(False, start_time, "context rollover has no complete retained unit")

        embed_parts: list[Any] = []
        token_ids: list[int] = []
        if state.audio_chunk_idx == 0 and state.context_embeds:
            embed_parts.extend(state.context_embeds)
            token_ids.extend(state.context_token_ids)

        # In a split Thinker, D owns generation.  Before P prefills the next
        # unit, replay every D-generated token whose KV D retained.  The final
        # sampled terminator is injected below with </unit>, exactly like the
        # native non-split scheduler.  On rollover these rows are already in
        # retained_output_token_ids and must not be duplicated.
        if state.pd_feedback_token_ids and not context_rollover:
            feedback_prefix = state.pd_feedback_token_ids[:-1]
            embed_parts.extend(self._embed_token(token_id) for token_id in feedback_prefix)
            token_ids.extend(feedback_prefix)

        # Consume every complete processor chunk in the buffer so the appended
        # span and the scheduler's slot reservation agree exactly. The serving
        # PCM buffer pads a final real residual before it reaches Stage0. Do not
        # pad again here: the first processor chunk is hop-aligned below 1035ms
        # and intentionally leaves a small carry that is not another model unit.
        units_built = 0
        audio_feature_ms = 0.0
        audio_encoder_ms = 0.0
        latest_unit_embed_parts: list[Any] = []
        latest_unit_token_ids: list[int] = []
        while True:
            if prepared_audio is not None:
                if prepared_unit_index >= len(prepared_units):
                    break
                prepared_unit = prepared_units[prepared_unit_index]
                if prepared_unit.chunk_idx != current_audio_chunk_idx:
                    return self._stage_prefill_result(
                        False,
                        start_time,
                        "prepared audio unit order does not match session state",
                    )
                batch_feature = prepared_unit.batch_feature
                consumed_samples = prepared_unit.consumed_samples
                audio_embeds = prepared_unit.audio_embeds
                if audio_embeds is None:
                    sync_device()
                    audio_encoder_start = time.perf_counter()
                    audio_embeds = self._stage_audio_embeddings(batch_feature, state=state)
                    sync_device()
                    audio_encoder_ms += (time.perf_counter() - audio_encoder_start) * 1000.0
                prepared_unit_index += 1
            else:
                if len(state.audio_buffer) < chunk_size:
                    break
                audio_chunk = state.audio_buffer[:chunk_size]
                feature_start = time.perf_counter()
                batch_feature = self._process_streaming_audio(
                    audio_chunk,
                    current_audio_chunk_idx,
                    processor=processor,
                )
                audio_feature_ms += (time.perf_counter() - feature_start) * 1000.0
                for name, value in (
                    ("chunk_idx", current_audio_chunk_idx),
                    ("use_extra_context", True),
                    ("prefix_extra_frames", 0 if current_audio_chunk_idx == 0 else 2),
                    ("suffix_extra_frames", 2),
                ):
                    with suppress(Exception):
                        setattr(batch_feature, name, value)
                sync_device()
                audio_encoder_start = time.perf_counter()
                audio_embeds = self._stage_audio_embeddings(batch_feature, state=state)
                sync_device()
                audio_encoder_ms += (time.perf_counter() - audio_encoder_start) * 1000.0
                consumed_samples = self._consumed_audio_samples(
                    current_audio_chunk_idx,
                    chunk_size,
                    processor=processor,
                )
            if audio_embeds is None:
                if units_built == 0:
                    return self._stage_prefill_result(False, start_time, "streaming audio embedding returned empty")
                break
            if current_audio_chunk_idx > 0:
                # Official duplex closes every unit (finalize_unit feeds the
                # sampled terminator + </unit>) before the next <unit> opens.
                # The scheduler session update discards the previous segment's
                # sampled terminator token, so it is re-injected here ahead of
                # the closure; the model's listen/speak policy depends on
                # seeing its own past decisions in context.
                pending_terminator = state.pending_terminator_token
                if pending_terminator is not None and units_built == 0:
                    state.pending_terminator_token = None
                    embed_parts.append(self._embed_token(pending_terminator))
                    token_ids.append(int(pending_terminator))
                embed_parts.append(self._embed_token(self.unit_end_token_id))
                token_ids.append(self.unit_end_token_id)
            unit_embed_offset = len(embed_parts)
            unit_token_offset = len(token_ids)
            embed_parts.append(self._embed_token(self.unit_token_id))
            token_ids.append(self.unit_token_id)
            if frame_blocks:
                # Official order inside a unit: one source-image block followed
                # by zero or more HD crop blocks, all ahead of audio.
                image_blocks = frame_blocks.pop(0)
                vision_block = self._as_2d_tensor(image_blocks[0])
                embed_parts.append(self._embed_token(self.image_start_token_id))
                token_ids.append(int(self.image_start_token_id))
                embed_parts.append(vision_block)
                token_ids.extend([self._vision_embedding_placeholder_token_id()] * int(vision_block.shape[0]))
                embed_parts.append(self._embed_token(self.image_end_token_id))
                token_ids.append(int(self.image_end_token_id))
                for crop in image_blocks[1:]:
                    crop_block = self._as_2d_tensor(crop)
                    embed_parts.append(self._embed_token(self.slice_start_token_id))
                    token_ids.append(int(self.slice_start_token_id))
                    embed_parts.append(crop_block)
                    token_ids.extend([self._vision_embedding_placeholder_token_id()] * int(crop_block.shape[0]))
                    embed_parts.append(self._embed_token(self.slice_end_token_id))
                    token_ids.append(int(self.slice_end_token_id))
            audio_block = self._as_2d_tensor(audio_embeds)
            if getattr(audio_block, "device", None) != self._model_device():
                audio_block = audio_block.to(
                    device=self._model_device(),
                    non_blocking=True,
                )
            embed_parts.append(audio_block)
            token_ids.extend(
                [self._audio_embedding_placeholder_token_id()] * int(audio_block.shape[0])
            )
            latest_unit_embed_parts = list(embed_parts[unit_embed_offset:])
            latest_unit_token_ids = list(token_ids[unit_token_offset:])
            if prepared_audio is None:
                state.audio_buffer = state.audio_buffer[consumed_samples:]
            current_audio_chunk_idx += 1
            units_built += 1
            if prepared_audio is None:
                chunk_size = self._streaming_chunk_size(processor)
        if frame_blocks:
            # Each frame reserves one image block in the append plan. A frame
            # without a matching audio unit would desynchronize that plan.
            return self._stage_prefill_result(
                False,
                start_time,
                f"{len(frame_blocks)} video frame(s) left without a matching audio unit",
            )
        if prepared_audio is not None:
            state.audio_buffer = prepared_audio.remaining_audio_buffer
            if prepared_audio.arrival_preencoded:
                # GPU2 owns the authoritative Mel/Whisper lineage. Formal P
                # advances only its lightweight consumption watermark.
                state.audio_sidecar_active = True
                state.audio_past_key_values = None
            else:
                if prepared_audio.encoded and hasattr(self.thinker, "audio_past_key_values"):
                    state.audio_past_key_values = prepared_audio.audio_past_key_values
                self._restore_streaming_mel_snapshot(
                    processor,
                    prepared_audio.mel_snapshot_after,
                )
            audio_feature_ms = prepared_audio.feature_ms
        state.audio_chunk_idx = current_audio_chunk_idx
        # Match official streaming_prefill: per chunk feed ONLY <unit>+audio. The assistant
        # turn is opened once at session init; re-emitting the turn-open prefix per chunk
        # re-opened the turn each chunk -> degenerate repetition. tts_bos/listen/turn_eos are
        # model-generated and tracked via current_turn_ended (mirrors streaming_generate).
        prompt_suffix_len = 0

        import torch

        compact_rebase_prefix_parts: tuple[Any, ...] = ()
        compact_rebase_prefix_token_ids: tuple[int, ...] = ()
        if not rebase_prompt and retained_unit_embeds is not None:
            # Keep one exact, bounded recovery snapshot alongside the suffix.
            # It is never concatenated on the normal path: these are immutable
            # prefix component references, while ``inputs_embeds`` below is
            # the already-materialized append suffix.
            compact_rebase_prefix_parts = (
                *state.context_embeds,
                retained_unit_embeds,
            )
            compact_rebase_prefix_token_ids = (
                *state.context_token_ids,
                *retained_unit_token_ids,
            )

        if context_rollover:
            retained_parts = list(state.context_embeds)
            retained_parts.append(retained_unit_embeds)
            retained_parts.extend(self._embed_token(token_id) for token_id in retained_output_token_ids)
            embed_parts = retained_parts + embed_parts
            token_ids = list(state.context_token_ids) + retained_unit_token_ids + retained_output_token_ids + token_ids
            state.generated_tokens = retained_output_token_ids[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :]
            state.context_rollovers += 1

        sync_device()
        assembly_start = time.perf_counter()
        inputs_embeds = torch.cat([self._as_2d_tensor(embed) for embed in embed_parts], dim=0)
        if int(inputs_embeds.shape[0]) != len(token_ids):
            raise RuntimeError(
                "MiniCPM-o duplex Stage0 produced mismatched embedding and "
                f"token rows: embeddings={int(inputs_embeds.shape[0])}, "
                f"tokens={len(token_ids)}"
            )
        sync_device()
        if prep_diag is not None:
            prep_diag.update(
                {
                    "audio_feature_ms": audio_feature_ms,
                    "audio_encoder_ms": audio_encoder_ms,
                    "assembly_ms": (time.perf_counter() - assembly_start) * 1000.0,
                }
            )
        result = self._stage_prefill_result(True, start_time)
        result.update(
            {
                "inputs_embeds": inputs_embeds,
                "input_token_ids": token_ids,
                "special_token_ids": self._special_token_ids(),
                "num_input_tokens": int(inputs_embeds.shape[0]),
                "prompt_suffix_len": prompt_suffix_len,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
                "context_rollover": context_rollover,
                "context_rollovers": state.context_rollovers,
                "rebase_prompt": rebase_prompt,
                "delta_num_tokens": len(token_ids),
            }
        )
        # Persist the physical source with the prepared append. vLLM may call
        # preprocess again while chunking/replaying one request; those later
        # calls must report the original consumption instead of relabelling an
        # arrival-cache hit as a request-local fallback.
        input_video_frames = len(video_frames) if isinstance(video_frames, list) else 0
        result["duplex_input_video_frames"] = input_video_frames
        result["duplex_arrival_video_frames"] = input_video_frames if vision_input_source == "arrival" else 0
        result["duplex_vision_fallback_frames"] = (
            input_video_frames if input_video_frames and vision_input_source != "arrival" else 0
        )
        result["duplex_arrival_audio_units"] = (
            units_built
            if prepared_audio is not None and prepared_audio.arrival_preencoded
            else 0
        )
        result["duplex_audio_fallback_units"] = (
            units_built
            if prepared_audio is None or not prepared_audio.arrival_preencoded
            else 0
        )
        if prep_diag is not None:
            prep_diag["stage_total_ms"] = (time.time() - start_time) * 1000.0
            result["prep_diag"] = prep_diag
        if latest_unit_embed_parts:
            state.last_unit_inputs_embeds = torch.cat(
                [self._as_2d_tensor(embed) for embed in latest_unit_embed_parts],
                dim=0,
            )
            state.last_unit_input_token_ids = latest_unit_token_ids
        state.current_segment_output_tokens.clear()
        state.pd_feedback_token_ids.clear()
        if is_speech and (append_identity is None or state.pending_speech_append_identity != append_identity):
            state.pending_speech_context = True
            state.pending_speech_append_identity = append_identity
        if append_identity is not None:
            state.prepared_append_identity = append_identity
            state.prepared_inputs_embeds = inputs_embeds
            state.prepared_input_token_ids = list(token_ids)
            if rebase_prompt:
                state.prepared_rebase_prefix_embeds = ()
                state.prepared_rebase_prefix_token_ids = ()
            else:
                compact_prefix_rows = sum(
                    int(self._as_2d_tensor(embed).shape[0]) for embed in compact_rebase_prefix_parts
                )
                if compact_prefix_rows != len(compact_rebase_prefix_token_ids):
                    raise RuntimeError(
                        "MiniCPM-o duplex compact rebase prefix embedding/token "
                        "row mismatch: "
                        f"embeddings={compact_prefix_rows}, "
                        f"tokens={len(compact_rebase_prefix_token_ids)}"
                    )
                state.prepared_rebase_prefix_embeds = compact_rebase_prefix_parts
                state.prepared_rebase_prefix_token_ids = compact_rebase_prefix_token_ids
            state.prepared_result = {k: v for k, v in result.items() if k not in {"inputs_embeds", "input_token_ids"}}
        return result

    def apply_pd_decode_feedback(
        self,
        state: _MiniCPMO45Stage0SessionState,
        token_ids: list[int],
        *,
        epoch: int | None,
        seq: int,
        force_listen: bool = False,
    ) -> None:
        """Align P's next prompt with the authoritative D segment.

        D retains all generated tokens except its final sampled terminator in
        KV.  P therefore prefills those retained tokens before the next media
        unit and re-injects the terminator at the native unit boundary.  The
        append identity makes retries idempotent.
        """
        identity = (epoch, seq)
        if state.pd_feedback_append_identity == identity:
            return
        state.pd_feedback_append_identity = identity
        state.pd_feedback_token_ids = [int(token_id) for token_id in token_ids]
        state.current_segment_output_tokens = list(state.pd_feedback_token_ids)
        if not state.pd_feedback_token_ids:
            return

        terminator = state.pd_feedback_token_ids[-1]
        terminators = {
            self.listen_token_id,
            self.chunk_eos_token_id,
            self.chunk_tts_eos_token_id,
            self.turn_eos_token_id,
        }
        if terminator in terminators:
            state.pending_terminator_token = terminator
            state.last_terminator_token = terminator
            if terminator == self.turn_eos_token_id or (terminator == self.listen_token_id and force_listen):
                state.current_turn_ended = True
                state.pending_speech_response_open = False
            return
        state.pending_terminator_token = None
        state.last_terminator_token = None
        state.current_turn_ended = False

    @staticmethod
    def _stage_prefill_result(success: bool, start_time: float, reason: str = "") -> dict[str, Any]:
        return {
            "success": success,
            "prefill_success": success,
            "is_buffering": not success,
            "reason": reason,
            "cost_all": time.time() - start_time,
            "stage_runtime_ready": True,
        }

    @staticmethod
    def _as_2d_tensor(value: Any) -> Any:
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    def _embed_token(self, token_id: int) -> Any:
        import torch

        token = torch.tensor([int(token_id)], dtype=torch.long, device=self._model_device())
        embedder = self._token_embedder()
        embeds = embedder(token)
        return self._as_2d_tensor(embeds)

    def _token_embedder(self) -> Any:
        nested_embed = getattr(getattr(getattr(self.thinker, "llm", None), "model", None), "embed_tokens", None)
        if callable(nested_embed):
            return nested_embed
        for target in (self.thinker, self.stage_model):
            embedder = getattr(target, "get_input_embeddings", None)
            if callable(embedder):
                try:
                    embeddings = embedder()
                    if callable(embeddings):
                        return embeddings
                except TypeError:
                    return embedder
        raise AttributeError("MiniCPM-o stage0 model does not expose token embeddings")

    def _model_device(self) -> Any:
        try:
            return next(self.thinker.parameters()).device
        except Exception:
            pass
        try:
            return next(self.stage_model.parameters()).device
        except Exception:
            pass
        return self.device

    def _streaming_chunk_size(self, processor: Any | None = None) -> int:
        processor = processor or self.processor
        get_chunk = getattr(processor, "get_streaming_chunk_size", None)
        if callable(get_chunk):
            return int(get_chunk())
        return 16000

    def _sample_rate(self, processor: Any | None = None) -> int:
        processor = processor or self.processor
        return int(
            self._stage_param(
                "sample_rate",
                getattr(getattr(processor, "_streaming_mel_processor", None), "sample_rate", 16000),
            )
        )

    def _first_chunk_samples(self, default_chunk_size: int, processor: Any | None = None) -> int:
        processor = processor or self.processor
        if getattr(processor, "_streaming_mel_processor", None) is None:
            return default_chunk_size
        return int(self._stage_param("first_chunk_ms", 1035) * self._sample_rate(processor) / 1000)

    def _pad_first_audio_chunk_if_needed(
        self,
        state: _MiniCPMO45Stage0SessionState,
        processor: Any | None = None,
    ) -> None:
        if state.audio_chunk_idx != 0 or len(state.audio_buffer) == 0:
            return
        first_chunk_samples = self._first_chunk_samples(
            self._streaming_chunk_size(processor),
            processor,
        )
        if len(state.audio_buffer) >= first_chunk_samples:
            return
        padding = np.zeros(first_chunk_samples - len(state.audio_buffer), dtype=np.float32)
        state.audio_buffer = np.concatenate([padding, state.audio_buffer])

    def _stage_param(self, name: str, default: Any) -> Any:
        for target in (self.stage_model, self.thinker, getattr(self.thinker, "llm", None)):
            value = getattr(target, name, None)
            if value is not None:
                return value
            value = getattr(target, name.upper(), None)
            if value is not None:
                return value
        return default

    def _consumed_audio_samples(
        self,
        chunk_idx: int,
        default_chunk_size: int,
        *,
        processor: Any | None = None,
    ) -> int:
        processor = processor or self.processor
        if chunk_idx != 0:
            chunk_ms = int(self._stage_param("chunk_ms", 1000))
            return int(chunk_ms * self._sample_rate(processor) / 1000)
        mel_processor = getattr(processor, "_streaming_mel_processor", None)
        get_config = getattr(mel_processor, "get_config", None)
        if callable(get_config):
            cfg = get_config()
            if isinstance(cfg, dict):
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self._stage_param("first_chunk_ms", 1035)))
                return int(consumed_ms * self._sample_rate(processor) / 1000)
        return default_chunk_size

    def _process_streaming_audio(
        self,
        audio_chunk: Any,
        chunk_idx: int,
        *,
        processor: Any | None = None,
    ) -> Any:
        processor = processor or self.processor
        process = getattr(processor, "process_audio_streaming", None)
        if callable(process):
            try:
                return process(audio_chunk, reset=False, return_batch_feature=True)
            except TypeError:
                return process(audio_chunk, chunk_idx=chunk_idx)
        return {"audio_features": audio_chunk, "audio_feature_lens": [[len(audio_chunk)]]}

    @staticmethod
    def _restore_streaming_mel_snapshot(processor: Any, snapshot: Any) -> bool:
        mel_processor = getattr(processor, "_streaming_mel_processor", None)
        restore = getattr(mel_processor, "restore_snapshot", None)
        if not callable(restore) or snapshot is None:
            return False
        restore(snapshot)
        return True

    def _prepare_streaming_audio_append(
        self,
        state: _MiniCPMO45Stage0SessionState,
        audio_waveform: Any,
    ) -> _MiniCPMO45PreparedAudioAppend | None:
        """Prepare exact streaming Mel features without advancing live state.

        Every session owns its own processor, so callers may run this method in
        parallel across sessions.  The official exact processor recomputes Mel
        features from its retained waveform and mutates its cursor.  Snapshot
        it around the work so an interrupted batch cannot corrupt the session;
        the resulting post-state is committed by the request-local path.
        """
        processor = self._configure_streaming_processor(state)
        mel_processor = getattr(processor, "_streaming_mel_processor", None)
        get_snapshot = getattr(mel_processor, "get_snapshot", None)
        restore_snapshot = getattr(mel_processor, "restore_snapshot", None)
        if not callable(get_snapshot) or not callable(restore_snapshot):
            return None
        waveform = np.asarray(audio_waveform, dtype=np.float32)
        if waveform.size == 0:
            return None

        before_snapshot = get_snapshot()
        working_buffer = np.concatenate([state.audio_buffer, waveform])
        start_chunk_idx = state.audio_chunk_idx
        if start_chunk_idx == 0 and len(working_buffer) > 0:
            first_chunk_samples = self._first_chunk_samples(
                self._streaming_chunk_size(processor),
                processor,
            )
            if len(working_buffer) < first_chunk_samples:
                working_buffer = np.concatenate(
                    [
                        np.zeros(
                            first_chunk_samples - len(working_buffer),
                            dtype=np.float32,
                        ),
                        working_buffer,
                    ]
                )

        units: list[_MiniCPMO45PreparedAudioUnit] = []
        chunk_idx = start_chunk_idx
        feature_start = time.perf_counter()
        try:
            chunk_size = self._streaming_chunk_size(processor)
            while len(working_buffer) >= chunk_size:
                batch_feature = self._process_streaming_audio(
                    working_buffer[:chunk_size],
                    chunk_idx,
                    processor=processor,
                )
                for name, value in (
                    ("chunk_idx", chunk_idx),
                    ("use_extra_context", True),
                    ("prefix_extra_frames", 0 if chunk_idx == 0 else 2),
                    ("suffix_extra_frames", 2),
                ):
                    with suppress(Exception):
                        setattr(batch_feature, name, value)
                consumed_samples = self._consumed_audio_samples(
                    chunk_idx,
                    chunk_size,
                    processor=processor,
                )
                units.append(
                    _MiniCPMO45PreparedAudioUnit(
                        chunk_idx=chunk_idx,
                        batch_feature=batch_feature,
                        consumed_samples=consumed_samples,
                    )
                )
                working_buffer = working_buffer[consumed_samples:]
                chunk_idx += 1
                chunk_size = self._streaming_chunk_size(processor)
            after_snapshot = get_snapshot()
        except Exception:
            restore_snapshot(before_snapshot)
            return None
        restore_snapshot(before_snapshot)
        if not units:
            return None
        return _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=start_chunk_idx,
            start_buffer_len=len(state.audio_buffer),
            units=units,
            remaining_audio_buffer=working_buffer,
            mel_snapshot_after=after_snapshot,
            feature_ms=(time.perf_counter() - feature_start) * 1000.0,
        )

    def _stage_audio_embeddings_batch(
        self,
        prepared: list[tuple[_MiniCPMO45PreparedAudioAppend, _MiniCPMO45Stage0SessionState]],
        *,
        return_on_audio_device: bool = False,
    ) -> bool:
        """Encode equal-shaped audio units from independent sessions together."""
        if len(prepared) < 2:
            return False
        target = next(
            (
                candidate
                for candidate in (self.stage_model, self.thinker)
                if callable(getattr(candidate, "get_audio_embedding_streaming_batch", None))
                and callable(getattr(candidate, "combine_audio_past_key_values", None))
                and callable(getattr(candidate, "split_audio_past_key_values", None))
                and callable(getattr(candidate, "should_reset_audio_past_key_values", None))
            ),
            None,
        )
        if target is None:
            return False

        import torch

        cache_by_plan = {id(plan): state.audio_past_key_values for plan, state in prepared}
        try:
            max_units = max(len(plan.units) for plan, _state in prepared)
            for unit_index in range(max_units):
                groups: dict[tuple[int, int, int], list[tuple[Any, ...]]] = {}
                for plan, _state in prepared:
                    if unit_index >= len(plan.units):
                        continue
                    unit = plan.units[unit_index]
                    feature = unit.batch_feature["audio_features"]
                    if not isinstance(feature, torch.Tensor) or feature.ndim != 3:
                        raise ValueError("unexpected streaming audio feature shape")
                    cache = cache_by_plan[id(plan)]
                    cache_len = int(target.audio_cache_seq_length(cache))
                    group_key = (
                        int(unit.chunk_idx == 0),
                        int(feature.shape[-1]),
                        cache_len,
                    )
                    groups.setdefault(group_key, []).append((plan, unit, feature, cache))

                for (_is_first, _feature_len, _cache_len), records in groups.items():
                    # A singleton gains nothing from the batching wrapper. Keep
                    # it on the request-local exact path.
                    if len(records) < 2:
                        continue
                    features = torch.cat([record[2] for record in records], dim=0)
                    feature_lens = []
                    for _plan, unit, _feature, _cache in records:
                        raw_lens = unit.batch_feature["audio_feature_lens"]
                        if isinstance(raw_lens, torch.Tensor):
                            lens = raw_lens.reshape(-1)
                        elif isinstance(raw_lens, (list, tuple)) and raw_lens:
                            lens = torch.as_tensor(raw_lens[0]).reshape(-1)
                        else:
                            raise ValueError("missing streaming audio feature length")
                        feature_lens.append(lens)
                    prefix_extra_frames = 0 if records[0][1].chunk_idx == 0 else 2
                    reset_cache = target.should_reset_audio_past_key_values(
                        records[0][3],
                        max_mel_seq_len=int(features.shape[-1]),
                        use_extra_context=True,
                        prefix_extra_frames=prefix_extra_frames,
                        suffix_extra_frames=2,
                    )
                    combined_cache = (
                        None
                        if reset_cache
                        else target.combine_audio_past_key_values([record[3] for record in records])
                    )
                    outputs, combined_cache = target.get_audio_embedding_streaming_batch(
                        {
                            "audio_features": features,
                            "audio_feature_lens": feature_lens,
                        },
                        past_key_values=combined_cache,
                        use_extra_context=True,
                        prefix_extra_frames=prefix_extra_frames,
                        suffix_extra_frames=2,
                        return_on_audio_device=return_on_audio_device,
                    )
                    if len(outputs) != len(records):
                        raise ValueError("batched audio encoder returned wrong batch size")
                    split_caches = target.split_audio_past_key_values(
                        combined_cache,
                        len(records),
                    )
                    if len(split_caches) != len(records):
                        raise ValueError("batched audio cache returned wrong batch size")
                    for record, output, split_cache in zip(
                        records,
                        outputs,
                        split_caches,
                        strict=True,
                    ):
                        plan, unit, _feature, _cache = record
                        audio_embeds = self._cat_nested_tensors(output)
                        if audio_embeds is None:
                            raise ValueError("batched streaming audio embedding is empty")
                        unit.audio_embeds = audio_embeds
                        cache_by_plan[id(plan)] = split_cache

            # Commit each plan independently. Mixed cache-length cohorts are
            # normal in long sessions: one cohort may form a useful batch while
            # another is a singleton.  Discarding every completed cohort just
            # because one plan could not batch makes the request-local path
            # redundantly run the successful Whisper forwards again.
            encoded_any = False
            for plan, _state in prepared:
                if all(unit.audio_embeds is not None for unit in plan.units):
                    plan.audio_past_key_values = cache_by_plan[id(plan)]
                    plan.encoded = True
                    encoded_any = True
                    continue
                # A partially encoded multi-unit plan cannot mix the explicit
                # batch cache with the request-local mutable cache. Clear only
                # that plan and let its ordinary transactional path own all of
                # its units.
                plan.audio_past_key_values = None
                plan.encoded = False
                for unit in plan.units:
                    unit.audio_embeds = None
            return encoded_any
        except Exception:
            logger.exception(
                "MiniCPM-o batched arrival audio encoding failed for sessions %s",
                [state.session_id for _plan, state in prepared],
            )
            for plan, _state in prepared:
                plan.audio_past_key_values = None
                plan.encoded = False
                for unit in plan.units:
                    unit.audio_embeds = None
            return False

    def _stage_audio_embeddings(
        self,
        batch_feature: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        lock = getattr(self, "_audio_execution_lock", None)
        if lock is None:
            lock = RLock()
            self._audio_execution_lock = lock
        with lock:
            return self._stage_audio_embeddings_unlocked(
                batch_feature,
                state=state,
            )

    def _stage_audio_embeddings_unlocked(
        self,
        batch_feature: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        if hasattr(batch_feature, "to"):
            batch_feature = batch_feature.to(self.device)
        self._ensure_dynamic_cache_compat()
        has_audio_cache = state is not None and hasattr(self.thinker, "audio_past_key_values")
        previous_audio_past_key_values = (
            getattr(self.thinker, "audio_past_key_values", None) if has_audio_cache else None
        )
        if has_audio_cache:
            self.thinker.audio_past_key_values = state.audio_past_key_values
        try:
            for target in (self.stage_model, self.thinker):
                get_streaming = getattr(target, "get_audio_embedding_streaming", None)
                if callable(get_streaming):
                    try:
                        result = self._cat_nested_tensors(
                            get_streaming(
                                batch_feature,
                                use_extra_context=True,
                                prefix_extra_frames=0 if int(getattr(batch_feature, "chunk_idx", 0)) == 0 else 2,
                                suffix_extra_frames=2,
                            )
                        )
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                    except TypeError:
                        result = self._cat_nested_tensors(get_streaming(batch_feature))
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    result = self._cat_nested_tensors(get_hidden(batch_feature))
                    if has_audio_cache:
                        state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                    return result
            return None
        finally:
            if has_audio_cache:
                self.thinker.audio_past_key_values = previous_audio_past_key_values

    @staticmethod
    def _decode_ref_audio_from_session_config(session_config: dict[str, Any]) -> Any | None:
        from vllm_omni.experimental.fullduplex.minicpmo45.input import decode_native_ref_audio_from_config

        return decode_native_ref_audio_from_config({"extra_body": session_config})

    def _stage_ref_audio_embeddings(
        self,
        ref_audio: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        process_audio = getattr(self.processor, "process_audio", None)
        if callable(process_audio):
            batch_feature = process_audio([ref_audio])
            if hasattr(batch_feature, "to"):
                batch_feature = batch_feature.to(self.device)
            self._ensure_dynamic_cache_compat()
            for target in (self.stage_model, self.thinker):
                get_audio_embedding = getattr(target, "get_audio_embedding", None)
                if callable(get_audio_embedding):
                    try:
                        chunk_length = getattr(getattr(target, "config", None), "audio_chunk_length", None)
                        if chunk_length is not None:
                            return self._cat_nested_tensors(
                                get_audio_embedding(batch_feature, chunk_length=chunk_length)
                            )
                    except TypeError:
                        pass
                    return self._cat_nested_tensors(get_audio_embedding(batch_feature))
                # The split vLLM stage0 wrapper ports official
                # get_audio_embedding(chunk_length=...) as
                # get_audio_hidden_states (chunk_length comes from config).
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    return self._cat_nested_tensors(get_hidden(batch_feature))
        # The split vLLM stage0 wrapper may only expose the streaming encoder
        # path.  Use it as a fallback so the server-resolved reference audio is
        # still represented in the same system context location as official
        # MiniCPM-o prepare().
        processor = state.streaming_processor if state is not None else self.processor
        batch_feature = self._process_streaming_audio(ref_audio, 0, processor=processor)
        for name, value in (
            ("chunk_idx", 0),
            ("use_extra_context", True),
            ("prefix_extra_frames", 0),
            ("suffix_extra_frames", 2),
        ):
            with suppress(Exception):
                setattr(batch_feature, name, value)
        return self._stage_audio_embeddings(batch_feature, state=state)

    @staticmethod
    def _ensure_dynamic_cache_compat() -> None:
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return
        if hasattr(DynamicCache, "get_usable_length"):
            return

        def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if not callable(get_seq_length):
                return 0
            try:
                return int(get_seq_length(layer_idx))
            except TypeError:
                return int(get_seq_length())

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    @staticmethod
    def _cat_nested_tensors(value: Any) -> Any | None:
        import torch

        tensors = []

        def collect(item: Any) -> None:
            if item is None:
                return
            if hasattr(item, "detach"):
                tensors.append(item)
                return
            if isinstance(item, dict):
                for child in item.values():
                    collect(child)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        if not tensors:
            return None
        return torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)

    def _encode_text(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text, add_special_tokens=False))
        return []

    def _special_token_ids(self) -> dict[str, int]:
        return {
            name: value
            for name, value in {
                "unit_token_id": self.unit_token_id,
                "unit_end_token_id": self.unit_end_token_id,
                "listen_token_id": self.listen_token_id,
                "speak_token_id": self.speak_token_id,
                "tts_bos_token_id": self.tts_bos_token_id,
                "tts_eos_token_id": self.tts_eos_token_id,
                "tts_pad_token_id": self.tts_pad_token_id,
                "chunk_eos_token_id": self.chunk_eos_token_id,
                "chunk_tts_eos_token_id": self.chunk_tts_eos_token_id,
                "turn_eos_token_id": self.turn_eos_token_id,
            }.items()
            if isinstance(value, int) and value >= 0
        }

    @staticmethod
    def _load_processor_from_path(model_path: str | None) -> Any | None:
        if not model_path:
            return None
        try:
            from transformers import AutoImageProcessor, AutoProcessor

            original_register = AutoImageProcessor.register

            def register_image_processor(config_class, *args, **kwargs):
                # The checkpoint's auto_map already loads this class. Its
                # legacy string registration is incompatible with some
                # Transformers versions and is otherwise redundant.
                if config_class == "MiniCPMVImageProcessor":
                    return None
                return original_register(config_class, *args, **kwargs)

            with _MINICPMO45_PROCESSOR_LOAD_LOCK:
                AutoImageProcessor.register = staticmethod(register_image_processor)
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                finally:
                    AutoImageProcessor.register = staticmethod(original_register)
        except Exception as exc:
            raise RuntimeError(f"Failed to load MiniCPM-o duplex processor from {model_path!r}") from exc
        if getattr(processor, "tokenizer", None) is None:
            raise RuntimeError(f"MiniCPM-o duplex processor loaded from {model_path!r} does not expose a tokenizer")
        return processor

    def _init_token_ids(self) -> None:
        if self.tokenizer is None:
            for field_name in _MINICPMO45_SPECIAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
            for field_name in _MINICPMO45_OPTIONAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
        else:
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))
            for field_name, token in _MINICPMO45_OPTIONAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))

    def _resolve_special_token_id(self, token: str) -> int:
        if self.tokenizer is None:
            return -1

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        candidate = None
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
            with suppress(TypeError, ValueError):
                candidate = int(value)
        if candidate is not None and candidate >= 0 and candidate != unk_token_id:
            return candidate

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            ids = list(encode(token, add_special_tokens=False))
            if len(ids) == 1:
                value = int(ids[0])
                if value >= 0 and value != unk_token_id:
                    return value
        return -1

    def _require_special_token_ids(self) -> None:
        missing = [
            token
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items()
            if not isinstance(getattr(self, field_name, None), int) or getattr(self, field_name) < 0
        ]
        if missing:
            raise ValueError(
                "MiniCPM-o 4.5 native duplex requires tokenizer-defined special "
                f"tokens, missing or unknown: {', '.join(missing)}"
            )

    def _required_token_id(self, field_name: str) -> int:
        token_id = getattr(self, field_name, None)
        if not isinstance(token_id, int) or token_id < 0:
            token = _MINICPMO45_SPECIAL_TOKEN_FIELDS.get(field_name, field_name)
            raise ValueError(f"MiniCPM-o 4.5 missing required special token id for {token}")
        return token_id

    def stage_padding_token_id(self) -> int:
        return self._required_token_id("unit_end_token_id")

    def _audio_embedding_placeholder_token_id(self) -> int:
        token_id = getattr(self, "audio_placeholder_token_id", -1)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return self.stage_padding_token_id()

    @staticmethod
    def _decode_audio_payload(payload: dict[str, Any]) -> Any:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            raise ValueError("audio append payload requires base64 audio")
        fmt = payload.get("format") or "pcm_f32le"
        if fmt != "pcm_f32le":
            raise ValueError(f"MiniCPM-o stage0 expects pcm_f32le audio, got {fmt!r}")
        return np.frombuffer(base64.b64decode(audio), dtype=np.float32)

    @staticmethod
    def _decode_video_frames_payload(payload: dict[str, Any]) -> list[Any]:
        """Decode omni-duplex camera frames (base64 JPEG/PNG) to PIL images."""
        frames = payload.get("video_frames")
        if not isinstance(frames, list) or not frames:
            return []
        from io import BytesIO

        from PIL import Image

        decoded: list[Any] = []
        for frame_b64 in frames:
            if not isinstance(frame_b64, str) or not frame_b64:
                continue
            try:
                raw = base64.b64decode(frame_b64, validate=True)
                image = Image.open(BytesIO(raw))
                image.load()
            except Exception as exc:  # noqa: BLE001 - normalized below
                raise ValueError("invalid omni duplex video frame payload") from exc
            decoded.append(image.convert("RGB"))
        return decoded

    def _vision_embedding_placeholder_token_id(self) -> int:
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if isinstance(unk_token_id, int) and unk_token_id >= 0:
            return unk_token_id
        return self._required_optional_token_id("image_start_token_id")

    def _required_optional_token_id(self, field_name: str) -> int:
        token_id = getattr(self, field_name, None)
        if not isinstance(token_id, int) or token_id < 0:
            token = _MINICPMO45_OPTIONAL_TOKEN_FIELDS.get(field_name, field_name)
            raise ValueError(f"MiniCPM-o 4.5 missing required special token id for {token}")
        return token_id

    def _require_vision_token_ids(self, *, include_slices: bool = False) -> None:
        fields = ["image_start_token_id", "image_end_token_id"]
        if include_slices:
            fields.extend(["slice_start_token_id", "slice_end_token_id"])
        missing = [
            _MINICPMO45_OPTIONAL_TOKEN_FIELDS[field_name]
            for field_name in fields
            if not isinstance(getattr(self, field_name, None), int) or getattr(self, field_name) < 0
        ]
        if missing:
            raise ValueError(
                "MiniCPM-o 4.5 omni duplex requires tokenizer-defined image tokens, "
                f"missing or unknown: {', '.join(missing)}"
            )

    def _stage_vision_embeddings(
        self,
        frames: list[Any],
        *,
        max_slice_nums: int | list[int] = 1,
        preprocessed: Any | None = None,
    ) -> list[list[Any]] | None:
        lock = getattr(self, "_vision_execution_lock", None)
        if lock is None:
            lock = RLock()
            self._vision_execution_lock = lock
        with lock:
            return self._stage_vision_embeddings_unlocked(
                frames,
                max_slice_nums=max_slice_nums,
                preprocessed=preprocessed,
            )

    def _stage_vision_embeddings_unlocked(
        self,
        frames: list[Any],
        *,
        max_slice_nums: int | list[int] = 1,
        preprocessed: Any | None = None,
    ) -> list[list[Any]] | None:
        """Encode camera frames for omni duplex via the loaded vision tower.

        Semantics mirror ``MiniCPMODuplex.streaming_prefill``: every frame has
        one 64-row source-image block and may have additional 64-row HD crops.
        The vLLM wrapper's ``get_vision_hidden_states`` runs vpm + resampler.
        """
        process_image = getattr(self.processor, "process_image", None)
        if not callable(process_image):
            return None
        if isinstance(max_slice_nums, int) and not isinstance(max_slice_nums, bool):
            slice_limits = [max(1, max_slice_nums)] * len(frames)
        elif isinstance(max_slice_nums, list) and len(max_slice_nums) == len(frames):
            slice_limits = [
                max(1, value) if isinstance(value, int) and not isinstance(value, bool) else 1
                for value in max_slice_nums
            ]
        else:
            return None
        if len(set(slice_limits)) != 1:
            # Official code permits a per-frame list. Process separately here
            # so the nested slice grouping remains explicit for the runner.
            groups: list[list[Any]] = []
            for frame, limit in zip(frames, slice_limits, strict=True):
                encoded = self._stage_vision_embeddings([frame], max_slice_nums=limit)
                if encoded is None:
                    return None
                groups.extend(encoded)
            return groups
        if preprocessed is None:
            try:
                processed = process_image(frames, max_slice_nums=slice_limits[0])
            except Exception:  # noqa: BLE001 - prefill fails with a reason
                return None
        else:
            processed = preprocessed
        targets = (self.stage_model, self.thinker, getattr(self.stage_model, "model", None))
        for target in targets:
            if target is None:
                continue
            get_hidden = getattr(target, "get_vision_hidden_states", None)
            vpm = getattr(target, "vpm", None)
            if not callable(get_hidden) or vpm is None:
                continue
            try:
                import torch

                vpm_param = next(vpm.parameters())
                device, dtype = vpm_param.device, vpm_param.dtype
                pixel_nested = processed["pixel_values"]
                tgt_nested = processed["tgt_sizes"]
                slice_counts = [len(image_slices) for image_slices in pixel_nested]
                flat_pixels: list[Any] = []
                flat_tgt: list[Any] = []
                for image_slices, image_tgt in zip(pixel_nested, tgt_nested):
                    for slice_pixels in image_slices:
                        flat_pixels.append(slice_pixels.to(device=device, dtype=dtype))
                    tgt_tensor = image_tgt if hasattr(image_tgt, "reshape") else torch.tensor(image_tgt)
                    flat_tgt.append(tgt_tensor.reshape(-1, 2))
                tgt_sizes = torch.cat(flat_tgt, dim=0).to(device=device, dtype=torch.int64)
                with torch.no_grad():
                    hidden = get_hidden({"pixel_values": flat_pixels, "tgt_sizes": tgt_sizes})
            except Exception:  # noqa: BLE001 - prefill fails with a reason
                return None
            expected = MiniCPMO45DuplexPolicy.VISION_EMBEDS_PER_BLOCK
            flat_blocks: list[Any] = []
            for block in hidden:
                block_2d = self._as_2d_tensor(block)
                if int(block_2d.shape[0]) != expected:
                    return None
                flat_blocks.append(block_2d)
            if len(flat_blocks) != sum(slice_counts):
                return None
            out: list[list[Any]] = []
            offset = 0
            for count in slice_counts:
                out.append(flat_blocks[offset : offset + count])
                offset += count
            return out
        return None

    def _stage_vision_embeddings_batch(
        self,
        processed_batch: list[Any],
        *,
        microbatch_size: int = 8,
    ) -> list[list[list[Any]]] | None:
        lock = getattr(self, "_vision_execution_lock", None)
        if lock is None:
            lock = RLock()
            self._vision_execution_lock = lock
        with lock:
            return self._stage_vision_embeddings_batch_unlocked(
                processed_batch,
                microbatch_size=microbatch_size,
            )

    def _stage_vision_embeddings_batch_unlocked(
        self,
        processed_batch: list[Any],
        *,
        microbatch_size: int = 8,
    ) -> list[list[list[Any]]] | None:
        """Encode independent camera slices across sessions in microbatches.

        MiniCPM HD slicing produces a source-image tensor and one or more crop
        tensors per frame. Mixing unlike shapes in one call makes the vision
        wrapper pad every item to the largest tensor. Grouping by exact pixel
        shape and target patch grid avoids that waste while preserving each
        request/frame/slice position for the streaming prompt builder.

        Vision encoding is stateless across sessions. Any malformed processor
        output or encoder failure returns ``None`` so the caller can retain the
        existing request-at-a-time fallback.
        """
        if not processed_batch:
            return []
        try:
            microbatch_size = max(1, int(microbatch_size))
        except (TypeError, ValueError):
            microbatch_size = 8

        targets = (self.stage_model, self.thinker, getattr(self.stage_model, "model", None))
        target = next(
            (
                candidate
                for candidate in targets
                if candidate is not None
                and callable(getattr(candidate, "get_vision_hidden_states", None))
                and getattr(candidate, "vpm", None) is not None
            ),
            None,
        )
        if target is None:
            return None

        try:
            import torch

            get_hidden = target.get_vision_hidden_states
            get_hidden_uniform = getattr(
                target,
                "get_vision_hidden_states_uniform",
                None,
            )
            vpm_param = next(target.vpm.parameters())
            device, dtype = vpm_param.device, vpm_param.dtype

            # bucket -> (request index, frame index, slice index, pixel tensor)
            buckets: dict[
                tuple[tuple[int, ...], tuple[int, int]],
                list[tuple[int, int, int, Any]],
            ] = {}
            output: list[list[list[Any | None]]] = []
            for request_index, processed in enumerate(processed_batch):
                pixel_nested = processed["pixel_values"]
                tgt_nested = processed["tgt_sizes"]
                if len(pixel_nested) != len(tgt_nested):
                    return None
                request_output: list[list[Any | None]] = []
                for frame_index, (image_slices, image_tgt) in enumerate(zip(pixel_nested, tgt_nested, strict=True)):
                    tgt_tensor = (image_tgt if hasattr(image_tgt, "reshape") else torch.tensor(image_tgt)).reshape(
                        -1, 2
                    )
                    if len(image_slices) != int(tgt_tensor.shape[0]):
                        return None
                    request_output.append([None] * len(image_slices))
                    for slice_index, (slice_pixels, tgt_row) in enumerate(zip(image_slices, tgt_tensor, strict=True)):
                        tgt_key_raw = tgt_row.detach().cpu().tolist()
                        if len(tgt_key_raw) != 2:
                            return None
                        tgt_key = (int(tgt_key_raw[0]), int(tgt_key_raw[1]))
                        shape_key = tuple(int(dim) for dim in slice_pixels.shape)
                        buckets.setdefault((shape_key, tgt_key), []).append(
                            (request_index, frame_index, slice_index, slice_pixels)
                        )
                output.append(request_output)

            expected_rows = MiniCPMO45DuplexPolicy.VISION_EMBEDS_PER_BLOCK
            with torch.no_grad():
                for (_shape, tgt_key), records in buckets.items():
                    for start in range(0, len(records), microbatch_size):
                        chunk = records[start : start + microbatch_size]
                        pixels_tensor = torch.stack(
                            [record[3] for record in chunk],
                            dim=0,
                        ).to(device=device, dtype=dtype)
                        if callable(get_hidden_uniform):
                            hidden = get_hidden_uniform(
                                pixels_tensor,
                                tgt_key,
                                return_on_vision_device=True,
                            )
                        else:
                            pixels = list(pixels_tensor.unbind(0))
                            tgt_sizes = torch.tensor(
                                [tgt_key] * len(chunk),
                                dtype=torch.int64,
                                device=device,
                            )
                            hidden = get_hidden(
                                {"pixel_values": pixels, "tgt_sizes": tgt_sizes},
                                return_on_vision_device=True,
                            )
                        if len(hidden) != len(chunk):
                            return None
                        for record, block in zip(chunk, hidden, strict=True):
                            block_2d = self._as_2d_tensor(block)
                            if int(block_2d.shape[0]) != expected_rows:
                                return None
                            request_index, frame_index, slice_index, _ = record
                            output[request_index][frame_index][slice_index] = block_2d
        except Exception:  # noqa: BLE001 - retain exact serial fallback
            return None

        finalized: list[list[list[Any]]] = []
        for request_output in output:
            finalized_request: list[list[Any]] = []
            for frame_output in request_output:
                if any(block is None for block in frame_output):
                    return None
                finalized_request.append(list(frame_output))
            finalized.append(finalized_request)
        return finalized
