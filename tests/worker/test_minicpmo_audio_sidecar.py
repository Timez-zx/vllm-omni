from __future__ import annotations

from collections import OrderedDict
from threading import Lock, RLock
from types import SimpleNamespace

import numpy as np
import torch


def test_arrival_audio_sidecar_advances_each_session_in_order() -> None:
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
    )

    class AudioTarget:
        @staticmethod
        def audio_cache_seq_length(cache):
            return 0 if cache is None else int(cache)

        @staticmethod
        def combine_audio_past_key_values(caches):
            assert len(set(caches)) == 1
            return 0 if caches[0] is None else int(caches[0])

        @staticmethod
        def split_audio_past_key_values(cache, batch_size):
            return [int(cache)] * batch_size

        @staticmethod
        def should_reset_audio_past_key_values(_cache, **_kwargs):
            return False

        @staticmethod
        def get_audio_embedding_streaming_batch(data, *, past_key_values, **_kwargs):
            features = data["audio_features"]
            outputs = [
                [torch.full((2, 4), float(features[row].mean()))]
                for row in range(int(features.shape[0]))
            ]
            cache = 0 if past_key_values is None else int(past_key_values)
            return outputs, cache + 1

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = AudioTarget()
    runtime.thinker = runtime.stage_model
    runtime._arrival_audio_cache = OrderedDict()
    runtime._arrival_audio_lineages = {}
    runtime._arrival_audio_retired = OrderedDict()
    runtime._arrival_audio_retired_sessions = OrderedDict()
    runtime._arrival_audio_cache_lock = Lock()
    runtime._audio_execution_lock = RLock()
    runtime._audio_prepare_executor = None
    runtime._configure_streaming_processor = lambda _state: object()
    runtime._restore_streaming_mel_snapshot = lambda _processor, _snapshot: True
    runtime._decode_audio_payload = lambda payload: np.asarray(
        [float(payload["marker"])],
        dtype=np.float32,
    )

    def prepare(state, waveform):
        marker = float(waveform[0])
        return _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=state.audio_chunk_idx,
            start_buffer_len=len(state.audio_buffer),
            units=[
                _MiniCPMO45PreparedAudioUnit(
                    chunk_idx=state.audio_chunk_idx,
                    batch_feature={
                        "audio_features": torch.full((1, 80, 4), marker),
                        "audio_feature_lens": torch.tensor([4]),
                    },
                    consumed_samples=1,
                )
            ],
            remaining_audio_buffer=np.empty(0, dtype=np.float32),
            mel_snapshot_after=object(),
        )

    runtime._prepare_streaming_audio_append = prepare
    jobs = [
        {
            "session_id": "a",
            "incarnation": 1,
            "epoch": 0,
            "audio_preencode_seq": 0,
            "audio_preencode_id": "a0",
            "payload": {"marker": 1},
        },
        {
            "session_id": "b",
            "incarnation": 1,
            "epoch": 0,
            "audio_preencode_seq": 0,
            "audio_preencode_id": "b0",
            "payload": {"marker": 2},
        },
        {
            "session_id": "a",
            "incarnation": 1,
            "epoch": 0,
            "audio_preencode_seq": 1,
            "audio_preencode_id": "a1",
            "payload": {"marker": 3},
        },
    ]

    result = runtime.preencode_arrival_audio(jobs)

    assert result["encoded_jobs"] == 3
    assert result["job_results"] == {"a0": True, "b0": True, "a1": True}
    assert runtime._arrival_audio_lineages[("a", 1)].state.audio_chunk_idx == 2
    assert runtime._arrival_audio_lineages[("b", 1)].state.audio_chunk_idx == 1
    for session_id, seq, preencode_id, marker in (
        ("a", 0, "a0", 1.0),
        ("b", 0, "b0", 2.0),
        ("a", 1, "a1", 3.0),
    ):
        plan = runtime.take_arrival_audio_append(
            session_id=session_id,
            incarnation=1,
            epoch=0,
            audio_preencode_seq=seq,
            audio_preencode_id=preencode_id,
        )
        assert plan is not None
        assert plan.arrival_preencoded is True
        assert plan.audio_past_key_values is None
        torch.testing.assert_close(
            plan.units[0].audio_embeds,
            torch.full((2, 4), marker),
        )

    runtime._audio_prepare_executor.shutdown(wait=True)


def test_arrival_audio_sidecar_resets_critical_cache_before_batch_combine() -> None:
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMForConditionalGeneration,
    )

    class AudioTarget:
        audio_streaming_seq_length = staticmethod(
            MiniCPMO45OmniLLMForConditionalGeneration.audio_streaming_seq_length
        )
        should_reset_audio_past_key_values = (
            MiniCPMO45OmniLLMForConditionalGeneration.should_reset_audio_past_key_values
        )

        def __init__(self) -> None:
            self.apm = SimpleNamespace(
                embed_positions=SimpleNamespace(weight=torch.empty((4, 1)))
            )
            self.combine_calls: list[list[int]] = []
            self.forward_past: list[int | None] = []

        @staticmethod
        def audio_cache_seq_length(cache):
            return 0 if cache is None else int(cache)

        def combine_audio_past_key_values(self, caches):
            normalized = [int(cache) for cache in caches]
            self.combine_calls.append(normalized)
            assert len(set(normalized)) == 1
            return normalized[0]

        @staticmethod
        def split_audio_past_key_values(cache, batch_size):
            return [int(cache)] * batch_size

        def get_audio_embedding_streaming_batch(
            self,
            data,
            *,
            past_key_values,
            **_kwargs,
        ):
            self.forward_past.append(past_key_values)
            features = data["audio_features"]
            outputs = [
                [torch.full((2, 4), float(features[row].mean()))]
                for row in range(int(features.shape[0]))
            ]
            cache = 0 if past_key_values is None else int(past_key_values)
            return outputs, cache + 1

    target = AudioTarget()
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = target
    runtime.thinker = target
    runtime._arrival_audio_cache = OrderedDict()
    runtime._arrival_audio_lineages = {}
    runtime._arrival_audio_retired = OrderedDict()
    runtime._arrival_audio_retired_sessions = OrderedDict()
    runtime._arrival_audio_cache_lock = Lock()
    runtime._audio_execution_lock = RLock()
    runtime._audio_prepare_executor = None
    runtime._configure_streaming_processor = lambda _state: object()
    runtime._restore_streaming_mel_snapshot = lambda _processor, _snapshot: True
    runtime._decode_audio_payload = lambda payload: np.asarray(
        [float(payload["marker"])],
        dtype=np.float32,
    )

    def prepare(state, waveform):
        marker = float(waveform[0])
        return _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=state.audio_chunk_idx,
            start_buffer_len=len(state.audio_buffer),
            units=[
                _MiniCPMO45PreparedAudioUnit(
                    chunk_idx=state.audio_chunk_idx,
                    batch_feature={
                        "audio_features": torch.full((1, 80, 7), marker),
                        "audio_feature_lens": torch.tensor([7]),
                    },
                    consumed_samples=1,
                )
            ],
            remaining_audio_buffer=np.empty(0, dtype=np.float32),
            mel_snapshot_after=object(),
        )

    runtime._prepare_streaming_audio_append = prepare
    for session_id in ("a", "b"):
        lineage = runtime._new_arrival_audio_lineage(
            session_id=session_id,
            epoch=0,
            first_seq=0,
        )
        lineage.state.audio_chunk_idx = 1
        lineage.state.audio_past_key_values = 2
        runtime._arrival_audio_lineages[(session_id, 1)] = lineage

    jobs = [
        {
            "session_id": session_id,
            "incarnation": 1,
            "epoch": 0,
            "audio_preencode_seq": seq,
            "audio_preencode_id": f"{session_id}{seq}",
            "payload": {"marker": marker},
        }
        for seq, marker in ((0, 1), (1, 2))
        for session_id in ("a", "b")
    ]

    try:
        result = runtime.preencode_arrival_audio(jobs)
    finally:
        if runtime._audio_prepare_executor is not None:
            runtime._audio_prepare_executor.shutdown(wait=True)

    assert result["encoded_jobs"] == 4
    assert result["job_results"] == {
        "a0": True,
        "b0": True,
        "a1": True,
        "b1": True,
    }
    # cache=2 plus this unit's seq=2 reaches the max=4 boundary. The first
    # cohort must enter Whisper with past=None, without materializing a batched
    # copy of the old cache. Its fresh cache then lets the next cohort proceed.
    assert target.forward_past == [None, 1]
    assert target.combine_calls == [[1, 1]]
    for session_id in ("a", "b"):
        lineage = runtime._arrival_audio_lineages[(session_id, 1)]
        assert lineage.next_seq == 2
        assert lineage.state.audio_past_key_values == 2


def test_arrival_audio_commit_failure_rolls_back_lineage_and_plan() -> None:
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45ArrivalAudioLineage,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
        _MiniCPMO45Stage0SessionState,
    )

    class FailingCache(OrderedDict):
        def __setitem__(self, _key, _value):
            raise RuntimeError("injected cache commit failure")

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime._arrival_audio_cache = FailingCache()
    runtime._arrival_audio_retired = OrderedDict()
    runtime._arrival_audio_retired_sessions = OrderedDict()
    runtime._arrival_audio_cache_lock = Lock()
    runtime._configure_streaming_processor = lambda _state: object()
    runtime._restore_streaming_mel_snapshot = lambda _processor, _snapshot: True

    old_buffer = np.asarray([9.0], dtype=np.float32)
    state = _MiniCPMO45Stage0SessionState(
        session_id="a",
        audio_buffer=old_buffer,
        audio_chunk_idx=5,
        audio_past_key_values="old-cache",
    )
    lineage = _MiniCPMO45ArrivalAudioLineage(
        epoch=0,
        next_seq=7,
        state=state,
    )
    original_embed = torch.ones((2, 4))
    original_feature = {"audio_features": torch.ones((1, 80, 7))}
    mel_snapshot = object()
    plan = _MiniCPMO45PreparedAudioAppend(
        start_chunk_idx=5,
        start_buffer_len=1,
        units=[
            _MiniCPMO45PreparedAudioUnit(
                chunk_idx=5,
                batch_feature=original_feature,
                consumed_samples=1,
                audio_embeds=original_embed,
            )
        ],
        remaining_audio_buffer=np.asarray([1.0], dtype=np.float32),
        mel_snapshot_after=mel_snapshot,
        audio_past_key_values="new-cache",
        encoded=True,
    )

    committed = runtime._commit_arrival_audio_plan(
        identity=("a", 1, 0, 7, "a7"),
        lineage=lineage,
        plan=plan,
    )

    assert committed is False
    assert state.audio_buffer is old_buffer
    assert state.audio_chunk_idx == 5
    assert state.audio_past_key_values == "old-cache"
    assert lineage.next_seq == 7
    assert plan.audio_past_key_values == "new-cache"
    assert plan.mel_snapshot_after is mel_snapshot
    assert plan.arrival_preencoded is False
    assert plan.units[0].audio_embeds is original_embed
    assert plan.units[0].batch_feature is original_feature
    assert not runtime._arrival_audio_cache


def test_preprocess_batch_consumes_arrival_audio_without_local_decode() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        _MINICPMO45_BATCHED_VISION_KEY,
        MiniCPMO45OmniForConditionalGeneration,
    )

    audio_plan = object()

    class Helper:
        def __init__(self):
            self.state = SimpleNamespace(prepared_append_identity=None)
            self.decode_calls = 0

        def get_or_create_session_state(self, *_args, **_kwargs):
            return self.state

        @staticmethod
        def take_arrival_audio_append(**_kwargs):
            return audio_plan

        def _decode_audio_payload(self, _payload):
            self.decode_calls += 1
            return np.ones(4, dtype=np.float32)

    helper = Helper()
    model = MiniCPMO45OmniForConditionalGeneration.__new__(
        MiniCPMO45OmniForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._duplex_data_plane_helper = lambda: helper
    info = {
        "duplex": {
            "data_plane": True,
            "session_id": "session-a",
            "incarnation": 1,
            "epoch": 2,
            "seq": 3,
            "payload": {
                "audio": "unused",
                "audio_preencode_id": "audio-a",
                "audio_preencode_seq": 7,
            },
        }
    }

    model.preprocess_batch(
        req_ids=["req-a"],
        model_intermediate_buffer={"req-a": info},
        device=torch.device("cpu"),
    )

    cached = info[_MINICPMO45_BATCHED_VISION_KEY]
    assert cached["identity"] == ("session-a", 1, 2, 3)
    assert cached["audio_plan"] is audio_plan
    assert cached["audio_source"] == "arrival"
    assert helper.decode_calls == 0


def test_arrival_audio_exact_identity_replay_is_idempotent() -> None:
    """A lost RPC acknowledgement must not re-encode or advance twice."""
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
    )

    class AudioTarget:
        def __init__(self) -> None:
            self.encode_calls = 0

        def get_audio_embedding_streaming_batch(
            self,
            data,
            *,
            past_key_values,
            **_kwargs,
        ):
            self.encode_calls += 1
            marker = float(data["audio_features"].mean())
            cache = 0 if past_key_values is None else int(past_key_values)
            return [[torch.full((2, 4), marker)]], cache + 1

    target = AudioTarget()
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(
        MiniCPMO45Stage0DuplexRuntime
    )
    runtime.stage_model = target
    runtime.thinker = target
    runtime._arrival_audio_cache = OrderedDict()
    runtime._arrival_audio_lineages = {}
    runtime._arrival_audio_retired = OrderedDict()
    runtime._arrival_audio_retired_sessions = OrderedDict()
    runtime._arrival_audio_cache_lock = Lock()
    runtime._audio_execution_lock = RLock()
    runtime._audio_prepare_executor = None
    runtime._configure_streaming_processor = lambda _state: object()
    runtime._restore_streaming_mel_snapshot = lambda _processor, _snapshot: True
    runtime._decode_audio_payload = lambda payload: np.asarray(
        [float(payload["marker"])],
        dtype=np.float32,
    )

    def prepare(state, waveform):
        marker = float(waveform[0])
        return _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=state.audio_chunk_idx,
            start_buffer_len=len(state.audio_buffer),
            units=[
                _MiniCPMO45PreparedAudioUnit(
                    chunk_idx=state.audio_chunk_idx,
                    batch_feature={
                        "audio_features": torch.full((1, 80, 4), marker),
                        "audio_feature_lens": torch.tensor([4]),
                    },
                    consumed_samples=1,
                )
            ],
            remaining_audio_buffer=np.empty(0, dtype=np.float32),
            mel_snapshot_after=object(),
        )

    runtime._prepare_streaming_audio_append = prepare
    first_job = {
        "session_id": "replay",
        "incarnation": 1,
        "epoch": 0,
        "audio_preencode_seq": 0,
        "audio_preencode_id": "replay-0",
        "payload": {"marker": 1},
    }
    second_job = {
        **first_job,
        "audio_preencode_seq": 1,
        "audio_preencode_id": "replay-1",
        "payload": {"marker": 2},
    }

    try:
        first = runtime.preencode_arrival_audio([first_job])
        lineage = runtime._arrival_audio_lineages[("replay", 1)]
        assert first["job_results"] == {"replay-0": True}
        assert lineage.next_seq == 1
        assert target.encode_calls == 1

        replay = runtime.preencode_arrival_audio([dict(first_job)])
        assert replay["job_results"] == {"replay-0": True}
        assert lineage.next_seq == 1
        assert target.encode_calls == 1

        following = runtime.preencode_arrival_audio([second_job])
        assert following["job_results"] == {"replay-1": True}
        assert lineage.next_seq == 2
        assert target.encode_calls == 2
    finally:
        if runtime._audio_prepare_executor is not None:
            runtime._audio_prepare_executor.shutdown(wait=True)


def test_arrival_audio_sidecar_remains_active_across_context_rollover() -> None:
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
        _MiniCPMO45Stage0SessionState,
    )

    class StageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(
        MiniCPMO45Stage0DuplexRuntime
    )
    runtime.stage_model = StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sidecar-rollover",
        context_embeds=[runtime._embed_token(50)],
        context_token_ids=[50],
    )

    def arrival_plan(chunk_idx: int, marker: float):
        return _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=chunk_idx,
            start_buffer_len=0,
            units=[
                _MiniCPMO45PreparedAudioUnit(
                    chunk_idx=chunk_idx,
                    batch_feature=None,
                    consumed_samples=4,
                    audio_embeds=torch.full((1, 2), marker),
                )
            ],
            remaining_audio_buffer=np.empty(0, dtype=np.float32),
            mel_snapshot_after=None,
            encoded=True,
            arrival_preencoded=True,
            arrival_preencode_seq=chunk_idx,
            arrival_preencode_id=f"audio-{chunk_idx}",
        )

    first = runtime._stage_prefill_embeddings_only(
        state,
        None,
        preprocessed_audio=arrival_plan(0, 0.25),
        epoch=0,
        seq=1,
    )
    assert first["success"] is True
    assert first["duplex_arrival_audio_units"] == 1
    assert first["duplex_audio_fallback_units"] == 0
    assert state.audio_sidecar_active is True
    assert state.audio_chunk_idx == 1

    state.pending_terminator_token = 3
    state.last_terminator_token = 3
    state.current_segment_output_tokens = [21, 3]
    rollover = runtime._stage_prefill_embeddings_only(
        state,
        None,
        preprocessed_audio=arrival_plan(1, 0.5),
        epoch=0,
        seq=2,
        context_rollover=True,
    )

    assert rollover["success"] is True
    assert rollover["context_rollover"] is True
    assert rollover["input_token_ids"] == [50, 1, 11, 21, 3, 2, 1, 11]
    assert rollover["duplex_arrival_audio_units"] == 1
    assert rollover["duplex_audio_fallback_units"] == 0
    assert state.audio_sidecar_active is True
    assert state.audio_chunk_idx == 2
    assert state.context_rollovers == 1
