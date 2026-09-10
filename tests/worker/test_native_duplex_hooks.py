from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pinned_prefix_must_cover_entire_system_and_reference_context():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import MiniCPMO45Stage0DuplexRuntime

    helper = SimpleNamespace(stage_model=SimpleNamespace(config=SimpleNamespace(vllm_omni_pinned_prefix_tokens=128)))
    state = SimpleNamespace(context_token_ids=[1] * 128)
    validate = MiniCPMO45Stage0DuplexRuntime._validate_pinned_session_context
    validate(helper, state)
    state.context_token_ids.append(1)
    with pytest.raises(ValueError, match="exceeding pinned prefix"):
        validate(helper, state)
    helper.stage_model.config.vllm_omni_pinned_prefix_tokens = 0
    validate(helper, state)


def test_pd_feedback_preserves_last_sample_even_without_native_terminator():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import MiniCPMO45Stage0DuplexRuntime

    helper = SimpleNamespace(listen_token_id=1, chunk_eos_token_id=2,
                             chunk_tts_eos_token_id=3, turn_eos_token_id=4)
    state = SimpleNamespace(pd_feedback_append_identity=None)
    MiniCPMO45Stage0DuplexRuntime.apply_pd_decode_feedback(helper, state, [99, 100], epoch=0, seq=278)
    assert state.pd_feedback_token_ids == [99, 100]
    assert state.pending_terminator_token == 100
    assert state.last_terminator_token is None
    # The replay prefix plus the last sampled token accounts for all feedback,
    # even if a finite request ended on a budget or a different EOS token.
    assert state.pd_feedback_token_ids[:-1] + [state.pending_terminator_token] == [99, 100]


@pytest.mark.parametrize(
    ("feedback", "delta_len"),
    [
        ([101, 151705], 491),
        ([101, 102, 151705], 492),
    ],
)
def test_minicpmo_duplex_delta_slice_accepts_exact_feedback_expanded_budget(feedback, delta_len):
    from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
        duplex_feedback_scheduler_rows,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    base_budget = 490
    scheduler_budget = base_budget + duplex_feedback_scheduler_rows(feedback)
    embeds = torch.arange(delta_len * 2, dtype=torch.float32).reshape(delta_len, 2)
    token_ids = list(range(delta_len))

    sliced, sliced_ids, delta_start = MiniCPMO45OmniForConditionalGeneration._slice_duplex_prompt_delta(
        embeds,
        token_ids,
        prompt_len=10_000 + delta_len,
        token_offset=10_001,
        span_len=delta_len - 1,
        rebase_prompt=False,
        scheduler_token_budget=scheduler_budget,
    )

    assert delta_start == 10_000
    assert torch.equal(sliced, embeds[1:])
    assert sliced_ids == token_ids[1:]


def test_minicpmo_duplex_delta_slice_rejects_budget_mismatch_and_prefix_replay():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    embeds = torch.zeros((3, 2), dtype=torch.float32)
    token_ids = [11, 12, 13]

    with pytest.raises(RuntimeError, match="refusing to pad or truncate"):
        MiniCPMO45OmniForConditionalGeneration._slice_duplex_prompt_delta(
            embeds,
            token_ids,
            prompt_len=103,
            token_offset=100,
            span_len=3,
            rebase_prompt=False,
            scheduler_token_budget=2,
        )

    with pytest.raises(RuntimeError, match="prefix KV is unavailable"):
        MiniCPMO45OmniForConditionalGeneration._slice_duplex_prompt_delta(
            embeds,
            token_ids,
            prompt_len=103,
            token_offset=99,
            span_len=3,
            rebase_prompt=False,
            scheduler_token_budget=3,
        )


def test_minicpmo_duplex_delta_slice_rebase_owns_complete_prompt():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    token_ids = [11, 12, 13, 14]
    sliced, sliced_ids, delta_start = MiniCPMO45OmniForConditionalGeneration._slice_duplex_prompt_delta(
        embeds,
        token_ids,
        prompt_len=4,
        token_offset=2,
        span_len=2,
        rebase_prompt=True,
        scheduler_token_budget=999,
    )

    assert delta_start == 0
    assert torch.equal(sliced, embeds[2:])
    assert sliced_ids == token_ids[2:]

    with pytest.raises(RuntimeError, match="complete prompt"):
        MiniCPMO45OmniForConditionalGeneration._slice_duplex_prompt_delta(
            embeds,
            token_ids,
            prompt_len=5,
            token_offset=0,
            span_len=4,
            rebase_prompt=True,
            scheduler_token_budget=4,
        )


def test_minicpmo_pd_prefill_preprocess_materializes_only_the_append_delta():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        @staticmethod
        def get_or_create_session_state(*_args, **_kwargs):
            return object()

        @staticmethod
        def _decode_audio_payload(_payload):
            return np.zeros(4, dtype=np.float32)

        @staticmethod
        def _decode_video_frames_payload(_payload):
            return []

        @staticmethod
        def _stage_prefill_embeddings_only(*_args, **_kwargs):
            return {
                "success": True,
                "inputs_embeds": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
                "input_token_ids": [11, 12, 13],
                "rebase_prompt": False,
                "delta_num_tokens": 3,
            }

        @staticmethod
        def stage_padding_token_id():
            pytest.fail("P must not materialize a full-prefix padding list")

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._minicpmo_pd_prefill = True
    model._duplex_data_plane_helper = lambda: _Helper()
    embedding_lookup_sizes = []

    def get_input_embeddings(ids):
        embedding_lookup_sizes.append(int(ids.numel()))
        return torch.zeros((ids.numel(), 2), dtype=torch.float32)

    model.get_input_embeddings = get_input_embeddings

    req_ids, req_embeds, update = model.preprocess(
        input_ids=torch.zeros(3, dtype=torch.long),
        duplex_prompt_len=103,
        duplex_token_offset=100,
        duplex={
            "data_plane": True,
            "session_id": "sid-delta-only",
            "incarnation": 0,
            "epoch": 0,
            "seq": 2,
            "payload": {},
            "scheduler_token_budget": 3,
        },
    )

    assert req_ids.tolist() == [11, 12, 13]
    assert req_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert update["duplex"]["duplex_prompt_delta_start"] == 100
    assert update["duplex"]["duplex_prompt_delta_token_ids"] == [11, 12, 13]
    assert "duplex_prompt_token_ids" not in update["duplex"]
    assert embedding_lookup_sizes == [1]


def test_minicpmo_failed_prefill_does_not_claim_video_frame_consumption():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        @staticmethod
        def get_or_create_session_state(*_args, **_kwargs):
            return object()

        @staticmethod
        def _decode_audio_payload(_payload):
            return np.zeros(4, dtype=np.float32)

        @staticmethod
        def _decode_video_frames_payload(_payload):
            return [object()]

        @staticmethod
        def _stage_prefill_embeddings_only(*_args, **_kwargs):
            return {"success": False, "reason": "stale transactional plan"}

    model = MiniCPMO45OmniForConditionalGeneration.__new__(
        MiniCPMO45OmniForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._minicpmo_pd_prefill = True
    model._duplex_data_plane_helper = lambda: _Helper()
    model.get_input_embeddings = lambda ids: torch.zeros(
        (ids.numel(), 2),
        dtype=torch.float32,
    )

    _, _, update = model.preprocess(
        input_ids=torch.zeros(3, dtype=torch.long),
        duplex_prompt_len=3,
        duplex_token_offset=0,
        duplex={
            "data_plane": True,
            "session_id": "sid-failed-frame",
            "incarnation": 0,
            "epoch": 0,
            "seq": 1,
            "payload": {},
            "scheduler_token_budget": 3,
        },
    )

    assert update["duplex"]["success"] is False
    assert "duplex_input_video_frames" not in update["duplex"]
    assert "duplex_arrival_video_frames" not in update["duplex"]
    assert "duplex_vision_fallback_frames" not in update["duplex"]


def test_minicpmo_chunk_replay_reuses_prepared_input_and_original_frame_audit():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    state = SimpleNamespace(
        prepared_append_identity=(0, 1),
        prepared_inputs_embeds=object(),
    )

    class _Helper:
        @staticmethod
        def get_or_create_session_state(*_args, **_kwargs):
            return state

        @staticmethod
        def _decode_audio_payload(_payload):
            pytest.fail("a cached physical append must not decode audio again")

        @staticmethod
        def _decode_video_frames_payload(_payload):
            pytest.fail("a cached physical append must not decode video again")

        @staticmethod
        def _stage_prefill_embeddings_only(
            _state,
            audio_waveform,
            *,
            video_frames,
            **_kwargs,
        ):
            assert audio_waveform is None
            assert video_frames is None
            return {
                "success": True,
                "inputs_embeds": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
                "input_token_ids": [11, 12, 13],
                "rebase_prompt": False,
                "delta_num_tokens": 3,
                "duplex_input_video_frames": 1,
                "duplex_arrival_video_frames": 1,
                "duplex_vision_fallback_frames": 0,
            }

    model = MiniCPMO45OmniForConditionalGeneration.__new__(
        MiniCPMO45OmniForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._minicpmo_pd_prefill = True
    model._duplex_data_plane_helper = lambda: _Helper()
    model.get_input_embeddings = lambda ids: torch.zeros(
        (ids.numel(), 2),
        dtype=torch.float32,
    )

    _, _, update = model.preprocess(
        input_ids=torch.zeros(3, dtype=torch.long),
        duplex_prompt_len=103,
        duplex_token_offset=100,
        duplex={
            "data_plane": True,
            "session_id": "sid-replayed-frame",
            "incarnation": 0,
            "epoch": 0,
            "seq": 1,
            "payload": {"video_frames": ["encoded-payload"]},
            "scheduler_token_budget": 3,
        },
    )

    assert update["duplex"]["duplex_input_video_frames"] == 1
    assert update["duplex"]["duplex_arrival_video_frames"] == 1
    assert update["duplex"]["duplex_vision_fallback_frames"] == 0


def test_minicpmo_pd_prefill_preprocess_accepts_only_exact_compact_engine_rebase():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    calls = []

    class _Helper:
        @staticmethod
        def get_or_create_session_state(*_args, **_kwargs):
            return object()

        @staticmethod
        def _decode_audio_payload(_payload):
            return np.zeros(4, dtype=np.float32)

        @staticmethod
        def _decode_video_frames_payload(_payload):
            return []

        @staticmethod
        def _stage_prefill_embeddings_only(*_args, **kwargs):
            calls.append(kwargs)
            return {
                "success": True,
                "inputs_embeds": torch.arange(
                    10,
                    dtype=torch.float32,
                ).reshape(5, 2),
                "input_token_ids": [21, 22, 23, 24, 25],
                "rebase_prompt": True,
                "engine_rebase_prompt": True,
                "delta_num_tokens": 5,
            }

        @staticmethod
        def stage_padding_token_id():
            pytest.fail("P must not materialize a padding prompt")

    model = MiniCPMO45OmniForConditionalGeneration.__new__(
        MiniCPMO45OmniForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._minicpmo_pd_prefill = True
    model._duplex_data_plane_helper = lambda: _Helper()
    model.get_input_embeddings = lambda ids: torch.zeros(
        (ids.numel(), 2),
        dtype=torch.float32,
    )
    duplex = {
        "data_plane": True,
        "session_id": "sid-engine-rebase",
        "incarnation": 0,
        "epoch": 1,
        "seq": 2,
        "payload": {},
        "scheduler_token_budget": 3,
        "compact_rebase_prefix_tokens": 2,
    }

    req_ids, req_embeds, update = model.preprocess(
        input_ids=torch.zeros(5, dtype=torch.long),
        duplex_prompt_len=5,
        duplex_token_offset=0,
        duplex=duplex,
    )

    assert calls[0]["engine_rebase"] is True
    assert req_ids.tolist() == [21, 22, 23, 24, 25]
    assert req_embeds.shape == (5, 2)
    assert update["duplex"]["duplex_prompt_delta_start"] == 0

    with pytest.raises(RuntimeError, match="did not install the exact"):
        model.preprocess(
            input_ids=torch.zeros(6, dtype=torch.long),
            duplex_prompt_len=6,
            duplex_token_offset=0,
            duplex=duplex,
        )
    assert len(calls) == 1


def test_gpu_ar_worker_routes_minicpmo_vision_preencode_to_loaded_model():
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker

    calls = []
    model = SimpleNamespace(
        preencode_duplex_vision=lambda jobs: calls.append(jobs) or {"supported": True, "encoded_frames": len(jobs)}
    )
    worker = GPUARWorker.__new__(GPUARWorker)
    worker.model_runner = SimpleNamespace(model=model)
    jobs = [{"preencode_ids": ["frame-a"]}]
    worker.device = torch.device("cpu")

    result = worker.preencode_minicpmo45_vision(jobs)

    assert result == {"supported": True, "encoded_frames": 1}
    assert calls == [jobs]


def test_gpu_ar_worker_routes_minicpmo_audio_preencode_to_loaded_model():
    from vllm_omni.worker.gpu_ar_worker import GPUARWorker

    calls = []
    model = SimpleNamespace(
        preencode_duplex_audio=lambda jobs: calls.append(jobs)
        or {
            "supported": True,
            "encoded_jobs": len(jobs),
            "job_results": {"audio-1": True},
        }
    )
    worker = GPUARWorker.__new__(GPUARWorker)
    worker.model_runner = SimpleNamespace(model=model)
    jobs = [{"session_id": "sid-audio", "audio": [0.0, 0.5]}]
    worker.device = torch.device("cpu")

    result = worker.preencode_minicpmo45_audio(jobs)

    assert result == {
        "supported": True,
        "encoded_jobs": 1,
        "job_results": {"audio-1": True},
    }
    assert calls == [jobs]


@pytest.mark.parametrize(
    (
        "pd_prefill",
        "pd_decode",
        "expects_hidden_payload",
        "expects_side_cache",
    ),
    [
        (True, False, False, False),
        (False, True, True, False),
        (False, False, True, True),
    ],
)
def test_minicpmo_pd_thinker_disables_redundant_prefix_tensor_cache(
    monkeypatch,
    pd_prefill: bool,
    pd_decode: bool,
    expects_hidden_payload: bool,
    expects_side_cache: bool,
):
    from vllm_omni.model_executor.models.minicpmo_4_5 import (
        minicpmo_4_5_omni as model_module,
    )

    class DummyThinker:
        def make_empty_intermediate_tensors(self):
            return None

    monkeypatch.setattr(
        model_module,
        "init_vllm_registered_model",
        lambda **_: DummyThinker(),
    )
    hf_config = SimpleNamespace(
        vllm_omni_minicpmo_pd_prefill=pd_prefill,
        vllm_omni_minicpmo_pd_decode=pd_decode,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            multimodal_config=SimpleNamespace(),
            model_stage="llm",
        )
    )

    model = model_module.MiniCPMO45OmniForConditionalGeneration(vllm_config=vllm_config)

    assert model.omni_pooler_payload_include_hidden is expects_hidden_payload
    assert getattr(model.thinker, "omni_pooler_payload_include_hidden", True) is expects_hidden_payload
    assert model.requires_full_prefix_cached_hidden_states is expects_side_cache
    assert model.requires_full_prefix_cached_multimodal_outputs is expects_side_cache
    assert getattr(model.thinker, "requires_full_prefix_cached_hidden_states", True) is expects_side_cache
    assert (
        getattr(
            model.thinker,
            "requires_full_prefix_cached_multimodal_outputs",
            True,
        )
        is expects_side_cache
    )


def _minicpmo_duplex_policy_case(
    state: SimpleNamespace,
    payload: dict[str, object],
):
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    session_key = ("sid-policy", 1)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": 7,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-policy",
        session_id=session_key[0],
        incarnation=session_key[1],
        seq=3,
        payload=payload,
        max_tokens=20,
    )
    return model, row


def test_minicpmo_model_hook_owns_duplex_sampling_rows_and_force_listen():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    listen_id = 7
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": listen_id,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-hook", 2): state})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-hook",
        session_id="sid-hook",
        incarnation=2,
        seq=3,
        payload={"force_listen": True, "is_speech": True},
        max_tokens=20,
    )

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert model._minicpmo45_active_duplex_rows == [0]
    assert model._minicpmo45_duplex_row_sessions == {0: ("sid-hook", 2)}
    assert model._minicpmo45_duplex_row_payloads == {0: row.payload}
    assert model._minicpmo45_duplex_row_max_tokens == {0: 20}
    assert logits[0, listen_id].item() == 0.0
    assert torch.isneginf(logits[0, :listen_id]).all()
    assert torch.isneginf(logits[0, listen_id + 1 :]).all()


@pytest.mark.parametrize("payload", [{"is_speech": False}, {"is_speech": None}, {}, {"force_listen": False}])
def test_minicpmo_model_hook_silence_does_not_override_native_decision(payload):
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, payload)
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 10] = 20.0
    original_logits = logits.clone()
    original_state = vars(state).copy()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    # Official streaming_generate allows a new reply even over silence; only
    # an explicit force-listen instruction may replace the model logits.
    assert torch.equal(logits, original_logits)
    assert vars(state) == original_state


@pytest.mark.parametrize("is_speech", [False, None, True])
def test_minicpmo_model_hook_explicit_force_listen_only_applies_to_first_sample(is_speech):
    from dataclasses import replace

    state = SimpleNamespace(current_turn_ended=True, last_terminator_token=9, pending_terminator_token=9)
    model, row = _minicpmo_duplex_policy_case(state, {"force_listen": True, "is_speech": is_speech})
    original_state = vars(state).copy()
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    assert torch.isfinite(logits).sum().item() == 1
    assert logits[0, 7].item() == 0.0
    assert vars(state) == original_state

    # A duplicate hook or continuation inside the unit is not another force.
    next_logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (row,))
    assert torch.equal(next_logits, torch.arange(16, dtype=torch.float32).reshape(1, 16))
    state.current_segment_output_tokens = [9]
    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (replace(row, seq=4),))
    assert torch.equal(next_logits, torch.arange(16, dtype=torch.float32).reshape(1, 16))

    # A new unit can independently request force-listen again.
    state.current_segment_output_tokens = []
    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (replace(row, seq=4),))
    assert torch.isfinite(next_logits).sum().item() == 1
    assert next_logits[0, 7].item() == 0.0


def test_minicpmo_model_hook_mixed_rows_only_force_explicit_request():
    from dataclasses import replace

    states = [
        SimpleNamespace(current_turn_ended=True, last_terminator_token=9, pending_terminator_token=9)
        for _ in range(4)
    ]
    payloads = [{"force_listen": True}, {"is_speech": False}, {}, {"is_speech": True}]
    model, template = _minicpmo_duplex_policy_case(states[0], payloads[0])
    rows = tuple(
        replace(template, row_idx=i, request_id=f"req-{i}", session_id=f"session-{i}", payload=payload)
        for i, payload in enumerate(payloads)
    )
    model._minicpmo45_duplex_data_plane_helper.sessions = {
        (row.session_id, row.incarnation): state for row, state in zip(rows, states)
    }
    logits = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    original_logits = logits.clone()
    original_states = [vars(state).copy() for state in states]
    model.prepare_duplex_sampling(logits, SimpleNamespace(), rows)
    assert torch.isfinite(logits[0]).sum().item() == 1
    assert logits[0, 7].item() == 0.0
    assert torch.equal(logits[1:], original_logits[1:])
    assert [vars(state) for state in states] == original_states


def test_minicpmo_model_hook_decode_does_not_reapply_p_force_listen():
    from vllm_omni.experimental.fullduplex.minicpmo45.sampling_state import (
        SAMPLING_STATE_KEY,
        DecodeSamplingState,
        pack_sampling_state,
    )

    state = DecodeSamplingState(current_segment_output_tokens=[7])
    snapshot = pack_sampling_state(state, incarnation=1, epoch=None, seq=3)
    model, row = _minicpmo_duplex_policy_case(
        state, {"force_listen": True, "is_speech": False, SAMPLING_STATE_KEY: snapshot}
    )
    model._minicpmo_pd_decode = True
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    original_logits = logits.clone()
    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    assert torch.equal(logits, original_logits)
    assert model._minicpmo45_pd_sampling_states[row.request_id].current_segment_output_tokens == [7]


def test_minicpmo_model_hook_pending_speech_after_turn_eos_allows_silence_sampling():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)
    assert state.pending_speech_context is True


def test_minicpmo_model_hook_speech_row_does_not_rewrite_model_state():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=False,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 8] = -2.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert state.current_turn_ended is True
    assert state.last_terminator_token == 9
    assert state.pending_speech_context is False
    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_old_response_output_does_not_clear_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=None,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}

    model._record_minicpmo45_duplex_terminator(
        0,
        10,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )

    assert state.pending_speech_context is True
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_new_response_output_clears_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=7,
        pending_terminator_token=7,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}

    model._record_minicpmo45_duplex_terminator(
        0,
        8,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )

    assert state.pending_speech_context is False
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_empty_speak_envelope_preserves_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
        pending_speech_response_open=False,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    token_ids = {
        "listen_token_id": 7,
        "tts_bos_token_id": 8,
        "chunk_eos_token_id": -1,
        "chunk_tts_eos_token_id": -1,
        "turn_eos_token_id": 9,
    }

    model._record_minicpmo45_duplex_terminator(0, 8, token_ids)
    model._record_minicpmo45_duplex_terminator(0, 9, token_ids)

    assert state.current_turn_ended is True
    assert state.pending_speech_context is True
    assert state.pending_speech_response_open is False

    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_second_new_response_step_is_not_forced_to_listen():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    model._record_minicpmo45_duplex_terminator(
        0,
        8,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )
    next_logits = torch.zeros((1, 16), dtype=torch.float32)
    next_logits[0, 7] = 30.0
    next_logits[0, 10] = 20.0
    next_original_logits = next_logits.clone()

    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)
    assert torch.equal(next_logits, next_original_logits)
    assert state.pending_speech_context is False
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_same_speech_row_second_step_does_not_rearm_pending_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    model._record_minicpmo45_duplex_terminator(
        0,
        10,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )
    next_logits = torch.zeros((1, 16), dtype=torch.float32)
    next_logits[0, 7] = 30.0
    next_logits[0, 8] = -2.0
    next_logits[0, 10] = 20.0
    original_next_logits = next_logits.clone()

    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (row,))

    assert state.pending_speech_context is False
    assert state.current_turn_ended is False
    assert torch.equal(next_logits, original_next_logits)


def test_minicpmo_model_hook_mid_turn_speech_preserves_logits_for_post_sample_redirect():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 8] = -2.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_ignores_serving_new_user_turn_marker():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=8,
        pending_terminator_token=8,
    )
    model, row = _minicpmo_duplex_policy_case(
        state,
        {"is_speech": True, "new_user_turn": True, "force_speak": True},
    )
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 5.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()
    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert state.current_turn_ended is False
    assert state.last_terminator_token == 8
    assert torch.equal(logits, original_logits)


def test_generic_ar_runner_builds_typed_duplex_sampling_rows():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingHelper
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-duplex", "req-plain"])
    runner.model_intermediate_buffer = {
        "req-duplex": {
            "duplex": {
                "data_plane": True,
                "session_id": "sid-runner-hook",
                "incarnation": 4,
                "seq": 4,
                "payload": {"is_speech": True},
            }
        }
    }
    runner.requests = {
        "req-duplex": SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=32),
        )
    }
    helper = DuplexSamplingHelper()
    helper.active_request_ids = {"req-duplex"}

    rows = helper.rows(runner)

    assert len(rows) == 1
    assert rows[0].row_idx == 0
    assert rows[0].request_id == "req-duplex"
    assert rows[0].session_id == "sid-runner-hook"
    assert rows[0].incarnation == 4
    assert rows[0].seq == 4
    assert rows[0].payload == {"is_speech": True}
    assert rows[0].should_sample is True
    runner.discard_request_mask = SimpleNamespace(np=[True, False])
    assert helper.rows(runner)[0].should_sample is False
    assert rows[0].max_tokens == 32


def test_generic_ar_runner_skips_duplex_rows_without_model_hook():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=object(),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(sample=lambda *_args, **_kwargs: None, prefer_model_sampler=False)
    runner.sampler = lambda **_kwargs: "standard-sampler"
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids={"req-duplex"},
        rows=lambda *_args: pytest.fail("duplex rows built without a model hook"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "standard-sampler"


def test_plain_model_hook_resolution_does_not_allocate_duplex_tracking():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.model = SimpleNamespace()
    runner._duplex_sampling_hook = None
    runner._duplex_sampling_hook_resolved = False

    assert runner._resolve_duplex_sampling_hook() is None
    assert not hasattr(runner, "_duplex_sampling_helper")


def test_minicpmo_non_duplex_sample_skips_duplex_row_scan():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    calls = []
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=[]),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(
        sample=lambda *_args, **_kwargs: "model-sampler",
        prefer_model_sampler=True,
        skips_model_sampler_output_token_history=True,
        prepare_duplex_sampling=lambda *_args: calls.append("prepare"),
    )
    runner.sampler = SimpleNamespace()
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids=set(),
        hook_active=False,
        rows=lambda *_args: pytest.fail("non-duplex MiniCPM scanned request rows"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert calls == []


def test_minicpmo_duplex_sample_clears_stale_rows_once_without_scanning():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    calls = []
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=[]),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(
        sample=lambda *_args, **_kwargs: "model-sampler",
        prefer_model_sampler=True,
        skips_model_sampler_output_token_history=True,
        prepare_duplex_sampling=lambda _logits, _metadata, rows: calls.append(rows),
    )
    runner.sampler = SimpleNamespace()
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids=set(),
        hook_active=True,
        rows=lambda *_args: pytest.fail("duplex cleanup scanned request rows"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert calls == [()]


def test_generic_ar_runner_has_no_minicpmo_sampler_state_or_typeerror_probe():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    source = inspect.getsource(GPUARModelRunner)

    assert "_minicpmo45_duplex_row" not in source
    assert "_minicpmo45_native_duplex_token_ids" not in source
    assert 'if "duplex_rows" not in str(exc)' not in source


def test_minicpmo_model_cleans_incarnation_state_when_request_finishes():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    request_id = "duplex-sid-cleanup-i3-e0-stage0"
    session_key = ("sid-cleanup", 3)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model = SimpleNamespace()
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: object()})
    model._minicpmo45_duplex_request_sessions = {request_id: session_key}
    model._minicpmo45_force_listen_applied_segments = {
        (request_id, 1),
        ("duplex-sid-other-e0-stage0", 2),
    }

    model.on_requests_finished({request_id})

    assert model._minicpmo45_duplex_data_plane_helper.sessions == {}
    assert model._minicpmo45_duplex_request_sessions == {}
    assert model._minicpmo45_force_listen_applied_segments == {
        ("duplex-sid-other-e0-stage0", 2),
    }


def test_minicpmo_stage0_routes_duplex_metadata_per_batched_request():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )
    from vllm_omni.utils.mm_outputs import to_payload_element

    class _Thinker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("device_anchor", torch.zeros(1))

        def forward(self, *, input_ids, **kwargs):
            del kwargs
            return torch.arange(input_ids.numel() * 4, dtype=torch.float32).reshape(input_ids.numel(), 4)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model.thinker = _Thinker()
    output = model.forward(
        input_ids=torch.tensor([1, 2, 3, 4]),
        positions=torch.arange(4),
        runtime_additional_information=[
            {
                "global_request_id": ["req-a"],
                "duplex": {
                    "duplex_prompt_token_ids": [101, 102],
                    "special_token_ids": {"listen_token_id": 701},
                    "duplex_input_video_frames": 1,
                    "duplex_arrival_video_frames": 1,
                    "duplex_vision_fallback_frames": 0,
                    "duplex_arrival_audio_units": 1,
                    "duplex_audio_fallback_units": 0,
                },
            },
            {
                "global_request_id": ["req-b"],
                "duplex": {
                    "duplex_prompt_token_ids": [201, 202, 203],
                    "special_token_ids": {"listen_token_id": 702},
                    "duplex_input_video_frames": 1,
                    "duplex_arrival_video_frames": 0,
                    "duplex_vision_fallback_frames": 1,
                    "duplex_arrival_audio_units": 0,
                    "duplex_audio_fallback_units": 0,
                },
            },
        ],
    )

    prompt_rows = output.multimodal_outputs["duplex_prompt_token_ids"]
    listen_rows = output.multimodal_outputs["meta"]["listen_token_id"]
    input_frame_rows = output.multimodal_outputs["duplex_input_video_frames"]
    arrival_frame_rows = output.multimodal_outputs["duplex_arrival_video_frames"]
    fallback_frame_rows = output.multimodal_outputs["duplex_vision_fallback_frames"]
    arrival_audio_rows = output.multimodal_outputs["duplex_arrival_audio_units"]
    fallback_audio_rows = output.multimodal_outputs["duplex_audio_fallback_units"]
    assert to_payload_element(prompt_rows, 0, 0, 2) == [101, 102]
    assert to_payload_element(prompt_rows, 1, 2, 4) == [201, 202, 203]
    assert to_payload_element(listen_rows, 0, 0, 2) == (701,)
    assert to_payload_element(listen_rows, 1, 2, 4) == (702,)
    assert to_payload_element(input_frame_rows, 0, 0, 2) == 1
    assert to_payload_element(input_frame_rows, 1, 2, 4) == 1
    assert to_payload_element(arrival_frame_rows, 0, 0, 2) == 1
    assert to_payload_element(arrival_frame_rows, 1, 2, 4) == 0
    assert to_payload_element(fallback_frame_rows, 0, 0, 2) == 0
    assert to_payload_element(fallback_frame_rows, 1, 2, 4) == 1
    assert to_payload_element(arrival_audio_rows, 0, 0, 2) == 1
    assert to_payload_element(arrival_audio_rows, 1, 2, 4) == 0
    assert to_payload_element(fallback_audio_rows, 0, 0, 2) == 0
    assert to_payload_element(fallback_audio_rows, 1, 2, 4) == 0

    # Exercise the actual tensor-only worker wire, not only the row splitter.
    # Metadata stays off CUDA but its encoded dtype/shape/value are unchanged.
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
        _special_token_ids_from_mm_output,
    )
    from vllm_omni.worker.gpu_ar_model_runner import _ensure_tensor_values

    for i, expected in enumerate((701, 702)):
        value = to_payload_element(listen_rows, i, i * 2, i * 2 + 2)
        wire = _ensure_tensor_values({"meta.listen_token_id": value})
        buffers = MsgpackEncoder().encode(wire)
        original = {"meta.listen_token_id": torch.tensor([expected], dtype=torch.int64)}
        assert [bytes(b) for b in buffers] == [bytes(b) for b in MsgpackEncoder().encode(original)]
        restored = MsgpackDecoder(dict[str, torch.Tensor]).decode(buffers)
        assert _special_token_ids_from_mm_output(restored) == {"listen_token_id": expected}


def test_minicpmo_stage0_rejects_invalid_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    with pytest.raises(ValueError, match="invalid native duplex ref_audio_data"):
        MiniCPMO45Stage0DuplexRuntime._decode_ref_audio_from_session_config(
            {
                "ref_audio_data": "a",
                "ref_audio_format": "pcm_f32le",
            }
        )


def test_minicpmo_stage0_special_token_ids_are_tokenizer_derived():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_bos|>": 105,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [self.ids[text]] if text in self.ids else [201, self.ids["<|tts_bos|>"]]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    runtime._require_special_token_ids()
    assert runtime.tts_bos_token_id == 105
    assert runtime.stage_padding_token_id() == 102
    assert runtime._special_token_ids()["chunk_tts_eos_token_id"] == 109


def test_minicpmo_stage0_rejects_unknown_special_token_fallbacks():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [self.unk_token_id]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    with pytest.raises(ValueError, match=r"<\|tts_bos\|>"):
        runtime._require_special_token_ids()


def _minicpmo45_tokenizer_stub():
    token_ids = {
        token: index
        for index, token in enumerate(
            (
                "<unit>",
                "</unit>",
                "<|listen|>",
                "<|speak|>",
                "<|tts_bos|>",
                "<|tts_eos|>",
                "<|tts_pad|>",
                "<|chunk_eos|>",
                "<|chunk_tts_eos|>",
                "<|turn_eos|>",
            ),
            start=101,
        )
    }
    return SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: token_ids.get(token, 0),
        encode=lambda token, add_special_tokens=False: (
            [token_ids[token]] if not add_special_tokens and token in token_ids else [0]
        ),
    )


def _transformers_processor_stub(auto_processor):
    class _AutoImageProcessor:
        @staticmethod
        def register(*_args, **_kwargs):
            return None

    return SimpleNamespace(
        AutoImageProcessor=_AutoImageProcessor,
        AutoProcessor=auto_processor,
    )


def test_minicpmo_stage0_loads_processor_from_hf_id(monkeypatch):
    import sys

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    model_id = "openbmb/MiniCPM-o-4_5"
    processor = SimpleNamespace(tokenizer=_minicpmo45_tokenizer_stub())
    load_calls = []

    class _AutoImageProcessor:
        @staticmethod
        def register(config_class, image_processor_class):
            del image_processor_class
            if isinstance(config_class, str):
                raise AttributeError("'str' object has no attribute '__module__'")

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, *, trust_remote_code):
            load_calls.append((model_path, trust_remote_code))
            _AutoImageProcessor.register("MiniCPMVImageProcessor", object)
            return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=_AutoImageProcessor,
            AutoProcessor=_AutoProcessor,
        ),
    )

    loaded = MiniCPMO45Stage0DuplexRuntime._load_processor_from_path(model_id)

    assert loaded is processor
    assert load_calls == [(model_id, True)]
    with pytest.raises(AttributeError, match="__module__"):
        _AutoImageProcessor.register("MiniCPMVImageProcessor", object)


def test_minicpmo_stage0_processor_load_preserves_original_exception(monkeypatch):
    import sys

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    original = ValueError("invalid processor config")

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, *, trust_remote_code):
            del cls, model_path, trust_remote_code
            raise original

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _transformers_processor_stub(_AutoProcessor),
    )

    with pytest.raises(RuntimeError, match="Failed to load MiniCPM-o duplex processor") as exc_info:
        MiniCPMO45Stage0DuplexRuntime._load_processor_from_path("openbmb/MiniCPM-o-4_5")

    assert exc_info.value.__cause__ is original


def test_minicpmo_stage0_processor_requires_tokenizer(monkeypatch):
    import sys

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, *, trust_remote_code):
            del cls, model_path, trust_remote_code
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _transformers_processor_stub(_AutoProcessor),
    )

    with pytest.raises(RuntimeError, match="does not expose a tokenizer"):
        MiniCPMO45Stage0DuplexRuntime._load_processor_from_path("openbmb/MiniCPM-o-4_5")


def test_minicpmo_stage0_loaded_processor_validates_special_tokens(monkeypatch):
    import sys

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    processor = SimpleNamespace(tokenizer=_minicpmo45_tokenizer_stub())

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, *, trust_remote_code):
            del cls, model_path, trust_remote_code
            return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _transformers_processor_stub(_AutoProcessor),
    )

    runtime = MiniCPMO45Stage0DuplexRuntime(
        SimpleNamespace(),
        model_path="openbmb/MiniCPM-o-4_5",
        device="cpu",
    )

    assert runtime.processor is processor
    assert set(runtime._special_token_ids()) == {
        "unit_token_id",
        "unit_end_token_id",
        "listen_token_id",
        "speak_token_id",
        "tts_bos_token_id",
        "tts_eos_token_id",
        "tts_pad_token_id",
        "chunk_eos_token_id",
        "chunk_tts_eos_token_id",
        "turn_eos_token_id",
    }


def test_minicpmo_stage0_data_plane_prefill_matches_official_unit_format():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
        encode=lambda text, add_special_tokens=False: [201, 5],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-data-plane-prefill")

    # Official duplex format: each unit is <unit> + audio embeddings with no
    # per-chunk assistant header or <|tts_bos|> boundary. Decoding starts right
    # after the audio so the first sampled token is the listen/speak decision.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=1)

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert result["prompt_suffix_len"] == 0


def test_minicpmo_stage0_data_plane_prefill_matches_official_hd_slice_format():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    token_map = {
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
        "<image>": 12,
        "</image>": 13,
        "<slice>": 14,
        "</slice>": 15,
    }
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: token_map.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    runtime._stage_vision_embeddings = lambda frames, max_slice_nums=1, **_kwargs: [
        [torch.ones((64, 2)), torch.full((64, 2), 2.0)]
    ]
    state = _MiniCPMO45Stage0SessionState(session_id="sid-hd-slice")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        video_frames=[object()],
        max_slice_nums=4,
        seq=1,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == ([1, 12] + [0] * 64 + [13, 14] + [0] * 64 + [15, 11])

    # Subsequent units must close the previous unit with </unit> first,
    # mirroring the official finalize_unit() feed.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=2)

    assert result["success"] is True
    assert result["input_token_ids"] == [2, 1, 11]
    assert result["prompt_suffix_len"] == 0

    runtime._stage_vision_embeddings = lambda *_args, **_kwargs: pytest.fail(
        "preencoded frame blocks must bypass request-at-a-time vision encoding"
    )
    preencoded_state = _MiniCPMO45Stage0SessionState(session_id="sid-hd-preencoded")
    preencoded = runtime._stage_prefill_embeddings_only(
        preencoded_state,
        np.zeros(4, dtype=np.float32),
        video_frames=[object()],
        max_slice_nums=4,
        preencoded_vision=[[torch.ones((64, 2)), torch.full((64, 2), 2.0)]],
        seq=1,
        vision_input_source="arrival",
    )
    assert preencoded["success"] is True
    assert preencoded["input_token_ids"] == ([1, 12] + [0] * 64 + [13, 14] + [0] * 64 + [15, 11])
    assert preencoded["duplex_input_video_frames"] == 1
    assert preencoded["duplex_arrival_video_frames"] == 1
    assert preencoded["duplex_vision_fallback_frames"] == 0

    # Chunked execution may preprocess the same physical request again. The
    # prepared append remains authoritative even if the replay no longer has
    # the one-shot arrival-cache marker.
    cached_preencoded = runtime._stage_prefill_embeddings_only(
        preencoded_state,
        None,
        video_frames=None,
        seq=1,
    )
    assert cached_preencoded["duplex_input_video_frames"] == 1
    assert cached_preencoded["duplex_arrival_video_frames"] == 1
    assert cached_preencoded["duplex_vision_fallback_frames"] == 0


def test_minicpmo_native_duplex_preprocess_batch_prepares_vision_across_sessions():
    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        _MINICPMO45_BATCHED_VISION_KEY,
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        def __init__(self):
            self.sessions = {}
            self.calls = []
            self.batch_calls = []
            self.processor = self

        @staticmethod
        def _decode_video_frames_payload(payload):
            return list(payload.get("video_frames", []))

        def process_image(self, frames, *, max_slice_nums=1):
            self.calls.append((list(frames), max_slice_nums))
            return {"processed": list(frames)}

        def _stage_vision_embeddings_batch(self, processed_batch, *, microbatch_size=8):
            self.batch_calls.append((list(processed_batch), microbatch_size))
            return [[[f"encoded-{processed['processed'][0]}"]] for processed in processed_batch]

    helper = _Helper()
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._duplex_data_plane_helper = lambda: helper

    infos = {
        "req-a": {
            "duplex": {
                "data_plane": True,
                "session_id": "session-a",
                "incarnation": 2,
                "epoch": 3,
                "seq": 4,
                "payload": {"video_frames": [1], "max_slice_nums": [4]},
            }
        },
        "req-b": {
            "duplex": {
                "data_plane": True,
                "session_id": "session-b",
                "incarnation": 5,
                "epoch": 6,
                "seq": 7,
                "payload": {"video_frames": [2], "max_slice_nums": [4]},
            }
        },
    }

    model.preprocess_batch(
        req_ids=["req-a", "req-b"],
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
    )

    assert sorted(helper.calls) == [([1], 4), ([2], 4)]
    cached_a = infos["req-a"][_MINICPMO45_BATCHED_VISION_KEY]
    cached_b = infos["req-b"][_MINICPMO45_BATCHED_VISION_KEY]
    assert cached_a["identity"] == ("session-a", 2, 3, 4)
    assert cached_b["identity"] == ("session-b", 5, 6, 7)
    assert cached_a["video_frames"] == [1]
    assert cached_b["video_frames"] == [2]
    assert cached_a["processed"] == {"processed": [1]}
    assert cached_b["processed"] == {"processed": [2]}
    assert cached_a["frame_blocks"] == [["encoded-1"]]
    assert cached_b["frame_blocks"] == [["encoded-2"]]
    assert helper.batch_calls == [([{"processed": [1]}, {"processed": [2]}], 8)]

    helper._stage_vision_embeddings_batch = lambda *_args, **_kwargs: None
    model.preprocess_batch(
        req_ids=["req-a", "req-b"],
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
    )
    assert "frame_blocks" not in infos["req-a"][_MINICPMO45_BATCHED_VISION_KEY]
    assert "frame_blocks" not in infos["req-b"][_MINICPMO45_BATCHED_VISION_KEY]
    assert infos["req-a"][_MINICPMO45_BATCHED_VISION_KEY]["processed"] == {"processed": [1]}


def test_minicpmo_native_duplex_preprocess_batch_batches_audio_only_and_reuses_pcm():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        _MINICPMO45_BATCHED_VISION_KEY,
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        def __init__(self):
            self.states = {}
            self.decode_calls = []
            self.audio_batch_calls = []
            self.stage_calls = []

        def get_or_create_session_state(self, session_id, incarnation, **_kwargs):
            return self.states.setdefault(
                (session_id, incarnation),
                SimpleNamespace(prepared_append_identity=None),
            )

        def _decode_audio_payload(self, payload):
            marker = int(payload["audio_marker"])
            self.decode_calls.append(marker)
            return np.full(4, marker, dtype=np.float32)

        @staticmethod
        def _prepare_streaming_audio_append(_state, audio_waveform):
            return ("audio-plan", int(audio_waveform[0]))

        def _stage_audio_embeddings_batch(self, prepared):
            self.audio_batch_calls.append(list(prepared))
            return True

        @staticmethod
        def _decode_video_frames_payload(_payload):
            return []

        def _stage_prefill_embeddings_only(
            self,
            _state,
            audio_waveform,
            *,
            preprocessed_audio=None,
            **_kwargs,
        ):
            self.stage_calls.append((audio_waveform, preprocessed_audio))
            return {
                "success": True,
                "inputs_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "input_token_ids": [1, 11],
                "rebase_prompt": True,
            }

    helper = _Helper()
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._minicpmo_pd_prefill = True
    model._duplex_data_plane_helper = lambda: helper
    model.get_input_embeddings = lambda ids: torch.zeros((ids.numel(), 2))
    infos = {
        f"req-{marker}": {
            "duplex": {
                "data_plane": True,
                "session_id": f"session-{marker}",
                "incarnation": 0,
                "epoch": 0,
                "seq": 1,
                "payload": {"audio_marker": marker},
                "scheduler_token_budget": 2,
            }
        }
        for marker in (1, 2)
    }

    model.preprocess_batch(
        req_ids=["req-1", "req-2"],
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
    )

    assert sorted(helper.decode_calls) == [1, 2]
    assert len(helper.audio_batch_calls) == 1
    assert len(helper.audio_batch_calls[0]) == 2
    cached = infos["req-1"][_MINICPMO45_BATCHED_VISION_KEY]
    prepared_waveform = cached["audio_waveform"]
    assert cached["audio_plan"] == ("audio-plan", 1)
    model.preprocess(
        input_ids=torch.tensor([1, 11]),
        duplex_prompt_len=2,
        duplex_token_offset=0,
        **infos["req-1"],
    )

    # The request-local transactional commit consumes the batch-prepared PCM
    # and Mel plan without decoding the payload a second time.
    assert sorted(helper.decode_calls) == [1, 2]
    assert helper.stage_calls == [(prepared_waveform, ("audio-plan", 1))]
    assert infos["req-1"][_MINICPMO45_BATCHED_VISION_KEY] == {}
    model._minicpmo45_audio_prepare_executor.shutdown(wait=True)


def test_minicpmo_native_duplex_preprocess_batch_never_speculates_two_appends_for_one_session():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        _MINICPMO45_BATCHED_VISION_KEY,
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        def __init__(self):
            self.states = {}
            self.prepared_markers = []

        def get_or_create_session_state(self, session_id, incarnation, **_kwargs):
            return self.states.setdefault(
                (session_id, incarnation),
                SimpleNamespace(prepared_append_identity=None),
            )

        @staticmethod
        def _decode_audio_payload(payload):
            return np.full(4, int(payload["audio_marker"]), dtype=np.float32)

        def _prepare_streaming_audio_append(self, _state, audio_waveform):
            marker = int(audio_waveform[0])
            self.prepared_markers.append(marker)
            return ("audio-plan", marker)

        @staticmethod
        def _stage_audio_embeddings_batch(_prepared):
            return True

    helper = _Helper()
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._duplex_data_plane_helper = lambda: helper

    def info(session_id, seq, marker):
        return {
            "duplex": {
                "data_plane": True,
                "session_id": session_id,
                "incarnation": 0,
                "epoch": 0,
                "seq": seq,
                "payload": {"audio_marker": marker},
            }
        }

    infos = {
        "same-1": info("same", 1, 1),
        "same-2": info("same", 2, 2),
        "other": info("other", 1, 3),
    }
    model.preprocess_batch(
        req_ids=["same-1", "same-2", "other"],
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
    )

    assert sorted(helper.prepared_markers) == [1, 3]
    assert _MINICPMO45_BATCHED_VISION_KEY in infos["same-1"]
    assert _MINICPMO45_BATCHED_VISION_KEY not in infos["same-2"]
    assert _MINICPMO45_BATCHED_VISION_KEY in infos["other"]
    model._minicpmo45_audio_prepare_executor.shutdown(wait=True)


def test_minicpmo_audio_batch_commits_complete_cohorts_and_falls_back_only_singletons():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45PreparedAudioAppend,
        _MiniCPMO45PreparedAudioUnit,
        _MiniCPMO45Stage0SessionState,
    )

    class _AudioTarget:
        def __init__(self):
            self.batch_sizes = []

        @staticmethod
        def audio_cache_seq_length(cache):
            return int(cache)

        @staticmethod
        def combine_audio_past_key_values(caches):
            assert len(set(caches)) == 1
            return ("combined", caches[0])

        def get_audio_embedding_streaming_batch(self, data, *, past_key_values, **_kwargs):
            batch_size = int(data["audio_features"].shape[0])
            self.batch_sizes.append(batch_size)
            outputs = [
                [torch.full((1, 2), float(row + 1))]
                for row in range(batch_size)
            ]
            return outputs, ("next", past_key_values)

        @staticmethod
        def split_audio_past_key_values(cache, batch_size):
            return [(cache, row) for row in range(batch_size)]

        @staticmethod
        def should_reset_audio_past_key_values(_cache, **_kwargs):
            return False

    def prepared(session_id, cache_len, marker):
        unit = _MiniCPMO45PreparedAudioUnit(
            chunk_idx=1,
            batch_feature={
                "audio_features": torch.full((1, 80, 4), float(marker)),
                "audio_feature_lens": torch.tensor([4]),
            },
            consumed_samples=4,
        )
        plan = _MiniCPMO45PreparedAudioAppend(
            start_chunk_idx=1,
            start_buffer_len=0,
            units=[unit],
            remaining_audio_buffer=np.empty(0, dtype=np.float32),
            mel_snapshot_after=object(),
        )
        state = _MiniCPMO45Stage0SessionState(
            session_id=session_id,
            audio_chunk_idx=1,
            audio_past_key_values=cache_len,
        )
        return plan, state

    target = _AudioTarget()
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = target
    runtime.thinker = target
    cohort_a = prepared("cohort-a", 10, 1)
    cohort_b = prepared("cohort-b", 10, 2)
    singleton = prepared("singleton", 11, 3)

    encoded = runtime._stage_audio_embeddings_batch(
        [cohort_a, cohort_b, singleton]
    )

    assert encoded is True
    assert target.batch_sizes == [2]
    for plan, _state in (cohort_a, cohort_b):
        assert plan.encoded is True
        assert plan.audio_past_key_values is not None
        assert plan.units[0].audio_embeds is not None
    singleton_plan, _singleton_state = singleton
    assert singleton_plan.encoded is False
    assert singleton_plan.audio_past_key_values is None
    assert singleton_plan.units[0].audio_embeds is None


def test_minicpmo_arrival_preencode_bypasses_request_local_vision_encoder():
    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        _MINICPMO45_BATCHED_VISION_KEY,
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Helper:
        def __init__(self):
            self.sessions = {}
            self.processor = self
            self.process_calls = []
            self.batch_calls = []
            self.cache = {}

        @staticmethod
        def _decode_video_frames_payload(payload):
            return list(payload.get("video_frames", []))

        def process_image(self, frames, *, max_slice_nums=1):
            self.process_calls.append((list(frames), max_slice_nums))
            return {"processed": list(frames)}

        def _stage_vision_embeddings_batch(self, processed_batch, *, microbatch_size=8):
            self.batch_calls.append((list(processed_batch), microbatch_size))
            return [[[f"arrival-encoded-{processed['processed'][0]}"]] for processed in processed_batch]

        def cache_arrival_vision_embeddings(
            self,
            *,
            session_id,
            incarnation,
            epoch,
            preencode_ids,
            frame_blocks,
        ):
            for preencode_id, blocks in zip(preencode_ids, frame_blocks, strict=True):
                self.cache[(session_id, incarnation, epoch, preencode_id)] = blocks
            return len(frame_blocks)

        def take_arrival_vision_embeddings(
            self,
            *,
            session_id,
            incarnation,
            epoch,
            preencode_ids,
        ):
            keys = [(session_id, incarnation, epoch, preencode_id) for preencode_id in preencode_ids]
            if any(key not in self.cache for key in keys):
                return None
            return [self.cache.pop(key) for key in keys]

    helper = _Helper()
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model._minicpmo_pd_decode = False
    model._duplex_data_plane_helper = lambda: helper
    jobs = [
        {
            "session_id": "session-a",
            "incarnation": 1,
            "epoch": 2,
            "preencode_ids": ["frame-a"],
            "video_frames": [1],
            "max_slice_nums": [4],
        },
        {
            "session_id": "session-b",
            "incarnation": 3,
            "epoch": 4,
            "preencode_ids": ["frame-b"],
            "video_frames": [2],
            "max_slice_nums": [4],
        },
    ]

    result = model.preencode_duplex_vision(jobs)

    assert result["encoded_frames"] == 2
    assert sorted(helper.process_calls) == [([1], 4), ([2], 4)]
    process_call_count = len(helper.process_calls)
    infos = {
        "req-a": {
            "duplex": {
                "data_plane": True,
                "session_id": "session-a",
                "incarnation": 1,
                "epoch": 2,
                "seq": 5,
                "payload": {
                    "video_frames": [1],
                    "max_slice_nums": [4],
                    "video_preencode_ids": ["frame-a"],
                },
            }
        },
        "req-b": {
            "duplex": {
                "data_plane": True,
                "session_id": "session-b",
                "incarnation": 3,
                "epoch": 4,
                "seq": 6,
                "payload": {
                    "video_frames": [2],
                    "max_slice_nums": [4],
                    "video_preencode_ids": ["frame-b"],
                },
            }
        },
    }

    model.preprocess_batch(
        req_ids=["req-a", "req-b"],
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
    )

    assert len(helper.process_calls) == process_call_count
    assert infos["req-a"][_MINICPMO45_BATCHED_VISION_KEY]["frame_blocks"] == [["arrival-encoded-1"]]
    assert infos["req-b"][_MINICPMO45_BATCHED_VISION_KEY]["frame_blocks"] == [["arrival-encoded-2"]]


def test_minicpmo_stage0_speech_append_sets_pending_context_once():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
    state = _MiniCPMO45Stage0SessionState(session_id="sid-speech-append")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert result["success"] is True
    assert result["rebase_prompt"] is True
    assert result["delta_num_tokens"] == len(result["input_token_ids"])
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (0, 1)

    state.pending_speech_context = False
    cached = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert cached["success"] is True
    assert cached["rebase_prompt"] is True
    assert cached["delta_num_tokens"] == len(cached["input_token_ids"])
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity == (0, 1)

    next_seq = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=2,
        is_speech=True,
    )

    assert next_seq["success"] is True
    assert next_seq["rebase_prompt"] is False
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (0, 2)

    state.pending_speech_context = False
    next_epoch = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=1,
        seq=1,
        is_speech=True,
    )

    assert next_epoch["success"] is True
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (1, 1)

    state.pending_speech_context = False
    silence = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=1,
        seq=2,
        is_speech=False,
    )

    assert silence["success"] is True
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity == (1, 1)


def test_minicpmo_stage0_failed_append_does_not_set_pending_speech_context():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
    state = _MiniCPMO45Stage0SessionState(session_id="sid-failed-speech")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(0, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert result["success"] is False
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity is None


def test_minicpmo_stage0_streaming_processor_is_isolated_per_session():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Mel:
        def __init__(self):
            self.counter = 0

    class _Processor:
        def __init__(self):
            self._streaming_mel_processor = _Mel()

        def set_streaming_mode(self, **_kwargs):
            return None

        def reset_streaming(self):
            self._streaming_mel_processor.counter = 0

        def process_audio_streaming(self, _audio, **_kwargs):
            self._streaming_mel_processor.counter += 1
            return self._streaming_mel_processor.counter

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.processor = _Processor()
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = runtime.stage_model
    a = _MiniCPMO45Stage0SessionState(session_id="a")
    b = _MiniCPMO45Stage0SessionState(session_id="b")
    processor_a = runtime._configure_streaming_processor(a)
    processor_b = runtime._configure_streaming_processor(b)

    observed = [
        runtime._process_streaming_audio([], 0, processor=processor_a),
        runtime._process_streaming_audio([], 0, processor=processor_b),
        runtime._process_streaming_audio([], 1, processor=processor_a),
        runtime._process_streaming_audio([], 1, processor=processor_b),
    ]

    assert observed == [1, 1, 2, 2]
    assert processor_a is not processor_b
    assert runtime.processor._streaming_mel_processor.counter == 0


def test_minicpmo_stage0_data_plane_next_append_reinjects_previous_listen():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
        session_id="sid-new-speech-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=3,
        last_terminator_token=3,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [3, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 3
    assert state.current_turn_ended is True


def test_minicpmo_stage0_data_plane_turn_eos_closes_previous_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
        session_id="sid-new-user-turn-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=10,
        last_terminator_token=10,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [10, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 10
    assert state.current_turn_ended is True


def test_minicpmo_stage0_context_rollover_keeps_latest_complete_unit():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
        session_id="sid-context-rollover",
        audio_chunk_idx=1,
        context_embeds=[runtime._embed_token(50)],
        context_token_ids=[50],
        pending_terminator_token=3,
        last_terminator_token=3,
        last_unit_inputs_embeds=torch.cat(
            [runtime._embed_token(1), runtime._embed_token(11)],
            dim=0,
        ),
        last_unit_input_token_ids=[1, 11],
        current_segment_output_tokens=[21, 3],
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
        context_rollover=True,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [50, 1, 11, 21, 3, 2, 1, 11]
    assert result["context_rollover"] is True
    assert result["rebase_prompt"] is True
    assert result["delta_num_tokens"] == len(result["input_token_ids"])
    assert state.context_rollovers == 1
    assert state.current_segment_output_tokens == []
    assert state.last_unit_input_token_ids == [1, 11]


def test_minicpmo_stage0_steady_suffix_keeps_lazy_exact_preemption_rebase():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(
        MiniCPMO45Stage0DuplexRuntime
    )
    runtime.stage_model = _StageModel()
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
    retained_unit = torch.cat(
        [runtime._embed_token(1), runtime._embed_token(11)],
        dim=0,
    )
    context_embed = runtime._embed_token(50)
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-preempt-rebase",
        audio_chunk_idx=1,
        context_embeds=[context_embed],
        context_token_ids=[50],
        pending_terminator_token=3,
        last_terminator_token=3,
        last_unit_inputs_embeds=retained_unit,
        last_unit_input_token_ids=[1, 11],
        current_segment_output_tokens=[21, 3],
        pd_feedback_token_ids=[21, 3],
    )

    steady = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=9,
        seq=2,
    )

    assert steady["success"] is True
    assert steady["rebase_prompt"] is False
    assert steady["input_token_ids"] == [21, 3, 2, 1, 11]
    assert steady["inputs_embeds"].shape[0] == 5
    assert state.prepared_inputs_embeds is steady["inputs_embeds"]
    assert len(state.prepared_rebase_prefix_embeds) == 2
    assert state.prepared_rebase_prefix_embeds[0] is context_embed
    assert state.prepared_rebase_prefix_embeds[1] is retained_unit
    assert state.prepared_rebase_prefix_token_ids == (50, 1, 11)

    rebased = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=9,
        seq=2,
        engine_rebase=True,
    )

    assert rebased["success"] is True
    assert rebased["rebase_prompt"] is True
    assert rebased["engine_rebase_prompt"] is True
    assert rebased["input_token_ids"] == [50, 1, 11, 21, 3, 2, 1, 11]
    assert rebased["inputs_embeds"].shape[0] == 8

    audio_chunk_idx = state.audio_chunk_idx
    stale = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=9,
        seq=3,
        engine_rebase=True,
    )
    assert stale["success"] is False
    assert "identity" in stale["reason"]
    assert state.audio_chunk_idx == audio_chunk_idx


def test_minicpmo_stage0_data_plane_model_owned_turn_boundary_preserves_audio_cache():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    stale_cache = object()

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)
            self.audio_past_key_values = stale_cache

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
        session_id="sid-new-user-turn-audio-cache",
        audio_chunk_idx=1,
        audio_past_key_values=stale_cache,
        pending_terminator_token=8,
        last_terminator_token=8,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert state.audio_past_key_values is stale_cache
    assert runtime.thinker.audio_past_key_values is stale_cache


def test_minicpmo_stage0_data_plane_final_first_chunk_does_not_add_silence_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        first_chunk_ms = 10
        sample_rate = 1000

        def __init__(self):
            super().__init__()
            self.seen_audio = None
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            self.seen_audio = np.asarray(data["audio_features"], dtype=np.float32)
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    class _MelProcessor:
        sample_rate = 1000

        def get_config(self):
            return {"effective_first_chunk_ms": 10}

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
    runtime.processor = SimpleNamespace(
        _streaming_mel_processor=_MelProcessor(),
        get_streaming_chunk_size=lambda: 10,
    )
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-first-chunk-padding")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.arange(8, dtype=np.float32),
        seq=1,
        final=True,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert runtime.stage_model.seen_audio is not None
    np.testing.assert_allclose(
        runtime.stage_model.seen_audio.reshape(-1),
        np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
    )


def test_minicpmo_stage0_final_does_not_promote_first_chunk_alignment_tail_to_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        first_chunk_ms = 10
        chunk_ms = 8
        sample_rate = 1000

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    class _MelProcessor:
        sample_rate = 1000
        is_first = True

        def get_config(self):
            return {"effective_first_chunk_ms": 9}

    class _Processor:
        def __init__(self):
            self._streaming_mel_processor = _MelProcessor()

        def get_streaming_chunk_size(self):
            return 9 if self._streaming_mel_processor.is_first else 8

        def process_audio_streaming(self, audio, **_kwargs):
            self._streaming_mel_processor.is_first = False
            return {"audio_features": audio, "audio_feature_lens": [[len(audio)]]}

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
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
    runtime.processor = _Processor()
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-first-chunk-alignment-tail")

    first = runtime._stage_prefill_embeddings_only(state, np.zeros(8, dtype=np.float32), seq=1)
    second = runtime._stage_prefill_embeddings_only(state, np.zeros(8, dtype=np.float32), seq=2)
    final = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(8, dtype=np.float32),
        seq=3,
        final=True,
    )

    assert first["input_token_ids"].count(1) == 1
    assert second["input_token_ids"].count(1) == 1
    assert final["input_token_ids"].count(1) == 1
    assert state.audio_chunk_idx == 3
    assert len(state.audio_buffer) == 1


def test_minicpmo_stage0_runtime_uses_loaded_vllm_embed_tokens_when_get_input_embeddings_is_broken():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, 2))
            self.calls = []

        def forward(self, input_ids):
            ids = input_ids.reshape(-1).tolist()
            self.calls.append(ids)
            return torch.tensor([[float(i), 0.0] for i in ids], dtype=torch.float32)

    class _Thinker:
        def __init__(self):
            self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=_Embed()))

        def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
            raise AttributeError("'Qwen3ForCausalLM' object has no attribute 'get_input_embeddings'")

    thinker = _Thinker()
    stage_model = SimpleNamespace(model_stage="llm", thinker=thinker, processor=None)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = stage_model
    runtime.thinker = thinker
    runtime.device = "cpu"

    embeds = runtime._embed_token(11)

    assert embeds.shape == (1, 2)
    assert thinker.llm.model.embed_tokens.calls == [[11]]


def test_minicpmo_remote_config_patch_handles_nested_and_dict_configs():
    from vllm_omni.experimental.fullduplex.minicpmo45.compat import (
        patch_minicpmo_remote_config,
    )

    nested = SimpleNamespace(base_model_tp_plan=None)
    config = SimpleNamespace(
        base_model_tp_plan=None,
        text_config=nested,
        tts_config={},
    )

    patch_minicpmo_remote_config(config)

    assert config.base_model_tp_plan == {}
    assert nested.base_model_tp_plan == {}
    assert config.tts_config["top_p"] == 0.8
    assert config.tts_config["top_k"] == 100
    assert config.tts_config["temperature"] == 0.8
    assert config.tts_config["repetition_penalty"] == 1.05


def test_minicpmo_stage0_short_audio_buffers_without_context_mutation():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Processor:
        def get_streaming_chunk_size(self):
            return 16000

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.processor = _Processor()
    runtime._require_special_token_ids = lambda: None
    state = _MiniCPMO45Stage0SessionState(session_id="sid")

    result = runtime._stage_prefill_embeddings_only(state, np.zeros(1600, dtype=np.float32))

    assert result["success"] is False
    assert result["reason"]
    assert len(state.audio_buffer) >= 1600
    assert state.context_embeds == []


def test_minicpmo_stage0_sampler_bulk_metadata_preserves_row_fallbacks():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    metadata = SimpleNamespace(
        temperature=torch.tensor([0.1, 0.2]),
        top_k=torch.tensor(7),
        top_p=torch.tensor([]),
        invalid="not-a-number",
    )
    read = MiniCPMO45OmniForConditionalGeneration._sampling_metadata_values

    assert read(metadata, "temperature", 4, 0.7) == pytest.approx([0.1, 0.2, 0.2, 0.2])
    assert read(metadata, "top_k", 4, 100) == [7.0] * 4
    assert read(metadata, "top_p", 4, 0.8) == [0.8] * 4
    assert read(metadata, "missing", 4, 0.8) == [0.8] * 4
    assert read(metadata, "invalid", 4, 0.8) == [0.8] * 4
    assert read(metadata, "temperature", 0, 0.7) == []


def test_minicpmo_stage0_sampler_bulk_metadata_is_seed_exact_and_keeps_logits():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 3
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
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
            }.get(token, -1)

    def make_model():
        model = MiniCPMO45OmniForConditionalGeneration.__new__(
            MiniCPMO45OmniForConditionalGeneration
        )
        model.model_stage = "llm"
        model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
        return model

    def make_metadata():
        return SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([0.8, 1.2]),
            top_k=torch.tensor([1, 1]),
            top_p=torch.tensor([1.0, 1.0]),
            generators={
                0: torch.Generator().manual_seed(101),
                1: torch.Generator().manual_seed(202),
            },
            prompt_token_ids=torch.tensor([[1, 1], [1, 1]]),
            output_token_ids=[[4], [4]],
        )

    logits = torch.full((2, 32), float("-inf"))
    logits[0, 8] = 10.0
    logits[1, 11] = 10.0
    original_logits = logits.clone()

    reference_model = make_model()
    reference_metadata = make_metadata()
    token_ids = reference_model._minicpmo45_native_duplex_token_ids()
    reference_ids = []
    for row_idx in range(2):
        sampled = reference_model._sample_minicpmo45_native_duplex_row(
            logits[row_idx : row_idx + 1].clone(),
            reference_metadata,
            row_idx=row_idx,
            token_ids=token_ids,
            temperature=float(reference_metadata.temperature[row_idx].item()),
            top_k=int(reference_metadata.top_k[row_idx].item()),
            top_p=float(reference_metadata.top_p[row_idx].item()),
        )
        reference_model._record_minicpmo45_duplex_terminator(row_idx, sampled, token_ids)
        reference_ids.append(sampled)
    reference_states = {
        row_idx: generator.get_state()
        for row_idx, generator in reference_metadata.generators.items()
    }

    model = make_model()
    metadata = make_metadata()
    sampled = model.sample(logits, metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[token_id] for token_id in reference_ids]
    assert torch.equal(logits, original_logits)
    for row_idx, generator in metadata.generators.items():
        assert torch.equal(generator.get_state(), reference_states[row_idx])


def test_minicpmo_stage0_native_sampler_penalizes_repeated_text_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    repeated = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, repeated] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[repeated] * 8],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_penalizes_text_from_prior_chunk():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    session_key = ("sid-cross-chunk-repetition", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    state.generated_tokens = [198] * 8
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}

    vocab_size = 151723
    repeated = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, repeated] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_matches_official_negative_logit_penalty():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    session_key = ("sid-negative-logit-repetition", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    repeated = 198
    alternative = 1234
    state.generated_tokens = [repeated]
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}

    logits = torch.full((1, 151723), -100.0)
    logits[0, repeated] = -1.0
    logits[0, alternative] = -0.97
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[repeated]]


def test_minicpmo_stage0_records_bounded_model_policy_history():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    session_key = ("sid-text-history", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}
    token_ids = {
        "unit_token_id": 1,
        "unit_end_token_id": 2,
        "listen_token_id": 3,
        "speak_token_id": 4,
        "tts_bos_token_id": 5,
        "tts_eos_token_id": 6,
        "tts_pad_token_id": 7,
        "chunk_eos_token_id": 8,
        "chunk_tts_eos_token_id": 9,
        "turn_eos_token_id": 10,
    }

    for sampled in range(1000, 1520):
        model._record_minicpmo45_duplex_generation_token(0, sampled)

    expected_history = list(range(1008, 1520))
    assert state.generated_tokens == expected_history

    for sampled in token_ids.values():
        model._record_minicpmo45_duplex_generation_token(0, sampled)

    assert state.generated_tokens == [*expected_history[len(token_ids) :], *token_ids.values()]


def test_minicpmo_stage0_native_sampler_does_not_override_model_at_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            200: "我",
            201: "喜",
            202: "欢",
            203: "。",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202, 203]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_cut_before_natural_boundary_minimum():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "。"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_rewrite_model_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            108386: "你",
            104256: "好",
            3837: "，",
            100644: "今",
            99172: "天",
            100281: "想",
            27442: "聊",
            99217: "什",
            1773: "。",
            99218: "么",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"。": [1773]}.get(text, [])

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    period = 1773
    continuation = 99218
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, period] = 30.0
    logits[0, continuation] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 108386, 104256, 3837, 100644, 99172, 100281, 27442, 99217]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[period]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_preserves_early_model_turn_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151717] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151717]]


def test_minicpmo_stage0_native_sampler_char_cap_does_not_override_special_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = [151706, 151717]

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "已经超过二十八个字符的当前语音分段文本"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151717] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151717]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_keeps_hard_chunk_cap():
    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[200] * (MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK - 1)],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK == 20
    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_cuts_before_request_length_cap():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.max_new_speak_tokens_per_chunk = 64
    model._minicpmo45_duplex_row_max_tokens = {0: 20}
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706] + [200] * 18],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_does_not_cut_on_decoded_text_length():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = [151706, 151718, 151717, 151705]

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            token_text = {
                200: "一二三四五六七八九十",
                201: "十一十二十三十四十五",
                202: "十六十七十八十九",
            }
            special = set(self.all_special_ids) if skip_special_tokens else set()
            return "".join(token_text.get(int(token_id), "") for token_id in ids if int(token_id) not in special)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    candidate = 202
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, candidate] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[candidate]]


def test_minicpmo_stage0_native_sampler_ignores_pending_placeholders():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    newline = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, newline] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[-1, -1, -1]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[newline]]


def test_minicpmo_stage0_native_sampler_converts_mid_turn_listen_to_tts_bos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151703]]
    assert state.current_turn_ended is False


def test_minicpmo_stage0_native_sampler_forced_listen_yields_floor():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_row_payloads = {0: {"force_listen": True}}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151705]]
    assert state.current_turn_ended is True


def test_minicpmo_stage0_native_sampler_uses_runner_duplex_rows():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    metadata = SimpleNamespace(
        prompt_token_ids=torch.tensor([[1, 2, 3]]),
    )

    rows = model._minicpmo45_native_duplex_prompt_rows(
        metadata,
        unit_id=151683,
        batch_size=1,
        duplex_rows=[0],
    )

    assert rows == [0]


def test_minicpmo_stage0_session_context_includes_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.unit_token_id = 151683
    runtime.processor = SimpleNamespace()
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.device = "cpu"
    token_map = {
        "<|im_start|>system\nUse speech.\n<|audio_start|>": [1, 2, 3],
        "<|audio_end|><|im_end|>": [4, 5],
    }
    runtime._stage_runtime_ready = lambda: True
    runtime._require_special_token_ids = lambda: None
    runtime._decode_ref_audio_from_session_config = lambda _config: np.array([0.1, -0.1], dtype=np.float32)
    runtime._encode_text = lambda text: token_map[text]
    runtime._embed_token = lambda token_id: torch.full((1, 2), float(token_id))
    runtime._stage_ref_audio_embeddings = lambda ref_audio, state=None: torch.tensor([[10.0, 11.0], [12.0, 13.0]])

    state = _MiniCPMO45Stage0SessionState(session_id="sid-ref")
    runtime._prepare_session_context(state, {"instructions": "Use speech.", "extra_body": {"ref_audio_data": "x"}})

    assert state.context_token_ids == [1, 2, 3, 151683, 151683, 4, 5]
    assert len(state.context_embeds) == 6


def test_minicpmo_stage0_reuses_identical_session_context_embeddings():
    from collections import OrderedDict

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.unit_token_id = 151683
    runtime.processor = SimpleNamespace(process_audio=lambda _audio: None)
    runtime.stage_model = SimpleNamespace(get_audio_hidden_states=lambda _audio: None)
    runtime.thinker = SimpleNamespace()
    runtime.device = "cpu"
    runtime._session_context_cache = OrderedDict()
    runtime._stage_runtime_ready = lambda: True
    runtime._require_special_token_ids = lambda: None
    runtime._decode_ref_audio_from_session_config = lambda _config: np.array(
        [0.1, -0.1],
        dtype=np.float32,
    )
    runtime._encode_text = lambda text: [1] if "audio_start" in text else [2]
    runtime._embed_token = lambda token_id: torch.full((1, 2), float(token_id))
    ref_calls = []

    def encode_ref_audio(ref_audio, state=None):
        ref_calls.append((ref_audio, state))
        return torch.tensor([[10.0, 11.0], [12.0, 13.0]])

    runtime._stage_ref_audio_embeddings = encode_ref_audio
    first = _MiniCPMO45Stage0SessionState(session_id="first")
    second = _MiniCPMO45Stage0SessionState(session_id="second")
    config = {"instructions": "Use speech."}

    runtime._prepare_session_context(first, config)
    runtime._prepare_session_context(second, config)

    assert len(ref_calls) == 1
    assert first.context_token_ids == second.context_token_ids
    assert all(
        first_embed is second_embed
        for first_embed, second_embed in zip(
            first.context_embeds,
            second.context_embeds,
            strict=True,
        )
    )
    assert first.context_embeds is not second.context_embeds
