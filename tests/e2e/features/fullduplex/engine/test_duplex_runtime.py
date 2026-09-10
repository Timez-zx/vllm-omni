import base64
from dataclasses import FrozenInstanceError
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine import duplex_runtime
from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_data_plane_request_info,
    duplex_resource_request_belongs_to_session,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeManager,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.minicpmo45 import runtime as minicpm_runtime
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
    MiniCPMO45DuplexRuntimeExtension,
    build_duplex_data_plane_prompt,
    duplex_feedback_scheduler_rows,
    duplex_scheduler_token_budget,
    duplex_vision_block_counts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_fence_is_immutable():
    fence = DuplexFence("session")

    with pytest.raises(FrozenInstanceError):
        fence.epoch = 1  # type: ignore[misc]

    assert not hasattr(fence, "__dict__")


def test_duplex_runtime_tracks_stage_bindings_and_barge_in_epoch():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-1"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_TOKENS},
        ),
    )
    session.bind_stage_request(stage_id=0, request_id="req-stage0", fence=session.fence)
    session.bind_stage_request(stage_id=1, request_id="req-stage1", fence=session.fence)

    update = session.append_input(mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)
    next_fence = DuplexFence("sid-1", epoch=1)
    stale_request_ids = session.release_fence(session.fence)
    session.accept_fence(next_fence)

    assert update.seq == 1
    assert session.fence == next_fence
    assert stale_request_ids == ["req-stage0", "req-stage1"]
    assert session.stage_bindings == {}


def test_duplex_append_commit_accepts_same_epoch_fence_advance_to_target():
    manager = DuplexSessionRuntimeManager()
    base = DuplexFence("sid-append-fence-advance")
    target = DuplexFence(base.session_id, turn_id=1)
    session = manager.open_session(
        base,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    reservation = session.prepare_append(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=target,
    )

    # A model/control event may publish the append's target fence while the
    # data-plane submission is in flight.  No input sequence was consumed.
    session.accept_fence(target)
    update = session.commit_append(reservation)

    assert update.seq == 1
    assert update.turn_seq == 1
    assert session.fence == target


def test_duplex_append_commit_rejects_fence_advance_beyond_target():
    manager = DuplexSessionRuntimeManager()
    base = DuplexFence("sid-append-fence-stale")
    target = DuplexFence(base.session_id, turn_id=1)
    session = manager.open_session(
        base,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    reservation = session.prepare_append(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=target,
    )

    session.accept_fence(DuplexFence(base.session_id, turn_id=2))

    with pytest.raises(RuntimeError, match="changed=fence"):
        session.commit_append(reservation)


def test_duplex_runtime_tracks_same_request_id_for_each_pipeline_stage():
    manager = DuplexSessionRuntimeManager()
    fence = DuplexFence("sid-shared-pipeline-request")
    session = manager.open_session(fence)

    session.reserve_stage_request(0, "req-shared", fence=fence)
    session.bind_stage_request(0, "req-shared", fence=fence)
    session.bind_stage_request(1, "req-shared", fence=fence)

    assert session.stage_bindings[0].request_id == "req-shared"
    assert session.stage_bindings[1].request_id == "req-shared"
    assert session.resource_request_ids() == ["req-shared"]
    assert session.input_seq == 0


def test_duplex_runtime_extension_validation_rejects_missing_methods():
    class IncompleteExtension:
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults

    with pytest.raises(TypeError, match="plan_append"):
        duplex_runtime.validate_duplex_runtime_extension(IncompleteExtension())


def test_duplex_runtime_extension_validation_rejects_stage_count_mismatch():
    class WrongStageCountExtension(MiniCPMO45DuplexRuntimeExtension):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults[:1]

    with pytest.raises(ValueError, match="one sampling parameter per stage"):
        duplex_runtime.validate_duplex_runtime_extension(
            WrongStageCountExtension(),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


def test_duplex_runtime_extension_validation_rejects_sampling_type_mismatch():
    class WrongSamplingTypeExtension(MiniCPMO45DuplexRuntimeExtension):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return tuple(object() for _ in defaults)

    with pytest.raises(TypeError, match="sampling parameter type"):
        duplex_runtime.validate_duplex_runtime_extension(
            WrongSamplingTypeExtension(),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


def test_duplex_output_decision_metadata_is_immutable():
    decision = DuplexOutputDecision(
        action=DuplexOutputAction.DIRECT_RESPONSE,
        metadata={"model_listen": True},
    )

    with pytest.raises(TypeError):
        decision.metadata["model_listen"] = False


def _decide_minicpmo_output(
    output: object,
    *,
    segment_token_ids: tuple[int, ...] = (),
    segment_output_metadata: dict | None = None,
):
    return MiniCPMO45DuplexRuntimeExtension().decide_output(
        stage_id=0,
        final_stage_id=1,
        segment_finished=True,
        segment_token_ids=segment_token_ids,
        segment_output_metadata=segment_output_metadata or {},
        output=output,
    )


def test_minicpmo_extension_owns_stage_sampling_overrides():
    defaults = (
        SamplingParams(max_tokens=4),
        SamplingParams(max_tokens=8),
    )

    configured = MiniCPMO45DuplexRuntimeExtension().configure_sampling_params(
        runtime_config={
            "duplex_stage_max_tokens": {"0": 20},
            "duplex_stage_sampling_params": {"1": {"stop_token_ids": [151645]}},
        },
        defaults=defaults,
    )

    assert configured[0].max_tokens == 20
    assert configured[1].stop_token_ids == [151645]
    assert defaults[0].max_tokens == 4
    assert 151645 not in (defaults[1].stop_token_ids or [])


@pytest.mark.parametrize("stage_count", [3, 4])
def test_native_thinker_uses_only_model_unit_boundaries_not_inherited_chat_stops(stage_count):
    from vllm.v1.core.sched.utils import check_stop

    unit_ends = [151705, 151718, 151721]
    chat_eos, extra_eos, turn_eos, old_stop = 151645, 151643, 151717, 9999
    defaults = tuple(
        SamplingParams(max_tokens=64, stop=["legacy stop"], stop_token_ids=[turn_eos, old_stop], min_tokens=3)
        for _ in range(stage_count)
    )
    for params in defaults:
        params.update_from_generation_config({"eos_token_id": [chat_eos, extra_eos]}, chat_eos)
    configured = MiniCPMO45DuplexRuntimeExtension().configure_sampling_params(
        runtime_config={
            "duplex_stage_max_tokens": {"0": 20, "1": 8192},
            "duplex_stage_sampling_params": {"0": {"stop_token_ids": unit_ends}},
        },
        defaults=defaults,
    )
    thinker_idx = 1 if stage_count == 4 else 0
    params = configured[thinker_idx]
    assert params.ignore_eos is True
    assert params.eos_token_id is None
    assert params.stop == []
    assert params.output_text_buffer_length == 0
    assert params.stop_token_ids == unit_ends
    assert params.all_stop_token_ids == set(unit_ends)
    assert params.min_tokens == 0
    # Engine input processing updates generation config once more after the
    # runtime override. This must not reintroduce chat EOS as a stopping rule.
    params.update_from_generation_config({"eos_token_id": [chat_eos, extra_eos]}, chat_eos)
    assert params.eos_token_id is None
    assert params.stop_token_ids == unit_ends
    for token in [*unit_ends, chat_eos, extra_eos, turn_eos, old_stop, 123]:
        request = SimpleNamespace(
            pooling_params=None,
            sampling_params=params,
            num_output_tokens=1,
            output_token_ids=[token],
            num_tokens=280,
            max_tokens=params.max_tokens,
        )
        assert check_stop(request, 262144) is (token in unit_ends)
    if stage_count == 4:
        assert configured[0].max_tokens == 1
        assert configured[0].stop_token_ids == []
        assert configured[0].eos_token_id is None
    for params in configured[thinker_idx + 1 :]:
        assert params.ignore_eos is False
        assert params.eos_token_id == chat_eos
        assert params.stop == ["legacy stop"]
        assert turn_eos in params.stop_token_ids
    for params in defaults:
        assert params.ignore_eos is False
        assert params.eos_token_id == chat_eos
        assert params.stop == ["legacy stop"]


def test_native_adapter_explicitly_disables_chat_eos_and_stop_strings(monkeypatch):
    from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45NativeDuplexServingAdapter as Adapter

    monkeypatch.setattr(Adapter, "_native_stage0_stop_token_ids", staticmethod(lambda _: [1, 3, 4]))
    monkeypatch.setattr(Adapter, "_native_scheduler_token_id", staticmethod(lambda _: 0))
    runtime_config = {}
    Adapter._apply_default_scheduler_policy(
        runtime_config, config=SimpleNamespace(max_tokens=20, temperature=0.7), model_config=None
    )
    params = runtime_config["duplex_stage_sampling_params"]["0"]
    assert params["ignore_eos"] is True
    assert params["stop"] == []
    assert params["min_tokens"] == 0
    assert params["stop_token_ids"] == [1, 3, 4]


def test_minicpmo_output_decision_uses_raw_streaming_token_snapshot():
    decision = _decide_minicpmo_output(
        SimpleNamespace(outputs=[SimpleNamespace()]),
        segment_token_ids=(151705,),
        segment_output_metadata={"special_token_ids": {"listen_token_id": 151705}},
    )

    assert decision is not None
    assert decision.action is DuplexOutputAction.DIRECT_RESPONSE
    assert decision.metadata["duplex_native_decision"] == "listen"
    assert decision.metadata["model_listen"] is True


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_minicpmo_output_decision_ignores_output_level_token_history(attr):
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace()],
        **{attr: [42, 151705]},
    )

    assert _decide_minicpmo_output(output) is None


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_minicpmo_output_decision_uses_completion_token_ids(attr):
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(**{attr: [42, 151705]})],
    )

    assert _decide_minicpmo_output(output) is not None


def test_minicpmo_output_decision_uses_completion_stop_reason():
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
        outputs=[SimpleNamespace(stop_reason=151705)],
    )

    assert _decide_minicpmo_output(output) is not None


def test_duplex_runtime_cancel_fence_rejects_late_append_and_accepts_next_epoch():
    manager = DuplexSessionRuntimeManager()
    cancelled_fence = DuplexFence("sid-cancel-race")
    next_fence = DuplexFence("sid-cancel-race", epoch=1)
    session = manager.open_session(
        cancelled_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(
        stage_id=0,
        request_id="req-cancelled",
        fence=cancelled_fence,
    )

    stale_request_ids = session.cancel_fence(cancelled_fence, next_fence)

    assert stale_request_ids == ["req-cancelled"]
    assert session.fence == next_fence
    assert session.stage_bindings == {}
    with pytest.raises(RuntimeError, match="fence mismatch"):
        session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=cancelled_fence,
        )
    update = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_fence,
    )
    assert update.seq == 1


def test_duplex_runtime_stale_close_preserves_live_session_and_bindings():
    manager = DuplexSessionRuntimeManager()
    current_fence = DuplexFence("sid-stale-close", epoch=1)
    session = manager.open_session(
        current_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(0, "req-live", fence=current_fence)

    with pytest.raises(RuntimeError, match="fence mismatch"):
        manager.close_session(DuplexFence("sid-stale-close"))

    assert manager.get("sid-stale-close") is session
    assert session.stage_request_ids() == ["req-live"]


def test_duplex_runtime_reopen_rejects_late_append_from_old_incarnation():
    manager = DuplexSessionRuntimeManager()
    old_fence = DuplexFence("sid-reopen", incarnation=0)
    old_session = manager.open_session(
        old_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    manager.close_session(old_fence)

    new_fence = DuplexFence("sid-reopen", incarnation=1)
    new_session = manager.open_session(
        new_fence,
        capabilities=old_session.capabilities,
    )

    with pytest.raises(RuntimeError, match="fence mismatch"):
        new_session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=old_fence,
        )
    assert (
        new_session.append_input(
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            fence=new_fence,
        ).seq
        == 1
    )


def test_duplex_prompt_expands_incarnation_metadata():
    fence = DuplexFence("sid-incarnation", incarnation=3)

    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={"is_speech": True},
        final=False,
    )

    assert prompt["model_intermediate_buffer"]["duplex"]["incarnation"] == 3


def test_duplex_runtime_tracks_turn_local_append_sequence():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-turn-seq"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    first = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    second = session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK, fence=session.fence)
    next_turn = DuplexFence("sid-turn-seq", turn_id=1, response_seq=1)
    third = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_turn,
    )
    fourth = session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=next_turn,
    )

    assert [first.seq, second.seq, third.seq, fourth.seq] == [1, 2, 3, 4]
    assert [first.turn_seq, second.turn_seq, third.turn_seq, fourth.turn_seq] == [1, 2, 1, 2]
    assert [first.turn_id, second.turn_id, third.turn_id, fourth.turn_id] == [0, 0, 1, 1]


def test_duplex_runtime_rejects_unsupported_append_mode():
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        DuplexFence("sid-2"),
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.TURN_COMMIT_ONLY}),
    )

    with pytest.raises(ValueError, match="not supported"):
        session.append_input(mode=DuplexInputMode.APPEND_TOKENS, fence=session.fence)


def test_duplex_data_plane_request_info_extracts_structured_stage_result():
    request_id, response_stage_id = duplex_data_plane_request_info(
        {
            "stage_results": [
                {"result": {"supported": True}},
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0-s1",
                        "response_stage_id": 1,
                    }
                },
            ]
        }
    )

    assert request_id == "duplex-sid-e0-stage0-s1"
    assert response_stage_id == 1


def test_duplex_data_plane_request_info_rejects_missing_request_id():
    assert duplex_data_plane_request_info(
        {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "",
                        "response_stage_id": 1,
                    }
                }
            ]
        }
    ) == (None, None)


def test_duplex_scheduler_token_budget_estimates_pcm_slots():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
        == 16
    )


def test_duplex_scheduler_token_budget_ignores_client_budget_fields():
    assert (
        duplex_scheduler_token_budget(
            {
                "audio": "AAAAAA==",
                "format": "pcm_f32le",
                "duplex_num_input_tokens": 999,
                "num_input_tokens": 999,
            }
        )
        == 16
    )


@pytest.mark.parametrize(
    ("feedback", "expected"),
    [
        (None, 0),
        ([], 0),
        ([151705], 0),
        ([101, 151705], 1),
        ([101, 102, 151705], 2),
    ],
)
def test_duplex_feedback_scheduler_rows_excludes_reserved_terminator(feedback, expected):
    assert duplex_feedback_scheduler_rows(feedback) == expected


def test_duplex_scheduler_token_budget_counts_official_hd_slices():
    image = Image.new("RGB", (960, 540), color="white")
    encoded = BytesIO()
    image.save(encoded, format="JPEG")
    payload = {
        "audio": base64.b64encode(np.zeros(16_000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
        "video_frames": [base64.b64encode(encoded.getvalue()).decode()],
        "max_slice_nums": 4,
    }

    # Official grid selection uses one global image plus a 2x1 crop grid.
    assert duplex_vision_block_counts(payload) == [3]
    assert duplex_scheduler_token_budget(payload) == 12 + 3 * 66


def test_minicpmo_plan_append_decodes_media_shape_once_per_transaction(monkeypatch):
    image = Image.new("RGB", (960, 540), color="white")
    encoded = BytesIO()
    image.save(encoded, format="JPEG")
    payload = {
        "audio": base64.b64encode(np.zeros(16_000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
        "video_frames": [base64.b64encode(encoded.getvalue()).decode()],
        "max_slice_nums": 4,
    }
    vision_calls = 0
    audio_calls = 0
    original_vision = minicpm_runtime.duplex_vision_block_counts
    original_audio = minicpm_runtime._duplex_pcm_sample_count

    def counted_vision(payload):
        nonlocal vision_calls
        vision_calls += 1
        return original_vision(payload)

    def counted_audio(payload):
        nonlocal audio_calls
        audio_calls += 1
        return original_audio(payload)

    monkeypatch.setattr(
        minicpm_runtime,
        "duplex_vision_block_counts",
        counted_vision,
    )
    monkeypatch.setattr(
        minicpm_runtime,
        "_duplex_pcm_sample_count",
        counted_audio,
    )
    extension = MiniCPMO45DuplexRuntimeExtension()
    for seq in (1, 2, 2):
        extension.plan_append(
            request_id="req-shape-once",
            fence=DuplexFence("sid-shape-once"),
            session_config={},
            runtime_config={"duplex_first_append_context_tokens": 48},
            seq=seq,
            turn_seq=seq,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
            sampling_params=SamplingParams(max_tokens=20),
        )

    assert vision_calls == 3
    assert audio_calls == 3


def test_minicpmo_context_window_starts_fresh_lineage_with_one_retained_unit():
    extension = MiniCPMO45DuplexRuntimeExtension()
    fence = DuplexFence("sid-context-window")
    payload = {
        "audio": base64.b64encode(np.zeros(16_000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
    }
    runtime_config = {
        "duplex_first_append_context_tokens": 48,
        "duplex_context_window_trigger_tokens": 1024,
        "duplex_stage_max_tokens": {"0": 20},
    }

    rollover = None
    for seq in range(1, 80):
        plan = extension.plan_append(
            request_id="req-context-window",
            fence=fence,
            session_config={},
            runtime_config=runtime_config,
            seq=seq,
            turn_seq=seq,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
            sampling_params=SamplingParams(max_tokens=20),
        )
        meta = plan.prompt["model_intermediate_buffer"].get("meta")
        if isinstance(meta, dict) and meta.get("replace_streaming_prompt") is True:
            rollover = plan.prompt
            break

    assert rollover is not None
    assert rollover["model_intermediate_buffer"]["duplex"]["payload"]["duplex_context_rollover"] is True
    assert rollover["model_intermediate_buffer"]["meta"]["retain_streaming_output_tokens"] is True
    # 48 context + 11 retained-unit rows + 13 rows for the current steady unit.
    assert len(rollover["prompt_token_ids"]) == 72
    rollover_duplex = rollover["model_intermediate_buffer"]["duplex"]
    assert rollover_duplex["compact_rebase_prefix_tokens"] == 0


def test_minicpmo_engine_window_never_recycles_prompt_after_limit():
    extension = MiniCPMO45DuplexRuntimeExtension()
    fence = DuplexFence("sid-engine-window")
    payload = {
        "audio": base64.b64encode(np.zeros(16_000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
    }
    for seq in range(1, 150):
        plan = extension.plan_append(
            request_id="req-engine-window",
            fence=fence,
            session_config={},
            runtime_config={
                "duplex_first_append_context_tokens": 48,
                "duplex_context_window_trigger_tokens": 1024,
                "duplex_kv_window_tokens": 1024,
            },
            seq=seq,
            turn_seq=seq,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
            sampling_params=SamplingParams(max_tokens=20),
        )
        buffer = plan.prompt["model_intermediate_buffer"]
        assert not buffer.get("meta", {}).get("replace_streaming_prompt")
        assert buffer["duplex"]["context_generation"] == 0
        if seq > 1:
            assert len(plan.prompt["prompt_token_ids"]) == 13


def test_minicpmo_steady_append_declares_exact_lazy_preemption_rebase_prefix():
    extension = MiniCPMO45DuplexRuntimeExtension()
    fence = DuplexFence("sid-preemption-rebase")
    payload = {
        "audio": base64.b64encode(np.zeros(16_000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
    }
    runtime_config = {
        "duplex_first_append_context_tokens": 48,
        "duplex_context_window_trigger_tokens": 36_000,
    }

    first = extension.plan_append(
        request_id="req-preemption-rebase",
        fence=fence,
        session_config={},
        runtime_config=runtime_config,
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=payload,
        final=False,
        sampling_params=SamplingParams(max_tokens=20),
    )
    second = extension.plan_append(
        request_id="req-preemption-rebase",
        fence=fence,
        session_config={},
        runtime_config=runtime_config,
        seq=2,
        turn_seq=2,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=payload,
        final=False,
        sampling_params=SamplingParams(max_tokens=20),
    )
    second_retry = extension.plan_append(
        request_id="req-preemption-rebase",
        fence=fence,
        session_config={},
        runtime_config=runtime_config,
        seq=2,
        turn_seq=2,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=payload,
        final=False,
        sampling_params=SamplingParams(max_tokens=20),
    )

    first_duplex = first.prompt["model_intermediate_buffer"]["duplex"]
    second_duplex = second.prompt["model_intermediate_buffer"]["duplex"]
    retry_duplex = second_retry.prompt["model_intermediate_buffer"]["duplex"]
    assert first_duplex["compact_rebase_prefix_tokens"] == 0
    # 48 immutable context rows + the latest complete 11-row audio unit.
    assert second_duplex["compact_rebase_prefix_tokens"] == 59
    assert retry_duplex["compact_rebase_prefix_tokens"] == 59
    assert second_duplex["scheduler_token_budget"] == 13


def test_resource_state_rejects_fence_regression_and_requires_explicit_fence():
    current = DuplexFence("sid", epoch=2, turn_id=3, response_seq=4)
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(
        current,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )

    for stale in (
        DuplexFence("sid", epoch=1, turn_id=99, response_seq=99),
        DuplexFence("sid", epoch=2, turn_id=2, response_seq=4),
        DuplexFence("sid", epoch=2, turn_id=3, response_seq=3),
    ):
        with pytest.raises(RuntimeError, match="fence mismatch"):
            session.accept_fence(stale)
        assert session.fence == current

    with pytest.raises(TypeError, match="fence"):
        session.bind_stage_request(0, "request")
    with pytest.raises(TypeError, match="fence"):
        session.append_input(mode=DuplexInputMode.APPEND_AUDIO_CHUNK)
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.open_session("legacy-session")
    with pytest.raises(TypeError, match="DuplexFence"):
        manager.close_session("legacy-session")


def test_resource_request_id_is_derived_from_fence_and_role():
    fence = DuplexFence("sid-with-dashes", epoch=7, turn_id=11, response_seq=13)

    assert duplex_resource_request_id(fence, "stage0") == "duplex-s.c2lkLXdpdGgtZGFzaGVz.i.0.e.7.r.stage0"
    assert duplex_resource_request_id(fence, "stage1") == "duplex-s.c2lkLXdpdGgtZGFzaGVz.i.0.e.7.r.stage1"


def test_resource_request_id_codec_separates_session_id_from_incarnation():
    embedded_incarnation = duplex_resource_request_id(
        DuplexFence("foo-i1", incarnation=0),
        "stage0",
    )
    actual_incarnation = duplex_resource_request_id(
        DuplexFence("foo", incarnation=1),
        "stage0",
    )

    assert embedded_incarnation != actual_incarnation


def test_resource_request_id_session_membership_uses_encoded_identity():
    request_id = duplex_resource_request_id(
        DuplexFence("sid-with-dashes", incarnation=2, epoch=7),
        "stage0",
    )

    assert duplex_resource_request_belongs_to_session(request_id, "sid-with-dashes") is True
    assert duplex_resource_request_belongs_to_session(request_id, "sid") is False
    assert duplex_resource_request_belongs_to_session("duplex-s.invalid.i.x.e.7.r.stage0", "sid") is False


def test_placeholder_budget_is_planned_inside_omni_engine_boundary():
    fence = DuplexFence("sid", turn_id=1, response_seq=1)
    prompt = build_duplex_data_plane_prompt(
        request_id=duplex_resource_request_id(fence, "stage0"),
        fence=fence,
        session_config={},
        runtime_config={},
        seq=2,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload={
            "audio": "AAAAAA==",
            "format": "pcm_f32le",
            "duplex_num_input_tokens": 999,
            "num_input_tokens": 999,
        },
        final=False,
    )

    assert len(prompt["prompt_token_ids"]) == 16
    assert prompt["model_intermediate_buffer"]["duplex"]["fence"] == fence
    assert prompt["model_intermediate_buffer"]["duplex"]["scheduler_token_budget"] == 16
