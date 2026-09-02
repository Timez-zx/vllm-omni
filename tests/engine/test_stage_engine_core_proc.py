import queue
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.metrics.stats import PrefillStats

from vllm_omni.engine import OmniEngineCoreOutputs
from vllm_omni.engine.stage_engine_core_proc import (
    StageEngineCoreProc,
    _StageInputQueue,
)


def test_auxiliary_vision_rpc_does_not_block_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._vision_preencode_executor = None
    core_thread = threading.get_ident()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return [threading.get_ident()]

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_VISION_ENCODER_DEVICE": "cuda:1"},
        ),
        patch.object(
            EngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        result = engine.collective_rpc("preencode_minicpmo45_vision")

    assert isinstance(result, Future)
    assert result.result(timeout=1)[0] != core_thread
    engine._vision_preencode_executor.shutdown(wait=True)


def test_auxiliary_audio_rpc_does_not_block_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._audio_preencode_executor = None
    core_thread = threading.get_ident()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return [threading.get_ident()]

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_AUDIO_ENCODER_DEVICE": "cuda:2"},
        ),
        patch.object(
            EngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        result = engine.collective_rpc("preencode_minicpmo45_audio")

    assert isinstance(result, Future)
    assert result.result(timeout=1)[0] != core_thread
    engine._audio_preencode_executor.shutdown(wait=True)


def test_ordinary_collective_rpc_stays_on_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._vision_preencode_executor = None
    core_thread = threading.get_ident()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return [threading.get_ident()]

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_VISION_ENCODER_DEVICE": ""},
        ),
        patch.object(
            EngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        result = engine.collective_rpc("preencode_minicpmo45_vision")

    assert result == [core_thread]
    assert engine._vision_preencode_executor is None


def test_audio_collective_rpc_without_sidecar_device_stays_on_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._audio_preencode_executor = None
    core_thread = threading.get_ident()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return [threading.get_ident()]

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_AUDIO_ENCODER_DEVICE": ""},
        ),
        patch.object(
            EngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        result = engine.collective_rpc("preencode_minicpmo45_audio")

    assert result == [core_thread]
    assert engine._audio_preencode_executor is None


def test_auxiliary_vision_reply_waits_for_cache_ready_and_bypasses_data_outputs():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.output_queue = queue.Queue()
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.output_queue.put_nowait("ordinary-data-output")
    result = Future()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return result

    with patch.object(
        StageEngineCoreProc,
        "collective_rpc",
        new=fake_collective_rpc,
    ):
        engine._handle_client_request(
            EngineCoreRequestType.UTILITY,
            (
                0,
                7,
                "collective_rpc",
                (
                    "preencode_minicpmo45_vision",
                    5.0,
                    ([{"video_frames": ["a", "b"]}],),
                    None,
                ),
            ),
        )

    assert engine.output_queue.get_nowait() == "ordinary-data-output"
    with pytest.raises(queue.Empty):
        engine.output_queue.get_nowait()

    result.set_result([{"supported": True, "encoded_frames": 2}])
    client_idx, outputs = engine.output_queue.get_nowait()
    assert client_idx == 0
    assert outputs.utility_output.call_id == 7
    assert outputs.utility_output.result.result == [
        {"supported": True, "encoded_frames": 2}
    ]


def test_auxiliary_audio_reply_waits_for_cache_ready_and_bypasses_data_outputs():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.output_queue = queue.Queue()
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.output_queue.put_nowait("ordinary-data-output")
    result = Future()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return result

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_AUDIO_ENCODER_DEVICE": "cuda:2"},
        ),
        patch.object(
            StageEngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        engine._handle_client_request(
            EngineCoreRequestType.UTILITY,
            (
                0,
                8,
                "collective_rpc",
                (
                    "preencode_minicpmo45_audio",
                    5.0,
                    ([{"audio": "chunk"}],),
                    None,
                ),
            ),
        )

    assert engine.output_queue.get_nowait() == "ordinary-data-output"
    with pytest.raises(queue.Empty):
        engine.output_queue.get_nowait()

    result.set_result(
        [
            {
                "supported": True,
                "encoded_jobs": 1,
                "job_results": {"audio-1": True},
            }
        ]
    )
    client_idx, outputs = engine.output_queue.get_nowait()
    assert client_idx == 0
    assert outputs.utility_output.call_id == 8
    assert outputs.utility_output.result.result == [
        {
            "supported": True,
            "encoded_jobs": 1,
            "job_results": {"audio-1": True},
        }
    ]


def test_auxiliary_vision_rpc_bypasses_core_busy_loop():
    dispatched = []
    input_queue = _StageInputQueue(dispatched.append)
    ordinary_add = (EngineCoreRequestType.ADD, "ordinary-add")
    vision_rpc = (
        EngineCoreRequestType.UTILITY,
        (
            0,
            7,
            "collective_rpc",
            ("preencode_minicpmo45_vision", 5.0, ([],), None),
        ),
    )

    input_queue.put_nowait(ordinary_add)
    input_queue.put_nowait(vision_rpc)

    assert dispatched == [vision_rpc]
    assert input_queue.get_nowait() == ordinary_add
    assert input_queue.empty()


def test_auxiliary_audio_rpc_bypasses_core_busy_loop_only_with_sidecar():
    dispatched = []
    input_queue = _StageInputQueue(dispatched.append)
    audio_rpc = (
        EngineCoreRequestType.UTILITY,
        (
            0,
            8,
            "collective_rpc",
            ("preencode_minicpmo45_audio", 5.0, ([],), None),
        ),
    )

    with patch.dict(
        "os.environ",
        {"MINICPMO45_AUDIO_ENCODER_DEVICE": ""},
    ):
        input_queue.put_nowait(audio_rpc)

    assert dispatched == []
    assert input_queue.get_nowait() == audio_rpc

    with patch.dict(
        "os.environ",
        {"MINICPMO45_AUDIO_ENCODER_DEVICE": "cuda:2"},
    ):
        input_queue.put_nowait(audio_rpc)

    assert dispatched == [audio_rpc]
    assert input_queue.empty()


def test_shutdown_drains_both_preencode_executors():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    vision_executor = MagicMock()
    audio_executor = MagicMock()
    engine._vision_preencode_executor = vision_executor
    engine._audio_preencode_executor = audio_executor

    with patch.object(EngineCoreProc, "shutdown") as base_shutdown:
        engine.shutdown()

    vision_executor.shutdown.assert_called_once_with(
        wait=True,
        cancel_futures=True,
    )
    audio_executor.shutdown.assert_called_once_with(
        wait=True,
        cancel_futures=True,
    )
    assert engine._vision_preencode_executor is None
    assert engine._audio_preencode_executor is None
    base_shutdown.assert_called_once_with()


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace()
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


def test_prepared_pd_decode_inherits_exact_cached_prefix_stats():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    prepared_stats = PrefillStats()
    prepared_stats.set(
        num_prompt_tokens=1123,
        num_local_cached_tokens=912,
        num_external_cached_tokens=211,
    )
    prepared_request = SimpleNamespace(
        prompt_token_ids=[0] * 1123,
        num_computed_tokens=1122,
        prefill_stats=prepared_stats,
        kv_transfer_params={
            "do_remote_prefill": False,
            "kv_transfer_selected_blocks": 14,
            "kv_transfer_selected_tokens": 211,
            "kv_transfer_selected_bytes": 13_762_560,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        },
        pd_transfer_evidence={
            "kv_transfer_selected_blocks": 14,
            "kv_transfer_selected_tokens": 211,
            "kv_transfer_selected_bytes": 13_762_560,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        },
    )
    request = SimpleNamespace(
        request_id="req-00000003",
        prompt_token_ids=[0] * 1124,
        num_prompt_tokens=1124,
        num_computed_tokens=0,
        prefill_stats=PrefillStats(),
        kv_transfer_params={"do_remote_prefill": True},
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": {"do_remote_prefill": True}},
        ),
        get_skip_reading_prefix_cache=lambda: False,
    )

    with patch.object(EngineCoreProc, "add_request") as add_request:
        engine._activate_pd_decode(
            request,
            request_wave=7,
            prepared={
                "request": prepared_request,
                "ready_mono": 0.0,
                "owns_blocks": True,
            },
        )

    add_request.assert_called_once_with(request, 7)
    assert request.num_computed_tokens == 1122
    assert request.prefill_stats.num_prompt_tokens == 1124
    assert request.prefill_stats.num_local_cached_tokens == 912
    assert request.prefill_stats.num_external_cached_tokens == 210
    assert request.prefill_stats.num_cached_tokens == 1122
    assert request.prefill_stats.num_computed_tokens == 2
    assert request.kv_transfer_params is None
    assert request.sampling_params.extra_args == {}
    assert request.pd_transfer_evidence == {
        "kv_transfer_selected_blocks": 14,
        "kv_transfer_selected_tokens": 211,
        "kv_transfer_selected_bytes": 13_762_560,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }


def test_shared_output_control_frame_uses_stable_copied_buffer():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.output_queue = queue.Queue()
    engine.output_queue.put((0, OmniEngineCoreOutputs()))
    engine.output_queue.put(EngineCoreProc.ENGINE_CORE_DEAD)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_id=0),
    )
    sender = MagicMock()
    sender.message_stats.return_value = (0, 0, 0.0)
    engine._output_tensor_ipc_sender = sender

    socket = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = context

    with (
        patch(
            "vllm_omni.engine.stage_engine_core_proc.zmq.Context",
            return_value=context,
        ),
        patch(
            "vllm_omni.engine.stage_engine_core_proc.make_zmq_socket",
            return_value=nullcontext(socket),
        ),
    ):
        engine.process_output_sockets(["unused"], None, engine_index=0)

    _, kwargs = socket.send_multipart.call_args
    assert kwargs == {"copy": True}


def test_core_publishes_finished_pd_blocks_before_another_model_step():
    events = []
    metadata = object()

    class Connector:
        def take_immediate_push_metadata(self):
            events.append("take-finished")
            return metadata

    class ModelExecutor:
        def collective_rpc(self, method, args=()):
            events.append((method, args))
            return [True]

    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace(connector=Connector())
    engine.model_executor = ModelExecutor()
    engine._pd_cache_sync_jobs = {}

    def finish_model_step(self):
        del self
        events.append("finish-a")
        return True

    with (
        patch.object(EngineCoreProc, "has_work", return_value=True),
        patch.object(
            EngineCoreProc,
            "_process_engine_step",
            new=finish_model_step,
        ),
        patch.object(
            StageEngineCoreProc,
            "_progress_pd_cache_sync_jobs",
            return_value=False,
        ),
    ):
        assert engine._process_engine_step() is True

    assert events == [
        "finish-a",
        "take-finished",
        ("publish_pd_finished_blocks", (metadata,)),
    ]
