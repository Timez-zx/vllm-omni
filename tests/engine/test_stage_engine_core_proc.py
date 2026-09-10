import queue
import threading
from collections import deque
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
    _batch_output_ready,
    _StageInputQueue,
)


def test_output_readiness_does_not_wait_for_builder_or_cuda():
    future = Future()
    assert not _batch_output_ready(future)
    event = MagicMock()
    output = SimpleNamespace(
        _model_runner_output=object(),
        _background_thread=SimpleNamespace(is_alive=lambda: True),
        async_copy_ready_event=event,
    )
    future.async_output = output
    assert not _batch_output_ready(future)
    event.query.assert_not_called()
    output._background_thread = None
    event.query.return_value = False
    assert not _batch_output_ready(future)
    event.query.return_value = True
    assert _batch_output_ready(future)
    event.synchronize.assert_not_called()


@pytest.mark.parametrize("producer,ready", [(True, True), (True, False), (False, True)])
def test_ready_prefill_result_retires_before_next_dispatch(producer, ready):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=producer))
    result, execution = Future(), Future()
    output = object()
    if ready:
        result.set_result(output)
    scheduled = object()
    engine.batch_queue = deque([(result, scheduled, execution)])
    engine.scheduler = MagicMock()
    engine.scheduler.update_from_output.return_value = {0: "done"}
    engine.capture_iteration_details = lambda _: nullcontext("details")
    engine.log_error_detail = lambda _: nullcontext()
    engine._process_aborts_queue = MagicMock()
    engine._attach_iteration_details = MagicMock()
    with patch.object(EngineCoreProc, "step_with_batch_queue", return_value=(None, True)) as native:
        actual = engine.step_with_batch_queue()
    if producer and ready:
        assert actual == ({0: "done"}, False)
        assert not engine.batch_queue
        native.assert_not_called()
        engine.scheduler.schedule.assert_not_called()
        engine._process_aborts_queue.assert_called_once()
        engine.scheduler.update_from_output.assert_called_once_with(scheduled, output)
        engine._attach_iteration_details.assert_called_once_with({0: "done"}, "details")
    else:
        assert actual == (None, True)
        native.assert_called_once()
        engine.scheduler.update_from_output.assert_not_called()
        assert len(engine.batch_queue) == 1


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
        result = engine.collective_rpc("unrelated_utility")

    assert result == [core_thread]
    assert engine._vision_preencode_executor is None


@pytest.mark.parametrize("method, attr", [
    ("preencode_minicpmo45_audio", "_audio_preencode_executor"),
    ("preencode_minicpmo45_vision", "_vision_preencode_executor"),
])
def test_colocated_encoder_always_uses_background_thread(method, attr):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._audio_preencode_executor = None
    core_thread = threading.get_ident()

    def fake_collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        del self, method, timeout, args, kwargs
        return [threading.get_ident()]

    with (
        patch.dict(
            "os.environ",
            {"MINICPMO45_AUDIO_ENCODER_DEVICE": "", "MINICPMO45_VISION_ENCODER_DEVICE": ""},
        ),
        patch.object(
            EngineCoreProc,
            "collective_rpc",
            new=fake_collective_rpc,
        ),
    ):
        result = engine.collective_rpc(method)

    assert isinstance(result, Future)
    assert result.result(timeout=1)[0] != core_thread
    getattr(engine, attr).shutdown(wait=True)


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


def test_auxiliary_audio_rpc_bypasses_core_busy_loop_on_both_placements():
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

    assert dispatched == [audio_rpc]
    assert input_queue.empty()
    dispatched.clear()

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


@pytest.mark.parametrize("extra_tokens", [0, 1])
def test_prepared_pd_decode_inherits_exact_cached_prefix_stats(extra_tokens):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    prepared_stats = PrefillStats()
    prepared_stats.set(
        num_prompt_tokens=1123,
        num_local_cached_tokens=912,
        num_external_cached_tokens=211,
    )
    prepared_request = SimpleNamespace(
        prompt_token_ids=[0] * 1123,
        num_computed_tokens=1123,
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
        prompt_token_ids=[0] * (1123 + extra_tokens),
        num_prompt_tokens=1123 + extra_tokens,
        num_tokens=1123 + extra_tokens,
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
    cached = 1122 + extra_tokens
    assert request.num_computed_tokens == cached
    assert request.prefill_stats.num_prompt_tokens == 1123 + extra_tokens
    assert request.prefill_stats.num_local_cached_tokens == 912
    assert request.prefill_stats.num_external_cached_tokens == cached - 912
    assert request.prefill_stats.num_cached_tokens == cached
    assert request.prefill_stats.num_computed_tokens == 1
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


@pytest.mark.parametrize("model_executed", [False, True])
def test_core_does_not_wait_for_import_notifications_after_model_work(model_executed):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._pd_cache_sync_jobs = {}
    with (
        patch.object(EngineCoreProc, "has_work", return_value=True),
        patch.object(EngineCoreProc, "_process_engine_step", return_value=model_executed),
        patch.object(StageEngineCoreProc, "_publish_finished_pd_blocks", return_value=False),
        patch.object(StageEngineCoreProc, "_progress_pd_cache_sync_jobs", return_value=False) as progress,
    ):
        assert engine._process_engine_step() is model_executed
    progress.assert_called_once_with(wait_for_completion=not model_executed)


@pytest.mark.parametrize("wait_for_completion", [False, True])
def test_cache_sync_poll_forwards_wait_policy_to_worker(wait_for_completion):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._pd_cache_sync_jobs = {
        "r": {"request": SimpleNamespace(kv_lineage_id=None), "phase": "loading"}
    }
    engine._pd_cache_sync_last_poll = 0.0
    engine.model_executor = MagicMock()
    engine.model_executor.collective_rpc.return_value = [set()]
    assert engine._progress_pd_cache_sync_jobs(wait_for_completion=wait_for_completion) is False
    engine.model_executor.collective_rpc.assert_called_once_with(
        "poll_pd_cache_sync", kwargs={"wait_for_completion": wait_for_completion}
    )
