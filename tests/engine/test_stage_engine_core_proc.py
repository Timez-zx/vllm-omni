import threading
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

from vllm_omni.engine.stage_engine_core_proc import (
    StageEngineCoreProc,
    _StageInputQueue,
)


def test_auxiliary_vision_rpc_does_not_block_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
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


def test_ordinary_collective_rpc_stays_on_core_thread():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
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
    assert getattr(engine, "_vision_preencode_executor", None) is None


def test_auxiliary_vision_reply_bypasses_pending_data_outputs():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.output_queue = Queue()
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

    client_idx, outputs = engine.output_queue.get_nowait()
    assert client_idx == 0
    assert outputs.utility_output.call_id == 7
    assert outputs.utility_output.result.result == [
        {"supported": True, "accepted": True, "encoded_frames": 2}
    ]
    assert not result.done()
    assert engine.output_queue.get_nowait() == "ordinary-data-output"
    result.set_result([{"encoded_frames": 2}])


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


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
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
