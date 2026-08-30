import queue
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine import OmniEngineCoreOutputs
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


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
