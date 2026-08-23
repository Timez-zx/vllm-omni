from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver, TensorIpcSender
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.engine.stage_engine_core_client import _SharedTensorIpcSender
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


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


def test_pd_decode_preserves_mrope_features_without_media_cache_lookup():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace()
    mm_features = [SimpleNamespace(identifier="pd-mrope:image", data=object())]
    pd_payload = object()
    request = SimpleNamespace(
        request_id="pd-d",
        external_req_id="pd-d",
        additional_information=None,
        model_intermediate_buffer=None,
        pd_prefill_payload=pd_payload,
        mm_features=mm_features,
    )
    scheduler_request = SimpleNamespace(mm_features=[])

    def preprocess_without_media(req):
        assert req.mm_features == []
        return scheduler_request, 4

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        side_effect=preprocess_without_media,
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert current_wave == 4
    assert request.mm_features is mm_features
    assert result.mm_features is mm_features
    assert result.pd_prefill_payload is pd_payload


def test_reverse_tensor_ipc_keeps_large_output_off_zmq_frames():
    tensor_queue = get_mp_context().Queue()
    try:
        tensor = torch.arange(1024, dtype=torch.float32).reshape(256, 4)
        outputs = OmniEngineCoreOutputs(
            outputs=[
                OmniEngineCoreOutput(
                    request_id="pd-p",
                    new_token_ids=[],
                    multimodal_output={"hidden": tensor},
                )
            ]
        )
        frames = MsgpackEncoder(
            oob_tensor_consumer=TensorIpcSender(tensor_queue),
        ).encode(outputs)

        # Only the compact msgpack control frame goes over ZMQ; the tensor is
        # shared through torch multiprocessing IPC.
        assert len(frames) == 1
        assert len(frames[0]) < 1024

        decoded = MsgpackDecoder(
            OmniEngineCoreOutputs,
            oob_tensor_provider=TensorIpcReceiver(tensor_queue),
        ).decode(frames)
        received = decoded.outputs[0].multimodal_output["hidden"]
        assert received.is_shared()
        torch.testing.assert_close(received, tensor)
    finally:
        tensor_queue.close()
        tensor_queue.join_thread()


def test_shared_only_input_tensor_ipc_does_not_stage_regular_tensors():
    tensor_queue = get_mp_context().Queue()
    try:
        shared = torch.arange(4096, dtype=torch.float32).share_memory_()
        regular = torch.arange(8, dtype=torch.float32)
        outputs = OmniEngineCoreOutputs(
            outputs=[
                OmniEngineCoreOutput(
                    request_id="pd-d",
                    new_token_ids=[],
                    multimodal_output={"snapshot": shared, "metadata": regular},
                )
            ]
        )
        frames = MsgpackEncoder(
            oob_tensor_consumer=_SharedTensorIpcSender(tensor_queue),
        ).encode(outputs)

        # The large shared snapshot is represented by a compact handle. The
        # ordinary tensor stays on the normal wire path instead of incurring a
        # synchronous share_memory_ staging copy.
        assert sum(len(frame) for frame in frames) < 2048

        decoded = MsgpackDecoder(
            OmniEngineCoreOutputs,
            oob_tensor_provider=TensorIpcReceiver(tensor_queue),
        ).decode(frames)
        payload = decoded.outputs[0].multimodal_output
        assert payload["snapshot"].is_shared()
        torch.testing.assert_close(payload["snapshot"], shared)
        torch.testing.assert_close(payload["metadata"], regular)
    finally:
        tensor_queue.close()
        tensor_queue.join_thread()
