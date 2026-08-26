from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm.multimodal.cache import ShmObjectStoreReceiverCache
from vllm.sampling_params import SamplingParams
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver, TensorIpcSender
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import (
    OmniEngineCoreOutput,
    OmniEngineCoreOutputs,
    OmniEngineCoreRequest,
)
from vllm_omni.engine.stage_engine_core_client import _SharedTensorIpcSender
from vllm_omni.engine.stage_engine_core_proc import (
    StageEngineCoreProc,
    _MaterializedShmReceiverCache,
    _SharedOutputTensorIpcSender,
)


def _mm_item(value: int) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "value": MultiModalFieldElem(
                data=torch.tensor([value]),
                field=MultiModalBatchedField(keep_on_cpu=True),
            )
        }
    )


def _mm_feature(key: str, data: MultiModalKwargsItem) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier=key,
        mm_hash=key,
        mm_position=PlaceholderRange(offset=0, length=1),
    )


def test_materialized_shm_cache_reuses_deserialized_media_but_still_touches():
    address = _mm_item(10)
    materialized = _mm_item(20)
    delegate = Mock(spec=ShmObjectStoreReceiverCache)
    delegate.get_and_update_item.return_value = materialized
    cache = _MaterializedShmReceiverCache(delegate, capacity_gb=0.001)

    first = _mm_feature("same-media", address)
    second = _mm_feature("same-media", address)

    assert cache.get_and_update_features([first])[0].data is materialized
    assert cache.get_and_update_features([second])[0].data is materialized
    assert delegate.get_and_update_item.call_count == 1
    assert delegate.touch_receiver_cache_item.call_count == 2

    info = cache.materialized_cache_info()
    assert info.hits == 1
    assert info.total == 2


def test_materialized_shm_cache_clear_drops_local_and_delegate_state():
    address = _mm_item(10)
    materialized = _mm_item(20)
    delegate = Mock(spec=ShmObjectStoreReceiverCache)
    delegate.get_and_update_item.return_value = materialized
    cache = _MaterializedShmReceiverCache(delegate, capacity_gb=0.001)

    cache.get_and_update_features([_mm_feature("same-media", address)])
    cache.clear_cache()
    cache.get_and_update_features([_mm_feature("same-media", address)])

    assert delegate.get_and_update_item.call_count == 2
    delegate.clear_cache.assert_called_once_with()


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


def test_pd_cache_sync_bypasses_media_lookup_without_talker_snapshot():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace()
    mm_features = [SimpleNamespace(identifier="pd-mrope:image", data=object())]
    request = SimpleNamespace(
        request_id="pd-cache-sync",
        external_req_id="pd-cache-sync",
        additional_information=None,
        model_intermediate_buffer=None,
        pd_prefill_payload=None,
        mm_features=mm_features,
        sampling_params=SimpleNamespace(
            extra_args={"kv_transfer_params": {"do_remote_prefill": True}}
        ),
    )
    scheduler_request = SimpleNamespace(mm_features=[])

    def preprocess_without_media(req):
        assert req.mm_features == []
        return scheduler_request, 5

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        side_effect=preprocess_without_media,
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert current_wave == 5
    assert request.mm_features is mm_features
    assert result.mm_features is mm_features
    assert result.pd_prefill_payload is None


def test_direct_pd_cache_sync_completes_without_model_request():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(request_id="cache-only")
    result = Future()
    engine._pd_cache_sync_jobs = {
        request.request_id: {
            "request": request,
            "future": result,
            "phase": "queued",
            "started": 0.0,
        }
    }
    engine._pd_cache_sync_last_poll = 0.0
    engine.scheduler = SimpleNamespace(
        prepare_direct_pd_cache_sync=Mock(return_value=("loading", object())),
        complete_direct_pd_cache_sync=Mock(),
        fail_direct_pd_cache_sync=Mock(),
    )
    engine.model_executor = SimpleNamespace(
        collective_rpc=Mock(side_effect=[[True], [{request.request_id}]])
    )

    assert engine._progress_pd_cache_sync_jobs() is True
    assert result.result()["request_id"] == request.request_id
    engine.scheduler.complete_direct_pd_cache_sync.assert_called_once_with(request)
    assert request.request_id not in engine._pd_cache_sync_jobs


def test_direct_pd_cache_sync_restores_generic_utility_request():
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "image_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([1, 2, 3]),
                    field=MultiModalBatchedField(keep_on_cpu=True),
                )
            }
        ),
        modality="image",
        identifier="pd-mrope:image",
        mm_position=PlaceholderRange(offset=0, length=2),
        mm_hash="pd-mrope:image",
    )
    request = OmniEngineCoreRequest(
        request_id="cache-wire",
        prompt_token_ids=[1, 2],
        mm_features=[mm_feature],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prefill_only=True,
    )
    generic = MsgpackDecoder().decode(MsgpackEncoder().encode(request))
    assert isinstance(generic, list)

    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._pd_cache_sync_jobs = {}
    scheduler_request = SimpleNamespace(request_id=request.request_id)
    engine.preprocess_add_request = Mock(return_value=(scheduler_request, 0))
    connector = Mock()
    engine.scheduler = SimpleNamespace(get_kv_connector=lambda: connector)

    future = engine.start_pd_cache_sync(generic)

    assert isinstance(future, Future)
    engine.preprocess_add_request.assert_called_once()
    converted = engine.preprocess_add_request.call_args.args[0]
    assert isinstance(converted, OmniEngineCoreRequest)
    assert converted.request_id == request.request_id
    assert isinstance(converted.mm_features[0].data, MultiModalKwargsItem)
    connector.on_new_request.assert_called_once_with(scheduler_request)


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


def test_shared_only_input_tensor_ipc_keeps_pd_snapshot_off_zmq_frames():
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

        # The large shared snapshot is represented by a compact handle.  The
        # ordinary tensor remains on the standard wire path.
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


def test_shared_output_tensor_ipc_does_not_copy_small_regular_tensors():
    tensor_queue = get_mp_context().Queue()
    try:
        sender = _SharedOutputTensorIpcSender(tensor_queue)
        regular = torch.arange(8, dtype=torch.float32)
        shared = torch.arange(16, dtype=torch.float32).share_memory_()

        assert sender(regular) is None
        assert not regular.is_shared()
        assert sender(shared) is not None
        shared_bytes, fallback_bytes, _ = sender.message_stats()
        assert shared_bytes == shared.nbytes
        assert fallback_bytes == regular.nbytes
    finally:
        tensor_queue.close()
        tensor_queue.join_thread()


def test_shared_output_tensor_ipc_preserves_large_fallback():
    tensor_queue = get_mp_context().Queue()
    try:
        sender = _SharedOutputTensorIpcSender(tensor_queue)
        regular = torch.zeros((1 << 20) // 4, dtype=torch.float32)

        assert sender(regular) is not None
        assert regular.is_shared()
        shared_bytes, fallback_bytes, _ = sender.message_stats()
        assert shared_bytes == 0
        assert fallback_bytes == regular.nbytes
    finally:
        tensor_queue.close()
        tensor_queue.join_thread()
