from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm.multimodal.cache import ShmObjectStoreReceiverCache
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import (
    StageEngineCoreProc,
    _MaterializedShmReceiverCache,
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
