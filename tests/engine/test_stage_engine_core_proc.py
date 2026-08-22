from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine.core import EngineCoreProc

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
