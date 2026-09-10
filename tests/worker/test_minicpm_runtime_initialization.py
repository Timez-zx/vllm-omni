from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace


def test_encoder_and_p_share_one_lazy_runtime(monkeypatch):
    from vllm_omni.experimental.fullduplex.minicpmo45 import stage0
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    entered = Event()
    duplicate = Event()
    release = Event()
    attempted = Event()
    runtimes = []

    def construct(*args, **kwargs):
        if entered.is_set():
            duplicate.set()
        result = SimpleNamespace(arrival_cache={"first_frame": object()})
        runtimes.append(result)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(stage0, "MiniCPMO45Stage0DuplexRuntime", construct)
    model = SimpleNamespace(
        _minicpmo45_helper_init_lock=Lock(),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="unused")),
        thinker=None,
        _module_device=lambda _: "cpu",
    )
    helper = MiniCPMO45OmniForConditionalGeneration._duplex_data_plane_helper

    def second_caller():
        attempted.set()
        return helper(model)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(helper, model)
        try:
            assert entered.wait(5)
            second = executor.submit(second_caller)
            assert attempted.wait(5)
            assert not duplicate.wait(0.1)
        finally:
            release.set()
        assert first.result() is second.result()
    assert len(runtimes) == 1
    assert helper(model) is runtimes[0]
    assert "first_frame" in helper(model).arrival_cache
