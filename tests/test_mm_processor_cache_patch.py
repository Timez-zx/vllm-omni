from vllm.multimodal.registry import MultiModalRegistry

from vllm_omni.patch import _safe_mm_processor_cache_type


def test_safe_mm_processor_cache_switches_lru_to_processor_only(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SAFE_MM_PROCESSOR_CACHE", "1")

    assert _safe_mm_processor_cache_type("lru") == "processor_only"
    assert _safe_mm_processor_cache_type("shm") == "shm"


def test_safe_mm_processor_cache_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_SAFE_MM_PROCESSOR_CACHE", raising=False)

    assert _safe_mm_processor_cache_type("lru") == "lru"


def test_safe_mm_processor_cache_registry_patch_is_installed():
    assert getattr(
        MultiModalRegistry._get_cache_type,
        "_omni_safe_mm_cache_patched",
        False,
    )
