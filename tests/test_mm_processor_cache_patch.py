from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldElem, MultiModalKwargsItem
from vllm.multimodal.registry import MultiModalRegistry

from vllm_omni.patch import (
    _extract_mm_layout_values,
    _safe_mm_processor_cache_type,
)


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


def test_extract_mm_layout_values_keeps_only_small_position_fields():
    item = MultiModalKwargsItem(
        {
            "video_grid_thw": MultiModalFieldElem(
                data=[[1, 2, 3]],
                field=MultiModalBatchedField(),
            ),
            "pixel_values_videos": MultiModalFieldElem(
                data="large-media-tensor",
                field=MultiModalBatchedField(),
            ),
        }
    )

    assert _extract_mm_layout_values(item) == {"video_grid_thw": [[1, 2, 3]]}
