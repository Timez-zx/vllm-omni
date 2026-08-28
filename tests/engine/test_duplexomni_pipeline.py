from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from vllm_omni.engine.duplexomni_pipeline import (
    PIPELINE_BASE_TURNS,
    PIPELINE_EPOCH,
    PIPELINE_FINAL,
    PIPELINE_SESSION_ID,
    PIPELINE_SLOT,
    DuplexOmniPipelineCoordinator,
    inject_codec_history,
    pipeline_identity_from_prompt,
)
from vllm_omni.engine.orchestrator import Orchestrator


def _codes(value: int) -> list[list[int]]:
    return [[value] * 6 for _ in range(16)]


def test_orchestrator_accepts_mapping_multimodal_output() -> None:
    metadata = MappingProxyType({"duplexomni_valid_turn": [True]})
    output = SimpleNamespace(multimodal_output=None)
    completion = SimpleNamespace(multimodal_output=metadata)

    assert Orchestrator._completion_multimodal_output(output, completion) is metadata


def _prompt(*, slot: int, base_codes: list[list[list[int]]] | None = None) -> dict:
    base_codes = base_codes or []
    return {
        "additional_information": {
            "codes": {"ref": base_codes},
            "ids": {"duplex_history_indices": list(range(len(base_codes)))},
            "meta": {
                PIPELINE_SESSION_ID: "session",
                PIPELINE_EPOCH: 0,
                PIPELINE_SLOT: slot,
                PIPELINE_FINAL: slot == 2,
                PIPELINE_BASE_TURNS: len(base_codes),
            },
        }
    }


def test_talker_slots_are_ordered_without_blocking_later_thinkers() -> None:
    coordinator = DuplexOmniPipelineCoordinator()
    identities = []
    for slot in range(3):
        prompt = _prompt(slot=slot)
        identity = pipeline_identity_from_prompt(prompt)
        assert identity is not None
        coordinator.register(f"req-{slot}", identity, prompt)
        identities.append(identity)

    assert coordinator.talker_ready(identities[0])
    assert not coordinator.talker_ready(identities[1])
    assert not coordinator.talker_ready(identities[2])

    coordinator.complete(identities[0], codec_codes=_codes(1), valid_turn=True)
    assert coordinator.talker_ready(identities[1])
    assert coordinator.history_for(identities[1]) == ([0], [_codes(1)])

    coordinator.complete(identities[1], codec_codes=_codes(2), valid_turn=False)
    assert coordinator.talker_ready(identities[2])
    # An invalid Talker turn advances ordering but is not added to history.
    assert coordinator.history_for(identities[2]) == ([0], [_codes(1)])


def test_pipeline_rejects_out_of_order_talker_completion() -> None:
    coordinator = DuplexOmniPipelineCoordinator()
    identities = []
    for slot in range(2):
        prompt = _prompt(slot=slot)
        identity = pipeline_identity_from_prompt(prompt)
        assert identity is not None
        coordinator.register(f"req-{slot}", identity, prompt)
        identities.append(identity)

    with pytest.raises(RuntimeError, match="out of order"):
        coordinator.complete(identities[1], codec_codes=_codes(2), valid_turn=True)


def test_injected_history_does_not_mutate_the_original_prompt() -> None:
    prompt = _prompt(slot=1)
    updated = inject_codec_history(prompt, history_indices=[0], codec_history=[_codes(7)])

    assert prompt["additional_information"]["codes"]["ref"] == []
    assert updated["additional_information"]["ids"]["duplex_history_indices"] == [0]
    assert updated["additional_information"]["codes"]["ref"] == [_codes(7)]
