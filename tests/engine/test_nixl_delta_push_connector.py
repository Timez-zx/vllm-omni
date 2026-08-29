# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_omni.engine.nixl_delta_push_connector import (
    NixlDeltaPushConnectorScheduler,
    _delta_registration_fields,
)


def test_explicit_remote_boundary_full_local_hit_does_not_include_decode_token() -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    request = SimpleNamespace(
        request_id="req-00000001",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_prompt_tokens": 912,
        },
    )

    assert scheduler.get_num_new_matched_tokens(request, 912) == (0, False)


def test_delta_registration_uses_aligned_local_prefix() -> None:
    assert _delta_registration_fields(
        remote_prompt_tokens=1123,
        external_tokens=211,
        block_size=16,
    ) == {
        "matched_prefix_tokens": 912,
        "source_block_offset": 57,
        "decode_block_size": 16,
        "remote_prompt_tokens": 1123,
    }


def test_delta_registration_rejects_unaligned_prefix() -> None:
    with pytest.raises(ValueError, match="block-aligned"):
        _delta_registration_fields(
            remote_prompt_tokens=912,
            external_tokens=1,
            block_size=16,
        )
