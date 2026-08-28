import torch

from vllm_omni.utils.mm_outputs import build_mm_cpu, to_shared_cpu_tensor


def test_to_shared_cpu_tensor_allocates_exact_contiguous_storage() -> None:
    base = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    output = to_shared_cpu_tensor(base[3:7])

    assert output.is_shared()
    assert output.is_contiguous()
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
    torch.testing.assert_close(output, base[3:7])


def test_build_mm_cpu_shares_only_selected_flat_keys() -> None:
    output = build_mm_cpu(
        {
            "hidden_states.layer_0": torch.arange(8).reshape(2, 4),
            "embed.tts_bos": torch.ones((1, 4)),
        },
        shared_tensor_keys={"hidden_states.layer_0"},
    )

    assert output["hidden_states.layer_0"].is_shared()
    assert not output["embed.tts_bos"].is_shared()
