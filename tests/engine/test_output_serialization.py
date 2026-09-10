import gc

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine.serialization import CPUOutputMsgpackEncoder


def wire_bytes(buffers):
    return [bytes(value) for value in buffers]


@pytest.mark.parametrize("dtype", [
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
    torch.float64, torch.float32, torch.float16, torch.bool,
    torch.complex64, torch.complex128, torch.bfloat16,
])
@pytest.mark.parametrize("shape", ["scalar", "empty", "matrix", "offset", "transpose"])
@pytest.mark.parametrize("threshold", [0, 256])
def test_cpu_output_wire_matches_native(dtype, shape, threshold):
    tensor = torch.arange(12).to(dtype)
    tensor = {
        "scalar": tensor[0], "empty": tensor[:0], "matrix": tensor.reshape(3, 4),
        "offset": tensor[3:9], "transpose": tensor.reshape(3, 4).T,
    }[shape]
    payload = {"value": tensor}
    native = MsgpackEncoder(size_threshold=threshold)
    encoder = CPUOutputMsgpackEncoder(size_threshold=threshold)
    encoded = encoder.encode(payload)
    assert wire_bytes(encoded) == wire_bytes(native.encode(payload))
    decoded = MsgpackDecoder(dict[str, torch.Tensor]).decode(encoded)["value"]
    assert decoded.dtype == tensor.dtype and decoded.shape == tensor.shape
    assert torch.equal(decoded, tensor)
    buffer = bytearray(b"old data" * 100)
    assert wire_bytes(encoder.encode_into(payload, buffer)) == wire_bytes(native.encode(payload))


@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("size", [1, 1024])
def test_output_encoder_preserves_oob_consumer_order(accept, size):
    class Consumer:
        def __init__(self):
            self.calls = []

        def new_message(self):
            self.calls.append("new")

        def __call__(self, tensor):
            self.calls.append(tensor)
            return {"shared": "same-handle"} if accept else None

    tensor = torch.arange(size)
    consumers = [Consumer(), Consumer()]
    native = MsgpackEncoder(oob_tensor_consumer=consumers[0])
    encoder = CPUOutputMsgpackEncoder(oob_tensor_consumer=consumers[1])
    assert wire_bytes(native.encode({"value": tensor})) == wire_bytes(encoder.encode({"value": tensor}))
    for consumer in consumers:
        assert consumer.calls[0] == "new"
        assert len(consumer.calls) == (2 if size * tensor.element_size() >= encoder.size_threshold else 1)
        if len(consumer.calls) == 2:
            assert consumer.calls[1] is tensor


def test_output_byte_view_keeps_tensor_storage_alive():
    def encode():
        tensor = torch.arange(4096, dtype=torch.int64)[16:2048]
        return CPUOutputMsgpackEncoder().encode({"value": tensor})

    buffers = encode()
    gc.collect()
    churn = [torch.full((4096,), -1, dtype=torch.int64) for _ in range(10)]
    decoded = MsgpackDecoder(dict[str, torch.Tensor]).decode(buffers)["value"]
    assert torch.equal(decoded, torch.arange(16, 2048))
    assert all(tensor[0].item() == -1 for tensor in churn)


def test_output_encoder_falls_back_for_special_tensor_semantics(monkeypatch):
    native_calls = []

    def original(self, tensor):
        native_calls.append(tensor)
        return "native"

    monkeypatch.setattr(MsgpackEncoder, "_encode_tensor", original)
    encoder = CPUOutputMsgpackEncoder()
    tensors = [
        torch.ones(2, requires_grad=True), torch.ones(2, dtype=torch.bfloat16),
        torch.ones(2, dtype=torch.complex64).conj(), torch._neg_view(torch.ones(2)),
        torch.ones(2, 3).T, torch.Tensor._make_subclass(type("Subclass", (torch.Tensor,), {}), torch.ones(2)),
    ]
    if torch.cuda.is_available():
        tensors.append(torch.ones(2, device="cuda"))
    for tensor in tensors:
        assert encoder._encode_tensor(tensor) == "native"
        assert native_calls[-1] is tensor
