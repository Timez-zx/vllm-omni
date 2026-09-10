"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_omni.data_entry_keys import OmniPayload, deserialize_payload, serialize_payload
from vllm_omni.engine import AdditionalInformationPayload

logger = init_logger(__name__)

_NUMPY_WIRE_DTYPES = frozenset({
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
    torch.float64, torch.float32, torch.float16, torch.bool,
    torch.complex64, torch.complex128,
})


class CPUOutputMsgpackEncoder(MsgpackEncoder):
    """Keep the native tensor wire; avoid extra CPU torch views on output IO.

    NumPy's byte view owns the tensor storage without copying. CUDA, BF16,
    subclasses and non-contiguous/lazy/autograd tensors retain native handling.
    """

    def _encode_tensor(self, obj):
        if not (
            type(obj) is torch.Tensor and obj.is_cpu and obj.layout == torch.strided
            and obj.dtype in _NUMPY_WIRE_DTYPES and not obj.requires_grad
            and not obj.is_conj() and not obj.is_neg() and obj.is_contiguous()
        ):
            return super()._encode_tensor(obj)
        # Offer the same tensors to the existing shared-memory consumer first.
        consumer = self.oob_tensor_consumer
        if obj.nbytes >= self.size_threshold and consumer is not None and (data := consumer(obj)) is not None:
            assert isinstance(data, dict)
        else:
            # Reuse native inline/aux-buffer handling on a 1D byte view;
            # restore the original torch dtype and shape in its wire header.
            _, _, data = self._encode_ndarray(obj.numpy().reshape(-1).view("uint8"))
        return str(obj.dtype).removeprefix("torch."), obj.shape, data


def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport.

    Delegates to ``serialize_payload`` which understands the nested
    ``OmniPayload`` TypedDict structure.
    """
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    payload: OmniPayload = raw_info  # type: ignore[assignment]
    return serialize_payload(payload)


def deserialize_additional_information(
    payload: dict | AdditionalInformationPayload | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return deserialize_payload(payload)  # type: ignore[return-value]
