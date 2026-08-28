# SPDX-License-Identifier: Apache-2.0
"""DuplexOmni stage adapters.

DuplexOmni reuses the Qwen3-Omni modules, but its Talker prompt is different:
each 480 ms turn contributes assistant conditioning followed by exactly six
16-codebook RVQ frames.  The application carries the previous RVQ frames in
``additional_information.codes.ref``; engine KV state remains disposable.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.engine.duplexomni_pipeline import (
    PIPELINE_BASE_TURNS,
    PIPELINE_EPOCH,
    PIPELINE_FINAL,
    PIPELINE_SESSION_ID,
    PIPELINE_SLOT,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors import qwen3_omni

logger = logging.getLogger(__name__)

IM_START = 151644
IM_END = 151645
ASSISTANT = 77091
CODEC_PAD = 2148
CODEC_BOS = 2149
CODEC_EOS = 2150
NUM_CODEBOOKS = 16
FRAMES_PER_SLICE = 6
FINAL_THINKER_LAYER = 48


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "_x"):
        value = value._x
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def _prompt_dict(prompt: Any, index: int = 0) -> dict[str, Any]:
    item = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    return item if isinstance(item, dict) else {}


def _codec_history_from_plain_prompt(prompt: Any, index: int = 0) -> tuple[torch.Tensor, list[int]]:
    info = _prompt_dict(prompt, index).get("additional_information")
    return _codec_history_from_info(info)


def _codec_history_from_request(request: Any) -> tuple[torch.Tensor, list[int]]:
    # The GPU runner materializes the serialized EngineCore payload on its
    # cached request state as ``additional_information_cpu`` before invoking
    # the full-payload adapter.  Keep the serialized field as a fallback for
    # direct/offline callers and older runners.
    raw_info = getattr(request, "additional_information_cpu", None)
    if not isinstance(raw_info, dict):
        raw_info = deserialize_additional_information(getattr(request, "additional_information", None))
    return _codec_history_from_info(raw_info)


def _request_additional_information(request: Any) -> dict[str, Any]:
    raw_info = getattr(request, "additional_information_cpu", None)
    if not isinstance(raw_info, dict):
        raw_info = deserialize_additional_information(getattr(request, "additional_information", None))
    return raw_info if isinstance(raw_info, dict) else {}


def _pd_prompt_chunks(payload: Any, field: str) -> tuple[torch.Tensor, ...]:
    chunks = getattr(payload, f"{field}_chunks", None)
    if chunks:
        return tuple(chunk.detach().cpu() for chunk in chunks)
    tensor = getattr(payload, field, None)
    if isinstance(tensor, torch.Tensor):
        return (tensor.detach().cpu(),)
    raise RuntimeError(f"DuplexOmni P/D snapshot is missing {field}")


def _pipeline_meta(raw_info: Any) -> dict[str, Any] | None:
    meta = raw_info.get("meta") if isinstance(raw_info, dict) else None
    if not isinstance(meta, dict) or not isinstance(meta.get(PIPELINE_SESSION_ID), str):
        return None
    return meta


def _codec_history_from_info(raw_info: Any) -> tuple[torch.Tensor, list[int]]:
    raw = raw_info.get("codes", {}).get("ref") if isinstance(raw_info, dict) else None
    history = _normalize_codec_history(raw)
    raw_indices = raw_info.get("ids", {}).get("duplex_history_indices") if isinstance(raw_info, dict) else None
    if raw_indices is None:
        # Backward-compatible default: every historical assistant turn has a
        # corresponding valid codec block.
        indices = list(range(int(history.shape[0])))
    else:
        indices = _as_list(raw_indices)
    if len(indices) != int(history.shape[0]):
        raise ValueError(
            "DuplexOmni history index/codec mismatch: "
            f"indices={len(indices)}, codec turns={history.shape[0]}"
        )
    if any(index < 0 for index in indices) or any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("DuplexOmni history indices must be non-negative and strictly increasing")
    return history, indices


def _normalize_codec_history(raw: Any) -> torch.Tensor:
    if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
        return torch.empty((0, NUM_CODEBOOKS, FRAMES_PER_SLICE), dtype=torch.long)
    codes = (
        raw.detach().cpu().to(torch.long) if isinstance(raw, torch.Tensor) else torch.as_tensor(raw, dtype=torch.long)
    )
    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    expected_tail = (NUM_CODEBOOKS, FRAMES_PER_SLICE)
    if codes.ndim != 3 or tuple(codes.shape[1:]) != expected_tail:
        raise ValueError(
            "DuplexOmni codec history must have shape "
            f"[turns,{NUM_CODEBOOKS},{FRAMES_PER_SLICE}], got {tuple(codes.shape)}"
        )
    if codes.numel() and (bool((codes < 0).any()) or bool((codes >= 2048).any())):
        raise ValueError("DuplexOmni codec history contains an RVQ id outside [0, 2048)")
    return codes.contiguous()


def _assistant_content_ranges(prompt_ids: list[int]) -> list[tuple[int, int]]:
    """Return historical assistant content spans in a rendered ChatML prompt."""
    starts = [index for index, token_id in enumerate(prompt_ids) if token_id == IM_START]
    ranges: list[tuple[int, int]] = []
    for number, start in enumerate(starts):
        if start + 2 >= len(prompt_ids) or prompt_ids[start + 1] != ASSISTANT:
            continue
        # The last ``assistant\n`` is the generation prefix, not history.
        search_end = starts[number + 1] if number + 1 < len(starts) else len(prompt_ids)
        try:
            end = prompt_ids.index(IM_END, start + 3, search_end)
        except ValueError:
            continue
        ranges.append((start + 3, end))
    return ranges


def _conditioning_layout(
    prompt_ids: list[int],
    all_ids: list[int],
    available_rows: int,
    history_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return selected Thinker rows and one conditioning length per turn."""
    positions: list[int] = []
    lengths: list[int] = []
    assistant_ranges = _assistant_content_ranges(prompt_ids)
    selected_indices = list(range(len(assistant_ranges))) if history_indices is None else history_indices
    for index in selected_indices:
        if index >= len(assistant_ranges):
            raise ValueError(
                "DuplexOmni codec history references assistant turn "
                f"{index}, but the prompt contains only {len(assistant_ranges)} historical assistant turns"
            )
        start, end = assistant_ranges[index]
        end = min(end, available_rows)
        start = min(start, end)
        positions.extend(range(start, end))
        lengths.append(end - start)

    # Pooling accumulates one row per causal input position.  The final sampled
    # token has no following row and is intentionally omitted, matching the
    # official ``assist[:-1]`` training alignment.
    current_start = min(len(prompt_ids), available_rows)
    current_end = min(max(0, len(all_ids) - 1), available_rows)
    current_end = max(current_start, current_end)
    positions.extend(range(current_start, current_end))
    lengths.append(current_end - current_start)
    return positions, lengths


def _talker_prompt_ids(
    conditioning_lengths: list[int],
    history: torch.Tensor,
    conditioning_ids: list[list[int]] | None = None,
) -> list[int]:
    if len(conditioning_lengths) != int(history.shape[0]) + 1:
        raise ValueError(
            "DuplexOmni assistant/codec history mismatch: "
            f"conditioning turns={len(conditioning_lengths) - 1}, codec turns={history.shape[0]}"
        )
    result: list[int] = []
    for turn, condition_len in enumerate(conditioning_lengths[:-1]):
        ids = [CODEC_PAD] * condition_len if conditioning_ids is None else conditioning_ids[turn]
        if len(ids) != condition_len:
            raise ValueError("DuplexOmni conditioning identity length mismatch")
        result.extend(ids)
        result.append(CODEC_BOS)
        result.extend(int(item) for item in history[turn, 0].tolist())
        result.append(CODEC_EOS)
    current_ids = (
        [CODEC_PAD] * conditioning_lengths[-1]
        if conditioning_ids is None
        else conditioning_ids[-1]
    )
    if len(current_ids) != conditioning_lengths[-1]:
        raise ValueError("DuplexOmni current conditioning identity length mismatch")
    result.extend(current_ids)
    result.append(CODEC_BOS)
    return result


def _media_hashes_before(source_prompt: dict[str, Any], end: int) -> list[str]:
    """Return media identities whose placeholders occur before ``end``."""
    hashes_by_modality = source_prompt.get("mm_hashes")
    placeholders_by_modality = source_prompt.get("mm_placeholders")
    if not isinstance(hashes_by_modality, Mapping) or not isinstance(placeholders_by_modality, Mapping):
        return []
    result: list[str] = []
    for modality in sorted(set(hashes_by_modality) | set(placeholders_by_modality)):
        hashes = hashes_by_modality.get(modality) or []
        placeholders = placeholders_by_modality.get(modality) or []
        for index, placeholder in enumerate(placeholders):
            offset = getattr(placeholder, "offset", None)
            if offset is None and isinstance(placeholder, Mapping):
                offset = placeholder.get("offset")
            if isinstance(offset, int) and offset < end and index < len(hashes):
                result.append(f"{modality}:{hashes[index]}")
    return result


def _conditioning_identity_ids(
    source_prompt: dict[str, Any],
    prompt_ids: list[int],
    output_ids: list[int],
    history_indices: list[int],
    conditioning_lengths: list[int],
) -> list[list[int]]:
    """Build cache-only token identities for dynamic Talker embeddings.

    Talker conditioning uses prompt embeddings, so a constant CODEC_PAD token
    cannot safely identify it to vLLM's prefix cache.  Each turn instead gets
    a deterministic identity derived from the complete Thinker prefix and all
    media hashes visible at that boundary.  ``cache_salt`` already isolates
    sessions/epochs; the digest prevents false reuse within one lineage.
    """
    assistant_ranges = _assistant_content_ranges(prompt_ids)
    ends = [assistant_ranges[index][1] for index in history_indices]
    ends.append(max(len(prompt_ids), len(prompt_ids) + len(output_ids) - 1))
    all_ids = [*prompt_ids, *output_ids]
    cache_salt = str(source_prompt.get("cache_salt") or "")
    result: list[list[int]] = []
    for end, length in zip(ends, conditioning_lengths, strict=True):
        digest = hashlib.sha256()
        digest.update(cache_salt.encode("utf-8"))
        digest.update(int(end).to_bytes(8, "little", signed=False))
        for token_id in all_ids[:end]:
            digest.update(int(token_id).to_bytes(4, "little", signed=True))
        for media_hash in _media_hashes_before(source_prompt, end):
            digest.update(media_hash.encode("utf-8"))
            digest.update(b"\0")
        seed = digest.digest()
        identities: list[int] = []
        counter = 0
        while len(identities) < length:
            block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for offset in range(0, len(block), 2):
                identities.append(int.from_bytes(block[offset : offset + 2], "little") % CODEC_PAD)
                if len(identities) == length:
                    break
            counter += 1
        result.append(identities)
    return result


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Send only assistant rows needed by the DuplexOmni Talker."""
    del transfer_manager
    if not isinstance(pooling_output, Mapping):
        return None
    thinker_emb = pooling_output.get("hidden_states.layer_0")
    thinker_top = pooling_output.get(f"hidden_states.layer_{FINAL_THINKER_LAYER}")
    if not isinstance(thinker_emb, torch.Tensor) or not isinstance(thinker_top, torch.Tensor):
        logger.warning(
            "DuplexOmni Thinker payload is missing layer-0/final-layer-%d tensors",
            FINAL_THINKER_LAYER,
        )
        return None

    prompt_ids = _as_list(getattr(request, "prompt_token_ids", []))
    all_ids = _as_list(getattr(request, "all_token_ids", []))
    if not all_ids:
        all_ids = prompt_ids + _as_list(getattr(request, "output_token_ids", []))

    # P/D transfers Thinker KV to D, but Talker also needs the prompt rows from
    # layer 0 and DuplexOmni's final layer 48. The wire-compatible
    # ``prompt_layer_24`` field carries whichever hidden layer the pipeline
    # selected; append D's newly decoded rows before applying the ordinary
    # DuplexOmni assistant/codec alignment below.
    pd_prefill = getattr(request, "pd_prefill_payload", None)
    if pd_prefill is not None:
        decode_steps = min(int(thinker_emb.shape[0]), int(thinker_top.shape[0]))
        if decode_steps <= 0:
            raise RuntimeError("DuplexOmni P/D Thinker produced no decode rows")
        prompt_layer_0 = _pd_prompt_chunks(pd_prefill, "prompt_layer_0")
        prompt_layer_hidden = _pd_prompt_chunks(pd_prefill, "prompt_layer_24")
        layer_0_rows = sum(int(chunk.shape[0]) for chunk in prompt_layer_0)
        hidden_rows = sum(int(chunk.shape[0]) for chunk in prompt_layer_hidden)
        if layer_0_rows != len(prompt_ids) or hidden_rows != len(prompt_ids):
            raise RuntimeError(
                "DuplexOmni P/D prompt snapshot is not row-aligned: "
                f"prompt_tokens={len(prompt_ids)} layer0_rows={layer_0_rows} "
                f"hidden_rows={hidden_rows}"
            )
        # vLLM intentionally leaves the final prompt token uncached so D can
        # recompute logits.  Therefore the accumulated D rows are:
        #   [last prompt row, generated token 0, ..., generated token N-2]
        # The P snapshot already contains that prompt row; append only the
        # N-1 training-aligned generated rows required by Talker.
        decoded_layer_0 = thinker_emb[-decode_steps:].detach().cpu()[1:]
        decoded_layer_hidden = thinker_top[-decode_steps:].detach().cpu()[1:]
        thinker_emb = torch.cat((*prompt_layer_0, decoded_layer_0), dim=0)
        thinker_top = torch.cat((*prompt_layer_hidden, decoded_layer_hidden), dim=0)
        # Remote-KV CachedRequestState may retain only its last output token.
        # The accumulated row count is the authoritative completion length;
        # _conditioning_layout uses only this suffix length, not token values.
        all_ids = [*prompt_ids, *([0] * decode_steps)]
        logger.info(
            "DuplexOmni P/D assembled Talker conditioning request=%s "
            "prompt_rows=%d decode_steps=%d conditioning_rows=%d",
            getattr(request, "request_id", "-"),
            len(prompt_ids),
            decode_steps,
            max(0, decode_steps - 1),
        )
    available_rows = min(int(thinker_emb.shape[0]), int(thinker_top.shape[0]), max(0, len(all_ids) - 1))
    raw_info = _request_additional_information(request)
    history, history_indices = _codec_history_from_info(raw_info)
    pipeline_meta = _pipeline_meta(raw_info)
    positions, lengths = _conditioning_layout(
        prompt_ids,
        all_ids,
        available_rows,
        # A pipelined Thinker may finish before the immediately preceding
        # Talker codec exists.  Transfer all assistant rows now; the Talker
        # request selects valid turns after its ordered predecessor completes.
        history_indices=None if pipeline_meta is not None else history_indices,
    )
    # Validate the one-to-one turn contract before shipping a downstream request.
    if pipeline_meta is None:
        _talker_prompt_ids(lengths, history)
    if lengths[-1] <= 0:
        raise RuntimeError("DuplexOmni Thinker produced no training-aligned assistant conditioning rows")

    selected = torch.tensor(positions, dtype=torch.long, device=thinker_emb.device)
    payload_meta = MetaStruct(duplexomni=True, finished=torch.tensor(True, dtype=torch.bool))
    if pipeline_meta is not None:
        payload_meta.duplexomni_pipeline = True
        payload_meta.duplexomni_pipeline_session_id = str(pipeline_meta[PIPELINE_SESSION_ID])
        payload_meta.duplexomni_pipeline_epoch = int(pipeline_meta.get(PIPELINE_EPOCH, 0))
        payload_meta.duplexomni_pipeline_slot = int(pipeline_meta.get(PIPELINE_SLOT, 0))
        payload_meta.duplexomni_pipeline_final = bool(pipeline_meta.get(PIPELINE_FINAL, False))
        payload_meta.duplexomni_pipeline_base_turns = int(pipeline_meta.get(PIPELINE_BASE_TURNS, 0))
    payload = OmniPayloadStruct(
        embed=EmbeddingsStruct(duplex_conditioning=thinker_emb.index_select(0, selected).detach().cpu()),
        hidden_states=HiddenStatesStruct(
            duplex_conditioning=thinker_top.index_select(0, selected.to(thinker_top.device)).detach().cpu()
        ),
        ids=IdsStruct(duplex_conditioning_lengths=lengths),
        # In pipeline mode the authoritative codec history is injected at the
        # Talker boundary.  Omitting ``codes`` prevents the earlier incomplete
        # snapshot from overwriting that history when connector payloads merge.
        codes=None if pipeline_meta is not None else CodesStruct(ref=history),
        meta=payload_meta,
    )
    return to_dict(payload)


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Build deterministic Talker cache identities for each completed slice."""
    del requires_multimodal_data, streaming_context
    results: list[OmniTokensPrompt] = []
    for index, thinker_output in enumerate(source_outputs):
        source_prompt = _prompt_dict(prompt, index)
        prompt_ids = _as_list(getattr(thinker_output, "prompt_token_ids", []))
        # The public OmniRequestOutput does not retain the Thinker prompt ids.
        # The processed source prompt does, and is also the authoritative
        # prompt after application-level canonical block rendering.  Without
        # this fallback every pipelined Talker request looks like only the
        # current 24-token assistant slice, so historical codec conditioning
        # is silently omitted and prefix caching can never match.
        if not prompt_ids:
            prompt_ids = _as_list(source_prompt.get("prompt_token_ids"))
        output = thinker_output.outputs[0]
        output_ids = _as_list(getattr(output, "cumulative_token_ids", getattr(output, "token_ids", [])))
        history, history_indices = _codec_history_from_plain_prompt(prompt, index)
        _, lengths = _conditioning_layout(
            prompt_ids,
            prompt_ids + output_ids,
            available_rows=max(0, len(prompt_ids) + len(output_ids) - 1),
            history_indices=history_indices,
        )
        logger.debug(
            "[DuplexOmni] built Talker input prompt_tokens=%d history_turns=%d conditioning_lengths=%s",
            len(prompt_ids),
            len(history_indices),
            lengths,
        )
        conditioning_ids = _conditioning_identity_ids(
            source_prompt,
            prompt_ids,
            output_ids,
            history_indices,
            lengths,
        )
        talker_ids = _talker_prompt_ids(lengths, history, conditioning_ids)
        item: OmniTokensPrompt = {
            "prompt_token_ids": talker_ids,
            "additional_information": {"meta": {"duplexomni": True}},
            "multi_modal_data": None,
            "mm_processor_kwargs": None,
        }
        pipeline_meta = _pipeline_meta(source_prompt.get("additional_information"))
        if pipeline_meta is not None:
            item["additional_information"] = {
                "codes": {"ref": history},
                "ids": {"duplex_history_indices": history_indices},
                "meta": {
                    "duplexomni": True,
                    "duplexomni_pipeline": True,
                    PIPELINE_SESSION_ID: pipeline_meta[PIPELINE_SESSION_ID],
                    PIPELINE_EPOCH: int(pipeline_meta.get(PIPELINE_EPOCH, 0)),
                    PIPELINE_SLOT: int(pipeline_meta.get(PIPELINE_SLOT, 0)),
                    PIPELINE_FINAL: bool(pipeline_meta.get(PIPELINE_FINAL, False)),
                    PIPELINE_BASE_TURNS: int(pipeline_meta.get(PIPELINE_BASE_TURNS, 0)),
                },
            }
        if source_prompt.get("cache_salt") is not None:
            item["cache_salt"] = source_prompt["cache_salt"]
        results.append(item)
    return results


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Require one complete six-frame DuplexOmni slice and expose its RVQ codes."""
    payload = qwen3_omni.talker2code2wav_full_payload(transfer_manager, pooling_output, request)
    if payload is None:
        return None
    flat_codes = payload.get("codes", {}).get("audio")
    if not isinstance(flat_codes, list) or len(flat_codes) < NUM_CODEBOOKS * FRAMES_PER_SLICE:
        raise RuntimeError(
            "DuplexOmni Talker must produce at least "
            f"{FRAMES_PER_SLICE} frames x {NUM_CODEBOOKS} codebooks; "
            f"received {len(flat_codes) if isinstance(flat_codes, list) else 'no'} codec ids"
        )
    if len(flat_codes) % NUM_CODEBOOKS:
        raise RuntimeError(
            "DuplexOmni Talker returned a non-frame-aligned codec payload: "
            f"{len(flat_codes)} ids"
        )

    # The official loop appends six real frames, samples once more to check
    # for codec EOS, and discards that seventh sample when it is not EOS.  A
    # single vLLM request cannot express that control-flow boundary, so retain
    # the first six rows here and report whether the validation EOS arrived.
    frame_count = len(flat_codes) // NUM_CODEBOOKS
    code_matrix = torch.as_tensor(flat_codes, dtype=torch.long).reshape(NUM_CODEBOOKS, frame_count)
    payload["codes"]["audio"] = code_matrix[:, :FRAMES_PER_SLICE].reshape(-1).tolist()
    output_ids = _as_list(getattr(request, "output_token_ids", []))
    eos_emitted = CODEC_EOS in output_ids
    valid_turn = eos_emitted and frame_count == FRAMES_PER_SLICE
    if frame_count > FRAMES_PER_SLICE:
        logger.warning(
            "DuplexOmni discarded %d post-slot Talker frame(s); eos=%s output_ids=%s",
            frame_count - FRAMES_PER_SLICE,
            eos_emitted,
            output_ids,
        )
    payload.setdefault("meta", {})["return_codec_codes"] = True
    payload["meta"]["duplexomni"] = True
    payload["meta"]["duplexomni_eos_emitted"] = eos_emitted
    payload["meta"]["duplexomni_valid_turn"] = valid_turn
    return payload


__all__ = [
    "FRAMES_PER_SLICE",
    "NUM_CODEBOOKS",
    "thinker2talker_full_payload",
    "thinker2talker_token_only",
    "talker2code2wav_full_payload",
]
