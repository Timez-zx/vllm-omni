# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
    prefill_only_channel,
    request_is_prefill_only,
)

logger = logging.getLogger(__name__)

# [live-vllm diagnosis] same env as the orchestrator's [AUDIO-CHUNK] stamps:
# per-chunk emit stamps at the stage-1 send point.
import os as _os

_LOG_CHUNK_EMIT = _os.environ.get("VLLM_OMNI_LOG_AUDIO_CHUNKS", "0") not in ("0", "", "false", "False")

# Drop payload fields the consumer never reads on the decode path; see the
# long note at the return site. Default ON -- it is a pure "stop sending unread
# bytes" change -- with an env to restore the fat payload for A/B.
_T2T_LEAN_DECODE = _os.environ.get("VLLM_OMNI_T2T_LEAN_DECODE", "1") not in ("0", "", "false", "False")



# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"


def _s01_edge_is_intra_process() -> bool:
    """True when stages 0 and 1 are colocated in one process.

    Mirrors ColocInProcConnector's flat-group parse of
    VLLM_OMNI_COLOCATE_STAGES. When true, payloads on the thinker->talker edge
    pass by reference through the in-process store, so the transport copy to
    CPU is pure waste (measured: 291 MB and 57-77% of the talker's added
    latency at long contexts).
    """
    import os

    raw = os.environ.get("VLLM_OMNI_COLOCATE_STAGES", "").strip()
    if not raw:
        return False
    members: set[int] = set()
    for pair in raw.split(","):
        guest_s, _, host_s = pair.partition(":")
        try:
            members.add(int(guest_s.strip()))
            members.add(int(host_s.strip()))
        except ValueError:
            return False
    return 0 in members and 1 in members


_S01_INTRA_PROCESS = _s01_edge_is_intra_process()
_SNAPSHOT_DEVICE_LOGGED = False


def _snapshot_for_talker(t: torch.Tensor) -> torch.Tensor:
    """Detach-and-snapshot a thinker tensor for shipment to the talker.

    The copy is load-bearing beyond transport: it decouples the payload from
    GPU buffers the next engine step overwrites (async scheduling relies on
    that). Separate-process mode must go through CPU anyway (SHM transport).
    In-process mode keeps the snapshot but takes it as a same-device clone --
    a D2D copy instead of a pageable D2H, and the consumer's .to(device)
    becomes a no-op. The synchronize matches the implicit sync today's
    .cpu() performs, so payload-readiness semantics are unchanged across the
    producer thread / consumer stream boundary.
    """
    global _SNAPSHOT_DEVICE_LOGGED
    if not _SNAPSHOT_DEVICE_LOGGED:
        _SNAPSHOT_DEVICE_LOGGED = True
        logger.warning(
            "[stage0->1] first payload tensor: device=%s intra_process=%s "
            "(cuda+intra = D2D snapshot active; cpu = an upstream copy already paid the D2H)",
            t.device,
            _S01_INTRA_PROCESS,
        )
    if _S01_INTRA_PROCESS and t.is_cuda:
        # Already a decoupled snapshot: under full tri-colocation the runner's
        # async output path ships its D2D clone (keep_on_device) instead of a
        # CPU copy, and this builder is that payload's sole owner -- pass the
        # reference through, no further copy needed.
        return t.detach()
    return t.detach().cpu()
# Per-model REPLACE-keys for the full-payload accumulator.  Keys in this
# set use REPLACE semantics (subsequent emissions discard prior chunks)
# instead of CONCAT.  qwen3-omni currently has none — model_outputs is
# not emitted by the thinker/talker forward.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()

_QWEN3_CODEC_CODEBOOK_SIZE = 2048
_QWEN3_CODEC_PAD_TOKEN_ID = 4196
_QWEN3_CODEC_BOS_TOKEN_ID = 4197
_QWEN3_CODEC_EOS_TOKEN_ID = 4198


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


def _compute_talker_prompt_ids_length(info: OmniPayload, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    ids = info.get("ids", {})
    thinker_sequences = torch.tensor(ids["all"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(ids["prompt"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    # The closing sentinel must be the end of THIS DELTA, not the end of the whole session.
    #
    # `ids["all"]` is the request's full accumulated sequence; `ids["prompt"]` is only the
    # rows of this forward. They are equal whenever every stage-0 forward is a talker
    # segment, which is why using the former was harmless -- and it is precisely wrong as
    # soon as they are not. A frames-on-arrival append is a stage-0 forward with no talker
    # segment, so `all` runs ahead of `prompt` and this sentinel overshoots: the final user
    # block's length is computed as (full_length - s), the placeholder is sized far too
    # large, and the talker reads past its codec embedding table --
    # `indexSelectSmallIndex: srcIndex < srcSelectDimSize`, stage 1 dead, engine gone.
    #
    # min() keeps it a no-op for every existing path and bounds it for the new one. This is
    # the fix three earlier attempts were working around from the outside: withholding the
    # append, passing it through, and stripping its chatml headers all left this arithmetic
    # untouched and all died identically on turn 1.
    _delta_end = min(int(thinker_sequences.shape[-1]), int(input_ids.shape[-1]))
    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([_delta_end], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _as_tensor_or_none(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return value[0].detach().cpu()
    return None


def _is_valid_qwen3_codec_token_id(token_id: Any) -> bool:
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return False
    return 0 <= token_id < _QWEN3_CODEC_CODEBOOK_SIZE


def _extract_qwen3_full_payload_codec_rows(
    code_predictor_codes: torch.Tensor,
    output_token_ids: list[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Filter full-payload codec rows by the authoritative output ids."""
    if code_predictor_codes.ndim != 2 or code_predictor_codes.numel() == 0:
        return code_predictor_codes, {
            "raw_rows": int(code_predictor_codes.shape[0]) if code_predictor_codes.ndim > 0 else 0,
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": 0,
        }

    trailing_placeholder_count = 0
    while (
        trailing_placeholder_count < len(output_token_ids) and output_token_ids[-1 - trailing_placeholder_count] == -1
    ):
        trailing_placeholder_count += 1

    aligned_len = min(int(code_predictor_codes.shape[0]), len(output_token_ids))
    if aligned_len <= 0:
        return code_predictor_codes[:0], {
            "raw_rows": int(code_predictor_codes.shape[0]),
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": trailing_placeholder_count,
        }

    aligned_rows = code_predictor_codes[-aligned_len:]
    aligned_token_ids = output_token_ids[-aligned_len:]
    aligned_token_mask = torch.tensor(
        [_is_valid_qwen3_codec_token_id(token_id) for token_id in aligned_token_ids],
        dtype=torch.bool,
        device=aligned_rows.device,
    )
    row_valid_mask = (aligned_rows.max(dim=1).values < _QWEN3_CODEC_CODEBOOK_SIZE) & (
        aligned_rows.min(dim=1).values >= 0
    )
    filtered_rows = aligned_rows[aligned_token_mask & row_valid_mask]
    if filtered_rows.numel() == 0:
        filtered_rows = aligned_rows[:0]
    return filtered_rows, {
        "raw_rows": int(code_predictor_codes.shape[0]),
        "aligned_rows": aligned_len,
        "valid_rows": int(filtered_rows.shape[0]) if filtered_rows.ndim > 0 else 0,
        "trailing_placeholder_count": trailing_placeholder_count,
    }


# =========================
# PD disaggregation helpers
# =========================


def _get_prefill_multimodal_output(
    request_id: str,
    streaming_context: Any | None,
) -> dict[str, Any] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    by_req = bridge_states.get("pd_prefill_multimodal_output_by_req")
    if not isinstance(by_req, dict):
        return None
    prefill_mm = by_req.get(request_id)
    return prefill_mm if isinstance(prefill_mm, Mapping) else None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    In PD mode the prefill engine processes the prompt and the decode engine
    generates tokens starting from position 1.  This function concatenates
    them, removing the overlapping token(s):

        merged = prefill[:P] + decode[overlap:]

    where overlap = P + D - expected_total.
    """
    try:
        p_layers = prefill_mm.get("hidden_states", {}).get("layers", {})
        p_emb = p_layers[int(_EMBED_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
        p_hid = p_layers[int(_HIDDEN_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError) as exc:
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, Mapping) else type(prefill_mm).__name__
        logger.error(
            "_merge_pd_embeddings: failed to extract prefill embeddings (%s). "
            "Expected keys %r and %r, got: %s. "
            "Falling back to decode-only embeddings – talker user-segment will be degraded.",
            exc,
            _EMBED_LAYER_KEY,
            _HIDDEN_LAYER_KEY,
            available_keys,
        )
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def _resolve_tts_token_embedding(
    key: str,
    *,
    thinker_mm: dict[str, Any],
    prefill_mm: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return TTS BOS/EOS/PAD embedding tensors for the talker projection path.

    Values are taken from the current thinker (decode) ``multimodal_output``; in
    PD mode, missing keys may be filled from the paired prefill stage output.
    """
    val = thinker_mm.get("embed", {}).get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get("embed", {}).get(key)
    return val.detach().to(device=device, dtype=torch.float) if val is not None else None


# =========================
# Streaming input helpers
# =========================


def _construct_thinker2talker_streaming_input_async_chunk(
    is_finished: bool,
    request,
    thinker_emb,
    thinker_hid,
    transfer_manager,
) -> OmniPayloadStruct | None:
    """Build Thinker -> Talker payloads for realtime streaming input chunks.

    A resumable realtime request reuses the same logical request id across
    audio segments. The first streaming prefill chunk is cached and returns ``None`` so the
    connector does not emit an incomplete downstream chunk. The following
    decode chunk flushes that cached prefill together with the current Thinker
    output, keeping Talker ids and tensor rows aligned.
    """
    request_id = request.external_req_id
    output_token_ids = request.output_token_ids
    # Convert ConstantList to regular list for OmniSerializer serialization
    output_token_ids = _ensure_list(output_token_ids)
    speaker = extract_speaker_from_request(request)
    language = extract_language_from_request(request)
    finished = torch.tensor(is_finished, dtype=torch.bool)
    emb_cpu = _snapshot_for_talker(thinker_emb)
    hid_cpu = _snapshot_for_talker(thinker_hid)

    # Has the engine finished prefilling every prompt token it has been given?
    # This is the ENGINE's own bookkeeping, and it is the only trustworthy answer:
    # a watermark of "prompt tokens shipped to the talker" cannot work, because
    # prefill-only appends (frames prefilled on arrival) deliberately ship nothing,
    # so their tokens would be counted against the next segment forever -- measured
    # as segments reported short by exactly 1, 2 or 4 frames' worth of tokens.
    # Placeholders subtracted for the same reason the chunk adapter's
    # _confirmed_num_computed_tokens does it: async scheduling advances
    # num_computed_tokens for output tokens that are not committed yet, so the raw
    # counter would call a segment prefilled while rows are still missing.
    prompt_len = len(request.prompt_token_ids)
    computed = max(
        0,
        int(getattr(request, "num_computed_tokens", 0) or 0)
        - int(getattr(request, "num_output_placeholders", 0) or 0),
    )
    fully_prefilled = computed >= prompt_len

    if output_token_ids:
        if thinker_emb.shape[0] > 1:
            # if thinker_emb.shape[0] > 1, new streaming input segment is added
            # and will transfer prefill embeddings and hidden states to talker.
            #
            # ACCUMULATE across steps, and size the ids from the SEGMENT, not from
            # this step. One segment's prefill can span several engine steps --
            # chunked prefill splits it whenever the batch token budget runs out,
            # which a long chunk (many frames, or a compression seed) does routinely
            # under load -- and every step arrives here. Sizing the ids by the last
            # step's row count shipped the segment's TAIL: the delta reached the
            # talker without its leading `<|im_end|>\n<|im_start|>user` and with only
            # the trailing `<|im_start|>assistant`. Measured consequence: stage 1 died
            # in _thinker_to_talker_prefill (a lone im_start collapses torch.nonzero's
            # [n,1] to a 0-d tensor that torch.cat refuses), and with that crash
            # guarded the failure goes SILENT instead -- the talker conditioned on an
            # assistant header alone, because compute_talker_prompt_ids_length reads
            # the same truncated ids and returns 9.
            prev = transfer_manager._pending_streaming_prefills.get(request_id)
            prompt_rows = int(thinker_emb.shape[0])
            if prev is not None:
                prev_emb = prev.get("embed", {}).get("prefill")
                prev_hid = prev.get("hidden_states", {}).get("output")
                if isinstance(prev_emb, torch.Tensor) and isinstance(prev_hid, torch.Tensor):
                    emb_cpu = torch.cat((prev_emb, emb_cpu), dim=0)
                    hid_cpu = torch.cat((prev_hid, hid_cpu), dim=0)
                    prompt_rows = int(prev.get("_prompt_rows", prev_emb.shape[0])) + int(
                        thinker_emb.shape[0]
                    )
            # A split segment is otherwise invisible, and it is the shape that broke
            # stage 1 -- so report its presence, at WARNING because this module's
            # logger is not part of vLLM's configured tree and its INFO lines never
            # reach the log at all.
            if prompt_rows != thinker_emb.shape[0]:
                logger.warning(
                    "[stage0->1] req %s: streaming segment split across prefill steps "
                    "(%d rows accumulated; computed=%d prompt_len=%d)",
                    request_id, prompt_rows, computed, prompt_len,
                )
            ids_len = prompt_rows
            payload = OmniPayloadStruct(
                meta=MetaStruct(finished=finished),
                embed=EmbeddingsStruct(prefill=emb_cpu),
                hidden_states=HiddenStatesStruct(output=hid_cpu),
                ids=IdsStruct(
                    all=_ensure_list(request.all_token_ids[-ids_len - 1 :]),
                    prompt=_ensure_list(request.prompt_token_ids[-ids_len:]),
                ),
                speaker=speaker,
                language=language,
            )
            pending = to_dict(payload)
            pending["_prompt_rows"] = prompt_rows
            transfer_manager._pending_streaming_prefills[request_id] = pending
            return None
        else:
            if (
                not fully_prefilled
                and transfer_manager._pending_streaming_prefills.get(request_id) is not None
            ):
                # Prompt tokens are still unprefilled, so the payload about to ship
                # covers only PART of the segment: the talker will be conditioned on a
                # fragment. Reported, not repaired -- and deliberately so. Withholding
                # the payload until the rest arrives was measured to kill stage 1 with
                # `KeyError: 'prefill'`: the stages are coupled step-by-step, so the
                # next step's decode-only payload is then read as this segment's
                # prefill. A real repair belongs in the payload framing (one payload
                # per SEGMENT rather than per step), which is an upstream change.
                logger.warning(
                    "[stage0->1] req %s: shipping a PARTIAL segment -- %d of %d prompt "
                    "tokens prefilled; the talker sees a fragment of this turn",
                    request_id, computed, prompt_len,
                )
            save_payload = transfer_manager._pending_streaming_prefills.pop(request_id, None)
            if save_payload is not None:
                saved_prefill = save_payload.get("embed", {}).get("prefill")
                saved_output = save_payload.get("hidden_states", {}).get("output")
                if isinstance(saved_prefill, torch.Tensor) and isinstance(saved_output, torch.Tensor):
                    return OmniPayloadStruct(
                        meta=MetaStruct(finished=finished),
                        embed=EmbeddingsStruct(prefill=torch.cat((saved_prefill, emb_cpu), dim=0)),
                        hidden_states=HiddenStatesStruct(output=torch.cat((saved_output, hid_cpu), dim=0)),
                        ids=IdsStruct(
                            all=save_payload.get("ids", {}).get("all"),
                            prompt=save_payload.get("ids", {}).get("prompt"),
                        ),
                        speaker=speaker,
                        language=language,
                    )
            # [lean decode payload] Ship only what the consumer reads. On this
            # path -- a plain decode step, no pending prefill to flush -- the
            # talker reads embed.decode and nothing else: hidden_states.output
            # is consumed only by talker_preprocess_prefill (qwen3_omni.py:888)
            # and ids is consumed only there too (ids.all / ids.prompt at
            # :892-899); ids.output has no reader anywhere on the receiving
            # side. Both were being sent per token per session anyway: the
            # hidden row is a fixed 4096 bytes (half the payload) and the id
            # list grows through the turn, which is what made per-turn transfer
            # bytes grow quadratically. Measured motivation: the talker's GPU
            # runs 2.5 ms per 28.7 ms pass (8.5% duty) at 64 users while ~1 ms
            # per session per pass goes to CPU -- deserialize and payload build
            # are 30% of that stage's samples, so the cheapest capacity left is
            # to stop sending unread bytes. VLLM_OMNI_T2T_LEAN_DECODE=0 restores
            # the fat payload for A/B.
            if _T2T_LEAN_DECODE:
                return OmniPayloadStruct(
                    meta=MetaStruct(finished=finished),
                    embed=EmbeddingsStruct(decode=emb_cpu),
                    speaker=speaker,
                    language=language,
                )
            return OmniPayloadStruct(
                meta=MetaStruct(
                    finished=finished,
                ),
                embed=EmbeddingsStruct(decode=emb_cpu),
                hidden_states=HiddenStatesStruct(output=hid_cpu),
                ids=IdsStruct(output=output_token_ids),
                speaker=speaker,
                language=language,
            )
    else:
        if not is_finished:
            # do not send async chunk mode placeholder token or embedding/hidden of the stop token
            return None
        return OmniPayloadStruct(
            meta=MetaStruct(finished=finished),
            embed=EmbeddingsStruct(decode=emb_cpu),
            hidden_states=HiddenStatesStruct(output=hid_cpu),
            speaker=speaker,
            language=language,
        )


@dataclass
class _Thinker2TalkerStreamingState:
    last_prompt_len: int = 0
    last_output_len: int = 0
    merged_sequences: list[int] = field(default_factory=list)


@dataclass
class _Qwen3OmniStreamingState:
    thinker2talker: _Thinker2TalkerStreamingState = field(default_factory=_Thinker2TalkerStreamingState)
    talker2code2wav_last_seq_len: int = 0


def _get_qwen3_streaming_state(
    request_id: str,
    streaming_context: Any | None,
) -> _Qwen3OmniStreamingState:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    per_model_state = bridge_states.setdefault("qwen3_omni", {})
    state = per_model_state.get(request_id)
    if state is None:
        state = _Qwen3OmniStreamingState()
        per_model_state[request_id] = state
    return state


def _get_streaming_talker_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    new_prompt_len_snapshot: int | None = None,
    streaming_context: Any | None = None,
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int]]:
    """Return prompt/output token deltas for the current streaming segment.

    In non-async-chunk streaming, Thinker's prompt may already include the
    next input segment. Remove that new prompt tail before building the Talker
    delta for the previous segment.

    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output


def _get_streaming_codec_delta_len(
    cur_seq_len: int,
    request_id: str,
    talker_output: Any,
    streaming_context: Any | None = None,
) -> int:
    """Return newly added seq_len for talker->code2wav in streaming mode."""
    state = _get_qwen3_streaming_state(request_id, streaming_context)
    prev_seq_len = state.talker2code2wav_last_seq_len
    seq_len = cur_seq_len - prev_seq_len
    state.talker2code2wav_last_seq_len = cur_seq_len + 1
    if bool(getattr(talker_output, "finished", False)):
        # Final segment: clear history to avoid cross-session carry-over.
        state.talker2code2wav_last_seq_len = 0
    return seq_len


# =========================
# Thinker -> Talker
# =========================


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]

    # A prefill-only append ships NOTHING to the talker, and this is now correct rather than
    # a guess -- two independent defects had to be fixed before it could be, and each one hid
    # the other:
    #
    # 1. `-1` reaching the codec embedding. Under async scheduling `token_ids_cpu` never holds
    #    a sampled id; vLLM writes the sentinel -1 and patches the real value onto the GPU row
    #    from `prev_sampled_token_ids`, which only reaches requests that were in the PREVIOUS
    #    forward's batch. An append parks the talker, so it leaves the batch and is re-admitted
    #    on an output row -- readable only from CPU, and reads -1. codec_embedding has 3072
    #    rows, so `indexSelectSmallIndex: srcIndex < srcSelectDimSize`, stage 1 dead. FIXED by
    #    `async_scheduling: false` on stage 1 in the deploy YAML, not here.
    #
    # 2. A prefill tensor labelled as a decode payload -- what this branch prevents. An append
    #    stops on the very forward that prefills it, and omni_ar_scheduler.py clears
    #    `_output_token_ids` before calling save_async, so the code below takes its
    #    decode-shaped path and ships the append's [N_rows, 1024] prefill tensor as
    #    `embed.decode`. The runner then copies it into a ONE-ROW decode slot:
    #    `RuntimeError: output with shape [1, 1024] doesn't match the broadcast shape
    #    [222, 1024]` at gpu_model_runner.py:1793.
    #
    # Defect 2 was invisible until defect 1 was fixed -- the CUDA assert killed the process
    # first, on the same turn, which is why four earlier attempts all "failed identically"
    # while actually failing for two different reasons at once.
    # STRUCTURAL, not marker-based. The marker (SamplingParams.extra_args) does reach the stage
    # processes -- proven by log -- but it does NOT survive to here: by the time either the
    # scheduler's save_async or this save-thread call reads it, the next streaming update has
    # replaced sampling_params, so both the live read and an enqueue-time snapshot came back
    # False while the crash they were meant to prevent happened. Two silent misses; stop
    # relying on transported state.
    #
    # The condition below cannot be lost, because it is a property of THIS forward: a forward
    # that both PREFILLS (more than one row of thinker embeddings) and ENDS the segment carries
    # no talker obligation -- the segment produced no text, so there is nothing to speak. Only
    # a frames-on-arrival append can do that, because it is submitted with max_tokens=1 and so
    # stops on the very forward that prefills it. With the feature off, a segment's first
    # forward never stops, so this branch is unreachable and the shipping path is untouched.
    if is_finished and isinstance(multimodal_output, Mapping):
        _emb = multimodal_output.get("hidden_states", {})
        _layers = _emb.get("layers", {}) if isinstance(_emb, dict) else {}
        _rows = _layer_tensor(_layers, _EMBED_LAYER_KEY)
        if _rows is not None and int(_rows.shape[0]) > 1:
            logger.info(
                "[prefill-only] context-only forward: %d rows prefilled and the segment ended, "
                "so nothing is shipped to the talker (req=%s chunk_id=%d)",
                int(_rows.shape[0]), request_id, chunk_id,
            )
            return None

    if not isinstance(multimodal_output, Mapping):
        logger.debug("thinker2talker_async_chunk: skip non-dict multimodal_output for req=%s", request_id)
        return None

    thinker_hs = multimodal_output.get("hidden_states", {})
    thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    thinker_embed_raw = multimodal_output.get("embed", {})
    thinker_embed = thinker_embed_raw if isinstance(thinker_embed_raw, dict) else {}
    thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None or thinker_hid is None:
        logger.debug(
            "thinker2talker_async_chunk: missing thinker layers for req=%s (embed=%s hidden=%s)",
            request_id,
            thinker_emb is not None,
            thinker_hid is not None,
        )
        return None
    speaker = extract_speaker_from_request(request)
    language = extract_language_from_request(request)

    def _maybe_cpu(t: Any) -> torch.Tensor | None:
        return _snapshot_for_talker(t) if isinstance(t, torch.Tensor) else None

    if chunk_id == 0:
        all_token_ids = _ensure_list(request.all_token_ids)
        prompt_token_ids = _ensure_list(request.prompt_token_ids)
        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=_snapshot_for_talker(thinker_emb),
                tts_bos=_maybe_cpu(thinker_embed.get("tts_bos")),
                tts_eos=_maybe_cpu(thinker_embed.get("tts_eos")),
                tts_pad=_maybe_cpu(thinker_embed.get("tts_pad")),
            ),
            hidden_states=HiddenStatesStruct(output=_snapshot_for_talker(thinker_hid)),
            ids=IdsStruct(all=all_token_ids, prompt=prompt_token_ids),
            meta=MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool)),
            speaker=speaker,
            language=language,
        )
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            payload.embed.prefill = torch.cat(
                (save_payload.get("embed", {}).get("prefill"), payload.embed.prefill), dim=0
            )
            payload.hidden_states.output = torch.cat(
                (save_payload.get("hidden_states", {}).get("output"), payload.hidden_states.output), dim=0
            )
            prefill_shape = payload.embed.prefill.shape[0]
            if not is_finished and prefill_shape <= len(prompt_token_ids):
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
    else:
        if request.resumable:
            return _construct_thinker2talker_streaming_input_async_chunk(
                is_finished, request, thinker_emb, thinker_hid, transfer_manager
            )
        if thinker_emb.shape[0] > 1:
            logger.warning(
                "Unexpected multiple embeddings in thinker2talker_async_chunk for chunk_id %d: "
                "request_id %s, num_computed_tokens%d %s. Expected shape [1, D].",
                chunk_id,
                request_id,
                request.num_computed_tokens,
                thinker_emb.shape,
            )
            return None
        meta = MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool))
        payload = OmniPayloadStruct(
            meta=meta,
            embed=EmbeddingsStruct(decode=_snapshot_for_talker(thinker_emb)),
            speaker=speaker,
            language=language,
        )
    return payload


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete thinker output for the non-async connector path."""
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "thinker2talker_full_payload: pooling_output not a dict (type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    layers = {
        0: pooling_output.get("hidden_states.layer_0"),
        24: pooling_output.get("hidden_states.layer_24"),
    }
    thinker_emb = _layer_tensor(layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None:
        hidden = pooling_output.get("hidden")
        thinker_emb = hidden if isinstance(hidden, torch.Tensor) else None
    if thinker_emb is None or thinker_hid is None:
        logger.warning(
            "thinker2talker_full_payload: missing thinker tensors for req=%s "
            "(embed=%s hidden=%s keys=%s); consumer wait gate may hang.",
            rid,
            thinker_emb is not None,
            thinker_hid is not None,
            list(pooling_output.keys()),
        )
        return None

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])
    all_token_ids = _ensure_list(getattr(request, "all_token_ids", None) or [])
    if not all_token_ids:
        output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = list(prompt_token_ids) + list(output_token_ids)

    # Drop the terminal stop-token row only when more than one row was
    # accumulated; trimming a single row would ship 0 conditioning tensors
    # while ids still has tokens and break talker prefill alignment.
    if isinstance(thinker_emb, torch.Tensor) and thinker_emb.shape[0] > 1:
        thinker_emb_prefill = thinker_emb[:-1]
    else:
        thinker_emb_prefill = thinker_emb
    if isinstance(thinker_hid, torch.Tensor) and thinker_hid.shape[0] > 1:
        thinker_hid_prefill = thinker_hid[:-1]
    else:
        thinker_hid_prefill = thinker_hid

    emb_rows = int(thinker_emb_prefill.shape[0]) if isinstance(thinker_emb_prefill, torch.Tensor) else 0
    hid_rows = int(thinker_hid_prefill.shape[0]) if isinstance(thinker_hid_prefill, torch.Tensor) else 0
    if len(all_token_ids) > 0 and (emb_rows == 0 or hid_rows == 0):
        logger.warning(
            "thinker2talker_full_payload: empty thinker conditioning for req=%s "
            "(ids_len=%s embed_rows=%s hidden_rows=%s); withholding payload.",
            rid,
            len(all_token_ids),
            emb_rows,
            hid_rows,
        )
        return None

    payload: OmniPayload = {
        "embed": {
            "prefill": thinker_emb_prefill.detach().cpu(),
            "tts_bos": _as_tensor_or_none(pooling_output.get("embed.tts_bos")),
            "tts_eos": _as_tensor_or_none(pooling_output.get("embed.tts_eos")),
            "tts_pad": _as_tensor_or_none(pooling_output.get("embed.tts_pad")),
        },
        "hidden_states": {"output": thinker_hid_prefill.detach().cpu()},
        "ids": {"all": list(all_token_ids), "prompt": list(prompt_token_ids)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
    speaker = extract_speaker_from_request(request)
    if speaker is not None:
        payload["speaker"] = speaker
    language = extract_language_from_request(request)
    if language is not None:
        payload["language"] = language
    return payload


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Orchestrator-side placeholder builder for Stage-1 (Talker) when
    ``async_chunk=False``.

    After the communication-layer refactor, this function only allocates a
    placeholder ``prompt_token_ids`` of the correct length so the scheduler can
    reserve KV-cache slots. It does **not** forward bulk tensors.

    Bulk talker conditioning is sent through the connector. Speaker and
    language are also copied from the original prompt so they survive when
    Stage-0 request metadata is unavailable to the connector payload.

    ``prompt`` / ``requires_multimodal_data`` are kept for call-site signature
    compatibility with other orchestrator input processors; they are unused.
    """
    talker_inputs: list[OmniTokensPrompt] = []
    for i, thinker_output in enumerate(source_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.cumulative_token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        thinker_sequences = prompt_token_ids + output_ids
        thinker_input_ids = prompt_token_ids
        info_for_len = {"ids": {"all": thinker_sequences, "prompt": thinker_input_ids}}
        prompt_len = _compute_talker_prompt_ids_length(info_for_len, device="cpu")
        # Keep this fallback until the connector reliably preserves voice metadata.
        additional_information = to_dict(
            OmniPayloadStruct(
                speaker=extract_speaker_from_prompt(prompt, index=i),
                language=extract_language_from_prompt(prompt, index=i),
            )
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=additional_information or None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Multimodal output version.
    """
    request_id = request.external_req_id
    code_predictor_codes = None
    if isinstance(multimodal_output, Mapping):
        talker_codes = multimodal_output.get("codes", {})
        if isinstance(talker_codes, dict):
            code_predictor_codes = talker_codes.get("audio")

    sampling_params = getattr(request, "sampling_params", None)
    stop_token_ids = set(getattr(sampling_params, "stop_token_ids", None) or [])
    stop_token_id = getattr(sampling_params, "stop_token_id", None)
    if stop_token_id is not None:
        stop_token_ids.add(stop_token_id)

    append_codes = (
        isinstance(code_predictor_codes, torch.Tensor)
        and code_predictor_codes.numel() > 0
        and bool(code_predictor_codes.any())
    )
    if append_codes:
        first_codebook = int(code_predictor_codes[0, 0].item())
        if first_codebook in stop_token_ids:
            logger.debug("skip stop-token codec frame: first_codebook=%s", first_codebook)
            append_codes = False
    if append_codes:
        transfer_manager.code_prompt_token_ids[request_id].append(code_predictor_codes)
    elif not is_finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    # Segment-local, NOT session-global. `put_req_chunk` survives segment
    # boundaries (the connector key needs continuity), but `code_prompt_token_ids`
    # is popped at every segment end -- so under session mode the two diverge from
    # the second turn onward, and this function's arithmetic runs against a list
    # that restarted while the counter kept going. Concretely, with the session-
    # global counter every post-first segment took the `length -=` branch below
    # against a list that never shipped an initial chunk: the second chunk of
    # every turn went out with left_context_size=0 (an audible seam every chunk
    # until the ramp caught up), and a segment ending with fewer frames than
    # initial_codec_chunk_frames drove `length` negative, sliced past the end,
    # and dropped the segment's audio entirely (the torch.cat error upstream of
    # here). qwen3_tts.py already uses the segment-local counter for exactly
    # this reason.
    chunk_id = transfer_manager.ramp_chunk_count[request_id]
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        return None

    if configured_initial_chunk_size > 0:
        if chunk_id == 0:
            chunk_size_config = configured_initial_chunk_size
        else:
            adjusted = length - configured_initial_chunk_size
            if adjusted < 0:
                # chunk_id >= 1 guarantees >= initial frames shipped from THIS
                # list, so this cannot happen unless the counter and the list
                # drift out of scope again. Ship unadjusted rather than slicing
                # past the end and losing the audio -- and say so.
                logger.warning(
                    "[code2wav-chunk] chunk_id=%d but only %d frame(s) in the "
                    "segment list -- counter/list scope drift; shipping unadjusted",
                    chunk_id, length,
                )
            else:
                length = adjusted

    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    if is_finished and not append_codes and chunk_length == 0:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    if configured_initial_chunk_size > 0 and chunk_id == 1:
        left_context_size = configured_initial_chunk_size
        end_index = length + configured_initial_chunk_size
    else:
        left_context_size = max(0, min(length - context_length, left_context_size_config))
        end_index = min(length, left_context_size + context_length)

    codes = (
        torch.cat(transfer_manager.code_prompt_token_ids[request_id][-end_index:], dim=0).transpose(0, 1).reshape(-1)
    )

    # [live-vllm diagnosis] per-chunk emit stamp at the stage-1 SEND point --
    # the last unstamped hop of the chunk pipeline (emit -> vocode ->
    # orchestrator stamp -> client). CLOCK_MONOTONIC is host-wide, so this
    # aligns with [SCHED-STEP] mono across processes.
    if _LOG_CHUNK_EMIT:
        import time as _t
        logger.info(
            "[CHUNK-EMIT] rid=%s chunk_id=%d frames=%d mono=%.6f",
            request_id, chunk_id, context_length, _t.monotonic(),
        )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete talker codec output for the non-async connector path."""
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "talker2code2wav_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None
    code_predictor_codes = pooling_output.get("codes.audio")
    if code_predictor_codes is None:
        codes = pooling_output.get("codes")
        if isinstance(codes, dict):
            code_predictor_codes = codes.get("audio")
    if code_predictor_codes is None:
        logger.warning(
            "talker2code2wav_full_payload: missing codes.audio (keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    if not isinstance(code_predictor_codes, torch.Tensor):
        code_predictor_codes = torch.as_tensor(code_predictor_codes)
    if code_predictor_codes.numel() == 0:
        logger.warning(
            "talker2code2wav_full_payload: empty codes.audio for req=%s; consumer wait gate may hang.",
            rid,
        )
        return None

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
    raw_shape = tuple(code_predictor_codes.shape)
    code_predictor_codes, codec_stats = _extract_qwen3_full_payload_codec_rows(
        code_predictor_codes.to(torch.long),
        list(output_token_ids),
    )
    if code_predictor_codes.numel() == 0:
        logger.warning(
            "talker2code2wav_full_payload: no valid codec rows after filtering "
            "(raw_shape=%s output_ids_len=%d aligned_rows=%s valid_rows=%s) for req=%s; "
            "consumer wait gate may hang.",
            raw_shape,
            len(output_token_ids),
            codec_stats["aligned_rows"],
            codec_stats["valid_rows"],
            rid,
        )
        return None

    codec_codes = code_predictor_codes.transpose(0, 1).cpu().reshape(-1).tolist()
    logger.debug(
        "talker2code2wav_full_payload: raw_shape=%s output_ids_len=%s aligned_rows=%s "
        "valid_rows=%s placeholders=%s flattened_len=%s pad4196=%s bos4197=%s eos4198=%s",
        raw_shape,
        len(output_token_ids),
        codec_stats["aligned_rows"],
        codec_stats["valid_rows"],
        codec_stats["trailing_placeholder_count"],
        len(codec_codes),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_PAD_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_BOS_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_EOS_TOKEN_ID),
    )
    return {
        "codes": {"audio": codec_codes},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
