from __future__ import annotations

from typing import TypedDict

NATIVE_PROMPT_TOKEN_IDS_KEY = "duplex_prompt_token_ids"
NATIVE_PROMPT_LEN_KEY = "duplex_prompt_len"
NATIVE_LAST_PROMPT_TOKEN_KEY = "duplex_last_prompt_token_id"
NATIVE_SEGMENT_TOKEN_IDS_KEY = "duplex_segment_token_ids"


class DuplexIntermediateBuffer(TypedDict, total=False):
    """Structured keys carried through ``model_intermediate_buffer``.

    The buffer remains a dict for scheduler and msgspec compatibility, but
    duplex-specific producers and consumers should use the helpers in this
    module instead of scattering nested string keys across serving, runner, and
    model code.
    """

    request_id: str
    global_request_id: list[str]
    prompt_token_ids: list[int]
    llm_output_token_ids: list[int]
    llm_output_text: list[str]
    stream_output: bool
    native_duplex: bool
    ids: dict[str, object]
    hidden_states: dict[str, object]
    codes: dict[str, object]
    meta: dict[str, object]
    duplex: dict[str, object]
    omni_payload: object
    waveform: object
    mel_spec: object


def build_duplex_intermediate_buffer(
    *,
    request_id: str,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    output_text: str | None = None,
    stream_output: bool = False,
    native_duplex: bool = False,
) -> DuplexIntermediateBuffer:
    buffer: DuplexIntermediateBuffer = {
        "global_request_id": [str(request_id)],
        "ids": {},
    }
    if prompt_token_ids is not None:
        prompt_ids = [int(token_id) for token_id in prompt_token_ids]
        buffer["prompt_token_ids"] = prompt_ids
        buffer["ids"]["prompt"] = prompt_ids
    if output_token_ids is not None:
        output_ids = [int(token_id) for token_id in output_token_ids]
        buffer["llm_output_token_ids"] = output_ids
        buffer["ids"]["output"] = output_ids
    if output_text is not None:
        buffer["llm_output_text"] = [output_text]
    if stream_output:
        buffer["stream_output"] = True
    if native_duplex:
        buffer["native_duplex"] = True
    return buffer


def set_ref_audio(buffer: dict[str, object], waveform: object, sample_rate_hz: int) -> None:
    buffer.setdefault("codes", {})["ref"] = waveform
    buffer.setdefault("meta", {})["ref_audio_sr"] = int(sample_rate_hz)


def set_ref_audio_handle(buffer: dict[str, object], handle: str) -> None:
    """Attach the stable session reference without copying its waveform."""
    buffer.setdefault("meta", {})["ref_audio_handle"] = str(handle)


def set_native_prompt_handoff_metadata(
    buffer: dict[str, object],
    *,
    prompt_len: int,
    last_prompt_token: int | None,
    current_segment_token_ids: list[int],
) -> None:
    """Store the O(segment) Thinker-to-Talker boundary contract.

    Native MiniCPM Talker conditioning needs the prompt boundary, not another
    copy of the complete cached prompt.  Keep the small current segment once in
    ``meta``; the actual Talker condition remains in ``ids.tts``.
    """
    meta = buffer.setdefault("meta", {})
    meta["prompt_len"] = max(0, int(prompt_len))
    if last_prompt_token is not None:
        meta["last_prompt_token"] = int(last_prompt_token)
    meta["current_segment_token_ids"] = [int(token_id) for token_id in current_segment_token_ids]


def set_tts_handoff(buffer: dict[str, object], token_ids: object | None, hidden_states: object | None) -> None:
    """Store the AR-to-TTS handoff used by the full-duplex stage bridge."""
    if token_ids is not None:
        buffer.setdefault("ids", {})["tts"] = token_ids
    if hidden_states is not None:
        buffer.setdefault("hidden_states", {})["tts"] = hidden_states


def get_tts_handoff(info: dict[str, object]) -> tuple[object | None, object | None]:
    """Read the canonical handoff, including the legacy flat aliases."""
    ids_info = info.get("ids")
    hidden_info = info.get("hidden_states")
    token_ids = ids_info.get("tts") if isinstance(ids_info, dict) else None
    hidden_states = hidden_info.get("tts") if isinstance(hidden_info, dict) else None
    return (
        info.get("tts_token_ids") if token_ids is None else token_ids,
        info.get("tts_hidden_states") if hidden_states is None else hidden_states,
    )


def get_stream_request_key(info: dict[str, object]) -> str:
    key = info.get("global_request_id") or info.get("request_id") or info.get("_omni_req_id")
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if key is None:
        raise ValueError(
            "Duplex streaming handoff requires a stable request id; "
            "expected global_request_id, request_id, or _omni_req_id."
        )
    return str(key)
