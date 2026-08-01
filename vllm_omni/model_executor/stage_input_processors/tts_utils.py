# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Shared TTS utility functions for speaker and language extraction.

These utilities are model-agnostic and can be used by any TTS model stage
processor (qwen3_omni, qwen2_5_omni, qwen3_tts, etc.).
"""

from typing import Any

# =============================================================================
# Speaker helpers
# =============================================================================


def extract_speaker_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract speaker from per-request runtime info dicts.

    Iterates through the list of per-request info dicts and returns the first
    non-empty speaker string found, normalized to lowercase.

    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        vt = info.get("speaker")
        if vt is None:
            continue
        if isinstance(vt, (list, tuple)) and len(vt) > 0:
            vt = vt[0]
        if isinstance(vt, str) and vt.strip():
            return vt.lower().strip()
        if vt is not None:
            return str(vt).lower().strip()
    return None


def extract_speaker_from_request(request: Any) -> str | None:
    """Extract speaker from a request's additional_information field.

    Reads from the structured ``additional_information.entries["speaker"]``
    field used by the engine serialization layer.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("speaker")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        val = list_data[0]
        return val.lower().strip() if isinstance(val, str) else str(val).lower().strip()
    return None


PREFILL_ONLY_KEY = "vllm_omni_prefill_only"


def request_is_prefill_only(request: Any) -> bool:
    """Is this append meant to be READ but not answered?

    A frames-on-arrival append exists to get the frames prefilled into stage 0's KV while
    the user is still talking. It must stop there: every payload that reaches the talker
    makes the model speak, and nobody asked it to.

    The marker rides on ``additional_information`` because that is the only channel that
    survives the trip -- the entrypoint and the stage processes are separate processes, so
    a flag in memory would not cross, and inferring it from the shape of the append (say,
    max_tokens == 1) would silently mislabel any future caller that happens to want one
    token.

    Reads both the structured engine form (``entries[key].list_data``) and a plain dict,
    because the prompt carries a dict and the engine request carries the structured form.
    """
    # The sampling-params channel first, because it is the one that actually arrives.
    #
    # additional_information was tried twice -- as a plain dict and as a real
    # AdditionalInformationPayload -- and neither reached the stage processes: the field is
    # populated by the CONNECTOR for stage-to-stage payloads, and nothing on the entrypoint's
    # streaming-update path transfers it from the prompt. Both attempts failed silently, with
    # the symptom (the model speaking unasked) three components away from the cause.
    #
    # SamplingParams.extra_args exists for exactly this, and it travels by construction:
    # StreamingInput carries per-chunk sampling params, async_omni puts them in
    # chunk_sampling_params_list[0], and they end up on request.sampling_params.
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None)
    if isinstance(extra, dict) and str(extra.get(PREFILL_ONLY_KEY, "")).strip().lower() in (
        "1", "true", "yes"
    ):
        return True

    info = getattr(request, "additional_information", None)
    if info is None:
        return False
    entries = getattr(info, "entries", None)
    if not isinstance(entries, dict):
        entries = info if isinstance(info, dict) else None
        if entries is None:
            return False
    entry = entries.get(PREFILL_ONLY_KEY)
    if entry is None:
        return False
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        entry = list_data[0]
    if isinstance(entry, list) and entry:
        entry = entry[0]
    return str(entry).strip().lower() in ("1", "true", "yes")


def extract_speaker_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract speaker from a prompt's additional_information dict.

    Used in non-async stage processors where the prompt is an
    OmniTokensPrompt / TextPrompt dict (or a list of them).

    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The speaker as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    speaker = add_info.get("speaker")
    if isinstance(speaker, list) and speaker:
        return speaker
    return None


# =============================================================================
# Language helpers
# =============================================================================


def extract_language_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract language from per-request runtime info dicts.
    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The language string (e.g. "Chinese", "English", "Auto"), or None.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        lang = info.get("language")
        if lang is None:
            continue
        if isinstance(lang, (list, tuple)) and len(lang) > 0:
            return lang
        if isinstance(lang, str) and lang.strip():
            return [lang.strip()]
    return None


def extract_language_from_request(request: Any) -> str | None:
    """Extract language from a request's additional_information field.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The language string, or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("language")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        return list_data
    return None


def extract_language_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract language from a prompt's additional_information dict.
    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The language as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    language = add_info.get("language")
    if isinstance(language, list) and language:
        return language
    return None
