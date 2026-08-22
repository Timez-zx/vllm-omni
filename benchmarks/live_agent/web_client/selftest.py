#!/usr/bin/env python3
"""Check the web client against the real server code, before any GPU is involved.

Everything here fails silently at runtime if it is wrong, which is why it is
checked statically:

* a session-config field name the server does not have (the browser reports a
  healthy session while running on defaults);
* a message type the server does not accept (it replies with an error the page
  logs but the user reads as "the model ignored me");
* a wrong audio container or sample rate (noise, not an exception).

    python selftest.py
"""

from __future__ import annotations

import base64
import io
import pathlib
import re
import sys
import wave

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE / "app"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def js_object_literal(js: str, start_marker: str) -> dict:
    """Pull the session.config literal out of app.js without executing it.

    Only the keys are needed, and only their names are load-bearing, so a
    tolerant key scan beats dragging in a JS engine.
    """
    idx = js.index(start_marker)
    depth = 0
    out = []
    for ch in js[idx:]:
        out.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
    body = "".join(out)
    keys = re.findall(r"^\s{4,}([a-z_][a-z0-9_]*)\s*:", body, re.M)
    return {k: None for k in keys}


def main() -> int:
    print("=== web client self-test (run on the server) ===\n")
    app_js = (APP / "static" / "app.js").read_text(encoding="utf-8")

    # ---- 1. every session.config key must exist on the server model ----
    print("1. session.config keys against StreamingVideoSessionConfig")
    from vllm_omni.entrypoints.openai.video_stream_base import StreamingVideoSessionConfig

    sent = js_object_literal(app_js, "return {\n      type: 'session.config'")
    sent.pop("type", None)
    known = set(StreamingVideoSessionConfig.model_fields)
    unknown = sorted(set(sent) - known)
    check(
        "every key the page sends exists on the server model",
        not unknown,
        f"unknown: {unknown}" if unknown else f"{len(sent)} keys, all known",
    )

    wanted = {
        "context_window_trigger_tokens",
        "context_window_target_tokens",
        "context_window_compaction_headroom_tokens",
        "max_frame_width",
        "max_frame_height",
        "enable_frame_filter",
        "frame_filter_min_gap",
        "frame_filter_max_gap",
        "use_audio_in_video",
        "enable_video_arrival_prefill",
        "system_prompt",
        "modalities",
        "thinker_max_response_tokens",
    }
    missing = sorted(wanted - set(sent))
    check(
        "the optimisations we care about are actually requested",
        not missing,
        f"not sent: {missing}" if missing else f"all {len(wanted)} present",
    )

    # ---- 2. message types must be ones the server accepts ----
    print("\n2. message types against the server's dispatch")
    base_src = pathlib.Path(StreamingVideoSessionConfig.__module__.replace(".", "/") + ".py")
    server_src = (pathlib.Path(__file__).resolve().parents[3] / base_src).read_text(encoding="utf-8")
    accepted = set(re.findall(r'msg_type == "([a-z._]+)"', server_src))
    accepted.add("session.config")  # handled in _receive_config, not the dispatch chain
    # Scope to what actually crosses the socket. A loose scan also catches the
    # playback worklet's internal {type: 'samples'} port message, which is not a
    # protocol message -- and a test that reports that as a protocol error trains
    # you to ignore the test.
    client_sends = set(re.findall(r"socket\.send\(JSON\.stringify\(\{\s*type:\s*'([a-z._]+)'", app_js))
    if "type: 'session.config'" in app_js:
        client_sends.add("session.config")  # built by buildSessionConfig, sent on open
    bad = sorted(client_sends - accepted)
    check(
        "every type the page sends is dispatched by the server",
        not bad,
        f"not accepted: {bad}" if bad else f"sends {sorted(client_sends)}",
    )

    # Only what the server really sends. Its module docstring also lists the
    # CLIENT->server messages, and counting those as "emitted" made this test
    # demand the page handle session.config, which it sends rather than receives.
    body = server_src.split('"""', 2)[-1]
    emitted = set(re.findall(r'send_json\(\s*\{\s*"type":\s*"([a-z._]+)"', body))
    emitted |= set(re.findall(r'"type":\s*"(response\.[a-z._]+|session\.[a-z._]+)"', body))
    handled = set(re.findall(r"case '([a-z._]+)':", app_js))
    # response.* and session.* are the ones a user would notice going unhandled.
    notable = {e for e in emitted if e.startswith(("response.", "session."))}
    unhandled = sorted(notable - handled)
    check(
        "the page handles every response./session. event the server emits",
        not unhandled,
        f"unhandled: {unhandled}" if unhandled else f"{len(notable)} events",
    )

    # probe.py claims to send "deliberately the same values app.js sends, so this probe and
    # the page exercise one configuration rather than two". That claim drifted the moment a
    # new field was added to one of them, and the symptom was a feature that simply never
    # fired with no error anywhere. Compare the key sets rather than trusting the comment.
    probe_src = (HERE / "probe.py").read_text(encoding="utf-8")
    probe_keys = set(re.findall(r'^\s*"([a-z_]+)":', probe_src, re.M)) & known
    page_only = sorted(set(sent) - probe_keys)
    probe_only = sorted(probe_keys - set(sent))
    check(
        "probe.py and the page send the same session config",
        not page_only and not probe_only,
        f"page-only {page_only}, probe-only {probe_only}"
        if (page_only or probe_only)
        else f"{len(probe_keys)} keys in both",
    )

    # ---- 3. the audio contract, both directions ----
    print("\n3. audio contract")
    check(
        "uplink is raw PCM16 at 16 kHz",
        "INPUT_RATE = 16000" in app_js and "type: 'audio.chunk'" in app_js,
        "audio.chunk carries base64 PCM16",
    )

    import numpy as np

    from vllm_omni.entrypoints.openai.video_stream_base import OmniStreamingVideoHandler as H

    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)).astype(np.float32)
    wav_bytes = base64.b64decode(H._encode_audio_wav_b64(tone))
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, width, ch, n = w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getnframes()
    check("downlink is a parsable WAV", True, f"{rate} Hz, {ch}ch, {width * 8}-bit, {n} frames")
    check(
        "the page parses the WAV header rather than assuming 44 bytes",
        "decodeWav" in app_js and "'data'" in app_js and "'fmt '" in app_js,
    )
    check("16-bit is what the page supports", width == 2, f"server emits {width * 8}-bit")

    # The whole reply has to survive, not just its first granule. Streaming requests are
    # coerced to RequestOutputKind.DELTA, and under DELTA the output processor drains the
    # audio payload after every snapshot -- so each output carries a bare tensor holding
    # only the newest granule. The extractor used to read that as a transient first state
    # and answer None to every output after the first, delivering 0.22 s of a 13.66 s reply.
    # Both delta modes had the bug, which is why an A/B between them showed no difference.
    import torch

    granules = [7125, 48000, 48000, 32640]  # measured shape of one 151-char reply
    for mode in ("fast", "slow"):
        drained, emitted = 0, 0
        for n_samples in granules:
            b64, drained = getattr(H, f"_delta_{mode}")(torch.zeros(n_samples), drained)
            if b64:
                with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
                    emitted += w.getnframes()
        # One codec frame comes off the first granule as a CausalConv artifact.
        expected = sum(granules) - 1920
        check(
            f"delta mode {mode} delivers the whole reply",
            emitted == expected,
            f"{emitted}/{expected} samples ({emitted / 24000:.2f}s of {sum(granules) / 24000:.2f}s produced)",
        )

    # ---- 4. the turn trigger must exist, because the server never fires one ----
    print("\n4. the turn trigger")
    handler_src = (
        pathlib.Path(__file__).resolve().parents[3] / "vllm_omni/entrypoints/openai/serving_video_stream.py"
    ).read_text(encoding="utf-8")
    m = re.search(
        r"def should_trigger_turn\(self[^)]*\)[^:]*:\s*\n\s*(?:\"\"\".*?\"\"\"\s*\n\s*)?return (\w+)", handler_src, re.S
    )
    server_never_triggers = bool(m) and m.group(1) == "False"
    check(
        "the server still never auto-starts a turn",
        server_never_triggers,
        "should_trigger_turn returns False -- so the client MUST send video.query",
    )
    check("the page sends video.query", "type: 'video.query'" in app_js)
    check(
        "the page offers both auto-silence and push-to-talk",
        "updateSilenceDetector" in app_js and "push to talk released" in app_js,
    )

    # ---- 5. the page must not lie about who decides ----
    print("\n5. honesty of the UI")
    index_html = (APP / "index.html").read_text(encoding="utf-8")
    check(
        "the page tells the user turn-taking is client-side",
        "no listen/speak decision of its own" in index_html,
        "otherwise this gets confused with the natively duplex model",
    )

    print()
    if FAILURES:
        print(f"!! {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
