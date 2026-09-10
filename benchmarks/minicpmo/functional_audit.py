"""Offline MiniCPM protocol/conservation audit, deliberately not a quality score.

No model/GPU imports and no instrumentation in the serving or sending path.
Missing evidence is UNKNOWN, never silently PASS. Repeated speech/PCM contents
are not duplicate-delivery proof: only transport identities can establish that.

Run after the load generator has closed and exported its quality captures:
    python benchmarks/minicpmo/functional_audit.py /path/to/run-directory
The current MiniCPM P/D protocol imports the prompt and computes one D prompt
token; this accounting invariant is not a generic prefill/decoding rule.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.minicpmo.capacity_audit import pd_input_backlog, sliding_window_engine_audit  # noqa: E402


def _result(errors, *, unknown=False, **evidence):
    return {"status": "fail" if errors else "unknown" if unknown else "pass", "errors": errors, **evidence}


def _physical_kv_result(engine_audit):
    # The engine audit deliberately requires both stages and window samples.
    # Before sliding starts there may be no samples at all: that is missing
    # evidence, not an observed residency violation and never a PASS.
    errors = []
    if isinstance(engine_audit, dict):
        if engine_audit.get("config_matches") is False:
            errors.append({"reason": "engine_window_config_mismatch"})
        if engine_audit.get("samples") and engine_audit.get("bounded_residency") is False:
            errors.append({"reason": "engine_window_residency_violation"})
    return _result(
        errors,
        unknown=not isinstance(engine_audit, dict) or engine_audit.get("valid") is not True,
        evidence=engine_audit,
    )


def _session_ids(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "session_id" and isinstance(item, str):
                yield item
            elif key == "session" and isinstance(item, dict) and isinstance(item.get("id"), str):
                yield item["id"]
            elif isinstance(item, (dict, list)):
                yield from _session_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _session_ids(item)


def _rid(event):
    return event.get("response_id") or (event.get("response") or {}).get("id")


def _physical_session(request_id):
    if not isinstance(request_id, str):
        return None
    match = re.match(r"^duplex-s\.([A-Za-z0-9_-]+)\.i\.", request_id or "")
    if not match:
        return None
    encoded = match.group(1)
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeError, binascii.Error):
        return None


def _response():
    return {
        "created": 0,
        "done": 0,
        "status": None,
        "audio_done": 0,
        "audio_bytes": 0,
        "audio_chunks": 0,
        "rates": set(),
        "audio_hash": hashlib.sha256(),
        "text": defaultdict(list),
        "text_done": defaultdict(list),
        "item_ids": set(),
        "final_output": None,
        "done_event_index": None,
        "alignment": defaultdict(lambda: {"seen": set(), "last": (0, 0)}),
        "alignment_marks": 0,
        "stream_alignment": [],
    }


def _check_audio_text_marks(marks, *, text_chars, audio_ms, state=None):
    """Check coordinates, not semantic alignment; cumulative repeats are legal."""
    if marks is None:
        return []
    if not isinstance(marks, list):
        return [{"reason": "invalid_audio_text_marks"}]
    errors, previous = [], (0, 0)
    for mark in marks:
        values = (mark.get("text_chars"), mark.get("audio_end_ms")) if isinstance(mark, dict) else (None, None)
        if any(not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v) for v in values):
            errors.append({"reason": "invalid_audio_text_mark"})
            continue
        chars, end_ms = values
        if chars < 0 or end_ms < 0 or chars > text_chars or (audio_ms is not None and end_ms > audio_ms):
            errors.append(
                {
                    "reason": "audio_text_mark_out_of_bounds",
                    "mark": mark,
                    "text_chars_bound": text_chars,
                    "audio_ms_bound": audio_ms,
                }
            )
        if chars < previous[0] or end_ms < previous[1]:
            errors.append({"reason": "audio_text_marks_nonmonotonic", "previous": previous, "mark": mark})
        previous = values
        if state is not None and values not in state["seen"]:
            if chars < state["last"][0] or end_ms < state["last"][1]:
                errors.append({"reason": "audio_text_marks_nonmonotonic", "previous": state["last"], "mark": mark})
            state["seen"].add(values)
            state["last"] = (max(chars, state["last"][0]), max(end_ms, state["last"][1]))
    return errors


def _played_text_prefix(text, cursor_ms, marks, duration_ms):
    """Independently apply the public piecewise-linear playback coordinates."""
    if cursor_ms <= 0:
        return ""
    points = [(0, 0)]
    for mark in marks or []:
        if isinstance(mark, dict):
            ms, chars = mark.get("audio_end_ms"), mark.get("text_chars")
            if isinstance(ms, int | float) and isinstance(chars, int | float) and ms >= 0 and chars >= 0:
                points.append((ms, min(len(text), chars)))
    points.sort()
    if duration_ms > points[-1][0]:
        points.append((duration_ms, len(text)))
    previous_ms, previous_chars = 0, 0
    for ms, chars in points[1:]:
        chars = max(previous_chars, chars)
        if cursor_ms <= ms:
            fraction = (cursor_ms - previous_ms) / max(1, ms - previous_ms)
            return text[: int(previous_chars + (chars - previous_chars) * fraction)].rstrip()
        previous_ms, previous_chars = ms, chars
    return text.rstrip()


def audit_session(user, events, exported=None, *, quality_dir=None):
    sid = user["session_id"]
    errors, ownership, protocol, export_errors = [], [], [], []
    sequence, identifiers, event_ids, physical = None, set(), set(), {}
    responses = defaultdict(_response)
    item_owners = {}
    truncations = {}
    missing_sequence = 0
    alignment_errors = []
    for index, row in enumerate(events):
        event = row["event"]
        kind = event.get("type", "")
        seq = event.get("server_event_seq")
        eid = event.get("event_id")
        if eid:
            if eid in event_ids:
                errors.append({"reason": "duplicate_event_id", "event_index": index, "event_id": eid})
            event_ids.add(eid)
        if isinstance(seq, int):
            if sequence is not None and seq != sequence + 1:
                errors.append({"reason": "nonconsecutive_server_event_sequence", "previous": sequence, "next": seq})
            sequence = seq
        elif kind not in {"session.created", "session.updated"}:
            missing_sequence += 1
        for other in set(_session_ids(event)) - {sid}:
            ownership.append({"reason": "wrong_session_in_event", "event_index": index, "actual": other})
        if kind == "error":
            protocol.append({"reason": "server_error_event", "event_index": index, "error": event.get("error")})
        if kind == "conversation.item.truncated":
            item = event.get("item_id")
            content_index = event.get("content_index", 0)
            cursor = event.get("audio_end_ms")
            if (
                not isinstance(item, str)
                or not isinstance(content_index, int)
                or content_index < 0
                or not isinstance(cursor, int | float)
                or isinstance(cursor, bool)
                or not math.isfinite(cursor)
                or cursor < 0
            ):
                protocol.append({"reason": "invalid_explicit_truncation", "event_index": index})
            else:
                truncations[(item, content_index)] = (cursor, index)
        witness = event.get("vllm_omni", {}).get("completion_witness")
        if kind == "response.model_unit.done" and isinstance(witness, dict):
            req = witness.get("engine_request_id")
            owner = _physical_session(req)
            if owner != sid:
                ownership.append(
                    {"reason": "physical_request_wrong_or_unreadable_session", "request": req, "owner": owner}
                )
            if req in physical:
                errors.append({"reason": "duplicate_physical_completion_event", "request": req})
            physical[req] = witness
        rid = _rid(event)
        if not rid:
            continue
        nested_rid = (event.get("response") or {}).get("id")
        if event.get("response_id") and nested_rid and event["response_id"] != nested_rid:
            protocol.append({"reason": "conflicting_response_ids", "event_index": index})
        response = responses[rid]
        identifiers.add(rid)
        if kind == "response.created":
            response["created"] += 1
        if kind in {"response.audio.delta", "response.audio_transcript.delta", "response.text.delta"}:
            if response["created"] != 1 or response["done"]:
                protocol.append(
                    {"reason": "delta_outside_response_lifecycle", "response_id": rid, "event_index": index}
                )
        if kind == "response.audio.delta":
            if response["audio_done"]:
                protocol.append({"reason": "audio_after_audio_done", "response_id": rid})
            try:
                pcm = base64.b64decode(event.get("delta") or event.get("audio") or "", validate=True)
                rate = event.get("sample_rate_hz", 24000)
                if not pcm or len(pcm) % 2 or not isinstance(rate, int) or rate <= 0:
                    raise ValueError("invalid PCM16 payload/rate")
                response["rates"].add(rate)
                response["audio_bytes"] += len(pcm)
                response["audio_chunks"] += 1
                response["audio_hash"].update(pcm)
            except (ValueError, TypeError, binascii.Error):
                protocol.append({"reason": "invalid_pcm16", "response_id": rid, "event_index": index})
            metadata = event.get("metadata") or {}
            marks = metadata.get("audio_text_marks", event.get("audio_text_marks"))
            channel = ("response.audio_transcript", event.get("output_index", 0), event.get("content_index", 0))
            # Chunk durations are rounded to integer ms upstream; allow at
            # most one rounding millisecond per received PCM chunk.
            audio_ms = (
                response["audio_bytes"] * 500 / next(iter(response["rates"])) + response["audio_chunks"]
                if len(response["rates"]) == 1
                else metadata.get("audio_duration_ms")
            )
            # Realtime serializes audio before its associated transcript
            # delta. Validate text bounds after collecting the response;
            # using the text observed at this event creates false failures.
            response["stream_alignment"].append((marks, channel, audio_ms, index))
            response["alignment_marks"] += len(marks) if isinstance(marks, list) else 0
        if kind in {"response.audio_transcript.delta", "response.text.delta"}:
            channel = (kind.removesuffix(".delta"), event.get("output_index", 0), event.get("content_index", 0))
            response["text"][channel].append(str(event.get("delta", "")))
        if kind in {"response.audio_transcript.done", "response.text.done"}:
            channel = (kind.removesuffix(".done"), event.get("output_index", 0), event.get("content_index", 0))
            response["text_done"][channel].append(str(event.get("transcript", event.get("text", ""))))
        if kind == "response.audio.done":
            response["audio_done"] += 1
        if kind == "response.done":
            response["done"] += 1
            response["status"] = event.get("response", {}).get("status")
            response["final_output"] = event.get("response", {}).get("output")
            response["done_event_index"] = index
        item_id = event.get("item_id") or event.get("item", {}).get("id")
        if item_id:
            response["item_ids"].add(item_id)
            previous_response = item_owners.setdefault(item_id, rid)
            if previous_response != rid:
                protocol.append({"reason": "item_shared_between_responses", "item_id": item_id})
    cancelled = []
    validated_truncations = set()
    summaries = []
    for rid, response in responses.items():
        if response["created"] != 1 or response["done"] != 1:
            protocol.append(
                {
                    "reason": "response_not_exactly_once_opened_and_closed",
                    "response_id": rid,
                    "created": response["created"],
                    "done": response["done"],
                }
            )
        if response["status"] not in {"completed", "cancelled"}:
            protocol.append(
                {"reason": "unsuccessful_or_missing_response_status", "response_id": rid, "status": response["status"]}
            )
        if response["status"] == "cancelled":
            cancelled.append(rid)
        if response["audio_chunks"] and response["audio_done"] != 1:
            protocol.append({"reason": "audio_not_exactly_once_closed", "response_id": rid})
        if len(response["rates"]) > 1:
            protocol.append({"reason": "mixed_pcm_sample_rates", "response_id": rid})
        for marks, channel, audio_ms, index in response["stream_alignment"]:
            chars = len("".join(response["text"].get(channel, [])))
            for error in _check_audio_text_marks(
                marks, text_chars=chars, audio_ms=audio_ms, state=response["alignment"][channel]
            ):
                alignment_errors.append({**error, "response_id": rid, "source": "stream", "event_index": index})
        for channel in response["text"].keys() | response["text_done"].keys():
            deltas = response["text"].get(channel, [])
            finals = response["text_done"].get(channel, [])
            # The protocol emits an empty transcript delta alongside pure
            # audio, but no transcript.done unless actual text was emitted.
            if channel[0] == "response.audio_transcript" and not "".join(deltas) and not finals:
                continue
            if len(finals) != 1 or "".join(deltas) != finals[0]:
                protocol.append(
                    {
                        "reason": "text_delta_done_mismatch",
                        "response_id": rid,
                        "channel": channel,
                        "delta_chars": len("".join(deltas)),
                        "done_count": len(finals),
                    }
                )
        for output_index, item in enumerate(response["final_output"] or []):
            if item.get("id") not in response["item_ids"]:
                protocol.append({"reason": "final_output_item_not_seen_in_response", "response_id": rid})
            for content_index, content in enumerate(item.get("content", [])):
                field = "transcript" if content.get("type") == "output_audio" else "text"
                channel_name = "response.audio_transcript" if field == "transcript" else "response.text"
                channel = (channel_name, output_index, content_index)
                generated_text = "".join(response["text"].get(channel, []))
                expected_text = generated_text
                truncation = truncations.get((item.get("id"), content_index))
                if truncation is not None and field == "transcript":
                    cursor, event_index = truncation
                    audio_ms = (
                        response["audio_bytes"] * 500 / next(iter(response["rates"])) + response["audio_chunks"]
                        if len(response["rates"]) == 1
                        else content.get("audio_duration_ms", 0)
                    )
                    if event_index < response["done_event_index"] and cursor <= audio_ms:
                        expected_text = _played_text_prefix(
                            generated_text,
                            cursor,
                            content.get("audio_text_marks", []),
                            content.get("audio_duration_ms") or audio_ms,
                        )
                        validated_truncations.add(rid)
                    else:
                        protocol.append({"reason": "invalid_explicit_truncation", "response_id": rid})
                if field in content and content[field] != expected_text:
                    protocol.append({"reason": "final_output_text_mismatch", "response_id": rid})
                if content.get("type") == "output_audio":
                    marks = content.get("audio_text_marks")
                    audio_ms = (
                        response["audio_bytes"] * 500 / next(iter(response["rates"])) + response["audio_chunks"]
                        if len(response["rates"]) == 1
                        else content.get("audio_duration_ms")
                    )
                    for error in _check_audio_text_marks(
                        # Marks describe the generated/sent audio timeline;
                        # an explicit close/cancel cursor may retain a shorter
                        # transcript in conversation history only.
                        marks, text_chars=len(generated_text), audio_ms=audio_ms
                    ):
                        alignment_errors.append({**error, "response_id": rid, "source": "response.done"})
                    response["alignment_marks"] += len(marks) if isinstance(marks, list) else 0
        summaries.append(
            {
                "response_id": rid,
                "audio_chunks": response["audio_chunks"],
                "audio_bytes": response["audio_bytes"],
                "status": response["status"],
                "text_chars": sum(len("".join(v)) for v in response["text"].values()),
            }
        )
    if exported is not None:
        if exported.get("metadata", {}).get("session_id") != sid:
            export_errors.append({"reason": "export_wrong_session_metadata"})
        export_rows = exported.get("responses", [])
        by_id = {row["response_id"]: row for row in export_rows}
        if len(by_id) != len(export_rows) or set(by_id) != set(responses):
            export_errors.append({"reason": "export_response_set_mismatch"})
        if exported.get("event_count") != len(events):
            export_errors.append({"reason": "export_event_count_mismatch"})
        for rid, response in responses.items():
            saved = by_id.get(rid, {})
            expected_text = "".join("".join(parts) for parts in response["text"].values())
            if saved.get("text") != expected_text:
                export_errors.append({"reason": "export_text_mismatch", "response_id": rid})
            if response["audio_bytes"] and quality_dir is not None:
                wav = quality_dir / Path(saved.get("wav", "")).name
                try:
                    with wave.open(str(wav), "rb") as audio:
                        raw = audio.readframes(audio.getnframes())
                        if (
                            audio.getnchannels() != 1
                            or audio.getsampwidth() != 2
                            or audio.getframerate() not in response["rates"]
                            or len(raw) != response["audio_bytes"]
                            or hashlib.sha256(raw).digest() != response["audio_hash"].digest()
                        ):
                            raise ValueError("WAV differs from received PCM")
                except (OSError, ValueError, wave.Error):
                    export_errors.append({"reason": "export_wav_pcm_mismatch", "response_id": rid})
    required_indices = {t["input_unit_index"] for t in user.get("input_unit_timings", [])}
    physical_real = [
        w
        for w in physical.values()
        if w.get("source") == "real_input" and w.get("input_unit_index") in required_indices
    ]
    count = Counter(w.get("input_unit_index") for w in physical_real)
    raw_expected = {r.get("request_id") for r in user.get("pd_completion_witness", {}).get("records", [])}
    physical_errors = []
    if events and (set(count) != required_indices or any(value != 1 for value in count.values())):
        physical_errors.append({"reason": "event_input_completion_not_one_to_one"})
    if events and {w["engine_request_id"] for w in physical_real} != raw_expected:
        physical_errors.append({"reason": "raw_event_and_run_completion_set_disagree"})
    return {
        "session_id": sid,
        "event_delivery": _result(
            errors,
            unknown=not events or missing_sequence > 0,
            events=len(events),
            missing_sequence_events=missing_sequence,
        ),
        "event_ownership": _result(ownership, unknown=not events),
        "response_lifecycle": _result(
            protocol,
            unknown=not responses or bool(set(cancelled) - validated_truncations),
            cancelled_responses=cancelled,
            explicit_truncations_validated=sorted(validated_truncations),
            responses=summaries,
        ),
        "quality_export": _result(export_errors, unknown=exported is None),
        "audio_text_alignment_coordinates": _result(
            alignment_errors,
            unknown=not responses or any(r["audio_chunks"] and not r["alignment_marks"] for r in responses.values()),
            marks_checked=sum(r["alignment_marks"] for r in responses.values()),
        ),
        "raw_input_completion": _result(physical_errors, unknown=not events),
        "speech_path_exercised": _result([], unknown=not any(r["audio_bytes"] for r in summaries)),
        "_owners": {
            "response_ids": sorted(identifiers),
            "event_ids": sorted(event_ids),
            "physical_ids": sorted(physical),
            "item_ids": sorted({item for response in responses.values() for item in response["item_ids"]}),
        },
    }


def audit_run(run, captures, exports=None, *, quality_dirs=None, engine_audit=None):
    exports, quality_dirs = exports or {}, quality_dirs or {}
    users = run.get("users", [])
    sids = [user.get("session_id") for user in users]
    unit_audit = pd_input_backlog(run)
    errors = []
    if not users or len(set(sids)) != len(sids) or not all(isinstance(s, str) and s for s in sids):
        errors.append({"reason": "missing_or_duplicate_session_ids"})
    for user in users:
        if user.get("units_sent") != len(user.get("input_unit_timings", [])):
            errors.append({"reason": "sent_count_timing_mismatch", "session_id": user["session_id"]})
    sessions = [
        audit_session(
            u,
            captures.get(u["session_id"], []),
            exports.get(u["session_id"]),
            quality_dir=quality_dirs.get(u["session_id"]),
        )
        for u in users
    ]
    owners, collisions = {}, []
    for session in sessions:
        for namespace, ids in session.pop("_owners").items():
            for identity in ids:
                key = (namespace, identity)
                prior = owners.setdefault(key, session["session_id"])
                if prior != session["session_id"]:
                    collisions.append(
                        {
                            "reason": "identity_shared_between_sessions",
                            "kind": namespace,
                            "identity": identity,
                            "sessions": [prior, session["session_id"]],
                        }
                    )
    incremental_errors, missing_stats, evidence = [], 0, []
    observed_D_rows = 0
    window = run.get("config", {}).get("expected_kv_window_tokens")
    for user in users:
        records = sorted(user.get("pd_completion_witness", {}).get("records", []), key=lambda r: r["input_unit_index"])
        previous = None
        for row in records:
            observed_D_rows += 1
            fields = [
                row.get(k)
                for k in (
                    "prompt_tokens",
                    "cached_tokens",
                    "local_cached_tokens",
                    "external_cached_tokens",
                    "computed_tokens",
                    "kv_transfer_selected_tokens",
                )
            ]
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in fields):
                missing_stats += 1
                continue
            prompt, cached, local, external, computed, transferred = fields
            if cached != local + external or prompt - cached != computed or computed != 1 or transferred != external:
                incremental_errors.append(
                    {
                        "reason": "D_cache_accounting_or_media_replay",
                        "session_id": user["session_id"],
                        "unit": row["input_unit_index"],
                        "fields": fields,
                    }
                )
            if previous is not None and prompt < previous:
                incremental_errors.append(
                    {
                        "reason": "logical_prompt_recycled",
                        "session_id": user["session_id"],
                        "unit": row["input_unit_index"],
                    }
                )
            previous = prompt
        max_prompt = max((r.get("prompt_tokens", 0) for r in records), default=0)
        post = sum(r.get("prompt_tokens", 0) >= window for r in records) if isinstance(window, int) else 0
        evidence.append(
            {
                "session_id": user["session_id"],
                "max_logical_prompt": max_prompt,
                "post_window_units": post,
                "window_replaced": bool(window and max_prompt >= 2 * window),
            }
        )
    checks = {
        "input_unit_conservation": _result(
            errors + [item for item in unit_audit["sessions"] if not item["valid"]], unknown=not unit_audit["valid"]
        ),
        "cross_session_identity_ownership": _result(collisions, unknown=len(users) < 2),
        "incremental_D_and_handoff": _result(
            incremental_errors, unknown=missing_stats > 0 or observed_D_rows == 0,
            missing_stats=missing_stats, observed_rows=observed_D_rows,
        ),
        "cross_window_coverage": _result(
            [], unknown=not evidence or not all(row["window_replaced"] for row in evidence), sessions=evidence
        ),
        "physical_KV_residency": _physical_kv_result(engine_audit),
        "distinct_user_phase_and_media": _result(
            [],
            unknown=len(users) < 2
            or len({u.get("phase_s") for u in users}) < 2
            or len({(u.get("media"), u.get("media_offset_units")) for u in users}) < 2,
        ),
    }
    observable = list(checks.values()) + [
        v for s in sessions for v in s.values() if isinstance(v, dict) and "status" in v
    ]
    return {
        "audit_version": 4,
        "scope": "received-protocol and cache-accounting contracts; not semantic quality or performance",
        "checks": checks,
        "sessions": sessions,
        "observable_contracts_pass": bool(observable) and all(c["status"] == "pass" for c in observable),
        "fail_count": sum(c["status"] == "fail" for c in observable),
        "unknown_count": sum(c["status"] == "unknown" for c in observable),
        "full_functionality_certified": False,
        "not_proven_by_events": [
            "Runtime cache salts and RNG seed/offset are not exported; routing does not prove cache/RNG isolation.",
            "P forward only computed new positions, KV numerical equality and every live block require engine probes.",
            "Contiguous server sequences prove received-event continuity, not an upstream chunk never emitted.",
            "Different event IDs with repeated content do not prove a bug: repeated speech/silence can be legitimate.",
            "Text/PCM response ownership and exported bytes are checked; spoken meaning and ASR/facts remain separate.",
        ],
        "complementary_CPU_tests": [
            "tests/core/sched/test_sliding_window_prefix_reuse.py::test_native_av_placeholders_are_salted_per_session_and_restart",
            "tests/worker/test_minicpm_pd_sampling_state.py::test_rng_restore_uses_request_identity_not_batch_row",
            "tests/worker/test_minicpm_pd_sampling_state.py::test_rng_wire_preserves_unsigned_seed_and_restores_once_across_units",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text())
    captures, exports, directories = {}, {}, {}
    for user in run.get("users", []):
        directory = args.run_dir / "quality" / f"user-{user['uid']}"
        if (directory / "events.json").exists():
            captures[user["session_id"]] = json.loads((directory / "events.json").read_text())
            exports[user["session_id"]] = json.loads((directory / "responses.json").read_text())
            directories[user["session_id"]] = directory
    engine = None
    if (args.run_dir / "server.log").exists() and (args.run_dir / "deploy.yaml").exists():
        import yaml

        engine = sliding_window_engine_audit(
            (args.run_dir / "server.log").read_text(),
            yaml.safe_load((args.run_dir / "deploy.yaml").read_text()),
            window_tokens=run["config"]["expected_kv_window_tokens"],
            users=len(run["users"]),
        )
    result = audit_run(run, captures, exports, quality_dirs=directories, engine_audit=engine)
    result["provenance"] = {
        "run_dir": str(args.run_dir.resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = args.out or args.run_dir / "functional-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(output),
                "fail_count": result["fail_count"],
                "unknown_count": result["unknown_count"],
                "observable_contracts_pass": result["observable_contracts_pass"],
            }
        )
    )


if __name__ == "__main__":
    main()
