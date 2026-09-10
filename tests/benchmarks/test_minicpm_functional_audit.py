"""Fault-injection tests for received-protocol checks, without a model/GPU."""

import base64
import copy
import wave

import pytest

from benchmarks.minicpmo.functional_audit import audit_run, audit_session


def _fixture(uid=0):
    sid = f"session-{uid}"
    salt = base64.urlsafe_b64encode(sid.encode()).decode().rstrip("=")
    records = []
    for index, prompt in enumerate((100, 2100), start=1):
        records.append(
            {
                "request_id": f"duplex-s.{salt}.i.1.e.0.r.stage0-{index}",
                "input_unit_index": index,
                "source": "real_input",
                "done_at_s": index + 0.1,
                "prompt_tokens": prompt,
                "cached_tokens": prompt - 1,
                "local_cached_tokens": prompt - 51,
                "external_cached_tokens": 50,
                "computed_tokens": 1,
                "kv_transfer_selected_tokens": 50,
            }
        )
    user = {
        "uid": uid,
        "session_id": sid,
        "units_sent": 2,
        "phase_s": 0.1 + uid * 0.5,
        "media": "same-video.mp4",
        "media_offset_units": uid * 5,
        "input_stream_complete": True,
        "input_unit_timings": [{"input_unit_index": i, "model_unit_ready_at_s": float(i)} for i in (1, 2)],
        "pd_completion_witness": {"records": records},
    }
    rid, item = f"response-{uid}", f"item-{uid}"
    pcm = b"\x01\x00" * 240
    events = [{"type": "session.created", "session": {"id": sid}}]
    events += [
        {
            "type": "response.model_unit.done",
            "vllm_omni": {
                "completion_witness": {
                    "engine_request_id": r["request_id"],
                    "input_unit_index": r["input_unit_index"],
                    "source": "real_input",
                    "session_id": sid,
                }
            },
        }
        for r in records
    ]
    events += [
        {"type": "response.created", "response": {"id": rid}},
        {"type": "response.audio_transcript.delta", "response_id": rid, "item_id": item, "delta": "你好"},
        {
            "type": "response.audio.delta",
            "response_id": rid,
            "item_id": item,
            "delta": base64.b64encode(pcm).decode(),
            "sample_rate_hz": 24000,
            "metadata": {"audio_text_marks": [{"text_chars": 2, "audio_end_ms": 10}]},
        },
        {"type": "response.audio.done", "response_id": rid, "item_id": item},
        {"type": "response.audio_transcript.done", "response_id": rid, "item_id": item, "transcript": "你好"},
        {
            "type": "response.done",
            "response_id": rid,
            "response": {
                "id": rid,
                "status": "completed",
                "output": [
                    {
                        "id": item,
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": "你好",
                                "audio_text_marks": [{"text_chars": 2, "audio_end_ms": 10}],
                            }
                        ],
                    }
                ],
            },
        },
    ]
    for seq, event in enumerate(events[1:], start=2):
        event.update(server_event_seq=seq, event_id=f"event-{uid}-{seq}")
    capture = [{"event": e, "relative_s": float(i)} for i, e in enumerate(events)]
    exported = {
        "metadata": {"session_id": sid},
        "event_count": len(events),
        "responses": [{"response_id": rid, "text": "你好", "wav": f"{rid}.wav"}],
    }
    return user, capture, exported, pcm


def _event(capture, kind):
    return next(row["event"] for row in capture if row["event"]["type"] == kind)


def _reason(result, check, reason):
    assert result[check]["status"] == "fail"
    assert reason in {error["reason"] for error in result[check]["errors"]}


def test_two_users_pass_observable_contracts_not_semantic_certification():
    pairs = [_fixture(i) for i in range(2)]
    run = {"config": {"expected_kv_window_tokens": 1000}, "users": [p[0] for p in pairs]}
    report = audit_run(
        run,
        {p[0]["session_id"]: p[1] for p in pairs},
        {p[0]["session_id"]: p[2] for p in pairs},
        engine_audit={"valid": True},
    )
    assert report["observable_contracts_pass"]
    assert report["fail_count"] == report["unknown_count"] == 0
    assert not report["full_functionality_certified"]
    assert any("RNG" in statement for statement in report["not_proven_by_events"])


def test_single_user_is_not_multiuser_evidence():
    user, capture, exported, _ = _fixture()
    result = audit_run({"users": [user]}, {user["session_id"]: capture}, {user["session_id"]: exported})
    for check in ("cross_session_identity_ownership", "distinct_user_phase_and_media", "physical_KV_residency"):
        assert result["checks"][check]["status"] == "unknown"


@pytest.mark.parametrize(
    "evidence,status,reason",
    [
        (None, "unknown", None),
        ({"valid": False}, "unknown", None),
        ({"valid": False, "config_matches": True, "bounded_residency": False, "samples": []}, "unknown", None),
        ({"valid": False, "config_matches": True, "bounded_residency": True, "samples": [{"stage": 0}]}, "unknown", None),
        ({"valid": False, "config_matches": False, "samples": []}, "fail", "engine_window_config_mismatch"),
        (
            {"valid": False, "config_matches": True, "bounded_residency": False, "samples": [{"stage": 0}]},
            "fail",
            "engine_window_residency_violation",
        ),
        ({"valid": True}, "pass", None),
    ],
)
def test_physical_kv_missing_evidence_is_unknown_not_a_runtime_fault(evidence, status, reason):
    user, capture, exported, _ = _fixture()
    report = audit_run(
        {"users": [user]}, {user["session_id"]: capture}, {user["session_id"]: exported}, engine_audit=evidence
    )
    result = report["checks"]["physical_KV_residency"]
    assert result["status"] == status
    assert result["evidence"] == evidence
    assert result["errors"] == ([] if reason is None else [{"reason": reason}])


@pytest.mark.parametrize("fault", ["duplicate_id", "sequence_gap", "duplicate_physical"])
def test_event_conservation_faults(fault):
    user, capture, exported, _ = _fixture()
    if fault == "duplicate_id":
        capture[2]["event"]["event_id"] = capture[1]["event"]["event_id"]
        reason = "duplicate_event_id"
    elif fault == "sequence_gap":
        capture[2]["event"]["server_event_seq"] += 1
        reason = "nonconsecutive_server_event_sequence"
    else:
        capture[2]["event"]["vllm_omni"] = copy.deepcopy(capture[1]["event"]["vllm_omni"])
        reason = "duplicate_physical_completion_event"
    _reason(audit_session(user, capture, exported), "event_delivery", reason)


def test_missing_raw_completion_and_mismatched_projection():
    user, capture, exported, _ = _fixture()
    del capture[1]
    result = audit_session(user, capture, exported)
    _reason(result, "raw_input_completion", "event_input_completion_not_one_to_one")
    _reason(result, "raw_input_completion", "raw_event_and_run_completion_set_disagree")


@pytest.mark.parametrize("fault", ["nested_session", "physical_session"])
def test_cross_session_delivery(fault):
    user, capture, exported, _ = _fixture()
    if fault == "nested_session":
        _event(capture, "response.created")["response"]["metadata"] = {"session_id": "wrong"}
        reason = "wrong_session_in_event"
    else:
        _event(capture, "response.model_unit.done")["vllm_omni"]["completion_witness"]["engine_request_id"] = _fixture(
            1
        )[0]["pd_completion_witness"]["records"][0]["request_id"]
        reason = "physical_request_wrong_or_unreadable_session"
    _reason(audit_session(user, capture, exported), "event_ownership", reason)


def test_identity_collision_across_two_connections():
    first, second = _fixture(), _fixture(1)
    second[1][1]["event"]["event_id"] = first[1][1]["event"]["event_id"]
    result = audit_run(
        {"users": [first[0], second[0]]}, {first[0]["session_id"]: first[1], second[0]["session_id"]: second[1]}
    )
    _reason(result["checks"], "cross_session_identity_ownership", "identity_shared_between_sessions")


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("pcm", "invalid_pcm16"),
        ("transcript", "text_delta_done_mismatch"),
        ("final_transcript", "final_output_text_mismatch"),
        ("late_audio", "audio_after_audio_done"),
        ("status", "unsuccessful_or_missing_response_status"),
        ("no_close", "response_not_exactly_once_opened_and_closed"),
    ],
)
def test_response_contract_faults(fault, reason):
    user, capture, exported, _ = _fixture()
    if fault == "pcm":
        _event(capture, "response.audio.delta")["delta"] = "not base64"
    elif fault == "transcript":
        _event(capture, "response.audio_transcript.done")["transcript"] = "another response"
    elif fault == "final_transcript":
        _event(capture, "response.done")["response"]["output"][0]["content"][0]["transcript"] = "wrong"
    elif fault == "late_audio":
        capture.append(copy.deepcopy(next(r for r in capture if r["event"]["type"] == "response.audio.delta")))
    elif fault == "status":
        _event(capture, "response.done")["response"]["status"] = "failed"
    else:
        capture.pop()
    _reason(audit_session(user, capture, exported), "response_lifecycle", reason)


def test_cancelled_is_not_successfully_exercised_audio_path():
    user, capture, exported, _ = _fixture()
    _event(capture, "response.done")["response"]["status"] = "cancelled"
    assert audit_session(user, capture, exported)["response_lifecycle"]["status"] == "unknown"


def _truncate_fixture(capture, cursor, *, item="item-0", index=0):
    event = {
        "type": "conversation.item.truncated",
        "item_id": item,
        "content_index": index,
        "audio_end_ms": cursor,
    }
    capture.insert(-1, {"event": event, "relative_s": 8.5})
    for seq, row in enumerate(capture[1:], start=2):
        row["event"].update(server_event_seq=seq, event_id=f"event-0-{seq}")
    _event(capture, "response.done")["response"]["status"] = "cancelled"


@pytest.mark.parametrize("cursor,expected", [(0, ""), (5, "你"), (10, "你好")])
def test_explicit_cancel_truncation_keeps_played_prefix_but_generated_marks(cursor, expected):
    user, capture, _, _ = _fixture()
    _truncate_fixture(capture, cursor)
    _event(capture, "response.done")["response"]["output"][0]["content"][0]["transcript"] = expected
    result = audit_session(user, capture)
    assert result["response_lifecycle"]["status"] == "pass"
    assert result["audio_text_alignment_coordinates"]["status"] == "pass"
    assert result["response_lifecycle"]["explicit_truncations_validated"] == ["response-0"]


@pytest.mark.parametrize("fault", ["wrong_item", "no_truncation", "wrong_prefix", "past_audio", "nonfinite"])
def test_cancel_is_not_an_excuse_for_arbitrary_text_loss(fault):
    user, capture, _, _ = _fixture()
    if fault != "no_truncation":
        _truncate_fixture(
            capture,
            100 if fault == "past_audio" else float("nan") if fault == "nonfinite" else 5,
            item="wrong-item" if fault == "wrong_item" else "item-0",
        )
    else:
        _event(capture, "response.done")["response"]["status"] = "cancelled"
    _event(capture, "response.done")["response"]["output"][0]["content"][0]["transcript"] = ""
    result = audit_session(user, capture)
    assert result["response_lifecycle"]["status"] == "fail"


def test_pure_audio_empty_transcript_delta_does_not_require_transcript_done():
    user, capture, _, _ = _fixture()
    _event(capture, "response.audio_transcript.delta")["delta"] = ""
    capture = [row for row in capture if row["event"]["type"] != "response.audio_transcript.done"]
    _event(capture, "response.done")["response"]["output"][0]["content"][0]["transcript"] = ""
    _event(capture, "response.audio.delta")["metadata"]["audio_text_marks"][0]["text_chars"] = 0
    _event(capture, "response.done")["response"]["output"][0]["content"][0]["audio_text_marks"][0]["text_chars"] = 0
    result = audit_session(user, capture)
    assert result["response_lifecycle"]["status"] == "pass"
    assert result["audio_text_alignment_coordinates"]["status"] == "pass"


def test_nonempty_transcript_still_requires_exactly_one_matching_done():
    user, capture, _, _ = _fixture()
    capture = [row for row in capture if row["event"]["type"] != "response.audio_transcript.done"]
    _reason(audit_session(user, capture), "response_lifecycle", "text_delta_done_mismatch")


def test_done_text_without_received_deltas_is_not_accepted():
    user, capture, _, _ = _fixture()
    capture = [row for row in capture if row["event"]["type"] != "response.audio_transcript.delta"]
    _reason(audit_session(user, capture), "response_lifecycle", "text_delta_done_mismatch")


def test_conflicting_nested_response_identity():
    user, capture, exported, _ = _fixture()
    _event(capture, "response.done")["response"]["id"] = "another-response"
    _reason(audit_session(user, capture, exported), "response_lifecycle", "conflicting_response_ids")


def test_repeated_pcm_content_with_distinct_event_ids_is_not_duplicate_proof():
    user, capture, _, _ = _fixture()
    copy_row = copy.deepcopy(capture[5])
    capture.insert(6, copy_row)
    for seq, row in enumerate(capture[1:], start=2):
        row["event"].update(server_event_seq=seq, event_id=f"distinct-{seq}")
    result = audit_session(user, capture)
    assert result["event_delivery"]["status"] == result["response_lifecycle"]["status"] == "pass"
    assert result["response_lifecycle"]["responses"][0]["audio_chunks"] == 2


def test_saved_wav_exactly_matches_received_pcm(tmp_path):
    user, capture, exported, pcm = _fixture()
    path = tmp_path / exported["responses"][0]["wav"]
    for payload, expected in [(pcm, "pass"), (b"\x00\x00" * 240, "fail")]:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(payload)
        assert audit_session(user, capture, exported, quality_dir=tmp_path)["quality_export"]["status"] == expected


@pytest.mark.parametrize("fault", ["replayed_media", "prompt_recycled", "missing_stats"])
def test_incremental_cache_accounting(fault):
    user, capture, _, _ = _fixture()
    row = user["pd_completion_witness"]["records"][1]
    if fault == "replayed_media":
        row["computed_tokens"] = 20
    elif fault == "prompt_recycled":
        row.update(prompt_tokens=80, cached_tokens=79, local_cached_tokens=29)
    else:
        del row["computed_tokens"]
    check = audit_run({"users": [user]}, {user["session_id"]: capture})["checks"]["incremental_D_and_handoff"]
    assert check["status"] == ("unknown" if fault == "missing_stats" else "fail")


def test_no_D_rows_cannot_certify_incremental_handoff():
    user, capture, _, _ = _fixture()
    user["pd_completion_witness"]["records"] = []
    check = audit_run({"users": [user]}, {user["session_id"]: capture})["checks"]["incremental_D_and_handoff"]
    assert check["status"] == "unknown"
    assert check["observed_rows"] == 0


def test_missing_capture_is_unknown_not_a_false_failure():
    user = _fixture()[0]
    result = audit_session(user, [])
    assert all(
        value["status"] == "unknown" for value in result.values() if isinstance(value, dict) and "status" in value
    )


@pytest.mark.parametrize("source", ["stream", "final"])
@pytest.mark.parametrize(
    "marks,reason",
    [
        (
            [{"text_chars": 2, "audio_end_ms": 5}, {"text_chars": 1, "audio_end_ms": 10}],
            "audio_text_marks_nonmonotonic",
        ),
        (
            [{"text_chars": 1, "audio_end_ms": 10}, {"text_chars": 2, "audio_end_ms": 5}],
            "audio_text_marks_nonmonotonic",
        ),
        ([{"text_chars": 3, "audio_end_ms": 10}], "audio_text_mark_out_of_bounds"),
        ([{"text_chars": 2, "audio_end_ms": 50}], "audio_text_mark_out_of_bounds"),
        ([{"text_chars": -1, "audio_end_ms": 10}], "audio_text_mark_out_of_bounds"),
        ([{"text_chars": 2, "audio_end_ms": -1}], "audio_text_mark_out_of_bounds"),
        ([{"text_chars": float("nan"), "audio_end_ms": 10}], "invalid_audio_text_mark"),
    ],
)
def test_alignment_coordinate_faults(source, marks, reason):
    user, capture, exported, _ = _fixture()
    target = (
        _event(capture, "response.audio.delta")["metadata"]
        if source == "stream"
        else _event(capture, "response.done")["response"]["output"][0]["content"][0]
    )
    target["audio_text_marks"] = marks
    _reason(audit_session(user, capture, exported), "audio_text_alignment_coordinates", reason)


def test_alignment_coordinates_cannot_restart_across_audio_deltas():
    user, capture, _, _ = _fixture()
    repeated = copy.deepcopy(capture[5])
    repeated["event"]["metadata"]["audio_text_marks"] = [{"text_chars": 1, "audio_end_ms": 5}]
    capture.insert(6, repeated)
    _reason(audit_session(user, capture), "audio_text_alignment_coordinates", "audio_text_marks_nonmonotonic")


def test_repeated_cumulative_alignment_is_legal_and_missing_marks_are_unknown():
    user, capture, _, _ = _fixture()
    repeated = copy.deepcopy(capture[5])
    capture.insert(6, repeated)
    assert audit_session(user, capture)["audio_text_alignment_coordinates"]["status"] == "pass"
    for row in capture:
        row["event"].get("metadata", {}).pop("audio_text_marks", None)
    _event(capture, "response.done")["response"]["output"][0]["content"][0].pop("audio_text_marks")
    assert audit_session(user, capture)["audio_text_alignment_coordinates"]["status"] == "unknown"


def test_audio_mark_may_precede_associated_transcript_delta_on_wire():
    user, capture, exported, _ = _fixture()
    capture[4], capture[5] = capture[5], capture[4]
    for seq, row in enumerate(capture[1:], start=2):
        row["event"].update(server_event_seq=seq)
    assert audit_session(user, capture, exported)["audio_text_alignment_coordinates"]["status"] == "pass"
