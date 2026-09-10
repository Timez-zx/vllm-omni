"""Audit non-P/D capacity using real input identities and semantic unit ends.

Use continuous_av.py --completion-mode nonpd and enable the existing
VLLM_OMNI_LOG_DUPLEX_CADENCE scheduler trace on the server. This is a traced
placement comparison, not a diagnostic-free production capacity certificate.
RTF starts at the CLIENT's first media arrival, never at engine admission.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

from benchmarks.minicpmo.analyze_rtf import _latency_summary, _request_session_id

ADMIT = re.compile(
    r"\[duplex_cadence\] stage=0 ADMIT req=(\S+) generation=(\d+) "
    r"admit_epoch=([\d.]+) prompt_tokens=(\S+) input_seq=(\S+) "
    r"input_origin=(\S+) input_unit_index=(\S+)"
)
DONE = re.compile(
    r"\[duplex_cadence\] stage=0 UNIT_DONE req=(\S+) generation=(\d+) done_epoch=([\d.]+)"
    r"(?: context_tokens=(\d+))?"
)
TTS_ADMIT = re.compile(r"\[duplex_cadence\] stage=1 ADMIT req=(\S+) generation=(\d+)")
TTS_DONE = re.compile(r"\[duplex_cadence\] stage=1 UNIT_DONE req=(\S+) generation=(\d+) done_epoch=([\d.]+)")


def input_backlog(timings, records):
    """Count earlier unfinished units at each subsequent complete-input arrival.

    A currently executing unit is normal; it becomes backlog when the next
    complete unit arrives. Do not confuse 200 ms audio chunks with 1 s units.
    No startup exclusion or grace period is applied.
    """
    pending = deque()
    events = []
    for timing in sorted(timings, key=lambda item: item["model_unit_ready_at_s"]):
        ready = timing["model_unit_ready_at_s"]
        while pending and pending[0][1] is not None and pending[0][1] <= ready:
            pending.popleft()
        if pending:
            known = [done for _, done in pending if done is not None]
            events.append(
                {
                    "input_unit_index": timing["input_unit_index"],
                    "ready_epoch": ready,
                    "earlier_unfinished_units": len(pending),
                    "oldest_unfinished_unit": pending[0][0],
                    "wait_for_earlier_units_ms": (max(known) - ready) * 1000 if len(known) == len(pending) else None,
                }
            )
        row = records.get(timing["input_unit_index"], {})
        pending.append((timing["input_unit_index"], row.get("done_epoch")))
    # A final unit has no observed successor. Give it the nominal 1 s unit
    # period, not an unlimited drain in which to hide end-of-stream backlog.
    last = max(timings, key=lambda item: item["model_unit_ready_at_s"]) if timings else None
    final_done = records.get(last["input_unit_index"], {}).get("done_epoch") if last else None
    final_late = bool(last) and (final_done is None or final_done > last["model_unit_ready_at_s"] + 1.0)
    return {
        "no_backlog": bool(timings) and not events and not final_late,
        "arrival_events_with_backlog": len(events),
        "max_earlier_unfinished_units": max((event["earlier_unfinished_units"] for event in events), default=0),
        "final_unit_exceeds_period": final_late,
        "first_event": events[0] if events else None,
        "worst_events": sorted(events, key=lambda event: event["earlier_unfinished_units"], reverse=True)[:5],
    }


def analyze(run, log_text, *, include_unit_records=False):
    admissions = {}
    ends = {}
    contexts = {}
    generation_metadata = {}
    pending = defaultdict(deque)
    pending_tts = defaultdict(deque)
    tts_admissions = defaultdict(set)
    tts_ends = defaultdict(dict)
    tts_queue_observations = defaultdict(list)
    for line in log_text.splitlines():
        if match := ADMIT.search(line):
            req, gen, timestamp, prompt, seq, origin, unit = match.groups()
            pending[req].append((req, int(gen)))
            if origin == "client" and unit.isdigit():
                admissions[(req, int(gen))] = {
                    "request_id": req,
                    "generation": int(gen),
                    "admit_epoch": float(timestamp),
                    "prompt_tokens": int(prompt) if prompt.isdigit() else None,
                    "input_unit_index": int(unit),
                }
        elif match := DONE.search(line):
            req, gen, timestamp, context = match.groups()
            # Streaming updates are consumed FIFO. The logged generation is
            # the newest ADMITTED update, which can be ahead of the one now
            # completing. Never overwrite two real completions with that ID.
            if not pending[req]:
                continue
            identity = pending[req].popleft()
            ends[identity] = float(timestamp)
            contexts[identity] = int(context) if context is not None else None
            detail = re.search(r"output_tokens=(\d+) max_tokens=\d+ last_token=(\S+)", line)
            if detail:
                generation_metadata[identity] = {
                    "decode_tokens": int(detail[1]),
                    "last_output_token": int(detail[2]) if detail[2].isdigit() else None,
                }
        elif match := TTS_ADMIT.search(line):
            req, gen = match.groups()
            timestamp = re.search(r"admit_epoch=([\d.]+)", line)
            if timestamp:
                tts_queue_observations[_request_session_id(req)].append((float(timestamp[1]), len(pending_tts[req])))
            pending_tts[req].append((req, int(gen)))
            tts_admissions[_request_session_id(req)].add((req, int(gen)))
        elif match := TTS_DONE.search(line):
            req, gen, timestamp = match.groups()
            if pending_tts[req]:
                tts_ends[_request_session_id(req)][pending_tts[req].popleft()] = float(timestamp)

    by_request = defaultdict(list)
    for identity, row in admissions.items():
        row["done_epoch"] = ends.get(identity)
        row["context_tokens"] = contexts.get(identity)
        row.update(generation_metadata.get(identity, {}))
        by_request[row["request_id"]].append(row)
    by_session = defaultdict(dict)
    for req, rows in by_request.items():
        # Require an actual completion for each FIFO input update. Do not
        # infer that a newer prompt completed an older missing model unit.
        for row in sorted(rows, key=lambda item: item["generation"], reverse=True):
            session = _request_session_id(req)
            index = row["input_unit_index"]
            if index not in by_session[session]:
                by_session[session][index] = row

    sessions = []
    all_latency = []
    for user in run["users"]:
        records = by_session[user["session_id"]]
        timings = user["input_unit_timings"]
        measured = []
        missing = []
        for timing in timings:
            index = timing["input_unit_index"]
            row = records.get(index)
            if row is None or row["done_epoch"] is None:
                missing.append(index)
                continue
            row = dict(row)
            row["input_ready_to_done_ms"] = (row["done_epoch"] - timing["model_unit_ready_at_s"]) * 1000
            row["ready_to_admit_ms"] = (row["admit_epoch"] - timing["model_unit_ready_at_s"]) * 1000
            row["admit_to_done_ms"] = (row["done_epoch"] - row["admit_epoch"]) * 1000
            measured.append(row)
            all_latency.append(row["input_ready_to_done_ms"])
        complete = bool(timings) and not missing and user["input_stream_complete"]
        last_done = max((row["done_epoch"] for row in measured), default=None)
        wall = last_done - timings[0]["first_media_arrival_at_s"] if last_done and timings else None
        budget = len(timings)
        rtf = budget / wall if complete and wall and wall > 0 else None
        tts_done = tts_ends[user["session_id"]]
        tts_missing = tts_admissions[user["session_id"]] - tts_done.keys()
        last_audio = user.get("audio", {}).get("last_received_epoch_s")
        pipeline_end = max([last_done or 0, last_audio or 0, *tts_done.values()])
        pipeline_wall = pipeline_end - timings[0]["first_media_arrival_at_s"] if timings else None
        pipeline_rtf = budget / pipeline_wall if complete and pipeline_wall and pipeline_wall > 0 else None
        minute_windows = defaultdict(list)
        for row in measured:
            minute_windows[(row["input_unit_index"] - timings[0]["input_unit_index"]) // 60].append(row)
        tts_queue = [
            count
            for timestamp, count in tts_queue_observations[user["session_id"]]
            if timings and timings[0]["model_unit_ready_at_s"] <= timestamp <= timings[-1]["model_unit_ready_at_s"]
        ]
        sessions.append(
            {
                "session_id": user["session_id"],
                "complete": complete,
                "completed_units": len(measured),
                "expected_units": budget,
                "missing_units": missing,
                "backlog": input_backlog(timings, records),
                "stream_rtf": rtf,
                "pipeline_rtf": pipeline_rtf,
                "talker_incomplete_units": len(tts_missing),
                "talker_queue": {
                    "audited_admissions": len(tts_queue),
                    "arrivals_with_previous_unfinished": sum(count > 0 for count in tts_queue),
                    "max_previous_unfinished": max(tts_queue, default=0),
                },
                "decode_tokens": sum(row.get("decode_tokens", 0) for row in measured),
                "listen_units": sum(row.get("last_output_token") == 151705 for row in measured),
                "minute_windows": [
                    {
                        "minute": minute + 1,
                        "units": len(rows),
                        "ready_to_done_ms": _latency_summary([row["input_ready_to_done_ms"] for row in rows]),
                        "ready_to_admit_ms": _latency_summary([row["ready_to_admit_ms"] for row in rows]),
                        "admit_to_done_ms": _latency_summary([row["admit_to_done_ms"] for row in rows]),
                        "decode_tokens": sum(row.get("decode_tokens", 0) for row in rows),
                        "max_context_tokens": max((row.get("context_tokens") or 0 for row in rows), default=0),
                    }
                    for minute, rows in sorted(minute_windows.items())
                ],
                "terminal_backlog_ms": (wall - budget) * 1000 if complete and wall else None,
                "input_ready_to_done_ms": _latency_summary([row["input_ready_to_done_ms"] for row in measured]),
                "ready_to_admit_ms": _latency_summary([row["ready_to_admit_ms"] for row in measured]),
                "admit_to_done_ms": _latency_summary([row["admit_to_done_ms"] for row in measured]),
                "max_context_tokens": max(
                    (row["context_tokens"] or row["prompt_tokens"] or 0 for row in measured), default=0
                ),
                "request_lineages": len({row["request_id"] for row in measured}),
                "context_rollovers": sum(
                    (previous.get("context_tokens") or 0) - (current.get("context_tokens") or 0) > 1024
                    for previous, current in zip(measured, measured[1:])
                    if current.get("context_tokens") is not None
                ),
                "slowest_units": sorted(measured, key=lambda row: row["input_ready_to_done_ms"], reverse=True)[:5],
            }
        )
        if include_unit_records:
            sessions[-1]["unit_records"] = measured
    rtfs = [item["stream_rtf"] for item in sessions if item["stream_rtf"] is not None]
    stage0_pass = bool(sessions) and len(rtfs) == len(sessions) and min(rtfs) >= 1
    audio_observed = int(run.get("audio_chunks", 0)) > 0
    pipeline_rtfs = [item["pipeline_rtf"] for item in sessions if item["pipeline_rtf"] is not None]
    no_skipped_units = bool(sessions) and all(item["complete"] for item in sessions)
    no_backlog = bool(sessions) and all(item["backlog"]["no_backlog"] for item in sessions)
    talker_complete = all(item["talker_incomplete_units"] == 0 for item in sessions)
    talker_backlog_events = sum(item["talker_queue"]["arrivals_with_previous_unfinished"] for item in sessions)
    return {
        "measurement_kind": "matched placement comparison with scheduler tracing",
        "definition": "input seconds / (last Thinker UNIT_DONE - first client media arrival)",
        "all_complete": all(item["complete"] for item in sessions),
        "stage0_rtf_pass": stage0_pass,
        "downstream_audio_observed": audio_observed,
        "audio_chunks": int(run.get("audio_chunks", 0)),
        "capacity_pass": no_backlog
        and no_skipped_units
        and talker_complete
        and audio_observed
        and not talker_backlog_events
        and not run.get("failed_users", 0),
        "capacity_criterion": (
            "No previous input unit may remain unfinished at the next complete-unit arrival; "
            "the final unit must finish within 1 s of readiness. No startup exclusion or grace period. "
            "All units and admitted Talker work must complete and audio must be observed. "
            "No same-session Talker queue backlog during the input streaming interval."
        ),
        "no_input_backlog": no_backlog,
        "talker_backlog_events": talker_backlog_events,
        "talker_queue_window": "complete input streaming interval; post-input automatic continuation excluded",
        "arrival_events_with_backlog": sum(item["backlog"]["arrival_events_with_backlog"] for item in sessions),
        "max_earlier_unfinished_units": max(
            (item["backlog"]["max_earlier_unfinished_units"] for item in sessions), default=0
        ),
        "pipeline_rtf_min": min(pipeline_rtfs, default=None),
        "pipeline_definition": (
            "input seconds / (last Thinker/Talker completion or received audio - first media arrival)"
        ),
        "pipeline_caveat": (
            "Diagnostic only: post-input automatic continuation may extend speech beyond the measured input stream"
        ),
        "no_skipped_units": no_skipped_units,
        "rtf_min": min(rtfs, default=None),
        "rtf_mean": sum(rtfs) / len(rtfs) if rtfs else None,
        "input_ready_to_done_ms": _latency_summary(all_latency),
        "sessions": sessions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-unit-records", action="store_true")
    args = parser.parse_args()
    result = analyze(
        json.loads(args.run_json.read_text()),
        args.server_log.read_text(errors="replace"),
        include_unit_records=args.include_unit_records,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
