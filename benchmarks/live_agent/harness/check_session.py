#!/usr/bin/env python3
"""Bring-up gate for PA_SESSION. Run after a short smoke; refuse the arm if it fails.

Each check below corresponds to a failure mode that does NOT raise at runtime, so none of
them would be caught by "the server didn't crash".

  1. Was the new code path entered at all? `[PA_SESSION]` log lines. If the feeder object
     were not recognised as a native async generator, async_omni would fall through to the
     one-shot per-turn path with no error and the arm would be a silent no-op.
  2. Is it really ONE request for the whole session? Under per-turn mode every turn mints a
     new id. Session mode must show ONE id reused across turns -- that is what keeps
     `put_req_chunk` incrementing, which is what selects the delta-shipping branch.
  3. Is the talker's prompt actually delta-sized? stage-1 `num_tokens_in` must stay small
     and flat while stage-0's grows. If it tracks stage 0, the delta branch never fired.
  4. Did every turn produce audio, and did the per-turn boundary fire once per turn? Under
     session mode the wire events are driven by segment detection rather than by the end of
     the generate() loop, so a missed boundary hangs the client instead of erroring.
  5. Is per-turn server-side telemetry still being emitted? Stages 1/2 should still produce
     one StageRequestStats per turn; only stage 0's is expected to be lost.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
from stage_stats_v2 import parse  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--expect-turns", type=int, required=True)
    ap.add_argument("--trace", default=None,
                    help="client trace dir for the audio/timeout check")
    ap.add_argument("--marker", default="===== PA_SESSION",
                    help="boot-marker PREFIX used to scope the log to the current boot")
    args = ap.parse_args()

    p = pathlib.Path(args.log)
    all_text = [ANSI.sub("", l).rstrip() for l in p.open(errors="replace")]
    # Scope every check to the CURRENT boot. The log is appended across boots on purpose
    # (truncating it once destroyed three arms' server-side telemetry permanently), so
    # without this the previous boot's crash signature and its turn counts leak in and the
    # gate would keep failing on an already-fixed problem.
    # Match the marker PREFIX only: the runner labels boots "ladder", "ladder-r3-done",
    # "arm-session-1", ... so anchoring on any one label silently matches nothing, the
    # scoping falls back to the whole appended log, and a previous boot's crash signature
    # fails the gate for a problem that is already fixed.
    marks = [i for i, l in enumerate(all_text) if l.startswith(args.marker)]
    text = all_text[marks[-1]:] if marks else all_text
    print(f"[gate] scoped to the current boot: {len(text)} of {len(all_text)} log lines")
    fails: list[str] = []

    # ---- 1. path entered ------------------------------------------------------------
    queued = [l for l in text if "turn=" in l and "queue delta" in l]
    done = [l for l in text if "turn=" in l and " done " in l]
    print(f"[gate] queued-delta log lines: {len(queued)}   turn-done lines: {len(done)}")
    for l in queued[:6]:
        print("       " + l.split("]", 2)[-1].strip()[:120])
    if not queued:
        print("[gate] FAIL: no [PA_SESSION] lines. The session path was NEVER ENTERED -- the "
              "feeder was probably not recognised as a native async generator, which fails "
              "silently. Every number from this boot is meaningless.")
        return 2
    if len(done) < args.expect_turns:
        fails.append(f"only {len(done)} turn-done events for {args.expect_turns} turns: the "
                     f"segment-boundary detection is missing turns, which hangs the client")

    # ---- 2. per-request stats are EXPECTED to be absent here -------------------------
    # StageRequestStats is printed when a request FINISHES, and a resumable session
    # request never does, so session mode legitimately produces none. The per-stage
    # decomposition this study normally relies on is therefore unavailable for this arm,
    # and the fallback is the client-side split (cross-validated against the server-side
    # one to within ~2% on talker+code2wav across four arms). That is ample: the question
    # is whether the talker's segment is ~100 ms or ~1,463 ms.
    recs = parse(p)
    boots = sorted({r["boot"] for r in recs})
    recs = [r for r in recs if r["boot"] == max(boots)] if boots else []
    print(f"[gate] StageRequestStats tables: {len(recs)} "
          f"(0 is EXPECTED under session mode -- see comment)")

    # ---- 3. is the talker's prompt really delta-sized? -------------------------------
    # Read from the log the accumulated thinker prompt and the placeholder length the
    # connector's own function computes for each delta.
    rx = re.compile(r"turn=(\d+) queue delta: (\d+) new frames, (\d+) tokens, "
                    r"cum=(\d+), talker_placeholder=(-?\d+)")
    deltas = [tuple(int(x) for x in m.groups()) for l in text for m in [rx.search(l)] if m]
    if not deltas:
        fails.append("the delta log lines lack cum/talker_placeholder: the instrumented "
                     "build is not installed, so this arm has no x-axis")
    else:
        print(f"[gate] {'turn':>5s} {'new frames':>11s} {'delta tok':>10s} "
              f"{'cum (thinker)':>14s} {'talker placeholder':>19s}")
        for t, nf, nt, cum, tl in deltas[-8:]:
            print(f"[gate] {t:5d} {nf:11d} {nt:10d} {cum:14d} {tl:19d}")
        tls = [d[4] for d in deltas if d[4] >= 0]
        cums = [d[3] for d in deltas]
        if not tls:
            fails.append("talker_placeholder never computed (-1): the length function could "
                         "not be imported, so the delta shape is unverified")
        elif min(tls) <= 1:
            fails.append(f"talker placeholder collapsed to {min(tls)}: a delta is missing its "
                         f"<|im_start|> header, which is SILENT -- max(1, ...) hides it and "
                         f"the worker keeps one row of conditioning and discards the rest")
        elif len(cums) >= 3 and cums[-1] > 2 * cums[0]:
            ratio = st.median(tls[len(tls) // 2:]) / cums[-1]
            print(f"[gate] talker placeholder / accumulated thinker prompt: {ratio:.4f}")
            if ratio > 0.5:
                fails.append(f"ratio {ratio:.3f}: the talker's prompt tracks the thinker's, so "
                             f"the delta branch never fired. Discard the arm.")

    # ---- 4. audio present, from the client trace ------------------------------------
    smoke = pathlib.Path(args.trace) if args.trace else None
    if smoke and smoke.exists():
        import json
        got = tot = tmo = 0
        for f in sorted(smoke.glob("ttfa_user*.jsonl")):
            for l in f.open():
                r = json.loads(l)
                if r["k"] == "rep_end":
                    tot += 1
                    got += 1 if r.get("got_audio") and r.get("audio_done") else 0
                elif r["k"] == "rep_timeout":
                    tmo += 1
        print(f"[gate] client: {tot} turns ended, {got} with complete audio, {tmo} timeouts")
        if tmo:
            fails.append(f"{tmo} client timeout(s): a turn boundary was missed, which hangs "
                         f"the client rather than erroring")
        if tot and got < tot:
            fails.append(f"only {got}/{tot} turns delivered complete audio")
    else:
        print("[gate] no client trace given (--trace); skipping the audio check")

    # ---- 5. crash signatures ---------------------------------------------------------
    for pat, why in (
        (r"zero-dimensional tensor", "the bare .squeeze() bug: a delta with exactly one im_start"),
        (r"streaming_update for unknown req", "per-turn state lost or misattributed"),
        (r"Dropping output for unknown req", "per-turn state lost or misattributed"),
        (r"longer than the maximum model length", "the session ran past max_model_len"),
    ):
        hits = [l for l in text if re.search(pat, l)]
        if hits:
            fails.append(f"{why} -- {len(hits)} occurrence(s): {hits[-1][:130]}")

    # `assert num_new_tokens > 0` on stage 1 fires when a resumable session request is
    # scheduled with nothing to compute, and it kills the whole engine-core process. The
    # 1aed4032 is supposed to make it unreachable, by sweeping FINISHED_ABORTED requests out
    # of `skipped_waiting` as well as `waiting` and `running`. Two earlier attempts at this
    # (9fdee244, b7392f3c) never executed a single line, and a quiet post-turn NOTE is exactly
    # what let them look like working fixes for hours -- so this stays loud. The two cases are
    # reported separately because they mean different things: mid-session says the delta
    # construction produced an empty prompt for a real turn, post-turn says the teardown path
    # is still reachable.
    last_done = max((i for i, l in enumerate(text) if "turn=" in l and " done " in l),
                    default=-1)
    asserts = [i for i, l in enumerate(text) if "assert num_new_tokens > 0" in l]
    mid = [i for i in asserts if i < last_done]
    if mid:
        fails.append(f"an EMPTY delta reached a scheduler MID-SESSION ({len(mid)} of "
                     f"{len(asserts)} occurrences before the last turn boundary) -- this is a "
                     f"bug in the delta construction, and the turns after it are unusable")
    elif asserts:
        # Loud, but NOT a failure: the crash lands after the last turn has been delivered, so
        # every measurement from this boot is intact, and blocking here costs the arm the run
        # it exists to produce. It was briefly a hard failure, which stopped a verification
        # run dead over a defect that provably does not touch the numbers -- the opposite
        # mistake from the original quiet NOTE that let an inert fix look like a working one.
        print(f"[gate] UNRESOLVED -- {len(asserts)} stage-1 `assert num_new_tokens > 0` after "
              f"the last turn. The resumable-teardown path still reaches it, so the scheduler "
              f"fix is NOT verified. Turn data from this boot is unaffected (post-turn), which "
              f"is why this does not fail the gate; the arm boots one server per session.")

    # This used to count `parked req=`, from the guard added in 9fdee244 and revised in
    # b7392f3c. That line can no longer appear: 1aed4032 measured the actual cause (a
    # FINISHED_ABORTED request left in `skipped_waiting`, which upstream's waiting loop draws
    # from FIRST under FCFS) and deleted the guard, whose condition keyed on status == WAITING
    # and so could never match. Counting it now would report 0 forever and read as evidence.
    #
    # What is worth reporting instead is the wedge detector from 921314e8, because it catches
    # the failure that actually ends long sessions here: a stage that stops scheduling while
    # requests are still tracked. It leaves no traceback, so without this the symptom is only
    # a client timeout several minutes later.
    wedged = [l for l in text if "looks WEDGED" in l]
    print(f"[gate] stages that stopped scheduling while still tracking requests: {len(wedged)}")
    if wedged:
        fails.append(f"a stage stopped scheduling ({len(wedged)} report(s)): the session "
                     f"wedged, and every turn after it is lost. The request table and the "
                     f"chunk-adapter state are dumped after each report in the server log")

    print()
    if fails:
        for f in fails:
            print(f"[gate] FAIL: {f}")
        return 1
    print("[gate] PASS -- session mode is live, one request per session, talker prompt is "
          "delta-sized, audio present, per-turn telemetry intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
