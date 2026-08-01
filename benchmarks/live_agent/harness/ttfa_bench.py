#!/usr/bin/env python3
"""TTFA benchmark for vllm-omni /v1/video/chat/stream, single- and multi-user.

Workload, per user session:

    [continuous video frames at `fps` for the whole session]
    [continuous audio chunks every `chunk_ms` for the whole session --
     SILENCE when the user is not speaking; never a gap in the stream,
     because a real microphone does not stop sending when nobody talks]

    repeat `reps` times:
        ... silence ...
        play the fixed question utterance      <- user speaks
        send video.query                       <- user has stopped speaking
        wait for the FIRST audio delta         <- this is the measurement
        drain the rest of the turn
        >= inter_turn_idle_s of silence         (upstream scheduler race
                                                 workaround, PR #2342)

Two anchors are recorded per repetition, because they answer different
questions:

    TTFA_from_speech_end    first audio  -  query sent
        the user-facing conversational gap; the SLO metric.

    TTFA_from_speech_start  first audio  -  utterance onset
        the diagnostic timeline. Only this one makes visible whether the
        engine did any work *while the user was still talking*, which is
        where deferred work shows up.

All timestamps are absolute wall clock (time.time()) so that this trace can be
aligned against the in-server stage-0 probe and the NVML sampler, which run in
other processes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pathlib
import sys
import time
import wave

import websockets

QWEN_OMNI_SPEECH_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def load_frames(d: pathlib.Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(d.glob("*.jpg"))]


def load_pcm(p: pathlib.Path) -> bytes:
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, \
            f"{p}: need mono/16k/16-bit"
        return w.readframes(w.getnframes())


class Trace:
    def __init__(self, path: pathlib.Path, meta: dict) -> None:
        self.f = path.open("w")
        self.rec("meta", **meta)

    def rec(self, k: str, **kw) -> None:
        self.f.write(json.dumps({"w": time.time(), "k": k, **kw}) + "\n")

    def close(self) -> None:
        self.f.flush()
        self.f.close()


async def one_user(uid: int, args, frames: list[bytes], utter: bytes,
                   barrier: asyncio.Barrier | None) -> dict:
    bytes_per_chunk = int(16000 * 2 * args.chunk_ms / 1000)
    silence = b"\x00\x00" * (bytes_per_chunk // 2)
    audio_period = args.chunk_ms / 1000.0
    frame_period = 1.0 / args.fps

    # --session only changes the FILENAME, never the uid. That matters: several
    # sequential single-user sessions must land in one directory without
    # overwriting each other, while ttfa_decompose keys per_user on the meta uid
    # and so still reports "1 user" rather than mistaking 4 sessions for 4
    # concurrent users. Keeping uid at 0 is what makes the concurrency label
    # honest; the session index is carried in meta for grouping.
    sfx = "" if args.session is None else f"_s{args.session}"
    outp = pathlib.Path(args.outdir) / f"ttfa_user{uid}{sfx}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tr = Trace(outp, {
        "uid": uid, "session": args.session,
        "users": args.users, "reps": args.reps, "fps": args.fps,
        "chunk_ms": args.chunk_ms, "num_frames": args.num_frames,
        "max_frames": args.max_frames, "evs": args.evs,
        "evs_threshold": args.evs_threshold, "query": args.query,
        "utterance_s": len(utter) / 32000.0, "frames_dir": args.frames,
        "think_s": args.think_s, "inter_turn_idle_s": args.inter_turn_idle_s,
    })

    # q_time gates delta attribution: a delta that arrives before this rep's
    # query belongs to the PREVIOUS rep (trailing output still in flight) and
    # must not be counted, or TTFA is measured against the wrong request.
    st = {"first_audio": None, "text_done": False, "audio_done": False,
          "rep": -1, "err": None, "done": False, "ntext": 0, "naudio": 0,
          "q_time": None, "stale_text": 0, "stale_audio": 0}
    results: list[dict] = []

    async with websockets.connect(args.uri, max_size=args.ws_max_mb * 1024 * 1024,
                                  ping_interval=None, open_timeout=60) as ws:
        cfg = {
            "type": "session.config", "model": args.model,
            "modalities": ["text", "audio"],
            "num_frames": args.num_frames, "max_frames": args.max_frames,
            "enable_frame_filter": args.evs,
            "frame_filter_threshold": args.evs_threshold,
            "use_audio_in_video": True,
        }
        # Server-side knobs, added in the fork. They live in session.config rather than in
        # server env vars so one deployment can serve clients with different policies, and
        # so an arm's configuration travels with its trace instead of with a shell script.
        if args.max_frame_width and args.max_frame_height:
            cfg["max_frame_width"] = args.max_frame_width
            cfg["max_frame_height"] = args.max_frame_height
        if args.frame_filter_min_gap:
            cfg["frame_filter_min_gap"] = args.frame_filter_min_gap
        if args.frame_filter_max_gap:
            cfg["frame_filter_max_gap"] = args.frame_filter_max_gap
        if args.session_scoped_request:
            cfg["session_scoped_request"] = True
        if args.session_talker_token_budget:
            cfg["session_talker_token_budget"] = args.session_talker_token_budget
        if args.session_roll_at_talker_tokens:
            cfg["session_roll_at_talker_tokens"] = args.session_roll_at_talker_tokens
        if args.session_roll_history_turns is not None:
            cfg["session_roll_history_turns"] = args.session_roll_history_turns
        if args.session_roll_settle_s is not None:
            cfg["session_roll_settle_s"] = args.session_roll_settle_s
        if args.system_prompt:
            cfg["system_prompt"] = args.system_prompt
        await ws.send(json.dumps(cfg))
        # Record the config VERBATIM. The comment above says an arm's configuration travels
        # with its trace, and until now it did not: this event carried only a timestamp, so a
        # run that silently came up in the wrong mode could not be told from one that came up
        # right. That happened -- a session-mode reproduction ran 30 turns entirely in per-turn
        # mode, and the only way to notice was counting log lines on the server. `meta` below
        # covers a hand-maintained subset of args and had already fallen behind every knob
        # added in the fork, so it is not a substitute.
        tr.rec("tx_config", cfg=cfg)

        async def receiver() -> None:
            try:
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except Exception:
                        continue
                    t = m.get("type")
                    r = st["rep"]
                    if t == "response.start":
                        tr.rec("rx_start", rep=r)
                    elif t == "response.text.delta":
                        if st["q_time"] is None or time.time() < st["q_time"]:
                            st["stale_text"] += 1
                            continue
                        st["ntext"] += 1
                        if st["ntext"] == 1:
                            tr.rec("rx_first_text", rep=r)
                        # Timestamping EVERY text delta is the only way to tell what
                        # the pipeline is doing between the first text and the first
                        # sound. The client-side `response.text.done` cannot answer
                        # it: the server emits that message from inside the audio
                        # branch, immediately before the first audio chunk (see
                        # video_stream_base.py:658), so first_text -> text_done is
                        # identical to first_text -> first_audio by construction and
                        # splitting on it yields a tautology.
                        if args.trace_deltas and st["ntext"] <= 512:
                            tr.rec("rx_text_delta", rep=r, i=st["ntext"],
                                   nchars=len(m.get("delta") or ""))
                    elif t == "response.text.done":
                        st["text_done"] = True
                        # Record the TEXT, not only its length, so a wrong answer can be
                        # spotted after the fact instead of only a differently-sized one.
                        #
                        # DO NOT USE EITHER FIELD AS A RESPONSE-LENGTH METRIC. The server
                        # emits response.text.done from inside its AUDIO branch
                        # (video_stream_base.py:658), so what arrives here is only the text
                        # generated BEFORE the first audio chunk -- the recorded text is
                        # visibly cut mid-word. It is therefore a proxy for the RAMP
                        # DURATION, roughly (thinker decode rate) x (talker+code2wav time):
                        # 863 ms of ramp gave 216 chars, 274 ms gave 72, 195 ms gave 73.
                        # Using it to check "did the fast arm just say less" compares the
                        # latency against a function of itself and manufactures a finding:
                        # it produced a confident, wrong claim that cutting frames to
                        # 640x352 flattened the model's output. The true length is stage-0
                        # num_tokens_out (52 vs 52 tokens for high motion at the two
                        # resolutions -- unchanged), cross-checked by stage-2
                        # audio_duration_s at ~3 tokens/second.
                        #
                        # Kept anyway because the partial text is still useful for spotting
                        # degenerate or repeated openings, and it costs a few hundred bytes
                        # against traces already ~1.6 MB per session.
                        #
                        # The galling part: recall_bench.py:127-130 already documents this
                        # exact trap and accumulates response.text.delta into answer_acc to
                        # get the real text. The fact was known and simply not carried over
                        # to this file, where `chars` then went unquestioned through several
                        # arms. TODO, deliberately NOT done mid-experiment because it would
                        # make two arms of a running comparison use different client code:
                        # accumulate the deltas here as well, and report that length.
                        _txt = m.get("text") or ""
                        tr.rec("rx_text_done", rep=r, chars=len(_txt), text=_txt)
                    elif t == "response.audio.delta":
                        nb = len(base64.b64decode(m.get("data") or ""))
                        if st["q_time"] is None or time.time() < st["q_time"]:
                            st["stale_audio"] += 1
                            tr.rec("rx_stale_audio", rep=r, nbytes=nb)
                            continue
                        st["naudio"] += 1
                        if st["first_audio"] is None:
                            st["first_audio"] = time.time()
                            tr.rec("rx_first_audio", rep=r, nbytes=nb)
                        if args.trace_deltas and st["naudio"] <= 32:
                            tr.rec("rx_audio_delta", rep=r, i=st["naudio"], nbytes=nb)
                    elif t == "response.audio.done":
                        st["audio_done"] = True
                        tr.rec("rx_audio_done", rep=r)
                    elif t == "session.rolled":
                        # The server retired the engine request and opened a fresh one seeded
                        # with recent text. Recorded because the turn that follows pays a cold
                        # prefill, and that cost is the whole price of an unbounded session --
                        # without this event in the trace it would show up as an unexplained
                        # TTFA outlier with nothing to attribute it to.
                        tr.rec("rx_session_rolled", rep=r, turn=m.get("turn"),
                               rolls=m.get("rolls"),
                               carried_messages=m.get("carried_messages"))
                    elif t == "session.done":
                        st["done"] = True
                        tr.rec("rx_session_done")
                        return
                    elif t == "error":
                        st["err"] = str(m.get("message"))[:300]
                        tr.rec("rx_error", rep=r, message=st["err"])
            except websockets.exceptions.ConnectionClosed as e:
                tr.rec("rx_closed", code=getattr(e, "code", None))

        rx = asyncio.create_task(receiver())
        loop = asyncio.get_running_loop()

        # Absolute deadline schedules. Frames and audio are open-loop: they keep
        # going on schedule no matter what the server is doing.
        next_frame = loop.time()
        next_audio = loop.time()
        frame_i = 0

        async def pump(until: float, utter_at: float | None) -> None:
            """Push frames + audio on schedule until `until`.

            If utter_at is set, the question utterance is played starting at
            that loop time, replacing silence; otherwise silence is sent.
            """
            nonlocal next_frame, next_audio, frame_i
            upos = 0
            spoke = False
            while True:
                now = loop.time()
                if now >= until and (utter_at is None or upos >= len(utter)):
                    return
                if now >= next_frame:
                    blob = frames[frame_i % len(frames)]
                    lag = now - next_frame
                    await ws.send(json.dumps({
                        "type": "video.frame",
                        "data": base64.b64encode(blob).decode()}))
                    if lag > frame_period:
                        tr.rec("frame_slip", rep=st["rep"], lag_s=lag)
                    frame_i += 1
                    next_frame += frame_period
                    if lag > frame_period:
                        next_frame += (lag // frame_period) * frame_period
                if now >= next_audio:
                    speaking = utter_at is not None and now >= utter_at and upos < len(utter)
                    if speaking:
                        if not spoke:
                            tr.rec("speech_start", rep=st["rep"])
                            spoke = True
                        chunk = utter[upos:upos + bytes_per_chunk]
                        upos += bytes_per_chunk
                        if len(chunk) < bytes_per_chunk:
                            chunk = chunk + silence[: bytes_per_chunk - len(chunk)]
                            tr.rec("speech_end", rep=st["rep"])
                    else:
                        chunk = silence
                    lag = now - next_audio
                    await ws.send(json.dumps({
                        "type": "audio.chunk",
                        "data": base64.b64encode(chunk).decode()}))
                    tr.rec("tx_audio", rep=st["rep"], speech=bool(speaking),
                           nbytes=len(chunk))
                    next_audio += audio_period
                    if lag > audio_period:
                        tr.rec("audio_slip", rep=st["rep"], lag_s=lag)
                        next_audio += (lag // audio_period) * audio_period
                nxt = min(next_frame, next_audio, until)
                await asyncio.sleep(max(0.0, nxt - loop.time()))
                if st["err"] or st["done"]:
                    return

        # A short lead-in so the frame buffer is not empty on the first query.
        await pump(loop.time() + args.lead_in_s, None)

        if barrier is not None:
            await barrier.wait()          # align users so contention overlaps

        for rep in range(args.reps):
            st.update(rep=rep, first_audio=None, text_done=False,
                      audio_done=False, ntext=0, naudio=0, q_time=None,
                      stale_text=0, stale_audio=0)
            tr.rec("rep_begin", rep=rep)

            # silence, then the utterance; query goes out the moment it ends
            t_utter = loop.time() + args.think_s
            await pump(t_utter + len(utter) / 32000.0, t_utter)

            st["q_time"] = time.time()
            tr.rec("tx_query", rep=rep, text=args.query, frames_sent=frame_i)
            await ws.send(json.dumps({"type": "video.query", "text": args.query}))

            # keep the media streams alive while waiting -- a real client does
            deadline = loop.time() + args.turn_timeout_s
            while loop.time() < deadline:
                if st["err"] or st["done"]:
                    break
                # require audio_done: text_done fires on the FIRST audio chunk
                # (see report 2.3), so it is not a completion signal and using
                # it lets the tail of this turn leak into the next one.
                if st["audio_done"]:
                    break
                await pump(min(loop.time() + 0.05, deadline), None)
            if st["first_audio"] is None:
                tr.rec("rep_timeout", rep=rep)
            tr.rec("rep_end", rep=rep, got_audio=st["first_audio"] is not None,
                   n_audio_deltas=st["naudio"], n_text_deltas=st["ntext"],
                   audio_done=st["audio_done"],
                   stale_audio=st["stale_audio"], stale_text=st["stale_text"])
            if st["err"] or st["done"]:
                break
            await pump(loop.time() + args.inter_turn_idle_s, None)

        await ws.send(json.dumps({"type": "video.done"}))
        tr.rec("tx_done")
        try:
            await asyncio.wait_for(rx, timeout=30)
        except asyncio.TimeoutError:
            rx.cancel()

    tr.rec("session_end", err=st["err"])
    tr.close()
    return {"uid": uid, "err": st["err"], "trace": str(outp)}


async def main_async(args) -> int:
    frames = load_frames(pathlib.Path(args.frames))
    if not frames:
        print(f"no frames in {args.frames}", file=sys.stderr)
        return 1
    utter = load_pcm(pathlib.Path(args.utterance))
    barrier = asyncio.Barrier(args.users) if args.users > 1 else None
    res = await asyncio.gather(
        *[one_user(i, args, frames, utter, barrier) for i in range(args.users)],
        return_exceptions=True)
    for r in res:
        if isinstance(r, Exception):
            print(f"user failed: {r!r}", file=sys.stderr)
        else:
            print(f"user {r['uid']}: err={r['err']} -> {r['trace']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8091/v1/video/chat/stream")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--frames", default="/data/zx/stimuli/frames/talkinghead")
    ap.add_argument("--utterance", default="/data/zx/stimuli/q_utterance.wav")
    ap.add_argument("--outdir", default="/data/zx/results/ttfa")
    ap.add_argument("--query", default="Describe what you can see in the camera right now.")
    ap.add_argument("--system-prompt", default=QWEN_OMNI_SPEECH_SYSTEM_PROMPT)
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--evs", action="store_true", default=True)
    ap.add_argument("--no-evs", dest="evs", action="store_false")
    ap.add_argument("--evs-threshold", type=float, default=0.95)
    # Fork-side session config. Previously these were server env vars (PA_*); moving them
    # into session.config is what makes an arm reproducible from its own trace.
    ap.add_argument("--max-frame-width", type=int, default=None,
                    help="server downscales arriving frames to fit this width")
    ap.add_argument("--max-frame-height", type=int, default=None)
    ap.add_argument("--frame-filter-min-gap", type=int, default=0)
    ap.add_argument("--frame-filter-max-gap", type=int, default=0)
    ap.add_argument("--session-scoped-request", action="store_true", default=False,
                    help="one resumable engine request for the whole session")
    ap.add_argument("--session-roll-at-talker-tokens", type=int, default=None,
                    help="roll the session (new engine request seeded with recent text) when "
                         "the talker's estimate reaches this, so the conversation continues "
                         "past stage-1 max_model_len")
    ap.add_argument("--session-roll-history-turns", type=int, default=None,
                    help="how many recent turns of text to carry across a roll")
    ap.add_argument("--session-roll-settle-s", type=float, default=None,
                    help="seconds to let the stages retire the old request before submitting "
                         "the first chunk of the rolled one")
    ap.add_argument("--session-talker-token-budget", type=int, default=None,
                    help="end the session cleanly at this many estimated talker tokens, "
                         "rather than letting it reach stage-1 max_model_len, which either "
                         "kills the stage or makes the scheduler skip the request forever")
    ap.add_argument("--think-s", type=float, default=2.0,
                    help="silence before the utterance in each repetition")
    ap.add_argument("--lead-in-s", type=float, default=4.0,
                    help="stream media this long before the first query")
    ap.add_argument("--inter-turn-idle-s", type=float, default=1.0)
    ap.add_argument("--turn-timeout-s", type=float, default=120.0)
    ap.add_argument("--session", type=int, default=None,
                    help="session index, used only to keep several sequential "
                         "runs from overwriting one another in the same outdir. "
                         "Does NOT change uid, so concurrency stays labelled "
                         "correctly downstream.")
    ap.add_argument("--trace-deltas", action="store_true",
                    help="timestamp every text/audio delta, not just the first. "
                         "Needed to split the first_text -> first_audio gap into "
                         "'thinker still generating' vs 'talker+code2wav'. Off by "
                         "default so existing runs stay byte-comparable.")
    ap.add_argument("--ws-max-mb", type=int, default=32)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
