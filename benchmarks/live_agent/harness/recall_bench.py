#!/usr/bin/env python3
"""Does the model remember earlier turns of a camera session?

`ttfa_bench.py` measures how *fast* a turn is. It cannot measure whether the
model remembers anything, because its stimulus is one continuous scene: a model
with perfect memory and a model with none produce the same output.

This bench shows a *different* scene each turn (see `gen_recall_stimuli.py`),
then asks questions that are only answerable from memory:

    turns 0..N-1   scene i, spoken "Describe what you can see..."
    probe turn     scene N, spoken "What word was on the very first screen?"
    probe turn     scene N, spoken "List every word you have seen so far."

Scoring is exact substring match against the scene manifest, which is why the
stimuli use rendered uncommon words rather than natural scenes.

The `listall` probe is the useful number: it is *graded* (how many of N words
survive) rather than one pass/fail bit, so a partially-working memory is
distinguishable from a broken one.

Pacing is the same open-loop discipline as ttfa_bench: frames and audio go out
on absolute deadlines regardless of what the server is doing, and silence is
sent as real PCM zero chunks between utterances -- a real microphone does not
stop streaming when nobody is talking, and the server has no VAD.

    recall_bench.py --scenes /data/zx/stimuli/recall --outdir OUT [--describe-turns 8]
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

# Same speech-output system prompt as the latency bench, so the response path
# behaves identically and the two studies stay comparable.
QWEN_OMNI_SPEECH_SYSTEM_PROMPT = (
    "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen. "
    "You are a virtual voice assistant with no gender or age.\n"
    "You are communicating with the user.\n"
    "In user messages, [] contains the transcription of the user's speech, "
    "and () contains the description of non-speech sounds.\n"
    "Interact with users using short(no more than 50 words), brief, "
    "straightforward language, maintaining a natural tone.\n"
    "Never use formal phrasing, mechanical expressions, bureaucratic jargon, "
    "or auto-reply templates.\n"
    "Refer to the user as 'you' and yourself as 'I'."
)

# Appended with --memory-hint. Tests whether the "first screen" probe fails
# because the notes are unreachable, or merely because the model does not think
# to look at them. The notes are demonstrably usable -- the same history scores
# 8/8 on the list-everything probe -- so this isolates retrieval from storage.
MEMORY_HINT = (
    "\nEarlier turns of this conversation are recorded in the history. Depending "
    "on configuration this may include written notes (each beginning with SCENE: "
    "what the camera showed at that time, and ASKED: what you were asked) and/or "
    "the earlier camera frames and audio themselves. When the user asks about "
    "anything from earlier in the session, answer from that history. Do not answer "
    "from what the camera shows right now unless the question is about the present "
    "moment."
)


def load_frames(d: pathlib.Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(d.glob("*.jpg"))]


def load_pcm(p: pathlib.Path) -> bytes:
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, \
            f"{p}: need mono/16k/16-bit"
        return w.readframes(w.getnframes())


def load_manifest(p: pathlib.Path) -> list[tuple[int, str, str, str | None]]:
    """(index, word, shape, corner_detail). Detail is None for older 3-column sets."""
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        i, word, shape = parts[0], parts[1], parts[2]
        det = parts[3] if len(parts) > 3 and parts[3] != "-" else None
        rows.append((int(i), word, shape, det))
    return rows


class Trace:
    def __init__(self, path: pathlib.Path, meta: dict) -> None:
        self.f = path.open("w")
        self.rec("meta", **meta)

    def rec(self, k: str, **kw) -> None:
        self.f.write(json.dumps({"w": time.time(), "k": k, **kw}) + "\n")

    def close(self) -> None:
        self.f.flush()
        self.f.close()


async def run_session(args, plan: list[dict], scenes: dict[int, list[bytes]],
                      utters: dict[str, bytes]) -> list[dict]:
    bytes_per_chunk = int(16000 * 2 * args.chunk_ms / 1000)
    silence = b"\x00\x00" * (bytes_per_chunk // 2)
    audio_period = args.chunk_ms / 1000.0
    frame_period = 1.0 / args.fps

    outp = pathlib.Path(args.outdir) / "recall.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tr = Trace(outp, {
        "policy": args.policy_label, "fps": args.fps, "chunk_ms": args.chunk_ms,
        "num_frames": args.num_frames, "max_frames": args.max_frames,
        "evs": args.evs, "evs_threshold": args.evs_threshold,
        "turns": len(plan), "scenes_dir": args.scenes,
        "think_s": args.think_s, "inter_turn_idle_s": args.inter_turn_idle_s,
    })

    # answer_acc accumulates response.text.delta. The text.done payload CANNOT
    # be used for scoring: it fires on the FIRST audio chunk rather than when
    # text generation completes, so it is truncated mid-word ('the word "PI').
    # done_text is kept only to document that gap.
    st = {"rep": -1, "err": None, "done": False, "audio_done": False,
          "q_time": None, "answer_acc": "", "done_text": "",
          "first_audio": None, "n_text_deltas": 0}
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
        if args.system_prompt:
            cfg["system_prompt"] = args.system_prompt
        await ws.send(json.dumps(cfg))
        tr.rec("tx_config")

        async def receiver() -> None:
            try:
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except Exception:
                        continue
                    t = m.get("type")
                    r = st["rep"]
                    if t == "response.text.delta":
                        # Gate on q_time for the same reason as ttfa_bench: a
                        # message that predates this turn's query belongs to the
                        # previous turn's trailing output.
                        if st["q_time"] is None or time.time() < st["q_time"]:
                            continue
                        st["answer_acc"] += m.get("delta") or ""
                        st["n_text_deltas"] += 1
                    elif t == "response.text.done":
                        if st["q_time"] is None or time.time() < st["q_time"]:
                            tr.rec("rx_stale_text_done", rep=r)
                            continue
                        st["done_text"] = m.get("text") or ""
                        tr.rec("rx_text_done", rep=r, chars=len(st["done_text"]),
                               text=st["done_text"][:1000])
                    elif t == "response.audio.delta":
                        if st["q_time"] is None or time.time() < st["q_time"]:
                            continue
                        if st["first_audio"] is None:
                            st["first_audio"] = time.time()
                            tr.rec("rx_first_audio", rep=r)
                    elif t == "response.audio.done":
                        st["audio_done"] = True
                        tr.rec("rx_audio_done", rep=r)
                    elif t == "session.done":
                        st["done"] = True
                        tr.rec("rx_session_done")
                        return
                    elif t == "error":
                        st["err"] = str(m.get("message"))[:400]
                        tr.rec("rx_error", rep=r, message=st["err"])
            except websockets.exceptions.ConnectionClosed as e:
                tr.rec("rx_closed", code=getattr(e, "code", None))

        rx = asyncio.create_task(receiver())
        loop = asyncio.get_running_loop()

        next_frame = loop.time()
        next_audio = loop.time()
        cur_scene = plan[0]["scene"]
        frame_i = 0
        frames_this_turn = 0

        async def pump(until: float, utter: bytes | None, utter_at: float | None) -> None:
            nonlocal next_frame, next_audio, frame_i, frames_this_turn
            upos = 0
            spoke = False
            while True:
                now = loop.time()
                if now >= until and (utter_at is None or upos >= len(utter or b"")):
                    return
                if now >= next_frame:
                    buf = scenes[cur_scene]
                    await ws.send(json.dumps({
                        "type": "video.frame",
                        "data": base64.b64encode(buf[frame_i % len(buf)]).decode()}))
                    frame_i += 1
                    frames_this_turn += 1
                    lag = now - next_frame
                    next_frame += frame_period
                    if lag > frame_period:
                        tr.rec("frame_slip", rep=st["rep"], lag_s=lag)
                        next_frame += (lag // frame_period) * frame_period
                if now >= next_audio:
                    speaking = (utter is not None and utter_at is not None
                                and now >= utter_at and upos < len(utter))
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
                    await ws.send(json.dumps({
                        "type": "audio.chunk",
                        "data": base64.b64encode(chunk).decode()}))
                    lag = now - next_audio
                    next_audio += audio_period
                    if lag > audio_period:
                        next_audio += (lag // audio_period) * audio_period
                nxt = min(next_frame, next_audio, until)
                await asyncio.sleep(max(0.0, nxt - loop.time()))
                if st["err"] or st["done"]:
                    return

        # lead-in so the buffer is not empty on the first query
        await pump(loop.time() + args.lead_in_s, None, None)

        for rep, step in enumerate(plan):
            cur_scene = step["scene"]
            frames_this_turn = 0
            st.update(rep=rep, answer_acc="", done_text="", audio_done=False,
                      q_time=None, first_audio=None, n_text_deltas=0)
            utter = utters[step["utter"]]
            tr.rec("rep_begin", rep=rep, scene=step["scene"],
                   kind=step["kind"], utter=step["utter"])

            t_utter = loop.time() + args.think_s
            await pump(t_utter + len(utter) / 32000.0, utter, t_utter)

            st["q_time"] = time.time()
            tr.rec("tx_query", rep=rep, frames_this_turn=frames_this_turn,
                   frames_total=frame_i)
            # No text in video.query: the question is in the audio, which is the
            # whole point -- a real camera user speaks, they do not type.
            await ws.send(json.dumps({"type": "video.query", "text": ""}))

            deadline = loop.time() + args.turn_timeout_s
            while loop.time() < deadline:
                if st["err"] or st["done"]:
                    break
                if st["audio_done"]:
                    break
                await pump(min(loop.time() + 0.05, deadline), None, None)

            results.append({
                "turn": rep, "scene": step["scene"], "kind": step["kind"],
                "shown_word": step.get("shown_word"),
                "shown_shape": step.get("shown_shape"),
                "answer": st["answer_acc"],          # full text, from deltas
                "done_text": st["done_text"],        # truncated at first audio
                "n_text_deltas": st["n_text_deltas"],
                "got_audio": st["first_audio"] is not None,
                "frames_this_turn": frames_this_turn,
                "err": st["err"],
            })
            tr.rec("rep_end", rep=rep, chars=len(st["answer_acc"]),
                   done_chars=len(st["done_text"]),
                   got_audio=st["first_audio"] is not None)
            print(f"  turn {rep:2d} scene={step['scene']:2d} "
                  f"({step.get('shown_word') or '-':9}) {step['kind']:11} "
                  f"-> {st['answer_acc'][:110]!r}", flush=True)
            if st["err"]:
                print(f"    ERROR: {st['err']}", file=sys.stderr)
                break
            await pump(loop.time() + args.inter_turn_idle_s, None, None)

        await ws.send(json.dumps({"type": "video.done"}))
        tr.rec("tx_done")
        try:
            await asyncio.wait_for(rx, timeout=30)
        except asyncio.TimeoutError:
            rx.cancel()

    tr.rec("session_end", err=st["err"])
    tr.close()
    return results


def score(results: list[dict], manifest: list[tuple[int, str, str, str | None]],
          describe_turns: int) -> dict:
    """Score the probes against ground truth."""
    words = {i: w for i, w, _, _ in manifest}
    details = {i: d for i, _, _, d in manifest}
    out: dict = {}

    # --- probe 1: the first word --------------------------------------------
    first_word = words[0]
    p1 = next((r for r in results if r["kind"] == "probe_first"), None)
    if p1:
        ans = (p1["answer"] or "").upper()
        out["probe_first"] = {
            "expected": first_word,
            "answer": p1["answer"],
            "correct": first_word in ans,
        }

    # --- probe 2: list everything -------------------------------------------
    p2 = next((r for r in results if r["kind"] == "probe_listall"), None)
    if p2:
        ans = (p2["answer"] or "").upper()
        # words the model had to *remember*: shown in the describe turns and no
        # longer in the rolling frame buffer by the time of the probe
        must_remember = [words[i] for i in range(describe_turns)]
        # currently-visible word needs no memory; report it separately
        visible = words.get(p2["scene"])
        found = [w for w in must_remember if w in ans]
        out["probe_listall"] = {
            "must_remember": must_remember,
            "recalled": found,
            "n_recalled": len(found),
            "n_total": len(must_remember),
            "frac": len(found) / len(must_remember) if must_remember else 0.0,
            "visible_word_mentioned": bool(visible and visible in ans),
            "answer": p2["answer"],
        }

    # --- probe 3: the corner detail ------------------------------------------
    # This is the probe that separates "keeps the pixels" from "keeps a note".
    # The note is one sentence about the salient objects, so it will not contain a
    # tiny corner number; a policy that still has the frame can read it, a policy
    # that only has the note cannot. Scored as an exact digit-string match, and
    # also checked against every OTHER scene's number so a lucky guess or a
    # confusion with a later scene is visible rather than counted as a hit.
    first_detail = details.get(0)
    p3 = next((r for r in results if r["kind"] == "probe_detail"), None)
    if p3 and first_detail:
        ans = p3["answer"] or ""
        others = [d for i, d in details.items() if i != 0 and d]
        out["probe_detail"] = {
            "expected": first_detail,
            "answer": p3["answer"],
            "correct": first_detail in ans,
            "other_scene_numbers_mentioned": [d for d in others if d in ans],
        }

    # --- sanity: did the model actually perceive each scene when shown? ------
    # What counts as "perceived" depends on what was asked. Under q_shape the
    # answer names the shape and deliberately does NOT name the word, so
    # checking for the word here would report 0% and look like a broken model
    # when the model is answering correctly.
    read_ok = []
    leaked_word = []
    for r in results:
        if r["kind"] != "describe":
            continue
        ans = (r["answer"] or "").upper()
        target = (r.get("shown_shape") or r.get("shown_word") or "").upper()
        if target:
            read_ok.append(target in ans)
        if r.get("shown_word"):
            leaked_word.append(r["shown_word"] in ans)
    out["scene_read_rate"] = {
        "checked": "shape" if any(r.get("shown_shape") for r in results) else "word",
        "n_ok": sum(read_ok), "n": len(read_ok),
        "frac": (sum(read_ok) / len(read_ok)) if read_ok else 0.0,
    }
    # If the describe answers already contain the word, then history text alone
    # carries it and the probe cannot separate full_text from text_memory. This
    # number says whether the experiment is actually isolating the note.
    out["word_leaked_into_answers"] = {
        "n": sum(leaked_word), "of": len(leaked_word),
        "frac": (sum(leaked_word) / len(leaked_word)) if leaked_word else 0.0,
    }
    return out


async def main_async(args) -> int:
    sdir = pathlib.Path(args.scenes)
    manifest = load_manifest(sdir / "manifest.txt")
    scene_dirs = sorted(d for d in sdir.iterdir() if d.is_dir() and d.name.startswith("scene_"))
    if len(scene_dirs) < args.describe_turns + 1:
        print(f"need >= {args.describe_turns + 1} scenes, found {len(scene_dirs)}",
              file=sys.stderr)
        return 1
    scenes = {i: load_frames(d) for i, d in enumerate(scene_dirs)}
    for i, f in scenes.items():
        if not f:
            print(f"scene {i} has no frames", file=sys.stderr)
            return 1
    has_detail = any(d for _, _, _, d in manifest)
    probes = [("probe_first", "q_first"), ("probe_listall", "q_listall")]
    if has_detail and (sdir / "q_detail.wav").exists():
        probes.append(("probe_detail", "q_detail"))
    names = [u for _, u in probes] + [args.describe_utter]
    utters = {n: load_pcm(sdir / f"{n}.wav") for n in dict.fromkeys(names)}

    words = {i: w for i, w, _, _ in manifest}
    shapes = {i: s for i, _, s, _ in manifest}
    plan: list[dict] = []
    for i in range(args.describe_turns):
        plan.append({"scene": i, "kind": "describe", "utter": args.describe_utter,
                     "shown_word": words[i], "shown_shape": shapes[i]})
    probe_scene = args.describe_turns
    for kind, utter in probes:
        plan.append({"scene": probe_scene, "kind": kind, "utter": utter,
                     "shown_word": words[probe_scene],
                     "shown_shape": shapes[probe_scene]})

    print(f"=== recall session: {len(plan)} turns, policy_label={args.policy_label} ===")
    results = await run_session(args, plan, scenes, utters)

    sc = score(results, manifest, args.describe_turns)
    outp = pathlib.Path(args.outdir) / "recall_score.json"
    outp.write_text(json.dumps({"policy": args.policy_label,
                                "results": results, "score": sc}, indent=2))

    print(f"\n=== score ({args.policy_label}) ===")
    rr = sc["scene_read_rate"]
    print(f"  scene read rate (sanity):  {rr['n_ok']}/{rr['n']} = {rr['frac']:.0%}")
    if "probe_first" in sc:
        p = sc["probe_first"]
        print(f"  probe 'first word'?        expected {p['expected']}  -> "
              f"{'CORRECT' if p['correct'] else 'WRONG'}")
        print(f"      answer: {p['answer'][:200]!r}")
    if "probe_listall" in sc:
        p = sc["probe_listall"]
        print(f"  probe 'list all':          {p['n_recalled']}/{p['n_total']} "
              f"= {p['frac']:.0%} recalled")
        print(f"      recalled: {p['recalled']}")
        print(f"      answer:   {p['answer'][:300]!r}")
    if "probe_detail" in sc:
        p = sc["probe_detail"]
        print(f"  probe 'corner number':     expected {p['expected']}  -> "
              f"{'CORRECT' if p['correct'] else 'WRONG'}")
        print(f"      answer: {p['answer'][:200]!r}")
        if p["other_scene_numbers_mentioned"]:
            print(f"      NOTE: also mentioned other scenes' numbers "
                  f"{p['other_scene_numbers_mentioned']} -- possible confusion")
    print(f"\n  -> {outp}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8091/v1/video/chat/stream")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--scenes", default="/data/zx/stimuli/recall")
    ap.add_argument("--outdir", default="/data/zx/results/recall")
    ap.add_argument("--policy-label", default="unknown",
                    help="recorded in the output; set it to the PA_HISTORY_POLICY in use")
    ap.add_argument("--describe-turns", type=int, default=8)
    # Which question the non-probe turns ask. This choice decides what the
    # experiment can distinguish:
    #   q_describe  the answer names the word, so it doubles as a caption --
    #               full_text then scores nearly as well as text_memory and the
    #               deliberate scene note looks worthless.
    #   q_shape     the answer names only the shape, so the word survives ONLY
    #               if something wrote it down on purpose. This isolates the
    #               contribution of the generated note.
    ap.add_argument("--describe-utter", default="q_shape",
                    choices=["q_shape", "q_describe"])
    ap.add_argument("--system-prompt", default=QWEN_OMNI_SPEECH_SYSTEM_PROMPT)
    ap.add_argument("--memory-hint", action="store_true",
                    help="append MEMORY_HINT to the system prompt: tell the model "
                         "written notes exist and to consult them for the past")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--chunk-ms", type=int, default=100)
    # The frame buffer is a rolling window that is NEVER cleared between turns,
    # so it must be sized to hold ONLY the current scene -- otherwise the model
    # answers the probes by *looking* at earlier scenes still in view, and a
    # memoryless system scores like a perfect one.
    #
    # 8 == 8 also disables subsampling (n_buf <= num_frames takes the whole
    # buffer), which matters because the stride formula picks index 0 -- the
    # OLDEST frame -- first.
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=8)
    # EVS OFF by default here, and this is load-bearing rather than a
    # preference. Measured by replaying the shipped filter over these stimuli:
    # within one scene the 3 px jitter leaves frames ~99% similar, so EVS
    # retains exactly 1 frame PER SCENE. The rolling buffer then spans up to
    # max_frames *scenes* at once -- in the first smoke run the model saw four
    # scenes simultaneously ("a series of four static images") and scored 3/3 on
    # the recall probe by reading them off the screen. With EVS off, all ~11
    # frames of the current scene enter the buffer and the last 8 are all from
    # the current turn.
    ap.add_argument("--evs", action="store_true", default=False)
    ap.add_argument("--no-evs", dest="evs", action="store_false")
    ap.add_argument("--evs-threshold", type=float, default=0.95)
    ap.add_argument("--think-s", type=float, default=2.0)
    ap.add_argument("--lead-in-s", type=float, default=4.0)
    ap.add_argument("--inter-turn-idle-s", type=float, default=3.0,
                    help="raise above the observed [PA_MEM] dur_ms so the "
                         "memory-note call does not overlap the next turn")
    ap.add_argument("--turn-timeout-s", type=float, default=120.0)
    ap.add_argument("--ws-max-mb", type=int, default=32)
    args = ap.parse_args()
    if args.memory_hint:
        args.system_prompt = (args.system_prompt or "") + MEMORY_HINT
        args.policy_label = f"{args.policy_label}+hint"
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
