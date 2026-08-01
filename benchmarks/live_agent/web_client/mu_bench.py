#!/usr/bin/env python3
"""Multi-user scaling driver: N simulated browsers against the live session engine.

    mu_bench.py --users 4 --content talkinghead --turns 30 --out /data/zx/results/mu_talkinghead_u4

Each simulated user is one real WebSocket to ws://127.0.0.1:8091 -- the same
door the browser page uses -- with the browser replaced by a script:

  * a continuous 2 fps frame pump playing a REAL frame set from
    /data/zx/stimuli/frames640/<content> (the July taxonomy: screencast /
    talkinghead / handheld_walk_talk), each user starting at a different
    offset so prefix caching cannot collapse the cohort into one user;
  * a closed loop: ask, wait for the reply's audio to finish, think 2-6 s,
    ask again -- 30 turns per user, questions rotated per user so no two
    users ask the same thing at the same turn index.

All timestamps are the client's own monotonic clock, stamped on message
receipt BEFORE any decoding, and measured from the moment the query was sent.
`response.text.done` fires just before the first audio chunk, so its length
is text-at-first-sound, not the reply (July trap); the full reply length is
accumulated from text deltas and audio seconds instead.

The one-user cell should be run with --repeat-sessions 2: two sequential
30-turn sessions rather than one 60-turn session, so its turn indices match
the multi-user cells and turn index cannot be read as concurrency.
"""
import argparse
import asyncio
import base64
import json
import pathlib
import random
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe import session_config  # noqa: E402

LOG = pathlib.Path("/data/zx/results/qwen_live.log")
URL = "ws://127.0.0.1:8091/v1/video/chat/stream"
FRAMES_ROOT = pathlib.Path("/data/zx/stimuli/frames640")

FRAME_INTERVAL_S = 0.5          # 2 fps, the browser page's rhythm
TURN_TIMEOUT_S = 180.0
THINK_S = (2.0, 6.0)            # closed-loop pause after each reply
STAGGER_S = (0.0, 8.0)          # spread session starts so turn 1 is not a stampede
WARMUP_S = 4.0                  # let the frame pump run before the first query
GIVE_UP_AFTER = 3               # consecutive timeouts before a user stops

SYSTEM_PROMPT = ("You are a voice assistant. "
                 "Answer each question out loud in one short sentence.")

QUESTIONS = [
    "Name one primary color.",
    "What is two plus three?",
    "Say a short greeting to a visitor.",
    "Name an animal that can fly.",
    "What season comes after winter?",
    "What do bees make?",
    "Name a fruit that is yellow.",
    "How many legs does a spider have?",
    "What is the opposite of hot?",
    "Name one planet in our solar system.",
    "What color is the sky on a clear day?",
    "What is ten minus four?",
    "Name a musical instrument with strings.",
    "What do you call frozen water?",
    "Name one day of the weekend.",
    "What animal says moo?",
    "What is the first letter of the alphabet?",
    "Name something you can drink.",
    "How many wheels does a bicycle have?",
    "What is the opposite of fast?",
    "Name a vegetable that is orange.",
    "What do you use to write on paper?",
    "Name one month of summer.",
    "What is five times two?",
    "What do you call a baby dog?",
    "Name something that flies in the sky.",
    "What is the opposite of up?",
    "Name a sport played with a ball.",
    "How many fingers are on one hand?",
    "What do you wear on your feet?",
]

# The engine-side probes worth counting over each cell's log slice. All of
# them are presence checks -- a healthy run has nonzero segment stops and
# zero everything in "bad".
LOG_PROBES_BAD = {
    "unowned_audio": r"UNOWNED",
    "torch_cat_error": r"expected a non-empty list of Tensors",
    "counter_leak_clamped": r"streaming-parked counter had leaked",
    "zero_output_wedge": r"sampled ZERO output tokens",
    "negative_slice": r"scope drift; shipping unadjusted",
}
LOG_PROBES_INFO = {
    "segment_stops": r"\[session\] audio segment stop",
    "prefill_only": r"\[prefill-only\]",
    "boundary_cap": r"\[boundary-cap\]",
    "preempt": r"(?i)preempt",
    "recompute": r"(?i)recomput",
}


def load_frames(content: str) -> list[str]:
    d = FRAMES_ROOT / content
    files = sorted(d.glob("*.jpg"))
    assert files, f"no frames under {d}"
    return [base64.b64encode(f.read_bytes()).decode() for f in files]


def pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class User:
    """One simulated browser: a connection, a frame pump, and a turn loop."""

    def __init__(self, uid: int, rep: int, frames: list[str], turns: int, seed: int):
        self.uid = uid
        self.rep = rep
        self.name = f"r{rep}u{uid}"
        self.frames = frames
        self.turns = turns
        self.rng = random.Random(seed * 10_000 + rep * 100 + uid)
        self.frame_pos = self.rng.randrange(len(frames))
        self.q_offset = (uid * 7 + rep * 3) % len(QUESTIONS)
        self.records: list[dict] = []
        self.acks_accepted = 0
        self.acks_filtered = 0
        self.rolls = 0
        self.stray_audio = 0
        self.errors: list[str] = []
        self.cur: dict | None = None
        self.done_evt = asyncio.Event()

    async def run(self) -> None:
        import websockets

        cfg = session_config(SYSTEM_PROMPT)
        cfg["frame_filter_min_gap"] = 0
        cfg["frame_filter_max_gap"] = 4
        cfg["prefill_frames_on_arrival"] = True

        try:
            async with websockets.connect(URL, max_size=None) as ws:
                await ws.send(json.dumps(cfg))
                reader = asyncio.create_task(self._reader(ws))
                pump = asyncio.create_task(self._frame_pump(ws))
                try:
                    await asyncio.sleep(WARMUP_S + self.rng.uniform(*STAGGER_S))
                    await self._turn_loop(ws)
                finally:
                    pump.cancel()
                    reader.cancel()
        except Exception as e:  # connection refused / dropped mid-run
            self.errors.append(f"connection: {e!r:.200}")
            self._mark_skipped(reason="connection_lost")

    async def _frame_pump(self, ws) -> None:
        seq = 0
        while True:
            await ws.send(json.dumps({
                "type": "video.frame",
                "data": self.frames[self.frame_pos],
                "frame_id": f"{self.name}-f{seq}",
            }))
            self.frame_pos = (self.frame_pos + 1) % len(self.frames)
            seq += 1
            await asyncio.sleep(FRAME_INTERVAL_S)

    async def _reader(self, ws) -> None:
        async for raw in ws:
            t_now = time.monotonic()          # stamp BEFORE any decoding
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t == "video.frame.ack":
                if msg.get("accepted"):
                    self.acks_accepted += 1
                else:
                    self.acks_filtered += 1
                continue
            if t == "session.rolled":
                self.rolls += 1
                continue
            if t == "error":
                self.errors.append(str(msg.get("message"))[:200])
                continue
            cur = self.cur
            if cur is None:
                if t == "response.audio.delta":
                    self.stray_audio += 1
                continue
            if t == "response.text.delta":
                if cur["t_first_text"] is None:
                    cur["t_first_text"] = t_now
                cur["text_stream"] += msg.get("delta") or ""
            elif t == "response.text.done":
                cur["text_at_first_sound"] = msg.get("text") or ""
            elif t == "response.audio.delta":
                if cur["t_first_audio"] is None:
                    cur["t_first_audio"] = t_now
                samples = (len(base64.b64decode(msg["data"])) - 44) // 2
                # starvation: a player that started at the first delta has
                # consumed (t_now - t_first) seconds; was that much delivered?
                played = t_now - cur["t_first_audio"]
                avail = cur["audio_samples"] / 24000.0
                cur["max_starve_s"] = max(cur["max_starve_s"], played - avail)
                cur["audio_samples"] += samples
                cur["n_deltas"] += 1
            elif t == "response.audio.done":
                cur["t_done"] = t_now
                self.done_evt.set()

    async def _turn_loop(self, ws) -> None:
        consecutive_timeouts = 0
        for i in range(self.turns):
            q = QUESTIONS[(self.q_offset + i) % len(QUESTIONS)]
            self.cur = {
                "t_first_text": None, "t_first_audio": None, "t_done": None,
                "text_stream": "", "text_at_first_sound": "",
                "audio_samples": 0, "n_deltas": 0, "max_starve_s": 0.0,
            }
            self.done_evt.clear()
            t_q = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": q}))
            try:
                await asyncio.wait_for(self.done_evt.wait(), timeout=TURN_TIMEOUT_S)
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                self.records.append({
                    "user": self.name, "turn": i + 1, "q": q, "status": "timeout",
                    "ttfa_ms": None, "ttft_ms": None, "wall_s": None,
                    "audio_s": self.cur["audio_samples"] / 24000.0,
                    "chars_stream": len(self.cur["text_stream"]),
                })
                self.cur = None
                if consecutive_timeouts >= GIVE_UP_AFTER:
                    self._mark_skipped(start=i + 1, reason="gave_up")
                    return
                continue
            consecutive_timeouts = 0
            cur, self.cur = self.cur, None
            ttfa = (cur["t_first_audio"] - t_q) * 1000 if cur["t_first_audio"] else None
            ttft = (cur["t_first_text"] - t_q) * 1000 if cur["t_first_text"] else None
            wall = cur["t_done"] - t_q
            audio_s = cur["audio_samples"] / 24000.0
            deliver_s = (cur["t_done"] - cur["t_first_audio"]) if cur["t_first_audio"] else None
            self.records.append({
                "user": self.name, "turn": i + 1, "q": q, "status": "ok",
                "ttfa_ms": ttfa, "ttft_ms": ttft, "wall_s": wall,
                "audio_s": audio_s,
                "rtf_deliver": (audio_s / deliver_s) if deliver_s and deliver_s > 0 else None,
                "max_starve_ms": cur["max_starve_s"] * 1000,
                "n_deltas": cur["n_deltas"],
                "chars_stream": len(cur["text_stream"]),
                "chars_at_first_sound": len(cur["text_at_first_sound"]),
                "text": cur["text_stream"][:200],
            })
            await asyncio.sleep(self.rng.uniform(*THINK_S))

    def _mark_skipped(self, start: int | None = None, reason: str = "") -> None:
        done = {r["turn"] for r in self.records}
        first = start if start is not None else 1
        for i in range(first, self.turns + 1):
            if i not in done:
                self.records.append({"user": self.name, "turn": i, "status": "skipped",
                                     "reason": reason})


def summarize(records: list[dict], users: list[User], meta: dict, log_slice: str) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    ttfa = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms") is not None]
    per_user_p50 = {}
    for u in users:
        xs = [r["ttfa_ms"] for r in u.records if r.get("ttfa_ms") is not None]
        if xs:
            per_user_p50[u.name] = pctl(xs, 0.5)
    probes = {k: len(re.findall(p, log_slice)) for k, p in
              {**LOG_PROBES_BAD, **LOG_PROBES_INFO}.items()}
    return {
        **meta,
        "n_ok": len(ok),
        "n_timeout": sum(1 for r in records if r.get("status") == "timeout"),
        "n_skipped": sum(1 for r in records if r.get("status") == "skipped"),
        "ttfa_p50_ms": pctl(ttfa, 0.5),
        "ttfa_p95_ms": pctl(ttfa, 0.95),
        "ttfa_over_1s_pct": 100 * sum(1 for x in ttfa if x > 1000) / len(ttfa) if ttfa else None,
        "ttfa_over_2s_pct": 100 * sum(1 for x in ttfa if x > 2000) / len(ttfa) if ttfa else None,
        "ttft_p50_ms": pctl([r["ttft_ms"] for r in ok if r.get("ttft_ms")], 0.5),
        "audio_s_p50": pctl([r["audio_s"] for r in ok], 0.5),
        "chars_stream_p50": pctl([float(r["chars_stream"]) for r in ok], 0.5),
        "rtf_deliver_p50": pctl([r["rtf_deliver"] for r in ok if r.get("rtf_deliver")], 0.5),
        "max_starve_ms_p95": pctl([r["max_starve_ms"] for r in ok if "max_starve_ms" in r], 0.95),
        "fairness_p50_spread": (max(per_user_p50.values()) / min(per_user_p50.values()))
                               if len(per_user_p50) > 1 and min(per_user_p50.values()) > 0 else None,
        "per_user_ttfa_p50": per_user_p50,
        "acks_accepted": sum(u.acks_accepted for u in users),
        "acks_filtered": sum(u.acks_filtered for u in users),
        "session_rolls": sum(u.rolls for u in users),
        "stray_audio_deltas": sum(u.stray_audio for u in users),
        "client_errors": sum(len(u.errors) for u in users),
        "engine_probes": probes,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--content", required=True,
                    choices=["screencast", "talkinghead", "handheld_walk_talk"])
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--repeat-sessions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.content)
    log_offset = LOG.stat().st_size if LOG.exists() else 0
    t_start = time.monotonic()

    all_users: list[User] = []
    for rep in range(args.repeat_sessions):
        cohort = [User(uid, rep, frames, args.turns, args.seed)
                  for uid in range(args.users)]
        await asyncio.gather(*(u.run() for u in cohort))
        all_users.extend(cohort)

    records = [r for u in all_users for r in u.records]
    records.sort(key=lambda r: (r["user"], r["turn"]))
    with (out / "turns.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    log_slice = ""
    if LOG.exists():
        with LOG.open("rb") as fh:
            fh.seek(log_offset)
            log_slice = re.sub(rb"\x1b\[[0-9;]*m", b"", fh.read()).decode(errors="replace")

    meta = {"users": args.users, "content": args.content, "turns_per_user": args.turns,
            "repeat_sessions": args.repeat_sessions, "seed": args.seed,
            "wall_s": time.monotonic() - t_start}
    summary = summarize(records, all_users, meta, log_slice)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"== {args.content} x {args.users} users: "
          f"ok={summary['n_ok']} timeout={summary['n_timeout']} skipped={summary['n_skipped']}")
    print(f"   ttfa p50={summary['ttfa_p50_ms']} p95={summary['ttfa_p95_ms']} "
          f">1s={summary['ttfa_over_1s_pct']}% rtf={summary['rtf_deliver_p50']}")
    bad = {k: summary["engine_probes"][k] for k in LOG_PROBES_BAD}
    print(f"   probes bad={bad} stops={summary['engine_probes']['segment_stops']}")
    return 0 if summary["n_ok"] > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
