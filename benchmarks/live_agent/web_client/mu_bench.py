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
import os
import base64
import json
import pathlib
import random
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe import session_config, synth_frame_jpeg, synth_speechlike_pcm  # noqa: E402

# The engine's CURRENT log file. Boots write wherever the launch redirected
# them, so the default here can silently go stale (probes then count an empty
# slice -- every engine_probes field reads 0). Point MU_ENGINE_LOG at the live
# boot log; verify with: ls -la /proc/<engine pid>/fd | grep log
LOG = pathlib.Path(os.environ.get("MU_ENGINE_LOG", "/data/zx/results/qwen_live.log"))
URL = "ws://127.0.0.1:8091/v1/video/chat/stream"
FRAMES_ROOT = pathlib.Path("/data/zx/stimuli/frames640")

FRAME_INTERVAL_S = 0.5          # 2 fps, the browser page's rhythm (see --video-interval-ms)
TURN_TIMEOUT_S = 180.0
THINK_S = (2.0, 6.0)            # closed-loop pause after each reply (see --think)
# Wait for the reply to finish PLAYING (not merely arriving) before the think
# pause. See the note at the sleep site: arrival-paced turns under-load any
# engine that delivers faster than realtime.
PLAYBACK_PACED = os.environ.get("MU_PLAYBACK_PACED", "1") not in ("0", "", "false", "False")
_stag = os.environ.get("MU_STAGGER_S")  # "lo,hi" override for arrival-spread控制实验
STAGGER_S = tuple(float(x) for x in _stag.split(",")) if _stag else (0.0, 8.0)
WARMUP_S = 4.0                  # let the frame pump run before the first query
GIVE_UP_AFTER = 3               # consecutive timeouts before a user stops

# Temporal-batching experiment additions (benchmarks/temporal_batching/DESIGN.zh.md):
# audio input is streamed in fixed-cadence chunks -- the AUDIO GRID (80 ms =
# 1 codec frame = the model's own 12.5 Hz rhythm). Video takes no clock of its
# own: its interval should be an integer multiple of the audio cadence
# (e.g. 480 = 6x80), set via --video-interval-ms.
AUDIO_CADENCE_MS = 80

SYSTEM_PROMPT = ("You are a voice assistant. "
                 "Answer each question out loud in one short sentence.")

# Optional listening artifacts: dump each ok turn's audio as
# <dir>/<user>_tNN.wav (24 kHz mono PCM). Env-driven like MU_SESSION_CFG_JSON
# so the ladder scripts need no new flags. Meant for small cells (u1); at 32
# users this writes ~1k files.
SAVE_WAV_DIR = os.environ.get("MU_SAVE_WAV_DIR")
if SAVE_WAV_DIR:
    os.makedirs(SAVE_WAV_DIR, exist_ok=True)

# Text-only sessions (MU_TEXT_ONLY=1): modalities ["text"], turns close on
# response.text.done, ttfa is undefined. Pair with a thinker-only engine to
# measure the brain with no talker anywhere in the building.
TEXT_ONLY = os.environ.get("MU_TEXT_ONLY") == "1"

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

# MU_QUESTIONS=long: answers of ~4 sentences (~15-25 s of speech) instead of
# one short sentence. Triples the sustained talker/code2wav duty per session,
# which is the knob that separates "idle box, everything is easy" from a load
# where mid-turn scheduling collisions can actually happen.
# MU_QUESTIONS=mixed: alternate short and long -- the realistic conversation
# shape (quick factual turns interleaved with long descriptive ones).
_SHORT_QUESTIONS = QUESTIONS
if os.environ.get("MU_QUESTIONS") in ("long", "mixed"):
    QUESTIONS = [
        "Describe what a sunrise over the ocean looks like, in about four sentences.",
        "Explain how bread is made, in about four sentences.",
        "Describe a walk through a quiet forest, in about four sentences.",
        "Explain why the sky is blue, in about four sentences.",
        "Describe what a busy train station feels like, in about four sentences.",
        "Explain how bees make honey, in about four sentences.",
        "Describe a thunderstorm from indoors, in about four sentences.",
        "Explain how a bicycle stays upright, in about four sentences.",
        "Describe a small mountain village in winter, in about four sentences.",
        "Explain how rain forms, in about four sentences.",
        "Describe the smell and sounds of a bakery in the morning, in about four sentences.",
        "Explain what makes autumn leaves change color, in about four sentences.",
    ]
if os.environ.get("MU_QUESTIONS") == "mixed":
    QUESTIONS = [q for pair in zip(_SHORT_QUESTIONS, QUESTIONS) for q in pair]

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
    "arrival_prefill": r"prefill-on-arrival",
    "compress_warm": r"COMPRESS: warming shadow",
    "compress_swap": r"COMPRESS #\d+ at turn",
    "warmup_queued": r"warm-up queued",
    "blocking_roll": r"(?i)blocking roll",
    # Count nonzero per-request preemption COUNTERS, not the word: our own
    # scheduler dumps print "preemptions=0" on every row, and a bare /preempt/i
    # counted 840 of those in a run whose real preemption count was zero.
    "preempted_reqs": r"preemptions=[1-9]",
    "recompute": r"(?i)recomput",
}


def load_frames(content: str) -> list[str]:
    if content == "none":
        # Audio-only scenario: no camera at all. The user side is a text query
        # (standing in for ASR'd speech); the model still answers with speech,
        # so the whole output pipeline is loaded -- only the visual input is gone.
        return []
    if content == "synthetic":
        # Locally generated frames -- no stimuli directory needed. Per-USER
        # variation happens in User.__init__ (labels carry the user name), so
        # prefix caching cannot collapse the cohort; this shared pool is only
        # the fallback for code paths that read frames before users exist.
        return [base64.b64encode(synth_frame_jpeg(f"shared {i}")).decode() for i in range(16)]
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

    def __init__(self, uid: int, rep: int, frames: list[str], turns: int, seed: int,
                 opts: argparse.Namespace | None = None):
        self.uid = uid
        self.rep = rep
        self.name = f"r{rep}u{uid}"
        self.opts = opts
        self.video_interval_s = (opts.video_interval_ms / 1000.0) if opts else FRAME_INTERVAL_S
        # Dynamic video cadence: a real camera client raises frame rate while
        # the user speaks / the reply plays (the model may want to see) and
        # drops it while idle. When set, the pump uses this interval from
        # speech start until response done, and video_interval_s otherwise.
        _act = getattr(opts, "video_interval_active_ms", None) if opts else None
        self.video_interval_active_s = (_act / 1000.0) if _act else None
        self.turn_active = False
        self.audio_input_s = opts.audio_input_s if opts else 0.0
        self.think_s = opts.think_range if opts else THINK_S
        if opts is not None and opts.content == "synthetic":
            # Per-user frames: the label carries the user name, so no two
            # users' frames byte-match and prefix caching cannot collapse them.
            frames = [base64.b64encode(synth_frame_jpeg(f"{self.name} {i}")).decode()
                      for i in range(16)]
        self.frames = frames
        self.turns = turns
        self.rng = random.Random(seed * 10_000 + rep * 100 + uid)
        self.frame_pos = self.rng.randrange(len(frames)) if frames else 0
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
        if TEXT_ONLY:
            cfg["modalities"] = ["text"]
            cfg["prefill_frames_on_arrival"] = False
        # Session-config overrides injected by the harness (e.g. context compression
        # knobs) without forking the bench: MU_SESSION_CFG_JSON='{"key": value}'.
        _extra = os.environ.get("MU_SESSION_CFG_JSON")
        if _extra:
            cfg.update(json.loads(_extra))

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
        if not self.frames:
            return
        seq = 0
        while True:
            await ws.send(json.dumps({
                "type": "video.frame",
                "data": self.frames[self.frame_pos],
                "frame_id": f"{self.name}-f{seq}",
            }))
            self.frame_pos = (self.frame_pos + 1) % len(self.frames)
            seq += 1
            interval = self.video_interval_s
            if self.video_interval_active_s is not None and self.turn_active:
                interval = self.video_interval_active_s
            await asyncio.sleep(interval)

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
                if TEXT_ONLY:
                    # Text-only turns have no audio.done; this close IS the end.
                    cur["t_done"] = t_now
                    self.done_evt.set()
            elif t == "response.audio.delta":
                if cur["t_first_audio"] is None:
                    cur["t_first_audio"] = t_now
                data = base64.b64decode(msg["data"])
                samples = (len(data) - 44) // 2
                # Full per-delta timeline (arrival stamp, samples): the raw
                # material for inter-chunk jitter, per-chunk playback deadlines
                # and tick-alignment checks. ~30 pairs per 10 s reply -- cheap.
                cur["deltas"].append((t_now, samples))
                if SAVE_WAV_DIR:
                    cur["pcm"] += data[44:]
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

    async def _stream_audio_input(self, ws) -> None:
        """Stream synthetic speech at the AUDIO GRID cadence (80 ms chunks).

        Real-time pacing, not a blob: the point of the temporal-batching
        experiment is that inputs ARRIVE on the grid, so every condition sees
        the same arrival process. 80 ms at 16 kHz = 1280 samples = 2560 bytes.
        """
        chunk_samples = int(16000 * AUDIO_CADENCE_MS / 1000)
        pcm = synth_speechlike_pcm(self.audio_input_s)
        step = chunk_samples * 2
        for off in range(0, len(pcm), step):
            await ws.send(json.dumps({
                "type": "audio.chunk",
                "data": base64.b64encode(pcm[off:off + step]).decode(),
            }))
            await asyncio.sleep(AUDIO_CADENCE_MS / 1000)

    async def _turn_loop(self, ws) -> None:
        consecutive_timeouts = 0
        for i in range(self.turns):
            q = QUESTIONS[(self.q_offset + i) % len(QUESTIONS)]
            self.turn_active = True
            if self.audio_input_s > 0:
                await self._stream_audio_input(ws)
            self.cur = {
                "t_first_text": None, "t_first_audio": None, "t_done": None,
                "text_stream": "", "text_at_first_sound": "",
                "audio_samples": 0, "n_deltas": 0, "max_starve_s": 0.0,
                "deltas": [], "pcm": bytearray(),
            }
            self.done_evt.clear()
            t_q = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": q}))
            try:
                await asyncio.wait_for(self.done_evt.wait(), timeout=TURN_TIMEOUT_S)
            except asyncio.TimeoutError:
                self.turn_active = False
                consecutive_timeouts += 1
                self.records.append({
                    "user": self.name, "turn": i + 1, "q": q, "status": "timeout",
                    "t_q": t_q, "t_ft": self.cur["t_first_text"],
                    "t_fa": self.cur["t_first_audio"], "t_done": None,
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
            self.turn_active = False
            cur, self.cur = self.cur, None
            ttfa = (cur["t_first_audio"] - t_q) * 1000 if cur["t_first_audio"] else None
            ttft = (cur["t_first_text"] - t_q) * 1000 if cur["t_first_text"] else None
            wall = cur["t_done"] - t_q
            audio_s = cur["audio_samples"] / 24000.0
            deliver_s = (cur["t_done"] - cur["t_first_audio"]) if cur["t_first_audio"] else None
            self.records.append({
                "user": self.name, "turn": i + 1, "q": q, "status": "ok",
                # Absolute monotonic stamps (one clock: all users share this
                # process). They let the analysis reconstruct, for any turn,
                # how many OTHER turns were mid-TTFA or mid-delivery when this
                # one arrived -- the split between "queued behind others" and
                # "everything got slower" that percentiles alone cannot give.
                "t_q": t_q, "t_ft": cur["t_first_text"],
                "t_fa": cur["t_first_audio"], "t_done": cur["t_done"],
                "ttfa_ms": ttfa, "ttft_ms": ttft, "wall_s": wall,
                "audio_s": audio_s,
                "rtf_deliver": (audio_s / deliver_s) if deliver_s and deliver_s > 0 else None,
                "max_starve_ms": cur["max_starve_s"] * 1000,
                # [seconds since t_q, samples] per audio delta, in arrival order.
                "deltas": [[round(t - t_q, 4), s] for t, s in cur["deltas"]],
                "n_deltas": cur["n_deltas"],
                "chars_stream": len(cur["text_stream"]),
                "chars_at_first_sound": len(cur["text_at_first_sound"]),
                "text": cur["text_stream"][:200],
            })
            if SAVE_WAV_DIR and cur["pcm"]:
                import wave
                p = pathlib.Path(SAVE_WAV_DIR) / f"{self.name}_t{i + 1:02d}.wav"
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(bytes(cur["pcm"]))
            # [closed loop, playback-paced] The turn ENDS for the server when
            # the last audio byte is sent, but a human ends it when the audio
            # finishes PLAYING. Starting the think-time at arrival lets an
            # engine that races ahead of realtime shorten its own session
            # cycle: measured at 64 nominal users, the engine that delivered a
            # 25 s answer in ~16 s ran at median 28 concurrent turns while the
            # paced engine ran at 49 -- i.e. the racing engine was quietly
            # tested at 57%% of the load. Wait out the remaining playback first
            # so both engines face the same number of simultaneous speakers.
            # MU_PLAYBACK_PACED=0 restores arrival-paced turns.
            if PLAYBACK_PACED and cur["t_first_audio"] and audio_s > 0:
                _play_end = cur["t_first_audio"] + audio_s
                _left = _play_end - time.monotonic()
                if _left > 0:
                    await asyncio.sleep(_left)
            await asyncio.sleep(self.rng.uniform(*self.think_s))

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
        # First few error strings per user: the 8-user wedge postmortem had to
        # be reconstructed from the engine log because these never hit disk.
        "per_user_errors": {u.name: u.errors[:5] for u in users if u.errors},
        "engine_probes": probes,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--content", required=True,
                    choices=["none", "synthetic", "screencast", "talkinghead", "handheld_walk_talk"])
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--repeat-sessions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    # Temporal-batching experiment knobs (DESIGN.zh.md): video rides the audio
    # grid at an integer multiple of 80 ms; audio input streams at 80 ms/chunk.
    ap.add_argument("--video-interval-active-ms", type=int, default=None,
                    help="when set, frame interval while a turn is active (speech start -> "
                         "response done); --video-interval-ms then applies only to idle/think time")
    ap.add_argument("--video-interval-ms", type=int, default=int(FRAME_INTERVAL_S * 1000),
                    help="frame pump interval; use a multiple of 80 (e.g. 480)")
    ap.add_argument("--audio-input-s", type=float, default=0.0,
                    help="stream this many seconds of synthetic speech before each query, "
                         "paced at 80 ms/chunk (0 = text-only queries, the old behavior)")
    ap.add_argument("--think", default=None, metavar="LO,HI",
                    help=f"think-time range in seconds (default {THINK_S[0]},{THINK_S[1]})")
    args = ap.parse_args()
    args.think_range = tuple(float(x) for x in args.think.split(",")) if args.think else THINK_S
    if args.video_interval_ms % AUDIO_CADENCE_MS:
        print(f"warning: --video-interval-ms {args.video_interval_ms} is not a multiple "
              f"of the {AUDIO_CADENCE_MS} ms audio grid", file=sys.stderr)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.content)
    log_offset = LOG.stat().st_size if LOG.exists() else 0
    t_start = time.monotonic()

    all_users: list[User] = []
    for rep in range(args.repeat_sessions):
        cohort = [User(uid, rep, frames, args.turns, args.seed, opts=args)
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
            "video_interval_ms": args.video_interval_ms, "audio_input_s": args.audio_input_s,
            "think_s": list(args.think_range),
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
