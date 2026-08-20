#!/usr/bin/env python3
"""Deterministic plans and media loading for the continuous-AV benchmark."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pathlib
import random
import wave
from array import array
from dataclasses import dataclass

AUDIO_RATE = 16_000
AUDIO_CADENCE_MS = 200
VIDEO_INTERVAL_MS = 500
ENDPOINT_SILENCE_MS = 700
ECHO_GUARD_MS = 300


@dataclass(frozen=True, slots=True)
class Utterance:
    utterance_id: str
    speaker: str
    transcript: str
    pcm: bytes
    source: str
    mic_condition: str | None = None
    source_asr_wer: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (2 * AUDIO_RATE)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.pcm).hexdigest()


@dataclass(frozen=True, slots=True)
class TurnPlan:
    turn: int
    utterance: Utterance
    think_s: float

    def event_fields(self) -> dict:
        return {
            "turn": self.turn,
            "utterance_id": self.utterance.utterance_id,
            "speaker": self.utterance.speaker,
            "transcript": self.utterance.transcript,
            "input_audio_s": round(self.utterance.duration_s, 4),
            "input_audio_sha256": self.utterance.sha256,
            "mic_condition": self.utterance.mic_condition,
            "source_asr_wer": self.utterance.source_asr_wer,
            "think_s": self.think_s,
        }


@dataclass(frozen=True, slots=True)
class UserPlan:
    user: str
    start_delay_s: float
    frame_start_offset: int
    turns: tuple[TurnPlan, ...]

    def event_fields(self) -> dict:
        return {
            "user": self.user,
            "start_delay_s": self.start_delay_s,
            "frame_start_offset": self.frame_start_offset,
            "turns": [turn.event_fields() for turn in self.turns],
        }


def _load_pcm16_mono(path: pathlib.Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV")
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected PCM16 WAV")
        if wav.getframerate() != AUDIO_RATE:
            raise ValueError(f"{path}: expected {AUDIO_RATE} Hz WAV")
        if wav.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed WAV")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError(f"{path}: empty audio")
    return pcm


def load_audio_manifest(path: pathlib.Path) -> tuple[list[Utterance], dict]:
    """Load JSONL records: id, speaker, transcript, and a relative WAV path."""
    path = path.resolve()
    records: list[Utterance] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = json.loads(raw)
        missing = {"id", "speaker", "transcript", "audio"} - set(item)
        if missing:
            raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
        utterance_id = str(item["id"])
        if utterance_id in seen_ids:
            raise ValueError(f"{path}:{line_no}: duplicate id {utterance_id!r}")
        seen_ids.add(utterance_id)
        audio_path = (path.parent / str(item["audio"])).resolve()
        records.append(
            Utterance(
                utterance_id=utterance_id,
                speaker=str(item["speaker"]),
                transcript=str(item["transcript"]),
                pcm=_load_pcm16_mono(audio_path),
                source=str(audio_path),
                mic_condition=str(item["mic_condition"]) if item.get("mic_condition") else None,
                source_asr_wer=float(item["source_asr_wer"]) if item.get("source_asr_wer") is not None else None,
            )
        )
    if not records:
        raise ValueError(f"{path}: no utterances")
    speakers = sorted({record.speaker for record in records})
    corpus_hash = hashlib.sha256()
    for record in sorted(records, key=lambda value: value.utterance_id):
        corpus_hash.update(record.utterance_id.encode())
        corpus_hash.update(record.speaker.encode())
        corpus_hash.update(record.transcript.encode())
        corpus_hash.update(record.pcm)
    return records, {
        "audio_manifest": str(path),
        "audio_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audio_corpus_sha256": corpus_hash.hexdigest(),
        "audio_utterances": len(records),
        "audio_speakers": speakers,
        "audio_duration_min_s": min(record.duration_s for record in records),
        "audio_duration_max_s": max(record.duration_s for record in records),
    }


def load_frame_set(path: pathlib.Path) -> tuple[list[str], dict]:
    path = path.resolve()
    files = sorted((*path.glob("*.jpg"), *path.glob("*.jpeg")))
    if not files:
        raise ValueError(f"{path}: no JPEG frames")
    encoded: list[str] = []
    digest = hashlib.sha256()
    for frame_path in files:
        payload = frame_path.read_bytes()
        digest.update(payload)
        encoded.append(base64.b64encode(payload).decode())
    return encoded, {
        "frames_dir": str(path),
        "frame_set_sha256": digest.hexdigest(),
        "frame_count": len(encoded),
    }


def build_user_plan(
    *,
    uid: int,
    rep: int,
    users: int,
    turns: int,
    seed: int,
    utterances: list[Utterance],
    frame_count: int,
    stagger_s: tuple[float, float],
) -> UserPlan:
    if turns <= 0:
        raise ValueError("turns must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    cohort_index = rep * users + uid
    rng = random.Random(seed * 1_000_003 + cohort_index * 9_973)
    speakers = sorted({record.speaker for record in utterances})
    speaker_index = cohort_index % len(speakers)
    speaker_round = cohort_index // len(speakers)
    speaker = speakers[speaker_index]
    speaker_turns = [record for record in utterances if record.speaker == speaker]
    if len(speaker_turns) < turns:
        raise ValueError(f"speaker {speaker!r} has {len(speaker_turns)} utterances; need at least {turns}")
    # Every concurrency point is a nested workload: user N receives the same
    # recording plan at 8, 16, 32, ... users.  When concurrency exceeds the
    # number of speakers, repeated speakers consume disjoint windows before
    # any recording is reused across sessions.
    speaker_seed = int.from_bytes(hashlib.sha256(f"{seed}:{speaker}".encode()).digest()[:8], "big")
    random.Random(speaker_seed).shuffle(speaker_turns)
    start = (speaker_round * turns) % len(speaker_turns)
    planned: list[TurnPlan] = []
    for turn in range(1, turns + 1):
        utterance = speaker_turns[(start + turn - 1) % len(speaker_turns)]
        # Median near 3 s with a realistic long tail, bounded for reproducibility.
        think_s = 0.0 if turn == 1 else min(12.0, max(1.0, rng.lognormvariate(math.log(3.0), 0.55)))
        planned.append(TurnPlan(turn=turn, utterance=utterance, think_s=round(think_s, 3)))
    start_delay = rng.uniform(*stagger_s)
    return UserPlan(
        user=f"r{rep}u{uid}",
        start_delay_s=round(start_delay, 3),
        frame_start_offset=(cohort_index * 17) % frame_count,
        turns=tuple(planned),
    )


def make_room_tone_chunks(*, variant: int, chunks: int = 32) -> tuple[bytes, ...]:
    """Quiet deterministic PCM that keeps the mic stream non-identical per user."""
    samples_per_chunk = AUDIO_RATE * AUDIO_CADENCE_MS // 1000
    rng = random.Random(0xA710_0000 + variant)
    values = array("h", (rng.randrange(-96, 97) for _ in range(samples_per_chunk * chunks)))
    raw = values.tobytes()
    step = samples_per_chunk * 2
    return tuple(raw[offset : offset + step] for offset in range(0, len(raw), step))


def plan_sha256(plans: list[UserPlan]) -> str:
    payload = [plan.event_fields() for plan in plans]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
