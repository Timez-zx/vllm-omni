#!/usr/bin/env python3
"""Build the canonical continuous-AV corpus from SLURP-real and DAVIS 2017.

SLURP contributes real voice-assistant requests with speaker IDs. Half of the
sessions use close-talk audio and half use distant microphones, with one fixed
condition per speaker/session. DAVIS contributes real video with camera and
object motion. The output is the WAV JSONL manifest and flat JPEG directory
consumed by ``mu_bench.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import wave
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Candidate:
    file: str
    speaker: str
    transcript: str
    scenario: str
    slurp_id: int
    duration_s: float
    mic_condition: str
    wer: float | None


def stable_shuffle(values: list, key: str) -> None:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    random.Random(seed).shuffle(values)


def annotation_commit(root: pathlib.Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def load_candidates(
    annotations: pathlib.Path,
    audio_root: pathlib.Path,
    min_duration_s: float,
    max_duration_s: float,
) -> dict[str, list[Candidate]]:
    metadata = json.loads((annotations / "dataset/slurp/metadata.json").read_text())
    file_to_speaker: dict[str, str] = {}
    for item in metadata.values():
        for filename, recording in item.get("recordings", {}).items():
            if recording.get("status") == "correct":
                file_to_speaker[filename] = recording["usrid"]

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    seen_files: set[str] = set()
    for split in ("train", "devel", "test"):
        path = annotations / f"dataset/slurp/{split}.jsonl"
        for raw in path.read_text().splitlines():
            item = json.loads(raw)
            transcript = " ".join(str(item["sentence"]).split())
            if not 2 <= len(transcript.split()) <= 24:
                continue
            for recording in item["recordings"]:
                filename = recording["file"]
                if filename in seen_files or filename not in file_to_speaker or recording.get("status") != "correct":
                    continue
                source = audio_root / filename
                if not source.is_file():
                    continue
                info = sf.info(source)
                duration_s = info.frames / info.samplerate
                if not min_duration_s <= duration_s <= max_duration_s:
                    continue
                seen_files.add(filename)
                speaker = file_to_speaker[filename]
                grouped[speaker].append(
                    Candidate(
                        file=filename,
                        speaker=speaker,
                        transcript=transcript,
                        scenario=str(item["scenario"]),
                        slurp_id=int(item["slurp_id"]),
                        duration_s=duration_s,
                        mic_condition="close" if filename.endswith("-headset.flac") else "distant",
                        wer=float(recording["wer"]) if recording.get("wer") is not None else None,
                    )
                )
    return grouped


def choose_speakers(
    grouped: dict[str, list[Candidate]],
    count: int,
    utterances_per_speaker: int,
    distant_share: float,
) -> dict[str, str]:
    target_distant = round(count * distant_share)
    targets = {"close": count - target_distant, "distant": target_distant}
    selected: dict[str, str] = {}
    for condition in ("close", "distant"):
        needed = targets[condition]
        by_group: dict[str, list[str]] = defaultdict(list)
        for speaker, candidates in grouped.items():
            eligible = sum(candidate.mic_condition == condition for candidate in candidates)
            if speaker not in selected and eligible >= utterances_per_speaker:
                by_group[speaker.split("-", 1)[0]].append(speaker)
        for group, speakers in by_group.items():
            stable_shuffle(speakers, f"speaker-group:{condition}:{group}")
        groups = sorted(by_group)
        chosen: list[str] = []
        while len(chosen) < needed and any(by_group.values()):
            for group in groups:
                if by_group[group] and len(chosen) < needed:
                    chosen.append(by_group[group].pop())
        if len(chosen) != needed:
            raise ValueError(
                f"only {len(chosen)} unused {condition} speakers have "
                f"{utterances_per_speaker} eligible utterances; need {needed}"
            )
        selected.update({speaker: condition for speaker in chosen})
    return dict(sorted(selected.items()))


def choose_utterances(candidates: list[Candidate], count: int, speaker: str) -> list[Candidate]:
    by_scenario: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[candidate.scenario].append(candidate)
    for scenario, values in by_scenario.items():
        stable_shuffle(values, f"utterance:{speaker}:{scenario}")
    chosen: list[Candidate] = []
    scenarios = sorted(by_scenario)
    while len(chosen) < count:
        progressed = False
        for scenario in scenarios:
            if by_scenario[scenario] and len(chosen) < count:
                chosen.append(by_scenario[scenario].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"speaker {speaker} ran out of utterances")
    return chosen


def write_wav(source: pathlib.Path, destination: pathlib.Path) -> float:
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != AUDIO_RATE:
        mono = resample_poly(mono, AUDIO_RATE, rate)
    pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(AUDIO_RATE)
        output.writeframes(pcm.tobytes())
    return len(pcm) / AUDIO_RATE


def build_audio(args: argparse.Namespace, output: pathlib.Path) -> dict:
    grouped = load_candidates(
        args.slurp_annotations,
        args.slurp_audio,
        args.min_audio_s,
        args.max_audio_s,
    )
    speakers = choose_speakers(grouped, args.speakers, args.utterances_per_speaker, args.distant_share)
    manifest: list[dict] = []
    durations: list[float] = []
    wers: list[float] = []
    for speaker, mic_condition in speakers.items():
        candidates = [candidate for candidate in grouped[speaker] if candidate.mic_condition == mic_condition]
        for candidate in choose_utterances(candidates, args.utterances_per_speaker, speaker):
            relative = pathlib.Path("audio") / speaker / f"{pathlib.Path(candidate.file).stem}.wav"
            duration = write_wav(args.slurp_audio / candidate.file, output / relative)
            durations.append(duration)
            manifest.append(
                {
                    "id": f"slurp-{candidate.slurp_id}-{pathlib.Path(candidate.file).stem}",
                    "speaker": speaker,
                    "transcript": candidate.transcript,
                    "audio": str(relative),
                    "scenario": candidate.scenario,
                    "mic_condition": candidate.mic_condition,
                    "source_asr_wer": candidate.wer,
                }
            )
            if candidate.wer is not None:
                wers.append(candidate.wer)
    manifest.sort(key=lambda item: (item["speaker"], item["id"]))
    manifest_path = output / "audio_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(item) + "\n" for item in manifest))
    return {
        "source": "SLURP real close-talk and distant-microphone recordings",
        "source_url": "https://zenodo.org/records/4274930",
        "license": "CC BY-NC 4.0",
        "annotation_commit": annotation_commit(args.slurp_annotations),
        "speakers": len(speakers),
        "speaker_ids": sorted(speakers),
        "mic_condition_speakers": {
            condition: sum(value == condition for value in speakers.values()) for condition in ("close", "distant")
        },
        "utterances": len(manifest),
        "utterances_per_speaker": args.utterances_per_speaker,
        "duration_min_s": min(durations),
        "duration_max_s": max(durations),
        "duration_mean_s": sum(durations) / len(durations),
        "source_asr_wer_mean": sum(wers) / len(wers),
        "source_asr_wer_nonzero_share": sum(value > 0 for value in wers) / len(wers),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def build_frames(args: argparse.Namespace, output: pathlib.Path) -> dict:
    destination = output / "frames"
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str, pathlib.Path]] = []
    for sequence_dir in sorted(path for path in args.davis_jpegs.iterdir() if path.is_dir()):
        frames = sorted(sequence_dir.glob("*.jpg"))
        for frame in frames[:: args.video_stride]:
            selected.append((sequence_dir.name, frame))
    if not selected:
        raise ValueError(f"no DAVIS JPEG frames under {args.davis_jpegs}")
    for index, (sequence, source) in enumerate(selected):
        target = destination / f"{index:06d}-{sequence}-{source.name}"
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return {
        "source": "DAVIS 2017 trainval 480p",
        "source_url": "https://davischallenge.org/davis2017/code.html",
        "stride": args.video_stride,
        "assumed_source_fps": args.video_source_fps,
        "effective_fps": args.video_source_fps / args.video_stride,
        "sequences": len({sequence for sequence, _ in selected}),
        "frames": len(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurp-annotations", type=pathlib.Path, required=True)
    parser.add_argument("--slurp-audio", type=pathlib.Path, required=True)
    parser.add_argument("--davis-jpegs", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--speakers", type=int, default=80)
    parser.add_argument("--utterances-per-speaker", type=int, default=60)
    parser.add_argument("--min-audio-s", type=float, default=0.8)
    parser.add_argument("--max-audio-s", type=float, default=8.0)
    parser.add_argument("--distant-share", type=float, default=0.5)
    parser.add_argument("--video-source-fps", type=float, default=24.0)
    parser.add_argument("--video-stride", type=int, default=12)
    args = parser.parse_args()
    for path in (args.slurp_annotations, args.slurp_audio, args.davis_jpegs):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.speakers <= 0 or args.utterances_per_speaker <= 0 or args.video_stride <= 0:
        parser.error("speaker, utterance, and stride counts must be positive")
    if not 0 <= args.distant_share <= 1:
        parser.error("distant-share must be in [0, 1]")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance_path = args.out / "corpus_provenance.json"
    if provenance_path.exists():
        parser.error(f"output already prepared: {provenance_path}")
    provenance = {
        "schema": 1,
        "audio": build_audio(args, args.out),
        "video": build_frames(args, args.out),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
