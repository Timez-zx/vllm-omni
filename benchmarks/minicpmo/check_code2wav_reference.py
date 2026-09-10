"""Real-weight Code2Wav parity diagnostic; not a capacity or latency test.

Compare released stepaudio2.Token2wav.stream against BatchedToken2Wav using
the same loaded modules, prompt features, codec chunks and HiFT random draws.
Flow's fixed rand_noise buffer is shared. HiFT's rand/randn_like results are
recorded from the unmodified official forward and replayed per batch row;
merely reseeding cannot equate two singleton RNG calls with one batched call.
This controls randomness without disabling noise or claiming that production
batch-size changes naturally preserve RNG bit equivalence.

By default run two synthetic users with distinct legal codec tokens over two
complete turns, including flow-cache truncation. --payloads also replays real
speech_probe codec_payload JSONL events. Captured waveform hashes cannot be
compared unless the original HiFT random draws were captured too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Chunk:
    request_id: str
    cache_epoch: int
    chunk_seq: int
    codes: tuple[int, ...]
    last_chunk: bool
    model_turn_id: int | None = None


def smoke_chunks(*, users: int = 2, rounds: int = 2, chunks: int = 12, new_codes: int = 16):
    """Two independent streams, not duplicated inputs; 3-code left context."""
    if min(users, rounds, chunks, new_codes) < 1:
        raise ValueError("Smoke dimensions must be positive")
    streams = []
    for user in range(users):
        stream = []
        for turn in range(rounds):
            previous = (4218, 4218, 4218)
            for seq in range(chunks):
                body = tuple((71 * user + 113 * turn + 17 * seq + index) % 6561 for index in range(new_codes))
                stream.append(Chunk(f"smoke-{user}", turn, seq, previous + body, seq == chunks - 1, turn))
                previous = body[-3:] if len(body) >= 3 else (previous + body)[-3:]
        streams.append(stream)
    return streams


def load_payloads(path: Path) -> list[list[Chunk]]:
    paths = sorted(path.glob("speech-*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No speech probe JSONL files in {path}")
    events = []
    for source in paths:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "codec_payload":
                events.append(row)
    events.sort(key=lambda row: (row.get("wall_time_ns", 0), row.get("pid", 0), row.get("seq", 0)))
    streams: dict[str, list[Chunk]] = {}
    for row in events:
        request_id = str(row["request_id"])
        codes = tuple(int(code) for code in row["codes"])
        numel = int(row.get("code_flat_numel", len(codes)))
        if numel == 0:
            codes = ()  # Strip the scheduler's control-only placeholder.
        elif numel != len(codes):
            raise ValueError(f"Truncated codec capture for {request_id}: {numel} != {len(codes)}")
        if any(code < 0 or code >= 6561 for code in codes):
            raise ValueError(f"Invalid speech token in {request_id}")
        chunk = Chunk(
            request_id,
            int(row["cache_epoch"]),
            int(row["chunk_seq"]),
            codes,
            bool(row["last_chunk"]),
            row.get("model_turn_id"),
        )
        stream = streams.setdefault(request_id, [])
        previous = stream[-1] if stream else None
        if previous is None or chunk.cache_epoch != previous.cache_epoch:
            if chunk.chunk_seq != 0 or (previous is not None and not previous.last_chunk):
                raise ValueError(f"Capture does not start at a clean cache boundary: {chunk}")
        elif previous.last_chunk or chunk.chunk_seq != previous.chunk_seq + 1:
            raise ValueError(f"Duplicate/reordered codec payload: {chunk}")
        stream.append(chunk)
    if not streams:
        raise ValueError("No codec_payload events; run with the speech probe enabled")
    return list(streams.values())


def compatible_streams(streams: list[list[Chunk]]) -> bool:
    if not streams or not streams[0]:
        return False
    signatures = [
        [(chunk.chunk_seq == 0, len(chunk.codes), chunk.last_chunk) for chunk in stream] for stream in streams
    ]
    return all(signature == signatures[0] for signature in signatures[1:])


def tensor_difference(actual, expected, *, atol: float, rtol: float):
    import torch

    actual = torch.as_tensor(actual).detach()
    # Keep large real-weight caches on their device. Copying every cache to
    # CPU for every comparison adds gigabytes of PCIe traffic per chunk.
    device = actual.device
    actual = actual.to(dtype=torch.float64)
    expected = torch.as_tensor(expected).detach().to(device=device, dtype=torch.float64)
    result = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.shape != expected.shape:
        return {**result, "passed": False, "reason": "shape_mismatch"}
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {**result, "passed": False, "reason": "nonfinite"}
    delta = actual - expected
    norm = float(torch.linalg.vector_norm(expected))
    return {
        **result,
        "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "rms": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta)) / max(norm, 1e-30),
    }


class RandomTape:
    """Diagnostic-only common random numbers; no serving source is patched."""

    def __init__(self):
        self.draws = []

    @contextmanager
    def record(self):
        import torch

        self.draws = []
        originals = {name: getattr(torch, name) for name in ("rand", "randn_like")}

        def capture(name):
            def call(*args, **kwargs):
                value = originals[name](*args, **kwargs)
                self.draws.append((name, value.detach().clone()))
                return value

            return call

        with patch.object(torch, "rand", capture("rand")), patch.object(torch, "randn_like", capture("randn_like")):
            yield

    @staticmethod
    @contextmanager
    def replay(tapes):
        import torch

        counts = {len(tape.draws) for tape in tapes}
        if len(counts) != 1:
            raise ValueError("Different random-call counts between batch rows")
        cursor = 0

        def replay_draw(name):
            def call(*args, **kwargs):
                nonlocal cursor
                if cursor >= len(tapes[0].draws):
                    raise ValueError("Batched forward made an extra random draw")
                values = []
                for tape in tapes:
                    recorded_name, value = tape.draws[cursor]
                    if recorded_name != name:
                        raise ValueError(f"Random call order changed: {recorded_name} != {name}")
                    values.append(value)
                if name == "randn_like":
                    shape, device, dtype = tuple(args[0].shape), args[0].device, args[0].dtype
                else:
                    shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
                    device = torch.device(kwargs.get("device", "cpu"))
                    dtype = kwargs.get("dtype", torch.get_default_dtype())
                value = torch.cat(values, dim=0)
                wrong_device = value.device.type != device.type or (
                    device.index is not None and value.device.index != device.index
                )
                if tuple(value.shape) != shape or wrong_device or value.dtype != dtype:
                    raise ValueError(f"Random tensor contract changed: {tuple(value.shape)} != {shape}")
                cursor += 1
                return value.clone()

            return call

        with (
            patch.object(torch, "rand", replay_draw("rand")),
            patch.object(torch, "randn_like", replay_draw("randn_like")),
        ):
            yield
        if cursor != len(tapes[0].draws):
            raise ValueError("Batched forward skipped an official random draw")


def clone_state(state):
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2WavState

    return BatchedToken2WavState(
        flow_cache={key: value.detach().clone() for key, value in state.flow_cache.items()},
        hift_cache={key: value.detach().clone() for key, value in state.hift_cache.items()},
    )


@contextmanager
def capture_hift(hift):
    """Observe where small flow differences become waveform differences."""
    tensors = {}

    def capture_input(_module, args):
        tensors["mel_input"] = args[0].detach().clone()
        tensors["cached_source"] = args[1].detach().clone()

    def capture_f0(_module, _args, output):
        tensors["f0"] = output.detach().clone()

    def capture_source(_module, _args, output):
        tensors["source"] = output[0].detach().clone()

    def capture_speech(_module, _args, output):
        tensors["raw_speech"] = output[0].detach().clone()

    handles = [
        hift.register_forward_pre_hook(capture_input),
        hift.f0_predictor.register_forward_hook(capture_f0),
        hift.m_source.register_forward_hook(capture_source),
        hift.register_forward_hook(capture_speech),
    ]
    try:
        yield tensors
    finally:
        for handle in handles:
            handle.remove()


def compare_states(actual, expected, *, atol, rtol, n_timesteps):
    result = {}
    for group in ("flow_cache", "hift_cache"):
        left, right = getattr(actual, group), getattr(expected, group)
        if left.keys() != right.keys():
            result[group] = {"passed": False, "reason": "cache_keys_mismatch"}
            continue
        for key in right:
            a, b = left[key], right[key]
            storage_shapes = None
            if group == "flow_cache" and key.startswith("estimator_"):
                # Official solve_euler_chunk allocates 16 timestep slots but
                # only reads/writes [0:n_timesteps]. Our cache is dynamic.
                # Compare every live slot, not the official unused padding.
                if a.shape[0] < n_timesteps or b.shape[0] < n_timesteps:
                    result[f"{group}.{key}"] = {"passed": False, "reason": "missing_live_timestep"}
                    continue
                storage_shapes = [list(a.shape), list(b.shape)]
                a, b = a[:n_timesteps], b[:n_timesteps]
            comparison = tensor_difference(a, b, atol=atol, rtol=rtol)
            if storage_shapes is not None:
                comparison["storage_shapes"] = storage_shapes
                comparison["live_timesteps"] = n_timesteps
            result[f"{group}.{key}"] = comparison
    return result


def run_case(name, streams, official, batched, features, prompt, *, seed, atol, rtol, emit):
    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2WavState

    if not compatible_streams(streams):
        raise ValueError("Case cannot be batched: token/state shapes or turn boundaries differ")
    count = len(streams)
    official_states = singleton_states = batched_states = None
    summaries = {"case": name, "users": count, "steps": 0, "resets": 0, "cache_truncations": 0, "passed": True}
    for step, chunks in enumerate(zip(*streams, strict=True)):
        if chunks[0].chunk_seq == 0:
            official_flow, official_hift = official.set_stream_cache(str(prompt))
            initial = BatchedToken2WavState(official_flow, official_hift)
            official_states = [clone_state(initial) for _ in chunks]
            singleton_states = [batched.setup_batch(features, 1)[0] for _ in chunks]
            batched_states = batched.setup_batch(features, count)
            summaries["resets"] += 1
            for user in range(count):
                for mode, state in (
                    ("singleton_setup", singleton_states[user]),
                    ("batched_setup", batched_states[user]),
                ):
                    diffs = compare_states(state, initial, atol=atol, rtol=rtol, n_timesteps=batched.n_timesteps)
                    passed = all(item["passed"] for item in diffs.values())
                    emit({"case": name, "step": step, "user": user, "mode": mode, "passed": passed, "diff": diffs})
                    summaries["passed"] &= passed
        if not chunks[0].codes:
            # Stage-level control markers do not call either decoder.
            emit({"case": name, "step": step, "mode": "empty_control_marker", "last_chunk": chunks[0].last_chunk})
            continue
        tapes, reference_audio, reference_profiles = [], [], []
        raw_shapes = []
        for user, chunk in enumerate(chunks):
            official.stream_cache = official_states[user].flow_cache
            official.hift_cache_dict = official_states[user].hift_cache
            tape = RandomTape()
            torch.manual_seed(seed + 100000 * user + step)
            inference_chunk = official.flow.inference_chunk
            before_trim = {}

            def capture_flow(*args, **kwargs):
                value = inference_chunk(*args, **kwargs)
                before_trim.update({key: list(tensor.shape) for key, tensor in value[1].items()})
                return value

            with (
                tape.record(),
                patch.object(official.flow, "inference_chunk", capture_flow),
                capture_hift(official.hift) as profile,
            ):
                wave = official.stream(chunk.codes, str(prompt), last_chunk=chunk.last_chunk, return_waveform=True)
            official_states[user] = clone_state(BatchedToken2WavState(official.stream_cache, official.hift_cache_dict))
            for key, shape in before_trim.items():
                if shape != list(official_states[user].flow_cache[key].shape):
                    summaries["cache_truncations"] += 1
            raw_shapes.append(before_trim)
            tapes.append(tape)
            reference_audio.append(torch.as_tensor(wave).reshape(-1))
            reference_profiles.append(profile)
        if count > 1 and step in (0, 15):
            # Isolate the vocoder from upstream Flow drift, then isolate the
            # sine/source path from batched F0-predictor numerical drift.
            mel = torch.cat([profile["mel_input"] for profile in reference_profiles])
            cached_source = torch.cat([profile["cached_source"] for profile in reference_profiles])
            same_f0 = torch.cat([profile["f0"] for profile in reference_profiles])
            for override_f0 in (False, True):
                from contextlib import nullcontext

                control = (
                    patch.object(batched.hift.f0_predictor, "forward", return_value=same_f0)
                    if override_f0
                    else nullcontext()
                )
                with control, RandomTape.replay(tapes), capture_hift(batched.hift) as common_profile:
                    batched.hift(mel, cached_source)
                for user in range(count):
                    diffs = {
                        key: tensor_difference(
                            value[user : user + 1],
                            reference_profiles[user][key],
                            atol=atol,
                            rtol=rtol,
                        )
                        for key, value in common_profile.items()
                    }
                    emit(
                        {
                            "case": name,
                            "step": step,
                            "user": user,
                            "mode": "hift_same_mel_and_f0" if override_f0 else "hift_same_mel",
                            "passed": all(item["passed"] for item in diffs.values()),
                            "diff": diffs,
                        }
                    )
        singleton_audio, singleton_profiles = [], []
        for user, chunk in enumerate(chunks):
            tokens = torch.tensor([chunk.codes], dtype=torch.long, device=features.speech_tokens.device)
            with RandomTape.replay([tapes[user]]), capture_hift(batched.hift) as profile:
                waves, states = batched.decode_batch(
                    tokens, features, [singleton_states[user]], last_chunk=chunk.last_chunk
                )
            singleton_audio.append(waves[0])
            singleton_profiles.append(profile)
            singleton_states[user] = states[0]
        tokens = torch.tensor([chunk.codes for chunk in chunks], dtype=torch.long, device=features.speech_tokens.device)
        with RandomTape.replay(tapes), capture_hift(batched.hift) as batch_profile:
            waves, batched_states = batched.decode_batch(
                tokens, features, batched_states, last_chunk=chunks[0].last_chunk
            )
        for user, chunk in enumerate(chunks):
            for mode, wave, state, expected_wave, expected_state, profile, reference_profile in (
                (
                    "singleton_vs_official",
                    singleton_audio[user],
                    singleton_states[user],
                    reference_audio[user],
                    official_states[user],
                    singleton_profiles[user],
                    reference_profiles[user],
                ),
                (
                    "batch_vs_official",
                    waves[user],
                    batched_states[user],
                    reference_audio[user],
                    official_states[user],
                    {key: value[user : user + 1] for key, value in batch_profile.items()},
                    reference_profiles[user],
                ),
                (
                    "batch_vs_singleton",
                    waves[user],
                    batched_states[user],
                    singleton_audio[user],
                    singleton_states[user],
                    {key: value[user : user + 1] for key, value in batch_profile.items()},
                    singleton_profiles[user],
                ),
            ):
                diffs = compare_states(state, expected_state, atol=atol, rtol=rtol, n_timesteps=batched.n_timesteps)
                diffs["waveform"] = tensor_difference(wave.reshape(-1), expected_wave.reshape(-1), atol=atol, rtol=rtol)
                for key in reference_profile:
                    diffs[f"hift_intermediate.{key}"] = tensor_difference(
                        profile[key],
                        reference_profile[key],
                        atol=atol,
                        rtol=rtol,
                    )
                threshold = batched.hift.m_source.l_sin_gen.voiced_threshold
                diffs["hift_intermediate.f0"]["voicing_disagreements"] = int(
                    ((profile["f0"] > threshold) != (reference_profile["f0"] > threshold)).sum()
                )
                passed = all(item["passed"] for item in diffs.values())
                emit(
                    {
                        "case": name,
                        "step": step,
                        "user": user,
                        "request_id": chunk.request_id,
                        "cache_epoch": chunk.cache_epoch,
                        "chunk_seq": chunk.chunk_seq,
                        "last_chunk": chunk.last_chunk,
                        "codec_tokens": len(chunk.codes),
                        "mode": mode,
                        "passed": passed,
                        "diff": diffs,
                        "official_cache_before_trim": raw_shapes[user],
                        "hift_random_calls": [entry[0] for entry in tapes[user].draws],
                    }
                )
                summaries["passed"] &= passed
        summaries["steps"] += 1
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--assets", type=Path, required=True, help="Directory containing flow.pt/hift.pt/flow.yaml")
    parser.add_argument("--prompt", type=Path, required=True, help="Exact reference WAV used by the captured run")
    parser.add_argument("--payloads", type=Path, help="speech probe directory or one JSONL file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument(
        "--disable-tf32", action="store_true", help="Diagnostic precision control; does not change serving"
    )
    parser.add_argument("--smoke-chunks", type=int, default=12)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"Refuse to overwrite {args.out}")
    for path in (args.prompt, *(args.assets / name for name in ("flow.pt", "hift.pt", "flow.yaml"))):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(key.startswith("CUDA_MPS_") for key in os.environ):
        raise RuntimeError("Refuse inherited MPS settings; use an idle, dedicated GPU")
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(uuid in line for line in processes.splitlines()):
        raise RuntimeError(f"GPU {args.gpu} is occupied; refusing to interfere")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from stepaudio2.token2wav import Token2wav

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True)
    provenance = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu_uuid": uuid,
        "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
        "randomness": "shared flow.rand_noise; captured official HiFT rand/randn_like replayed per batch row",
        "prompt_provenance": "operator-supplied exact reference WAV; capture does not contain its waveform",
        "scope": "real-weight offline numerical diagnostic, not production waveform identity or capacity",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, path in (
        ("official_source", Path(sys.modules[Token2wav.__module__].__file__)),
        ("batched_source", Path(sys.modules[BatchedToken2Wav.__module__].__file__)),
    ):
        provenance[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.out / "setup.json").write_text(json.dumps(provenance, indent=2))
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    official = Token2wav(str(args.assets), float16=args.float16, n_timesteps=args.timesteps)
    batched = BatchedToken2Wav(official).eval()
    summaries = []

    def emit(record):
        with (args.out / "comparisons.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    with torch.inference_mode():
        prompt_tuple = official._prepare_prompt(str(args.prompt))
        official.cache = prompt_tuple
        # Keep prompt feature extraction identical; compare setup_cache and
        # setup_batch computation independently, rather than hiding init drift.
        with patch.object(official, "_prepare_prompt", return_value=prompt_tuple):
            features = batched.prepare_prompt("reference", str(args.prompt))
            cases = [("synthetic_two_users_two_turns", smoke_chunks(chunks=args.smoke_chunks))]
            if args.payloads:
                captured = load_payloads(args.payloads)
                cases.extend((f"captured_singleton_{index}", [stream]) for index, stream in enumerate(captured))
                for index in range(0, len(captured) - 1, 2):
                    pair = captured[index : index + 2]
                    if compatible_streams(pair):
                        cases.append((f"captured_two_users_{index}", pair))
            for name, streams in cases:
                result = run_case(
                    name,
                    streams,
                    official,
                    batched,
                    features,
                    args.prompt,
                    seed=args.seed,
                    atol=args.atol,
                    rtol=args.rtol,
                    emit=emit,
                )
                summaries.append(result)
                print(json.dumps(result), flush=True)
    result = {
        "cases": summaries,
        "passed": all(case["passed"] for case in summaries),
        "long_cache_truncation_exercised": any(case["cache_truncations"] for case in summaries),
        "real_weights_executed": True,
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    if not result["passed"] or not result["long_cache_truncation_exercised"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
