"""Offline, single-user official-model correctness control (not capacity).

Uses the *same decoded/JPEG-encoded* input as continuous_av.py, but bypasses
vLLM, P/D, Talker and wall-clock pacing. No serving implementation is patched.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--units", type=int, default=150)
    parser.add_argument("--offset-s", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--decode-mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--capture-units", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="openbmb/MiniCPM-o-4_5",
    )
    parser.add_argument(
        "--media",
        type=Path,
        default=Path("/home/ubuntu/data/minicpmo-benchmark-assets/assets/omni_duplex1.mp4"),
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=Path("/home/ubuntu/data/minicpmo-benchmark-assets/assets/HT_ref_audio.wav"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"Refuse to overwrite {args.out}")
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(uuid in line for line in active.splitlines()):
        raise RuntimeError(f"GPU {args.gpu} is occupied; refusing to interfere")
    if any(key.startswith("CUDA_MPS_") for key in os.environ):
        raise RuntimeError("Refuse inherited MPS settings")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.out.mkdir(parents=True)

    # Reuse the exact input decoding functions without importing vLLM (the
    # official reference needs Transformers 4.51; the serving env uses 5.x).
    import av
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoProcessor

    media_source = (ROOT / "benchmarks/minicpmo/continuous_av.py").read_text()
    wanted = {"_align_loop_media", "_load_media", "_jpeg_b64"}
    media_ast = ast.parse(media_source)
    media_ast.body = [node for node in media_ast.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    media_namespace = {
        "Path": Path,
        "Image": Image,
        "base64": base64,
        "io": io,
        "av": av,
        "PCM16_SAMPLE_RATE": 16000,
        "PCM16_BYTES_PER_SAMPLE": 2,
    }
    exec(compile(media_ast, "continuous_av.py:media_helpers", "exec"), media_namespace)
    _load_media = media_namespace["_load_media"]
    _align_loop_media = media_namespace["_align_loop_media"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pcm, encoded_frames, duration = _load_media(args.media, frame_max_side=0)
    pcm, encoded_frames, duration = _align_loop_media(pcm, encoded_frames)
    wave = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frames = [Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB") for frame in encoded_frames]
    ref, ref_rate = sf.read(args.ref_audio, dtype="float32")
    if ref.ndim == 2:
        ref = ref.mean(axis=1)
    if ref_rate != 16000:
        from scipy.signal import resample_poly

        ref = resample_poly(ref, 16000, ref_rate).astype(np.float32)
    provenance = {
        **vars(args),
        "out": str(args.out),
        "media": str(args.media),
        "ref_audio": str(args.ref_audio),
        "gpu_uuid": uuid,
        "loop_duration_s": duration,
        "frame_sizes": sorted({frame.size for frame in frames}),
        "media_sha256": hashlib.sha256(args.media.read_bytes()).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "precision": "BF16 weights and KV; torch SDPA; native HF explicit duplex",
        "sliding_window": "off (150 units stay below native 40960 positions)",
        "tts": "disabled; TTS-only initialization skipped; native LLM generation unchanged",
        "caveat": "Diagnostic, not a capacity or bit-equivalence measurement. No serving-side VAD gating.",
    }
    (args.out / "setup.json").write_text(json.dumps(provenance, indent=2))
    print("Loading cached official BF16 model", flush=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    config.init_tts = False
    model = (
        AutoModel.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )
    model.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    # from_existing_model() initializes the waveform decoder even with
    # generate_audio=False. Skip only this unused audio-output constructor.
    model.init_tts = lambda *unused_args, **unused_kwargs: None
    duplex = model.as_duplex(generate_audio=False, device="cuda", sliding_window_mode="off")
    capture_state = {"unit": -1, "phase": "prepare"}
    captures = []

    def capture_forward(module, positional, kwargs, output):
        if capture_state["unit"] >= args.capture_units:
            return
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            return
        captures.append(
            {
                **capture_state,
                "inputs_embeds": embeddings.detach().cpu().clone(),
                "position_ids": kwargs["position_ids"].detach().cpu().clone(),
                "last_hidden": output.hidden_states[-1].detach().cpu().clone(),
            }
        )

    hook = model.llm.register_forward_hook(capture_forward, with_kwargs=True)
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        duplex.prepare(ref_audio=ref)
        for index in range(args.units):
            source_second = (args.offset_s + index) % len(frames)
            capture_state.update(unit=index, phase="prefill")
            prefill = duplex.streaming_prefill(
                audio_waveform=wave[source_second * 16000 : (source_second + 1) * 16000],
                frame_list=[frames[source_second]],
                max_slice_nums=4,
                batch_vision_feed=True,
            )
            if not prefill["success"]:
                raise RuntimeError(f"Native prefill failed at unit {index}: {prefill}")
            before_ids = len(duplex.total_ids)
            capture_state["phase"] = "decode"
            result = duplex.streaming_generate(
                max_new_speak_tokens_per_chunk=20,
                decode_mode=args.decode_mode,
                temperature=0.7,
                top_k=args.top_k,
                top_p=0.8,
                text_repetition_penalty=1.05,
                text_repetition_window_size=512,
            )
            raw_ids = duplex.total_ids[before_ids:]
            record = {
                "unit": index,
                "source_second": source_second,
                "context_tokens": duplex.decoder.get_cache_length(),
                "prefill": prefill,
                "result": {key: value for key, value in result.items() if key != "audio_waveform"},
                "raw_token_ids": raw_ids,
                "raw_tokens": duplex.tokenizer.convert_ids_to_tokens(raw_ids),
                "current_turn_ended": duplex.current_turn_ended,
                "elapsed_s": time.monotonic() - started,
            }
            records.append(record)
            with (args.out / "units.jsonl").open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if result["text"] or index % 10 == 0:
                print(
                    f"unit={index:3d} ctx={record['context_tokens']:5d} listen={result['is_listen']} "
                    f"text={result['text']!r}",
                    flush=True,
                )
            if index + 1 == args.capture_units:
                torch.save(captures, args.out / "native-forwards.pt")
                captures.clear()
    hook.remove()
    if captures:
        torch.save(captures, args.out / "native-forwards.pt")
    (args.out / "transcript.txt").write_text(
        "\n".join(f"{row['unit']:4d} {row['result']['text']}" for row in records if row["result"]["text"])
    )
    (args.out / "complete.json").write_text(
        json.dumps(
            {
                "units": len(records),
                "elapsed_s": time.monotonic() - started,
                "final_context_tokens": duplex.decoder.get_cache_length(),
                "max_gpu_allocated_bytes": torch.accelerator.max_memory_allocated(),
            },
            indent=2,
        )
    )
    print("Completed; GPU memory releases when this process exits", flush=True)


if __name__ == "__main__":
    main()
