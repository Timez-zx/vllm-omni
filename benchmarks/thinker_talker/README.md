# Native deployment capacity bench (Qwen3-Omni thinker → talker → code2wav)

Multi-user, multi-turn TTFA measurement against the **native vllm-omni
deployment** — no fork, no engine patches. The branch tracks `origin/main`
and every file here is additive.

What "native" means concretely:

```
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config origin_deploy.yaml
```

the upstream multi-stage AR pipeline, upstream shm connectors, upstream
`/v1/video/chat/stream` WebSocket handler. Load is applied by `mu_bench.py`
over that WebSocket: N concurrent users, each running `TURNS` turns, each turn
streaming video frames at a fixed cadence plus an audio question.

Prompt shape is upstream's own and worth knowing before reading any number:
every turn is a **fresh request** carrying `message_history[-2:]` text-only
plus `num_frames` sampled from the frame buffer. The prompt does not grow with
dialog length — it stays around 300–400 tokens no matter how many turns run.

## Run one cell

```bash
setsid bash benchmarks/thinker_talker/run_origin_cell.sh <NAME> <USERS> [SEED]
```

Results land in `/home/ubuntu/data/results/<NAME>/`:

| file | contents |
|---|---|
| `turns.jsonl` | one record per turn: `ttfa_ms`, `ttft_ms`, `wall_s`, `audio_s`, `rtf_deliver`, `max_starve_ms`, `deltas`, `text` |
| `summary.json` | aggregate percentiles |
| `gpu.csv` | `nvidia-smi` sample at 200 ms |
| `engine_slice.log` | the engine's stdout for this cell only |
| `meta.json` | cell parameters |

`run_origin_cell.sh` calls `analyze.py --warmup-turns 2` at the end. Its
pass/fail column comes from a different harness and reads FAIL for this mode —
look at the `ttfa` columns, not `pass`.

Useful env overrides: `TURNS` (default 10), `ORIGIN_DEPLOY` (which yaml),
`RESULTS_DIR`, `MU_STAGGER_S` (user arrival stagger, default `0,40`),
`MU_UNIQUE_INPUTS=1` (make every user/turn's frames *and* audio bytes unique,
so the multimodal encoder cache cannot flatter the result).

## Deploy configs

All four are `vllm_omni/deploy/qwen3_omni_moe.yaml` plus config-only deltas —
fp8 weights + fp8 KV (to fit VRAM) and stage-0 prefix caching.

| yaml | delta | why it exists |
|---|---|---|
| `origin_deploy.yaml` | baseline, stages on GPU 0/1/1 | upstream's own device split |
| `origin_deploy_3gpu.yaml` | code2wav → GPU 2 | talker and code2wav sharing GPU 1 is the 128-user cliff |
| `origin_deploy_3gpu_s1b128.yaml` | + talker `max_num_seqs` 64→128 | the talker concurrency cap is the 160-user cliff |
| `origin_deploy_3gpu_ic2.yaml` | + `initial_codec_chunk_frames` 4→2 | shortens the serial chain to first audio |

## Analysis

- `analyze.py <dir> --warmup-turns 2` — the standard per-cell summary.
- `p99_attribution.py --cells '<glob>'` — splits the TTFA tail into thinker
  time (query → first text) vs speech time (first text → first audio), reports
  which side owns the excess above p95, and counts how many other turns were
  mid-TTFA when each turn arrived. This is the diagnostic for a capacity ladder.
  Defaults to `--warmup-turns 2` so its percentiles line up with `analyze.py`.
- `parse_server_timing.py --log <dir>/engine_slice.log` — the handler's own
  `[TIMING]` lines, i.e. the server-side view of first_text / first_audio.
  Differencing it against `turns.jsonl` isolates admission + transport.

## Environment prerequisites

- `HF_HOME=/home/ubuntu/data/hf-omni` and `HF_HUB_OFFLINE=1` (the runner sets
  both). Without them the engine stalls on Hub requests at boot.
- flashinfer JIT needs `nvcc` and `ninja`: the runner builds a CUDA shim from
  the `cudatk13` conda env and puts the omni env's bin on PATH.
- Engine runs in the `omni` conda env; the bench client runs in `mage`.
