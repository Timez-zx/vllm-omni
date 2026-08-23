# vLLM-Omni Realtime Multi-User Serving Workflow

## Goal and scope

The goal is to serve more continuous audio-video sessions with fewer GPUs while meeting TTFA and playback-continuity targets. The research focus is engine capacity, scheduling, KV cache behavior, and tail latency rather than model quality.

No locally deployable open realtime model currently matches the interaction semantics of Seed Realtime or Gemini Live. This branch approximates that workload with Qwen3-Omni's Thinker → Talker → Code2Wav pipeline: clients continuously upload audio and video, while the model answers in turns. This produces useful multimodal-prefill, speech-decode, and multi-tenant contention, but it is not native full duplex or semantic barge-in.

## Phase 1: application baseline

The current design is stateful at the application and finite-lived at the engine request layer:

```text
continuous media arrival
  → video frames launch silent finite Thinker requests to warm prefix KV
  → end of user speech submits complete canonical history + complete WAV
  → Thinker → Talker → Code2Wav
  → response finishes and the request is destroyed
  → the application commits the turn; the next turn gets a new request ID
```

Key constraints:

- The WebSocket application owns session state, media reception, and complete canonical multimodal history.
- Every silent warm-up and final response is a new ordinary finite request. The engine owns no live request across turns.
- Every turn submits the complete canonical prompt. Prefix/KV cache entries are disposable; a miss adds prefill work but cannot change correctness.
- The application stores processed canonical message blocks. It renders only the new user and assistant blocks, then assembles the full prompt without reprocessing all historical media.
- Video is append-only within a turn. The similarity/freshness filter decides admission; accepted frames are not subjected to an eight-frame sliding window or another sampling pass.
- New frames launch or coalesce into low-priority `video-warm-<uuid>` requests. They run Thinker only with `max_tokens=1`, emit no text, and never enter Talker. A query immediately cancels unfinished warm-ups instead of waiting for cache fill.
- User audio is a single complete WAV at query time, preserving Qwen's whole-audio semantics. Only the final query may produce speech.
- Media arriving during a response belongs to the next turn, and each media item is consumed once.
- A normal response is capped at 256 Thinker tokens. Video is bounded to 640×352, and JPEG work runs in subprocesses.
- The hard history threshold is 49,152 tokens. A 16,384-token headroom normally causes proactive complete-turn compaction near 32,768 tokens to a target below 16,384, rotating the cache lineage.

`enable_audio_arrival_prefill_approximation` is off by default. It seals audio into one-second chunks for silent arrival prefill and exists only to emulate duplex engine load; it is not semantically equivalent to Qwen whole-audio inference.

The old cross-turn persistent request, resumable append, Talker 45k rolling, Thinker shadow compression, and session ledger have been removed. The current baseline follows a general engine request abstraction and can connect naturally to routing, replication, and P/D separation.

## Phase 2: fixed deployment

Formal experiments use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`:

| Stage | GPU | Configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights/KV, prefix cache, priority scheduler |
| Talker | 1 | FP8 weights/KV, session-isolated conditioning prefix cache |
| Code2Wav | 2 | separate process |

Final responses have priority 0 and silent warm-ups priority 10. The YAML remains unchanged across user counts and session-policy controls. `run_qwen_server.sh` locates the CUDA toolkit and checks that Thinker selected FlashInfer. The multimodal processor cache uses API-side `processor_only` mode.

## Phase 3: formal workload

The only formal workload is continuous AV sessions:

- each user keeps one long-lived WebSocket;
- video uploads continuously at 2 FPS, and accepted frames immediately launch or coalesce into silent Thinker warm-ups;
- the microphone uploads PCM16 at 5 Hz, pausing during assistant playback plus a 300 ms echo guard;
- every turn uses a unique real 16 kHz mono SLURP recording followed by 700 ms endpoint silence, with empty query text;
- audio is appended as one complete WAV to the warmed video prefix at query time, and only this request invokes Talker;
- a session keeps one speaker, while DAVIS video uses a fixed sequence with different starting offsets;
- the next turn starts after response playback completes at 1×, producing a playback-paced closed loop;
- users start at deterministic offsets over 0–40 seconds.

A formal cell has 30 turns per user and excludes the first two as warm-up. The ladder runs 8, 16, 32, ... users and stops each seed at the first SLO failure. The engine restarts for every cell. Synthetic media in `probe.py` is only for protocol validation.

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
RESULT_PREFIX=finite_request USERS="8 16 32" SEEDS="7 17" \
TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_av_session_ladder.sh
```

## Phase 4: metrics and pass rule

- **TTFA**: query to the first audio packet. It depends on packet size and is only a within-deployment diagnostic.
- **Audio-ready-500**: time until 500 ms of playable audio has accumulated; this is the packetization-independent startup metric.
- **Stall max**: the largest single playback underflow at 1×.
- **RTF deliver**: generated audio duration divided by delivery duration, measuring sustained supply.

A cell passes only when:

- every post-warm-up turn completes;
- Audio-ready-500 p99 is below 1 second;
- Stall-max p99 is below 50 ms;
- there is no client, protocol, or fatal engine error.

Results must use workload schema 4 and pass `benchmarks/live_agent/analysis/verify_run.py`. The verifier checks unique finite requests, arrival warm-ups, the frame ledger, real prefix-cache hits, and separate stage processes.

## Phase 5: current baseline and conclusions

Exact setup for the latest formal control:

- source: clean commit `854535bb85789a882ff5995362e5528a5f52f83d`;
- deploy: `origin_deploy_3gpu.yaml`, SHA256 `ae7cbeb615b24ee8654cf6c867887b2920324fc81ec93be6aaaa8995c55e4e24`;
- workload: schema 4, seed 7, eight users × 30 turns, two warm-up turns, plan SHA256 `99dc083388931ffcbf9479a7f4344613229350546824371fec3744b548db0ed4`;
- default arm: no `MU_SESSION_CFG_JSON`;
- aligned arm: `MU_SESSION_CFG_JSON='{"context_window_trigger_tokens":32000,"context_window_target_tokens":0,"context_window_compaction_headroom_tokens":0}'`.

| Path | Context p50/p95/p99/max | Audio-ready-500 p50/p95/p99/max |
|---|---:|---:|
| Current default history | 22.5k/32.0k/32.4k/32.8k | 478/723/1186/1215 ms |
| Current `32k -> 0` aligned control | 16.8k/30.6k/31.8k/31.9k | 410/635/796/883 ms |
| Archived persistent path | 14.8k/30.9k/34.5k/36.8k | 515/740/824/900 ms |

Result paths:

- default: `/home/ubuntu/data/results/current_default_854535bb_20260822/current_default_seed7_u8`
- aligned: `/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`
- persistent archive: `/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`

The comparison establishes:

1. **Raw TTFA is not comparable.** The persistent first packet held only about 217 ms of audio, producing raw TTFA p50/p95/p99 of 244/392/479 ms. Recomputing its audio deltas at a fixed cumulative 500 ms gives the table's 515/740/824 ms.
2. **History policy materially affects tail latency.** At the same commit, the aligned control reduced p99 from 1186 to 796 ms. GPU0 SM-active p95 in p99-tail windows was 94.7% for default and 52.4% when aligned. The difference includes context length, retained multimodal content, compaction/cache-lineage churn, and closed-loop trajectory; it is not a token-count-only result.
3. **Finite request lifetime is not the performance problem.** After context alignment, current p99 is 796 ms versus 824 ms for persistent, which is the same performance class. Application-owned sessions, finite engine requests, and disposable prefix/KV reuse are a valid architecture.
4. **Repeated application work is no longer the primary tail source.** Processed canonical blocks remove full-history re-rendering, warm-ups do not block queries, and residual upstream work is cleaned after completion. Default tail contains both Thinker delay and post-first-text pipeline waiting, while Talker and Code2Wav GPUs are not saturated.
5. **`32k -> 0` is not a production policy.** It discards completed turns and exists only to isolate request lifetime and compute envelope. A single-seed live closed loop is also not a bit-identical replay.

The application architecture is suitable as the engine-research baseline. The unresolved application decision is production-grade history compression, not request lifetime.

## Phase 6: audio arrival-prefill research arm

Qwen's audio encoder uses bidirectional attention within an approximately eight-second window, so independently encoding one-second audio chunks changes semantics. This path is off by default and exists only to study arrival-prefill load.

This short A/B used schema 4, seed 7, eight users × six turns, one warm-up turn, a 0–8 second stagger, and plan SHA256 `b6160ec3d78a2126fb07830e5c1b75f9319ba579543555c8474852ad9f9ddf9b`. The baseline had no override; the arrival arm used `MU_SESSION_CFG_JSON='{"enable_audio_arrival_prefill_approximation":true}'`. Result metadata reports dirty source `cbf2226a`, so this supports a directional conclusion only and is not a formal result reproducible from one commit.

Short eight-user × six-turn A/B:

| Input mode | Audio-ready-500 p50/p95/p99 |
|---|---:|
| Query-time complete WAV | 368/475/654 ms |
| One-second audio-arrival approximation | 343/554/692 ms |

Arrival saves only about 25 ms at the median and worsens p95/p99 because additional warm-ups compete with foreground queries. The formal workload therefore keeps complete WAV at query time. Semantically equivalent audio arrival prefill requires a native causal/streaming audio encoder.

Results: `/home/ubuntu/data/results/audio_arrival_ab_baseline_20260822/u8_t6` and `/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u8_t6`.

## Phase 7: Thinker P/D short run

`thinker-talker-pd` keeps the same application protocol and finite requests while splitting Thinker onto GPU 0 for prefill and GPU 1 for decode. Talker and Code2Wav use GPUs 2 and 3. The application still owns session/history, and each engine request ends with its turn.

The implementation fixes three bridge issues: D reuses P's media-position metadata instead of re-encoding media; Talker receives P's saved prompt hidden states; and the background sender snapshots tokens and tensors when enqueueing. P→D uses NIXL, while later stage edges use shared memory.

Strict short runs use schema 4, seed 7, eight users × six turns, one warm-up turn, a 0–8 second stagger, complete query-time WAV, and video arrival prefill. Plan SHA256 is `b6160ec3d78a2126fb07830e5c1b75f9319ba579543555c8474852ad9f9ddf9b`. Each arm has 40 scored turns with no timeout or stall.

| P→D connector | TTFA p50/p95/p99 | TTFT p50/p99 | D-stage p99 |
|---|---:|---:|---:|
| NIXL pull | 4164/13347/19656 ms | 2709/13223 ms | 12.77 s |
| packed cross-layer pull | 4510/7783/9124 ms | 2999/8047 ms | 7.36 s |
| packed cross-layer push, incorrectly serialized | 2146/3700/5034 ms | 671/1188 ms | 0.59 s |
| packed cross-layer push, async pipeline fixed | 770/1083/1228 ms | 576/1081 ms | 0.52 s |

The old ten-second D wait was a connector engineering failure, not an inherent cost of PCIe or P/D:

- GPU 0↔1 is PCIe Gen5 x16; raw PyTorch P2P measures about `50 GiB/s`.
- Pull sustains only about `0.55 GiB/s`; cross-layer packing alone does not improve sustained bandwidth.
- Push issues background WRITE bursts. NIXL telemetry measures about `35 GiB/s` when warm; a 3,820-token prompt transfers exactly `187,957,248 B` in `4.922 ms`.
- Thinker KV costs `48 KiB/token`. D prefix caching is disabled, so every final request still transfers its complete prompt KV; arrival warm-ups run only on P and do not transfer to D.
- D prefix caching cannot simply be enabled: the current connector lacks a delta source offset/cache lineage, so P's complete block table cannot be safely aligned to only D's missing tail. Incremental transfer needs an explicit cache handle and block range.

The first packed-push result still had an application confounder: stage P's local `async_chunk: false` was incorrectly used as the pipeline-wide switch. D therefore waited for complete text before starting Talker, and Code2Wav waited for the complete codec sequence. Its connector bandwidth result remains valid, but its 5.03-second TTFA cannot evaluate P/D.

After the fix, the orchestrator enables async mode when any downstream stage uses chunks while P itself keeps its dedicated KV route. The eight-user short run completed 48/48 turns, scored 40, had no timeout or stall, and passed `verify_run.py`. Engine-side p99 progresses from P first output at 831 ms to D first text at 1044 ms, Talker first codec at 1102 ms, and first audio at 1191 ms. Talker-to-audio is now only 250 ms instead of 3881 ms, confirming Talker and Code2Wav overlap again.

This validation uses the same workload parameters but remains a live closed loop. Its plan SHA256 is `fd9c7288bca42529d9f3117174890853f611b6e0af10b24d9ed78d0fc8e1301d`, not an exact replay of the old run, so the numbers validate the regression fix rather than a strict capacity A/B.

Results: `/home/ubuntu/data/results/pd_finite_short_u8_v14_20260822`, `/home/ubuntu/data/results/pd_crosslayer_u8_t6_stagger8_20260822_v3`, `/home/ubuntu/data/results/pd_crosslayer_push_u8_t6_stagger8_20260822_v2`, and `/home/ubuntu/data/results/pd_async_fix_20260822_v1/pd_async_fix_seed7_u8`. The exact NIXL telemetry smoke is in `/home/ubuntu/data/results/pd_push_telemetry_smoke_u1_20260822_v1`.

### Exact replay and P snapshot optimization

The ladder now passes the canonical `0–8 s` stagger explicitly, and `verify_run.py` rejects any other value. The recorded trace fixes all media arrivals and queries: trace SHA256 `198d956dda10d7f91ed9d932fa3d9abd1d546cb79063accf1dad0d9028fd0109`. Every arm consumes the same 687 frames.

Prefix KV reuse did not make the old P path fully incremental: P rebuilt both complete Talker-conditioning hidden layers on CPU and transported them through the control plane for every finite request. The fix keeps a bounded, disposable snapshot cache in the orchestrator:

- When the actual prefix hit does not exceed the exact lineage parent, P returns only the miss tail and the orchestrator fills the verified prefix from the parent snapshot.
- When a global cache hit extends past the parent, P returns only the `parent→hit` gap plus the miss tail instead of falling back to the complete history. Unverified lineage rows are still never reused.
- Arrival-prefill stores shared tensor chunks without concatenating the complete history. Only a final query materializes one complete tensor for D/Talker.
- The cache defaults to 8 GiB and is configurable through `VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES`.

Exact eight-user replay:

| P snapshot path | TTFA p50/p95/p99 | Stage-0 first-output p50/p95/p99 | Final audio stage p99 |
|---|---:|---:|---:|
| Full snapshot baseline | 760/1275/1326 ms | 362/791/852 ms | 1290 ms |
| Delta P output, eager full concat | 661/1034/1116 ms | 284/582/692 ms | 1080 ms |
| Delta P output, deferred chunk chain | 627/915/988 ms | 232/434/561 ms | 953 ms |

The final arm passes the one-second p99 SLO: 40/40 scored turns, no timeout or stall, and `verify_run.py` passes. The remaining tail is still Thinker-dominated; D's own p99 is about 216 ms, while GPU samples show no sustained saturation. Further work should therefore study P admission/scheduling and delta P→D KV transfer rather than tune Talker or Code2Wav.

Exact replay results: `/home/ubuntu/data/results/pd_async_0_8_record_20260822_v1/pd_async_0_8_record_seed7_u8`, `/home/ubuntu/data/results/pd_snapshot_delta_replay_20260822_v2/u8_t6`, and `/home/ubuntu/data/results/pd_snapshot_chunk_replay_20260822_v1/u8_t6`.

### P-output control-plane optimization

RCA showed that the remaining apparent ingress wait was not P→D KV transfer. P runner work was usually only tens to roughly one hundred milliseconds, but EngineCore emitted about 50–255 MiB of Talker-conditioning hidden states as ZMQ multipart output. Serialization and memory copies occupied CPU/GIL time in the same process, delaying both output handoff and admission of new requests.

Local stage 0 now uses vLLM tensor IPC through a reverse shared-memory queue. ZMQ carries only tensor handles and small metadata; tensor storage is no longer copied through the control plane. This is output-only: using the same mechanism in both directions added an input staging copy and regressed latency, so that experiment was reverted.

In a diagnostic same-trace A/B, P output-ready→orchestrator-receive p50/p95/p99 fell from `46/133/204 ms` to `27/60/94 ms`; P submit→receive fell from `282/576/633 ms` to `238/426/440 ms`. End-to-end exact eight-user replay results are:

| Implementation | TTFA p50/p95/p99 | Scored turns | timeout/stall |
|---|---:|---:|---:|
| Deferred snapshot, ZMQ tensor | 627/915/988 ms | 40/40 | 0/0 |
| Output tensor IPC | 592/849/931 ms | 40/40 | 0/0 |
| Output tensor IPC + relative gap | 577/923/977 ms | 40/40 | 0/0 |

The final arm passes `verify_run.py`; trace SHA256, users, media arrivals, and all 687 consumed frames remain unchanged. The stable conclusion is limited to the approximately halved control-plane handoff tail. End-to-end p99 moves only from the original `988 ms` baseline to `977 ms`, effectively flat, so this does not show that overall tail is solved. Paired turns show the old two slowest turns improving by about 215/254 ms while new concurrency collisions regress other turns by about 112/189 ms. Diagnostics locate the remaining dominant interval before the stage-0 EngineCore scheduler: client send p99 is `7.8 ms` and core request build p99 is `3.8 ms`, but send-complete→core-preprocess-start p99 is `279 ms`; P runner p99 is `152 ms` and output handoff p99 is `94 ms`. Socket receive/deserialize/input-thread admission before scheduler entry is therefore the largest current interval. Results: `/home/ubuntu/data/results/pd_output_shm_replay_20260822_v1/u8_t6_seed7_u8` and `/home/ubuntu/data/results/pd_gap_shm_replay_20260822_v1/u8_t6_seed7_u8`.

### P-input control-plane fix

The added input-thread timestamps identify the pre-scheduler ingress root cause. The old benchmark forced `VLLM_OMNI_SAFE_MM_PROCESSOR_CACHE=1`, downgrading vLLM's mirrored multimodal cache to `processor_only`. Sending the complete token history each turn is correct, but this fallback also put every historical media feature back into the local ZMQ request, making each final input 127–260 MiB.

The fix makes “stage-0 input processor updates sender cache→orchestrator enqueue” one atomic ordered operation, keeping sender and EngineCore receiver LRU access order identical. The benchmark no longer enables the old fallback by default. Semantics, complete history, arrival prefill, and the media ledger are unchanged; a cache hit only omits media tensors already owned by EngineCore.

Diagnostic same-trace A/B:

| Stage-0 input | wire MiB p50/p95/p99 | send-complete→socket-ready p50/p95/p99 | P submit→receive p50/p95/p99 |
|---|---:|---:|---:|
| `processor_only` | 127/236/253 | 96/284/341 ms | 243/515/547 ms |
| Ordered mirrored cache | 0.8/5.7/6.3 | 1.0/5.0/5.9 ms | 148/240/263 ms |

The final eight-user replay without diagnostic logging records TTFT p50/p99 `311/473 ms` and TTFA p50/p95/p99 `476/693/714 ms`. All 40 scored turns succeed, with no timeout/stall, the same 687 consumed frames, a passing `verify_run.py`, and no receiver-cache miss. Relative to the same-trace pre-fix `577/923/977 ms`, TTFA p99 falls by 263 ms. Results: `/home/ubuntu/data/results/pd_ingress_split_replay_20260822_v1/u8_t6_seed7_u8`, `/home/ubuntu/data/results/pd_mirrored_mm_cache_replay_20260822_v1/u8_t6_seed7_u8`, and `/home/ubuntu/data/results/pd_mirrored_mm_cache_clean_replay_20260822_v1/u8_t6_seed7_u8`.

## Phase 8: next steps

1. Use ordered mirrored media cache plus output tensor IPC and relative-gap replay as the eight-user P/D baseline.
2. Continue at 16, 32, ... users until the SLO fails, then attribute the new capacity boundary.
3. Keep input/output control-plane diagnostics available for RCA but disabled in formal capacity runs.
4. Delta-only P→D remains an engine research item: D may retain disposable prefix KV, but the interface must carry a cache handle, lineage version, and explicit block range.

## Recovery map

| Purpose | Path |
|---|---|
| Session and finite-request lifecycle | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Canonical multimodal history | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Prefix-cache observation | `vllm_omni/worker/gpu_model_runner.py` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload plan and media loading | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| Capacity ladder | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| Formal deployment | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| P/D deployment | `benchmarks/thinker_talker/pd_deploy_4gpu.yaml` |
| P/D capacity entry point | `benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh` |
| Run verification | `benchmarks/live_agent/analysis/verify_run.py` |
| GPU sampling | `benchmarks/live_agent/harness/gpu_sampler.py` |
