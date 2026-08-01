# Where this stands

## Done and verified

| | |
|---|---|
| Branch `live-agent-web` | upstream `main` + our 26 Qwen commits. **One** merge conflict, and it was not a real one |
| Optimisations survive the merge | all 19 `StreamingVideoSessionConfig` fields, `TALKER_TEXT_ONLY` on, the session roll / orphan recovery / heartbeat intact — checked by import, not assumed |
| Browser client | written; `node --check` clean on all three JS files |
| Page server | renders, injects its config, serves all four assets — verified with a FastAPI TestClient, no GPU |
| `selftest.py` | **14/14 passing**, no GPU |
| `probe.py` | written; drives the whole chain with synthetic media, no browser |

## In progress when this was written

**The Qwen server coming up.** Two real obstacles were hit and fixed on the way,
both of them consequences of moving from vLLM 0.24.0 to 0.26.0:

1. **A 402-second silent phase** that looks exactly like a hang. Qwen3-Omni is 30B
   MoE and FlashInfer JIT-compiles a CUTLASS fused-MoE kernel on first use. GPU at
   0%, log silent, process in `pipe_read` — and `ptxas` at 100% five forks down.
   Cached after the first build. See the README for the one command that tells
   compiling from hung.
2. **Stage 2 OOM, then my own overcorrection.** The measurement config
   (`harness/deploy_pc_stage0.yaml`, 0.74/0.12/0.06) fits on 0.24.0 and does not on
   0.26.0 — `Tried to allocate 1.41 GiB ... 1.04 GiB is free`, after stage 1's CUDA
   graph capture had taken resident memory to 85 GB. FlashInfer's kernels want
   workspace the fractions did not budget, and another user's MPS server holds a
   slice they cannot see.

   Cutting stage 0 to 0.62 to make room **killed stage 0 instead**: its weights are
   59.4 GiB and 0.62 × 94.97 = 58.9, so it reported
   `Available KV cache memory: 0.0 GiB` and died. **Stage 0's fraction is a floor,
   not a preference.**

   The working shape is `deploy_web_demo.yaml`: stage 0 back at **0.74**, and the
   headroom taken from three other places — `enforce_eager` on stages 1 and 2,
   `max_num_seqs` 16→4, and `max_num_batched_tokens` 16384→8192 on the thinker.
   A **new** file rather than a retune of the measurement config, which must stay
   byte-identical or every number in `workflow.md` loses its baseline.

Check where it got to:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8091/health   # want 200
tail -5 /data/zx/results/qwen_live.log
```

## Then, in order

```bash
cd /home/zx/voice-agent/vllm-omni
PYTHONPATH=$PWD python benchmarks/live_agent/web_client/probe.py --direct   # no browser
bash benchmarks/live_agent/web_client/run_page_server.sh                    # then ssh -L 7870
```

## A REAL BUG, found by the probe

**The reply's audio is truncated to about 0.22 s regardless of how long the reply
is.** Established, not suspected:

| text length | audio delivered |
|---|---|
| 34 chars — "Hello! How can I assist you today?" | 0.22 s |
| 104 chars — "One, two, three, … fifteen." | **0.22 s** |

0.22 s at 24 kHz is ~5,280 samples, which is exactly
`initial_codec_chunk_frames: 4` (4 × 1920) minus the one frame the first emit
strips as a CausalConv artifact. **Only the first small codec granule reaches the
client.**

Server-side accounting agrees that more was produced: `audio_chunks=3` per turn,
three `type=audio stage=2` outputs with the third carrying `finish_reason=stop`.
So stage 2 ran three times and only the first delta carried new samples.

Latency itself is fine and repeatable: **first audio 340–376 ms** across five
turns, which is the same order as the 513 ms baseline.

### A/B 1: our talker change is NOT the cause — ruled out

```
TALKER_TEXT_ONLY=1 (default) :  0.22 s
TALKER_TEXT_ONLY=0 (control) :  0.22 s     identical
```

Same probe, same query, same config otherwise. So the truncation is **upstream's
behaviour on this path**, not something the merge introduced. Our change stays on.

That also rules out the obvious reading of `initial_codec_chunk_frames: 4`. A small
FIRST granule is by design; the bug is that **the granules after it never arrive**.
Server-side, `audio_chunks=3` while only one delta carried new samples — so
`_extract_audio_delta_b64` found `audio_data[chunks_drained:]` empty twice, meaning
stage 2's audio tensor list did not grow across its three outputs.

### A/B 2: which delta path

`VLLM_VIDEO_AUDIO_DELTA_MODE` selects between `_delta_fast` (emit only the new
tail) and `_delta_slow` (re-concatenate everything each call). If `slow` delivers
full-length audio, the fast path's `chunks_drained` accounting is wrong on this
branch — a precise and fixable finding.

```bash
tail -8 /data/zx/results/qwen_slow_launch.out
```

**Until this is resolved, do not judge audio quality in the browser** — you would
be listening to a fifth of a second. Everything else about the chain is verified.

## Genuinely unknown

The Qwen numbers — 513 ms TTFA, 40/40 turns with 2 rolls, the 817→33 talker cut —
were **all measured on vLLM 0.24.0**, and this branch runs **0.26.0** with a
different deploy config (fewer sequences, stage 1 eager). **They are not
comparable here until re-measured.** The merge proves the code is present; it
proves nothing about performance. `benchmarks/live_agent/harness/` is
branch-agnostic, so re-running it is how to find out — and `DEPLOY_CONFIG=` on the
launcher can select the original config if the card is free enough to take it.

## The honest regression

The turn trigger is in the browser, not the model. Qwen3-Omni has no listen/speak
token and `should_trigger_turn()` returns `False`, so nothing server-side can
decide you have stopped talking. `selftest.py` asserts that is still true, so if
upstream ever changes it, the test says so.

The MiniCPM arm (`live-agent-minicpm`) has the model deciding, measured at 470 ms
to first audio and **rtf 1.37 — it cannot keep pace with real time**. That is the
arm to compare against, and the difference is real rather than an implementation
shortcut.
