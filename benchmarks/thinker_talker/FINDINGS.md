# Findings — native deployment capacity, Qwen3-Omni thinker → talker → code2wav

All numbers below come from the scripts in this directory run against the
**unmodified upstream engine** at `0a51a185`. Milliseconds, TTFA (time to first
audio) measured client-side at the WebSocket. First 2 turns of each session
dropped as warmup. `TURNS=10`, `seed=7`, video frames every 480 ms plus a 3 s
audio question per turn, fp8 weights + fp8 KV. Hardware: 4× RTX PRO 6000
(Blackwell), 96 CPU cores.

Every cell ran with `timeout=0` and all client-side probes clean.

## Capacity ladder — upstream default device split (thinker GPU0, talker + code2wav both GPU1)

| users | 4 | 24 | 48 | 64 | 96 | 112 | 128 |
|---|---|---|---|---|---|---|---|
| p50 | 169 | 235 | 324 | 384 | 523 | 599 | 671 |
| p99 | 170 | 391 | 516 | 584 | 769 | **835** | **2539** |

Up to 112 users the tail grows linearly and stays under 1 s. 128 users falls off
a cliff.

**Cause of the 128-user cliff, verified causally.** GPU1 hosts talker *and*
code2wav; it sat above 90 % utilization 48 % of the time while GPU0's thinker
never once reached 90 %. Moving code2wav to a third GPU (`origin_deploy_3gpu.yaml`,
a `devices:` field change and nothing else) took 128-user p99 from **2539 → 740**
and GPU1's >90 % share from 48 % → 9 %.

## Ladder continued on 3 GPUs

| config | users | p50 | p99 |
|---|---|---|---|
| `origin_deploy_3gpu.yaml` | 128 | 514 | 740 |
| `origin_deploy_3gpu.yaml` | 160 | 647 | **3904** |
| `origin_deploy_3gpu_s1b128.yaml` | 160 | 633 | **1032** |

**Cause of the 160-user cliff, verified causally.** Per-stage TTFT decomposition
showed thinker flat while talker p90 jumped 324 → 816 ms: the talker's
`max_num_seqs: 64` admission cap. Raising it to 128 recovered p99 to 1032 ms and
talker p90 to 386 ms.

**Ladder endpoint: 160 users at p99 = 1032 ms on 3 GPUs.**

## Full-history variant (prompt carries the entire multimodal history)

Upstream's own prompt is a sliding window — `message_history[-2:]`, text-only,
plus `num_frames` sampled from the buffer — so the prompt never grows. To compare
against an architecture that keeps full session state, we ran a variant where
every turn carries the whole history (a 15-line env-gated edit to
`serving_video_stream.py`, kept as a patch outside this branch).

| users | 24 | 48 | 96 | 112 |
|---|---|---|---|---|
| p50 | 281 | 362 | 551 | 591 |
| p99 | 473 | 615 | 959 | 1077 |

Cost of full history ≈ +100–150 ms p99 per rung; capacity drops from ~160 users
to ~96–100. `num_tokens_in` grew 297 → 3382 across turns, confirming the
mechanism was actually engaged, with zero preemptions at 96 users × 3.4 k tokens.

## Test-set validity

Synthetic frames and audio can repeat byte-for-byte, letting the multimodal
encoder cache flatter the result. A 96-user cell with `MU_UNIQUE_INPUTS=1` —
every (user, turn) gets unique frame bytes *and* unique audio bytes — cost only
+40 ms (578/999 vs 551/959). The ladder is not a caching artifact.

## The 112-user case — the deadline is missed while the GPUs idle

This is the most informative rung, and the one worth reading first. 112 users
were run **seven times** across configurations, so the numbers below are not one
sample. Reproduce any row with:

```bash
python3 p99_attribution.py --cells '/home/ubuntu/data/results/<cell>'
```

| cell | config | p50 | p99 | thinker p50/p99 | speech p50/p99 | tail share thk/spc | in-flight avg/tail |
|---|---|---|---|---|---|---|---|
| `origin_u112` | 2 GPU, sliding window | 599 | 835 | 122 / 291 | 469 / 653 | 0.53 / 0.50 | 4.3 / 4.5 |
| `originfh_u112` | 3 GPU, full history | 591 | 1068 | 187 / 371 | 403 / 758 | 0.28 / **0.72** | 4.6 / 6.8 |
| `originfhg_u112` | + GPM sampling | 586 | 1087 | 191 / 397 | 398 / 752 | 0.29 / **0.71** | 4.8 / 8.1 |
| `originspy_u112` | + py-spy probes | 593 | 1058 | 195 / 355 | 403 / 799 | 0.19 / **0.80** | 4.5 / 6.1 |
| `originic2_u112` | + codec chunk 4→2 | 499 | 1306 | 195 / 554 | 320 / 701 | 0.43 / 0.54 | 4.1 / 7.9 |

"thinker" is query → first text; "speech" is first text → first audio. "tail
share" is how much of the excess above p95 each side owns. "in-flight" is how
many other users' turns were mid-TTFA at this turn's arrival — average, then
restricted to the tail turns. (p99 here is computed by `p99_attribution.py`;
`analyze.py` interpolates percentiles slightly differently and reports 1077 for
`originfh_u112` against 1068 here. The two agree exactly on n and p50.)

Four things follow:

**The four full-history runs land within 5 % of each other** (1058, 1068, 1087,
1111 by `analyze.py`) despite two of them carrying heavy instrumentation. The
miss is a property of the system at this load, not noise.

**The median is already dominated by the speech side.** Even in the passing
2-GPU cell, first text arrives in 122 ms but first audio takes another 469 ms.
The thinker is not the bottleneck at any percentile here.

**The tail is almost entirely downstream.** Going from sliding-window to
full-history prompts, the thinker p99 grows modestly (291 → 371 ms) but the
speech p99 grows more (653 → 758 ms), and the *share* of tail excess owned by
the speech side jumps from 0.50 to 0.72–0.80. Longer prompts make the thinker
work harder, but what breaks the deadline is the talker → code2wav path
downstream of it.

**Tail turns are the ones that arrive into a crowded engine.** Across the
full-history cells the average turn arrives with ~4.6 other turns mid-TTFA;
tail turns arrive with 6.1–8.1. Nothing is preempted and nothing times out —
they simply queue, at step granularity, behind work that is itself mostly
waiting.

And the counter-test rules out the obvious fix. Halving the first codec chunk
(`origin_deploy_3gpu_ic2.yaml`) produced the **best p50 of any 112-user
configuration** — 499 ms, with the speech median dropping 403 → 320 — and the
**worst p99**, 1306 ms, with tail in-flight rising to 7.9. Shortening the serial
chain helps the typical turn and hurts the unlucky one, because more frequent
chunk handoffs mean more opportunities to queue.

So at 112 users the engine misses a 1 s deadline while, as the next section
shows, three quarters of the machine's compute sits unused.

## The tail is not a resource problem

`nvidia-smi utilization.gpu` reports kernel residency — the fraction of time
any kernel was resident — not how much of the machine that kernel used. It reads
high while the GPU does almost nothing. These are NVML **GPM** hardware
counters instead, sampled 4×/s during `originfhg2_u112`, averaged over the
middle 50 % of the run (162 samples per GPU, ~160 s of full concurrency).

This is the same run that recorded **p99 = 1111 ms** — the deadline miss and the
idle silicon are one measurement, not two experiments stitched together.

| GPU | role | SM util | SM occupancy | tensor pipe | DRAM bw | power | memory |
|---|---|---|---|---|---|---|---|
| 0 | thinker | 30.0 % (max 54.7) | 5.4 % | 4.1 % | 27.2 % (max 51.1) | 188 W | 89.0 GB |
| 1 | talker | 29.5 % (max 57.7) | 3.6 % | 14.6 % | 9.2 % (max 19.7) | 230 W | 60.1 GB |
| 2 | code2wav | 11.3 % (max 36.3) | 4.3 % | 2.6 % | 4.8 % (max 28.4) | 141 W | 6.9 GB |
| 3 | unused | 0 % | 0 % | 0 % | 0 % | 31 W | 0.6 GB |

SM occupancy — resident warps as a fraction of what the SMs could hold — is the
damning one: **3–5 %**. Even during the busiest window the machine is running a
handful of warps per SM. The tensor pipes, the part that actually does the
matrix work, are at 3–15 %. Power sits at roughly a quarter to a third of the
600 W rating. A fourth GPU is completely idle.

Reproduce with the sampler in the scratchpad
(`gpm_sampler2.py`, columns `ts,gpu,sm_util,sm_occ,tensor,fp16,fp32,int,dram_bw,
pcie_tx_MBs,pcie_rx_MBs,power_W,mem_GB`). Caveat: the PCIe counters read 0 on
this SKU and should not be trusted; use `nvidia-smi dmon` for link traffic.

So when TTFA is pushing 1 s, compute, bandwidth, memory and power are all far
from any wall. The ceiling is pipeline serialization, not capacity.

## TTFA structure

TTFA = a **floor** times a **load amplification factor**.

The floor is ~255 ms at 24 users: roughly a dozen serial steps — one prefill
step, ~4–5 thinker tokens, 4 codec frames at ~54 ms each, then code2wav's first
chunk. Each segment amplifies with load by ~1.7× / 1.9× / 2.6× respectively, the
further downstream the more sensitive, because admission happens at step
granularity and bursts queue behind whole steps.

Causal check: halving `initial_codec_chunk_frames` 4 → 2 cut the talker→audio
segment p50 exactly in half (227 → 119 ms) and overall p50 591 → 499 — but p99
got *worse*, 1077 → 1332. **Lowering the floor does not fix the amplification
factor**; more frequent chunk handoffs make burst contention worse. Fixing p99
requires making steps cheaper or changing admission granularity, which means
engine changes.

## Where the time actually goes

py-spy at 200 Hz over a burst window at 112 users. Each of the three main loops
is dominated by a single blocking call:

| process | share of loop | call |
|---|---|---|
| stage0 (thinker) | 53 % | `OmniTensorPrefixCache._coerce_to_cpu_tensor` — a synchronous `.detach().cpu()` of hidden states into the prefix cache |
| stage1 (talker) | 51 % | `stream.synchronize()` in `synchronize_input_prep` — waiting for the payload to land on GPU |
| API server | 48 % of busy time | PIL decode/resize, running on the asyncio event loop alongside 112 WebSocket sessions |

This is why the talker — a much smaller model — has a *higher* per-step cost
than the 30 B thinker (34 ms vs 24 ms ITL): fixed overhead dominates, not FLOPs.
CUDA graphs are captured and in use. stage0 burns 616 % CPU, stage1 136 %.

Two structural issues found by source audit corroborate this:

1. **A skipped scheduler tick per chunk.** When a request consumes a chunk it is
   unconditionally removed from the running queue and parked in
   `WAITING_FOR_CHUNK`, and `load_async` for the *next* chunk only fires at that
   moment — so polling starts one tick after the chunk was consumed, even if the
   upstream stage already wrote it. The request cannot be re-admitted until the
   following scheduler pass, costing a full skipped round plus the engine's 1 ms
   idle sleep on every chunk hop.
   (`chunk_transfer_adapter.py:596-624`, `omni_ar_scheduler.py:258-281`)

2. **No GPU-direct path between stages on the same host.** Hidden states go
   GPU → CPU → `/dev/shm` → CPU → GPU. `SharedMemoryConnector` serializes every
   payload to CPU bytes (`supports_raw_data = False`), and the registered
   alternatives target RDMA NICs; there is no CUDA-IPC connector for the default
   same-node deployment.
   (`shm_connector.py:45`, `connectors/base.py:15-18`)

## Practical summary

- On 3 GPUs the native deployment serves **160 concurrent users at p99 ≈ 1 s**
  with sliding-window prompts, or **~96–100** if every turn carries full history.
- Both capacity cliffs found so far are **configuration**, not code: don't let
  talker and code2wav share a GPU, and raise the talker's `max_num_seqs`.
- Below those cliffs the latency floor is **serialization**, not resources —
  GPUs are ~26 % utilized when the deadline is missed. Reducing it needs work on
  the synchronous hidden-state copy, the blocking talker input wait, the
  event-loop image decode, and the per-chunk scheduler round trip.
