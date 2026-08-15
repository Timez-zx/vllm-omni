# What A adds on top of stock vLLM-omni + Qwen3-Omni

This document is a **delta list**. Nothing here describes what vLLM-omni already does; every entry
is something this branch adds, with what stock does, what A does instead, and what it measured. The
chronological build log this replaced is still in git, at `f9406958:workflow.md`.

**Where we started.** Stock vLLM-omni serves Qwen3-Omni as three stages (thinker → talker →
code2wav) and already has a streaming-video WebSocket entrypoint with EVS near-duplicate frame
filtering. What it does well is *a request*: a prompt arrives, tokens come out, the request ends.

**What A has to be.** Dozens of people, each with camera and microphone on, each in a conversation
running tens of turns, each owed **12.5 codec frames per second for as long as they stay connected**.
That is not a bigger version of one request. It needs three things stock does not have:

1. **session semantics** — a conversation is one long-lived thing, not a series of requests;
2. **multi-tenancy** — the walls (KV pools, per-pass budgets) are shared, so every guard has to
   divide by the number of tenants;
3. **a per-frame and per-token cost low enough that N of them fit in real time** — and the binding
   constraint turned out to be scheduling and transport, not FLOPs.

---

## How A is judged

Two numbers, computed by `benchmarks/thinker_talker/analyze.py` from the client's own arrival
timestamps, the first two turns of each session dropped as warm-up:

- **deadline miss %** — of all audio chunks, the fraction that arrived after the audio already
  delivered had finished playing. The playback clock **re-anchors** at each miss, so one late chunk
  is not charged again to every chunk behind it.
- **stall ms per turn** — the silence the listener actually heard, per turn.

**Pass = miss < 1% and stall < 50 ms/turn.** Chunks arriving early are free: generating ahead of
playback is allowed and the client buffers it; only a gap that outruns the buffer counts. Time to
first audio is reported but is *not* the judge — "starts late" and "stutters" are different failures
with different causes, and averaging them together hid the real one for weeks.

---

## The whole delta at a glance

| # | what | stock | A | measured |
|---|---|---|---|---|
| **Session semantics** ||||
| 1 | conversation shape | one request per turn, history re-prefilled | one resumable request per session; a turn appends only what is new | keeping history costs *less* than discarding it |
| 2 | when frames become tokens | at question time | on arrival, while the user is still speaking | the largest single lever on time-to-first-audio |
| 3 | when mic audio becomes tokens | at question time | on arrival in 1 s chunks, last second reserved for the tail splice | encoder is 1 s conv / 8 s attention; 1 s = latency-optimal |
| 4 | conversation length | ends at the context wall | rolls and compresses under the user, invisibly | 40+ turns; unbounded in principle |
| 5 | which frame answers "now" | whatever survived the filter | the newest frame rides the query | freshness is bought with position, not frame rate |
| 6 | a stage that quietly stops | silence in the log | three kinds of silent freeze report themselves | the observed failure was text-then-no-audio, forever |
| **Multi-tenancy** ||||
| 7 | overload | queued | **refused at the door** | queueing a periodic stream is already a violation |
| 8 | shared KV pools | per-request limits only | admission divides each pool by live tenants (thinker and talker ledgers) | at 64 sessions on a 116k talker pool every request was preempted, p99 57 s |
| 9 | everyone compressing at once | n/a (no compression) | deterministic 16-step phase ladder + 2 s global calendar | wave seconds slipped 10.7% vs 2.5% event-free — that is the gap it closes |
| **Cost per frame / per token** ||||
| 10 | frame size | as sent (1280×720 = 880 tokens) | downscaled on arrival to 640×352 = 220 tokens | 4× less prompt per frame |
| 11 | frame dedup bounds | similarity filter, unbounded gaps | `max_gap` forces one through, `min_gap` caps the rate | bounds blindness (static scene) and cost (busy scene) |
| 12 | what the talker receives | the thinker's full payload, pictures included | text only | **25× less** to hold per turn |
| 13 | decode payload | embeddings + hidden row + full output-id list | only the field the talker reads | 17,458 → **8,922 bytes**, constant in turn length instead of quadratic |
| 14 | JPEG decode | on the event loop | 8 prewarmed subprocess workers | the loop stops competing with per-frame decode |
| 15 | chunk transport | create/flock/write/unlink **per chunk** | one persistent two-slot segment per (session, edge) | seqlock + ACK cell; oversized or unacked falls back, so no chunk is lost |
| 16 | sending a chunk | handed to a sender thread | written to the edge at T+0 on the scheduler thread | one less hop per payload |
| 17 | vocoder convolution | full window | windowed | samples **bit-identical**, about half the work |
| **Engine rhythm and deployment** ||||
| 18 | chunk-fed consumer | **parked**, re-admitted a pass later | delivery taken inline on the scheduler thread | 56 sessions: misses **20.1% → 0.9%**, stall 35 ms, flat timeline |
| 19 | code2wav placement | its own process | its own process — we added colocation, measured it, and turned it back off | colocated: GPU1 **50%** vs 96%, 64 users 5.88%/307 ms vs 0.41%/17 ms |
| 20 | talker CPU/GPU overlap | unset on stages 0/1 | explicit: **on** for stage 1, off for stage 0 | halves the stage-1 pass; safe only with #18 |
| 21 | precision | bf16 (nothing set) | FP8 weights + FP8 KV on thinker and talker | thinker 59.4 → 31.7 GiB, KV per token halved, pool ~6× |
| 22 | CUDA graphs | already on for all three stages | same — our earlier shared-card config used eager, and we reverted | eager talker was kernel-active only 29% at 32 users |
| 23 | vocoder chunk | 25 codec frames per call (2 s of audio) | **4** frames (320 ms) | the granularity playback smoothness is made of |
| 24 | sizing (`qwen3_omni_moe.yaml` → `deploy_2gpu.yaml`) | 64 seqs · 0.9/0.6/0.1 · 32768/32768/65536 batched · no prefix caching | **80** seqs · 0.90/**0.30**/0.10 · **16384/8192**/65536 · prefix caching **on** for the thinker | stage-1 pool ~1M tokens, ~50k per session |
| 25 | stage→card mapping | already thinker GPU0, talker+vocoder GPU1 | unchanged | no NVLink here, so only the text payload crosses cards |
| **Per-step data movement** ||||
| 26 | the per-step payload row | pageable `.to(device)`, once per session per step | staged through pinned memory, copy enqueued async | 4 KB at 147 μs → 176 sessions: miss 27.2% → 2.03%, capacity ~168 → ~180 |
| 27 | the code predictor's 15 AR steps | re-forwards the whole 17-position window every step | fixed-size KV cache: one position per step, attention over 17 masked slots | 35.4 → 8.6 ms per pass; 184 sessions: miss 3.6-6.8% → 0.12%, capacity ~180 → ~200 |

---

## Session semantics

### 1. One resumable request per session

**Stock.** Each question is a fresh request carrying the whole conversation, so turn 20 re-prefills
19 turns of pictures and words before it can answer.

**A.** `session_scoped_request` submits the conversation as ONE engine request and feeds each turn as
an incremental streaming update. A turn prefills only its new frames and its new sentence; everything
earlier stays in GPU KV. The stage-0 → stage-1 connector ships the delta instead of re-copying the
whole prompt's embeddings every turn.

**Effect.** The speech stage stops slowing down just because the chat is long. What used to be the
worst case — tens of thousands of tokens of history — becomes *cheaper* than throwing the history
away.

### 2 and 3. Frames and microphone audio become tokens on arrival

**Stock.** Everything the user sent since the last answer is prefilled when the question arrives, on
the critical path to the first sound.

**A.** `prefill_frames_on_arrival` and `prefill_audio_on_arrival` append each retained frame — and
each whole second of speech — to the live request as a **prefill-only** chunk while the user is still
talking. Audio uses `audio_prefill_chunk_s = 1` with `audio_prefill_reserve_s = 1` held back so the
turn's own tail splices correctly. Qwen3-Omni's audio encoder is 1 s convolutional and 8 s attention:
8 s chunks are bit-faithful to a single-shot encode, 1 s chunks are latency-optimal, and this
deployment chooses latency.

**The limit, stated plainly.** An arrival append is refused while the session is answering, while the
question has already been claimed, or while the input queue is non-empty. The reason is ordering:
audio segments come back from the engine carrying no identity, so the only way to know which
submitted chunk a segment belongs to is that they return in submission order. The application keeps a
FIFO of tags and pops one per segment end; interleaving an append into a turn in flight
mis-attributes the *next* segment boundary and tells the client a turn ended when it did not. So
**during an answer, newly arrived frames are not prefilled** — they wait, and the next question pays
for them.

### 4. The conversation does not end

**Stock.** Context grows until the model's limit, then the session is over. At 48 KB per token (48
layers × 2 × 4 KV heads × 128 dims) a 13k context is ~624 MB of KV per session, so with many tenants
the wall arrives much sooner than with one.

**A.** Two mechanisms under one goal — the user never notices:

- **roll** (`session_roll_at_talker_tokens = 45000`): close the engine request and open a fresh one
  seeded with the recent transcript.
- **context compression** (`context_compression_trigger_tokens`, target = half the trigger):
  summarize into a rolling window, warm a **shadow** request on it, then swap the live session onto
  the shadow. `context_compression_carry_frames` decides whether recent frames ride along; A carries
  text only, which keeps seeds cheap at the cost of forgetting what the camera showed.

### 5. The newest frame answers the question

`fresh_frame_on_query` appends the latest arrived frame — similarity-filter-agnostic — at the end of
the query delta. Freshness of the answer is bought by the frame's *position* in the prompt, not by
raising the frame rate: a higher rate costs tokens linearly and still leaves the newest frame buried.

### 6. Silence is a bug report

A stage that crashes leaves a traceback; a stage that quietly stops leaves nothing, and that was the
observed failure — the client gets a turn's text and then no audio, forever. A detects and dumps state
for a stage that stops making progress while requests are tracked, logs the park/continue/wake
lifecycle of a streaming segment (`VLLM_OMNI_LOG_SEG_CYCLES`), and prints a heartbeat that reports
presence of work rather than absence of errors.

---

## Multi-tenancy

### 7 and 8. Admission refuses, and every guard divides by the tenant count

**Stock.** Requests queue. A per-request length limit cannot see a pool shared with 63 other
sessions.

**A.** A session is a standing obligation, so overload is refused at the door rather than degrading
everyone already connected:

- a hard session cap, `VLLM_OMNI_ADMIT_MAX_SESSIONS`;
- a **talker** ledger: refuse when `0.75 × stage1_pool / active < 2048` tokens;
- a **thinker** ledger: refuse when `0.75 × stage0_pool / active < 4096` tokens.

Why the ledgers exist: at 64 sessions on a 116,384-token talker pool, every talker request was
preempted and the recompute storms pushed p99 to 57 s while bandwidth and compute sat idle. The wall
is shared, so the guard must divide.

The same arithmetic sets the **automatic compression trigger**: `0.75 × stage0_pool / cap`. Stock's
anchor — 75% of the model's own context limit — is the wrong one for a multi-tenant deployment: 64
sessions at 75% of a 65,536 limit would want 3.1M tokens out of a 1.13M pool.

### 9. Nobody compresses at the same moment as everybody else

**Stock.** n/a — there is no compression to synchronize.

**A.** Sessions that start together fill up together, cross the trigger together, and their shadow
warm-ups prefill in the same few seconds. Measured inside that window: seconds containing a warm-up
launch carried a **10.7%** slip rate against **2.5%** for event-free seconds. The cliff was
reproducible — starting at t+120 s, peaking at 150 s, back to zero by 210 s, aligned with the first
compression event. Two fixes, both aimed at simultaneity rather than at the work:

- a **deterministic phase ladder**: admission ordinal *i* takes `0.55 + 0.45·(i mod 16)/16` of the
  trigger, downward only. A ±20% random jitter was measured insufficient — growth-rate homogeneity
  re-bunched 108 first compressions into a ~90 s window.
- a **global calendar**: `VLLM_OMNI_COMPRESS_MIN_INTERVAL_S = 2`, capping launches at 0.5/s against a
  steady-state need of ~0.25/s.

A third change here is stability rather than latency: at wave ignition an empty-payload race could
kill the talker's engine core outright. It is guarded now (zero rows built from the talker's text
width).

---

## Cost per frame and per token

### 10 and 11. Frames: smaller, and paced with both bounds

A frame becomes `(W/32)·(H/32)` tokens, so halving each edge cuts its prompt cost 4×. A downscales on
arrival to **640×352** (220 tokens, against 880 for 1280×720) and never upscales. It rides the
upstream similarity filter with two bounds stock does not have: `frame_filter_max_gap` forces a frame
through after N consecutive drops — the filter's metric is a whole-frame MSE on a 64×64 thumbnail,
which barely moves when a small region changes, so a screen share can go minutes retaining nothing —
and `frame_filter_min_gap` caps the retention rate when the scene is busy. `max_frames = 8` bounds
the buffer and evicts the oldest, which is the right one to lose on a live feed.

### 12. The talker stops carrying pictures it never looks at

`VLLM_OMNI_TALKER_TEXT_ONLY` (on by default) drops the visual part of the thinker → talker payload.
The talker's job is text → codec tokens; it never reads the pictures. **25× less** to hold per turn.
This is the one A mechanism that is not a `session.config` field, because it changes what one model
stage hands another — below the level a per-conversation setting can reach.

### 13. The decode payload ships only what is read

On a plain decode step the talker reads `embed.decode` and nothing else. Stock also ships the
hidden-state row (a fixed 4 KB, half the payload) and the full output-id list, which grows through
the turn and made per-turn transfer bytes quadratic. Neither has a reader on the receiving side.
Measured **17,458 → 8,922 bytes**, and constant in turn length. `VLLM_OMNI_T2T_LEAN_DECODE=0`
restores the fat payload for an A/B.

### 14. Media decoding leaves the event loop

JPEG decode and resize move to a prewarmed pool of 8 subprocess workers (`media_pipeline.py`,
`media_worker.py`); the loop rebuilds the image from the worker's bytes with a memcpy. The event loop
is what stamps and delivers audio to every client, so it is the one thread that must never be busy.

### 15 and 16. Chunk transport: a mailbox instead of a filesystem dance

Stock creates, locks, writes and unlinks a POSIX shared-memory segment **per chunk**. A keeps one
persistent segment per (session, edge) with two 256 KB slots addressed by chunk parity, a seqlock so
torn reads are detectable, and a reader-maintained ACK cell for flow control (`VLLM_OMNI_MAILBOX`).
Anything oversized or unacked falls back to the per-chunk path, so a chunk cannot be lost.
`VLLM_OMNI_INLINE_SEND` then writes the outgoing payload at T+0 on the scheduler thread instead of
handing it to a sender thread.

### 17. Windowed vocoder

`VLLM_OMNI_STREAM_VOCODER` computes code2wav's convolution over a window instead of the full context.
Emitted samples are **bit-identical**; about half the work disappears.

---

## Engine rhythm and deployment

### 18. The consumer is no longer parked every other pass

**This is the one that decided capacity.** Stock's chunk-fed stage works like this: the scheduler
sees that the next payload has not arrived, takes the request **out** of `running`, registers an
asynchronous load, and re-admits it a pass later. The request advances **one step per two scheduler
passes** — so a session's frame rate is capped at half the pass rate, however fast the GPU is.

```
cap = 1 ÷ (interval_tax × pass_time)   frames per second,   and 12.5 is required
```

The interval tax is 2 under the park protocol and 1 without it. Measured caps across six cells were
20.8, 12.5 and 9.7 frames/s, and those cells passed, failed and failed — 12.5 fails because meeting
the requirement exactly leaves no margin for a pass that runs long. **In none of the three was the
GPU the binding constraint**, which is why every hypothesis aimed at compute missed.

A takes delivery **synchronously on the scheduler thread**, at the last moment before the request
would have been parked (`_try_inline_receive`). An ownership registry guarantees one fetcher per
request, the flock is non-blocking, and the path is scoped to `model_mode == "ar"`. At 56 sessions:
deadline misses **20.1% → 0.9%**, stall 35 ms/turn, and the slip timeline goes flat.

`VLLM_OMNI_INLINE_RECV_ASYNC` exists because a placeholder guard made inline receive and vLLM's async
scheduling mutually exclusive; the talker runs async here, so without it the mechanism skipped every
single time (measured: 0 hits, 23,553 skips) while looking enabled.

### 19. code2wav stays in its own process — a decision, not an addition

Stock already runs the three stages as separate processes. `VLLM_OMNI_COLOCATE_STAGES` is ours: we
added it to save CPU by hosting stage 2 as a *thread* inside the talker's process, and at 64 sessions
that was measurably wrong. Both stages are Python-driven, so they blocked each other on the
interpreter lock: the talker spent most of its 28.5 ms pass waiting, the vocoder most of its 71 ms,
and GPU1 reported 50% while the client heard 5.9% of its chunks late. The row is here because the
numbers are the useful part — they say *do not colocate these two*.

| | colocated (`=2:1`) | separate processes (default) |
|---|---:|---:|
| GPU1 kernels-resident | 50% | **96%** |
| talker pass | 28.5 ms | 14.7 ms |
| vocoder call | 71.2 ms | 23.0 ms |
| miss / stall at 64 users | 5.88% / 307 ms | **0.41% / 17 ms** |
| TTFA p50 | 1287 ms | 693 ms |

Separate processes pay one real IPC hop per codec chunk, which shows up as a worse TTFA **tail** at
moderate load (56 users: p99 1800 → 2577 ms) while the median improves (698 → 563 ms). At 64 sessions
it is the difference between failing and passing, so colocation is off by default and
`VLLM_OMNI_COLOCATE_STAGES=2:1` is how to reproduce the bad arrangement.

### 20 to 25. Deployment (`benchmarks/thinker_talker/deploy_2gpu.yaml`)

The reference to compare against is stock's own two-card example,
`vllm_omni/deploy/qwen3_omni_moe.yaml`, which already maps the thinker to GPU0 and the speech pair to
GPU1 and already runs CUDA graphs on all three stages. The deltas are these:

- **stage-1 `async_scheduling: true`** — vLLM's own CPU/GPU overlap, which halves the stage-1 pass.
  Safe only together with #18: the park path is exactly what breaks under it, which is why the two
  arrived together. Stage 0 keeps it off, because its arrival-append path reads the −1 sentinel out of
  `token_ids_cpu` and kills the stage.
- **FP8 weights + FP8 KV on all three stages**, quantized at load time from the official bf16
  checkpoint (a community block-FP8 checkpoint produced degenerate text on this fork). Thinker weights
  59.4 → 31.7 GiB, KV per token halved, stage-0 pool about 6× larger. Read the boot log's "GPU KV
  cache size" line for the real number; never trust an estimate.
- **CUDA graphs stay on** — a decision, not an addition. Stock already runs them; our earlier
  single-card configs used `enforce_eager` as a memory measure, which left the talker kernel-active
  only 29% of the time at 32 users. The vocoder's graph wrapper is known to mishandle heterogeneous
  *video* co-batches, and stage 2's input here is audio-only, measured clean for hours.
- **`codec_chunk_frames` 25 → 4.** One vocoder call covers 4 codec frames = 320 ms of audio instead of
  25 frames = 2 s. This is the granularity everything about playback smoothness is made of: with 2 s
  chunks the first sound cannot arrive before 2 s of speech exists, and one late call is a 2 s hole.
- **Sizing.** `max_num_seqs` 64 → **80** on every stage (one slot per session, plus headroom for the
  compression shadow that must coexist with the request it replaces), stage-1 memory fraction
  0.6 → **0.30** (enough for ~1M KV tokens, ~50k per session, since GPU1 shares only with the
  vocoder), per-pass batch budgets 32768 → **16384** on the thinker and 32768 → **8192** on the talker
  (a periodic stage wants short, predictable passes, not maximal ones), and prefix caching **on** for
  the thinker, which a session-scoped request benefits from and stock leaves off.
- **The mapping is stock's.** thinker on GPU0, talker and vocoder on GPU1. The host has no NVLink, so
  the only inter-GPU traffic is the thinker → talker text payload — which is also why prefill/decode
  disaggregation was rejected: the state to move *is* the KV cache.

---

### 26. The per-step payload row goes through pinned memory

**Stock.** The decode payload arrives as a CPU tensor built with `torch.frombuffer` over the
shared-memory segment — pageable memory. `.to(device)` on pageable memory is a *synchronous* copy:
it stages through a driver buffer and orders itself against the stream, so the calling thread
blocks. The talker owes every session a step every 80 ms, so this is paid once per session per step
on the scheduler thread.

**Measured before the change**, 176 audio sessions: **147 μs per session per pass** for a 4 KB row —
28 MB/s against a PCIe path that does 20 GB/s, so essentially all per-call overhead and stream
waiting, none of it transfer. It was the largest single item on the stage's critical thread (32.7% of
its py-spy samples).

**A.** Stage the row through the caching host allocator's pinned memory and enqueue an async H2D.
The allocator is what makes buffer reuse safe: it does not hand a pinned block back until the copies
recorded against it have completed, so sharing buffers across sessions and passes cannot race.
Consumer ordering is the stream's — every later op is enqueued behind the copy.

**Effect** (audio-only, mixed reply lengths, 10 turns per session, first two dropped):

| load | deadline miss | stall/turn | verdict |
|---|---|---|---|
| 168 sessions | 2.92 / 4.67% → **1.75%** | 65 / 87 → 31 ms | passed, now with headroom |
| 176 sessions | 27.2% → **2.03%** | 338 → 45 ms | **failed → passed** |
| 200 sessions | 80.9% → 50.3% | 4052 → 2376 ms | still fails |

Max servable **~168 → ~180 sessions**. Three instruments agree on the attribution: the marginal cost
per session per pass went 745 → 612 μs (the −133 matches the copy's own 147 μs), the line left the
critical thread's top eight, and the thread's busy share fell 58% → 44% while the pass shortened
81.4 → 68.4 ms at the same load.

The session-count gain (+7%) is smaller than the per-pass gain (−16%) because the pass/fail metric is
the deadline-miss **tail**, and the delivery-gap p99 improved less than the median: async copies
enqueue rather than serialize, trading some per-delivery determinism for throughput.

Only the **decode** intake changed. The prefill intake does the same pageable copies on the
turn-opening path and is left alone, so this change's effect stays attributable.

---

### 27. The code predictor keeps a fixed-size KV cache

**Stock.** A codec frame is 16 code groups; the first comes from the talker and the code predictor
produces the other 15 in sequence. Each of those steps re-forwards the **whole** 17-position window
(`seq_len = max_seq`), so at batch 105 padded to 128 a talker pass does 128 × 17 × 15 = 32,640
token-forwards. Measured: **35.4 ms of GPU time per pass**, four times the talker's own forward, and
46.6% of the pass.

**A.** Each step forwards one position and attends over a cache of 17 fixed slots, with the slots
not yet written hidden by an additive mask. Two properties matter together: the arithmetic drops
17× **and** every step keeps the same shape, so one compiled function serves all of them. The second
half is not optional — sizing each step to the prefix it needs (step+1 positions, 40% less
arithmetic) was measured 8% SLOWER, because it loses the single shape.

**Measured**, 184 audio sessions: code-predictor time **35.4 → ~8.6 ms** per pass and now flat in
batch (8.4-9.6 ms from batch 3 to 67, i.e. a fixed per-pass cost rather than a per-session one);
talker pass 76.0 → 34.7 ms; late audio 3.6-6.8% → **0.12%**; stall 90-180 → 1.2 ms/turn; rtf
1.15 → 2.51; TTFA p50 873 → 504 ms. Max servable **~180 → ~200 sessions**.

**Equivalence** is numerical, not bit-exact: the two paths reduce in different orders, so bf16 cannot
match bit-for-bit. `benchmarks/` has no test for this; the check was an offline script that builds
the inner model directly and compares "full window, take row s" against "step per position" at all
17 positions — relative error 0.7-1.3%, the level a single SDPA call already shows (3.3e-3 = bf16's
2⁻⁸). End to end: probes clean, deterministic answers identical, and the produced audio's RMS, peak
and zero-crossing rate all in the normal speech range. `VLLM_OMNI_CP_KV_CACHE=0` restores the
re-prefill path for comparison; `VLLM_OMNI_CP_KV_VERIFY=1` runs both paths and logs the difference
(it self-disables under CUDA graph capture, where its device sync is illegal).

---

## The defaults are this configuration

Until 2026-08-14 every mechanism above defaulted OFF for upstream compatibility, and the measured
configuration lived in a benchmark script — so the server and the thing being measured were two
different systems. On this branch **a client that connects and sends only a system prompt gets A**:

`session_scoped_request` on · `prefill_frames_on_arrival` on · `prefill_audio_on_arrival` on ·
`audio_prefill_chunk_s` 1 s · `audio_prefill_reserve_s` 1 s · `max_frames` 8 ·
`max_frame_width/_height` 640×352 · `frame_jpeg_quality` 90 · `enable_frame_filter` on (0.95) ·
`frame_filter_min_gap` 0 / `max_gap` 4 · `fresh_frame_on_query` on ·
`session_roll_at_talker_tokens` 45,000 · `session_roll_history_turns` 8 ·
`context_compression_carry_frames` off · `context_compression_target_tokens` half the trigger ·
`context_compression_warmup_timeout_s` 30 s

The engine-side mechanisms default ON through `vllm_omni/core/sched/runtime_flags.py`:
`VLLM_OMNI_INLINE_RECV`, `..._RECV_ASYNC`, `VLLM_OMNI_INLINE_SEND`, `VLLM_OMNI_MAILBOX`,
`VLLM_OMNI_STREAM_VOCODER`, `VLLM_OMNI_T2T_LEAN_DECODE`. *Unset* resolves to the ON default, so a
banner printing the raw environment shows 0 for a flag that is on. Setting one to `0` is how an
ablation is built — one mechanism off, never the baseline switched on.

Three numbers are properties of the **machine**, not of a conversation, so they come from the
environment at launch (`run_engine.sh` passes them): `VLLM_OMNI_STAGE0_KV_POOL_TOKENS` and
`VLLM_OMNI_STAGE1_KV_POOL_TOKENS` from the boot log's "GPU KV cache size" lines, and
`VLLM_OMNI_ADMIT_MAX_SESSIONS`. They arm both admission ledgers and set the automatic compression
trigger. The resolved values appear once per session in the `[session] context compression armed:`
log line.

---

## Running and measuring it

```bash
cd benchmarks/thinker_talker
bash run_engine.sh                  # boot (~3-4 min), one warm-up turn
bash run_cell.sh MyCell 56          # 56 sessions x 6 turns, everything recorded
bash run_engine.sh stop
```

`run_cell.sh` writes `turns.jsonl`, `metrics.json`, `meta.json` (every `VLLM_OMNI_*` in the
environment at launch), `gpu.csv` (both cards at 10 Hz, timestamped), `gpu_pmon.txt`
(per-**process** sm%, because GPU1 hosts two stages and the aggregate number cannot say which one
holds the card), and `engine_slice.log`. Then `analyze.py` for the two numbers,
`sched_steps.py --stage 1` for the inter-pass interval that feeds the capacity inequality above, and
`verify_cell.py` to check that the mechanisms claiming to be on left a trace in the log.

That last one exists because of two failures that are invisible in client metrics. **A treatment that
never ran**: the first inline-receive cell measured `irecv=0/0` — the code sat in a branch this
deployment does not take — and reproduced the baseline exactly, which reads as "the idea does not
work". **Configuration drift between compared runs**: five times, two runs differed in more than the
thing under test, each worth about 2× on the headline number, one of them reversing a conclusion. And
one on the client side: think-time must start when the previous answer finishes **playing**, not when
its last byte arrives (`MU_PLAYBACK_PACED=1`), or a faster engine gets its next question sooner and
silently runs at lower concurrency — one run measured 28 concurrent sessions against another's 49 with
both labelled the same N.

### Pass criteria

```
first audio   TTFA p99 < 1000 ms      measured from the end of user speech
stutter       seam silence p99 < 50 ms   played on arrival, no client prebuffer
```

**How the stutter number is built.** Simulate playback: when the previous chunk finishes and the next
one has not arrived, the speaker is silent — record how long. Every seam contributes one number (0
when it is not late) and the p99 is taken over all turns. The input is `deltas` in `turns.jsonl`
(each chunk's arrival time and sample count).

Not "fraction of late chunks": lateness there is binary — 1 ms and 3 s count the same — and the
denominator moves with chunk size. Measured: of the 0.6% late chunks at 200 sessions, **77% were
shorter than 50 ms and inaudible**.

**Where 1000 ms comes from.** Gaps past ~700 ms are heard as hesitation
([Kendrick & Torreira 2015](https://doi.org/10.1080/0163853X.2014.955997)); 1 s is the limit for
uninterrupted flow of thought ([Miller 1968](https://doi.org/10.1145/1476589.1476628)). The modal
gap between turns in human conversation is near 0
([Stivers et al. 2009](https://doi.org/10.1073/pnas.0903616106)). The looser of the two is taken.

That budget is meant to **include endpointing** (deciding the user has stopped), which `t_q` does
not — a real product spends another 200–700 ms there. A full-duplex architecture is what saves it.

**Where 50 ms comes from.** A stop closure (p/t/k) in natural speech is already 50–100 ms of
near-silence, so a shorter seam is masked by the speech itself; VoIP loss concealment is likewise
transparent under 30 ms and clearly audible past 80 ms. Moving the threshold anywhere in 30–80 ms
flips none of the verdicts below.

**Capacity curve** (current HEAD, `deploy_2gpu_seq256.yaml`, audio load, 10 turns less 2 warmup):

| requested | admitted | TTFA p50 | TTFA p99 | stutter p99 | rtf |
|---:|---:|---:|---:|---:|---:|
| 200 | 187 | 369 ms | **578 ms PASS** | **0 ms PASS** | 3.27 |
| 240 | 223 | 542 ms | 1570 ms FAIL | **0 ms PASS** | 1.90 |
| 280 | 258 | 970 ms | 2344 ms FAIL | 146 ms FAIL | 1.11 |
| 300 | 275 | 2005 ms | 3886 ms FAIL | 199 ms FAIL | 1.01 |

**First audio breaks first, between 200 and 240; the stutter knee is between 240 and 280.** At 240
sessions the stutter p99 is still 0 ms — seamless — while TTFA p99 is already 1570 ms. **Capacity is
set by first audio, so the next target is the thinker on GPU0, not GPU1.**

### The load generator does not compress its websocket

`mu_bench.py` connects with `compression=None`. The `websockets` library offers permessage-deflate by
default and the server accepts it, so before this every audio delta was zlib-compressed on the way
out and inflated on the way in. py-spy on the API server at 200 sessions put **42% of its event loop
in `permessage_deflate.encode`** — audio PCM, which barely compresses. A/B at 200 sessions, same
seed, back to back:

| | TTFA p50 | TTFA p99 | deadline miss | API server's busiest thread |
|---|---|---|---|---|
| deflate on | 645 ms | 1654 ms | 0.44% | 65% of a core |
| deflate off | 558 ms | **728 ms** | 0.48% | **34% of a core** |

The three stage processes did not move (99 / 67-68 / 81%), so the change is confined to the API
server. Two further signatures confirm the mechanism rather than a coincidence: at fixed concurrency
(190-200 sessions) the growth of question→first-text across a session's turns went from 106→396 ms to
a flat 69→88 ms, and the control arm reproduced an earlier run to within 6% on p99.

This is a **serial** cost on one event loop shared by every session, which is why it showed up as a
tail rather than as saturation — the loop averaged 65% of a core while carrying a 1.6 s p99. A high
tail with a mid-range average CPU is the shape to look for; the first pass at this dismissed the API
server precisely because 58% "was not saturated".

**The server is deliberately left alone.** A real browser negotiates the extension and should keep
saving the bandwidth (36% on a real reply, measured). Only the load generator opts out, so the
capacity number is not spent on compression the engine never asked for. `MU_WS_DEFLATE=1` puts it
back, which is how the A/B was run.

With it off at 200 sessions the TTFA p50 of 558 ms decomposes as **75 ms** question→thinker's first
text (the whole web-inbound path plus the thinker), **304 ms** talker turning that text into its first
codec chunk, **179 ms** code2wav plus orchestration plus delivery. The first chunk is fixed at 4 codec
frames = 320 ms of speech (`initial_codec_chunk_frames`), so more than half of TTFA is the talker
filling that first chunk — that, not the web layer, is what a lower TTFA has to attack next.

---

## Where it stands

6 turns per session, compression active, both cards FP8:

| load | deadline miss | stall/turn | TTFA p50 | TTFA p99 | GPU kernels-resident |
|---|---:|---:|---:|---:|---:|
| 56 sessions | 1.17% | 85 ms | 833 ms | 2271 ms | 82-85% |
| 64 sessions | 6.15% | 364 ms | 997 ms | 3591 ms | 82-85% |
| 64 sessions, arrivals spread out | **0.07%** | **3.5 ms** | 346 ms | 776 ms | — |

The third row is the point: the same 64 sessions doing the same total work pass comfortably when their
compression waves do not coincide. What remains at 64 is **simultaneity**, not capacity.

**Measured and rejected** — recorded so they are not retried blind:

- batching or clock-coalescing the thinker → talker text: no change to the pass interval once the park
  was gone;
- encoder disaggregation: the vision encoder is **10.7%** of the thinker's GPU forward time (14.3 ms
  p50 on the 15% of passes that carry it), which cannot pay for a PCIe hop;
- prefill/decode disaggregation: the state to move is the KV cache, and this host has no NVLink.

### Where the audio limit is now

**After #27 the limit moved off the talker.** At 200 sessions: late audio 0.55%, stall 7.8 ms/turn,
TTFA p99 1747 ms, no admission refusals — passing. At 240: late audio still only 1.00%, but TTFA p99
is 3409 ms (fails) and the thinker's pool guard refuses 33 sessions, because
`0.75 × 1,132,672 / 240 = 3540` is below the 4096-token admission floor. So the audio workload is now
bounded by the THINKER — its TTFA tail, and a hard ceiling of about 207 sessions from the stage-0 KV
pool — not by the speech pair.

The decomposition below is the pre-#27 state, kept because it is what identified the code predictor:

184 sessions, batch 105, a 76.0 ms pass (`VLLM_OMNI_LOG_MTP_GPU=1`):

| part | per pass | share | per session |
|---|---:|---:|---:|
| **code predictor forward (GPU)** | **35.41 ms** | **46.6%** | **339 μs** |
| talker model forward (GPU) | 8.81 ms | 11.6% | 84 μs |
| critical thread (CPU) | 33.99 ms | 44.7% | 325 μs |
| sum | 78.2 ms | vs a measured 76.0 ms pass — 3% apart, so nothing large is unaccounted | |

The limit is now **GPU**, and it is the **code predictor**, not the talker: a 5-layer, hidden-1024
model costing 4× the talker's own forward per session. Three reasons, from the config and the code:

- `num_code_groups=16`, so **15 sequential AR steps per codec frame** — structural;
- **no KV cache**: each step re-forwards the whole growing sequence (length 2→17), which is 15,960
  token-forwards per pass where a cache needs 1,680 — **9.5× redundant**. The wrapper's docstring
  calls the extra O(T²) negligible for short sequences; that holds at batch 1, not at batch 105 and
  13 passes/s;
- `use_cuda_graphs=current_omni_platform.is_npu()`, i.e. **eager on CUDA**: 2176 GFLOP in 35.4 ms is
  61.5 TFLOPS, roughly 20% of this card.

Ruled out with evidence: the per-row fallback that runs one code-predictor forward per session
requires a request seed, and this deploy sets none.

So about nine tenths of that GPU time is work that does not have to happen: with a cache (the
sequence is 2→17 and the batch is known, so it is a fixed-shape buffer, ~44 MB) and graphs, the same
arithmetic should land near 4-8 ms instead of 35.

**Open:** the 352 ms thinker phase of TTFA is not decomposed — that needs a request-id ↔ turn mapping
the logs do not carry — and the TTFA p99 tail is not fully attributed: compression is the suspect and
the isolating run, same load with compression disabled, has not been completed.

### Above that, the wall at 300 sessions is GPU1

With websocket compression off, 200 sessions have room to spare: TTFA p99 **728 ms** and 0.48% late
audio. (Both are the old metrics; the criteria were later replaced by TTFA p99 < 1000 ms and stutter
p99 < 50 ms — see "Pass criteria" above.) 300 sessions collapse: rtf **0.83** (a second of speech takes
1.19 s to produce), 78.6% late, 3128 ms of stall per turn, 25 clients dropped. Aggregate throughput
rises only from 82 to 94 audio-seconds per wall-second.

**Not admission.** With `stage0_admission_floor_tokens` at 2048 and the pool-share factor at 0.8, all
300 sessions are admitted. A session's measured thinker context is about 1000 tokens, and 200 of them
use **18%** of the stage-0 pool — the 4096-token floor is 4× the real usage.

**Not memory.** GPU0's 88 GiB is reserved once at boot by `gpu_memory_utilization` and does not track
session count; GPU1 uses 47 of 95 GiB.

**GPU1's compute.** The talker's pass time grows linearly with concurrency (fit over 91 batch points):

```
pass = 6.1 ms + 529 us x concurrently speaking sessions
```

A pass advances every session in the batch by one codec frame = 80 ms of speech, so a pass must
finish inside 80 ms: a ceiling near **139 simultaneous speakers**. At 300 sessions the peak is 176
(a 104 ms pass, slower than real time); at 200 sessions it is 74.

Those 529 us are spent WAITING on the GPU, not on CPU work. Timing the same region inside
`_preprocess` twice — once on the wall clock, once on the CUDA timeline, with the events read 8
passes late so the probe never synchronizes (the probe is local, not committed):

| concurrency | wall | CUDA timeline |
|---|---:|---:|
| 61–110 | 8.99 ms | 31.45 ms |
| 151–190 | 29.98 ms | 51.39 ms |

The GPU timeline spans LONGER than the wall clock, which can only mean the CPU queues work faster
than the GPU retires it.

**The biggest consumer of GPU1 is code2wav, not the talker.** Per-process occupancy (`pmon`, 300
sessions): code2wav **54%**, talker **28%**. Moving code2wav to GPU0 so the talker owns GPU1 drops
that same region's GPU timeline from 31.45 to **16.04 ms** at matched concurrency (61–110), of which
the MTP forward itself is 9.55 ms — the half that disappears is the time code2wav was holding the
card.

Moving it is **not** a fix, though: GPU0 goes to 94% and GPU1 idles at 38%, and late audio rises from
62.3% to 89.2%. Either placement saturates one card first (68%/88% against 94%/38%). (That run has a
confound — the thinker's `gpu_memory_utilization` was cut from 0.90 to 0.78 to make room. The GPU
timeline measurement is unaffected: matched concurrency, same stream.)

**The next thing to look at is code2wav** — the largest consumer on the GPU, and never profiled;
every profile so far pointed at stage 1. (Profiled; see the section below.)

One measurement note, recorded so it is not repeated: **high CPU utilization does not mean the thread
is working.** CUDA synchronization spins by default, so "main thread at 78% CPU" and "the stack is
parked on this line" read identically whether the thread computes or waits, and a sampling profiler
sees the line that QUEUED the GPU work while the wait lands on the next synchronizing call.
Optimizing the talker's CPU path line by line against that reading took three rounds and returned
4–5%; all of it was reverted. Separating "computing" from "waiting" requires paired CPU/GPU timing.

### code2wav: the cause of the stutter, and a third of it removed

**It is the cause.** Short-circuiting the vocoder's compute entirely
(`VLLM_OMNI_CW_BYPASS=1`, which ships silence of exactly the length the real decode would have
produced, so chunk count, chunk duration and arrival timing are unchanged and only the samples are
garbage), at 300 sessions:

| | baseline | code2wav bypassed |
|---|---:|---:|
| late chunks | 75.4% | **4.27%** |
| stall per turn | 2826 ms | **93.6 ms** |
| rtf | 0.84 (behind) | **1.22 (ahead)** |

Both runs admitted 275 sessions and measured 2200 turns; one variable apart.

**Its cost is a straight line.** Measured offline with both cards idle, real weights, bf16, and the
production shape (25 frames of left context + 4 new): past batch 8 every additional
simultaneously-speaking session costs a flat **+1.4 ms of GPU**, and a bigger batch amortizes
nothing. That is **4.24 ms of GPU per second of speech**, so a dedicated card sustains **236
simultaneous streams**.

**The time is not in the convolutions, it is in memory traffic.** It reaches **14%** of this card's
measured bf16 peak (57 of 411 TFLOP/s). The cause is the Snake activation `x + (1/β)·sin²(αx)`:
transformers evaluates it as five separate kernels (mul, sin, square, mul, add), each reading and
writing the whole tensor. The decoder's last stage carries `[B, 96, 28800]` — 177 MB at batch 32 —
so five steps are eleven passes over HBM where a fused kernel needs two. The activation does 2.5
flops per byte moved on a card whose break-even is ~200, so it is purely bandwidth-bound: cutting
traffic is the only lever, and extra arithmetic is nearly free.

**The fused kernel already existed in this repo, unconnected.** `SnakeBeta` in
`common/snake_activation.py` carries a Triton implementation that reads once, writes once, and keeps
the intermediates in fp32 registers. Qwen3-Omni's code2wav has 29 Snake activations, of which **28**
come from transformers' `Qwen3OmniMoeCode2WavDecoderBlock` and only the last one was the fused
module. They are swapped at build time; parameter names and shapes are identical, so weight loading
is untouched, and the boot line "Precomputed exp caches for N SnakeBeta activations" goes from 1 to
29, which is the check that it took effect.

**Offline this saturates the available win.** At batch 80 a call goes **108.7 → 72.0 ms**; deleting
the activation outright costs **71.99 ms**, so the fused version is as fast as not doing the work at
all and there is nothing left on this axis. Card capacity goes **236 → 356 streams**.

**Accuracy improves rather than degrades.** Against the same model in fp32, the transformers path
scores 32.36 dB SNR and the fused path **32.73 dB** — the five-kernel chain rounds to bf16 at every
intermediate, the fused one rounds once on store. Note also that bf16 itself is only ~32 dB against
fp32 through this decoder, so the difference between the two bf16 paths is bf16's own noise, not an
artifact of fusing.

**At 300 sessions:**

| | baseline | fused Snake | bypassed (ceiling) |
|---|---:|---:|---:|
| late chunks | 75.4% | **40.4%** | 4.27% |
| stall per turn | 2826 ms | **1077 ms** | 93.6 ms |
| rtf | 0.84 | **1.01** | 1.22 |
| TTFA p99 | 4628 ms | 3886 ms | 3253 ms |
| code2wav's share of GPU1 | 52.8% | **39.0%** | 10.5% |

That is half the achievable gain: (75.4 − 40.4) / (75.4 − 4.27) = 49%. The freed card went to the
talker (34.5% → 39.5%).

**300 sessions still fail, and not because of code2wav.** Read the ceiling column: even with the
vocoder's compute deleted, TTFA p99 is **3253 ms** against a 1000 ms criterion. First audio happens
before any audio is being generated, so GPU1's compute cannot reach it. **Further code2wav work can
only improve the stutter, never the TTFA tail.**

**Two levers remain**, both stutter-side only: raising `codec_chunk_frames` from 4 to 8 turns the
convolution window from "15 frames to emit 4" into "19 frames to emit 8", 1.58× less work, config
only, and the first chunk is a separate knob so TTFA is unaffected; below that, per-layer streaming
convolution state would take the window from 15 frames to 4, 3.75× less, but requires per-session
state for ~30 conv layers and reworking both the batching and the CUDA-graph path.

One measurement note: **profiling this offline requires gradients disabled.** The fused path is
gated on `not torch.is_grad_enabled()`, so with grad on the Triton kernel is never reached, and
autograd additionally retains every intermediate at 1920× upsampled resolution, which exhausts the
card. Production runs under `inference_mode`; offline measurement has to match.

---

## Where the code is

| what | where |
|---|---|
| session layer, admission, compression, arrival prefill | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| frame filter gap bounds | `vllm_omni/entrypoints/openai/video_frame_filter.py` |
| media decode workers | `vllm_omni/entrypoints/openai/media_pipeline.py`, `media_worker.py` |
| inline receive / inline send | `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` |
| persistent mailbox | `vllm_omni/distributed/omni_connectors/connectors/shm_mailbox_connector.py` |
| engine-side flags | `vllm_omni/core/sched/runtime_flags.py` |
| lean decode payload, talker text-only | `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py` |
| windowed vocoder | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py` |
| probes (`[SCHED-STEP]`, `[STEP-GPU]`, `[MTP-GPU]`, `[ENC-GPU]`, `[PF]`, `[SEG-CYCLE]`) | `omni_ar_scheduler.py`, `gpu_model_runner.py`, `gpu_ar_model_runner.py` |
| deployment, runner, metrics | `benchmarks/thinker_talker/` |
| client and multi-user driver | `benchmarks/live_agent/web_client/` |
