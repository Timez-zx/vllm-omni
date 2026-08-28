# DuplexOmni capacity baseline

## Setup

- Hardware: 3 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB.
- Deployment: FP8 Thinker on GPU 0, FP8 Talker/MTP on GPU 1, and BF16
  Code2Wav on GPU 2.
- Input: one 480 ms PCM slice and one video frame per slot; each session has a
  reproducible random phase in `[0, 480 ms)`.
- Session path: the server owns history and submits one finite request per
  slot. Thinker slot `t+1` may overlap Talker/Code2Wav slot `t`; Talker remains
  ordered within a session.
- Context: append-only prefix lineage with compaction at 6,144 Thinker tokens,
  retaining the system prompt and latest complete slot.
- Capacity points: warmed server, seed 8001, 60 slots per user, and distinct
  media/cache lineages across users.

A slot misses the strict realtime deadline when its wall-clock E2E latency is
over 480 ms. A run is throughput-unstable when median application queueing in
the final third grows by more than one 480 ms slot relative to the first third.

## Current capacity

| Users | E2E p50/p99 | Request p50/p99 | Thinker p50/p99 | Queue p99 | Miss rate | Queue growth | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 373/462 ms | 368/457 ms | 282/354 ms | 0.46 ms | 0% | 0 ms | strict realtime |
| 2 | 444/570 ms | 437/555 ms | 343/446 ms | 26.9 ms | 22.5% | 0 ms | throughput-stable, SLO failure |
| 3 | 1,067/1,758 ms | 639/836 ms | 466/622 ms | 1,202 ms | 100% | 569 ms | throughput collapse |

Strict capacity is one user. Two users still sustain the arrival rate but miss
the per-slot deadline. Three users are the capacity knee because backlog grows
continuously.

Unlike the query-driven Qwen workload, DuplexOmni decodes about 24 Thinker
tokens every 480 ms, or roughly 50 Thinker tokens/s/user, and runs Talker for
every slot. Continuous decode is therefore the first-order capacity cost.

## Prefill-interference validation

The causal experiment keeps eight full Duplex probe sessions at 12 slots each
and adds 0/4/8/16 AV sessions that materialize Thinker KV but generate no token
and never enter Talker. It was repeated twice.

| Prefill-only users | Thinker p99, run 1 | Thinker p99, run 2 |
| ---: | ---: | ---: |
| 0 | 879 ms | 851 ms |
| 4 | 1,123 ms | 1,059 ms |
| 8 | 1,393 ms | 1,435 ms |
| 16 | 2,168 ms | 2,720 ms |

With 16 prefill-only sessions:

- Decode gaps without concurrent prefill have a 16--17 ms p99; gaps exposed
  to prefill have a 315--316 ms p99.
- Decode-only GPU batches have an approximately 8.4 ms p50; mixed
  prefill/decode batches have an approximately 52.7 ms p50.
- Running the same prefills to completion before starting the probes yields an
  820 ms Thinker p99. Only concurrent execution raises it to 2.17--2.72 s.

Concurrent small AV prefills therefore stretch continuous decode cadence and
amplify the tail. The effect is not hidden background decode and cannot be
explained only by the total amount of prefill work.

## Long-session check

A warmed one-user 300-slot AV run completed all 300 codec/EOS turns with no
deadline miss. E2E p50/p95/p99/max was `376/442/462/463 ms`; request p99/max
was `456/459 ms`, and application-queue p99 was below `0.5 ms`. Eight context
compactions occurred without a visible latency spike.

## Conclusion

- Thinker is the first capacity limit; Talker and Code2Wav are not the primary
  bottleneck.
- Continuous decode explains the main capacity difference from query-driven
  workloads; concurrent AV prefill is a proven tail amplifier.
- Prefix caching follows the append-only lineage, so the system is not
  repeatedly prefilling the complete history.
- The application/session/finite-request boundary is sound. The engine
  research target is deadline/QoS-aware scheduling and batching for continuous
  short decode mixed with small multimodal prefills.

Use `multi_user.py` to generate manifests, `analyze_capacity.py` for capacity
summaries, and `analyze_prefill_causal.py` for the causal experiment.
