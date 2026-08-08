# Tri-colocation design: thinker joins the unified speech process

Recon 2026-08-08 (full file:line survey of the colocation stack). Goal: stages
0+1+2 in ONE process so the s0->s1 edge (291 MB/turn `.detach().cpu()` pair,
57-77% of the talker's added latency per FRAME_PIPELINE.md) becomes an
in-process reference handoff.

## Decision: host = stage 1 (talker)

Stage 0 as host avoids reordering but starves the talker: a late stage-1 guest
gets `0.15*total - thinker(~65 GiB) -> 0` KV. Host=1 keeps the talker pool
honest; the thinker (late guest) is the stage whose budget we can compute from
a profiling delta and whose fraction we can raise. Requires a host-before-guest
reordering pass in `stage_runtime._init_group` (stage_id==index is hard-coded;
"host must appear earlier" is enforced).

## Two prerequisites (in order)

1. **Per-engine memory accounting** (`vllm_omni/worker/base.py:125-133`): the
   NVML per-PID snapshot diff subtracts EVERYTHING already built in the
   process, so the 2nd KV-bearing engine's pool collapses to ~0. Fix: in
   colocated mode force the profiling fallback (`base.py:147-154`, delta-based,
   already exists) or subtract a per-engine NVML baseline captured in
   `init_device`. Escape hatch: explicit `kv_cache_memory_bytes` per stage
   (precedent: deploy/voxcpm2.yaml:23). NOTHING ELSE MATTERS UNTIL THIS LANDS.
2. **MoE workspace partition question**: eea63d41 made WorkspaceManager shared
   (guest re-init keeps existing manager, lock is a no-op). With TWO MoE
   engines (thinker + talker) stepping from two threads on two streams into
   ONE scratch arena, silent numerical corruption is possible. READ
   `vllm/v1/worker/workspace.py` first: if the arena is handed out whole (not
   sliced per caller), tri-colocation needs per-engine managers — the pair's
   fix is the trio's bug.

## Mechanical change set (after prerequisites)

- `stage_runtime.py`: allow multiple guests (delete single-guest raise at
  :602-604), ordered `list[SiblingStageLaunch]`, raise on guest-and-host
  chains (today `"1:0,2:1"` silently strands stage 2), stash per guest
  (dict already keyed by stage id), host-first reordering in `_init_group`.
- `stage_engine_startup.py`: sibling_addresses -> dict[int, addresses]; one
  pre-bound ROUTER per guest; `wait_for_engine_startup` per guest IN THE SAME
  ORDER the child builds them (order mismatch = silent boot hang).
- `stage_engine_core_proc.py`: loop over guests — per-guest CUDA stream
  (third engine otherwise lands on the legacy default stream and re-serializes
  everyone), build ALL cores sequentially FIRST then start loop threads
  (graph captures must not interleave), per-guest death fanout, guests in the
  shutdown `finally`.
- `stage_engine_core_proc_manager.py`: sibling_kwargs -> list.
- Deploy yaml: `VLLM_OMNI_COLOCATE_STAGES=2:1,0:1`; the coloc yaml already
  routes from_stage_0 through the coloc connector, and the connector's flat
  group parser flips that edge to intra-process automatically.
- `coloc_inproc_connector.py`: BEFORE enabling the s0->s1 edge, add
  cross-stream safety (ship (tensor, cuda_event), consumer waits; pattern =
  `gpu_ar_model_runner._snapshot_tensor_payload_to_cpu_async`). Note
  cleanup() has no engine-path caller: aborted requests leak GPU references.
- LAST + SEPARATE: teach `stage_input_processors/qwen3_omni.py:304-305,
  617-622` to skip `.detach().cpu()` when the edge is intra-process. That
  copy is ALSO the snapshot that makes async scheduling safe — removing it
  needs the event-based lifetime story first. Even without this step,
  in-process routing kills 3 of 4 copies (msgpack + SHM passes + rebuild).

## Thread census of the merged process

3 busy loops + 6 ZMQ socket threads + 6 adapter I/O + 4 mixin I/O + per-step
transient ≈ 19-20 Python threads, ≥10 polling at 1-10 ms, one GIL. Precedent:
pair merge cost rtf 0.75->0.54 before graphs/streams/in-proc clawed to 0.79;
async then bought 1.13. The thinker's Python-side per-step work is the largest
of the three. Helper threads run on the DEFAULT stream (thread-local scoping!)
— the 291 MB D2H currently issues there = global barrier across all private
streams.

## Ordered risks

1. Shared MoE workspace x2 MoE engines (silent corruption) — blocker question.
2. KV collapse of the 2nd KV engine (deterministic, loud) — prerequisite 1.
3. Handshake-order deadlock (silent hang).
4. Graph-capture interleaving if a guest loop starts early.
5. GIL saturation (~20 threads).
6. Default-stream helper threads = cross-engine barriers.
7. Cross-stream reference handoff lifetimes (step "LAST").
8. GPU-byte leaks in _store / _pending_streaming_prefills (no cleanup caller).
9. cleanup() prefix cross-talk between two edges in one store.
10. Blast radius: one fatal error kills all three stages; guest-thread hangs
    invisible at manager granularity.
11. runtime.env / VLLM_OMNI_REPLICA_ID / proc title become host-scoped.
12. Profiling peaks now stack inside one allocator (headroom re-check).

## Why this is harder than the pair (fundamentals)

The pair survived wrong memory accounting because its guest had no KV budget;
the trio cannot. The pair had one MoE engine, so sharing process globals was
the fix; with two, partitioning is. The expensive edge's copy is also its
snapshot — transport and memory-safety are the same line of code. And the
ordering constraints (stage_id==index, host-earlier, parent-wait==child-build
order) go from avoidable to mandatory.

## P1+P2a scoreboard (2026-08-08, all gates green throughout)

| cell | separate procs (all-opt) | tri P1 | tri P2a (on-device payloads) |
|---|---|---|---|
| audio 32u p50/p95 | 297 / 486 | 337 / 1649 | 349 / 1663 |
| video 13u p50 / p95 / >1s | 457 / 2200 / 11.5% | 639 / 1702 / 40% | 648 / **1363 / 17.3%** |

Findings that supersede parts of the plan above:
- The builder-level `.detach().cpu()` sites were NOT paying the D2H -- the device
  probe showed tensors already arrive on CPU there. The real payment splits in two:
  (a) the async output snapshot (now gated: full colocation ships the D2D clone,
  landed as P2a -- this is what bought the video tail back), and (b)
  `staged_hidden_states_cpu`, pre-staged during forward and HARD-REQUIRED on CPU by
  the prefix-cache pooler payload ("Prefix-cache hidden-state payload requires
  staged CPU hidden states", gpu_ar_model_runner.py:888). Removing (b) = redesign
  that consumer's feed (GPU-side pooler slicing, or a lazy staged copy taken only
  when a prefix-cache payload is actually needed). NEXT DIG SITE.
- With streams + in-proc connector + on-device payloads all in, the residual audio
  gap (349 vs 297, p95 x3) is the GIL: three Python busy loops + ~20 threads. The
  pair had the same shape and clawed back with graphs/streams/connector; the trio
  already HAS all three, so the next lever is structural (engine loop off Python /
  free-threading) -- thesis-scale, not a config.
- Verdict for serving TODAY: separate-process async config remains the
  performance choice; tri-coloc is the RESEARCH PLATFORM (one process, unified
  visibility, per-engine arenas/accounting proven) awaiting the GIL work.
