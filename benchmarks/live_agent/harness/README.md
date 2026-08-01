# harness/ — what each script is, and which finding it produced

Flat on purpose. The scripts hardcode `$H/…` paths and the reports reference them, so grouping
them into subdirectories would mean editing a lot of references for no measurable gain. This
index is the substitute: it says which study each file belongs to and whether it is still live.

`env.sh` at the repo root defines `$PY_OMNI` (the serving env) and `$PY_PA0` (the analysis env).
All raw data goes to `/data/zx/results`; figures to `../figures/`.

## Live — the current line of work (session mode, the fork)

| file | what it does |
|---|---|
| `ttfa_bench.py` | **the client for every arm.** Streams frames + audio over the websocket, sends a query per turn, records a JSONL trace. Every fork-side knob is a flag here and is echoed into the trace (`k=tx_config`). |
| `run_fork_verify.sh` | does the fork reproduce the overlay's numbers? Ladder (3, 12 turns) + a 50-turn arm, gated. → `FRAME_PIPELINE.md` §5.10 |
| `run_wedge_probe.sh` | one long session, default query. Produced the pre-fix and post-fix 50-turn baselines (p50 346 / 348). |
| `run_roll_probe.sh` | session rolling: `run_roll_probe.sh <reps> <roll_at>`. Produced the 40-turn / 2-roll result. → §5.12 |
| `check_session.py` | bring-up gate. Refuses an arm unless session mode really engaged, the talker's prompt is delta-sized, every turn produced audio, and telemetry survived. Catches the silent fallback to per-turn mode. |
| `session_prompt_selftest.py` | 30 zero-GPU assertions on the delta prompt shape. Run before anything that costs GPU time. |
| `deploy_pc_stage0.yaml` | **the reference deployment** — three stages on one card. All published numbers use this. |
| `gpu_sampler.py` | NVML sampler, 50 Hz. |

## Live — recall / accuracy

| file | what it does |
|---|---|
| `recall_bench.py`, `gen_recall_stimuli.py` | the memory protocol: spoken words + a 17 px corner digit, the load-bearing item a text-only history cannot carry. |
| `run_session_recall.sh`, `score_session_recall.py` | recall under session mode. → `project-3-session-mode-wins` |
| `run_longmem.sh`, `run_longmem_matrix.sh`, `run_mm_memory.sh`, `run_multiuser_memory.sh` | → `LONG_MEMORY.md`, `MM_MEMORY.md`, `MULTIUSER_MEMORY.md` |

## Earlier studies — kept for provenance

These produced the published doc pairs. They are a few KB each and deleting them would turn
those documents into unrepeatable assertions.

| file(s) | study |
|---|---|
| `run_sweep.sh`, `run_contention.sh`, `stage0_probe.py`, `paced_client.py` | `TTFA_STUDY.md` |
| `evs_probe.py`, `frame_selection_headroom.py` | `PHASE0_FINDINGS.md` (frame filtering) |
| `run_video_latency.sh`, `run_video_tail.sh`, `run_user1_tail.sh`, `run_ramp_probe.sh` | `VIDEO_LATENCY.md`, tail-latency work |
| `run_fix_latency_high.sh`, `run_fix_latency_S.sh`, `run_fix_latency_rest.sh`, `run_fix_tuned.sh`, `run_detail_probe.sh` | the C640 resolution result → `FRAME_PIPELINE.md` §5.6–5.8 |
| `run_talker_delta_probe.sh`, `check_talker_probe.py` | what the talker charges per prompt token → §5.7, §5.9 |
| `run_append_functional.sh`, `run_turnblocks_functional.sh` | append-only and turn-block variants (both lost to session mode) |
| `run_session_arm.sh` | the overlay-era session arm — the 2.13 ms/1k reference the fork is compared against |
| `run_longsession.sh`, `run_pc_probe.sh`, `deploy_pc_on.yaml`, `deploy_icf1.yaml`, `deploy_icf8.yaml`, `deploy_qwen3_omni_1gpu.yaml`, `deploy_pc_stage01.yaml` | prefix-caching and single-GPU/topology probes |

Every `.sh` / `.py` / `.yaml` in this directory is named above, literally rather than by glob, so
the index can be checked mechanically:

```
python3 -c "
import pathlib; h=pathlib.Path('.'); idx=(h/'README.md').read_text()
print([f.name for f in h.iterdir() if f.suffix in ('.sh','.py','.yaml') and f.name not in idx] or 'complete')"
```

## Not here

- `../../../../archive/overlay-pre-fork/` — the pre-fork patched copies of `vllm_omni`, superseded by the
  fork but the only way to re-run its reference baseline.
- `../../../../archive/refuted-probes/` — probes whose hypotheses were refuted, kept runnable so the
  refutations can be repeated.
