#!/usr/bin/env bash
# Experiment B only, extended: tail latency (p90/p95/p99) up to 8 concurrent users.
#
#   run_video_tail.sh [users_csv] [reps]      default: 8,4,2,1  60
#
# Why this exists separately from run_video_latency.sh:
#
#   * Experiment A (frame count forced, filter off) is deliberately NOT a realistic
#     workload -- it exists to isolate causality. It is not repeated here.
#
#   * The earlier B runs used 11 turns = 10 scored samples per user. p99 from 10
#     samples is just the maximum wearing a different name. 60 turns is used here for
#     every user count, so sample count scales with concurrency and the estimator
#     quality is stated rather than assumed:
#         1 user -> 59    2 -> 118    4 -> 236    8 -> 472 scored turns
#     A percentile needs n >= 10/(1-q): p95 wants 200 (met at 4 and 8 users), p99
#     wants 1000 (met nowhere, but at 472 samples p99 lands on the 6th-worst turn,
#     which is a usable if noisy estimate rather than the maximum).
#
#   * User counts run in DESCENDING order (8 first). The 8-user arms are the ones
#     that might break, so they surface in the first ten minutes instead of after
#     three hours.
#
#   * 8 users is added because 4 was not enough to see saturation: at 4 users the
#     device already measured 88% busy, so the interesting behaviour is just past it.
#
# A known risk at 8 users x high motion: each request carries ~14,400 prompt tokens
# and stage-0 KV holds 106,880, so 8 x 14,400 = 115,200 does not fit. vLLM will have
# to preempt or recompute. That is a real production behaviour rather than a setup
# error, so it is allowed to happen and the log is checked for it afterwards.
#
# Everything varied here is client-side session config, so all runs share ONE server.
set -uo pipefail
source /home/zx/voice-agent/env.sh

USERS_CSV="${1:-8,4,2,1}"
REPS="${2:-60}"
IFS=',' read -r -a USERS <<< "$USERS_CSV"

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
F=/data/zx/stimuli/frames
LOG=$RES/server_vt.log
EVENTS=$RES/stage0_events_vt.jsonl

kill_all() {
  for pid in $(ps -eo pid=,args= | awk '/vllm serve|cli.main serve/ && !/awk/ {print $1}'); do
    kill "$pid" 2>/dev/null
  done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill "$pid" 2>/dev/null ;; esac
  done
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1000 ] && break; sleep 3
  done
}

run_one() {          # tag users stimulus
  local tag="$1" users="$2" stim="$3"
  local out=$RES/vt_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "---------------- $tag : users=$users stim=$(basename "$stim") reps=$REPS ----------------"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_$tag.jsonl --hz 50 --duration-s 4200 &
  local gs=$!
  sleep 1
  # shipped defaults throughout: EVS on at 0.95, num_frames 16, max_frames 64
  timeout 4000 $PY_OMNI $H/ttfa_bench.py \
    --users "$users" --reps "$REPS" --outdir "$out" \
    --frames "$stim" --num-frames 16 --max-frames 64 --evs --evs-threshold 0.95 \
    2>&1 | tail -"$((users+1))"
  kill $gs 2>/dev/null; wait $gs 2>/dev/null

  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$out/ttfa_user*.jsonl" --events "$EVENTS" \
    --gpu $RES/gpu_vt_$tag.jsonl --skip-reps 1 \
    --out $RES/decomp_vt_$tag.json 2>&1 | sed -n '/=== aggregate/,$p' | head -14
  # did the engine have to preempt / recompute? that is the 8-user KV risk
  echo "  --- engine pressure signals in this window ---"
  grep -ciE "preempt|recompute|swap" "$LOG" 2>/dev/null | sed 's/^/    cumulative preempt-ish log lines: /'
  sleep 5
}

kill_all
: > "$LOG"; : > "$EVENTS"
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_vt.json \
PA_STAGE0_PROBE_EVENTS="$EVENTS" \
nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "TIMEOUT"; tail -20 "$LOG" | cut -c1-180; exit 1; }

echo ""
echo "################################################################"
echo "# B EXTENDED: shipped config, content x users, tail latency"
echo "################################################################"
for U in "${USERS[@]}"; do
  run_one "u${U}_static" "$U" "$F/screencast"
  run_one "u${U}_low"    "$U" "$F/talkinghead"
  run_one "u${U}_high"   "$U" "$F/handheld_walk_talk"
done

kill_all
echo ""
echo "################################################################"
echo "# video tail sweep done"
echo "################################################################"
