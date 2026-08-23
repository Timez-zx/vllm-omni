#!/usr/bin/env bash
# Capacity ladder for the single supported workload: continuous AV sessions.
#
# Required:
#   MU_FRAMES_DIR=/path/to/ordered/jpeg/frames
#   MU_AUDIO_MANIFEST=/path/to/utterances.jsonl
#
# Optional:
#   USERS="8 16 32 48 64 96 128 160" SEEDS="7" TURNS=30
#   RESULT_PREFIX=avsession RESULTS_DIR=/path/to/results
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYBIN=${MU_PYTHON:-python3}
RESULTS_DIR=${RESULTS_DIR:-/tmp/vllm-omni-results}
ENGINE_LOG=${MU_ENGINE_LOG:-$RESULTS_DIR/qwen_live.log}
PORT=${MU_PORT:-8091}
MU_DEPLOY=${MU_DEPLOY:-$REPO_ROOT/benchmarks/thinker_talker/origin_deploy_3gpu.yaml}
EXPECTED_DEPLOY_BASENAME=${MU_EXPECTED_DEPLOY_BASENAME:-origin_deploy_3gpu.yaml}

: "${MU_FRAMES_DIR:?set MU_FRAMES_DIR to the ordered JPEG frame directory}"
: "${MU_AUDIO_MANIFEST:?set MU_AUDIO_MANIFEST to the real-utterance JSONL manifest}"

[ -f "$MU_DEPLOY" ] || {
  echo "deploy YAML not found: $MU_DEPLOY" >&2
  exit 2
}
[ "${MU_DEPLOY##*/}" = "$EXPECTED_DEPLOY_BASENAME" ] || {
  echo "AV capacity run is pinned to $EXPECTED_DEPLOY_BASENAME: $MU_DEPLOY" >&2
  exit 2
}
[ -d "$MU_FRAMES_DIR" ] || {
  echo "frame directory not found: $MU_FRAMES_DIR" >&2
  exit 2
}
[ -f "$MU_AUDIO_MANIFEST" ] || {
  echo "audio manifest not found: $MU_AUDIO_MANIFEST" >&2
  exit 2
}

USERS=${USERS:-"8 16 32 48 64 96 128 160"}
SEEDS=${SEEDS:-"7"}
TURNS=${TURNS:-30}
WARMUP_TURNS=${WARMUP_TURNS:-2}
STAGGER=${MU_STAGGER_S:-0,8}
RESULT_PREFIX=${RESULT_PREFIX:-avsession}
SKIP_DONE=${SKIP_DONE:-1}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-14400}

mkdir -p "$RESULTS_DIR"
log() { echo "[$(date +%H:%M:%S)] $*"; }

stop_engine() {
  local pids pgids pgid
  pids=$(pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" || true)
  if [ -z "$pids" ]; then return 0; fi
  pgids=$(ps -o pgid= -p $pids 2>/dev/null | tr -d " " | sort -u)
  for pgid in $pgids; do kill -TERM -- "-$pgid" 2>/dev/null; done
  for _ in $(seq 1 24); do
    pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" >/dev/null || return 0
    sleep 5
  done
  log "engine did not stop after 120 s"
  return 1
}

healthy() {
  [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:$PORT/health")" = "200" ]
}

run_cell() {
  local users=$1 seed=$2 reps=1 out sampler rc
  local -a trace_args=()
  [ "$users" = "1" ] && reps=2
  out=$RESULTS_DIR/${RESULT_PREFIX}_seed${seed}_u${users}
  mkdir -p "$out"
  case "${MU_INPUT_TRACE_MODE:-live}" in
    live) ;;
    record) trace_args+=(--record-input-trace) ;;
    replay)
      : "${MU_REPLAY_INPUT_TRACE:?set MU_REPLAY_INPUT_TRACE when MU_INPUT_TRACE_MODE=replay}"
      trace_args+=(--replay-input-trace "$MU_REPLAY_INPUT_TRACE")
      ;;
    *) echo "MU_INPUT_TRACE_MODE must be live, record, or replay" >&2; return 2 ;;
  esac
  log "continuous AV: users=$users seed=$seed turns=$TURNS reps=$reps -> $out"
  "$PYBIN" "$REPO_ROOT/benchmarks/live_agent/harness/gpu_sampler.py" \
    --devices "${MU_GPU_IDS:-0,1,2}" --hz "${MU_GPU_SAMPLE_HZ:-5}" \
    --out "$out/gpu_samples.jsonl" > "$out/gpu_sampler.log" 2>&1 &
  sampler=$!

  MU_URL="${MU_URL:-ws://127.0.0.1:$PORT/v1/video/chat/stream}" \
  MU_DEPLOY_CONFIG="$MU_DEPLOY" MU_ENGINE_LOG="$ENGINE_LOG" \
    timeout "$CELL_TIMEOUT_S" "$PYBIN" "$SCRIPT_DIR/mu_bench.py" \
      --users "$users" --turns "$TURNS" --repeat-sessions "$reps" \
      --warmup-turns "$WARMUP_TURNS" --seed "$seed" \
      --stagger "$STAGGER" \
      --frames-dir "$MU_FRAMES_DIR" --audio-manifest "$MU_AUDIO_MANIFEST" \
      --out "$out" "${trace_args[@]}" 2>&1 | tee "$out/driver.log"
  rc=${PIPESTATUS[0]}

  kill -INT "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null || true
  cp -f "$ENGINE_LOG" "$out/engine.log" 2>/dev/null || true
  if ! healthy; then echo "engine_died_after=true" >> "$out/summary_note.txt"; fi
  return "$rc"
}

for seed in $SEEDS; do
  for users in $USERS; do
    out=$RESULTS_DIR/${RESULT_PREFIX}_seed${seed}_u${users}
    if [ "$SKIP_DONE" = "1" ] && [ -f "$out/summary.json" ]; then
      log "skip seed=$seed users=$users: summary exists"
      continue
    fi
    stop_engine || exit 1
    RESULTS_DIR="$RESULTS_DIR" QWEN_LOG="$ENGINE_LOG" MU_PORT="$PORT" \
      DEPLOY_CONFIG="$MU_DEPLOY" bash "$SCRIPT_DIR/run_qwen_server.sh" || exit 1
    run_cell "$users" "$seed"
    rc=$?
    if [ "$rc" -eq 3 ]; then
      log "capacity boundary reached at users=$users seed=$seed; stopping this seed"
      if [ "${LEAVE_ENGINE_RUNNING:-0}" != "1" ]; then stop_engine || exit 1; fi
      break
    elif [ "$rc" -ne 0 ]; then
      if [ "${LEAVE_ENGINE_RUNNING:-0}" != "1" ]; then stop_engine || true; fi
      exit "$rc"
    fi
  done
done

if [ "${LEAVE_ENGINE_RUNNING:-0}" != "1" ]; then stop_engine || exit 1; fi
log "continuous AV ladder complete"
