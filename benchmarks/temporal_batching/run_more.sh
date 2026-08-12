#!/usr/bin/env bash
# Round-3 extension: (a) frontier cells u52 (below the mns-56 admission cap,
# leaving shadow-warmup slots) to see where T's QoS also gives out; (b) a
# 6x-spread high-dynamic video cell (idle 960 ms -> active 160 ms).
#
#   bash run_more.sh M|T
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_more.sh M|T}"
case "$ARM" in
  M) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_BARRIER=1) ;;
  T) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_BARRIER=1
               VLLM_OMNI_TEMPORAL_ENGINE=1 VLLM_OMNI_TEMPORAL_INLINE_SEND=1
               VLLM_OMNI_TEMPORAL_REPLAY=1 VLLM_OMNI_TEMPORAL_MAILBOX=1) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

boot() {
  bash "$HERE/run_engine.sh" stop; sleep 12
  env "${PACE_ENV[@]}" TB_DEPLOY="$HERE/deploy_temporal_2gpu_mns56.yaml" \
    bash "$HERE/run_engine.sh"
}
ensure_up() {
  if ! curl -fsS --max-time 3 http://127.0.0.1:8091/health >/dev/null 2>&1; then
    echo "!! engine dead -- rebooting" >&2; boot >&2 || return 1; echo 1
  fi
}

echo "=== MORE arm $ARM ==="
boot || { echo "!! initial boot failed"; exit 1; }

run_cell() {  # out-name users turns questions extra-bench-args...
  local NAME=$1 U=$2 TURNS=$3 WL=$4; shift 4
  local OUT="$RESULTS/${NAME}"
  local REBOOTED
  REBOOTED=$(ensure_up) || { echo "!! cannot revive; skip $NAME"; return 1; }
  mkdir -p "$OUT"
  [ -n "${REBOOTED:-}" ] && touch "$OUT/PRECEDED_BY_REBOOT"
  local LOG_OFF; LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
    --format=csv,noheader -l 1 > "$OUT/gpu.csv" 2>/dev/null &
  local SMI_PID=$!
  local TRIGGER=$(( 275008 * 3 / 4 / U ))
  local TARGET=$(( TRIGGER / 2 ))
  local CFG='{"context_compression_trigger_tokens": '$TRIGGER', "context_compression_target_tokens": '$TARGET
  [ "$ARM" = T ] && CFG="$CFG"', "prefill_audio_on_arrival": true, "audio_prefill_chunk_s": 1.0'
  CFG="$CFG"'}'
  local QENV=(MU_SESSION_CFG_JSON="$CFG")
  [ "$WL" != short ] && QENV+=(MU_QUESTIONS=$WL)
  echo "--- cell: $NAME (turns=$TURNS)"
  (cd "$WEB" && env MU_ENGINE_LOG="$ENGINE_LOG" ${QENV[@]:+"${QENV[@]}"} \
     timeout 3600 "$MAGE_PY" mu_bench.py \
     --users "$U" --content synthetic --turns "$TURNS" \
     --audio-input-s 3 --think 2,6 --seed 7 \
     "$@" --out "$OUT") || echo "!! cell $NAME failed (continuing)"
  kill "$SMI_PID" 2>/dev/null
  tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
}

run_cell "press_${ARM}_mixed_u52" 52 8 mixed --video-interval-ms 480
run_cell "press_${ARM}_long_u52"  52 6 long  --video-interval-ms 480
run_cell "vf_${ARM}_dyn960a160"   24 8 mixed --video-interval-ms 960 --video-interval-active-ms 160

echo "=== arm $ARM MORE done ==="
"$MAGE_PY" "$HERE/analyze.py" "$RESULTS"/press_${ARM}_*_u52 "$RESULTS"/vf_${ARM}_dyn960a160 --warmup-turns 2 || true
