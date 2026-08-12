#!/usr/bin/env bash
# Video-frequency experiment: does raising (or dynamically raising) the video
# input rate change the M vs T picture?
#
#   bash run_video_freq.sh M|T
#
# Fixed workload (mixed, u24, turns=8, same seed as the campaign); the only
# axis is the frame cadence:
#   v480          -- campaign baseline (480 ms, 6x the 80 ms audio grid)
#   v240          -- 2x rate (3x grid)
#   v160          -- 3x rate (2x grid)
#   dyn480a160    -- dynamic: 480 ms idle, 160 ms while a turn is active
#                    (speech start -> response done), the realistic camera
#                    client shape.
# Results: /home/ubuntu/data/results/vf_<ARM>_<CADENCE>.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_video_freq.sh M|T}"
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
    echo "!! engine dead -- rebooting for next cell" >&2
    boot >&2 || return 1
    echo 1
  fi
}

echo "=== VIDEO-FREQ arm $ARM (${PACE_ENV[*]}) ==="
boot || { echo "!! initial boot failed"; exit 1; }

run_cell() {  # name extra-bench-args...
  local NAME=$1; shift
  local OUT="$RESULTS/vf_${ARM}_${NAME}"
  local REBOOTED
  REBOOTED=$(ensure_up) || { echo "!! cannot revive engine; skipping $NAME"; return 1; }
  mkdir -p "$OUT"
  [ -n "${REBOOTED:-}" ] && touch "$OUT/PRECEDED_BY_REBOOT"
  local LOG_OFF
  LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
    --format=csv,noheader -l 1 > "$OUT/gpu.csv" 2>/dev/null &
  local SMI_PID=$!
  local QENV=(MU_QUESTIONS=mixed)
  # u24 with faster video grows context faster; same compression sizing as
  # run_pressure.sh (0.75 * 275,008-token stage-0 pool / 24 users).
  local CFG='{"context_compression_trigger_tokens": 8594, "context_compression_target_tokens": 4297'
  [ "$ARM" = T ] && CFG="$CFG"', "prefill_audio_on_arrival": true, "audio_prefill_chunk_s": 1.0'
  CFG="$CFG"'}'
  QENV+=(MU_SESSION_CFG_JSON="$CFG")
  echo "--- cell: $ARM/$NAME"
  (cd "$WEB" && env MU_ENGINE_LOG="$ENGINE_LOG" ${QENV[@]:+"${QENV[@]}"} \
     timeout 3600 "$MAGE_PY" mu_bench.py \
     --users 24 --content synthetic --turns 8 \
     --audio-input-s 3 --think 2,6 --seed 7 \
     "$@" --out "$OUT") || echo "!! cell $NAME failed (continuing)"
  kill "$SMI_PID" 2>/dev/null
  tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
}

run_cell v480       --video-interval-ms 480
run_cell v240       --video-interval-ms 240
run_cell v160       --video-interval-ms 160
run_cell dyn480a160 --video-interval-ms 480 --video-interval-active-ms 160

echo "=== arm $ARM video-freq done ==="
"$MAGE_PY" "$HERE/analyze.py" "$RESULTS"/vf_${ARM}_* --warmup-turns 2 || true
