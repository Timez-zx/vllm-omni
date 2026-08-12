#!/usr/bin/env bash
# Three-arm campaign: A (original) / M (simple tick gates only) / T (fully
# optimized engine) x three workloads (short / long / mixed answers) x N.
#
#   bash run_campaign.sh A|M|T
#
# One boot per arm; health-checked between cells with automatic reboot if a
# known instability killed the engine (reboots are logged -- a rebooted cell
# is marked in the cell dir). Every cell saves its engine-log slice and a
# 1 Hz GPU utilization trace for the bottleneck analysis.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_campaign.sh A|M|T}"
case "$ARM" in
  A) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=0) ;;
  M) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_BARRIER=1) ;;
  T) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_BARRIER=1
               VLLM_OMNI_TEMPORAL_ENGINE=1 VLLM_OMNI_TEMPORAL_INLINE_SEND=1) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

boot() {
  bash "$HERE/run_engine.sh" stop; sleep 12
  env "${PACE_ENV[@]}" TB_DEPLOY="$HERE/deploy_temporal_2gpu_mns56.yaml" \
    bash "$HERE/run_engine.sh"
}

ensure_up() {  # reboot if dead; echo 1 if a reboot happened
  if ! curl -fsS --max-time 3 http://127.0.0.1:8091/health >/dev/null 2>&1; then
    echo "!! engine dead -- rebooting for next cell" >&2
    boot >&2 || return 1
    echo 1
  fi
}

echo "=== CAMPAIGN arm $ARM (${PACE_ENV[*]}) ==="
boot || { echo "!! initial boot failed"; exit 1; }

run_cell() {  # workload users turns
  local WL=$1 U=$2 TURNS=$3
  local OUT="$RESULTS/camp_${ARM}_${WL}_u${U}"
  local REBOOTED
  REBOOTED=$(ensure_up) || { echo "!! cannot revive engine; skipping $WL u$U"; return 1; }
  mkdir -p "$OUT"
  [ -n "${REBOOTED:-}" ] && touch "$OUT/PRECEDED_BY_REBOOT"
  local LOG_OFF
  LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
    --format=csv,noheader -l 1 > "$OUT/gpu.csv" 2>/dev/null &
  local SMI_PID=$!
  # Engine-process CPU trace: the M-vs-T question ("does engine-level
  # periodicity matter once the MODEL is already paced?") is answered here,
  # not in client metrics -- M's loop hot-spins between ticks, T sleeps.
  local ENG_PIDS
  ENG_PIDS=$(pgrep -d, -f "StageEngineCore|APIServer" 2>/dev/null || true)
  local PIDSTAT_PID=""
  if [ -n "$ENG_PIDS" ]; then
    pidstat -h -u -p "$ENG_PIDS" 2 > "$OUT/cpu.txt" 2>/dev/null &
    PIDSTAT_PID=$!
  fi
  local QENV=()
  [ "$WL" != short ] && QENV=(MU_QUESTIONS=$WL)
  echo "--- cell: $ARM/$WL/u$U (turns=$TURNS)"
  (cd "$WEB" && env MU_ENGINE_LOG="$ENGINE_LOG" ${QENV[@]:+"${QENV[@]}"} \
     timeout 3600 "$MAGE_PY" mu_bench.py \
     --users "$U" --content synthetic --turns "$TURNS" \
     --audio-input-s 3 --video-interval-ms 480 --think 2,6 --seed 7 \
     --out "$OUT") || echo "!! cell $WL u$U failed (continuing)"
  kill "$SMI_PID" 2>/dev/null
  [ -n "$PIDSTAT_PID" ] && kill "$PIDSTAT_PID" 2>/dev/null
  tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
}

# short answers: low duty -> deeper ladder, more turns for sample size
for U in 16 24 32; do run_cell short "$U" 10; done
# long answers: high duty
for U in 16 24;    do run_cell long  "$U" 6;  done
# mixed: the realistic shape
for U in 16 24;    do run_cell mixed "$U" 8;  done

echo "=== arm $ARM campaign done ==="
"$MAGE_PY" "$HERE/analyze.py" "$RESULTS"/camp_${ARM}_* --warmup-turns 2 || true
