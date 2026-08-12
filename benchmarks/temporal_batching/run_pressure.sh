#!/usr/bin/env bash
# Post-bugfix pressure ladder: M vs T at higher user counts.
#
#   bash run_pressure.sh M|T
#
# Two workloads (long / mixed) x u32/u40/u48. Same cell instrumentation as
# run_campaign.sh (gpu.csv, cpu.txt, engine_slice.log, reboot markers).
# Results: /home/ubuntu/data/results/press_<ARM>_<WL>_u<N>.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_pressure.sh M|T}"
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

echo "=== PRESSURE arm $ARM (${PACE_ENV[*]}) ==="
boot || { echo "!! initial boot failed"; exit 1; }

run_cell() {  # workload users turns
  local WL=$1 U=$2 TURNS=$3
  local OUT="$RESULTS/press_${ARM}_${WL}_u${U}"
  local REBOOTED
  REBOOTED=$(ensure_up) || { echo "!! cannot revive engine; skipping $WL u$U"; return 1; }
  mkdir -p "$OUT"
  [ -n "${REBOOTED:-}" ] && touch "$OUT/PRECEDED_BY_REBOOT"
  local LOG_OFF
  LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
    --format=csv,noheader -l 1 > "$OUT/gpu.csv" 2>/dev/null &
  local SMI_PID=$!
  local ENG_PIDS
  ENG_PIDS=$(pgrep -d, -f "StageEngineCore|APIServer" 2>/dev/null || true)
  local PIDSTAT_PID=""
  if [ -n "$ENG_PIDS" ]; then
    pidstat -h -u -p "$ENG_PIDS" 2 > "$OUT/cpu.txt" 2>/dev/null &
    PIDSTAT_PID=$!
  fi
  local QENV=()
  [ "$WL" != short ] && QENV=(MU_QUESTIONS=$WL)
  # Context compression, scaled to N -- the hard capacity rule is
  # sessions <= stage-0 KV pool / per-session context (pool 275,008 tokens;
  # measured: 24 unbounded mixed sessions froze the stage at turn 6-8 with
  # sum(computed) == 100.0% of the pool). trigger = 0.75*pool/N keeps the
  # steady state near half the pool and leaves shadow-warmup headroom.
  # Identical for both arms: this knob is workload realism, not a treatment.
  local TRIGGER=$(( 275008 * 3 / 4 / U ))
  local TARGET=$(( TRIGGER / 2 ))
  local CFG='{"context_compression_trigger_tokens": '$TRIGGER', "context_compression_target_tokens": '$TARGET
  [ "$ARM" = T ] && CFG="$CFG"', "prefill_audio_on_arrival": true, "audio_prefill_chunk_s": 1.0'
  CFG="$CFG"'}'
  QENV+=(MU_SESSION_CFG_JSON="$CFG")
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

# Ladder from just above the old capacity edge. long = high duty (talker
# ceiling); mixed = realistic KV growth (stage-0 ceiling).
for U in 32 40 48; do run_cell long  "$U" 6; done
for U in 32 40 48; do run_cell mixed "$U" 8; done

echo "=== arm $ARM pressure ladder done ==="
"$MAGE_PY" "$HERE/analyze.py" "$RESULTS"/press_${ARM}_* --warmup-turns 2 || true
