#!/usr/bin/env bash
# Capacity stress test: original engine (A) vs tick engine (T), same
# config-identical deploy (mns56). Long answers (~26 s audio/turn) at
# realistic think time; N ladder until QoS breaks.
#
#   bash run_stress.sh A            # greedy baseline ladder
#   bash run_stress.sh T            # tick engine ladder
#   bash run_stress.sh T 32 48      # only selected N cells
#
# QoS (analyze.py metrics): deadline miss < 1%, TTFA p99 < 2000 ms,
# rtf_deliver p50 >= 0.98. Max N passing all three = engine capacity.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_stress.sh A|T [users ...]}"
shift || true
USERS=("${@}")
[ ${#USERS[@]} -eq 0 ] && USERS=(16 24 32 40 48)

case "$ARM" in
  A) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=0) ;;
  T) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_BARRIER=1
               VLLM_OMNI_TEMPORAL_ENGINE=1 VLLM_OMNI_TEMPORAL_INLINE_SEND=1) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

echo "=== STRESS arm $ARM (${PACE_ENV[*]}) users: ${USERS[*]} ==="
bash "$HERE/run_engine.sh" stop
sleep 12
env "${PACE_ENV[@]}" TB_DEPLOY="$HERE/deploy_temporal_2gpu_mns56.yaml" \
  bash "$HERE/run_engine.sh" || { echo "!! engine failed to boot"; exit 1; }

for U in "${USERS[@]}"; do
  OUT="$RESULTS/stress_${ARM}_u${U}"
  echo "--- cell: arm=$ARM users=$U -> $OUT"
  (cd "$WEB" && MU_ENGINE_LOG="$ENGINE_LOG" MU_QUESTIONS=long timeout 3600 \
      "$MAGE_PY" mu_bench.py \
      --users "$U" --content synthetic --turns 6 \
      --audio-input-s 3 --video-interval-ms 480 --think 2,6 --seed 7 \
      --out "$OUT") || echo "!! cell u$U failed (continuing)"
done

echo "=== arm $ARM done ==="
"$MAGE_PY" "$HERE/analyze.py" "$RESULTS"/stress_${ARM}_u* --warmup-turns 1 || true
