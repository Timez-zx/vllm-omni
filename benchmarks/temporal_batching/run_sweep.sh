#!/usr/bin/env bash
# One experiment arm end-to-end: boot the engine in that arm's pacing mode,
# run the N-user ladder, leave results under /home/ubuntu/data/results/.
#
#   bash run_sweep.sh A            # greedy baseline
#   bash run_sweep.sh B            # tick=80ms
#   bash run_sweep.sh C            # tick=160ms
#   bash run_sweep.sh D            # rate-limit only, no grid (ablation)
#   bash run_sweep.sh B 4 16       # only the N=4 and N=16 cells
#
# Measured cells run WITHOUT [SCHED-STEP]/[AUDIO-CHUNK] logging (observer
# effect); use `INSTRUMENTED=1 bash run_sweep.sh B 8` for a short evidence
# run whose engine logs carry the batch-formation timeline.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
RESULTS=/home/ubuntu/data/results
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log

ARM="${1:?usage: run_sweep.sh A|B|C|D [users ...]}"
shift || true
USERS=("${@}")
[ ${#USERS[@]} -eq 0 ] && USERS=(1 2 4 8 16)

case "$ARM" in
  A) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=0) ;;
  B) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80) ;;
  C) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=160) ;;
  D) PACE_ENV=(VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_NO_QUANT=1) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

INSTR_ENV=()
SUFFIX=""
if [ "${INSTRUMENTED:-0}" = "1" ]; then
  INSTR_ENV=(VLLM_OMNI_LOG_SCHED_STEPS=0,1 VLLM_OMNI_LOG_AUDIO_CHUNKS=1)
  SUFFIX="_instr"
fi

echo "=== arm $ARM (${PACE_ENV[*]} ${INSTR_ENV[*]:-}) users: ${USERS[*]} ==="
bash "$HERE/run_engine.sh" stop
sleep 12
env "${PACE_ENV[@]}" ${INSTR_ENV[@]:+"${INSTR_ENV[@]}"} bash "$HERE/run_engine.sh" || {
  echo "!! engine failed to boot for arm $ARM"; exit 1; }

for U in "${USERS[@]}"; do
  OUT="$RESULTS/tb_${ARM}_u${U}${SUFFIX}"
  echo "--- cell: arm=$ARM users=$U -> $OUT"
  (cd "$WEB" && MU_ENGINE_LOG="$ENGINE_LOG" timeout 3000 "$MAGE_PY" mu_bench.py \
      --users "$U" --content synthetic --turns 10 \
      --audio-input-s 3 --video-interval-ms 480 --think 1,4 --seed 7 \
      --out "$OUT") || echo "!! cell u$U failed (continuing)"
done

echo "=== arm $ARM done; analyze with:"
echo "  $MAGE_PY $HERE/analyze.py $RESULTS/tb_${ARM}_u*${SUFFIX:-}"
