#!/usr/bin/env bash
# Full long-session-memory matrix: for each history policy, start the server
# once and run BOTH benches against it.
#
#   run_longmem_matrix.sh [policy ...]      default: shipped full_text text_memory
#
# One server start per policy (~150 s) instead of one per bench. PA_HISTORY_POLICY
# is read from the environment on every request, but the environment belongs to
# the server process, so it cannot be varied from the client -- hence a restart
# per arm.
#
# Two benches per arm, answering the two different questions:
#   recall_bench   does the model REMEMBER earlier turns   (distinct scene/turn)
#   ttfa_bench     what does the memory COST               (same stimulus as the
#                  earlier TTFA study, so numbers stay comparable)
set -uo pipefail
source /home/zx/voice-agent/env.sh

POLICIES=("$@")
[ ${#POLICIES[@]} -eq 0 ] && POLICIES=(shipped full_text text_memory)

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness

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

# Fail fast rather than after 150 s of model loading.
$PY_OMNI - <<'EOF' || { echo "PATCH NOT INSTALLED -- run $H/patches/apply.sh" >&2; exit 1; }
import sys
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler as C
sys.exit(0 if hasattr(C, "_pa_generate_memory_note") else 1)
EOF

for POL in "${POLICIES[@]}"; do
  TAG="lm_$POL"
  LOG=$RES/server_$TAG.log
  echo ""
  echo "################################################################"
  echo "# arm: $POL"
  echo "################################################################"
  kill_all
  : > "$LOG"

  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_HISTORY_POLICY="$POL" PA_MEMORY_LOG=1 \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

  READY=0
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then READY=1; echo "[$TAG] ready after ~$((i*5))s"; break; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
      echo "[$TAG] SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; break
    fi
    sleep 5
  done
  if [ "$READY" != 1 ]; then echo "[$TAG] NOT READY, skipping arm"; continue; fi

  # ---- 1. recall -------------------------------------------------------------
  OUT=$RES/longmem_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
  echo "[$TAG] --- recall bench ---"
  timeout 2400 $PY_OMNI $H/recall_bench.py \
    --outdir "$OUT" --policy-label "$POL" --describe-turns 8 2>&1 | tail -30

  # ---- 2. latency ------------------------------------------------------------
  # Same stimulus and knobs as the earlier TTFA study so the cost of memory can
  # be read against those numbers directly.
  LOUT=$RES/longmem_lat_$TAG; rm -rf "$LOUT"; mkdir -p "$LOUT"
  echo "[$TAG] --- latency bench (11 turns, talking-head) ---"
  timeout 2400 $PY_OMNI $H/ttfa_bench.py \
    --users 1 --reps 11 --outdir "$LOUT" \
    --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -3

  echo "[$TAG] --- per-turn server timing ---"
  grep -o "\[TIMING\].*" "$LOG" | tail -14
  n=$(grep -c "\[PA_MEM\]" "$LOG" 2>/dev/null || echo 0)
  echo "[$TAG] [PA_MEM] lines: $n"
  grep -oE "\[PA_MEM\] dur_ms=[0-9.]+ chars=[0-9]+" "$LOG" | tail -6
  echo "[$TAG] --- errors ---"
  grep -iE "Query processing failed|longer than the maximum|PA_MEM.*failed" "$LOG" \
    | tail -4 | cut -c1-190 || echo "  (none)"
done

kill_all
echo ""
echo "################################################################"
echo "# matrix done -- all arms complete"
echo "################################################################"
