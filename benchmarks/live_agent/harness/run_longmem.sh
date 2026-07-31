#!/usr/bin/env bash
# Long-session memory experiment driver.
#
#   run_longmem.sh <tag> <PA_HISTORY_POLICY> <mode> [extra bench flags]
#
#     mode = recall   -> recall_bench.py  (does it REMEMBER?)
#            latency  -> ttfa_bench.py    (what does it COST?)
#
# Policies (see harness/patches/serving_video_stream.LONGMEM.py):
#   shipped      last turn only, text-only          <- upstream default
#   full_text    all history, text-only
#   full_mm      all history, frames kept
#   text_memory  all history as generated text notes <- the thing under test
#
# The patch must be installed first (harness/patches/apply.sh). With
# PA_HISTORY_POLICY=shipped the patched module takes the identical code path as
# upstream, so the baseline is measured through the same binary -- no
# apply/restore cycling between arms, which would be a confound.
set -uo pipefail
source /home/zx/voice-agent/env.sh

TAG="$1"; POL="$2"; MODE="${3:-recall}"; EXTRA="${4:-}"
RES=/data/zx/results
LOG=$RES/server_$TAG.log
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml   # prefix caching on stage 0

# --- kill any survivors -------------------------------------------------------
# Patterns deliberately avoid matching this script's own cmdline: pkill -f on
# "run_longmem" would kill the invoking shell (learned the hard way, twice).
for pid in $(ps -eo pid=,args= | awk '/vllm serve/ && !/awk/ {print $1}'); do kill "$pid" 2>/dev/null; done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill "$pid" 2>/dev/null ;; esac
done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break; sleep 3
done

echo "[$TAG] policy=$POL mode=$MODE cfg=$(basename "$CFG") GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# --- confirm the patch is actually installed before burning 10 min on startup -
$PY_OMNI - <<'EOF' || { echo "[$TAG] PATCH NOT INSTALLED -- run harness/patches/apply.sh" >&2; exit 1; }
import sys
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler as H
sys.exit(0 if hasattr(H, "_pa_generate_memory_note") else 1)
EOF

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
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed|Traceback" "$LOG"; then
    echo "[$TAG] SERVER FAILED"; grep -iE "not enough GPU|TimeoutError|Error" "$LOG" | tail -5 | cut -c1-200; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "[$TAG] TIMEOUT"; tail -20 "$LOG" | cut -c1-200; exit 1; }

OUT=$RES/longmem_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"

if [ "$MODE" = "recall" ]; then
  timeout 1800 $PY_OMNI /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/recall_bench.py \
    --outdir "$OUT" --policy-label "$POL" $EXTRA 2>&1 | tail -40
else
  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/gpu_sampler.py --out "$OUT/gpu.jsonl" --hz 50 --duration-s 1500 &
  GS=$!
  sleep 1
  timeout 2400 $PY_OMNI /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/ttfa_bench.py \
    --users 1 --reps 11 --outdir "$OUT" $EXTRA \
    --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -3
  kill $GS 2>/dev/null; wait $GS 2>/dev/null
fi

echo ""
echo "[$TAG] ===== memory-note generations ([PA_MEM]) ====="
grep -o "\[PA_MEM\].*" "$LOG" | head -20
n=$(grep -c "\[PA_MEM\]" "$LOG" 2>/dev/null || echo 0)
echo "[$TAG] total [PA_MEM] lines: $n"
echo ""
echo "[$TAG] ===== server-side per-turn timing ====="
grep -o "\[TIMING\].*" "$LOG" | tail -14
echo ""
echo "[$TAG] ===== any errors ====="
grep -iE "PA_MEM.*failed|Query processing failed|longer than the maximum" "$LOG" | tail -5 | cut -c1-200 || echo "  (none)"
