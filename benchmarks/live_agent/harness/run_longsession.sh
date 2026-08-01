#!/usr/bin/env bash
# Long-session experiment: does a 10-turn conversation actually share KV context?
#
#   run_longsession.sh <tag> <deploy-config> <history-policy>
#
# history-policy (PA_HISTORY_POLICY, read by the patched build_engine_prompt):
#   shipped    last 2 messages, text-only            <- vllm-omni default
#   full_text  all history, text-only
#   full_mm    all history, multimodal kept          <- accumulating visual context
set -uo pipefail
source /home/zx/voice-agent/env.sh
TAG="$1"; CFG="$2"; POL="$3"; REPS="${4:-11}"; EXTRA="${5:-}"
RES=/data/zx/results
LOG=$RES/server_$TAG.log

for pid in $(ps -eo pid=,args= | awk '/vllm serve/ && !/awk/ {print $1}'); do kill "$pid" 2>/dev/null; done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill "$pid" 2>/dev/null ;; esac
done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break; sleep 3
done
echo "[$TAG] policy=$POL cfg=$(basename "$CFG")  GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

: > "$LOG"; : > $RES/stage0_events_$TAG.jsonl
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_HISTORY_POLICY="$POL" \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_$TAG.json \
PA_STAGE0_PROBE_EVENTS=$RES/stage0_events_$TAG.jsonl \
nohup /home/zx/miniconda3/envs/omni/bin/vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 240); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "[$TAG] ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "[$TAG] SERVER FAILED"; grep -iE "not enough GPU memory|TimeoutError" "$LOG" | tail -3 | cut -c1-160; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "[$TAG] TIMEOUT"; exit 1; }

OUT=$RES/ttfa_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
$PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/gpu_sampler.py --out $RES/gpu_$TAG.jsonl --hz 50 --duration-s 1200 &
GS=$!
sleep 1
# one user, many turns -- the point is turn INDEX, not concurrency
timeout 2400 $PY_OMNI /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/ttfa_bench.py \
  --users 1 --reps "$REPS" --outdir "$OUT" $EXTRA \
  --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -2
kill $GS 2>/dev/null; wait $GS 2>/dev/null

# keep all reps: the whole question is how latency moves with turn index
$PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
  --traces "$OUT/ttfa_user*.jsonl" --events $RES/stage0_events_$TAG.jsonl \
  --gpu $RES/gpu_$TAG.jsonl --skip-reps 0 \
  --out $RES/ttfa_$TAG.json 2>&1 | sed -n '/=== per turn ===/,/=== aggregate ===/p'
echo "[$TAG] prefill token counts per turn (from stage-0 probe, vision encoder ntok):"
grep -o '"ntok": [0-9]*' $RES/stage0_events_$TAG.jsonl | awk -F': ' '{print $2}' | sort -n | uniq -c | tail -5
