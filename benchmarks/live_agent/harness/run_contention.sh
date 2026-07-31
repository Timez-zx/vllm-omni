#!/usr/bin/env bash
# Part 2: contention sweep. One server instance, then 1/2/4/8 concurrent users
# against it, so the only thing varying across runs is the number of users.
set -uo pipefail
source /home/zx/voice-agent/env.sh
RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_qwen3_omni_1gpu.yaml
LOG=$RES/server_cont.log

# free the GPU (avoid patterns that match this script's own cmdline)
for pid in $(ps -eo pid=,args= | awk '/vllm serve/ && !/awk/ {print $1}'); do kill "$pid" 2>/dev/null; done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill "$pid" 2>/dev/null ;; esac
done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break; sleep 3
done
echo "GPU free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

: > "$LOG"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_cont.json \
PA_STAGE0_PROBE_EVENTS=$RES/stage0_events_cont.jsonl \
nohup /home/zx/miniconda3/envs/omni/bin/vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
echo "waiting for health..."
READY=0
for i in $(seq 1 240); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "SERVER FAILED"; grep -iE "not enough GPU memory|TimeoutError" "$LOG" | tail -3 | cut -c1-160; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo TIMEOUT; exit 1; }

for U in 1 2 4 8; do
  echo ""
  echo "################ USERS=$U ################"
  OUT=$RES/ttfa_c$U
  rm -rf "$OUT"; mkdir -p "$OUT"
  : > $RES/stage0_events_c$U.jsonl
  # point the probe's event file at a per-run path by copying afterwards; the
  # server writes to one file, so we snapshot offsets instead
  OFF=$(wc -c < $RES/stage0_events_cont.jsonl 2>/dev/null || echo 0)
  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/gpu_sampler.py --out $RES/gpu_c$U.jsonl --hz 50 --duration-s 900 &
  GS=$!
  sleep 1
  timeout 2400 $PY_OMNI /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/ttfa_bench.py \
    --users "$U" --reps 11 --outdir "$OUT" \
    --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -"$((U+1))"
  kill $GS 2>/dev/null; wait $GS 2>/dev/null
  tail -c +$((OFF+1)) $RES/stage0_events_cont.jsonl > $RES/stage0_events_c$U.jsonl 2>/dev/null || true
  echo "--- decomposition, USERS=$U ---"
  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$OUT/ttfa_user*.jsonl" \
    --events $RES/stage0_events_c$U.jsonl \
    --gpu $RES/gpu_c$U.jsonl --skip-reps 1 \
    --out $RES/ttfa_c$U.json 2>&1 | sed -n '/client health/,/=== stage-0/p' | head -40
  sleep 5
done
echo ""
echo "ALL CONTENTION RUNS DONE"
