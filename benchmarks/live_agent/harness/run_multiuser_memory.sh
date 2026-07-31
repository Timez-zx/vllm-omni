#!/usr/bin/env bash
# Multi-user contention sweep, with and without long-session memory.
#
#   run_multiuser_memory.sh [users_csv] [reps]      default: 1,2,4  11
#
# One server instance PER POLICY, all user counts run against it, so within a
# policy the ONLY thing varying is the number of concurrent users. (The prior
# contention study did the same; PA_HISTORY_POLICY is read from the server's
# environment, so changing policy is the one thing that needs a restart.)
#
# Each user streams video AND audio continuously for the whole session --
# silence goes out as real PCM zero chunks between utterances, because a real
# microphone does not stop and the server has no VAD.
#
# Probes, all on simultaneously (verified compatible in the Phase 0 study):
#   gpu_sampler   50 Hz NVML device + per-PID  -> per-STAGE busy time
#   stage0_probe  CUDA-event forward hooks     -> vision/audio encoder, exact
#   PA_MEMORY_LOG [PA_MEM] w0/w1 epoch windows -> memory-note call, bracketed
#
# The stage-0 event stream and the server log are shared across the user counts
# of one policy; the analysis separates them by each run's session wall clock.
set -uo pipefail
source /home/zx/voice-agent/env.sh

USERS_CSV="${1:-1,2,4}"
REPS="${2:-11}"
IFS=',' read -r -a USERS <<< "$USERS_CSV"

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
POLICIES=(shipped text_memory)

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

# Fail before burning 150 s on model load.
$PY_OMNI - <<'EOF' || { echo "PATCH NOT INSTALLED -- run harness/patches/apply.sh" >&2; exit 1; }
import sys
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler as C
sys.exit(0 if hasattr(C, "_pa_generate_memory_note") else 1)
EOF

for POL in "${POLICIES[@]}"; do
  LOG=$RES/server_mu_$POL.log
  EVENTS=$RES/stage0_events_mu_$POL.jsonl
  echo ""
  echo "################################################################"
  echo "# policy=$POL   users=${USERS[*]}   reps=$REPS"
  echo "################################################################"
  kill_all
  : > "$LOG"; : > "$EVENTS"

  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_HISTORY_POLICY="$POL" PA_MEMORY_LOG=1 \
  PA_STAGE0_PROBE=1 \
  PA_STAGE0_PROBE_OUT=$RES/stage0_probe_mu_$POL.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

  READY=0
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then READY=1; echo "[$POL] ready after ~$((i*5))s"; break; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
      echo "[$POL] SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; break
    fi
    sleep 5
  done
  [ "$READY" = 1 ] || { echo "[$POL] NOT READY, skipping policy"; continue; }

  for U in "${USERS[@]}"; do
    TAG="mu_${POL}_u${U}"
    OUT=$RES/$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
    echo ""
    echo "---------------- policy=$POL users=$U ----------------"
    # 50 Hz sampler, generous duration; killed as soon as the run ends
    $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_$TAG.jsonl --hz 50 --duration-s 1800 &
    GS=$!
    sleep 1
    timeout 2700 $PY_OMNI $H/ttfa_bench.py \
      --users "$U" --reps "$REPS" --outdir "$OUT" \
      --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -"$((U+1))"
    kill $GS 2>/dev/null; wait $GS 2>/dev/null

    echo "--- latency decomposition, policy=$POL users=$U ---"
    $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
      --traces "$OUT/ttfa_user*.jsonl" --events "$EVENTS" \
      --gpu $RES/gpu_$TAG.jsonl --skip-reps 1 \
      --out $RES/decomp_$TAG.json 2>&1 | sed -n '/=== aggregate/,$p' | head -24

    nmem=$(grep -c "\[PA_MEM\]" "$LOG" 2>/dev/null || echo 0)
    echo "[$TAG] cumulative [PA_MEM] lines: $nmem"
    # brief idle so the next run does not inherit a busy device
    sleep 5
  done
done

kill_all
echo ""
echo "################################################################"
echo "# multiuser sweep done"
echo "################################################################"
