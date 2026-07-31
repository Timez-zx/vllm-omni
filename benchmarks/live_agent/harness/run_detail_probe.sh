#!/usr/bin/env bash
# What does the 640x352 resolution cut actually cost? Measure it, do not assume it.
#
#   run_detail_probe.sh
#
# The A/B/C functional arms showed 640x352 costing nothing: 8/8 words recalled at both
# resolutions, same 62% scene read rate. But that test only asks the model to read a
# word rendered at 150 px, which becomes 75 px after halving and is still large. It does
# NOT probe the thing predicted to break.
#
# The recall_detail stimuli exist for exactly this. Each scene carries, besides the big
# word, a two-digit number in the bottom-right corner rendered at 34 px -- about a fifth
# the word's size, deliberately built as "the detail a text note cannot carry". After
# the cut it is 17 px. Claiming the resolution cut is free without testing this would be
# claiming it on the strength of the one measurement that could not detect the cost.
#
# ARMS: identical configuration (append-only, gaps [4,10]), one server, only the
# stimulus resolution differs. The detail probe is added by the bench AUTOMATICALLY
# when the manifest carries a 4th column and q_detail.wav exists -- it must NOT be
# requested via --describe-utter, whose only valid choices are q_shape and q_describe.
# Passing it there made the bench exit on an argparse error, and this script's closing
# kill_all then took down a server that a DIFFERENT run was using. Reading the corner digit is scored by exact substring
# match against the manifest.
#
# EXPECTATION, recorded before running: the word survives at both resolutions and the
# digit does not survive at 640x352. If the digit survives, the cut is cheaper than
# predicted. If the WORD stops surviving at 640, something other than resolution is
# wrong, because 75 px is not small.
set -uo pipefail
source /home/zx/voice-agent/env.sh

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
LOG=$RES/server_dt.log
EVENTS=$RES/stage0_events_dt.jsonl

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

run_arm() {        # tag scenes_dir label
  local tag="$1" scenes="$2" label="$3"
  local out=$RES/dt_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ DETAIL ARM $tag : $label ================"
  timeout 1800 $PY_OMNI $H/recall_bench.py \
    --scenes "$scenes" --outdir "$out" --policy-label "$tag" \
    --describe-turns 8 --num-frames 64 --max-frames 512 \
    --evs --evs-threshold 0.95 2>&1 | tail -26
  sleep 3
}

kill_all
printf "\n===== detail probe %s =====\n" "$(date -Is)" >> "$LOG"; : > "$EVENTS"
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_dt.json \
PA_STAGE0_PROBE_EVENTS="$EVENTS" \
PA_APPEND_ONLY=1 PA_EVS_MAX_GAP=10 PA_EVS_MIN_GAP=4 \
nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -4 | cut -c1-180; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "TIMEOUT"; tail -20 "$LOG" | cut -c1-180; exit 1; }

run_arm "detail_720" /data/zx/stimuli/recall_detail    "34 px corner digit, 1280x720"
run_arm "detail_640" /data/zx/stimuli/recall_detail640 "17 px corner digit, 640x352"

kill_all
echo ""
echo "################################################################"
echo "# detail probe done -- compare the digit scores across 720 / 640"
echo "################################################################"
