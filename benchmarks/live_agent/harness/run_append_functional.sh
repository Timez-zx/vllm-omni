#!/usr/bin/env bash
# Does the model still WORK after the frame-handling change? Correctness before speed.
#
#   run_append_functional.sh
#
# WHAT CHANGED, and why each arm below exists
#
# Three things were changed at once, so the arms are ordered to attribute a failure to
# exactly one of them rather than to "the new code":
#
#   1. append-only frames   build_engine_prompt no longer re-picks frames with a stride
#                           anchored at index 0; it takes the buffer in arrival order.
#   2. gap-bracketed EVS    the similarity filter is untouched, but the gap between
#                           retained frames is forced into [MIN_GAP, MAX_GAP] frames
#                           (4 and 10). Retained fraction goes 0.1/0.7/61.7% ->
#                           10.0/10.3/24.9% across screencast/talkinghead/handheld.
#   3. 640x352 frames       220 prompt tokens per frame instead of 880.
#
# ARMS
#
#   A  baseline    shipped frame handling (PA_APPEND_ONLY=0, gaps disabled), original
#                  1280x720 scenes. Reproduces the known-good behaviour on THIS server
#                  build, so arms B and C have something to be compared against that is
#                  not a number from a different night.
#   B  new code    append-only + gap bracketing, still 1280x720. Isolates changes 1+2.
#   C  new code    same, but 640x352 scenes. Isolates change 3 on top of B.
#
# WHAT IS MEASURED
#
# recall_bench shows a DIFFERENT scene each turn and then asks questions only answerable
# from memory, scored by exact substring match against the manifest. That is the right
# instrument here: ttfa_bench streams one continuous scene, so a model with perfect
# memory and one with none would produce identical output on it.
#
# The `listall` score is the one to read -- it is graded (how many of N words survive)
# rather than one pass/fail bit, so a partly-working memory is distinguishable from a
# broken one.
#
# EXPECTATION, stated before running so a bad result cannot be reinterpreted as fine:
#   * B should score at least as well as A. Append-only strictly increases what the
#     model can see (no frame is ever re-picked away), and gap bracketing strictly
#     increases retention on these synthetic scenes, whose consecutive frames are
#     0.998 similar and would otherwise be dropped.
#   * C is the resolution test. The scene word is rendered at 150 px and survives
#     halving (75 px). The corner detail digit is 34 px -> 17 px and is expected to
#     become unreadable. If C's word recall drops too, 640x352 is too aggressive.
#
# max_frames is raised to 64 on purpose: eviction pops from the FRONT, which is the one
# thing that still breaks the append-only prefix, and at 8 (the bench default) it would
# fire constantly and confound the arm.
set -uo pipefail
source /home/zx/voice-agent/env.sh

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
LOG=$RES/server_fn.log
EVENTS=$RES/stage0_events_fn.jsonl

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

start_server() {   # $1 = human label for the log
  : > "$LOG"; : > "$EVENTS"
  echo "################ starting server ($1) ################"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_fn.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_APPEND_ONLY="${PA_APPEND_ONLY:-1}" \
  PA_EVS_MAX_GAP="${PA_EVS_MAX_GAP:-10}" PA_EVS_MIN_GAP="${PA_EVS_MIN_GAP:-4}" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "ready after ~$((i*5))s"; return 0; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
      echo "SERVER FAILED"; grep -iE "not enough GPU|TimeoutError|Traceback" "$LOG" | tail -5 | cut -c1-180
      return 1
    fi
    sleep 5
  done
  echo "TIMEOUT"; tail -20 "$LOG" | cut -c1-180; return 1
}

run_arm() {        # tag scenes_dir label
  local tag="$1" scenes="$2" label="$3"
  local out=$RES/fn_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================================================================"
  echo "# ARM $tag : $label"
  echo "#   scenes=$scenes  PA_APPEND_ONLY=${PA_APPEND_ONLY:-1}"
  echo "#   gaps max=${PA_EVS_MAX_GAP:-10} min=${PA_EVS_MIN_GAP:-4}"
  echo "================================================================"
  timeout 1800 $PY_OMNI $H/recall_bench.py \
    --scenes "$scenes" --outdir "$out" --policy-label "$tag" \
    --describe-turns 8 --num-frames 64 --max-frames 64 \
    --evs --evs-threshold 0.95 2>&1 | tail -30
  sleep 3
}

kill_all

# ---- arm A: shipped behaviour on this build, original resolution
PA_APPEND_ONLY=0 PA_EVS_MAX_GAP=0 PA_EVS_MIN_GAP=0 start_server "A: shipped frame handling" || exit 1
PA_APPEND_ONLY=0 PA_EVS_MAX_GAP=0 PA_EVS_MIN_GAP=0 \
  run_arm "A_shipped_720" /data/zx/stimuli/recall "shipped stride re-pick, 1280x720"
kill_all

# ---- arms B and C: new behaviour, one server for both
PA_APPEND_ONLY=1 PA_EVS_MAX_GAP=10 PA_EVS_MIN_GAP=4 start_server "B/C: append-only + gap bracket" || exit 1
run_arm "B_append_720" /data/zx/stimuli/recall    "append-only + gaps, 1280x720"
run_arm "C_append_640" /data/zx/stimuli/recall640 "append-only + gaps, 640x352"
kill_all

echo ""
echo "################################################################"
echo "# functional arms done -- compare the listall scores across A/B/C"
echo "################################################################"
