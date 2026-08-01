#!/usr/bin/env bash
# The configuration the phase-1 arms say should actually work, plus its control.
#
#   run_fix_tuned.sh [reps] [sessions]      default: 60 4
#
# WHAT PHASE 1 SHOWED, and why this arm exists
#
# Append-only at 1280x720 with a 64-frame cap measured 10,975 ms against the shipped
# baseline's 2,043 -- 5.4x WORSE. The turn-index breakdown says exactly why:
#
#   turns 1-3   prefill   896 ms   (~14 frames, full-recompute prediction 976)
#   turns 4-6             999 ms   (~35 frames, prediction 2,439 -> cache IS working)
#   turns 7-9           1,551 ms   (~56 frames, prediction 3,903 -> still working)
#   turns 10+           7,205 ms   (cap reached; prediction 4,461 -> WORSE than a full
#                                   recompute, i.e. eviction costs more than no cache)
#
# So append-only trades "short prompt, no cache" for "long prompt, partial cache", and
# the long prompt wins as soon as the cap forces eviction. The shipped stride re-pick is
# cache-hostile but it keeps the prompt at 16 frames; append-only lets it reach 64 and
# then pays full price on all of it, every turn.
#
# THE CONDITION FOR APPEND-ONLY TO PAY: the frame cap must never be reached inside the
# session, so the prefix is never broken. Measured retained rate at MIN_GAP=4 was 7.05
# frames/turn (41 frames arriving per turn, 17.2% retained), against a budget of
# cap/turns. At 640x352 the cap that fits max_model_len=65,536 is ~284 frames, so a
# 60-turn session needs <= 4.7 frames/turn. 7.05 does not fit; the arm hits the cap
# around turn 36.
#
# There is a feedback loop worth naming, because it works in both directions. Frames
# arrive for the whole turn cycle, and the cycle length is dominated by TTFA. N720's
# 11 s TTFA stretched its cycle to ~20 s, so 41 frames arrived per turn, so it retained
# more, so the prompt grew faster, so it hit the cap sooner -- a vicious circle. If the
# fix works, TTFA falls to a few hundred ms, the cycle drops to ~8 s, ~17 frames arrive,
# and retention per turn falls with it. This arm tests the virtuous direction.
#
# ARMS
#   T640   append-only, 640x352, gap [8,16], cap 284.
#          MIN_GAP=8 halves the retained rate. If the loop closes the virtuous way,
#          ~17 frames/turn arrive and ~2.1 are retained, i.e. ~130 frames over 60 turns,
#          comfortably inside 284, so the prefix is never broken and prefill should land
#          near 62 ms fixed + new tokens x 79.2 ms/1k = roughly 150-200 ms.
#   C640   CONTROL: shipped frame handling at 640x352. Without this, a fast T640 could
#          not be separated from "smaller frames are just cheaper" -- the shipped path
#          at 640 already has a 4x smaller prompt (16 x 220 + 461 = 3,981 tokens) and
#          would be expected to beat the 1280x720 baseline on its own.
#
# The control matters more than the treatment here: phase 1 has already shown that the
# frame-handling change alone can be harmful, so the resolution cut has to be credited
# separately rather than folded into the same arm.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-60}"
SESSIONS="${2:-4}"

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
F640=/data/zx/stimuli/frames640/handheld_walk_talk
LOG=$RES/server_ft.log
EVENTS=$RES/stage0_events_ft.jsonl

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

start_server() {   # label
  printf "\n===== %s %s =====\n" "$1" "$(date -Is)" >> "$LOG"
  : > "$EVENTS"
  echo ""
  echo "################ server up for: $1 ################"
  echo "  PA_APPEND_ONLY=$PA_APPEND_ONLY  gaps=[$PA_EVS_MIN_GAP,$PA_EVS_MAX_GAP]"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_ft.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_APPEND_ONLY="$PA_APPEND_ONLY" \
  PA_EVS_MAX_GAP="$PA_EVS_MAX_GAP" PA_EVS_MIN_GAP="$PA_EVS_MIN_GAP" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "  ready after ~$((i*5))s"; return 0; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
      echo "  SERVER FAILED for $1"; return 1
    fi
    sleep 5
  done
  echo "  TIMEOUT"; return 1
}

run_arm() {        # tag max_frames
  local tag="$1" mf="$2"
  if [ -z "$(find "$F640" -maxdepth 1 -type f -name '*.jpg' -print -quit)" ]; then
    echo "!! ARM $tag ABORTED: no frames in $F640"; return 1
  fi
  local out=$RES/vt_u1_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ ARM $tag : $SESSIONS x $REPS turns, max_frames=$mf ================"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_$tag.jsonl --hz 50 --duration-s 14000 &
  local gs=$!
  sleep 1
  for s in $(seq 1 "$SESSIONS"); do
    echo "  --- session $s/$SESSIONS ---"
    timeout 4000 $PY_OMNI $H/ttfa_bench.py \
      --users 1 --reps "$REPS" --outdir "$out" --session "$s" \
      --frames "$F640" --num-frames 16 --max-frames "$mf" \
      --evs --evs-threshold 0.95 --trace-deltas 2>&1 | tail -2
    sleep 4
  done
  kill $gs 2>/dev/null; wait $gs 2>/dev/null
  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$out/ttfa_user*.jsonl" --events "$EVENTS" \
    --gpu $RES/gpu_vt_u1_$tag.jsonl --skip-reps 1 \
    --out $RES/decomp_vt_u1_$tag.json 2>&1 | sed -n '/=== aggregate/,$p' | head -12
  sleep 4
}

kill_all

# ---- C640: the control. shipped frame handling, only the resolution changed --------
export PA_APPEND_ONLY=0 PA_EVS_MAX_GAP=0 PA_EVS_MIN_GAP=0
if start_server "C640: shipped frame handling at 640x352"; then
  run_arm "C640_high" 64
fi
kill_all

# ---- T640: append-only with a gap tight enough to never reach the cap --------------
export PA_APPEND_ONLY=1 PA_EVS_MAX_GAP=16 PA_EVS_MIN_GAP=8
if start_server "T640: append-only, gap [8,16], 640x352"; then
  run_arm "T640_high" 284
fi
kill_all

echo ""
echo "################################################################"
echo "# tuned arms done: compare C640 (resolution only) vs T640 (plus append-only)"
echo "################################################################"
