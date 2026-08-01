#!/usr/bin/env bash
# Phase 1 of the fix measurement: high motion only, four arms, one variable each.
#
#   run_fix_latency_high.sh [reps] [sessions]      default: 60 4
#
# High motion is measured first and alone because it is the only broken case (1-user
# p50 2,043 ms, 100% of turns over 1 s) and the fastest arm (~10 min per session), so a
# wrong configuration is caught in 40 minutes instead of after three hours. Low motion
# and static are deferred to phase 2, run with whichever configuration wins here.
#
# PROTOCOL, identical to the baseline it is compared against: 1 user, FOUR SEQUENTIAL
# sessions of 60 turns. 4 x 59 = 236 scored turns, turn indices 2-60. Four short
# sessions rather than one long one because TTFA drifts within a session as the frame
# buffer fills, so a single 237-turn session would put most of its samples at turn
# indices the baseline never reached.
#
# ARMS -- each differs from the previous one in exactly one thing
#
#   S     shipped frame handling, 1280x720, num_frames 16. PA_APPEND_ONLY=0 and both
#         gap bounds 0, so the patched file takes the byte-equivalent upstream branch.
#         This arm exists to CHECK THE PATCH, not to learn anything new: it must
#         reproduce the baseline's 2,043 ms. If it does not, the patched file differs
#         from upstream somewhere it should not, and every other arm is suspect.
#
#   N720  append-only frames + retained-gap bracketed to [4,10] frames, still 1280x720.
#         Isolates the frame-handling change. Predicted: prefill collapses because the
#         prompt becomes a growing sequence whose prefix stage 0 can cache, the way
#         low-motion content already accidentally did (its prompt grew 1,374 -> 9,668
#         tokens while its prefill stayed flat at 73-85 ms).
#
#   N640  same, at 640x352. Adds the resolution change: 220 prompt tokens per frame
#         instead of 880, since one token covers a 32x32 pixel tile.
#
#   P640  same as N640 plus enable_prefix_caching on stage 1. Isolates that config bit.
#         Rationale: the measured "one un-interruptible pause per turn, 17 ms per 1,000
#         prompt tokens" is stage 1 re-prefilling its whole prompt with no cache. Under
#         append-only the prompt grows, so that pause grows with it -- 233 ms at 14.4k
#         tokens today, extrapolating to ~1.1 s by the end of a 60-turn session. If
#         stage 1 can cache, the pause should stop scaling. If it cannot, N640 buys a
#         cheap thinker at the price of an expensive talker and that trade has to be
#         reported rather than hidden.
#
# --trace-deltas is on for every arm, so the per-turn blocking pause is measurable and
# not just inferred from TTFA.
#
# max_frames IS THE PROMPT LENGTH under append-only, so it is set per arm against
# the binding context limit rather than to one number for all of them.
#
# THE BINDING LIMIT IS max_model_len = 65,536, NOT the 106,880-token KV. Sizing against
# KV was wrong and killed an arm at turn 10 with
#   ValueError: The decoder prompt (length 71013) is longer than the maximum model
#   length of 65536
# so the caps below leave room for generation (max_tokens 2048) plus audio, query text
# and system prompt (~1,000), i.e. a video budget of ~62,000 tokens:
#
#   at 880 tok/frame (1280x720)   62,000/880 =  70 frames  -> cap 64
#   at 220 tok/frame (640x352)    62,000/220 = 281 frames  -> cap 256
#
# MEASURED growth at the gap bracket's ceiling: ~7 retained frames per turn, i.e. ~6,200
# tokens/turn at 720p and ~1,540 at 640x352. So:
#
#   N720 fills its 64-frame cap around turn 9 and evicts from then on, breaking the
#        append-only prefix for most of the session. That is the honest result: a 65,536
#        context holds only ~70 frames of 720p video, about 2.3 minutes at this rate, so
#        append-only barely works at full resolution.
#   N640 fills its 256-frame cap around turn 36, so roughly the first three fifths of
#        each session is clean append-only and the rest evicts. The turn-index tables are
#        where that shows up, and it is a property of the context limit rather than a
#        setup error.
#
# The resolution cut therefore does not just make frames cheaper, it multiplies how long
# a session can stay append-only by 4x. Neither setting is unbounded.
#
# The shipped pydantic bound on max_frames is le=256, which 320 exceeds; the patched
# file raises it, because under append-only the binding constraint is KV and not that
# constant. S_high keeps 64 -- it re-picks 16 frames every turn regardless, so its cap
# is irrelevant, and 64 is what the baseline used.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-60}"
SESSIONS="${2:-4}"

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG0=$H/deploy_pc_stage0.yaml       # stage-1 prefix caching OFF (as shipped)
CFG1=$H/deploy_pc_stage01.yaml      # stage-1 prefix caching ON
F720=/data/zx/stimuli/frames/handheld_walk_talk
F640=/data/zx/stimuli/frames640/handheld_walk_talk
LOG=$RES/server_fx.log
EVENTS=$RES/stage0_events_fx.jsonl

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

start_server() {   # cfg label
  local cfg="$1" label="$2"
  : > "$LOG"; : > "$EVENTS"
  echo ""
  echo "################ server up for: $label ################"
  echo "  config=$(basename "$cfg")  PA_APPEND_ONLY=$PA_APPEND_ONLY" \
       "gaps=[$PA_EVS_MIN_GAP,$PA_EVS_MAX_GAP]"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_fx.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_APPEND_ONLY="$PA_APPEND_ONLY" \
  PA_EVS_MAX_GAP="$PA_EVS_MAX_GAP" PA_EVS_MIN_GAP="$PA_EVS_MIN_GAP" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$cfg" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "  ready after ~$((i*5))s"; return 0; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed|AssertionError" "$LOG"; then
      echo "  SERVER FAILED for $label"
      grep -iE "not enough GPU|TimeoutError|AssertionError|Error" "$LOG" | tail -6 | cut -c1-180
      return 1
    fi
    sleep 5
  done
  echo "  TIMEOUT"; tail -20 "$LOG" | cut -c1-180; return 1
}

run_arm() {        # tag frames_dir num_frames max_frames
  local tag="$1" frames="$2" nf="$3" mf="$4"
  # --frames wants the STIMULUS directory, not the parent that holds all three of them.
  # Passing the parent makes ttfa_bench print "no frames in ..." and exit 0 per session,
  # so the arm produces an empty output directory and no decomposition -- a silent
  # four-session no-op.
  #
  # Use find, not `ls a/*.jpg a/*.jpeg a/*.png`: ls exits nonzero when ANY operand is
  # missing, so a directory holding 253 .jpg files but no .png was rejected as empty.
  # That guard aborted a valid arm, which is a worse failure than the one it was added
  # to catch.
  if [ -z "$(find "$frames" -maxdepth 1 -type f \
              \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) -print -quit)" ]; then
    echo "!! ARM $tag ABORTED: no image files directly in $frames"
    echo "!! (expected a stimulus dir such as .../frames/handheld_walk_talk)"
    return 1
  fi
  local out=$RES/vt_u1_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ ARM $tag : $SESSIONS x $REPS turns ================"
  echo "  frames=$frames  num_frames=$nf  max_frames=$mf"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_$tag.jsonl --hz 50 --duration-s 14000 &
  local gs=$!
  sleep 1
  for s in $(seq 1 "$SESSIONS"); do
    echo "  --- session $s/$SESSIONS ---"
    timeout 4000 $PY_OMNI $H/ttfa_bench.py \
      --users 1 --reps "$REPS" --outdir "$out" --session "$s" \
      --frames "$frames" --num-frames "$nf" --max-frames "$mf" \
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

# ---- S: the patch's off-path must reproduce the baseline -----------------------
export PA_APPEND_ONLY=0 PA_EVS_MAX_GAP=0 PA_EVS_MIN_GAP=0
if start_server "$CFG0" "S: shipped frame handling, 1280x720"; then
  run_arm "S_high" "$F720" 16 64
fi
kill_all

# ---- N720: frame handling only ------------------------------------------------
export PA_APPEND_ONLY=1 PA_EVS_MAX_GAP=10 PA_EVS_MIN_GAP=4
if start_server "$CFG0" "N720: append-only + gaps, 1280x720"; then
  run_arm "N720_high" "$F720" 16 64
fi
kill_all

# ---- N640: plus the resolution cut --------------------------------------------
if start_server "$CFG0" "N640: append-only + gaps, 640x352"; then
  run_arm "N640_high" "$F640" 16 256
fi
kill_all

# ---- P640: plus stage-1 prefix caching ----------------------------------------
if start_server "$CFG1" "P640: N640 + stage-1 prefix caching"; then
  run_arm "P640_high" "$F640" 16 256
else
  echo ""
  echo "!! stage-1 prefix caching did not come up. That is a RESULT, not a skip:"
  echo "!! the talker cannot cache its prefix on this build, so the append-only"
  echo "!! design keeps a growing per-turn pause. Reported as such."
fi
kill_all

echo ""
echo "################################################################"
echo "# phase 1 done: compare S / N720 / N640 / P640 on high motion"
echo "################################################################"
