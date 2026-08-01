#!/usr/bin/env bash
# Correctness companion to run_talker_delta_probe.sh. Two questions, neither of which a
# latency number can answer.
#
#   run_turnblocks_functional.sh
#
# QUESTION 1 -- does PA_TURN_BLOCKS give the model real multimodal history?
#
# Upstream sends `message_history[-2:]` converted to TEXT ONLY
# (serving_video_stream.py, `_text_only_message`), so past frames never reach the prompt:
# the model can only ever see the frames of the current turn. That is why this project
# concluded a stateful long session does not exist in this stack, and why a text-note
# workaround was built. PA_TURN_BLOCKS keeps every past turn's frames in their own user
# block, so history becomes genuinely multimodal for free. The recall bench measures it:
# 8 scenes each carrying a word, then two probes asking for the FIRST word and for ALL
# words, at a point where none of those scenes is on screen any more.
#
#   baseline for comparison (already measured, same bench, 1 session/arm):
#     shipped stride re-pick 1280x720      7/8 words
#     append-only single block 1280x720    8/8
#     append-only single block 640x352     8/8
#   Fisher two-sided on 7/8 vs 8/8 is p = 1.00, so those three are indistinguishable and
#   this arm cannot beat them on words alone. What it CAN do is the detail probe: a
#   two-digit number rendered at 34 px in a corner, which a text note cannot carry.
#
# QUESTION 2 -- does the talker truncation in the F640 arm break anything?
#
# PA_TALKER_LAST_BLOCK shortens only the TALKER's prompt. Text is produced by the thinker
# and should be bit-identical, so the recall scores of the two arms below must match; if
# they do not, the switch is reaching further than intended and the F640 latency numbers
# are not interpretable. The audio is the real question: the talker loses conditioning it
# would normally have, so `got_audio`, the audio duration and the text/audio length
# agreement are the things to watch. A talker that emits less audio is trivially faster,
# which would invalidate the latency comparison unless it is reported.
#
# PRE-REGISTERED EXPECTATIONS
#   - recall word scores IDENTICAL across the two arms (text is the thinker's work)
#   - LAST_BLOCK=1 still produces audio on every turn; duration may shrink
#   - if LAST_BLOCK=1 produces NO audio, the F640 latency arm measures nothing and must
#     be withdrawn, not reported
set -uo pipefail
source /home/zx/voice-agent/env.sh

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
LOG=$RES/server_tb.log
EVENTS=$RES/stage0_events_tb.jsonl

# Matched to the latency arms.
export PA_APPEND_ONLY=1 PA_EVS_MIN_GAP=8 PA_EVS_MAX_GAP=16 PA_TURN_BLOCKS=1

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
  echo "  PA_TURN_BLOCKS=$PA_TURN_BLOCKS  PA_TALKER_LAST_BLOCK=$PA_TALKER_LAST_BLOCK"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_tb.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_APPEND_ONLY="$PA_APPEND_ONLY" \
  PA_EVS_MAX_GAP="$PA_EVS_MAX_GAP" PA_EVS_MIN_GAP="$PA_EVS_MIN_GAP" \
  PA_TURN_BLOCKS="$PA_TURN_BLOCKS" PA_TALKER_LAST_BLOCK="$PA_TALKER_LAST_BLOCK" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "  ready after ~$((i*5))s"; return 0; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed|RuntimeError: PA_TURN_BLOCKS" "$LOG"; then
      echo "  SERVER FAILED for $1"; grep -iE "not enough GPU|TimeoutError|RuntimeError" "$LOG" | tail -4 | cut -c1-180; return 1
    fi
    sleep 5
  done
  echo "  TIMEOUT"; return 1
}

run_recall() {     # tag scenes_dir
  local tag="$1" scenes="$2"
  local out=$RES/tb_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ RECALL $tag ================"
  # num_frames/max_frames generous: under turn blocks the whole session's frames are in
  # the prompt, and the recall stimuli are tiny, so nothing is at risk of the context cap.
  timeout 1800 $PY_OMNI $H/recall_bench.py \
    --scenes "$scenes" --outdir "$out" --policy-label "$tag" \
    --describe-turns 8 --num-frames 64 --max-frames 512 \
    --evs --evs-threshold 0.95 2>&1 | tail -26
  sleep 3
}

kill_all

# ---- control: talker sees every block (matches the E640 latency arm) ---------------
export PA_TALKER_LAST_BLOCK=0
if start_server "turn blocks, talker sums all blocks (E640 config)"; then
  run_recall "E640_words"  /data/zx/stimuli/recall640
  run_recall "E640_detail" /data/zx/stimuli/recall_detail640
fi
kill_all

# ---- treatment: talker sees only the newest block (matches F640) -------------------
export PA_TALKER_LAST_BLOCK=1
if start_server "turn blocks, talker sizes from the LAST block only (F640 config)"; then
  run_recall "F640_words"  /data/zx/stimuli/recall640
  run_recall "F640_detail" /data/zx/stimuli/recall_detail640
fi
kill_all

echo ""
echo "################################################################"
echo "# turn-blocks functional done."
echo "# Compare E640_words vs F640_words: the WORD scores must match, because"
echo "# text is the thinker's work and only the talker's prompt changed. If they"
echo "# differ, PA_TALKER_LAST_BLOCK reaches further than intended and the F640"
echo "# latency arm is not interpretable."
echo "# Then compare got_audio and audio length: a quieter talker is a faster one."
echo "################################################################"
