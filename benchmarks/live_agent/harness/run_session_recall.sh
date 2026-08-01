#!/usr/bin/env bash
# The one verification the session arm was missing: does the model still REMEMBER?
#
#   run_session_recall.sh
#
# WHY THIS IS THE OPEN QUESTION. The S640 arm established that one resumable engine request
# per websocket session drops the speech stage to 2.13 ms per 1,000 accumulated prompt tokens
# (against 36.5 for the best per-turn arm), so a 33k-token session ends up with a LOWER TTFA
# than the configuration that discards history. But the whole point of keeping history is
# being able to use it, and session mode changes how history reaches the model in two ways
# that could plausibly break recall:
#
#   1. Each turn is submitted as its own delta and the engine appends it to a live request,
#      rather than the entrypoint rebuilding one prompt containing every frame. If
#      `_update_request_as_session` rebased the new multimodal features' offsets wrongly, the
#      model would be reading image tokens at the wrong positions -- which produces plausible
#      but wrong answers, not an error.
#   2. The talker receives only the delta's conditioning. That should not affect TEXT at all,
#      since text is the thinker's work, so the text scores are also a check on whether the
#      change is reaching further than intended.
#
# THE CONTROL already exists and was run on the same stimuli with the same EVS settings:
# tb_E640_words / tb_E640_detail (per-turn requests, one user block per turn) scored
#   words:  read rate 8/8, first-word probe correct, 8/8 words recalled
#   detail: read rate 7/8, first-word correct, 8/8 recalled, 17 px corner digit CORRECT
# Session mode must match that. A drop would mean the latency win costs memory, which would
# change the recommendation completely.
#
# PRE-REGISTERED EXPECTATIONS
#   - words recalled: 8/8, matching the per-turn control
#   - first-word probe: correct (it asks about turn 0, the oldest thing in the session)
#   - 17 px corner digit: correct -- this is the one a text-only history cannot carry, so it
#     is the sharpest test that the frames themselves are still reachable
#   - every turn produces complete audio, as the ladder showed (3/3 and 12/12)
#
# ONE BOOT PER ARM. Ending a resumable request kills the stage-1 engine core
# (`assert num_new_tokens > 0`, by both the terminal sentinel and the abort route), strictly
# after the last turn is delivered. Data is unaffected but the process is gone, so a second
# session on the same server would fail.
set -uo pipefail
source /home/zx/voice-agent/env.sh

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
LOG=$RES/server_sr.log
EVENTS=$RES/stage0_events_sr.jsonl

# Identical to the S640 latency arm and to the tb_E640 control.
export PA_SESSION=1 PA_APPEND_ONLY=1 PA_EVS_MIN_GAP=8 PA_EVS_MAX_GAP=16

kill_all() {
  local pids
  pids=$(ps -eo pid=,args= | awk -v me=$$ '/cli.main serve|vllm serve/ && !/awk/ && $1!=me {print $1}')
  [ -n "$pids" ] && for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  sleep 6
  [ -n "$pids" ] && for p in $pids; do kill -KILL "$p" 2>/dev/null; done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill -KILL "$pid" 2>/dev/null ;; esac
  done
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    left=$(ps -eo pid=,args= | awk -v me=$$ '/cli.main serve/ && !/awk/ && $1!=me {print $1}' | wc -l)
    [ "$used" -lt 1000 ] && [ "$left" -eq 0 ] && break
    sleep 3
  done
}

boot() {          # label
  printf "\n===== PA_SESSION recall %s %s =====\n" "$1" "$(date -Is)" >> "$LOG"
  : > "$EVENTS"
  echo ""
  echo "################ server up: $1 (PA_SESSION=$PA_SESSION) ################"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_sr.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_SESSION="$PA_SESSION" PA_APPEND_ONLY=1 \
  PA_EVS_MAX_GAP="$PA_EVS_MAX_GAP" PA_EVS_MIN_GAP="$PA_EVS_MIN_GAP" \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "  ready after ~$((i*5))s"; return 0; fi
    if grep -qiE "not enough GPU memory|Engine core initialization failed" "$LOG"; then
      echo "  SERVER FAILED"; return 1
    fi
    sleep 5
  done
  echo "  TIMEOUT"; return 1
}

run_recall() {    # tag scenes_dir
  local tag="$1" scenes="$2"
  local out=$RES/sr_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ RECALL $tag ================"
  # recall_bench accumulates response.text.delta into answer_acc, so its scoring uses the
  # FULL answer rather than the truncated response.text.done payload. That matters: the
  # done payload only contains text emitted before the first audio chunk.
  timeout 1800 $PY_OMNI $H/recall_bench.py \
    --scenes "$scenes" --outdir "$out" --policy-label "$tag" \
    --describe-turns 8 --num-frames 64 --max-frames 512 \
    --evs --evs-threshold 0.95 2>&1 | tail -28
  sleep 3
}

kill_all

# ---- session mode, words ------------------------------------------------------------
export PA_SESSION=1
boot "session-words" || { kill_all; exit 1; }
run_recall "S640_words" /data/zx/stimuli/recall640
kill_all

# ---- session mode, 17 px detail digit ------------------------------------------------
boot "session-detail" || { kill_all; exit 1; }
run_recall "S640_detail" /data/zx/stimuli/recall_detail640
kill_all

echo ""
echo "################################################################"
echo "# session recall done. Compare against the per-turn control:"
echo "#   tb_E640_words  : read 8/8, first-word correct, 8/8 recalled"
echo "#   tb_E640_detail : read 7/8, first-word correct, 8/8 recalled, digit CORRECT"
echo "# analysis: harness/score_session_recall.py"
echo "################################################################"
