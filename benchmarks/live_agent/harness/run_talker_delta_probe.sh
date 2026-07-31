#!/usr/bin/env bash
# THE DECISIVE EXPERIMENT: does the talker's cost stop tracking prompt length if the
# talker only has to prefill the NEW frames?
#
#   run_talker_delta_probe.sh [reps] [sessions]        default: 60 4
#
# WHY THIS EXPERIMENT AND NOT THE SESSION-SCOPED REWRITE
#
# The open question is Xiao's: prefill each frame as it arrives, on BOTH stages, and the
# talker should stop paying for old frames. The obvious way to test that is to convert
# the video endpoint from one engine request per TURN to one per SESSION, driven by an
# AsyncGenerator, which is the path the engine already has
# (async_omni.py:398 -> add_streaming_update_async, resumable=True). Reconnaissance found
# that route to be a multi-day job with three fatal blockers, one of which destroys the
# measurement itself:
#
#   - the session-long request is torn down after turn 1, because
#     finished_final_output_stage_ids accumulates and is never reset
#     (orchestrator.py:897-901, 969-970);
#   - per-turn wire events (response.audio.done, on_turn_complete) are all emitted after
#     the `async for` loop exits, which under a session-long request means once per
#     SESSION (video_stream_base.py:761-803), so every turn boundary must be rebuilt;
#   - StageRequestStats is printed once per request id, so a 60-turn session would
#     produce ONE table instead of 60 -- the entire server-side measurement basis for
#     this study disappears.
#
# But the QUANTITY in question does not need any of that. The talker's cost is set by the
# length of its placeholder prompt, and that length is computed by one function from the
# thinker's prompt (adapter.py:compute_talker_prompt_ids_length, which sums EVERY user
# block). So the question becomes answerable with two small, reversible switches on the
# existing one-request-per-turn path:
#
#   PA_TURN_BLOCKS=1        put each turn's new frames in its OWN user block, so that
#                           "the last user block" means "this turn's new frames".
#                           Also makes the prompt append-only by construction.
#   PA_TALKER_LAST_BLOCK=1  size the talker's prompt from the last user block only.
#
# ARMS (identical except the second switch; both need their own boot because the
# switches are read at import time)
#
#   E640  control:   TURN_BLOCKS=1, TALKER_LAST_BLOCK=0. The talker still pays for every
#                    frame. This is the reference slope, and it also isolates what the
#                    prompt restructure alone costs versus the existing T640 arm.
#   F640  treatment: TURN_BLOCKS=1, TALKER_LAST_BLOCK=1. The talker pays only for the
#                    delta.
#
# WHAT EACH OUTCOME MEANS, written down BEFORE the run
#
#   F640's talker cost goes FLAT while its thinker prompt grows to ~35k tokens
#       -> the talker's cost IS its placeholder length. Incremental talker prefill is
#          worth having, Xiao's proposal is vindicated on both stages, and the
#          session-scoped rewrite has a known payoff to justify its cost.
#   F640's talker cost still grows at roughly E640's slope
#       -> the cost is NOT the placeholder length. Something else scales with the
#          thinker's prompt (the connector shipping per-position conditioning is the
#          leading suspect, and transfers=[0->1=0.00ms] in the OmniTiming line is
#          unmeasured rather than zero). The rewrite would not have helped, and that is
#          a real finding rather than a failure.
#
# THIS IS A LATENCY PROBE, NOT A FEATURE. Requests stay one-per-turn, so the talker's KV
# does not persist; F640 withholds conditioning the talker would legitimately need and
# its AUDIO MAY DEGRADE. Stage-2 audio_duration_s and stage-1 num_tokens_out are
# recorded per turn as covariates and must be reported alongside the latency, because a
# talker that says less is trivially faster.
#
# The bring-up gate (check_talker_probe.py) runs a 6-turn smoke on each boot and refuses
# to spend the 40-minute arm unless the server-side evidence proves the arm is the arm it
# claims to be. This study has twice produced clean-looking null results from arms that
# never entered the new code.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-60}"
SESSIONS="${2:-4}"

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
F640=/data/zx/stimuli/frames640/handheld_walk_talk
LOG=$RES/server_td.log
EVENTS=$RES/stage0_events_td.jsonl
SMOKE=$RES/td_smoke

# Matched to the existing T640 arm so the slopes are comparable: 640x352, append-only,
# gap bracket [8,16], frame cap 284 (the ~281 that fit max_model_len=65,536 at 220
# tokens/frame).
export PA_APPEND_ONLY=1 PA_EVS_MIN_GAP=8 PA_EVS_MAX_GAP=16
MAXF=284
NUMF=16

mkdir -p "$SMOKE"

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

# APPEND to the log, never truncate. Phase 1 of this study truncated per boot and
# destroyed the server-side telemetry of three arms permanently.
start_server() {   # label
  printf "\n===== %s %s =====\n" "$1" "$(date -Is)" >> "$LOG"
  : > "$EVENTS"
  echo ""
  echo "################ server up for: $1 ################"
  echo "  PA_TURN_BLOCKS=$PA_TURN_BLOCKS  PA_TALKER_LAST_BLOCK=$PA_TALKER_LAST_BLOCK"
  echo "  PA_APPEND_ONLY=$PA_APPEND_ONLY  gaps=[$PA_EVS_MIN_GAP,$PA_EVS_MAX_GAP]"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_td.json \
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
      echo "  SERVER FAILED for $1"
      grep -iE "not enough GPU|TimeoutError|RuntimeError" "$LOG" | tail -5 | cut -c1-180
      return 1
    fi
    sleep 5
  done
  echo "  TIMEOUT"; return 1
}

n_boots() { grep -c "^===== " "$LOG" 2>/dev/null || echo 0; }

run_arm() {        # tag gate_kind
  local tag="$1" kind="$2"
  if [ -z "$(find "$F640" -maxdepth 1 -type f -name '*.jpg' -print -quit)" ]; then
    echo "!! ARM $tag ABORTED: no frames in $F640"; return 1
  fi

  # ---- bring-up gate: 6 turns, then prove the arm is what it claims to be ----------
  echo ""
  echo "---- $tag bring-up smoke: 6 turns ----"
  rm -rf "$SMOKE/$tag"; mkdir -p "$SMOKE/$tag"
  timeout 900 $PY_OMNI $H/ttfa_bench.py \
    --users 1 --reps 6 --outdir "$SMOKE/$tag" \
    --frames "$F640" --num-frames "$NUMF" --max-frames "$MAXF" \
    --evs --evs-threshold 0.95 --trace-deltas 2>&1 | tail -2
  sleep 3
  if ! $PY_PA0 $H/check_talker_probe.py --log "$LOG" --arm "$kind" --boot "$(n_boots)"; then
    echo "!! ARM $tag ABORTED by the bring-up gate. No measurement time spent."
    return 1
  fi

  # ---- the arm ---------------------------------------------------------------------
  local out=$RES/vt_u1_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ ARM $tag : $SESSIONS x $REPS turns ================"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_$tag.jsonl --hz 50 --duration-s 14000 &
  local gs=$!
  sleep 1
  for s in $(seq 1 "$SESSIONS"); do
    echo "  --- session $s/$SESSIONS ---"
    timeout 4000 $PY_OMNI $H/ttfa_bench.py \
      --users 1 --reps "$REPS" --outdir "$out" --session "$s" \
      --frames "$F640" --num-frames "$NUMF" --max-frames "$MAXF" \
      --evs --evs-threshold 0.95 --trace-deltas 2>&1 | tail -2
    sleep 4
  done
  kill $gs 2>/dev/null; wait $gs 2>/dev/null
  sleep 3
}

kill_all

# ---- E640: control. per-turn blocks, talker still pays for every frame -------------
export PA_TURN_BLOCKS=1 PA_TALKER_LAST_BLOCK=0
if start_server "E640 control: turn blocks, talker sums all user blocks"; then
  run_arm "E640_high" "control"
fi
kill_all

# ---- F640: treatment. talker pays only for the newest block ------------------------
export PA_TURN_BLOCKS=1 PA_TALKER_LAST_BLOCK=1
if start_server "F640 treatment: turn blocks, talker sizes from the LAST block only"; then
  run_arm "F640_high" "treatment"
fi
kill_all

echo ""
echo "################################################################"
echo "# talker-delta probe done. Analyse with:"
echo "#   analysis/stage_stats_v2.py --log td=$LOG --split-boot"
echo "#   analysis/talker_delta_report.py"
echo "# Compare F640's talker slope against E640's. Report audio_duration_s"
echo "# and stage-1 num_tokens_out for both arms -- a quieter talker is"
echo "# trivially a faster one."
echo "################################################################"
