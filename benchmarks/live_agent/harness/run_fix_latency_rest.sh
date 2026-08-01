#!/usr/bin/env bash
# Phase 2: low motion and static, with whichever configuration phase 1 selected.
#
#   run_fix_latency_rest.sh CONFIG FRAMES_DIR NUM_FRAMES APPEND MAXGAP MINGAP [reps] [sessions]
#
# example (the full fix, stage-1 prefix caching on):
#   run_fix_latency_rest.sh deploy_pc_stage01.yaml frames640 16 1 10 4 60 4
#
# Phase 1 measured high motion alone -- the only broken case and the fastest arm -- so
# that a wrong configuration costs 40 minutes rather than three hours. This script
# finishes the other two contents with the winner, using the SAME protocol as the
# baseline so the numbers are comparable: 1 user, four sequential sessions of 60 turns,
# 4 x 59 = 236 scored turns at turn indices 2-60.
#
# Low motion is run before static because static's turn cycle is 25 s (the model talks
# for ~30 s per answer), so it alone is ~100 minutes; getting the cheaper arm banked
# first means an interruption costs less.
#
# Everything is parameterised rather than hard-coded because the point of phase 1 is to
# choose, and a script that silently assumed the answer would make the choice
# unfalsifiable.
set -uo pipefail
source /home/zx/voice-agent/env.sh

CFG_NAME="${1:?config file name under harness/}"
FRAMES_NAME="${2:?stimulus dir name under /data/zx/stimuli/}"
NUM_FRAMES="${3:-16}"
export PA_APPEND_ONLY="${4:-1}"
export PA_EVS_MAX_GAP="${5:-10}"
export PA_EVS_MIN_GAP="${6:-4}"
REPS="${7:-60}"
SESSIONS="${8:-4}"
MAXF="${9:-64}"

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/$CFG_NAME
FDIR=/data/zx/stimuli/$FRAMES_NAME
LOG=$RES/server_fx2.log
EVENTS=$RES/stage0_events_fx2.jsonl

# Tag encodes the configuration so an arm can never be mistaken for a different one
# after the fact. Phase 1 used S_/N720_/N640_/P640_; keep that vocabulary.
RES_TAG=$([ "$FRAMES_NAME" = "frames640" ] && echo 640 || echo 720)
if [ "$PA_APPEND_ONLY" = "0" ]; then
  # shipped frame handling. "C" for control, matching the phase-1 vocabulary where
  # C640_high was the arm that won: shipped re-pick, only the resolution changed.
  TAG_PREFIX="C${RES_TAG}"
elif [ "$CFG_NAME" = "deploy_pc_stage01.yaml" ]; then TAG_PREFIX="P${RES_TAG}"
else TAG_PREFIX="N${RES_TAG}"; fi

echo "phase 2 configuration"
echo "  config      $CFG_NAME"
echo "  frames      $FRAMES_NAME  (num_frames=$NUM_FRAMES, max_frames=${MAXF:-64})"
echo "  append-only $PA_APPEND_ONLY   gaps [$PA_EVS_MIN_GAP,$PA_EVS_MAX_GAP]"
echo "  protocol    1 user, $SESSIONS x $REPS turns per content"
echo "  tags        ${TAG_PREFIX}_low, ${TAG_PREFIX}_static"

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

run_arm() {        # tag stimulus_subdir
  local tag="$1" stim="$2"
  local out=$RES/vt_u1_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ ARM $tag : $SESSIONS x $REPS turns ================"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_$tag.jsonl --hz 50 --duration-s 20000 &
  local gs=$!
  sleep 1
  for s in $(seq 1 "$SESSIONS"); do
    echo "  --- session $s/$SESSIONS ---"
    timeout 5000 $PY_OMNI $H/ttfa_bench.py \
      --users 1 --reps "$REPS" --outdir "$out" --session "$s" \
      --frames "$FDIR/$stim" --num-frames "$NUM_FRAMES" --max-frames "${MAXF:-64}" \
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
: > "$LOG"; : > "$EVENTS"
echo ""
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_fx2.json \
PA_STAGE0_PROBE_EVENTS="$EVENTS" \
nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed|AssertionError" "$LOG"; then
    echo "SERVER FAILED"; grep -iE "not enough GPU|TimeoutError|AssertionError" "$LOG" | tail -5 | cut -c1-180
    exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "TIMEOUT"; tail -20 "$LOG" | cut -c1-180; exit 1; }

run_arm "${TAG_PREFIX}_low"    "talkinghead"
run_arm "${TAG_PREFIX}_static" "screencast"

kill_all
echo ""
echo "################################################################"
echo "# phase 2 done"
echo "################################################################"
