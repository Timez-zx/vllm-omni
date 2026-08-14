#!/usr/bin/env bash
# One measurement cell: N sessions x TURNS turns, with the evidence needed to
# interpret it attached.
#
#   run_cell.sh NAME USERS [DEPLOY] [profile]
#
# Writes /home/ubuntu/data/results/NAME/
#   turns.jsonl       per-turn client record (mu_bench)
#   metrics.json      the two SLO numbers (analyze.py)
#   meta.json         users, turns, deploy, session config, and EVERY
#                     VLLM_OMNI_* in the environment at launch
#   gpu.csv           both cards at 10 Hz, timestamped
#   gpu_pmon.txt      per-PROCESS sm% (GPU1 hosts talker AND vocoder; the
#                     aggregate number cannot say which one holds the card)
#   engine_slice.log  exactly the engine log this run produced
#
# meta.json exists because "which knobs did that cell actually have" has
# invalidated more measurements here than any analysis error: a knob passed as a
# command-line prefix otherwise lives only in shell history, and unset does not
# mean off (vllm_omni/core/sched/runtime_flags.py has ON defaults).
#
# With `profile`, also records py-spy folded stacks for both engine-core
# processes through steady state -- that pair is what settles "is this wall the
# hardware or the software".
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=${MAGE_PY:-/home/ubuntu/miniconda3/envs/mage/bin/python}
PYSPY=${PYSPY:-/home/ubuntu/miniconda3/envs/mage/bin/py-spy}
ENGINE_LOG=${ENGINE_LOG:-/home/ubuntu/data/logs/thinker_talker_engine.log}

NAME=${1:?usage: run_cell.sh NAME USERS [DEPLOY] [profile]}
U=${2:?}
DEPLOY=${3:-deploy_2gpu.yaml}
PROFILE=${4:-}
OUT=${RESULTS_DIR:-/home/ubuntu/data/results}/$NAME
TURNS=${TURNS:-6}

# The session config is EMPTY by default: the server's defaults are the
# baseline (see StreamingVideoSessionConfig), and the compression trigger is
# derived from the deployment facts run_engine.sh passes in (0.75 * stage-0 pool
# / session cap) rather than computed here. This is deliberate -- while these
# numbers lived in this script, the server and the thing being measured were two
# different systems.
#
# TRIG overrides the trigger for one cell: set it high enough that compression
# never fires within the run and the cell isolates "is there a latency problem at
# all" from "is compression handled badly".
if [ -n "${TRIG:-}" ]; then
  CFG='{"context_compression_trigger_tokens": '"$TRIG"', "context_compression_target_tokens": '"$(( TRIG / 2 ))"'}'
else
  CFG='{}'
fi

bash "$HERE/run_engine.sh" stop; sleep 12
env VLLM_OMNI_LOG_AUDIO_CHUNKS=1 VLLM_OMNI_LOG_SCHED_STEPS=1 \
  ${STEP_GPU:+VLLM_OMNI_LOG_STEP_GPU=1} \
  TT_DEPLOY="$HERE/$DEPLOY" bash "$HERE/run_engine.sh" || { echo "!! boot failed"; exit 1; }
mkdir -p "$OUT"
INHERITED=$(env | grep -E "^VLLM_OMNI_" | sort | tr '\n' ' ')
printf '{"name":"%s","users":%d,"deploy":"%s","turns":%d,"inherited_env":"%s","session_cfg":%s}\n' \
  "$NAME" "$U" "$DEPLOY" "$TURNS" "$INHERITED" "$CFG" > "$OUT/meta.json"

LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
S1=$(grep -o "StageEngineCoreProc_stage1_replica0 pid=[0-9]*" "$ENGINE_LOG" | tail -1 | grep -o "[0-9]*$")
S0=$(grep -o "StageEngineCoreProc_stage0_replica0 pid=[0-9]*" "$ENGINE_LOG" | tail -1 | grep -o "[0-9]*$")

# Timestamp first: without it a utilization sample cannot be matched to the
# latency event it is supposed to explain, which is the difference between "the
# GPU is busy 9% of seconds" and "the GPU was busy during THIS spike".
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader,nounits -lms 100 > "$OUT/gpu.csv" 2>/dev/null &
SMI=$!
nvidia-smi pmon -s um -d 1 -o DT > "$OUT/gpu_pmon.txt" 2>/dev/null &
PMON=$!

(cd "$WEB" && env MU_STAGGER_S="${MU_STAGGER_S:-0,40}" MU_ENGINE_LOG="$ENGINE_LOG" \
  MU_QUESTIONS=long MU_SESSION_CFG_JSON="$CFG" \
  timeout 3600 "$MAGE_PY" mu_bench.py \
  --users "$U" --content synthetic --turns "$TURNS" \
  --audio-input-s 3 --video-interval-ms 480 --think 2,6 --seed 7 \
  --out "$OUT") &
BENCH=$!

if [ "$PROFILE" = "profile" ]; then
  sleep 75
  sudo env PATH="$(dirname "$PYSPY"):$PATH" "$PYSPY" record -p "$S1" -d 45 -r 200 -f raw \
    -o "$OUT/stage1.folded" --nonblocking >"$OUT/pyspy.log" 2>&1 || echo "!! py-spy stage1 failed"
  sudo env PATH="$(dirname "$PYSPY"):$PATH" "$PYSPY" record -p "$S0" -d 20 -r 150 -f raw \
    -o "$OUT/stage0.folded" --nonblocking >>"$OUT/pyspy.log" 2>&1 || echo "!! py-spy stage0 failed"
fi

wait $BENCH
kill $SMI $PMON 2>/dev/null
tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
bash "$HERE/run_engine.sh" stop
"$MAGE_PY" "$HERE/analyze.py" "$OUT" --warmup-turns 2 || true
"$MAGE_PY" "$HERE/verify_cell.py" "$OUT" || true
echo "=== cell $NAME (u$U) done ==="
