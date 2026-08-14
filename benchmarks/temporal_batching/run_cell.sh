#!/usr/bin/env bash
# One measurement cell, with its attribution evidence attached.
#
#   run_cell.sh NAME USERS ARM [DEPLOY] [profile]
#
# ARM decides the env, and the env IS the experiment -- so it is recorded into
# the output directory (meta.json) rather than living only in a shell history:
#   A      native vLLM semantics. Every temporal organ zeroed AND
#          VLLM_OMNI_INLINE_RECV=0. That last one is not optional: the inline
#          receive fix defaults ON, so a baseline run that forgets it silently
#          measures a patched engine (this happened once; see LIVE_VLLM.zh.md
#          section 15).
#   Astar  A plus APPLICATION-level work only: the frame mailbox and the
#          arrival prefill live in the API server, not the scheduler, so an
#          optimally-written application would have them. Engine organs
#          (barrier, pacer, replay, inline send/recv) stay off.
#   T      every live-vllm default ON.
#
# Always samples both GPUs at 1 Hz. With `profile`, also records py-spy folded
# stacks for both engine-core processes through steady state -- that pair is
# what settles "is this wall the hardware or the software".
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE/../live_agent/web_client
MAGE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
PYSPY=/home/ubuntu/miniconda3/envs/mage/bin/py-spy
ENGINE_LOG=/home/ubuntu/data/logs/temporal_engine.log
POOL0=1135280

NAME=${1:?usage: run_cell.sh NAME USERS ARM [DEPLOY] [profile]}
U=${2:?}
ARM=${3:?}
DEPLOY=${4:-deploy_temporal_2gpu_mns80.yaml}
PROFILE=${5:-}
OUT=/home/ubuntu/data/results/$NAME
TURNS=${TURNS:-6}
TRIG=$(( POOL0 * 3 / 4 / U )); TGT=$(( TRIG / 2 ))

A_OFF=(VLLM_OMNI_TEMPORAL_TICK_MS=0 VLLM_OMNI_TEMPORAL_BARRIER=0
       VLLM_OMNI_TEMPORAL_ENGINE=0 VLLM_OMNI_TEMPORAL_INLINE_SEND=0
       VLLM_OMNI_TEMPORAL_REPLAY=0 VLLM_OMNI_TEMPORAL_MAILBOX=0
       VLLM_OMNI_TEMPORAL_FRAME_TICK=0 VLLM_OMNI_TEMPORAL_SLACK_TOKENS=0
       VLLM_OMNI_TEMPORAL_VOCODE_PHASES=0 VLLM_OMNI_STREAM_VOCODER=0
       VLLM_OMNI_TEXT_COALESCE_TOKENS=1 VLLM_OMNI_TEXT_COALESCE_TICKS=0)

case "$ARM" in
  A)     ARM_ENV=("${A_OFF[@]}" VLLM_OMNI_INLINE_RECV=0) ; AP=0 ;;
  # A* = A plus application-level work ONLY. The frame mailbox is API-server
  # code, so it gets its own knob rather than borrowing the pacing tick; the
  # audio arrival prefill is an entrypoint feature arm A never enabled (and
  # some T runs did -- that asymmetry is the reason this arm exists). Engine
  # organs stay off, INLINE_RECV stays off: the point is to find A's wall with
  # the application optimized, not to patch the engine.
  # VLLM_OMNI_STREAM_VOCODER belongs here, not in A_OFF: it is a windowed
  # convolution whose output is bit-identical, i.e. a kernel optimization any
  # deployment would take, not a scheduling organ. Classifying it as a temporal
  # organ kept it off for the A arms while T had it, which tilted the vocoder
  # stage against A by ~2x.
  # The frame mailbox is NOT here. Criterion for arm membership, sharpened:
  # a mechanism that REMOVES self-inflicted synchronization (the compression
  # trigger phase ladder, the warmup calendar) is hygiene any multi-tenant
  # application owes its users, and belongs to the baseline. A mechanism that
  # CREATES synchronization -- holding independent arrivals until a shared
  # clock edge so they batch -- is the periodic idea itself, applied at the
  # input boundary, and belongs to the periodic arm. The frame mailbox is the
  # second kind, so it moves to T.
  Astar) ARM_ENV=("${A_OFF[@]}" VLLM_OMNI_INLINE_RECV=0
                  VLLM_OMNI_STREAM_VOCODER=1) ; AP=1 ;;
  # A* plus the one ENGINE change (inline chunk receive). Separates "does the
  # application work still buy anything once the engine stops parking the
  # consumer" from "does either one alone fix the wall".
  # Everything that is NOT a scheduling decision, on both arms. The test for
  # membership: would someone running stock vLLM want this on its own merits?
  # Windowed vocoder (bit-identical output), inline send and the tick mailbox
  # are transport/kernel optimizations and pass that test; the barrier, pacer,
  # replay, phase groups and slack slot are the scheduling policy under test and
  # do not. Getting this wrong is not a wrong conclusion, it is a broken
  # experiment -- it happened twice (vocoder off for A, inline receive off for
  # T*) and each time it was worth ~2x on the headline metric.
  AstarFix) ARM_ENV=("${A_OFF[@]}" VLLM_OMNI_INLINE_RECV=1
                     VLLM_OMNI_INLINE_RECV_ASYNC=1
                     VLLM_OMNI_STREAM_VOCODER=1
                     VLLM_OMNI_TEMPORAL_INLINE_SEND=1
                     VLLM_OMNI_TEMPORAL_MAILBOX=1) ; AP=1 ;;
  T)     ARM_ENV=(VLLM_OMNI_CELL_ARM=T) ; AP=0 ;;
  # T* = the periodic arm with the SAME application-level treatment the A arms
  # get. Without this the comparison is asymmetric in the other direction: the
  # A arms would carry the audio arrival prefill and T would not.
  # INLINE_RECV_ASYNC is required here too: T runs on the overlap deploy, and
  # without it the placeholder guard disables inline receive entirely (measured:
  # 0 hits / 31468 skips), i.e. the periodic arm silently pays the very
  # per-payload tax the comparison is supposed to hold constant.
  Tstar) ARM_ENV=(VLLM_OMNI_INLINE_RECV_ASYNC=1) ; AP=1 ;;
  *)     echo "unknown arm: $ARM"; exit 2 ;;
esac

CFG='{"context_compression_trigger_tokens": '$TRIG', "context_compression_target_tokens": '$TGT', "context_compression_carry_frames": false'
[ "$AP" = 1 ] && CFG="$CFG"', "prefill_audio_on_arrival": true, "audio_prefill_chunk_s": 1.0'
CFG="$CFG}"

bash "$HERE/run_engine.sh" stop; sleep 12
env "${ARM_ENV[@]}" VLLM_OMNI_LOG_AUDIO_CHUNKS=1 VLLM_OMNI_LOG_SCHED_STEPS=1 \
  ${STEP_GPU:+VLLM_OMNI_LOG_STEP_GPU=1} \
  TB_DEPLOY="$HERE/$DEPLOY" bash "$HERE/run_engine.sh" || { echo "!! boot failed"; exit 1; }
mkdir -p "$OUT"
printf '{"name":"%s","users":%d,"arm":"%s","deploy":"%s","turns":%d,"env":"%s","session_cfg":%s}\n' \
  "$NAME" "$U" "$ARM" "$DEPLOY" "$TURNS" "${ARM_ENV[*]}" "$CFG" > "$OUT/meta.json"

LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
S1=$(grep -o "StageEngineCoreProc_stage1_replica0 pid=[0-9]*" "$ENGINE_LOG" | tail -1 | grep -o "[0-9]*$")
S0=$(grep -o "StageEngineCoreProc_stage0_replica0 pid=[0-9]*" "$ENGINE_LOG" | tail -1 | grep -o "[0-9]*$")

# timestamp first: without it a utilization sample cannot be matched to the
# latency event it is supposed to explain, which is the difference between
# "the GPU is busy 9% of seconds" and "the GPU was busy during THIS spike".
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader,nounits -lms 100 > "$OUT/gpu.csv" 2>/dev/null &
SMI=$!
# Per-PROCESS sm%: GPU1 hosts the talker AND the vocoder, and the aggregate
# number cannot say which of them is holding the card.
nvidia-smi pmon -s um -d 1 -o DT > "$OUT/gpu_pmon.txt" 2>/dev/null &
PMON=$!

(cd "$WEB" && env MU_STAGGER_S=0,40 MU_ENGINE_LOG="$ENGINE_LOG" MU_QUESTIONS=long \
  MU_SESSION_CFG_JSON="$CFG" \
  timeout 3600 "$MAGE_PY" mu_bench.py \
  --users "$U" --content synthetic --turns "$TURNS" \
  --audio-input-s 3 --video-interval-ms 480 --think 2,6 --seed 7 \
  --out "$OUT") &
BENCH=$!

if [ "$PROFILE" = "profile" ]; then
  sleep 75
  sudo env PATH="$(dirname $PYSPY):$PATH" "$PYSPY" record -p "$S1" -d 45 -r 200 -f raw \
    -o "$OUT/stage1.folded" --nonblocking >"$OUT/pyspy.log" 2>&1 || echo "!! py-spy stage1 failed"
  sudo env PATH="$(dirname $PYSPY):$PATH" "$PYSPY" record -p "$S0" -d 20 -r 150 -f raw \
    -o "$OUT/stage0.folded" --nonblocking >>"$OUT/pyspy.log" 2>&1 || echo "!! py-spy stage0 failed"
fi

wait $BENCH
kill $SMI $PMON 2>/dev/null
tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
bash "$HERE/run_engine.sh" stop
"$MAGE_PY" "$HERE/analyze.py" "$OUT" --warmup-turns 2 >/dev/null 2>&1 || true
echo "=== cell $NAME ($ARM, u$U) done ==="
