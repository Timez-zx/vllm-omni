#!/usr/bin/env bash
# The multi-user scaling matrix: {1,2,4,8,16} users x {screencast, talkinghead,
# handheld_walk_talk}, 30 turns per user, on the session-mode engine.
#
#   bash run_mu_matrix.sh                    # full matrix, ~3-5 h
#   ONLY_CONTENT=talkinghead ONLY_USERS=2 bash run_mu_matrix.sh   # one cell
#
# The engine is restarted before EVERY cell. Per-content restarts were the
# original design; the 8-user screencast cell then demonstrated why that is
# not enough: at ~13.8k tokens x 8 sessions the aggregate crosses the stage-0
# KV pool and the engine does not degrade -- sessions WEDGE (the in-flight
# segment's stop signal is lost, every later query is refused as an overlap),
# and the NEXT cell inherits a poisoned engine. A cell must start clean to be
# interpretable. SKIP_DONE=1 (default) skips cells that already have a
# summary.json, so an interrupted matrix resumes where it left off.
# Restarts go through run_qwen_server.sh, which keeps the previous log.
set -uo pipefail

FORK=/home/zx/voice-agent/vllm-omni
WC=$FORK/benchmarks/live_agent/web_client
PYBIN=/home/zx/miniconda3/envs/omni-minicpm/bin/python
RES=/data/zx/results
# MU_DEPLOY picks the engine config; RESULT_PREFIX keeps result sets apart
# (mu_* = the bf16 baseline matrix, mufp8_* = the FP8 re-measure).
MU_DEPLOY="${MU_DEPLOY:-$WC/deploy_web_multiuser.yaml}"
RESULT_PREFIX="${RESULT_PREFIX:-mu}"
PORT=8091

CONTENTS=${ONLY_CONTENT:-"screencast talkinghead handheld_walk_talk"}
USERS=${ONLY_USERS:-"1 2 4 8 16"}
TURNS=${TURNS:-30}
SKIP_DONE=${SKIP_DONE:-1}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Kill OUR engine only: the zx-owned `vllm-omni serve` session group. The card
# is shared -- pid matching must never widen beyond our own user and our own
# command line (jkim38's MPS server lives on this card permanently).
stop_engine() {
  local pids pgids pgid
  pids=$(pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" || true)
  if [ -z "$pids" ]; then log "no engine running"; return 0; fi
  # A wedged cell can leave more than one process group behind; sweep them all.
  pgids=$(ps -o pgid= -p $pids 2>/dev/null | tr -d ' ' | sort -u)
  for pgid in $pgids; do
    log "stopping engine pgid=$pgid"
    kill -TERM -- "-$pgid" 2>/dev/null
  done
  for _ in $(seq 1 24); do
    pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" >/dev/null || break
    sleep 5
  done
  if pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" >/dev/null; then
    log "engine still up after 120s; SIGKILL"
    for pgid in $pgids; do kill -KILL -- "-$pgid" 2>/dev/null; done
    sleep 5
  fi
  # run_qwen_server.sh refuses to start below 80 GB free; wait for the freeing
  # to actually land in nvidia-smi rather than racing it.
  for _ in $(seq 1 12); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge 80000 ] && { log "GPU freed (${free} MiB)"; return 0; }
    sleep 5
  done
  log "!! GPU did not free (${free} MiB) -- aborting rather than fighting the card"
  return 1
}

start_engine() {  # $1 = deploy yaml
  log "starting engine with $(basename "$1")"
  DEPLOY_CONFIG="$1" bash "$WC/run_qwen_server.sh"
}

healthy() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")" = "200" ]
}

run_cell() {  # $1 content, $2 users
  local content=$1 u=$2 reps=1
  [ "$u" = "1" ] && reps=2   # two sequential sessions: turn indices match other cells
  local out=$RES/${RESULT_PREFIX}_${content}_u${u}
  mkdir -p "$out"
  log "=== cell $content x ${u} users (reps=$reps) -> $out"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader -l 2 > "$out/gpu.csv" 2>/dev/null &
  local sampler=$!
  MU_DEPLOY_CONFIG="$MU_DEPLOY" timeout 7200 "$PYBIN" "$WC/mu_bench.py" \
    --users "$u" --content "$content" --turns "$TURNS" \
    --repeat-sessions "$reps" --out "$out" 2>&1 | tee "$out/driver.log"
  local rc=${PIPESTATUS[0]}
  kill "$sampler" 2>/dev/null
  log "cell done rc=$rc"
  # No mid-loop repair: the next cell boots a fresh engine regardless. Just
  # record whether this cell left the engine dead, for the analysis step.
  if ! healthy; then
    log "NOTE: engine unhealthy after cell $content/u$u (next cell boots fresh anyway)"
    echo "engine_died_after=true" >> "$out/summary_note.txt"
  fi
}

log "matrix start: contents=[$CONTENTS] users=[$USERS] turns=$TURNS skip_done=$SKIP_DONE"
for content in $CONTENTS; do
  for u in $USERS; do
    if [ "$SKIP_DONE" = "1" ] && [ -f "$RES/${RESULT_PREFIX}_${content}_u${u}/summary.json" ]; then
      log "skip $content x $u -- summary.json already present"
      continue
    fi
    stop_engine || exit 1
    start_engine "$MU_DEPLOY" || { log "!! engine failed to boot for $content/u$u"; exit 1; }
    run_cell "$content" "$u" || exit 1
  done
done

log "matrix done; restoring the single-user config"
RESTORE_DEPLOY="${RESTORE_DEPLOY:-$WC/deploy_web_demo.yaml}"
stop_engine || exit 1
start_engine "$RESTORE_DEPLOY" || exit 1
log "all done"
