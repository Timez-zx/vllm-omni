#!/usr/bin/env bash
# The multi-user scaling matrix: {1,2,4,8,16} users x {screencast, talkinghead,
# handheld_walk_talk}, 30 turns per user, on the session-mode engine.
#
#   bash run_mu_matrix.sh                    # full matrix, ~3-5 h
#   ONLY_CONTENT=talkinghead ONLY_USERS=2 bash run_mu_matrix.sh   # one cell
#
# The engine is restarted at each content boundary so every content starts
# from a clean engine (no cross-contamination from a previous cell's KV pool
# or any slow leak), and restored to the single-user default config at the
# end. Restarts go through run_qwen_server.sh, which keeps the previous log.
set -uo pipefail

FORK=/home/zx/voice-agent/vllm-omni
WC=$FORK/benchmarks/live_agent/web_client
PYBIN=/home/zx/miniconda3/envs/omni-minicpm/bin/python
RES=/data/zx/results
MU_DEPLOY=$WC/deploy_web_multiuser.yaml
PORT=8091

CONTENTS=${ONLY_CONTENT:-"screencast talkinghead handheld_walk_talk"}
USERS=${ONLY_USERS:-"1 2 4 8 16"}
TURNS=${TURNS:-30}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Kill OUR engine only: the zx-owned `vllm-omni serve` session group. The card
# is shared -- pid matching must never widen beyond our own user and our own
# command line (jkim38's MPS server lives on this card permanently).
stop_engine() {
  local main
  main=$(pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" | head -1 || true)
  if [ -z "$main" ]; then log "no engine running"; return 0; fi
  local pgid
  pgid=$(ps -o pgid= -p "$main" | tr -d ' ')
  log "stopping engine pid=$main pgid=$pgid"
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 1 24); do
    pgrep -g "$pgid" >/dev/null 2>&1 || break
    sleep 5
  done
  if pgrep -g "$pgid" >/dev/null 2>&1; then
    log "engine still up after 120s; SIGKILL"
    kill -KILL -- "-$pgid" 2>/dev/null
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
  local out=$RES/mu_${content}_u${u}
  mkdir -p "$out"
  log "=== cell $content x ${u} users (reps=$reps) -> $out"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader -l 2 > "$out/gpu.csv" 2>/dev/null &
  local sampler=$!
  timeout 7200 "$PYBIN" "$WC/mu_bench.py" \
    --users "$u" --content "$content" --turns "$TURNS" \
    --repeat-sessions "$reps" --out "$out" 2>&1 | tee "$out/driver.log"
  local rc=${PIPESTATUS[0]}
  kill "$sampler" 2>/dev/null
  log "cell done rc=$rc"
  if ! healthy; then
    log "!! engine unhealthy after cell $content/u$u -- restarting before next cell"
    echo "engine_died_after=true" >> "$out/summary_note.txt"
    stop_engine && start_engine "$MU_DEPLOY" || return 1
  fi
}

log "matrix start: contents=[$CONTENTS] users=[$USERS] turns=$TURNS"
for content in $CONTENTS; do
  stop_engine || exit 1
  start_engine "$MU_DEPLOY" || { log "!! engine failed to boot for $content"; exit 1; }
  for u in $USERS; do
    run_cell "$content" "$u" || exit 1
  done
done

log "matrix done; restoring the single-user default config"
stop_engine || exit 1
start_engine "$WC/deploy_web_demo.yaml" || exit 1
log "all done"
