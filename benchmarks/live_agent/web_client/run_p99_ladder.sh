#!/usr/bin/env bash
# Audio-only P99 attribution ladder: {1,4,8,16,24,32} users, each a 30-turn
# LONG session (session-scoped request, the default), on the two-process
# default engine (deploy_mu_fp8_s128_async.yaml + VLLM_OMNI_COLOCATE_STAGES=2:1
# from run_qwen_server.sh).
#
# Question under test: as audio-only users scale, p99 TTFA rises -- WHERE does
# the increase come from? mu_bench.py now records absolute per-turn stamps
# (t_q/t_ft/t_fa/t_done), so the analysis can split thinker-side vs speech-side
# and queued-behind-others vs everything-slower.
#
#   bash run_p99_ladder.sh          # full ladder, resumes past done cells
#
# Conventions inherited from run_mu_matrix.sh:
#   - the engine is restarted before EVERY cell (a wedged cell must not poison
#     the next, and each cell needs a clean log slice);
#   - kill only zx-owned vllm-omni processes (jkim38's MPS server shares the card);
#   - u=1 runs two sequential sessions so its turn indices match other cells.
# One addition: the engine log is COPIED into the cell directory after each
# cell -- run_qwen_server.sh keeps only one previous generation, so six cells
# would otherwise destroy four logs needed for server-side attribution.
set -uo pipefail

FORK=/home/zx/voice-agent/vllm-omni
WC=$FORK/benchmarks/live_agent/web_client
PYBIN=/home/zx/miniconda3/envs/omni-minicpm/bin/python
RES=/data/zx/results
LOG="${QWEN_LOG:-/data/zx/results/qwen_live.log}"
# Follows the project default (talker 4k window + FP8 KV since 2026-08-08);
# LADDER_DEPLOY overrides for A/B ladders against other configs.
DEPLOY="${LADDER_DEPLOY:-$WC/deploy_mu_sw4k_kvfp8.yaml}"
PORT=8091

USERS=${ONLY_USERS:-"1 4 8 16 24 32"}
TURNS=${TURNS:-30}
PREFIX=${RESULT_PREFIX:-p99aud}

log() { echo "[$(date +%H:%M:%S)] $*"; }

stop_engine() {
  local pids pgids pgid free
  pids=$(pgrep -u "$USER" -f "vllm-omni.*serve.*Qwen3-Omni" || true)
  if [ -z "$pids" ]; then log "no engine running"; return 0; fi
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
  for _ in $(seq 1 12); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge 80000 ] && { log "GPU freed (${free} MiB)"; return 0; }
    sleep 5
  done
  log "!! GPU did not free (${free} MiB) -- aborting rather than fighting the card"
  return 1
}

healthy() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")" = "200" ]
}

run_cell() {  # $1 = users
  local u=$1 reps=1
  [ "$u" = "1" ] && reps=2
  local out=$RES/${PREFIX}_none_u${u}
  mkdir -p "$out"
  log "=== cell audio-only x ${u} users (reps=$reps, ${TURNS} turns) -> $out"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader -l 2 > "$out/gpu.csv" 2>/dev/null &
  local sampler=$!
  timeout 7200 "$PYBIN" "$WC/mu_bench.py" \
    --users "$u" --content none --turns "$TURNS" \
    --repeat-sessions "$reps" --out "$out" 2>&1 | tee "$out/driver.log"
  local rc=${PIPESTATUS[0]}
  kill "$sampler" 2>/dev/null
  log "cell done rc=$rc"
  cp -f "$LOG" "$out/engine.log" 2>/dev/null
  if ! healthy; then
    log "NOTE: engine unhealthy after u$u (next cell boots fresh anyway)"
    echo "engine_died_after=true" >> "$out/summary_note.txt"
  fi
}

log "ladder start: users=[$USERS] turns=$TURNS prefix=$PREFIX"
for u in $USERS; do
  if [ -f "$RES/${PREFIX}_none_u${u}/summary.json" ]; then
    log "skip u$u -- summary.json already present"
    continue
  fi
  stop_engine || exit 1
  DEPLOY_CONFIG="$DEPLOY" bash "$WC/run_qwen_server.sh" || { log "!! boot failed for u$u"; exit 1; }
  run_cell "$u" || exit 1
done

log "ladder done; restoring the default server"
stop_engine || exit 1
DEPLOY_CONFIG="$DEPLOY" bash "$WC/run_qwen_server.sh" || exit 1
log "all done"
