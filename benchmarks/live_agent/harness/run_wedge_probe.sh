#!/usr/bin/env bash
# One session, long enough to wedge, on the FINAL fork code. Answers two open questions in a
# single boot:
#
#   1. Is the resumable-teardown crash gone? Both fork verification runs ended with stage 1
#      dying on upstream's `assert num_new_tokens > 0`. The cause turned out to be an aborted
#      request left in `skipped_waiting`, which `schedule()`'s FINISHED_ABORTED sweep did not
#      cover (1aed4032). Two earlier attempts at this keyed on `status == WAITING` and could
#      never match, so "no asserts" is the only acceptable evidence -- not "the guard ran".
#
#   2. WHY does the talker wedge? Runs A and B both stopped producing audio mid-session (turn
#      28 and turn 31) with no crash, no traceback and no log line from stage 1 or stage 2,
#      while stage 0 kept generating and kept shipping across the 0->1 edge. 921314e8 makes
#      the scheduler dump its full request table when it schedules nothing for 45s while
#      requests are still tracked, so this run should name the stuck request and its state.
#
# 35 turns, not 50: both runs wedged by turn 31, so 35 reaches it with margin and saves the
# GPU time. If this session gets to 35 clean, that is itself informative -- it would mean the
# wedge is not a simple function of turn count.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-35}"
FORK=/home/zx/voice-agent/vllm-omni
RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
LOG=$RES/server_wedge.log
OUT=$RES/vt_u1_WEDGE
F720=/data/zx/stimuli/frames/handheld_walk_talk

rm -rf "$OUT"; mkdir -p "$OUT"

# Pre-flight: refuse rather than clear the GPU by killing whatever matches a ps pattern. A
# pattern broad enough to catch a stale server also catches the shell commands used to look
# at one, and that has already cost a run here.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then
  echo "!! GPU already holds ${used} MiB -- something is still running. Refusing to start,"
  echo "   because a second server would either fail to allocate or silently contend with"
  echo "   the first and make the timings meaningless. Free it and re-run."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

# Identical to the fork verification arm, so the wedge is reproduced under the same regime
# rather than a new one.
CLIENT_ARGS=(--num-frames 16 --max-frames 256 --evs --evs-threshold 0.95 --trace-deltas
             --max-frame-width 640 --max-frame-height 352
             --frame-filter-min-gap 8 --frame-filter-max-gap 16
             --session-scoped-request)

# The first attempt at this probe was cut short at turn 16 by a SIGTERM that arrived from
# OUTSIDE, mid-session, with a live client -- vllm's launcher logged "shutdown triggered",
# which only its SIGINT/SIGTERM handler emits. So the run is isolated and instrumented rather
# than re-attempted and hoped over:
#   * the script runs in its own session (launched under setsid), so a signal aimed at the
#     launching shell's process group cannot reach it or the server
#   * signals it does receive are LOGGED, so "the probe was killed" can never again be
#     confused with "the session ended on its own"
#   * the server is killed by its RECORDED PID, never by matching a pattern against ps
#     output. A pattern wide enough to catch the server is also wide enough to catch the
#     inspection commands used while debugging it, which has already cost one run here.
SERVER_PID=""
trap 'echo "[probe] !! received SIGTERM at $(date -Is) -- the run was killed from outside"' TERM
trap 'echo "[probe] !! received SIGINT at $(date -Is) -- the run was killed from outside"' INT
trap 'echo "[probe] !! received SIGHUP at $(date -Is)"' HUP

kill_all() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # Negative PID targets the process group, which is the server plus its three stage
    # workers. setsid made the server a group leader, so its PID is its PGID.
    kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
    sleep 8
    kill -KILL -"$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1000 ] && break
    sleep 3
  done
}

printf "\n===== WEDGE probe %s =====\n" "$(date -Is)" >> "$LOG"
echo "################ server up ################"
echo "  probe pid $$  session $(ps -o sid= -p $$ | tr -d ' ')"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FORK" \
VLLM_OMNI_LOG_SESSION_OUTPUTS=1 VLLM_OMNI_LOG_TRANSFER=1 \
setsid $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$H/deploy_pc_stage0.yaml" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
echo "  launched under setsid; the real PID is read from the server's own log below"
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  [ "$code" = "200" ] && { echo "  ready after ~$((i*5))s"; break; }
  if grep -qiE "not enough GPU memory|Engine core initialization failed" "$LOG"; then
    echo "  SERVER FAILED"; tail -20 "$LOG" | cut -c1-170; kill_all; exit 1
  fi
  sleep 5
  [ "$i" = 300 ] && { echo "  TIMEOUT"; kill_all; exit 1; }
done

# `$!` is not trustworthy under setsid -- it forks when it is already a process group leader,
# in which case $! is the short-lived parent and the server has a different PID. uvicorn logs
# its own PID, so take it from there and verify it is alive before relying on it.
SERVER_PID=$(sed -n 's/.*Started server process \[\([0-9]\+\)\].*/\1/p' "$LOG" | tail -1)
if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "  server pid $SERVER_PID  pgid $(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')"
else
  echo "  !! could not determine the server PID; teardown will fall back to the GPU wait"
  SERVER_PID=""
fi

echo ""
echo "================ one session, $REPS turns ================"
timeout 2400 $PY_OMNI $H/ttfa_bench.py --users 1 --reps "$REPS" --outdir "$OUT" --session 1 \
  --frames "$F720" "${CLIENT_ARGS[@]}" 2>&1 | tail -3

# Give the session-end teardown time to land in the log before the server is killed, or the
# question "is the teardown crash gone" gets answered by a race instead of by the fix.
sleep 12
kill_all

echo ""
echo "################ verdict ################"
# Scope to the LAST boot: the log is appended across probes, and counting asserts over the
# whole file would report a previous boot's crash as this one's.
seg=$(awk '/^===== WEDGE probe/{buf=""} {buf = buf $0 "\n"} END{printf "%s", buf}' "$LOG")
n_assert=$(printf '%s\n' "$seg" | grep -c "assert num_new_tokens > 0")
n_wedge=$(printf '%s\n' "$seg" | grep -c "looks WEDGED")
n_done=$(printf '%s\n' "$seg" | grep -c "\] \[session\] turn=.* done ")
echo "  turns completed        : $n_done / $REPS"
echo "  teardown asserts       : $n_assert   (0 = 1aed4032 works)"
echo "  wedge reports          : $n_wedge"
if [ "$n_wedge" -gt 0 ]; then
  echo ""
  echo "  ---- the wedged-stage request table ----"
  printf '%s\n' "$seg" | grep -A 12 "looks WEDGED" | sed 's/^.*\[OmniARScheduler\]/   /' | head -20
fi
echo ""
echo "  full log: $LOG"
echo "  analysis: analysis/session_outputs.py --log $LOG --marker '===== WEDGE'"
