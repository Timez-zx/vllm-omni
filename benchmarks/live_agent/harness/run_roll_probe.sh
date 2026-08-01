#!/usr/bin/env bash
# Can the session outlive stage-1's max_model_len by ROLLING the engine request?
#
#   run_roll_probe.sh [reps]        default: 30
#
# WHAT IS BEING TESTED. A session-scoped request dies when the TALKER's stored token array
# reaches stage 1's max_model_len (65,536): either the worker crashes writing it into a
# max_model_len-sized buffer, or the scheduler's clamp reaches 0 and the running loop skips the
# request silently forever. The array grows every segment by the delta PLUS the audio codes the
# talker generated, so the limit arrives sooner the more the model SPEAKS. Measured: 66% of the
# wall after 50 short-answer turns, 102% after 27 verbose ones.
#
# The roll retires the request while it is still healthy and opens a fresh one seeded with the
# recent TEXT transcript. Text crosses; the accumulated visual KV does not.
#
# THE SETTINGS ARE CHOSEN TO FORCE SEVERAL ROLLS IN ONE SHORT RUN. The verbose query costs
# ~2,300 talker tokens per turn, and rolling at 20,000 therefore fires roughly every 9 turns,
# so 30 turns should produce about three. A backstop budget of 60,000 is set as well: if
# rolling works it must never trip, and if it does trip the roll is not doing its job.
#
# PREDICTIONS, written before running:
#   * 30/30 turns complete. This is the whole point -- the same settings without rolling died
#     at turn 27.
#   * ~3 rolls, 0 broadcast crashes, 0 wedge reports, 0 budget refusals.
#   * The turn immediately after each roll is SLOWER, because the new request has to prefill
#     the seed from cold. That cost is the price of an unbounded session and is measured here
#     rather than assumed.
#   * talker_est resets to near zero after each roll and climbs again.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-30}"
# Roll threshold, in estimated talker tokens. Parameterised because a run at 20,000 finished
# 12 turns at 18,528 without ever rolling -- response length varies enough that a fixed
# threshold cannot be relied on to fire, and a probe that does not fire tests nothing.
ROLL_AT="${2:-20000}"
FORK=/home/zx/voice-agent/vllm-omni
RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
LOG=$RES/server_roll.log
OUT=$RES/vt_u1_ROLL
F720=/data/zx/stimuli/frames/handheld_walk_talk

VERBOSE_QUERY="Describe in as much detail as you possibly can everything you see in the camera right now: the person, their clothing and posture, the background, the lighting, any objects or text visible, and what they seem to be doing. Be thorough and take your time."

rm -rf "$OUT"; mkdir -p "$OUT"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then
  echo "!! GPU already holds ${used} MiB -- refusing to start."; exit 1
fi

echo "==== which vllm_omni will the server load? ===="
PYTHONPATH="$FORK" $PY_OMNI -c "
import vllm_omni, os
p = os.path.dirname(vllm_omni.__file__)
print('  ', p)
assert '/voice-agent/vllm-omni/' in p, 'NOT the fork -- aborting'
import vllm_omni.entrypoints.openai.video_stream_base as B
for k in ('session_scoped_request','session_roll_at_talker_tokens','session_roll_history_turns'):
    assert k in B.StreamingVideoSessionConfig.model_fields, k
print('   roll config fields present')
" 2>&1 | grep -vE "NVFP4|RuntimeWarning|from .version|^This typically|^Using fallback" \
  || { echo "!! fork not importable -- aborting"; exit 1; }

SERVER_PID=""
trap 'echo "[roll] !! received SIGTERM at $(date -Is) -- killed from outside"' TERM
trap 'echo "[roll] !! received SIGINT at $(date -Is) -- killed from outside"' INT

kill_all() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
    sleep 8
    kill -KILL -"$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 1000 ] && break
    sleep 3
  done
}

CLIENT_ARGS=(--num-frames 16 --max-frames 256 --evs --evs-threshold 0.95 --trace-deltas
             --max-frame-width 640 --max-frame-height 352
             --frame-filter-min-gap 8 --frame-filter-max-gap 16
             --session-scoped-request
             --session-roll-at-talker-tokens "$ROLL_AT"
             --session-roll-history-turns 8
             --session-talker-token-budget 60000)

printf "\n===== ROLL %s =====\n" "$(date -Is)" >> "$LOG"
echo "################ session roll: $REPS turns, roll at $ROLL_AT talker tokens ################"
# Nothing may be inserted between these assignments and the command: a comment after a
# backslash continuation comments out the rest of the line, PYTHONPATH stops being exported,
# and the server silently comes up on site-packages vllm_omni with no session mode at all.
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FORK" \
VLLM_OMNI_LOG_SESSION_OUTPUTS=1 VLLM_OMNI_LOG_TRANSFER=1 \
setsid $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$H/deploy_pc_stage0.yaml" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  [ "$code" = "200" ] && { echo "  ready after ~$((i*5))s"; break; }
  if grep -qiE "not enough GPU memory|Engine core initialization failed" "$LOG"; then
    echo "  SERVER FAILED"; tail -20 "$LOG" | cut -c1-170; kill_all; exit 1
  fi
  sleep 5
  [ "$i" = 300 ] && { echo "  TIMEOUT"; kill_all; exit 1; }
done
SERVER_PID=$(sed -n 's/.*Started server process \[\([0-9]\+\)\].*/\1/p' "$LOG" | tail -1)
kill -0 "$SERVER_PID" 2>/dev/null || SERVER_PID=""
echo "  server pid ${SERVER_PID:-unknown}"

echo ""
timeout 3000 $PY_OMNI $H/ttfa_bench.py --users 1 --reps "$REPS" --outdir "$OUT" --session 1 \
  --frames "$F720" --query "$VERBOSE_QUERY" "${CLIENT_ARGS[@]}" 2>&1 | tail -3
sleep 12
kill_all

echo ""
echo "################ verdict ################"
seg=$(awk '/^===== ROLL/{buf=""} {buf = buf $0 "\n"} END{printf "%s", buf}' "$LOG")
n_sess=$(printf '%s\n' "$seg" | grep -c "\[session\] turn=")
if [ "$n_sess" -eq 0 ]; then
  echo "  !! ABORT: session mode was NEVER ENTERED -- this run says nothing about rolling."
  exit 2
fi
n_done=$(printf '%s\n' "$seg" | grep -c "session\] turn=.* done ")
n_roll=$(printf '%s\n' "$seg" | grep -c "ROLL #")
n_crash=$(printf '%s\n' "$seg" | grep -c "could not broadcast")
n_wedge=$(printf '%s\n' "$seg" | grep -c "looks WEDGED")
n_refuse=$(printf '%s\n' "$seg" | grep -c "REFUSED")
n_never=$(printf '%s\n' "$seg" | grep -c "boundary NEVER ARRIVED")
echo "  turns completed  : $n_done / $REPS      <- 30/30 is the result being sought"
echo "  rolls            : $n_roll"
echo "  broadcast crashes: $n_crash   (must be 0)"
echo "  wedge reports    : $n_wedge   (must be 0)"
echo "  budget refusals  : $n_refuse   (must be 0 -- the roll should get there first)"
echo "  boundary lost    : $n_never   (must be 0)"
echo ""
printf '%s\n' "$seg" | grep -o "ROLL #.*carrying [0-9]* transcript message(s)" | sed 's/^/  /'
echo ""
echo "  talker_est trajectory (should saw-tooth, resetting at each roll):"
printf '%s\n' "$seg" | grep -o "talker_est=[0-9]*" | cut -d= -f2 | paste -sd' ' - | fold -w 100 | sed 's/^/    /'
echo ""
echo "  cold-turn cost of a roll: analysis/roll_report.py"
echo "  log: $LOG"
