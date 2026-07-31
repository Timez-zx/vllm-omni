#!/usr/bin/env bash
# PA_SESSION: one engine request per websocket session. Bring-up ladder, then the arm.
#
#   run_session_arm.sh [reps] [sessions]        default: 50 2
#
# WHAT THIS TESTS
#
# Xiao's four steps -- process each frame incrementally, ship it to the talker
# incrementally, have the talker keep it, and at query time handle only the new material --
# all reduce to ONE condition in this codebase. In stage_input_processors/qwen3_omni.py the
# full stage-0 -> stage-1 payload is shipped only when chunk_id == 0; otherwise, if
# request.resumable, it calls _construct_thinker2talker_streaming_input_async_chunk, which
# ships new_prompt_len = thinker_emb.shape[0] rows and ids sliced to
# prompt_token_ids[-new_prompt_len:] -- delta-sized tensors AND delta-sized token ids, so
# the talker's placeholder shrinks with no further change. chunk_id comes from
# put_req_chunk[external_req_id], which is never reset per segment, so one request id for
# the whole session keeps it incrementing and every turn after the first takes that branch.
#
# WHAT IT SHOULD BE WORTH. Measured decomposition of the talker's cost at ~35k prompt
# tokens: ~25% its own prefill of the placeholder, ~75% the payload copy (two [L, 2048]
# bf16 tensors = 8 KB per prompt position, 291 MB at 160 frames, ~225 MB/s). E640/F640
# showed that removing the prefill term alone buys 24-29%. This arm removes both.
#
#   E640 (talker pays for everything)   1,915 ms of talker at ~35k tokens
#   F640 (talker pays for the delta)    1,463 ms          -- prefill term gone
#   THIS ARM, if the mechanism holds      ~100 ms         -- copy term gone as well
#
# ---------------------------------------------------------------------------------------
# THREE THINGS LEARNED DURING BRING-UP, all of which shape the protocol below
# ---------------------------------------------------------------------------------------
#
# 1. ONE BOOT PER SESSION, not four sessions per boot. Ending a resumable session request
#    kills the stage-1 engine core with `assert num_new_tokens > 0`, by either route: the
#    terminal resumable=False sentinel (its prompt is TokensPrompt([0]), so the talker's
#    placeholder is zero-length) or the abort. It happens strictly after the last turn is
#    delivered, so no turn is affected -- but stage 1 is dead afterwards, so a second
#    session on the same server would fail. Hence one boot per session.
#
# 2. 50 TURNS, NOT 60. Under session mode the accumulated sequence is deltas PLUS the
#    generated text folded back in, and nothing truncates it: max_model_len 65,536 is a
#    hard wall. At ~860 tokens/delta and ~150 generated tokens/turn, 60 turns would reach
#    roughly 61k -- too close. 50 turns lands near 50k, still well past the ~35k point
#    where E640/F640 are compared.
#
# 3. THE PER-STAGE TELEMETRY IS GONE for this arm and that is expected, not a bug.
#    StageRequestStats is printed when a request FINISHES, and a resumable session request
#    never does. So the thinker/talker split comes from the CLIENT trace here, which was
#    cross-validated against the server-side split to within ~2% on talker+code2wav across
#    four arms. The x-axis comes from the [PA_SESSION] log line, which reports the running
#    prompt total and the talker placeholder length computed by the connector's own
#    function. Caveat carried into the analysis: that running total counts DELTAS ONLY, so
#    it understates the true accumulated prompt by the folded-in generated text.
#
# LADDER: 3 turns, gate; 12 turns, gate; then the session. All on the same boot as the
# session itself, because the failure modes here are silent rather than slow -- an
# unrecognised feeder object falls back to the per-turn path with no error at all.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-50}"
SESSIONS="${2:-2}"

RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
CFG=$H/deploy_pc_stage0.yaml
F640=/data/zx/stimuli/frames640/handheld_walk_talk
LOG=$RES/server_ss.log
EVENTS=$RES/stage0_events_ss.jsonl
SMOKE=$RES/ss_smoke
OUT=$RES/vt_u1_S640_high

export PA_SESSION=1 PA_APPEND_ONLY=1 PA_EVS_MIN_GAP=8 PA_EVS_MAX_GAP=16
NUMF=16
MAXF=284

mkdir -p "$SMOKE"
rm -rf "$OUT"; mkdir -p "$OUT"

kill_all() {
  # SIGTERM then SIGKILL, and wait for the PROCESS to be gone rather than only for the GPU
  # to look free. Observed failure: three boots of this script left two API-server processes
  # alive 33 and 46 minutes later. They had released their GPU memory -- so the
  # "GPU under 1000 MiB" check passed and the script happily booted another server -- but
  # they were still holding port 8091, which means a client could have been talking to a
  # stale server while the new one failed to bind. Nothing in the data would have looked
  # wrong.
  #
  # Patterns are matched against the recorded PID list only, never re-derived later: a
  # pattern broad enough to catch the runner also matches any monitor tailing its .out file,
  # and killing that yields a bare "exit 144" with no explanation.
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
  printf "\n===== PA_SESSION %s %s =====\n" "$1" "$(date -Is)" >> "$LOG"   # APPEND, never truncate
  : > "$EVENTS"
  echo ""
  echo "################ server up: $1 ################"
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_ss.json \
  PA_STAGE0_PROBE_EVENTS="$EVENTS" \
  PA_SESSION=1 PA_APPEND_ONLY=1 \
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

run_turns() {     # n_turns outdir extra_args...
  local n="$1" od="$2"; shift 2
  timeout 4000 $PY_OMNI $H/ttfa_bench.py \
    --users 1 --reps "$n" --outdir "$od" "$@" \
    --frames "$F640" --num-frames "$NUMF" --max-frames "$MAXF" \
    --evs --evs-threshold 0.95 --trace-deltas 2>&1 | tail -2
}

# ---- ladder, on its own boot (each session needs a fresh one anyway) -----------------
kill_all
boot "ladder" || { kill_all; exit 1; }
for n in 3 12; do
  echo ""
  echo "---- rung: $n turns ----"
  rm -rf "$SMOKE/r$n"; mkdir -p "$SMOKE/r$n"
  run_turns "$n" "$SMOKE/r$n"
  sleep 3
  if ! $PY_PA0 $H/check_session.py --log "$LOG" --expect-turns "$n" --trace "$SMOKE/r$n"; then
    echo "!! rung $n FAILED the gate. Stopping before the arm."
    kill_all; exit 1
  fi
  # A rung ends the session, which kills stage 1 (see note 1), so reboot before the next.
  kill_all
  boot "ladder-r$n-done" || { kill_all; exit 1; }
done

# ---- the arm: one boot per session ---------------------------------------------------
for s in $(seq 1 "$SESSIONS"); do
  echo ""
  echo "================ ARM S640_high session $s/$SESSIONS : $REPS turns ================"
  if [ "$s" -gt 1 ] || true; then kill_all; boot "arm-session-$s" || { kill_all; exit 1; }; fi
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_S640_high_s$s.jsonl --hz 50 --duration-s 5000 &
  gs=$!
  sleep 1
  run_turns "$REPS" "$OUT" --session "$s"
  kill $gs 2>/dev/null; wait $gs 2>/dev/null
  sleep 3
done

kill_all
echo ""
echo "################################################################"
echo "# PA_SESSION arm done.  analysis/session_report.py"
echo "# Compare the talker column against E640 (1,915 ms) and F640 (1,463 ms)"
echo "# at ~35k thinker-prompt tokens."
echo "################################################################"
