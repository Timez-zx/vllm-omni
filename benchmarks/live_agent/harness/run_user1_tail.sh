#!/usr/bin/env bash
# Single-user tail latency, comparable to the 4- and 8-user arms.
#
#   run_user1_tail.sh [reps] [sessions]        default: 60 4
#
# WHY THIS IS NOT SIMPLY "--users 1 --reps 240"
#
# A percentile needs samples: p95 wants n >= 200. At 8 users, 60 turns per user
# gave 472 scored turns in ~20 minutes. At 1 user, 236 scored turns from a SINGLE
# session would need 237 consecutive turns -- and TTFA is already known to degrade
# over a session, because the rolling frame buffer (cap 64) keeps filling and the
# prompt keeps growing. Measured on the low-motion arm: turn 1-10 p50 499 ms rising
# to 1,024 ms by turn 51-60, with prompt tokens going 1,441 -> 12,957.
#
# So a 237-turn single session would spend most of its samples at turn indices the
# 8-user run never reached, and the 1-vs-8 comparison would be reading session
# ageing as if it were concurrency. That is the same class of confound that already
# cost this study once (max_frames=3 invalidating a recall comparison).
#
# Instead: FOUR sequential sessions of 60 turns. 4 x 59 = 236 scored turns, which
# matches the 4-user arm exactly (236) and covers turn indices 2-60 -- the same
# range as the 4- and 8-user arms. Turn index is therefore held constant across the
# whole 1/4/8 comparison, and each turn index gets 4 independent samples.
#
#   p90 needs 100 -> SUPPORTED     p95 needs 200 -> SUPPORTED (236)
#   p99 needs 1000 -> NOT supported, and is reported as "worst few turns"
#
# --trace-deltas is on, so this run also answers the ramp question at 1 user with
# real statistics instead of the 14 samples the short probe gave. The added client
# work is one small json line per text delta (~16 before the first sound), which is
# sub-millisecond against a 300 ms TTFA. That assumption is CHECKED rather than
# assumed: the report compares these turns 2-10 against the earlier 11-turn
# vl_B_u1_* arms, which ran without delta tracing. If tracing perturbed anything,
# those two disagree.
#
# Everything else is the shipped config, identical to run_video_tail.sh:
# EVS on at 0.95, num_frames 16, max_frames 64, one server for all arms.
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-60}"
SESSIONS="${2:-4}"

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
F=/data/zx/stimuli/frames
LOG=$RES/server_u1t.log
EVENTS=$RES/stage0_events_u1t.jsonl

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

run_content() {      # tag stimulus
  local tag="$1" stim="$2"
  local out=$RES/vt_u1_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "================ u1_$tag : $SESSIONS sessions x $REPS turns, stim=$(basename "$stim") ================"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vt_u1_$tag.jsonl --hz 50 --duration-s 14000 &
  local gs=$!
  sleep 1
  for s in $(seq 1 "$SESSIONS"); do
    echo "  --- session $s/$SESSIONS ---"
    timeout 4000 $PY_OMNI $H/ttfa_bench.py \
      --users 1 --reps "$REPS" --outdir "$out" --session "$s" \
      --frames "$stim" --num-frames 16 --max-frames 64 --evs --evs-threshold 0.95 \
      --trace-deltas 2>&1 | tail -2
    sleep 4
  done
  kill $gs 2>/dev/null; wait $gs 2>/dev/null

  # One decomposition over all sessions. per_user keys on the meta uid, which every
  # session leaves at 0, so this correctly reports "1 user" while `rows` carries all
  # 4 x 59 turns.
  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$out/ttfa_user*.jsonl" --events "$EVENTS" \
    --gpu $RES/gpu_vt_u1_$tag.jsonl --skip-reps 1 \
    --out $RES/decomp_vt_u1_$tag.json 2>&1 | sed -n '/=== aggregate/,$p' | head -14
  echo "  --- engine pressure signals so far ---"
  grep -ciE "preempt|recompute|swap" "$LOG" 2>/dev/null | sed 's/^/    cumulative preempt-ish log lines: /'
  sleep 5
}

kill_all
: > "$LOG"; : > "$EVENTS"
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_u1t.json \
PA_STAGE0_PROBE_EVENTS="$EVENTS" \
nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 300); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "TIMEOUT"; tail -20 "$LOG" | cut -c1-180; exit 1; }

echo ""
echo "################################################################"
echo "# SINGLE-USER TAIL: 3 contents, $SESSIONS x $REPS turns each"
echo "# high motion first -- it is the arm that matters most"
echo "################################################################"
run_content "high"   "$F/handheld_walk_talk"
run_content "low"    "$F/talkinghead"
run_content "static" "$F/screencast"

kill_all
echo ""
echo "################################################################"
echo "# single-user tail done"
echo "################################################################"
