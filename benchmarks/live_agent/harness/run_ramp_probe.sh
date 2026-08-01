#!/usr/bin/env bash
# Why is the gap between the first TEXT and the first SOUND so long, and why does
# camera motion make it worse?
#
#   run_ramp_probe.sh [users_csv] [reps]        default: 8,1  15
#
# The tail sweep measured that gap ("speech ramp") at 369 / 683 / 2532 ms for
# static / low / high motion at 8 users. It could not explain it, and one obvious
# way to try is a dead end worth recording:
#
#   NOT USABLE: first_text -> response.text.done. The server sends text.done from
#   inside the audio branch, immediately before the first audio chunk
#   (video_stream_base.py:658), so that interval EQUALS first_text -> first_audio
#   by construction. Splitting on it yields exactly 0 ms for talker+code2wav in
#   every arm -- a tautology, not a finding.
#
# So this run timestamps EVERY text delta (--trace-deltas). Text deltas are sent
# as the thinker produces them (video_stream_base.py:696), which makes the thinker
# observable token by token. That distinguishes the two candidate explanations:
#
#   A  THE THINKER IS STILL TALKING TO ITSELF. The talker cannot start until the
#      thinker has emitted enough tokens, and each thinker decode step attends
#      over the whole prompt -- so 16 frames of video make every step slower.
#      Signature: text deltas keep arriving right up to the first audio, and the
#      inter-delta interval grows with prompt size.
#
#   B  THE THINKER IS DONE AND THE TALKER IS THE BOTTLENECK. Stage 1 and 2 are
#      slow or queued behind other users.
#      Signature: text deltas stop early, then a long silence before first audio.
#
# These predict opposite fixes, which is why guessing is not good enough:
# A says cut frames or make thinker decode cheaper, B says give stage 1/2 more room.
#
# 1 user is included as the no-contention reference: whatever remains at 1 user is
# the architecture, and the difference to 8 users is contention.
#
# Same server, same shipped config as the tail sweep (EVS 0.95, 16 frames, 64 cap),
# so the numbers are directly comparable to decomp_vt_*.
set -uo pipefail
source /home/zx/voice-agent/env.sh

USERS_CSV="${1:-8,1}"
REPS="${2:-15}"
IFS=',' read -r -a USERS <<< "$USERS_CSV"

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
F=/data/zx/stimuli/frames
LOG=$RES/server_rp.log
EVENTS=$RES/stage0_events_rp.jsonl

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

run_one() {          # tag users stimulus
  local tag="$1" users="$2" stim="$3"
  local out=$RES/rp_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "---------------- $tag : users=$users stim=$(basename "$stim") reps=$REPS ----------------"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_rp_$tag.jsonl --hz 50 --duration-s 1800 &
  local gs=$!
  sleep 1
  timeout 1700 $PY_OMNI $H/ttfa_bench.py \
    --users "$users" --reps "$REPS" --outdir "$out" \
    --frames "$stim" --num-frames 16 --max-frames 64 --evs --evs-threshold 0.95 \
    --trace-deltas 2>&1 | tail -"$((users+1))"
  kill $gs 2>/dev/null; wait $gs 2>/dev/null

  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$out/ttfa_user*.jsonl" --events "$EVENTS" \
    --gpu $RES/gpu_rp_$tag.jsonl --skip-reps 1 \
    --out $RES/decomp_rp_$tag.json 2>&1 | sed -n '/=== aggregate/,$p' | head -10
  sleep 5
}

kill_all
: > "$LOG"; : > "$EVENTS"
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_rp.json \
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
echo "# RAMP PROBE: every text delta timestamped"
echo "################################################################"
for U in "${USERS[@]}"; do
  run_one "u${U}_static" "$U" "$F/screencast"
  run_one "u${U}_low"    "$U" "$F/talkinghead"
  run_one "u${U}_high"   "$U" "$F/handheld_walk_talk"
done

kill_all
echo ""
echo "################################################################"
echo "# ramp probe done"
echo "################################################################"
