#!/usr/bin/env bash
# Does video data drive multi-user latency, and through which mechanism?
#
#   run_video_latency.sh [A|B|AB]        default: AB
#
# Two experiments, both against ONE server instance. Everything that varies here
# -- stimulus, num_frames, max_frames, EVS on/off, EVS threshold -- lives in the
# client's `session.config`, so none of it needs a restart. Only the history
# policy would, and this study fixes it at `shipped` so memory cannot confound the
# video question.
#
#   A  CAUSAL. EVS off, so `num_frames` is exactly what the model sees. Sweep
#      1..16 frames on ONE stimulus at 1 and 4 users. Content is held constant, so
#      whatever moves is caused by frame count and nothing else.
#
#   B  REALISTIC. EVS on at the shipped 0.95, sweep stimuli that differ 107x in
#      retained frames (measured offline by analysis/characterize_stimuli.py):
#        screencast            ~1 frame retained of 434   (static screen)
#        talkinghead           ~7 of 502                  (person sitting)
#        handheld_walk_talk    ~107 of 127                (walking, high motion)
#      at 1, 2 and 4 users. This is what real content does through the filter.
#
# Probes on for every run: 50 Hz NVML per-stage, and CUDA-event hooks on the
# vision/audio encoders so the number of frames that actually reached the model is
# measured rather than assumed.
set -uo pipefail
source /home/zx/voice-agent/env.sh

WHICH="${1:-AB}"
RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
F=/data/zx/stimuli/frames
LOG=$RES/server_vl.log
EVENTS=$RES/stage0_events_vl.jsonl
REPS=11

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

# one run = one client-side config against the already-running server
run_one() {          # tag users stimulus num_frames max_frames evs_flag
  local tag="$1" users="$2" stim="$3" nf="$4" mf="$5" evs="$6"
  local out=$RES/vl_$tag
  rm -rf "$out"; mkdir -p "$out"
  echo ""
  echo "---------------- $tag : users=$users stim=$(basename "$stim") nf=$nf mf=$mf $evs ----------------"
  $PY_PA0 $H/gpu_sampler.py --out $RES/gpu_vl_$tag.jsonl --hz 50 --duration-s 1500 &
  local gs=$!
  sleep 1
  timeout 2400 $PY_OMNI $H/ttfa_bench.py \
    --users "$users" --reps "$REPS" --outdir "$out" \
    --frames "$stim" --num-frames "$nf" --max-frames "$mf" $evs 2>&1 | tail -"$((users+1))"
  kill $gs 2>/dev/null; wait $gs 2>/dev/null

  $PY_PA0 /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis/ttfa_decompose.py \
    --traces "$out/ttfa_user*.jsonl" --events "$EVENTS" \
    --gpu $RES/gpu_vl_$tag.jsonl --skip-reps 1 \
    --out $RES/decomp_vl_$tag.json 2>&1 | sed -n '/=== aggregate/,$p' | head -16
  # how many frames actually reached the encoder in this run's window
  $PY_PA0 - "$out" "$EVENTS" <<'EOF'
import json, pathlib, sys, statistics as st
out, ev = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ws = []
for t in out.glob("ttfa_user*.jsonl"):
    for line in t.read_text().splitlines():
        if line.strip():
            ws.append(json.loads(line)["w"])
if not ws or not ev.exists():
    raise SystemExit
lo, hi = min(ws), max(ws)
vt, vc, at, ac = [], 0, [], 0
for line in ev.read_text().splitlines():
    if not line.strip():
        continue
    e = json.loads(line)
    ls = e.get("launch_start")
    if ls is None or not (lo <= ls <= hi):
        continue
    if e["module"] == "vision_encoder":
        vt.append(e.get("ntok", 0)); vc += 1
    elif e["module"] == "audio_encoder":
        at.append(e.get("ntok", 0)); ac += 1
print(f"  vision encoder: {vc} calls, ntok p50={st.median(vt) if vt else 0:.0f}, "
      f"total={sum(vt)}  (~{st.median(vt)/3520:.1f} frames/call; 1 frame = 3520 patches)")
print(f"  audio  encoder: {ac} calls, ntok p50={st.median(at) if at else 0:.0f}")
EOF
  sleep 4
}

kill_all
: > "$LOG"; : > "$EVENTS"
echo "################ starting the one server ################"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_STAGE0_PROBE=1 PA_STAGE0_PROBE_OUT=$RES/stage0_probe_vl.json \
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

if [[ "$WHICH" == *A* ]]; then
  echo ""
  echo "################################################################"
  echo "# A  CAUSAL: EVS off, frame count swept, content held constant"
  echo "################################################################"
  for U in 1 4; do
    for NF in 1 2 4 8 16; do
      run_one "A_u${U}_f${NF}" "$U" "$F/talkinghead" "$NF" 64 "--no-evs"
    done
  done
fi

if [[ "$WHICH" == *B* ]]; then
  echo ""
  echo "################################################################"
  echo "# B  REALISTIC: EVS on (0.95), content swept by motion level"
  echo "################################################################"
  for U in 1 2 4; do
    run_one "B_u${U}_static"  "$U" "$F/screencast"         16 64 "--evs"
    run_one "B_u${U}_low"     "$U" "$F/talkinghead"        16 64 "--evs"
    run_one "B_u${U}_high"    "$U" "$F/handheld_walk_talk" 16 64 "--evs"
  done
fi

kill_all
echo ""
echo "################################################################"
echo "# video-latency sweep done"
echo "################################################################"
