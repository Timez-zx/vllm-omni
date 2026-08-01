#!/usr/bin/env bash
# Does keeping the pixels (not just a note) buy visual detail that text cannot carry?
#
#   run_mm_memory.sh [policy ...]     default: text_memory mm_memory full_mm
#
# EVS stays ON, which is the point: with the shipped filter retaining roughly one
# frame per distinct scene, a whole session's visual history is a handful of frames
# rather than a handful per turn -- and mm_memory dedupes by content-addressed frame
# id (md5 of the jpeg bytes) so each is stored once.
#
# max_frames = num_frames = 1 by default (override with FRAMES_OVERRIDE).
#
# The first run of this used 3, to give dedup some overlap to remove. That was a
# mistake and it invalidated the comparison: with EVS retaining one frame per
# distinct scene, a 3-frame buffer holds THREE scenes, and the subsample takes the
# whole buffer oldest-first -- so the note for a turn described an EARLIER scene.
# Measured on that run: of six notes inspected, three named the wrong scene and two
# scenes (PIANO, VIOLIN) were never written down at all, which dropped text_memory
# to 2/8 recall for reasons that have nothing to do with the policy. The same cause
# put "circle" in a turn whose scene was a square (scene read rate 7/8, 6/8).
#
# With 1, the buffer holds only the newest retained frame, so the current view is
# unambiguously the current scene and the notes describe it. Dedup then has no
# overlap to remove -- that effect has to be measured separately (FRAMES_OVERRIDE=3),
# and its 27 -> 5 saving is reported from that run rather than this one.
#
# All arms get --memory-hint, so the comparison is about WHAT IS STORED rather than
# about whether the model thinks to consult it (the earlier study showed the "first
# screen" phrasing fails without the hint even when the data is there).
#
set -uo pipefail
source /home/zx/voice-agent/env.sh

POLICIES=("$@")
[ ${#POLICIES[@]} -eq 0 ] && POLICIES=(text_memory mm_memory full_mm)

RES=/data/zx/results
CFG=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/deploy_pc_stage0.yaml
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
STIM=/data/zx/stimuli/recall_detail
FRAMES=${FRAMES_OVERRIDE:-1}

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

$PY_OMNI - <<'EOF' || { echo "PATCH NOT INSTALLED -- run harness/patches/apply.sh" >&2; exit 1; }
import sys
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler as C
import inspect
src = inspect.getsource(C)
sys.exit(0 if ("_pa_generate_memory_note" in dir(C) and "mm_memory" in src) else 1)
EOF

for POL in "${POLICIES[@]}"; do
  TAG="mmm_$POL"
  LOG=$RES/server_$TAG.log
  echo ""
  echo "################################################################"
  echo "# policy=$POL   EVS=on   max_frames=$FRAMES"
  echo "################################################################"
  kill_all
  : > "$LOG"

  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
  PA_HISTORY_POLICY="$POL" PA_MEMORY_LOG=1 \
  nohup $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$CFG" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

  READY=0
  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    if [ "$code" = "200" ]; then READY=1; echo "[$POL] ready after ~$((i*5))s"; break; fi
    if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
      echo "[$POL] SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; break
    fi
    sleep 5
  done
  [ "$READY" = 1 ] || { echo "[$POL] NOT READY, skipping"; continue; }

  OUT=$RES/longmem_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
  timeout 2400 $PY_OMNI $H/recall_bench.py \
    --scenes "$STIM" --outdir "$OUT" --policy-label "$POL" \
    --describe-turns 8 --memory-hint \
    --evs --evs-threshold 0.95 \
    --num-frames $FRAMES --max-frames $FRAMES 2>&1 | tail -36

  echo ""
  echo "[$TAG] --- dedup effect ([PA_MM] history frames before -> after) ---"
  grep -o "\[PA_MM\].*" "$LOG" | tail -10 || echo "  (none -- expected for text_memory / full_mm)"
  echo "[$TAG] --- prompt size per request (num_tokens_in) ---"
  grep -o "num_tokens_in *|[^|]*|" "$LOG" | tail -12
  echo "[$TAG] --- notes generated ---"
  echo "  $(grep -c '\[PA_MEM\]' "$LOG" 2>/dev/null || echo 0) lines"
  echo "[$TAG] --- errors ---"
  grep -iE "longer than the maximum|Query processing failed|PA_MEM.*failed" "$LOG" \
    | tail -3 | cut -c1-190 || echo "  (none)"
done

kill_all
echo ""
echo "################################################################"
echo "# mm_memory comparison done"
echo "################################################################"
