#!/usr/bin/env bash
# Bring up Qwen3-Omni for the browser live session, with the merged optimisations.
#
# A script rather than an inline command on purpose: backgrounding a long
# `cd X && ENV=1 setsid ... &` chain has repeatedly lost either the working
# directory or the redirect in this project, and the symptom (an empty log, or a
# stale one) looks like the server failing rather than the launcher failing.
#
#   bash run_qwen_server.sh            # start, wait for health, report
#
# Nothing may be inserted between the env assignments and the command: a comment
# after a backslash continuation comments out the rest of the line, PYTHONPATH
# silently stops being exported, and the server comes up on site-packages with
# none of our changes.
set -uo pipefail

FORK=/home/zx/voice-agent/vllm-omni
PY=/home/zx/miniconda3/envs/omni-minicpm/bin/vllm-omni
LOG="${QWEN_LOG:-/data/zx/results/qwen_live.log}"
PORT=8091
# DEFAULT MODE (2026-08-08): two processes -- the speech pair (talker +
# code2wav) colocated in one process, the thinker in its own. This is the
# split the evidence picked: the pair's fine-grained per-chunk handoff wants
# one process (in-proc references, kernel overlap where speech starved), the
# thinker's heavy Python bookkeeping wants its own GIL. It also set the
# extreme-load record (128-user rtf 1.13). Override with
#   VLLM_OMNI_COLOCATE_STAGES=""        # three separate processes
#   VLLM_OMNI_COLOCATE_STAGES="2:1,0:1" # tri-colocation (research platform)
# The async deploy config routes both edges through ColocInProcConnector,
# which delegates to SharedMemory automatically for cross-process edges, so
# ONE yaml serves every mode.
export VLLM_OMNI_COLOCATE_STAGES="${VLLM_OMNI_COLOCATE_STAGES-2:1}"
DEPLOY="${DEPLOY_CONFIG:-$FORK/benchmarks/live_agent/web_client/deploy_mu_fp8_s128_async.yaml}"
# QWEN_MODEL swaps the checkpoint without touching this file -- used for the
# FP8 experiment: marksverdhei/Qwen3-Omni-30B-A3B-FP8 is a block-FP8 E4M3
# quant of the SAME Instruct base (thinker+talker FP8; encoders, code2wav,
# embeddings, norms, MoE gates kept bf16).
MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

[ -f "$DEPLOY" ] || { echo "!! deploy config missing: $DEPLOY"; exit 1; }

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free" -lt 80000 ]; then
  echo "!! only ${free} MiB free -- Qwen3-Omni needs ~80 GB. Someone else may be on the card."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  exit 1
fi

echo "== which vllm_omni will load =="
PYTHONPATH="$FORK" $(dirname "$PY")/python -c "
import vllm_omni, os
p = os.path.dirname(vllm_omni.__file__)
print('  ', p)
assert '/voice-agent/vllm-omni/' in p, 'NOT the fork -- aborting'
from vllm_omni.entrypoints.openai.video_stream_base import StreamingVideoSessionConfig as C
need = {'session_scoped_request','session_roll_at_talker_tokens','max_frame_width','frame_filter_min_gap'}
assert need <= set(C.model_fields), 'merged optimisations missing from the config model'
from vllm_omni.distributed.omni_connectors.adapter import TALKER_TEXT_ONLY
print('   optimisations present; talker text-only =', TALKER_TEXT_ONLY)
" 2>&1 | grep -vE "NVFP4|RuntimeWarning|^This typically|^Using fallback|from .version|_version'|patch.py" || exit 1

# Keep the PREVIOUS log instead of truncating it. A crash is investigated after the fact,
# by which time the natural next move is to restart -- and truncating here destroyed the only
# copy of a stage-1 CUDA device-side assert that had just killed the engine. One generation
# back is enough and costs nothing.
[ -s "$LOG" ] && mv -f "$LOG" "$LOG.prev"
: > "$LOG"
# TALKER_TEXT_ONLY is passed through so the A/B can be run without editing this
# file: VLLM_OMNI_TALKER_TEXT_ONLY=0 bash run_qwen_server.sh gives the control arm.
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FORK" \
VLLM_OMNI_TALKER_TEXT_ONLY="${VLLM_OMNI_TALKER_TEXT_ONLY:-1}" \
VLLM_OMNI_LOG_SESSION_OUTPUTS=1 \
setsid "$PY" serve "$MODEL" \
  --omni --deploy-config "$DEPLOY" \
  --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

echo "== waiting for health (about 2-3 minutes) =="
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY after ~$((i*5))s"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    exit 0
  fi
  # Report the deepest exception rather than the first line that contains the
  # word "error", which is usually a benign warning.
  if grep -qE "Engine core initialization failed|not enough GPU memory|ModuleNotFoundError|FileNotFoundError" "$LOG" 2>/dev/null; then
    echo "!! STARTUP FAILED"
    sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "^\S*\s*(\w+Error|\w+Exception):" | tail -4 | cut -c1-200
    exit 1
  fi
  sleep 5
done
echo "!! TIMEOUT after 1000s"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | tail -6 | cut -c1-170
exit 2
