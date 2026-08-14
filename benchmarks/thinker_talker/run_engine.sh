#!/usr/bin/env bash
# Start (or stop) the two-card thinker+talker+code2wav engine.
#
#   bash run_engine.sh          # start, wait for /health, run one warmup turn
#   bash run_engine.sh stop     # stop it (kills the whole process group)
#
# Layout: thinker on GPU0, talker and code2wav in SEPARATE processes on GPU1.
# Set VLLM_OMNI_COLOCATE_STAGES=2:1 to put the vocoder back inside the talker
# process -- measured worse at scale, see deploy_2gpu.yaml.
#
# Both cards must be essentially free (weights need ~60G on GPU0, ~10G on
# GPU1). This script checks and refuses; it never kills anything itself.
#
# Three deployment facts are passed in as environment, because the server needs
# them and no client can know them: the two KV pool sizes (read off the boot
# log's "GPU KV cache size" lines -- they are deterministic for a given deploy
# yaml and card, and the defaults below are this box's) and the number of
# sessions the deployment is sized for. Together they set the admission ledgers
# and the automatic compression trigger (0.75 * stage0_pool / cap). Get the pool
# numbers wrong and every session gets the wrong trigger, so re-read them after
# any change to the deploy yaml.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FORK=$(cd "$HERE/../.." && pwd)
OMNI_PY=${OMNI_PY:-/home/ubuntu/miniconda3/envs/omni/bin}
PROBE_PY=${PROBE_PY:-/home/ubuntu/miniconda3/envs/mage/bin/python}
LOGDIR=${LOGDIR:-/home/ubuntu/data/logs}
ENGINE_LOG=$LOGDIR/thinker_talker_engine.log
PIDFILE=$LOGDIR/thinker_talker_engine.pid
PORT=${PORT:-8091}
MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY="${TT_DEPLOY:-$HERE/deploy_2gpu.yaml}"

mkdir -p "$LOGDIR"

if [ "${1:-}" = "stop" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    # setsid gave the engine its own process group; kill the group so the
    # stage engine-core subprocesses die too.
    kill -- -"$(cat "$PIDFILE")" 2>/dev/null || kill "$(cat "$PIDFILE")"
    echo "engine stop signal sent"
  else
    echo "engine was not running"
  fi
  exit 0
fi

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "engine already up on :$PORT"; exit 0
fi

for gpu in 0 1; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
  if [ "$free" -lt 80000 ]; then
    echo "!! GPU$gpu has only ${free} MiB free. Stop whatever holds it first:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu"
    exit 1
  fi
done

# CUDA shim: FlashInfer JIT needs a full cu13 toolkit laid out with include/ at
# the top (conda hides the headers under targets/...). Created idempotently.
CUDATK=${CUDATK:-/home/ubuntu/miniconda3/envs/cudatk13}
SHIM=${SHIM:-/home/ubuntu/data/cuda-shim-13.0}
mkdir -p "$SHIM"
ln -sfn "$CUDATK/targets/x86_64-linux/include" "$SHIM/include"
ln -sfn "$CUDATK/lib" "$SHIM/lib"
ln -sfn "$CUDATK/lib" "$SHIM/lib64"
ln -sfn "$CUDATK/bin" "$SHIM/bin"
[ -d "$CUDATK/nvvm" ] && ln -sfn "$CUDATK/nvvm" "$SHIM/nvvm"

[ -s "$ENGINE_LOG" ] && mv -f "$ENGINE_LOG" "$ENGINE_LOG.prev"

echo "engine starting: thinker->GPU0, talker+code2wav->GPU1 (deploy: $(basename "$DEPLOY"))"
echo "  log: $ENGINE_LOG"

HF_HOME=${HF_HOME:-/home/ubuntu/data/hf-omni} \
CUDA_HOME="$SHIM" \
PATH="$SHIM/bin:$PATH" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
VLLM_OMNI_COLOCATE_STAGES="${VLLM_OMNI_COLOCATE_STAGES-}" \
VLLM_OMNI_TALKER_TEXT_ONLY="${VLLM_OMNI_TALKER_TEXT_ONLY:-1}" \
VLLM_OMNI_LOG_SCHED_STEPS="${VLLM_OMNI_LOG_SCHED_STEPS:-0}" \
VLLM_OMNI_STAGE0_KV_POOL_TOKENS="${VLLM_OMNI_STAGE0_KV_POOL_TOKENS:-1132672}" \
VLLM_OMNI_STAGE1_KV_POOL_TOKENS="${VLLM_OMNI_STAGE1_KV_POOL_TOKENS:-2531264}" \
VLLM_OMNI_ADMIT_MAX_SESSIONS="${VLLM_OMNI_ADMIT_MAX_SESSIONS:-64}" \
VLLM_OMNI_LOG_AUDIO_CHUNKS="${VLLM_OMNI_LOG_AUDIO_CHUNKS:-0}" \
TMPDIR=${TMPDIR:-/home/ubuntu/data/tmp} \
setsid "$OMNI_PY/vllm-omni" serve "$MODEL" \
  --omni --deploy-config "$DEPLOY" \
  --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --init-timeout 3000 --stage-init-timeout 1500 ${QWEN_EXTRA_ARGS:-} >> "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
echo "$ENGINE_PID" > "$PIDFILE"

echo "waiting for /health ..."
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo "READY after ~$((i*5))s"; break; }
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "!! engine process died; last log lines:"; tail -25 "$ENGINE_LOG"; exit 1
  fi
  sleep 5
done
curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "!! engine never became healthy; see $ENGINE_LOG"; exit 1; }

# The KV pool lines are the only trusted capacity numbers.
grep -h "GPU KV cache size" "$ENGINE_LOG" | tail -3 || true
# Compare what this boot actually got against what was passed in: a stale pool
# number silently gives every session the wrong compression trigger.
_p0=$(grep -h "GPU KV cache size" "$ENGINE_LOG" | head -1 | grep -o "[0-9,]*$" | tr -d ,)
if [ -n "$_p0" ] && [ "$_p0" != "${VLLM_OMNI_STAGE0_KV_POOL_TOKENS:-1132672}" ]; then
  echo "!! stage-0 pool is $_p0 but VLLM_OMNI_STAGE0_KV_POOL_TOKENS=${VLLM_OMNI_STAGE0_KV_POOL_TOKENS:-1132672}"
  echo "   the automatic compression trigger is derived from that number -- update it."
fi

# Absorb first-request JIT before any measurement.
echo "warmup turn ..."
(cd "$FORK/benchmarks/live_agent/web_client" && \
  timeout 240 "$PROBE_PY" probe.py --direct --turns 1 >/dev/null 2>&1) \
  && echo "warmup done" || echo "warmup skipped (non-fatal)"
