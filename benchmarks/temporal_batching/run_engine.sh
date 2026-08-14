#!/usr/bin/env bash
# Engine launcher for the temporal-batching experiment (see DESIGN.zh.md).
#
#   GPU 0  thinker (whole card)
#   GPU 1  talker + code2wav (SEPARATE PROCESSES on the same card)
#
# code2wav used to run as a THREAD inside the talker process
# (VLLM_OMNI_COLOCATE_STAGES=2:1). Measured at 64 sessions: the two stages
# block each other on the GIL -- the talker spent most of its 28.5 ms pass
# waiting and the vocoder most of its 71 ms. Splitting them:
#   GPU1 kernels-resident   50% -> 96%      talker pass 28.5 -> 14.7 ms
#   client miss/stall       5.88%/307ms -> 0.41%/17ms   TTFA p50 1287 -> 693
# i.e. 64 users goes from failing the SLO to passing it. The cost is one real
# IPC hop for codec frames, which shows up as a worse TTFA TAIL at moderate
# load (u56: p99 1800 -> 2577 ms) while the median improves (698 -> 563).
# Default is now split; set VLLM_OMNI_COLOCATE_STAGES=2:1 to get the old
# thread-in-process arrangement back.
#
# Both GPUs must be free: the SoulX avatar server (GPU0) and any web-demo
# engine (GPU1) have to be stopped first -- this script checks and refuses,
# it does not kill anything itself.
#
#   bash run_engine.sh          # start + wait for health + warmup turn
#   bash run_engine.sh stop     # stop the engine
#
# Pacing arms are selected via env at launch (schedulers read them once,
# in the stage engine-core processes):
#
#   VLLM_OMNI_TEMPORAL_TICK_MS=80  bash run_engine.sh     # condition B
#   VLLM_OMNI_TEMPORAL_TICK_MS=160 bash run_engine.sh     # condition C
#   VLLM_OMNI_TEMPORAL_TICK_MS=80 VLLM_OMNI_TEMPORAL_NO_QUANT=1 \
#                                  bash run_engine.sh     # condition D
#   (unset / 0 = condition A, greedy baseline)
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FORK=/home/ubuntu/data/vllm-omni
OMNI_PY=/home/ubuntu/miniconda3/envs/omni/bin
PROBE_PY=/home/ubuntu/miniconda3/envs/mage/bin/python
LOGDIR=/home/ubuntu/data/logs
ENGINE_LOG=$LOGDIR/temporal_engine.log
PORT=8091
MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY="${TB_DEPLOY:-$HERE/deploy_temporal_2gpu.yaml}"

mkdir -p "$LOGDIR"

if [ "${1:-}" = "stop" ]; then
  if [ -f "$LOGDIR/temporal_engine.pid" ] && kill -0 "$(cat "$LOGDIR/temporal_engine.pid")" 2>/dev/null; then
    # setsid gave the engine its own process group; kill the whole group so
    # stage engine-core subprocesses die too.
    kill -- -"$(cat "$LOGDIR/temporal_engine.pid")" 2>/dev/null || kill "$(cat "$LOGDIR/temporal_engine.pid")"
    echo "engine stop signal sent"
  else
    echo "engine was not running"
  fi
  exit 0
fi

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "engine already up on :$PORT"; exit 0
fi

# Both cards must be essentially empty (weights need ~60G + ~10G).
for gpu in 0 1; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
  if [ "$free" -lt 80000 ]; then
    echo "!! GPU$gpu has only ${free} MiB free. Stop whatever holds it first:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$gpu"
    exit 1
  fi
done

# CUDA shim: FlashInfer JIT needs a full cu13 toolkit laid out with include/
# at the top (conda hides headers in targets/...). Same shim run_live_person
# uses; created idempotently.
CUDATK=/home/ubuntu/miniconda3/envs/cudatk13
SHIM=/home/ubuntu/data/cuda-shim-13.0
mkdir -p "$SHIM"
ln -sfn "$CUDATK/targets/x86_64-linux/include" "$SHIM/include"
ln -sfn "$CUDATK/lib" "$SHIM/lib"
ln -sfn "$CUDATK/lib" "$SHIM/lib64"
ln -sfn "$CUDATK/bin" "$SHIM/bin"
[ -d "$CUDATK/nvvm" ] && ln -sfn "$CUDATK/nvvm" "$SHIM/nvvm"

[ -s "$ENGINE_LOG" ] && mv -f "$ENGINE_LOG" "$ENGINE_LOG.prev"

TICK="${VLLM_OMNI_TEMPORAL_TICK_MS:-80}"  # live-vllm: tick default ON
echo "engine starting: thinker->GPU0, talker+code2wav->GPU1  (deploy: $(basename "$DEPLOY"))"
echo "  pacing: tick=${TICK}ms barrier=${VLLM_OMNI_TEMPORAL_BARRIER:-0} engine_loop=${VLLM_OMNI_TEMPORAL_ENGINE:-0} inline_send=${VLLM_OMNI_TEMPORAL_INLINE_SEND:-0} replay=${VLLM_OMNI_TEMPORAL_REPLAY:-0} mailbox=${VLLM_OMNI_TEMPORAL_MAILBOX:-0} lead=${VLLM_OMNI_TEMPORAL_LEAD_MS:-240} thinker_tps=${VLLM_OMNI_TEMPORAL_THINKER_TPS:-25}"
echo "  log: $ENGINE_LOG"

HF_HOME=/home/ubuntu/data/hf-omni \
CUDA_HOME="$SHIM" \
PATH="$SHIM/bin:$PATH" \
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_OMNI_COLOCATE_STAGES="${VLLM_OMNI_COLOCATE_STAGES-}" \
VLLM_OMNI_TALKER_TEXT_ONLY="${VLLM_OMNI_TALKER_TEXT_ONLY:-1}" \
VLLM_OMNI_TEMPORAL_TICK_MS="$TICK" \
VLLM_OMNI_TEMPORAL_LEAD_MS="${VLLM_OMNI_TEMPORAL_LEAD_MS:-240}" \
VLLM_OMNI_TEMPORAL_INITIAL_FRAMES="${VLLM_OMNI_TEMPORAL_INITIAL_FRAMES:-4}" \
VLLM_OMNI_TEMPORAL_THINKER_TPS="${VLLM_OMNI_TEMPORAL_THINKER_TPS:-25}" \
VLLM_OMNI_TEMPORAL_THINKER_BURST="${VLLM_OMNI_TEMPORAL_THINKER_BURST:-16}" \
VLLM_OMNI_TEMPORAL_NO_QUANT="${VLLM_OMNI_TEMPORAL_NO_QUANT:-0}" \
VLLM_OMNI_LOG_SCHED_STEPS="${VLLM_OMNI_LOG_SCHED_STEPS:-}" \
VLLM_OMNI_LOG_AUDIO_CHUNKS="${VLLM_OMNI_LOG_AUDIO_CHUNKS:-0}" \
TMPDIR=/home/ubuntu/data/tmp \
setsid "$OMNI_PY/vllm-omni" serve "$MODEL" \
  --omni --deploy-config "$DEPLOY" \
  --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --init-timeout 3000 --stage-init-timeout 1500 ${QWEN_EXTRA_ARGS:-} >> "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
echo "$ENGINE_PID" > "$LOGDIR/temporal_engine.pid"

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

# KV pool lines: the only trusted capacity numbers.
grep -h "GPU KV cache size" "$ENGINE_LOG" | tail -3 || true

# Absorb first-request JIT before any measurement.
echo "warmup turn ..."
(cd "$FORK/benchmarks/live_agent/web_client" && \
  timeout 240 "$PROBE_PY" probe.py --direct --turns 1 >/dev/null 2>&1) \
  && echo "warmup done" || echo "warmup skipped (non-fatal)"
