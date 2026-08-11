#!/usr/bin/env bash
# Bring up the live-person stack on THIS box (AWS g7e.12xlarge, 2x RTX PRO 6000):
#
#   GPU 0  SoulX-LiveAct avatar server        (mage-liveact-video-call repo, port 6006)
#   GPU 1  Qwen3-Omni engine, all 3 stages    (this repo, port 8091)
#   CPU    page server + avatar-teeing proxy  (this file starts it, port 7870)
#
# The avatar is started by its own repo's scripts and only CHECKED here: it takes
# minutes to warm up and survives engine restarts, so tying its lifetime to this
# script would just make iteration slower.
#
#   bash run_live_person.sh          # start engine (if down) + proxy, wait, report
#   bash run_live_person.sh stop     # stop engine + proxy (avatar left running)
#
# Then from a laptop:   ssh -N -L 7870:127.0.0.1:7870 <this box>
#                       open http://localhost:7870/
set -uo pipefail

FORK=/home/ubuntu/data/vllm-omni
OMNI_PY=/home/ubuntu/miniconda3/envs/omni/bin
PROXY_PY=/home/ubuntu/miniconda3/envs/mage/bin/python   # has fastapi/uvicorn/websockets
LOGDIR=/home/ubuntu/data/logs
ENGINE_LOG=$LOGDIR/omni_engine.log
PROXY_LOG=$LOGDIR/omni_proxy.log
PORT_ENGINE=8091
PORT_PAGE=7870
AVATAR_WS="ws://127.0.0.1:6006/ws"
MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY="${DEPLOY_CONFIG:-$FORK/benchmarks/live_agent/web_client/deploy_web_demo.yaml}"

mkdir -p "$LOGDIR"

if [ "${1:-}" = "stop" ]; then
  pkill -f "vllm-omni serve" 2>/dev/null && echo "engine stopped" || echo "engine was not running"
  pkill -f "server.py --port $PORT_PAGE" 2>/dev/null && echo "proxy stopped" || echo "proxy was not running"
  exit 0
fi

# --- avatar (GPU0): check, don't manage -------------------------------------
if curl -fsS --max-time 3 http://127.0.0.1:6006/health 2>/dev/null | grep -q '"ready":true'; then
  echo "avatar    ready on :6006 (GPU0)"
else
  echo "!! avatar not ready on :6006 -- the call will be voice-only."
  echo "   start it with: cd /home/ubuntu/mage-liveact-video-call && bash liveact/start_avatar.sh"
fi

# --- engine (GPU1) -----------------------------------------------------------
if curl -fsS --max-time 3 http://127.0.0.1:$PORT_ENGINE/health >/dev/null 2>&1; then
  echo "engine    already up on :$PORT_ENGINE"
else
  # The web demo config wants ~80 GB; the old repo's Mage/speech services also
  # live on GPU1 and must not be competing for it.
  for svc in "server.py --host 0.0.0.0 --port 6008" "speech_worker"; do
    pgrep -f "$svc" >/dev/null && {
      echo "!! stopping old service on GPU1: $svc"
      pkill -f "$svc"; sleep 2
    }
  done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  if [ "$free" -lt 80000 ]; then
    echo "!! GPU1 has only ${free} MiB free; Qwen3-Omni needs ~80 GB:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    exit 1
  fi
  [ -s "$ENGINE_LOG" ] && mv -f "$ENGINE_LOG" "$ENGINE_LOG.prev"
  echo "engine    starting on GPU1 (log: $ENGINE_LOG; first start compiles for ~3-6 min)"
  # CUDA_HOME: FlashInfer JIT-compiles MoE kernels during KV-cache init and dies
  # with "Could not find nvcc" without a full toolkit. Must match torch's CUDA
  # major (cu130 -> 13.x); the 12.8 toolkit that builds the avatar's NVFP4
  # kernels is NOT interchangeable here.
  #
  # And it must be a shim, not the conda env itself: conda keeps the headers in
  # targets/x86_64-linux/include, which nvcc resolves internally but the ninja
  # build's HOST g++ units (-I$cuda_home/include) do not -- they died with
  # "fatal error: cublasLt.h: No such file or directory" fifteen minutes into
  # an otherwise-clean compile. Same trick as the 12.8 shim this box already
  # uses for the avatar's NVFP4 build.
  CUDATK=/home/ubuntu/miniconda3/envs/cudatk13
  SHIM=/home/ubuntu/data/cuda-shim-13.0
  mkdir -p "$SHIM"
  ln -sfn "$CUDATK/targets/x86_64-linux/include" "$SHIM/include"
  ln -sfn "$CUDATK/lib" "$SHIM/lib"
  ln -sfn "$CUDATK/lib" "$SHIM/lib64"
  ln -sfn "$CUDATK/bin" "$SHIM/bin"
  [ -d "$CUDATK/nvvm" ] && ln -sfn "$CUDATK/nvvm" "$SHIM/nvvm"
  HF_HOME=/home/ubuntu/data/hf-omni \
  CUDA_HOME="$SHIM" \
  PATH="$SHIM/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=1 \
  VLLM_OMNI_COLOCATE_STAGES="${VLLM_OMNI_COLOCATE_STAGES-2:1}" \
  VLLM_OMNI_TALKER_TEXT_ONLY="${VLLM_OMNI_TALKER_TEXT_ONLY:-1}" \
  TMPDIR=/home/ubuntu/data/tmp \
  setsid "$OMNI_PY/vllm-omni" serve "$MODEL" \
    --omni --deploy-config "$DEPLOY" \
    --trust-remote-code --host 127.0.0.1 --port "$PORT_ENGINE" \
    --init-timeout 3000 --stage-init-timeout 1500 ${QWEN_EXTRA_ARGS:-} >> "$ENGINE_LOG" 2>&1 &
  ENGINE_PID=$!
  echo "$ENGINE_PID" > "$LOGDIR/omni_engine.pid"

  echo "          waiting for /health ..."
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT_ENGINE/health" 2>/dev/null)
    [ "$code" = "200" ] && { echo "          READY after ~$((i*5))s"; break; }
    # Liveness by PID, not by name: vLLM rewrites its process title to
    # "APIServer", so a pgrep for "vllm-omni serve" stops matching seconds
    # after launch and reads as a death that never happened.
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      echo "!! engine process died; last log lines:"; tail -25 "$ENGINE_LOG"; exit 1
    fi
    sleep 5
  done
  curl -fsS --max-time 3 "http://127.0.0.1:$PORT_ENGINE/health" >/dev/null 2>&1 || {
    echo "!! engine never became healthy; see $ENGINE_LOG"; exit 1; }

  # Absorb the first-request cliff before a human hits it. The engine's own
  # [TIMING] log showed first_audio=44.7s on the first request after boot
  # (Triton kernels JIT-compiling during inference: rotary, mrope, SnakeBeta)
  # and ~0.45s on every request since. One synthetic turn pays that here.
  echo "          warmup turn (absorbs first-request JIT) ..."
  (cd "$FORK/benchmarks/live_agent/web_client" && \
    timeout 180 "$PROXY_PY" probe.py --direct --turns 1 >/dev/null 2>&1) \
    && echo "          warmup done" || echo "          warmup skipped (non-fatal)"
fi

# --- page/proxy (CPU) ---------------------------------------------------------
pkill -f "server.py --port $PORT_PAGE" 2>/dev/null && sleep 1
cd "$FORK/benchmarks/live_agent/web_client"
setsid "$PROXY_PY" server.py --port "$PORT_PAGE" \
  --ws-backend "ws://127.0.0.1:$PORT_ENGINE" \
  --avatar "$AVATAR_WS" >> "$PROXY_LOG" 2>&1 &
sleep 2
curl -fsS --max-time 3 "http://127.0.0.1:$PORT_PAGE/healthz" >/dev/null 2>&1 \
  && echo "proxy     up on :$PORT_PAGE (avatar tee -> $AVATAR_WS)" \
  || { echo "!! proxy failed; see $PROXY_LOG"; tail -10 "$PROXY_LOG"; exit 1; }

echo
echo "open from your laptop:"
echo "  ssh -N -L $PORT_PAGE:127.0.0.1:$PORT_PAGE ubuntu@<this-box>"
echo "  http://localhost:$PORT_PAGE/"
