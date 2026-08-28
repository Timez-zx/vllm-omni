#!/usr/bin/env bash
# Start a DuplexOmni deployment and wait for health.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
MODE=${1:-fp8}
PORT=${DUPLEXOMNI_PORT:-8092}
MODEL=${DUPLEXOMNI_MODEL:-MuyeHuang/DuplexOmni}
RESULTS_DIR=${DUPLEXOMNI_RESULTS_DIR:-/tmp/vllm-omni-duplexomni}
VLLM_OMNI_BIN=${VLLM_OMNI_BIN:-$(command -v vllm-omni || true)}

case "$MODE" in
  bf16)
    DEPLOY=${DUPLEXOMNI_DEPLOY_CONFIG:-$SCRIPT_DIR/deploy_bf16_3gpu.yaml}
    EXPECTED_GPUS=3
    DEFAULT_GPU_IDS=0,1,2
    ;;
  fp8)
    DEPLOY=${DUPLEXOMNI_DEPLOY_CONFIG:-$SCRIPT_DIR/deploy_fp8_3gpu.yaml}
    EXPECTED_GPUS=3
    DEFAULT_GPU_IDS=0,1,2
    ;;
  pd-bf16)
    DEPLOY=${DUPLEXOMNI_DEPLOY_CONFIG:-$SCRIPT_DIR/deploy_pd_bf16_4gpu.yaml}
    EXPECTED_GPUS=4
    DEFAULT_GPU_IDS=0,1,2,3
    ;;
  pd)
    DEPLOY=${DUPLEXOMNI_DEPLOY_CONFIG:-$SCRIPT_DIR/deploy_pd_fp8_4gpu.yaml}
    EXPECTED_GPUS=4
    DEFAULT_GPU_IDS=0,1,2,3
    ;;
  *) echo "usage: $0 [bf16|fp8|pd-bf16|pd]" >&2; exit 2 ;;
esac
GPU_IDS=${DUPLEXOMNI_GPU_IDS:-$DEFAULT_GPU_IDS}

[ -n "$VLLM_OMNI_BIN" ] && [ -x "$VLLM_OMNI_BIN" ] || {
  echo "set VLLM_OMNI_BIN to the vllm-omni executable" >&2
  exit 2
}
[ -f "$DEPLOY" ] || { echo "deploy config missing: $DEPLOY" >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
[ "${#gpu_ids[@]}" -eq "$EXPECTED_GPUS" ] || {
  echo "DUPLEXOMNI_GPU_IDS must contain exactly $EXPECTED_GPUS GPU ids for mode $MODE" >&2
  exit 2
}

mkdir -p "$RESULTS_DIR"
LOG=${DUPLEXOMNI_LOG:-$RESULTS_DIR/server_${MODE}.log}
PID_FILE=${DUPLEXOMNI_PID_FILE:-$RESULTS_DIR/server.pid}
if [ -s "$PID_FILE" ]; then
  old_pid=$(tr -cd '0-9' < "$PID_FILE")
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "server already running with pid $old_pid" >&2
    exit 2
  fi
fi
for gpu in "${gpu_ids[@]}"; do
  free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free_mib" -ge 60000 ] || {
    echo "GPU $gpu has only ${free_mib} MiB free" >&2
    exit 2
  }
done

PYTHON_BIN=$(dirname -- "$VLLM_OMNI_BIN")/python
if [ -z "${CUDA_HOME:-}" ]; then
  python_env=$(dirname -- "$(dirname -- "$VLLM_OMNI_BIN")")
  conda_cuda=$(dirname -- "$python_env")/cudatk13
  if [ -x "$conda_cuda/bin/nvcc" ] && [ -e "$conda_cuda/lib/libnvrtc.so" ]; then
    export CUDA_HOME=$conda_cuda
  fi
fi
if [ -z "${CUDA_HOME:-}" ]; then
  cuda_home=$(
    "$PYTHON_BIN" -c '
from pathlib import Path
import nvidia
for root in nvidia.__path__:
    candidate = Path(root) / "cu13"
    if (candidate / "bin" / "nvcc").is_file():
        print(candidate)
        break
' 2>/dev/null
  )
  if [ -n "$cuda_home" ]; then
    export CUDA_HOME=$cuda_home
  fi
fi
if [ -n "${CUDA_HOME:-}" ]; then
  cuda_link_dirs="$CUDA_HOME/lib:$CUDA_HOME/lib64"
  # Conda CUDA packages place headers/libs below targets/x86_64-linux while
  # torch extensions conventionally probe CUDA_HOME/include and lib64.
  cuda_target_include="$CUDA_HOME/targets/x86_64-linux/include"
  if [ -d "$cuda_target_include" ]; then
    export CPATH="$cuda_target_include:${CPATH:-}"
  fi
  # NVIDIA's pip CUDA bundle may omit the unversioned libnvrtc.so symlink.
  # Reuse a sibling conda CUDA toolkit as a library supplement in that case.
  python_env=$(dirname -- "$(dirname -- "$VLLM_OMNI_BIN")")
  conda_cuda=$(dirname -- "$python_env")/cudatk13
  if [ -e "$conda_cuda/lib/libnvrtc.so" ]; then
    cuda_link_dirs="$cuda_link_dirs:$conda_cuda/lib:$conda_cuda/targets/x86_64-linux/lib"
  fi
  export PATH="$(dirname -- "$VLLM_OMNI_BIN"):$CUDA_HOME/bin:$PATH"
  export LIBRARY_PATH="$cuda_link_dirs:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$cuda_link_dirs:${LD_LIBRARY_PATH:-}"
fi
export VLLM_OMNI_SAFE_MM_PROCESSOR_CACHE=${VLLM_OMNI_SAFE_MM_PROCESSOR_CACHE:-1}
export VLLM_OMNI_TALKER_TEXT_ONLY=0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" setsid "$VLLM_OMNI_BIN" serve "$MODEL" \
  --omni --deploy-config "$DEPLOY" --served-model-name DuplexOmni \
  --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &
server_pid=$!
echo "$server_pid" > "$PID_FILE"

echo "starting DuplexOmni $MODE on GPUs $GPU_IDS (pid $server_pid)"
for attempt in $(seq 1 600); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ]; then
    echo "ready after about $((attempt * 5)) s; log: $LOG"
    nvidia-smi --id="$GPU_IDS" --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited; log tail:" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

echo "health timeout; log tail:" >&2
tail -40 "$LOG" >&2
exit 1
