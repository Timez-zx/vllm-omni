#!/usr/bin/env bash
# Start a pinned Qwen3-Omni benchmark deployment and wait for health.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
RESULTS_DIR=${RESULTS_DIR:-/tmp/vllm-omni-results}
LOG=${QWEN_LOG:-$RESULTS_DIR/qwen_live.log}
PORT=${MU_PORT:-8091}
DEPLOY=${DEPLOY_CONFIG:-$REPO_ROOT/benchmarks/thinker_talker/origin_deploy_3gpu.yaml}
MODEL=${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
VLLM_OMNI_BIN=${VLLM_OMNI_BIN:-$(command -v vllm-omni || true)}
GPU_IDS=${MU_GPU_IDS:-0,1,2}
EXPECTED_DEPLOY_BASENAME=${MU_EXPECTED_DEPLOY_BASENAME:-origin_deploy_3gpu.yaml}
EXPECTED_GPU_COUNT=${MU_EXPECTED_GPU_COUNT:-3}
MIN_FREE_MIB=${MU_MIN_FREE_MIB:-60000}

[ -n "$VLLM_OMNI_BIN" ] && [ -x "$VLLM_OMNI_BIN" ] || {
  echo "set VLLM_OMNI_BIN to the vllm-omni executable" >&2
  exit 2
}
[ -f "$DEPLOY" ] || {
  echo "deploy config missing: $DEPLOY" >&2
  exit 2
}
deploy_name=${DEPLOY##*/}
[ "$deploy_name" = "$EXPECTED_DEPLOY_BASENAME" ] || {
  echo "benchmark run is pinned to $EXPECTED_DEPLOY_BASENAME: $DEPLOY" >&2
  exit 2
}
[ -z "${VLLM_OMNI_COLOCATE_STAGES:-}" ] || {
  echo "$EXPECTED_DEPLOY_BASENAME requires separate stage processes; unset VLLM_OMNI_COLOCATE_STAGES" >&2
  exit 2
}


for variable in \
  VLLM_OMNI_MAILBOX VLLM_OMNI_STREAM_VOCODER VLLM_OMNI_FUSED_SNAKE \
  VLLM_OMNI_TALKER_TEXT_ONLY; do
  if [ -n "${!variable+x}" ]; then
    echo "unset $variable for the canonical baseline" >&2
    exit 2
  fi
done

IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
[ "${#gpu_ids[@]}" -eq "$EXPECTED_GPU_COUNT" ] || {
  echo "MU_GPU_IDS must contain exactly $EXPECTED_GPU_COUNT GPU ids: $GPU_IDS" >&2
  exit 2
}
for gpu in "${gpu_ids[@]}"; do
  free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  [ -n "$free" ] || {
    echo "cannot query GPU $gpu" >&2
    exit 2
  }
  if [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "GPU $gpu has only ${free} MiB free; need at least ${MIN_FREE_MIB} MiB" >&2
    exit 2
  fi
done

PYTHON_BIN=$(dirname -- "$VLLM_OMNI_BIN")/python
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
  export PATH="$(dirname -- "$VLLM_OMNI_BIN"):$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
# vLLM otherwise generates this name independently in spawned stage
# processes.  A single explicit name lets the stage-0 worker attach to the
# object store created by the API-side multimodal processor cache.
if grep -qE '^[[:space:]]*VLLM_OMNI_STAGE0_SHM_MM_CACHE:' "$DEPLOY"; then
  export VLLM_OMNI_STAGE0_SHM_MM_CACHE=1
  export VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME=${VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME:-vllm_omni_mm_${BASHPID}_${RANDOM}}
fi
REPO_ROOT_ENV="$REPO_ROOT" PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c '
import os
from pathlib import Path
import vllm_omni
from vllm_omni.entrypoints.openai.video_stream_base import StreamingVideoSessionConfig

repo = Path(os.environ["REPO_ROOT_ENV"]).resolve()
loaded = Path(vllm_omni.__file__).resolve()
assert repo in loaded.parents, f"loaded {loaded}, expected checkout under {repo}"
required = {
    "context_window_trigger_tokens",
    "context_window_target_tokens",
    "context_window_compaction_headroom_tokens",
    "max_frame_width",
    "frame_filter_min_gap",
}
assert required <= set(StreamingVideoSessionConfig.model_fields)
print(f"vllm_omni: {loaded}")
' || exit 1

mkdir -p "$(dirname -- "$LOG")"
[ -s "$LOG" ] && mv -f "$LOG" "$LOG.prev"
: > "$LOG"

extra_args=()
if [ -n "${QWEN_EXTRA_ARGS:-}" ]; then
  read -r -a extra_args <<< "$QWEN_EXTRA_ARGS"
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$REPO_ROOT" \
VLLM_OMNI_TALKER_TEXT_ONLY="${VLLM_OMNI_TALKER_TEXT_ONLY:-1}" \
setsid "$VLLM_OMNI_BIN" serve "$MODEL" \
  --omni --deploy-config "$DEPLOY" \
  --trust-remote-code --host 127.0.0.1 --port "$PORT" \
  --init-timeout 3000 --stage-init-timeout 1500 \
  "${extra_args[@]}" >> "$LOG" 2>&1 &
server_pid=$!

echo "waiting for engine health on port $PORT"
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$code" = "200" ]; then
    if ! grep -qE "StageEngineCoreProc_stage0.*FlashInfer resolved" "$LOG"; then
      echo "Thinker did not select FlashInfer; refusing a non-comparable formal run" >&2
      kill -TERM -- "-$server_pid" 2>/dev/null || true
      exit 1
    fi
    echo "engine ready after about $((i * 5)) s"
    nvidia-smi --id="$GPU_IDS" --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  if grep -qE "Engine core initialization failed|EngineCore failed to start|not enough GPU memory|ModuleNotFoundError|FileNotFoundError|Could not find nvcc|[Vv]alidation error" "$LOG" 2>/dev/null; then
    echo "engine startup failed; see $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

echo "engine health timeout; see $LOG" >&2
tail -20 "$LOG" >&2
exit 2
