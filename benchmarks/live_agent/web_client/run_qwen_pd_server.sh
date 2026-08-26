#!/usr/bin/env bash
# Start the four-GPU finite-request Thinker P/D exploration deployment.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

export DEPLOY_CONFIG=${DEPLOY_CONFIG:-$REPO_ROOT/benchmarks/thinker_talker/pd_deploy_4gpu.yaml}
export MU_GPU_IDS=${MU_GPU_IDS:-0,1,2,3}
export MU_EXPECTED_GPU_COUNT=4
export MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml
# Keep the latest conditioning snapshot for every active long session. This is
# host memory owned by the orchestrator, not vLLM's GPU attention-KV cache.
export VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES=${VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES:-34359738368}

exec bash "$SCRIPT_DIR/run_qwen_server.sh"
