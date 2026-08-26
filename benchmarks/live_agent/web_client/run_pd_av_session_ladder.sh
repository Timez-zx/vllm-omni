#!/usr/bin/env bash
# Run the canonical continuous-AV workload against the four-GPU P/D deployment.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

export MU_DEPLOY=${MU_DEPLOY:-$REPO_ROOT/benchmarks/thinker_talker/pd_deploy_4gpu.yaml}
export MU_GPU_IDS=${MU_GPU_IDS:-0,1,2,3}
export MU_EXPECTED_GPU_COUNT=4
export MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml
export MU_EXPECTED_STAGE_IDS=0,1,2,3
# The orchestrator owns conditioning snapshots in host memory. Keep one latest
# long-session lineage per active user without changing GPU attention-KV space.
export VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES=${VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES:-34359738368}

exec bash "$SCRIPT_DIR/run_av_session_ladder.sh"
