#!/usr/bin/env bash
# Fair three-arm RCA for continuous arrival prefill vs query-time batching.
# The arrival arm records one real closed-loop input trace. Both query-time
# arms replay that exact trace; media_fairness.py rejects any content drift.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYBIN=${MU_PYTHON:-python3}
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
RESULTS_ROOT=${RESULTS_ROOT:-/home/ubuntu/data/results/av_prefill_fair_${COMMIT}}
USERS=${USERS:-16}
SEED=${SEED:-17}
TURNS=${TURNS:-10}
WARMUP_TURNS=${WARMUP_TURNS:-2}

: "${MU_FRAMES_DIR:?set MU_FRAMES_DIR to the ordered JPEG frame directory}"
: "${MU_AUDIO_MANIFEST:?set MU_AUDIO_MANIFEST to the real-utterance JSONL manifest}"

if [ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ] && [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
  echo "formal paired RCA requires a clean checkout; commit or set ALLOW_DIRTY_SOURCE=1" >&2
  exit 2
fi

export VLLM_OMNI_BIN=${VLLM_OMNI_BIN:-/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni}
export MU_PYTHON=${MU_PYTHON:-/home/ubuntu/miniconda3/envs/omni/bin/python}
export CUDA_HOME=${CUDA_HOME:-/home/ubuntu/miniconda3/envs/cudatk13}
OMNI_BIN_DIR=$(dirname -- "$VLLM_OMNI_BIN")
export PATH="$OMNI_BIN_DIR:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_OMNI_LOG_SCHED_STEPS=1
export VLLM_OMNI_LOG_REQ_STEPS=1
export VLLM_OMNI_LOG_AUDIO_CHUNKS=1

run_arm() {
  local label=$1 trace_mode=$2 config=$3
  local replay_trace=${4:-}
  echo "== prefill timing RCA: $label ($trace_mode)"
  (
    export RESULTS_DIR="$RESULTS_ROOT/$label"
    export RESULT_PREFIX="$label"
    export USERS SEEDS="$SEED" TURNS WARMUP_TURNS
    export SKIP_DONE=0
    export MU_INPUT_TRACE_MODE="$trace_mode"
    export MU_SESSION_CFG_JSON="$config"
    if [ -n "$replay_trace" ]; then export MU_REPLAY_INPUT_TRACE="$replay_trace"; fi
    bash "$SCRIPT_DIR/run_av_session_ladder.sh"
  )
}

common='"max_frames":256,"fresh_frame_force_append_on_query":true,"log_media_ledger":true'
run_arm arrival record "{$common,\"prefill_frames_on_arrival\":true,\"prefill_audio_on_arrival\":true}"

arrival_cell="$RESULTS_ROOT/arrival/arrival_seed${SEED}_u${USERS}"
trace="$arrival_cell/input_trace.jsonl.gz"
[ -s "$trace" ] || { echo "arrival trace missing: $trace" >&2; exit 1; }

run_arm video_query replay \
  "{$common,\"prefill_frames_on_arrival\":false,\"prefill_audio_on_arrival\":true}" "$trace"
run_arm all_query replay \
  "{$common,\"prefill_frames_on_arrival\":false,\"prefill_audio_on_arrival\":false}" "$trace"

video_cell="$RESULTS_ROOT/video_query/video_query_seed${SEED}_u${USERS}"
all_cell="$RESULTS_ROOT/all_query/all_query_seed${SEED}_u${USERS}"

"$PYBIN" "$REPO_ROOT/benchmarks/live_agent/analysis/media_fairness.py" \
  --cell "arrival=$arrival_cell" --cell "video_query=$video_cell" --cell "all_query=$all_cell" \
  --baseline arrival --require-exact --json-out "$RESULTS_ROOT/media_fairness.json"
"$PYBIN" "$REPO_ROOT/benchmarks/live_agent/analysis/audio_chunk_rca.py" \
  --cell "arrival=$arrival_cell" --cell "video_query=$video_cell" --cell "all_query=$all_cell" \
  --json-out "$RESULTS_ROOT/root_cause.json"

echo "paired RCA complete: $RESULTS_ROOT"
