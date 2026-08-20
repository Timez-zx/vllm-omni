#!/usr/bin/env bash
# Fair application/session comparison on one workload and one deploy YAML.
#
#   ONLY_CONTENT=talkinghead ONLY_USERS=4 \
#     MU_DEPLOY=/path/to/deploy.yaml bash run_session_baselines.sh
#
# Every mode delegates to run_mu_matrix.sh, which restarts the engine before
# each cell. Model, GPU topology, questions, media cadence, think time and SLO
# calculation therefore stay fixed; only the application/session policy moves.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODES=${SESSION_BASELINE_MODES:-"stateless_full_replay persistent_incremental stateful_evict_rebuild"}

run_mode() {
  local mode=$1 config=$2
  echo "== session baseline: $mode"
  (
    export RESULT_PREFIX="${RESULT_PREFIX_BASE:-session}_${mode}"
    export MU_SESSION_CFG_JSON="$config"
    bash "$SCRIPT_DIR/run_mu_matrix.sh"
  )
}

for mode in $MODES; do
  case "$mode" in
    stateless_full_replay)
      run_mode "$mode" '{"session_scoped_request":false,"history_max_turns":null,"prefill_frames_on_arrival":false,"prefill_audio_on_arrival":false,"context_compression_trigger_tokens":0}'
      ;;
    persistent_incremental)
      run_mode "$mode" '{"session_scoped_request":true,"evict_engine_request_after_turn":false}'
      ;;
    stateful_evict_rebuild)
      run_mode "$mode" '{"session_scoped_request":true,"evict_engine_request_after_turn":true,"session_roll_settle_s":0,"context_compression_trigger_tokens":0}'
      ;;
    *)
      echo "unknown session baseline mode: $mode" >&2
      exit 2
      ;;
  esac
done
