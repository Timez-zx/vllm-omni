#!/usr/bin/env bash
# Compare session/KV policies on one identical continuous-AV workload plan.
# Required inputs are documented by run_av_session_ladder.sh.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODES=${SESSION_BASELINE_MODES:-"stateless_full_replay persistent_incremental stateful_evict_rebuild"}

run_mode() {
  local mode=$1 config=$2
  echo "== continuous AV session baseline: $mode"
  (
    export RESULT_PREFIX="${RESULT_PREFIX_BASE:-avsession}_${mode}"
    export MU_SESSION_CFG_JSON="$config"
    bash "$SCRIPT_DIR/run_av_session_ladder.sh"
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
      run_mode "$mode" '{"session_scoped_request":true,"evict_engine_request_after_turn":true,"session_roll_history_turns":8,"session_roll_settle_s":0,"context_compression_trigger_tokens":0}'
      ;;
    *)
      echo "unknown session baseline mode: $mode" >&2
      exit 2
      ;;
  esac
done
