#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${DUPLEXOMNI_RESULTS_DIR:-/tmp/vllm-omni-duplexomni}
PID_FILE=${DUPLEXOMNI_PID_FILE:-$RESULTS_DIR/server.pid}
[ -s "$PID_FILE" ] || { echo "no DuplexOmni pid file"; exit 0; }
server_pid=$(tr -cd '0-9' < "$PID_FILE")
if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
  kill -TERM -- "-$server_pid"
  for _ in $(seq 1 30); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 1
  done
fi
: > "$PID_FILE"
