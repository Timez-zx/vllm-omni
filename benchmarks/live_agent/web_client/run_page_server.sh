#!/usr/bin/env bash
# Serve the browser and proxy its WebSocket to the Qwen engine.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${MU_PYTHON:-python3}
PORT=${MU_PAGE_PORT:-${1:-7870}}
BACKEND=${MU_BACKEND_URL:-${2:-ws://127.0.0.1:8091}}

echo "page:   http://127.0.0.1:$PORT/"
echo "engine: $BACKEND/v1/video/chat/stream"
PYTHONPATH="$REPO_ROOT" exec "$PYTHON_BIN" "$SCRIPT_DIR/server.py" \
  --port "$PORT" --ws-backend "$BACKEND"
