#!/usr/bin/env bash
# Serve the live-session page and proxy its websocket to the engine.
#
# A script for the same reason run_qwen_server.sh is one: inline backgrounding
# has lost the working directory and the redirect in this project, and the
# symptom looks like the server failing rather than the launcher failing.
#
#   bash run_page_server.sh          # foreground; Ctrl-C to stop
set -uo pipefail

FORK=/home/zx/voice-agent/vllm-omni
PY=/home/zx/miniconda3/envs/omni-minicpm/bin/python
PORT="${1:-7870}"
BACKEND="${2:-ws://127.0.0.1:8091}"

echo "page      http://127.0.0.1:${PORT}/"
echo "engine    ${BACKEND}/v1/video/chat/stream"
echo
echo "On your Mac:  ssh -N -L ${PORT}:127.0.0.1:${PORT} <server>"
echo "Then open:    http://localhost:${PORT}/"
echo

cd "$FORK"
PYTHONPATH="$FORK" exec "$PY" benchmarks/live_agent/web_client/server.py \
  --port "$PORT" --ws-backend "$BACKEND"
