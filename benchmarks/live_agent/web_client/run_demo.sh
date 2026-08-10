#!/bin/bash
# One-command demo bring-up.
#
# Boots the STABLE, video-validated engine config:
#   - separate stage processes (no colocation)
#   - talker CUDA graphs on, async_scheduling OFF, vocoder eager
# This is the family every video cell of the three-scenario study ran on.
# The experimental configs (coloc / async) are for benchmarks only — async
# killed the talker on the first real browser video session (2026-08-05,
# device-side assert right after a boundary-only segment; see
# boot_coloc11.log around 14:05).
#
# Usage:  bash run_demo.sh
# Then open http://127.0.0.1:8091 (forward the port if remote) and Start Call.
set -u
WC=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/web_client
LOG=/data/zx/results/boot_demo_$(date +%m%d_%H%M).log

# Kill ONLY zx-owned vllm-omni engine process groups (GPU is shared).
PID=$(pgrep -u zx -f "vllm-omni serve" | head -1)
if [ -n "$PID" ]; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  echo "== stopping existing engine (pgid $PGID) =="
  kill -- "-$PGID" 2>/dev/null; sleep 15; kill -9 -- "-$PGID" 2>/dev/null; sleep 5
fi

cd "$WC"
echo "== booting demo engine (log: $LOG) =="
QWEN_LOG="$LOG" DEPLOY_CONFIG=$WC/deploy_mu_fp8_s128_graph_t15.yaml \
  timeout 900 bash run_qwen_server.sh
