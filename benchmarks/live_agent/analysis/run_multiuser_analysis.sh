#!/usr/bin/env bash
# One command for the whole multi-user analysis, so it is reproducible.
#
#   run_multiuser_analysis.sh [users_csv]        default: 1,2,4
#
# Each policy ran all its user counts against ONE server instance, so the arms of
# a policy share a stage-0 event stream and a server log. session_breakdown needs
# those passed explicitly (it separates them by each run's session wall clock),
# which makes the invocation long -- hence this wrapper.
set -euo pipefail
PY=${PY_PA0:-/home/zx/miniconda3/envs/pa0/bin/python}
A=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/analysis
RES=/data/zx/results
USERS_CSV="${1:-1,2,4}"
IFS=',' read -r -a USERS <<< "$USERS_CSV"

ARMS=()
for POL in shipped text_memory; do
  for U in "${USERS[@]}"; do
    [ -d "$RES/mu_${POL}_u${U}" ] || continue
    ARMS+=("mu_${POL}_u${U}=${POL} u${U}:stage0_events_mu_${POL}.jsonl:server_mu_${POL}.log")
  done
done
[ ${#ARMS[@]} -gt 0 ] || { echo "no mu_* result dirs found under $RES" >&2; exit 1; }

echo "################ resource breakdown ################"
"$PY" "$A/session_breakdown.py" --arms "${ARMS[@]}" \
  --outdir-tpl "{tag}" --out "$RES/session_breakdown.json"

echo ""
echo "################ latency + overhead ################"
"$PY" "$A/multiuser_overhead.py" --users "$USERS_CSV" \
  --policies shipped,text_memory \
  --breakdown "$RES/session_breakdown.json" \
  --out "$RES/multiuser_overhead.json"

echo ""
echo "################ figures ################"
"$PY" "$A/plot_multiuser.py" --users "$USERS_CSV" --policies shipped,text_memory
