#!/usr/bin/env bash
# Does prefix caching actually do anything in vllm-omni?
#
# Runs one multi-turn session and logs, per request, how many prompt tokens
# came from the prefix cache (PA_PC_PROBE instrumentation in
# vllm/v1/core/kv_cache_manager.get_computed_blocks).
#
#   run_pc_probe.sh <tag> <deploy-config> <history-policy> [extra bench flags]
#
# Interpretation:
#   hit_frac ~0 on every turn        -> prefix caching is not helping at all
#   hit_frac grows with turn index   -> accumulated context IS being reused
set -uo pipefail
source /home/zx/voice-agent/env.sh
TAG="$1"; CFG="$2"; POL="$3"; EXTRA="${4:-}"
RES=/data/zx/results
LOG=$RES/server_$TAG.log

for pid in $(ps -eo pid=,args= | awk '/vllm serve/ && !/awk/ {print $1}'); do kill "$pid" 2>/dev/null; done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *nvidia-cuda-mps*|"") : ;; *) kill "$pid" 2>/dev/null ;; esac
done
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && break; sleep 3
done
echo "[$TAG] policy=$POL cfg=$(basename "$CFG") extra='$EXTRA'"

: > "$LOG"
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 \
PA_HISTORY_POLICY="$POL" PA_PC_PROBE=1 \
nohup /home/zx/miniconda3/envs/omni/bin/vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni --port 8091 --deploy-config "$CFG" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$LOG" 2>&1 &

READY=0
for i in $(seq 1 240); do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "[$TAG] ready after ~$((i*5))s"; break; fi
  if grep -qiE "not enough GPU memory|TimeoutError|Engine core initialization failed" "$LOG"; then
    echo "[$TAG] SERVER FAILED"; grep -iE "not enough GPU|TimeoutError" "$LOG" | tail -3 | cut -c1-160; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "[$TAG] TIMEOUT"; exit 1; }

OUT=$RES/ttfa_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
timeout 1200 $PY_OMNI /home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness/ttfa_bench.py \
  --users 1 --reps 6 --outdir "$OUT" $EXTRA \
  --frames /data/zx/stimuli/frames/talkinghead 2>&1 | tail -2

echo ""
echo "[$TAG] ===== prefix cache hits per request (stage 0 only shown; all stages logged) ====="
grep -o "\[PC_PROBE\].*" "$LOG" | sed 's/req=video-[0-9a-f]*/req=<video>/' | head -40
echo ""
echo "[$TAG] ===== summary ====="
grep -o "\[PC_PROBE\].*" "$LOG" | $PY_PA0 -c "
import sys, re
rows=[]
for ln in sys.stdin:
    m=re.search(r'prompt_tokens=(\d+) cache_hit_tokens=(\d+) hit_frac=([\d.]+)', ln)
    if m: rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
if not rows:
    print('  no PC_PROBE lines captured'); raise SystemExit
# only requests with a non-trivial prompt are interesting (skip talker/code2wav tiny ones)
big=[r for r in rows if r[0]>500]
print(f'  total requests logged: {len(rows)}   with prompt>500 tok: {len(big)}')
for i,(p,h,f) in enumerate(big):
    print(f'    #{i}: prompt={p:>7} hit={h:>7} frac={f:.3f}')
if big:
    print(f'  max hit_frac among big prompts: {max(f for _,_,f in big):.3f}')
"
