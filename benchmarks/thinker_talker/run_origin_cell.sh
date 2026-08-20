#!/usr/bin/env bash
# One capacity cell against the NATIVE vllm-omni deployment: `vllm-omni serve
# --omni --deploy-config <yaml>` driving the upstream thinker -> talker ->
# code2wav pipeline over /v1/video/chat/stream. Engine code is untouched.
#
# Prompt shape is upstream's own: each turn is a fresh request carrying
# message_history[-2:] text-only plus num_frames sampled from the frame buffer,
# so the prompt stays O(1) (~300-400 tokens) no matter how long the dialog runs.
#
# The ONLY delta from vllm_omni/deploy/qwen3_omni_moe.yaml is configuration:
# fp8 weights + fp8 KV (to fit VRAM) and stage-0 enable_prefix_caching.
# No VLLM_OMNI_* env vars -- upstream reads none of them.
#
#   run_origin_cell.sh NAME USERS [SEED]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WEB=$HERE
MAGE_PY=${MAGE_PY:-/home/ubuntu/miniconda3/envs/mage/bin/python}
OMNI_BIN=${OMNI_BIN:-/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni}
export HF_HOME=${HF_HOME:-/home/ubuntu/data/hf-omni}
export HF_HUB_OFFLINE=1
MODEL="${QWEN_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
ENGINE_LOG=${ENGINE_LOG:-/home/ubuntu/data/logs/thinker_talker_origin_engine.log}

NAME=${1:?usage: run_origin_cell.sh NAME USERS [SEED]}
U=${2:?}
SEED=${3:-7}
TURNS=${TURNS:-10}
OUT=${RESULTS_DIR:-/home/ubuntu/data/results}/$NAME

# Toolchain env copied from run_engine_pd.sh (fork branch): flashinfer JIT needs
# nvcc (CUDA_HOME shim over the cudatk13 conda env) and torch.cpp_extension
# shells out to bare `ninja` (omni bin on PATH). Pure toolchain -- no fork
# semantics; upstream reads none of the VLLM_OMNI_* vars so none are set.
CUDATK=${CUDATK:-/home/ubuntu/miniconda3/envs/cudatk13}
SHIM=${SHIM:-/home/ubuntu/data/cuda-shim-13.0}
mkdir -p "$SHIM"
ln -sfn "$CUDATK/targets/x86_64-linux/include" "$SHIM/include"
ln -sfn "$CUDATK/lib" "$SHIM/lib"
ln -sfn "$CUDATK/lib" "$SHIM/lib64"
ln -sfn "$CUDATK/bin" "$SHIM/bin"
[ -d "$CUDATK/nvvm" ] && ln -sfn "$CUDATK/nvvm" "$SHIM/nvvm"

# A stale instance of this script sitting in its health-wait loop will race a
# fresh launch (three did exactly that on 08-19). Kill older instances -- but
# never an ancestor: the launching shell's command line contains this script's
# name too, so a plain pgrep match takes down the caller as well.
_anc=" $$ "
_p=$$
while [ -n "$_p" ] && [ "$_p" != "0" ] && [ "$_p" != "1" ]; do
  _p=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
  [ -n "$_p" ] && _anc="$_anc$_p "
done
for sib in $(pgrep -f "run_origin_cell\.sh" 2>/dev/null); do
  case "$_anc" in *" $sib "*) continue ;; esac
  kill "$sib" 2>/dev/null
done
unset _anc _p sib
pkill -f "bin/vllm-omni serve" 2>/dev/null; sleep 8
: > "$ENGINE_LOG"
PYTHONHASHSEED=0 \
CUDA_HOME="$SHIM" \
PATH="$SHIM/bin:$(dirname "$OMNI_BIN"):$PATH" \
TMPDIR=${TMPDIR:-/home/ubuntu/data/tmp} \
setsid "$OMNI_BIN" serve "$MODEL" --omni --port 8091 \
  --deploy-config "${ORIGIN_DEPLOY:-$HERE/origin_deploy.yaml}" \
  --init-timeout 3000 --stage-init-timeout 1500 >> "$ENGINE_LOG" 2>&1 &
echo "origin engine starting (upstream per-turn mode); log: $ENGINE_LOG"
for i in $(seq 1 900); do
  curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:8091/health >/dev/null || { echo "!! boot failed"; exit 1; }
echo "engine healthy"

mkdir -p "$OUT"
printf '{"name":"%s","users":%d,"seed":%d,"turns":%d,"deploy":"origin_deploy.yaml","mode":"per_turn_upstream"}\n' \
  "$NAME" "$U" "$SEED" "$TURNS" > "$OUT/meta.json"

LOG_OFF=$(stat -c%s "$ENGINE_LOG" 2>/dev/null || echo 0)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 200 > "$OUT/gpu.csv" 2>/dev/null &
SMI=$!

(cd "$WEB" && env \
  MU_STAGGER_S="${MU_STAGGER_S:-0,40}" \
  MU_QUESTIONS=mixed \
  MU_ENGINE_LOG="$ENGINE_LOG" \
  timeout 3600 "$MAGE_PY" mu_bench.py \
  --users "$U" --content synthetic --video-interval-ms 480 \
  --turns "$TURNS" --audio-input-s 3 --think 2,6 --seed "$SEED" \
  --out "$OUT") &
BENCH=$!
wait $BENCH
kill $SMI 2>/dev/null
tail -c +$((LOG_OFF + 1)) "$ENGINE_LOG" > "$OUT/engine_slice.log" 2>/dev/null || true
pkill -f "vllm-omni serve" 2>/dev/null
"$MAGE_PY" "$HERE/analyze.py" "$OUT" --warmup-turns 2 || true
echo "=== ORIGIN cell $NAME (u$U seed=$SEED) done ==="
