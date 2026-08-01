#!/usr/bin/env bash
# Does the talker need the user block's IMAGE/AUDIO positions, or only its text?
#
#   run_talker_textonly_ab.sh [reps]        default: 6  (turn 1 dropped as cold -> 5 analysed)
#
# WHAT IS BEING TESTED. Qwen3-Omni feeds its talker every user block, with the multimodal
# positions filled by hidden_projection(thinker last-layer hidden). One 720p frame is 880 such
# positions and they are never released, so they are the larger half of what drives stage 1
# toward max_model_len -- the wall session rolling exists to work around.
#
# MiniCPM-o 4.5, an independently designed model in the same thinker/talker family, gives its
# speech decoder only the GENERATED text tokens and no visual positions at all. So the
# architecture does not require them. These weights were TRAINED with them, which is why this
# is an experiment and not a cleanup.
#
# VLLM_OMNI_TALKER_TEXT_ONLY=1 drops those positions from the talker's input entirely --
# shortening its array, not blanking the values.
#
# PREDICTIONS, written before running:
#   * talker_placeholder per turn collapses. At 640x352 a frame is 220 positions, so a turn
#     carrying ~8 frames should fall from ~1,900 to well under 100.
#   * Every turn still produces audio. This is the gross-breakage check and the one that
#     matters: if the talker cannot speak without visual conditioning, it fails here.
#   * chars and audio_chunks stay in the same range. The thinker is untouched, so a large
#     change in either means something OTHER than the intended switch moved.
#   * TTFA drops somewhat -- less for the talker to prefill.
#   * ZERO "SHORT:" warnings. Those mean the placeholder and the embeddings disagree, in which
#     case the audio is conditioned on a prefix and every other number here is meaningless.
#
# WHAT THIS RUN CANNOT SETTLE: prosody. n=5 turns, one session per arm, no listening test. It
# detects "the talker stopped working", not "the talker sounds slightly flatter".
set -uo pipefail
source /home/zx/voice-agent/env.sh

REPS="${1:-6}"
FORK=/home/zx/voice-agent/vllm-omni
RES=/data/zx/results
H=/home/zx/voice-agent/vllm-omni/benchmarks/live_agent/harness
F640=/data/zx/stimuli/frames640/handheld_walk_talk

QUERY="Describe what you can see in the camera right now."

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then
  echo "!! GPU already holds ${used} MiB -- refusing to start."; exit 1
fi

echo "==== preflight: which vllm_omni, and is the switch present? ===="
PYTHONPATH="$FORK" $PY_OMNI -c "
import vllm_omni, os
p = os.path.dirname(vllm_omni.__file__)
print('   vllm_omni:', p)
assert '/voice-agent/vllm-omni/' in p, 'NOT the fork -- aborting'
from vllm_omni.distributed.omni_connectors.adapter import TALKER_TEXT_ONLY, QWEN3_OMNI_MM_TOKEN_IDS
assert TALKER_TEXT_ONLY is True, 'text-only is the default now; both arms set it explicitly anyway'
print('   switch present, defaults ON, mm ids', sorted(QWEN3_OMNI_MM_TOKEN_IDS))
" 2>&1 | grep -vE "NVFP4|RuntimeWarning|from .version|^This typically|^Using fallback" \
  || { echo "!! preflight failed -- aborting"; exit 1; }

SERVER_PID=""
kill_all() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
    sleep 8
    kill -KILL -"$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 1000 ] && break
    sleep 3
  done
  SERVER_PID=""
}
trap 'echo "[ab] killed from outside at $(date -Is)"; kill_all' TERM INT

CLIENT_ARGS=(--num-frames 16 --max-frames 256 --evs --evs-threshold 0.95
             --max-frame-width 640 --max-frame-height 352
             --frame-filter-min-gap 8 --frame-filter-max-gap 16
             --session-scoped-request)

# $1 = arm tag, $2 = value for VLLM_OMNI_TALKER_TEXT_ONLY
run_arm() {
  local tag="$1" flag="$2"
  local log="$RES/server_textonly_$tag.log"
  local out="$RES/vt_textonly_$tag"
  rm -rf "$out"; mkdir -p "$out"; : > "$log"

  printf "\n################ ARM %s (VLLM_OMNI_TALKER_TEXT_ONLY=%s) ################\n" "$tag" "$flag"

  # Nothing may be inserted between these assignments and the command: a comment after a
  # backslash continuation comments out the rest of the line, PYTHONPATH stops being exported,
  # and the server silently comes up on site-packages vllm_omni with no session mode at all.
  HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FORK" \
  VLLM_OMNI_TALKER_TEXT_ONLY="$flag" VLLM_OMNI_LOG_SESSION_OUTPUTS=1 \
  setsid $PY_OMNI -m vllm.entrypoints.cli.main serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni --port 8091 --deploy-config "$H/deploy_pc_stage0.yaml" \
    --init-timeout 3000 --stage-init-timeout 1500 >> "$log" 2>&1 &

  for i in $(seq 1 300); do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8091/health 2>/dev/null)
    [ "$code" = "200" ] && { echo "  ready after ~$((i*5))s"; break; }
    if grep -qiE "not enough GPU memory|Engine core initialization failed" "$log"; then
      echo "  SERVER FAILED"; tail -20 "$log" | cut -c1-170; kill_all; return 1
    fi
    sleep 5
    [ "$i" = 300 ] && { echo "  TIMEOUT"; kill_all; return 1; }
  done
  SERVER_PID=$(sed -n 's/.*Started server process \[\([0-9]\+\)\].*/\1/p' "$log" | tail -1)
  kill -0 "$SERVER_PID" 2>/dev/null || SERVER_PID=""

  timeout 1800 $PY_OMNI $H/ttfa_bench.py --users 1 --reps "$REPS" --outdir "$out" --session 1 \
    --frames "$F640" --query "$QUERY" "${CLIENT_ARGS[@]}" 2>&1 | tail -3
  sleep 10
  kill_all

  # ---- gates. Each one has killed a run in this project before. ----
  local n_sess n_probe n_short
  n_sess=$(grep -c "\[session\] turn=" "$log")
  n_probe=$(grep -c "\[talker-text-only\]" "$log")
  n_short=$(grep -c "SHORT: placeholder and embeddings disagree" "$log")
  echo "  gates: session_lines=$n_sess  textonly_probe_lines=$n_probe  SHORT_warnings=$n_short"
  if [ "$n_sess" -eq 0 ]; then
    echo "  !! ABORT: session mode never entered -- this arm says nothing."; return 2
  fi
  if [ "$flag" = "1" ] && [ "$n_probe" -eq 0 ]; then
    echo "  !! ABORT: flag was set but the switch never executed -- arm is invalid."; return 2
  fi
  if [ "$flag" = "0" ] && [ "$n_probe" -ne 0 ]; then
    echo "  !! ABORT: control arm entered the switch -- the flag is leaking."; return 2
  fi
  if [ "$n_short" -ne 0 ]; then
    echo "  !! placeholder/embedding LENGTH MISMATCH -- audio is conditioned on a prefix."
    echo "     Every number from this arm is meaningless. Not aborting so the report shows it."
  fi
  return 0
}

run_arm off 0 || { echo "control arm failed"; exit 1; }
run_arm on  1 || { echo "treatment arm failed"; exit 1; }

echo ""
echo "################ verdict ################"
$PY_OMNI - "$RES/server_textonly_off.log" "$RES/server_textonly_on.log" <<'PY'
import re, sys, statistics as st

DELTA = re.compile(r"\[session\] turn=(\d+) queue delta: (\d+) new frames, (\d+) tokens, "
                   r"cum=(\d+), talker_placeholder=(\d+), talker_est=(\d+)")
DONE  = re.compile(r"\[session\] turn=(\d+) done first_text=(-?[\d.]+)s "
                   r"first_audio=(-?[\d.]+)s audio_chunks=(\d+) chars=(\d+)")

def parse(path):
    d, done = {}, {}
    for line in open(path, errors="replace"):
        m = DELTA.search(line)
        if m:
            t = int(m.group(1))
            d[t] = dict(frames=int(m.group(2)), tokens=int(m.group(3)),
                        placeholder=int(m.group(5)), est=int(m.group(6)))
        m = DONE.search(line)
        if m:
            t = int(m.group(1))
            done[t] = dict(ttft=float(m.group(2)), ttfa=float(m.group(3)),
                           chunks=int(m.group(4)), chars=int(m.group(5)))
    return d, done

def med(xs):
    return st.median(xs) if xs else float("nan")

arms = {}
for tag, path in (("off", sys.argv[1]), ("on", sys.argv[2])):
    d, done = parse(path)
    # Drop the first turn seen: cold weights, cold caches, and the only turn carrying the
    # system prompt -- its placeholder is not comparable to the others by construction.
    keys = sorted(set(d) & set(done))
    keys = keys[1:]
    arms[tag] = dict(
        n=len(keys),
        placeholder=[d[k]["placeholder"] for k in keys],
        est_last=d[keys[-1]]["est"] if keys else 0,
        frames=[d[k]["frames"] for k in keys],
        ttfa=[done[k]["ttfa"] for k in keys],
        chunks=[done[k]["chunks"] for k in keys],
        chars=[done[k]["chars"] for k in keys],
        silent=[k for k in keys if done[k]["chunks"] == 0],
    )

def row(label, f, unit=""):
    a, b = f(arms["off"]), f(arms["on"])
    if isinstance(a, float):
        ratio = f"{b/a:.2f}x" if a else "--"
        print(f"  {label:<34} {a:>10.1f}{unit:<4} {b:>10.1f}{unit:<4}  {ratio}")
    else:
        ratio = f"{b/a:.2f}x" if a else "--"
        print(f"  {label:<34} {a:>10}{unit:<4} {b:>10}{unit:<4}  {ratio}")

print(f"  {'':<34} {'OFF':>10}     {'ON':>10}       ratio")
print(f"  {'-'*34} {'-'*14} {'-'*14}  -----")
print(f"  {'turns analysed':<34} {arms['off']['n']:>10}     {arms['on']['n']:>10}")
row("frames per turn (median)",        lambda a: med(a["frames"]))
row("talker_placeholder (median)",     lambda a: med(a["placeholder"]))
row("talker_est after last turn",      lambda a: float(a["est_last"]))
row("TTFA median",                     lambda a: med(a["ttfa"]), " s")
row("audio_chunks (median)",           lambda a: med(a["chunks"]))
row("chars out (median)",              lambda a: med(a["chars"]))

print()
for tag in ("off", "on"):
    s = arms[tag]["silent"]
    verdict = "ALL TURNS PRODUCED AUDIO" if not s else f"SILENT TURNS: {s}  <-- BROKEN"
    print(f"  {tag:<4} {verdict}")

print()
p_off, p_on = med(arms["off"]["placeholder"]), med(arms["on"]["placeholder"])
if p_on and p_off and p_on < p_off * 0.25:
    print(f"  SWITCH TOOK EFFECT: talker_placeholder {p_off:.0f} -> {p_on:.0f} "
          f"({p_off/p_on:.1f}x smaller per turn)")
else:
    print(f"  !! placeholder barely moved ({p_off:.0f} -> {p_on:.0f}) -- the switch did not do "
          f"what it claims; treat the rest as invalid.")

print()
print("  What this does NOT show: prosody. n=5, one session per arm, nobody listened.")
print("  Logs: /data/zx/results/server_textonly_{off,on}.log")
print("  Model text output is in those logs (VLLM_OMNI_LOG_SESSION_OUTPUTS=1) -- read a few")
print("  answers from each arm before believing the table.")
PY
