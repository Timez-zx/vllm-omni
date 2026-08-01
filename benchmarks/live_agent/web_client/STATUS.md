# Where this stands

Written while the Qwen server was still loading, so the last row is the one to
check first when you come back.

## Done and verified

| | |
|---|---|
| Branch `live-agent-web` | upstream `main` + our 26 Qwen commits. **One** merge conflict, and it was not a real one |
| Optimisations survive the merge | all 19 `StreamingVideoSessionConfig` fields, `TALKER_TEXT_ONLY` on, the session roll / orphan recovery / heartbeat intact — checked by import, not assumed |
| Browser client | `app/` + `server.py`, page and same-origin websocket proxy |
| `selftest.py` | **14/14 passing**, no GPU needed |
| `probe.py` | written; runs the whole chain with synthetic media and no browser |

## Not yet verified

**The end-to-end run.** The Qwen server was still loading weights when this was
written. The probe is queued behind its health check and will run by itself.

Check it with:

```bash
tail -20 /data/zx/results/qwen_live.log        # server
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8091/health
```

Then, if health is 200:

```bash
cd /home/zx/voice-agent/vllm-omni
PYTHONPATH=$PWD python benchmarks/live_agent/web_client/probe.py --direct --turns 2
bash benchmarks/live_agent/web_client/run_page_server.sh          # then ssh -L 7870
```

## One thing that is genuinely unknown

The original Qwen numbers — 513 ms TTFA, 40/40 turns with 2 rolls, the 817→33
talker cut — were **all measured on vLLM 0.24.0** in the `omni` env. This branch
runs on **0.26.0**, because `main` requires it.

So those numbers are not directly comparable here until re-measured, and the
merge is not proof of that: it proves the code is present, not that it performs
the same. The measurement harness that produced them is in
`benchmarks/live_agent/harness/` and is model- and branch-agnostic, so re-running
it on this branch is the way to find out.

## And one honest regression

The turn trigger is in the browser, not the model. Qwen3-Omni has no
listen/speak token and `should_trigger_turn()` returns `False`, so nothing
server-side can decide you have stopped talking. The MiniCPM duplex client
(`live-agent-minicpm` branch) does have the model deciding — that is the arm to
compare against, and the difference is real rather than an implementation
shortcut.

To go back to it:

```bash
git checkout live-agent-minicpm
HF_HOME=/data/zx/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zx/voice-agent/vllm-omni \
  /home/zx/miniconda3/envs/omni-minicpm/bin/vllm-omni serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8099
python -m examples.online_serving.minicpmo.realtime_web --port 7862 \
  --ws-backend ws://127.0.0.1:8099 --ref-audio /data/zx/stimuli/minicpm/ref_minicpm_signature.wav
```
