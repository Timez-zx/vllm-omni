# live_agent — latency benchmarks for continuous, stateful, multimodal serving

Measures **time to first audio** for a long-running video+audio conversation against the
three-stage Qwen3-Omni topology (thinker / talker / code2wav), and the tooling that produced
every number in the `live-agent` branch's commit messages.

It lives in the fork rather than beside it because it measures this code and has to move with
it. Nothing here is imported by `vllm_omni`; it is a leaf.

```
harness/    experiment drivers + the websocket client. See harness/README.md for each file.
analysis/   one report script per finding; each reads /data/zx/results and prints or plots it.
figures/    the plots referenced by the study documents.
```

## Running one

```bash
source /home/zx/voice-agent/env.sh        # $PY_OMNI serving env, $PY_PA0 analysis env
python harness/session_prompt_selftest.py # 30 assertions, no GPU — run this first
bash harness/run_roll_probe.sh 40 45000   # 40 turns, roll at 45k talker tokens
python analysis/roll_report.py            # what it cost
```

Every driver boots its own server, asserts that the FORK is what gets imported (a run that
silently came up on `site-packages` produced 30 turns of meaningless data once), refuses to start
if the GPU is already busy, and gates the result before reporting it.

## The headline

One resumable engine request per session drops the speech stage from 36.5 to ~2 ms per 1,000
accumulated prompt tokens, so a long conversation is cheaper to KEEP than to discard. Rolling
that request as the talker nears `max_model_len` lets the conversation continue indefinitely:
3.2x TTFA on the turn that rolls, ~56 ms/turn amortised.

Single user, one GPU, one stimulus set.
