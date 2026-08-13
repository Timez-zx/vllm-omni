"""Offline conservation test for the T2T coalescing patch (real import)."""
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/ubuntu/data/vllm-omni")
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as m

print("consts:", m._T2T_COALESCE_TOKENS, m._T2T_COALESCE_EXEMPT)
assert m._T2T_COALESCE_TOKENS == 8 and m._T2T_COALESCE_EXEMPT == 4

D = 8
tm = SimpleNamespace(_pending_streaming_prefills={}, request_payload={}, put_req_chunk=defaultdict(int))

def mk_req(n_out):
    return SimpleNamespace(
        external_req_id="req-A",
        output_token_ids=list(range(n_out)),
        prompt_token_ids=[1] * 10,
        all_token_ids=[1] * (10 + n_out),
        num_computed_tokens=10 + n_out,
        num_output_placeholders=0,
        resumable=True,
        sampling_params=None,
    )

def row(i):
    return torch.full((1, D), float(i))

shipped = []          # flat list of row values shipped, in order
flush_sizes = []

def ship(payload):
    if payload is None:
        return
    e = payload.embed.prefill if payload.embed.prefill is not None else payload.embed.decode
    vals = [float(e[j, 0]) for j in range(e.shape[0])]
    shipped.append(vals)
    flush_sizes.append(len(vals))

# 1) segment prefill step (multi-row) -> cached, returns None
r = mk_req(1)
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, r, torch.zeros(5, D), torch.zeros(5, D), tm)
assert p is None, "prefill step must cache"

# 2) first decode step -> opener flush (prefill + this row)
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, mk_req(1), row(100), torch.zeros(1, D), tm)
assert p is not None and p.embed.prefill is not None and p.embed.prefill.shape[0] == 6
print("opener ships prefill rows:", p.embed.prefill.shape[0])

# 3) 30 decode steps with rows 0..29
for i in range(30):
    p = m._construct_thinker2talker_streaming_input_async_chunk(
        False, mk_req(i + 2), row(i), torch.zeros(1, D), tm)
    ship(p)

# 4) segment end via the cleared-outputs branch (output_token_ids empty, is_finished)
r_end = mk_req(0)
p = m._construct_thinker2talker_streaming_input_async_chunk(
    True, r_end, row(999), torch.zeros(1, D), tm)
ship(p)

flat = [v for chunk in shipped for v in chunk]
expect = [float(i) for i in range(30)] + [999.0]
print("flush sizes:", flush_sizes)
print("rows shipped:", len(flat), "expected:", len(expect))
assert flat == expect, f"conservation/order violated:\n{flat}\nvs\n{expect}"
assert flush_sizes[:4] == [1, 1, 1, 1], "first 4 decode flushes must be per-token"
assert flush_sizes[4:7] == [2, 4, 8], f"ramp wrong: {flush_sizes[4:7]}"

# 5) new segment resets the counter: prefill again, then decode flushes restart at 1
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, mk_req(1), torch.zeros(4, D), torch.zeros(4, D), tm)
assert p is None
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, mk_req(1), row(500), torch.zeros(1, D), tm)   # opener
assert p is not None and p.embed.prefill is not None
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, mk_req(2), row(501), torch.zeros(1, D), tm)   # decode #1 -> per-token again
assert p is not None and p.embed.decode is not None and p.embed.decode.shape[0] == 1
print("per-segment reset OK")

# 6) control arm: TOKENS=1 must reproduce per-token behavior exactly
m._T2T_COALESCE_TOKENS = 1
p = m._construct_thinker2talker_streaming_input_async_chunk(
    False, mk_req(3), row(600), torch.zeros(1, D), tm)
assert p is not None and p.embed.decode.shape[0] == 1
print("control arm (TOKENS=1) OK")
print("ALL PASS")
