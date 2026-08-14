"""Payload size: does the lean decode payload actually shrink, and by how much?"""
import sys
from collections import defaultdict
from types import SimpleNamespace
import torch
sys.path.insert(0, "/home/ubuntu/data/vllm-omni")
from vllm_omni.model_executor.stage_input_processors import qwen3_omni as m
from vllm_omni.data_entry_keys import to_dict

D = 2048
tm = SimpleNamespace(_pending_streaming_prefills={}, request_payload={}, put_req_chunk=defaultdict(int))

def mk_req(n_out):
    return SimpleNamespace(external_req_id="r", output_token_ids=list(range(n_out)),
        prompt_token_ids=[1]*10, all_token_ids=[1]*(10+n_out),
        num_computed_tokens=10+n_out, num_output_placeholders=0, resumable=True, sampling_params=None)

def size_of(p):
    import pickle
    return len(pickle.dumps(to_dict(p)))

for lean in (False, True):
    m._T2T_LEAN_DECODE = lean
    tm._pending_streaming_prefills.clear()
    m._construct_thinker2talker_streaming_input_async_chunk(False, mk_req(1), torch.zeros(5,D), torch.zeros(5,D), tm)
    m._construct_thinker2talker_streaming_input_async_chunk(False, mk_req(1), torch.zeros(1,D), torch.zeros(1,D), tm)  # opener
    sizes = []
    for n in (1, 100, 400):
        p = m._construct_thinker2talker_streaming_input_async_chunk(False, mk_req(n+2), torch.zeros(1,D), torch.zeros(1,D), tm)
        sizes.append((n, size_of(p)))
    tag = "lean " if lean else "full "
    print(f"{tag}: " + "  ".join(f"第{n}个token时 {s} 字节" for n, s in sizes))
