import os

import pytest
import torch

from vllm_omni.engine.pinned_prefix_window import compact_window_view


@pytest.mark.skipif(os.environ.get("MINICPM_RUN_PINNED_GPU_TEST") != "1", reason="explicit exclusive-GPU diagnostic")
@pytest.mark.parametrize("qlen", [1, 7, 220])
@pytest.mark.parametrize(
    ("seq", "window"),
    [(240, 128), (512, 128), (4096, 128), (18352, 18000), (36768, 18000), (36352, 36000), (72768, 36000)],
)
@pytest.mark.parametrize("split_k", [False, True])
def test_native_rswa_compacted_view_matches_dense_causal_prefix_window(qlen, seq, window, split_k):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    torch.manual_seed(7)
    bs, hq, hk, dim = 16, 4, 2, 64
    pin = 32
    if window >= 18000:
        # Exercise both serving windows, including full replacement of their
        # original history, with both native split-K and 2D execution.
        pin = 128
    pages = (seq + bs - 1) // bs
    k = torch.randn(pages + 1, bs, hk, dim, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn_like(k.float()).to(torch.float8_e4m3fn)
    k[0] = float("nan")
    v[0] = float("nan")
    q = torch.randn(qlen, hq, dim, dtype=torch.bfloat16, device="cuda")
    table = torch.arange(1, pages + 1, dtype=torch.int32, device="cuda").unsqueeze(0)
    # The expired logical middle really is unavailable, not a hidden full cache.
    gap_end = max(pin // bs, max(0, seq - qlen - window + 1) // bs)
    table[0, pin // bs : gap_end] = 0
    view, lens = compact_window_view(
        table,
        torch.tensor([seq], dtype=torch.int32),
        torch.tensor([0, qlen], dtype=torch.int32),
        prefix_tokens=pin,
        window_tokens=window,
        block_size=bs,
    )
    out = torch.empty_like(q)
    scale = torch.tensor(1.0, device="cuda")
    split = {}
    if split_k:
        split = dict(
            num_par_softmax_segments=16,
            softmax_segm_output=torch.empty((2, hq, 16, dim), device="cuda"),
            softmax_segm_max=torch.empty((2, hq, 16), device="cuda"),
            softmax_segm_expsum=torch.empty((2, hq, 16), device="cuda"),
        )
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.tensor([0, qlen], device="cuda", dtype=torch.int32),
        max_seqlen_q=qlen,
        seqused_k=lens.cuda(),
        max_seqlen_k=int(lens.max()),
        softmax_scale=dim**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=view,
        softcap=0,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
        seq_threshold_3D=2 if split_k else 0,
        rswa_prefix_lens=torch.tensor([pin], device="cuda", dtype=torch.int32),
        rswa_window=window,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        **split,
    )
    keys = k[1:].float().reshape(-1, hk, dim)[:seq].repeat_interleave(hq // hk, dim=1)
    vals = v[1:].float().reshape(-1, hk, dim)[:seq].repeat_interleave(hq // hk, dim=1)
    qpos = torch.arange(seq - qlen, seq, device="cuda")[:, None]
    kpos = torch.arange(seq, device="cuda")[None, :]
    mask = (kpos <= qpos) & ((kpos < pin) | (qpos - kpos < window))
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * dim**-0.5
    probs = scores.masked_fill(~mask[None], float("-inf")).softmax(-1)
    expected = torch.einsum("hqk,khd->qhd", probs, vals)
    torch.testing.assert_close(out.float(), expected, atol=0.008, rtol=0.02)
