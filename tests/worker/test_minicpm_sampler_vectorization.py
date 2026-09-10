"""The optimized penalty must preserve native sampling and generator state."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.worker.test_minicpm_pd_sampling_state import IDS, model, row, snapshot
from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.minicpmo45.sampling_state import SAMPLING_STATE_KEY, DecodeSamplingState
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration as Model,
    _MiniCPMFilteredDraw,
)


def scalar_penalty(logits, history, penalty):
    """Literal pre-optimization implementation (also for negative logits)."""
    for token_id in set(history[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :]):
        if token_id < 0 or token_id >= logits.shape[-1]:
            continue
        logits[0, token_id] /= penalty


def uncached_forbidden_mask(m, logits, token_ids):
    forbidden = m._minicpmo45_native_forbidden_token_ids(token_ids)
    valid = [token_id for token_id in forbidden if 0 <= token_id < logits.shape[-1]]
    if valid:
        logits[:, valid] = float("-inf")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA comparison requires an idle GPU")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("history", [[], [-1, 32, 1000], [0, 1, 1, 2, 3, -1, 32], [31] + [1, 2] * 256])
def test_penalty_is_bitwise_equal_to_scalar_updates(device, dtype, history):
    base = torch.linspace(-8, 8, 32, dtype=dtype, device=device).unsqueeze(0)
    base[0, 2] = float("-inf")  # A forbidden token remains forbidden.
    expected, actual = base.clone(), base.clone()
    scalar_penalty(expected, history, 1.05)
    Model._apply_minicpmo45_repetition_penalty(actual, history, 1.05)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("vocab", [256, 8192])
def test_native_sampling_and_rng_match_scalar_reference_across_reordered_rows(device, greedy, decode, vocab):
    optimized, reference = model(decode), model(decode)
    reference._draw_minicpmo45_token = lambda probs, generator: torch.multinomial(
        probs, num_samples=1, generator=generator
    )
    reference._apply_minicpmo45_repetition_penalty = scalar_penalty
    reference._mask_minicpmo45_forbidden_tokens = lambda logits, ids: uncached_forbidden_mask(reference, logits, ids)
    models = (optimized, reference)
    if vocab == 8192:
        for m in models:
            m._minicpmo45_tokenizer_cache.bad_token_ids = list(range(100, 7065))
    initial = {
        "a": DecodeSamplingState(generated_tokens=[8, 9, 9, 12] * 128),
        "b": DecodeSamplingState(generated_tokens=[10, 11] * 256),
    }
    states = [deepcopy(initial) for _ in models]
    generators = [
        {key: torch.Generator(device=device).manual_seed(92 + i) for i, key in enumerate(initial)} for _ in models
    ]
    if not decode:
        for m, ss in zip(models, states):
            m._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={(key, 0): s for key, s in ss.items()})
    # Includes stochastic text, native boundary, forced token-budget boundary,
    # P's already-final decision on D, and a skipped intermediate prefill row.
    for step in range(8):
        order = ["a", "b"] if step % 2 == 0 else ["b", "a"]
        logits = torch.randn(2, vocab, generator=torch.Generator().manual_seed(100 + step)).to(device)
        logits[:, IDS["chunk_eos_token_id"]] = -20
        if step == 2:
            logits.fill_(float("-inf"))
            logits[:, IDS["chunk_eos_token_id"]] = 10
        results = []
        for j, m in enumerate(models):
            if decode:
                for key in initial:
                    if key in getattr(m, "_minicpmo45_pd_sampling_states", {}):
                        states[j][key] = m._minicpmo45_pd_sampling_states[key]
            if step in (3, 4):
                for s in states[j].values():
                    s.current_segment_output_tokens = [8] * 19 if step == 3 else [IDS["listen_token_id"]]
            recent = [list(states[j][key].current_segment_output_tokens) for key in order]
            metadata = SimpleNamespace(
                all_greedy=greedy,
                output_token_ids=recent,
                generators={i: generators[j][key] for i, key in enumerate(order)},
                temperature=0.7,
                top_k=20,
                top_p=0.8,
            )
            rows = []
            for i, key in enumerate(order):
                payload = {SAMPLING_STATE_KEY: snapshot(states[j][key]), "is_speech": True}
                r = row(key, idx=i, session=key, payload=payload)
                if step == 5 and key == "a":
                    from dataclasses import replace

                    r = replace(r, should_sample=False)
                rows.append(r)
            batch_logits = logits.clone()
            m.prepare_duplex_sampling(batch_logits, metadata, tuple(rows))
            out = m.sample(batch_logits, metadata)
            results.append((out.sampled_token_ids.cpu().tolist(), out.skipped_sampling_request_ids))
        assert results[0] == results[1]
        for key in initial:
            assert torch.equal(generators[0][key].get_state(), generators[1][key].get_state())
        for i in range(2):
            assert snapshot(optimized._minicpmo45_duplex_state_for_row(i)) == snapshot(
                reference._minicpmo45_duplex_state_for_row(i)
            )


def test_forbidden_cache_matches_reference_and_invalidates_on_semantic_changes(device):
    m = model(True)
    m._minicpmo45_tokenizer_cache.bad_token_ids = [-1, 2, 2, *range(100, 7065), 20000]
    ids = dict(IDS)

    def check(vocab=8192, dtype=torch.float32, where=device):
        logits = torch.zeros(2, vocab, device=where, dtype=dtype)
        expected = logits.clone()
        uncached_forbidden_mask(m, expected, ids)
        m._mask_minicpmo45_forbidden_tokens(logits, ids)
        assert torch.equal(logits, expected)
        return m._minicpmo45_forbidden_index_cache[1]

    cached = check()
    assert check() is cached
    assert check(dtype=torch.bfloat16) is cached
    m._minicpmo45_tokenizer_cache.bad_token_ids.append(80)
    changed = check()
    assert changed is not cached
    ids["chunk_eos_token_id"] = 90
    changed_again = check()
    assert changed_again is not changed
    smaller = check(vocab=128)
    assert smaller is not changed_again
    if device == "cuda":
        assert check(vocab=128, where="cpu").device.type == "cpu"
        assert check(vocab=128).device.type == "cuda"
    m._minicpmo45_tokenizer_cache.bad_token_ids = []
    m._mask_minicpmo45_forbidden_tokens(torch.zeros(1, 128, device=device), {})
    assert m._minicpmo45_forbidden_index_cache[1].numel() == 0


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("rng_mode", ["independent", "shared", "unseeded"])
def test_batched_feedback_matches_sequential_rows_and_rng(device, greedy, rng_mode):
    """CUDA batch versus one-row calls, including changing membership/order."""
    from dataclasses import replace

    batched, sequential = model(True), model(True)
    sequential._draw_minicpmo45_token = lambda probs, generator: torch.multinomial(
        probs, num_samples=1, generator=generator
    )
    gens = [{key: torch.Generator(device=device).manual_seed(100+i) for i, key in enumerate('abc')}
            for _ in range(2)]
    if rng_mode == "shared":
        gens = [{key: group['a'] for key in 'abc'} for group in gens]
    if rng_mode == "unseeded":
        gens = [{}, {}]
    states = [{key: DecodeSamplingState(generated_tokens=[8, 9, 11]*100) for key in 'abc'} for _ in range(2)]
    for step in range(8):
        order = list('abc' if step % 2 == 0 else 'cba')
        logits = torch.randn(3, 8192, generator=torch.Generator().manual_seed(90+step)).to(device)
        logits[:, IDS['chunk_eos_token_id']] = -30
        if step == 2:
            logits[1].fill_(float('-inf'))
            logits[1, IDS['chunk_eos_token_id']] = 1
        outcomes = []
        global_rng_states = []
        for impl, (m, ss, gs) in enumerate(zip((batched, sequential), states, gens)):
            torch.manual_seed(500 + step)  # compare the unseeded global RNG too
            for key in order:
                ss[key] = getattr(m, '_minicpmo45_pd_sampling_states', {}).get(key, ss[key])
                if step in (3, 4):
                    ss[key].current_segment_output_tokens = [8]*19 if step == 3 else [IDS['listen_token_id']]
            results = []
            for indices in ([list(range(3))] if impl == 0 else [[i] for i in range(3)]):
                rows = [row(order[i], idx=j, session=order[i],
                            payload={SAMPLING_STATE_KEY: snapshot(ss[order[i]]), 'is_speech': True})
                        for j, i in enumerate(indices)]
                rows = [replace(r, should_sample=False) if step == 5 and r.request_id == 'b' else r for r in rows]
                metadata = SimpleNamespace(all_greedy=greedy, output_token_ids=[[] for _ in rows],
                    generators={j: gs[order[i]] for j,i in enumerate(indices) if order[i] in gs},
                    temperature=.7, top_k=20, top_p=.8)
                x = logits[indices]
                m.prepare_duplex_sampling(x, metadata, tuple(rows))
                results.extend(m.sample(x, metadata).sampled_token_ids.cpu().reshape(-1).tolist())
            outcomes.append(results)
            global_rng_states.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
            states[impl] = {key: m._minicpmo45_pd_sampling_states[key] for key in order}
        assert outcomes[0] == outcomes[1]
        assert torch.equal(global_rng_states[0], global_rng_states[1])
        for key in gens[0]:
            assert torch.equal(gens[0][key].get_state(), gens[1][key].get_state())
        for key in order:
            assert snapshot(states[0][key]) == snapshot(states[1][key])


@pytest.mark.parametrize("tied", [False, True])
def test_filter_batches_only_matching_rules_and_preserves_draw_order(device, tied):
    from unittest.mock import patch

    m = model(True)
    logits = torch.randn(4, 8192, generator=torch.Generator().manual_seed(20)).to(device)
    if tied:
        logits = logits.round()
    rules = [(100, .8), (20, .9), (100, .8), (20, .9)]
    gens = [{i: torch.Generator(device=device).manual_seed(200+i) for i in range(4)} for _ in range(2)]
    expected = []
    for i, (k, p) in enumerate(rules):
        probs = m._top_k_top_p_filter(logits[i:i+1], top_k=k, top_p=p).softmax(-1)
        expected.append(m._draw_minicpmo45_token(probs, gens[0][i]))
    decisions = [_MiniCPMFilteredDraw(logits[i:i+1], k, p, gens[1][i]) for i, (k, p) in enumerate(rules)]
    # An already-resolved boundary can coexist with filtered text decisions.
    boundary = torch.tensor([77], device=device)
    decisions.insert(2, boundary)
    expected.insert(2, boundary)
    with patch.object(Model, "_top_k_top_p_filter", wraps=Model._top_k_top_p_filter) as filter_call:
        actual = m._materialize_minicpmo45_draws(decisions)
    assert filter_call.call_count == 2
    assert torch.equal(torch.cat([x.reshape(-1) for x in actual]), torch.cat([x.reshape(-1) for x in expected]))
    assert actual[2] is boundary
    assert all(torch.equal(gens[0][i].get_state(), gens[1][i].get_state()) for i in range(4))


def test_independent_cuda_draws_use_one_native_batch(device):
    if device != "cuda":
        pytest.skip("GPU-only native draw batching")
    from unittest.mock import patch
    import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni as module

    m = model(True)
    logits = torch.randn(3, 8192, device=device)
    decisions = [
        _MiniCPMFilteredDraw(logits[i:i+1], 100, .8, torch.Generator(device=device).manual_seed(i))
        for i in range(3)
    ]
    with patch.object(module, "random_sample", wraps=module.random_sample) as draw:
        values = m._materialize_minicpmo45_draws(decisions)
    assert draw.call_count == 1
    assert draw.call_args.args[0].shape == (3, 8192)
    assert list(draw.call_args.args[1]) == [0, 1, 2]
    assert all(value.shape == (1, 1) for value in values)


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_deferred_policy_preparation_matches_original_row_operations(device, greedy, tied):
    from unittest.mock import patch

    m = model(True)
    m._minicpmo45_tokenizer_cache.bad_token_ids = [3, 4, 5]
    logits = torch.randn(4, 8192, generator=torch.Generator().manual_seed(27)).to(device)
    if tied:
        logits = logits.round()
    original = logits.clone()
    histories = [(8, 9, 9, -1, 99999), (), (5, 8, 10), (17, 17)]
    temperatures = [.7, .9, .7, .9]
    gens = [[torch.Generator(device=device).manual_seed(70+i) for i in range(4)] for _ in range(2)]
    expected, decisions = [], []
    for i, (history, temperature) in enumerate(zip(histories, temperatures)):
        x = logits[i:i+1].clone()
        uncached_forbidden_mask(m, x, IDS)
        scalar_penalty(x, history, 1.05)
        if greedy:
            expected.append(torch.argmax(x, dim=-1))
        else:
            probs = m._top_k_top_p_filter(x / temperature, top_k=100, top_p=.8).softmax(-1)
            expected.append(m._draw_minicpmo45_token(probs, gens[0][i]))
        decisions.append(_MiniCPMFilteredDraw(logits[i:i+1], 100, .8, gens[1][i],
            token_ids=IDS, repetition_tokens=history, temperature=temperature, greedy=greedy))
    with patch.object(Model, "_mask_minicpmo45_forbidden_tokens", autospec=True,
                      side_effect=Model._mask_minicpmo45_forbidden_tokens) as mask:
        actual = m._materialize_minicpmo45_draws(decisions)
    assert mask.call_count == 2
    assert torch.equal(torch.cat([x.reshape(-1) for x in actual]), torch.cat([x.reshape(-1) for x in expected]))
    assert torch.equal(original, logits)
    assert all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(gens[0], gens[1]))
