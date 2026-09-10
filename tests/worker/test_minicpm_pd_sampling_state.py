from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.experimental.fullduplex.minicpmo45.sampling_state import (
    SAMPLING_STATE_KEY,
    SAMPLING_STATE_WIRE_KEY,
    DecodeSamplingState,
    pack_sampling_state,
    restore_sampling_state,
    resume_sampling_rng,
    snapshot_sampling_rng,
    unpack_sampling_state,
)
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration as Model,
)

IDS = dict(
    unit_token_id=0,
    listen_token_id=1,
    tts_bos_token_id=2,
    chunk_eos_token_id=3,
    chunk_tts_eos_token_id=4,
    turn_eos_token_id=5,
)


class FakePhilox:
    device = torch.device("cuda")

    def __init__(self, seed=42):
        self.manual_seed(seed)

    def manual_seed(self, seed):
        self.seed, self.offset = seed, 0

    def initial_seed(self):
        return self.seed

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset

    def draw(self):
        result = self.seed + self.offset
        self.offset += 4
        return result


def test_rng_wire_preserves_unsigned_seed_and_restores_once_across_units():
    seed = (1 << 64) - 2
    full, pgen = FakePhilox(seed), FakePhilox(seed)
    pstate = DecodeSamplingState()
    for unit in range(3):
        resume_sampling_rng(pstate, pgen, ("p", unit))
        assert pgen.draw() == full.draw()
        snapshot_sampling_rng(pstate, pgen)
        dstate, _ = unpack_sampling_state(torch.tensor(snapshot(pstate)))
        assert dstate.rng_seed == seed
        dgen = FakePhilox()  # Every finite D request starts with its own seed.
        for _ in range(3):
            resume_sampling_rng(dstate, dgen, ("d", unit))
            assert dgen.draw() == full.draw()
        snapshot_sampling_rng(dstate, dgen)
        returned, _ = unpack_sampling_state(snapshot(dstate))
        restore_sampling_state(pstate, returned)
    assert pstate.rng_offset == full.get_offset() == 48


def test_rng_acknowledgement_consumes_no_draws_and_cleanup_drops_state():
    m = model(True)
    wire = snapshot(DecodeSamplingState(current_segment_output_tokens=[1], rng_seed=17, rng_offset=40))
    m.prepare_duplex_sampling(torch.zeros(1, 12), None, (row(payload={SAMPLING_STATE_KEY: wire}),))
    gen = FakePhilox()
    metadata = SimpleNamespace(all_greedy=False, output_token_ids=[[]], generators={0: gen})
    assert m.sample(torch.zeros(1, 12), metadata).sampled_token_ids.item() == 1
    assert (gen.initial_seed(), gen.get_offset()) == (17, 40)
    assert list(m.snapshot_duplex_sampling_outputs(["r"])[SAMPLING_STATE_WIRE_KEY][0]) == wire
    m.on_requests_finished(["r"])
    assert not m._minicpmo45_pd_sampling_states


def test_rng_restore_uses_request_identity_not_batch_row():
    m = model(True)
    a = row(
        "a",
        payload={
            SAMPLING_STATE_KEY: snapshot(
                DecodeSamplingState(current_segment_output_tokens=[1], rng_seed=17, rng_offset=40)
            )
        },
    )
    b = row(
        "b",
        idx=1,
        session="other",
        payload={
            SAMPLING_STATE_KEY: snapshot(
                DecodeSamplingState(current_segment_output_tokens=[1], rng_seed=23, rng_offset=80)
            )
        },
    )
    ga, gb = FakePhilox(), FakePhilox()
    logits = torch.zeros(2, 12)
    m.prepare_duplex_sampling(logits, None, (a, b))
    m.sample(logits, SimpleNamespace(all_greedy=False, generators={0: ga, 1: gb}))
    m.prepare_duplex_sampling(
        logits, None, (row("b", session="other", payload=b.payload), row("a", idx=1, payload=a.payload))
    )
    m.sample(logits, SimpleNamespace(all_greedy=False, generators={0: gb, 1: ga}))
    assert (ga.initial_seed(), ga.get_offset()) == (17, 40)
    assert (gb.initial_seed(), gb.get_offset()) == (23, 80)


def test_unseeded_and_greedy_rng_are_unchanged():
    state = DecodeSamplingState(rng_seed=17, rng_offset=40)
    resume_sampling_rng(state, None, "p")
    snapshot_sampling_rng(state, None)
    m = model(True)
    wire = snapshot(DecodeSamplingState(current_segment_output_tokens=[1], rng_seed=17, rng_offset=40))
    m.prepare_duplex_sampling(torch.zeros(1, 12), None, (row(payload={SAMPLING_STATE_KEY: wire}),))
    gen = FakePhilox()
    m.sample(torch.zeros(1, 12), SimpleNamespace(all_greedy=True, generators={0: gen}))
    assert (gen.initial_seed(), gen.get_offset()) == (42, 0)
    assert (state.rng_seed, state.rng_offset) == (17, 40)


@pytest.mark.parametrize("decode", [False, True])
def test_partial_prefill_has_no_sampling_policy_or_rng_effects_in_mixed_batch(monkeypatch, decode):
    m = model(decode)
    states = [DecodeSamplingState(), DecodeSamplingState(current_turn_ended=False)]
    if not decode:
        m._minicpmo45_duplex_data_plane_helper = SimpleNamespace(
            sessions={("s", 0): states[0], ("other", 0): states[1]}
        )
    rows = (
        replace(row("a", payload={"force_listen": True, SAMPLING_STATE_KEY: snapshot(states[0])}), should_sample=False),
        row("b", idx=1, session="other", payload={"is_speech": True, SAMPLING_STATE_KEY: snapshot(states[1])}),
    )
    generators = {0: FakePhilox(), 1: FakePhilox()}
    draws = []

    def multinomial(_probs, num_samples, generator):
        draws.append(generator.draw())
        return torch.tensor([[9]])

    monkeypatch.setattr(torch, "multinomial", multinomial)
    logits = torch.zeros(2, 12)
    before = snapshot(states[0])
    metadata = SimpleNamespace(all_greedy=False, output_token_ids=[[], []], generators=generators)
    m.prepare_duplex_sampling(logits, metadata, rows)
    assert torch.equal(logits[0], torch.zeros(12))  # No forced LISTEN on an incomplete prefill.
    out = m.sample(logits, metadata)
    assert out.skipped_sampling_request_ids == ("a",)
    assert out.sampled_token_ids.tolist() == [[0], [9]]
    assert generators[0].get_offset() == 0
    assert generators[1].get_offset() == 8
    assert draws == [42, 46]
    assert snapshot(states[0]) == before
    if decode:
        assert "a" not in m._minicpmo45_pd_sampling_states
    published = m.snapshot_duplex_sampling_outputs(["a", "b"])[SAMPLING_STATE_WIRE_KEY]
    assert published[0] is None
    assert published[1] is not None


@pytest.mark.parametrize("raise_error", [False, True])
def test_bookkeeping_does_not_rewind_unsampled_rows_and_restores_mapping(monkeypatch, raise_error):
    import numpy as np

    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplerOutput
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    ga, gb = FakePhilox(), FakePhilox()
    runner.input_batch = SimpleNamespace(req_id_to_index={"a": 0, "b": 1}, generators={0: ga, 1: gb})
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False]))

    def upstream(self, *args):
        assert 0 not in self.input_batch.generators
        assert self.input_batch.generators[1] is gb
        assert self.discard_request_mask.np.tolist() == [True, False]
        if raise_error:
            raise ValueError("mock upstream error")
        return "done"

    monkeypatch.setattr(OmniGPUModelRunner, "_bookkeeping_sync", upstream)
    out = DuplexSamplerOutput(torch.zeros(2, 1, dtype=torch.int32), None, ("a",))
    if raise_error:
        with pytest.raises(ValueError, match="mock upstream"):
            runner._bookkeeping_sync(None, out, None, None, 2)
    else:
        assert runner._bookkeeping_sync(None, out, None, None, 2) == "done"
    assert runner.input_batch.generators == {0: ga, 1: gb}
    assert ga.get_offset() == 0  # No negative rewind at the first prefill chunk.


def test_plain_sampler_bookkeeping_retains_original_rng_behavior(monkeypatch):
    from vllm.v1.outputs import SamplerOutput

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    marker = object()
    out = SamplerOutput(torch.zeros(1, 1, dtype=torch.int32), None)

    def upstream(self, *args):
        assert args[1] is out
        return marker

    monkeypatch.setattr(OmniGPUModelRunner, "_bookkeeping_sync", upstream)
    assert runner._bookkeeping_sync(None, out, None, None, 1) is marker


def test_d_refuses_to_replace_last_audio_embedding_with_text_embedding():
    m = model(True)
    duplex = {"data_plane": True, "pd_media_prefix_tokens": 279}
    # Old cursor: it silently embedded the audio placeholder at position 278.
    with pytest.raises(RuntimeError, match="refusing to recompute"):
        m.preprocess(torch.tensor([6, 2]), torch.ones(2, 4), duplex=duplex, duplex_token_offset=278)
    _, embeds, _ = m.preprocess(torch.tensor([2]), torch.ones(1, 4), duplex=duplex, duplex_token_offset=279)
    assert embeds.shape == (1, 4)


def model(decode=False):
    m = Model.__new__(Model)
    m.model_stage = "llm"
    m.model = SimpleNamespace()
    m._minicpmo_pd_decode = decode
    m._minicpmo_pd_thinker = True
    m._minicpmo45_native_duplex_token_ids_cache = IDS
    m._minicpmo45_tokenizer_cache = SimpleNamespace(bad_token_ids=[], all_special_ids=[])
    return m


def row(request="r", seq=10, payload=None, idx=0, session="s"):
    return DuplexSamplingRow(idx, request, session, 0, seq, payload, 50, epoch=0)


def snapshot(state, seq=10):
    return pack_sampling_state(state, incarnation=0, epoch=0, seq=seq)


def test_decode_installs_once_retains_control_and_cleans_up():
    m = model(True)
    state = DecodeSamplingState(current_turn_ended=False, generated_tokens=[8], current_segment_output_tokens=[2])
    r = row(payload={SAMPLING_STATE_KEY: snapshot(state), "is_speech": True})
    logits = torch.zeros(1, 12)
    m.prepare_duplex_sampling(logits, None, (r,))
    assert m._finalize_minicpmo45_native_duplex_sample(0, 1, IDS) == 2
    m._record_minicpmo45_duplex_generation_token(0, 9)
    m.prepare_duplex_sampling(logits, None, (r,))
    assert m._minicpmo45_duplex_state_for_row(0).generated_tokens == [8, 9]
    frozen = m.snapshot_duplex_sampling_outputs(["r"])[SAMPLING_STATE_WIRE_KEY][0]
    m.on_requests_finished(["r"])
    assert not m._minicpmo45_pd_sampling_states
    assert unpack_sampling_state(frozen)[0].generated_tokens == [8, 9]
    assert not hasattr(m, "_minicpmo45_duplex_data_plane_helper")


def test_decode_mixed_sessions_and_reordered_rows_are_isolated():
    m = model(True)
    a = row("a", payload={SAMPLING_STATE_KEY: snapshot(DecodeSamplingState(current_turn_ended=False))})
    b = row("b", idx=1, session="other", payload={SAMPLING_STATE_KEY: snapshot(DecodeSamplingState())})
    m.prepare_duplex_sampling(torch.zeros(2, 12), None, (a, b))
    assert m._finalize_minicpmo45_native_duplex_sample(0, 1, IDS) == 2
    assert m._finalize_minicpmo45_native_duplex_sample(1, 1, IDS) == 1
    m.prepare_duplex_sampling(
        torch.zeros(2, 12), None, (row("b", session="other", payload=b.payload), row("a", idx=1, payload=a.payload))
    )
    assert m._finalize_minicpmo45_native_duplex_sample(0, 1, IDS) == 1
    assert m._finalize_minicpmo45_native_duplex_sample(1, 1, IDS) == 2


@pytest.mark.parametrize(
    "payload", [{}, {SAMPLING_STATE_KEY: [1]}, {SAMPLING_STATE_KEY: snapshot(DecodeSamplingState(), seq=9)}]
)
def test_decode_fails_closed_on_missing_or_stale_state(payload):
    with pytest.raises(ValueError):
        model(True).prepare_duplex_sampling(torch.zeros(1, 12), None, (row(payload=payload),))


def test_split_sampler_matches_unsplit_given_identical_logits_across_units():
    p, d, native = model(), model(True), model()
    p_state, native_state = DecodeSamplingState(), DecodeSamplingState()
    p._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("s", 0): p_state})
    native._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("s", 0): native_state})
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import MiniCPMO45Stage0DuplexRuntime as Runtime

    helper = SimpleNamespace(**IDS)
    # Include redirected LISTEN, repeated text, CHUNK_EOS and TURN_EOS.
    for seq, targets in enumerate(([2, 8, 1, 3], [2, 8, 5], [2, 8, 3]), start=10):
        native_state.current_segment_output_tokens = []
        p_state.current_segment_output_tokens = []
        native_tokens, split_tokens = [], []
        for step, target in enumerate(targets):
            logits = torch.full((1, 12), -100.0)
            logits[0, target] = 20.0
            if target == 8:
                logits[0, 9] = 19.5  # Cross-unit repetition can change the winner.
            metadata = SimpleNamespace(all_greedy=True, output_token_ids=[list(native_tokens)], generators={})
            nr = row("native", seq=seq, payload={"is_speech": True})
            native.prepare_duplex_sampling(logits.clone(), metadata, (nr,))
            expected = native.sample(logits.clone(), metadata).sampled_token_ids.item()
            native_tokens.append(expected)
            if step == 0:
                p.prepare_duplex_sampling(logits.clone(), metadata, (row("p", seq=seq, payload={"is_speech": True}),))
                actual = p.sample(logits.clone(), metadata).sampled_token_ids.item()
                wire = list(p.snapshot_duplex_sampling_outputs(["p"])[SAMPLING_STATE_WIRE_KEY][0])
            else:
                dr = row("d", seq=seq, payload={SAMPLING_STATE_KEY: wire, "is_speech": True})
                d.prepare_duplex_sampling(logits.clone(), metadata, (dr,))
                # Engine D output excludes P's first token, unlike native.
                metadata.output_token_ids = [split_tokens[1:]]
                actual = d.sample(logits.clone(), metadata).sampled_token_ids.item()
            split_tokens.append(actual)
            assert actual == expected
        returned = list(d.snapshot_duplex_sampling_outputs(["d"])[SAMPLING_STATE_WIRE_KEY][0])
        p_state.pd_feedback_append_identity = None
        Runtime.apply_pd_decode_feedback(helper, p_state, split_tokens, epoch=0, seq=seq + 1, sampling_state=returned)
        assert snapshot(p_state, seq) == snapshot(native_state, seq)
        # Retrying feedback must not duplicate the history.
        Runtime.apply_pd_decode_feedback(helper, p_state, split_tokens, epoch=0, seq=seq + 1, sampling_state=returned)
        assert snapshot(p_state, seq) == snapshot(native_state, seq)
        d.on_requests_finished(["d"])


def test_wire_state_is_replaced_not_concatenated():
    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.multimodal_accumulation import replace_snapshot_keys

    a = MultimodalPayload.from_dict({SAMPLING_STATE_WIRE_KEY: torch.tensor(snapshot(DecodeSamplingState()))})
    b = MultimodalPayload.from_dict(
        {SAMPLING_STATE_WIRE_KEY: torch.tensor(snapshot(DecodeSamplingState(generated_tokens=[7])))}
    )
    replace_snapshot_keys(a, b)
    a = a.merged_with(b)
    assert unpack_sampling_state(a[SAMPLING_STATE_WIRE_KEY])[0].generated_tokens == [7]


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_native_sampler_does_not_replace_long_text_with_untrained_character_boundary(decode, greedy):
    m = model(decode)
    # Official streaming_generate bounds a unit by model tokens, not decoded
    # characters. A text token crossing 28 characters must still enter KV.
    m._minicpmo45_tokenizer_cache.decode = lambda *args, **kwargs: "字" * 40
    state = DecodeSamplingState(current_turn_ended=False, current_segment_output_tokens=[2, 8])
    payload = {"is_speech": True}
    if decode:
        payload[SAMPLING_STATE_KEY] = snapshot(state)
    else:
        m._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("s", 0): state})
    metadata = SimpleNamespace(all_greedy=greedy, output_token_ids=[[2, 8]], generators={})
    logits = torch.full((1, 12), float("-inf"))
    logits[0, 9] = 100.0
    m.prepare_duplex_sampling(logits, metadata, (row(payload=payload),))
    actual = m.sample(logits, metadata).sampled_token_ids.item()
    assert actual == 9
    actual_state = m._minicpmo45_duplex_state_for_row(0)
    assert actual_state.generated_tokens == [9]
    assert actual_state.current_segment_output_tokens == [2, 8, 9]


@pytest.mark.parametrize("decode", [False, True])
def test_native_sampler_keeps_official_twenty_token_unit_budget(decode):
    m = model(decode)
    recent = [2] + [8] * 18
    state = DecodeSamplingState(current_turn_ended=False, current_segment_output_tokens=recent.copy())
    payload = {"is_speech": True}
    if decode:
        payload[SAMPLING_STATE_KEY] = snapshot(state)
    else:
        m._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("s", 0): state})
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[recent], generators={})
    logits = torch.zeros(1, 12)
    logits[0, 9] = 100.0
    m.prepare_duplex_sampling(logits, metadata, (row(payload=payload),))
    assert m.sample(logits, metadata).sampled_token_ids.item() == IDS["chunk_eos_token_id"]
    assert m._minicpmo45_duplex_state_for_row(0).generated_tokens == []


def test_forced_listen_is_preserved_without_reapplying_first_token_gate():
    m = model(True)
    r = row(payload={SAMPLING_STATE_KEY: snapshot(DecodeSamplingState(current_turn_ended=False)), "force_listen": True})
    logits = torch.zeros(1, 12)
    m.prepare_duplex_sampling(logits, None, (r,))
    assert torch.isfinite(logits).all()
    assert m._finalize_minicpmo45_native_duplex_sample(0, 1, IDS) == 1
    m._record_minicpmo45_duplex_generation_token(0, 1)
    assert m._minicpmo45_duplex_state_for_row(0).generated_tokens == []


@pytest.mark.parametrize("terminal", [1, 3, 4])
def test_p_terminator_is_acknowledged_without_a_second_decode_decision(terminal):
    m = model(True)
    wire = snapshot(DecodeSamplingState(current_segment_output_tokens=[terminal]))
    m.prepare_duplex_sampling(torch.zeros(1, 12), None, (row(payload={SAMPLING_STATE_KEY: wire}),))
    logits = torch.zeros(1, 12)
    logits[0, 8] = 100  # Must not restart generation after P already ended the unit.
    actual = m.sample(logits, SimpleNamespace(all_greedy=True, output_token_ids=[[]]))
    assert actual.sampled_token_ids.item() == terminal
    assert list(m.snapshot_duplex_sampling_outputs(["r"])[SAMPLING_STATE_WIRE_KEY][0]) == wire


def test_integer_state_snapshot_survives_batch_split_wire_and_live_cleanup():
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.utils.mm_outputs import to_payload_element
    from vllm_omni.worker.gpu_ar_model_runner import (
        GPUARModelRunner,
        _clone_cuda_tensor_payload,
        _copy_tensor_payload_to_cpu,
    )

    m = model(True)
    states = [DecodeSamplingState(generated_tokens=[7, 8], rng_seed=(1 << 64)-1, rng_offset=123),
              DecodeSamplingState(generated_tokens=[9], current_segment_output_tokens=[2])]
    rows = (row("a", payload={SAMPLING_STATE_KEY: snapshot(states[0])}),
            row("b", idx=1, session="other", payload={SAMPLING_STATE_KEY: snapshot(states[1])}))
    m.prepare_duplex_sampling(torch.zeros(2, 12), None, rows)
    payload = m.snapshot_duplex_sampling_outputs(["b", "a"])
    frozen = payload[SAMPLING_STATE_WIRE_KEY]
    assert all(type(v) is tuple and all(type(n) is int for n in v) for v in frozen)
    m._minicpmo45_duplex_state_for_row(0).generated_tokens.append(10)
    m.on_requests_finished(["a", "b"])
    cuda_sources = []
    copied = _copy_tensor_payload_to_cpu(_clone_cuda_tensor_payload(payload, cuda_sources), pin_memory=False)
    assert cuda_sources == []
    for i, expected in enumerate((states[1], states[0])):
        selected = to_payload_element(copied, idx=i, start=i, end=i+1, seq_len=2)
        restored = MsgpackDecoder().decode(MsgpackEncoder().encode(selected))
        mm = MultimodalPayload.from_dict(restored)
        assert SAMPLING_STATE_WIRE_KEY in mm.metadata
        actual, identity = unpack_sampling_state(mm[SAMPLING_STATE_WIRE_KEY])
        assert identity == (0, 0, 10)
        assert pack_sampling_state(actual, incarnation=0, epoch=0, seq=10) == snapshot(expected)
        # The real worker output remains tensor-only. This optimization only
        # removes intermediate tensor copies, not final wire serialization.
        runner = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(engine_output_type="latent")))
        wire = GPUARModelRunner._build_multimodal_outputs(runner, [selected])[0]
        assert isinstance(wire[SAMPLING_STATE_WIRE_KEY], torch.Tensor)
        encoded = MsgpackEncoder().encode(wire)
        native = {SAMPLING_STATE_WIRE_KEY: torch.tensor(snapshot(expected), dtype=torch.int64)}
        assert [bytes(b) for b in encoded] == [bytes(b) for b in MsgpackEncoder().encode(native)]
        restored = MsgpackDecoder(dict[str, torch.Tensor]).decode(encoded)
        actual, identity = unpack_sampling_state(restored[SAMPLING_STATE_WIRE_KEY])
        assert identity == (0, 0, 10)
        assert pack_sampling_state(actual, incarnation=0, epoch=0, seq=10) == snapshot(expected)


def test_turn_eos_does_not_end_the_model_unit_or_reapply_input_gating():
    m = model(True)
    wire = snapshot(DecodeSamplingState(current_turn_ended=True, current_segment_output_tokens=[5]))
    m.prepare_duplex_sampling(torch.zeros(1, 12), None, (row(payload={SAMPLING_STATE_KEY: wire}),))
    logits = torch.zeros(1, 12)
    logits[0, 3] = 100  # Native next step emits the unit's CHUNK_EOS.
    actual = m.sample(logits, SimpleNamespace(all_greedy=True, output_token_ids=[[]]))
    assert actual.sampled_token_ids.item() == 3
    assert m._minicpmo45_duplex_state_for_row(0).current_segment_output_tokens == [5, 3]
    m = model()
    m._minicpmo45_duplex_data_plane_helper = SimpleNamespace(
        sessions={("s", 0): DecodeSamplingState(current_turn_ended=True, current_segment_output_tokens=[2, 5])}
    )
    m.prepare_duplex_sampling(logits, None, (row(payload={"is_speech": False}),))
    assert logits[0, 3] == 100


def test_native_stop_ids_match_official_chunk_not_turn_boundaries(monkeypatch):
    from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45NativeDuplexServingAdapter as Adapter

    mapping = {"<|listen|>": 1, "<|chunk_eos|>": 3, "<|chunk_tts_eos|>": 4, "<|turn_eos|>": 5}
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: mapping[token], unk_token_id=-1)
    monkeypatch.setattr(Adapter, "_load_native_tokenizer", lambda _: tokenizer)
    assert set(Adapter._native_stage0_stop_token_ids(None)) == {1, 3, 4}
