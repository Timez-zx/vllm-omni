"""CPU-only checks against the ACTUAL production compaction algorithm."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.experimental.fullduplex.minicpmo45 import numerical_probe as probe
from vllm_omni.experimental.fullduplex.minicpmo45.numerical_probe import (
    capture_forward,
    gap_blocks,
    map_absolute_slots,
    parse_seq_selection,
    prefix_positions,
    raw_kv,
    selected_rows,
    snapshot_runner_rows,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    # A preceding GPU test may already own a context; these CPU probes must
    # neither use CUDA nor change that process-wide state themselves.
    initialized = torch.cuda.is_initialized()

    def unexpected_cuda_use(*args, **kwargs):
        pytest.fail("CPU numerical-probe tests must not initialize or use CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected_cuda_use)
    yield
    assert torch.cuda.is_initialized() == initialized

# Importing the full vLLM package may initialize a CUDA platform context even
# when every test tensor is CPU. Compile the actual standalone production
# function body instead; this executes precisely its original tensor formula.
_source = (Path(__file__).resolve().parents[2] / "vllm_omni/engine/pinned_prefix_window.py").read_text()
_function = next(
    node for node in ast.parse(_source).body if isinstance(node, ast.FunctionDef) and node.name == "compact_window_view"
)
_namespace = {"torch": torch, "cdiv": lambda a, b: (a + b - 1) // b}
exec(compile(ast.Module(body=[_function], type_ignores=[]), "production_compact_window_view", "exec"), _namespace)
compact_window_view = _namespace["compact_window_view"]


def table_view(history, qlen, *, pin=128, window=8000, block_size=16, offset=0):
    seq = history + qlen
    width = (seq + block_size - 1) // block_size
    logical = torch.arange(offset + 1, offset + width + 1).int().unsqueeze(0)
    gap = gap_blocks(history=history, window=window, pin=pin, block_size=block_size)
    logical[:, pin // block_size : pin // block_size + gap] = 0
    view, lengths = compact_window_view(
        logical,
        torch.tensor([seq]),
        torch.tensor([0, qlen]),
        prefix_tokens=pin,
        window_tokens=window,
        block_size=block_size,
    )
    return logical[0], view[0], int(lengths[0])


@pytest.mark.parametrize("history", [0, 127, 128, 8000, 8127, 8143, 8200, 52345, 89514, 262100])
@pytest.mark.parametrize("qlen", [1, 7, 220])
def test_absolute_mapping_matches_actual_production_compact_table(history, qlen):
    logical, view, compact_length = table_view(history, qlen)
    kwargs = dict(history=history, window=8000, pin=128, block_size=16)
    prefix = prefix_positions(**kwargs)
    query = torch.arange(history, history + qlen)
    selected = torch.cat((prefix, query))
    actual = map_absolute_slots(view, selected, **kwargs)
    expected = logical[selected // 16].long() * 16 + selected % 16
    assert torch.equal(actual, expected)
    assert compact_length == history + qlen - 16 * gap_blocks(**kwargs)
    assert not bool((logical[selected // 16] == 0).any())


def test_partial_boundary_retains_modulo_offset_but_excludes_masked_prefix_samples():
    _, view, _ = table_view(8200, 7)
    kwargs = dict(history=8200, window=8000, pin=128, block_size=16)
    assert gap_blocks(**kwargs) == 4
    # First tail block is abs192..207; query8200 only attends abs201 onward.
    assert int(map_absolute_slots(view, [201], **kwargs)[0] % 16) == 9
    selected = prefix_positions(**kwargs).tolist()
    assert 201 in selected and 207 in selected and 208 in selected
    assert not any(128 <= position < 201 for position in selected)
    with pytest.raises(ValueError, match="middle-gap"):
        map_absolute_slots(view, [128], **kwargs)


def test_mixed_session_rows_do_not_share_gap_or_physical_blocks():
    histories, qlens = [279, 8200, 52800], [1, 220, 7]
    seqs = torch.tensor([h + q for h, q in zip(histories, qlens)])
    width = (int(seqs.max()) + 15) // 16
    logical = torch.stack([torch.arange(1 + row * 10000, width + 1 + row * 10000) for row in range(3)]).int()
    for row, history in enumerate(histories):
        gap = gap_blocks(history=history, window=8000, pin=128, block_size=16)
        logical[row, 8 : 8 + gap] = 0
    starts = torch.tensor([0, 1, 221, 228])
    view, lengths = compact_window_view(logical, seqs, starts, prefix_tokens=128, window_tokens=8000, block_size=16)
    for row, (history, qlen) in enumerate(zip(histories, qlens)):
        kwargs = dict(history=history, window=8000, pin=128, block_size=16)
        positions = torch.cat((prefix_positions(**kwargs), torch.arange(history, history + qlen)))
        expected = logical[row, positions // 16] * 16 + positions % 16
        assert torch.equal(map_absolute_slots(view[row], positions, **kwargs), expected)
        assert int(lengths[row]) == history + qlen - gap_blocks(**kwargs) * 16


def test_old_absolute_column_lookup_is_explicitly_wrong_after_first_window():
    logical, view, _ = table_view(8200, 220)
    position = 8180  # Old probe's last256 span, in bounds but WRONG physical block.
    old = int(view[position // 16]) * 16 + position % 16
    correct = int(logical[position // 16]) * 16 + position % 16
    assert old != correct
    assert int(map_absolute_slots(view, [position], history=8200, window=8000, pin=128, block_size=16)[0]) == correct


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float8_e4m3fn])
def test_capture_preserves_storage_bytes_without_round_trip(dtype):
    values = torch.arange(4 * 2 * 16 * 8).reshape(4, 2, 16, 8) % 197
    cache = values.to(dtype)
    slots = torch.tensor([16, 19, 35, 63])
    record = raw_kv(cache, slots)
    expected = cache[slots // 16, :, slots % 16, :].contiguous()
    assert record["shape"] == tuple(expected.shape)
    assert record["storage_dtype"] == str(dtype)
    assert torch.equal(record["raw_bytes"], expected.view(torch.uint8))


def test_inputs_only_capture_never_reads_kv_and_is_labelled(tmp_path, monkeypatch):
    model, rows, positions, _cache, _logical = _pre_capture_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_LAYERS", "none")
    embeddings = torch.tensor([[1.0, 2.0]])
    hidden = torch.tensor([[3.0, 4.0]])
    monkeypatch.setattr(probe, "raw_kv", lambda *args: pytest.fail("inputs-only capture must not read KV"))
    probe.capture_forward(model, positions, positions, embeddings, hidden, rows=rows)
    record = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert record["kv_layer_selection"] == "none"
    assert record["kv"] == record["prefix_kv"] == {}
    assert torch.equal(record["inputs_embeds"], embeddings)
    assert torch.equal(record["hidden"], hidden)
    assert torch.equal(record["positions"], positions)


def test_query_and_prefix_across_pd_have_separate_absolute_positions():
    kwargs_p = dict(history=8200, window=8000, pin=128, block_size=16)
    kwargs_d = dict(history=8420, window=8000, pin=128, block_size=16)
    _, view_p, _ = table_view(8200, 220, offset=1)
    _, view_d, _ = table_view(8420, 1, offset=9999)
    p_positions = set(prefix_positions(**kwargs_p).tolist()) | set(range(8200, 8420))
    d_positions = set(prefix_positions(**kwargs_d).tolist())
    common = sorted(p_positions & d_positions)
    assert set(range(128)) <= set(common)
    assert set(range(8200, 8420)) <= set(common)
    assert gap_blocks(**kwargs_p) != gap_blocks(**kwargs_d)
    assert not torch.equal(
        map_absolute_slots(view_p, common, **kwargs_p), map_absolute_slots(view_d, common, **kwargs_d)
    )


def test_null_block_and_out_of_bounds_are_not_silently_captured():
    _, view, _ = table_view(8200, 1)
    kwargs = dict(history=8200, window=8000, pin=128, block_size=16)
    bad = view.clone()
    bad[0] = 0
    with pytest.raises(ValueError, match="null"):
        map_absolute_slots(bad, [0], **kwargs)
    with pytest.raises(ValueError, match="out of range"):
        map_absolute_slots(view, [999999], **kwargs)


def test_current_runner_identity_is_not_previous_sampling_identity():
    def info(sid, seq):
        return {"duplex": {"data_plane": True, "session_id": sid, "incarnation": 0, "epoch": 0, "seq": seq}}

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["new-b", "new-a"]),
        model_intermediate_buffer={"new-a": info("alice", 38), "new-b": info("bob", 240)},
        model=SimpleNamespace(_minicpmo45_sampling_rows=["old-unrelated"]),
    )
    snapshot = snapshot_runner_rows(runner)
    assert [(r["row_index"], r["request_id"], r["session_id"], r["seq"]) for r in snapshot] == [
        (0, "new-b", "bob", 240),
        (1, "new-a", "alice", 38),
    ]
    runner.model_intermediate_buffer["new-a"]["duplex"]["seq"] = 39
    assert snapshot[1]["seq"] == 38
    selected = selected_rows(
        snapshot,
        {"MINICPMO45_NUMERICAL_PROBE_SEQS": "37-39,239-241", "MINICPMO45_NUMERICAL_PROBE_SESSIONS": '["alice"]'},
    )
    assert selected == [snapshot[1]]


def test_diagnostic_selection_does_not_assume_global_position_is_unit_index():
    rows = [dict(row_index=0, request_id="r", session_id="a", incarnation=0, epoch=0, seq=240)]
    assert selected_rows(rows, {"MINICPMO45_NUMERICAL_PROBE_SEQS": "239-241"}) == rows
    assert selected_rows(rows, {"MINICPMO45_NUMERICAL_PROBE_SEQS": "37-39"}) == []
    assert parse_seq_selection("37-39,239-241") == {37, 38, 39, 239, 240, 241}
    with pytest.raises(ValueError):
        parse_seq_selection("0-99999999")


def test_actual_capture_labels_mixed_rows_and_records_raw_prefix_scale(tmp_path, monkeypatch):
    import vllm.forward_context

    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "37-39,239-241")
    monkeypatch.delenv("MINICPMO45_NUMERICAL_PROBE_SESSIONS", raising=False)
    histories, lengths = (8200, 279), (7, 1)
    tables, views, compact_lengths = [], [], []
    for row, (history, qlen) in enumerate(zip(histories, lengths)):
        logical, view, compact = table_view(history, qlen, offset=row * 1000)
        tables.append(logical)
        views.append(view)
        compact_lengths.append(compact)
    width = max(v.numel() for v in views)
    block_table = torch.stack([torch.nn.functional.pad(v, (0, width - v.numel())) for v in views])
    positions = torch.cat([torch.arange(h, h + q) for h, q in zip(histories, lengths)])
    slots = torch.cat(
        [
            table[torch.arange(h, h + q) // 16] * 16 + torch.arange(h, h + q) % 16
            for table, h, q in zip(tables, histories, lengths)
        ]
    )
    meta = SimpleNamespace(
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 7, 8]),
        seq_lens=torch.tensor(compact_lengths),
        slot_mapping=slots,
        block_table=block_table,
        rswa_prefix_lens=torch.tensor([128, 128]),
        rswa_window=8000,
    )
    cache = (torch.arange(1100 * 2 * 16 * 8) % 191).reshape(1100, 2, 16, 8).to(torch.bfloat16)
    module = SimpleNamespace(
        kv_cache=cache,
        kv_cache_dtype="bfloat16",
        _k_scale=torch.tensor(0.5),
        _v_scale=torch.tensor(0.25),
        layer_name="layer0",
    )
    model = SimpleNamespace(
        _minicpmo45_numerical_probe_dir=str(tmp_path),
        _minicpmo_pd_decode=True,
        thinker=SimpleNamespace(
            llm=SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [("layers.0.self_attn.attn", module)]))
        ),
    )
    monkeypatch.setattr(
        vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"layer0": meta})
    )
    rows = [
        dict(row_index=i, request_id=f"r{i}", session_id=f"s{i}", incarnation=0, epoch=0, seq=seq)
        for i, seq in enumerate((240, 38))
    ]
    capture_forward(model, positions, positions, torch.ones(8, 2), torch.zeros(8, 2), rows=rows)
    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == 2
    for row, path in enumerate(files):
        record = torch.load(path, weights_only=False)
        assert record["identity"] == rows[row]
        assert record["positions"].tolist() == list(range(histories[row], histories[row] + lengths[row]))
        assert record["first_forward_for_request_seq"]
        prefix = record["prefix_kv"]["layers.0.self_attn.attn"]
        assert prefix["k_scale"] == 0.5 and prefix["v_scale"] == 0.25
        p = prefix["positions"]
        expected = raw_kv(cache, tables[row][p // 16] * 16 + p % 16)
        assert torch.equal(prefix["raw_bytes"], expected["raw_bytes"])
    # Every generated position may be captured; the large historical sample
    # is emitted only once for each request/seq, even if decode runs 20 steps.
    capture_forward(model, positions, positions, torch.ones(8, 2), torch.zeros(8, 2), rows=rows)
    repeated = [torch.load(path, weights_only=False) for path in sorted(tmp_path.glob("*.pt"))[2:]]
    assert all(not record["first_forward_for_request_seq"] and record["prefix_kv"] == {} for record in repeated)


def _pre_capture_fixture(tmp_path, monkeypatch, *, stage="d", history=8200, qlen=1, window=8000):
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "37-39,239-241")
    monkeypatch.delenv("MINICPMO45_NUMERICAL_PROBE_SESSIONS", raising=False)
    args = dict(history=history, pin=128, window=window, block_size=16)
    gap = probe.gap_blocks(**args)
    logical = torch.arange(1, (history + qlen + 15) // 16 + 1)
    compact = torch.cat((logical[:8], logical[8 + gap :]))
    positions = torch.arange(history, history + qlen)
    slots = logical[positions // 16] * 16 + positions % 16
    meta = SimpleNamespace(
        num_actual_tokens=qlen,
        query_start_loc=torch.tensor([0, qlen]),
        seq_lens=torch.tensor([history + qlen - gap * 16]),
        slot_mapping=slots,
        block_table=compact.unsqueeze(0),
        rswa_prefix_lens=torch.tensor([128]),
        rswa_window=window,
    )
    cache = (torch.arange((len(logical) + 1) * 16 * 4) % 191).reshape(len(logical) + 1, 1, 16, 4).to(torch.uint8)
    attention = SimpleNamespace(
        kv_cache=cache,
        kv_cache_dtype="fp8",
        _k_scale=torch.tensor(0.5),
        _v_scale=torch.tensor(0.25),
        layer_name="layer0",
    )
    model = SimpleNamespace(
        _minicpmo45_numerical_probe_dir=str(tmp_path),
        _minicpmo_pd_decode=stage == "d",
        thinker=SimpleNamespace(
            llm=SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [("layers.0.self_attn.attn", attention)]))
        ),
    )
    module = ModuleType("vllm.forward_context")
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata={"layer0": meta})
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)
    rows = [
        dict(
            row_index=0, request_id="D38" if stage == "d" else "P38", session_id="alice", incarnation=0, epoch=0, seq=38
        )
    ]
    return model, rows, positions, cache, logical


def _pre_capture_files(tmp_path):
    return [torch.load(path, map_location="cpu", weights_only=True) for path in sorted(tmp_path.glob("*.pt"))]


def test_before_snapshot_is_immutable_prefix_only_and_after_remains_separate(tmp_path, monkeypatch):
    model, rows, positions, cache, logical = _pre_capture_fixture(tmp_path, monkeypatch)
    probe.capture_forward(model, None, positions, None, None, rows=rows, phase="before_backbone_forward")
    before = _pre_capture_files(tmp_path)[0]
    assert before["capture_phase"] == "before_backbone_forward"
    assert before["kv"] == {} and "hidden" not in before
    prefix = before["prefix_kv"]["layers.0.self_attn.attn"]
    assert int(prefix["positions"].max()) < int(positions[0])
    assert before["capture_begin_ns"] <= before["capture_end_ns"]
    old_bytes = prefix["raw_bytes"].clone()
    # Simulate a later writer changing KV. Pre snapshot must not alias it.
    cache[:] ^= 0x7F
    inputs = torch.ones(len(positions), 4)
    probe.capture_forward(model, positions, positions, inputs, inputs, rows=rows)
    pre, post = _pre_capture_files(tmp_path)
    assert pre["first_forward_for_request_seq"] and post["first_forward_for_request_seq"]
    assert torch.equal(pre["prefix_kv"]["layers.0.self_attn.attn"]["raw_bytes"], old_bytes)
    assert not torch.equal(post["prefix_kv"]["layers.0.self_attn.attn"]["raw_bytes"], old_bytes)
    assert post["capture_phase"] == "after_backbone_forward"
    assert post["kv"]["layers.0.self_attn.attn"]["positions"].tolist() == positions.tolist()
    assert post["capture_begin_ns"] >= pre["capture_end_ns"]
    # A later D token must not trigger a second historical pre capture.
    probe.capture_forward(model, None, positions, None, None, rows=rows, phase="before_backbone_forward")
    assert len(_pre_capture_files(tmp_path)) == 2


@pytest.mark.parametrize(
    "window,history,qlen",
    [
        (8000, 8200, 220),
        (8000, 52345, 220),
        (8000, 89514, 12),
        (8000, 0, 8400),
        (36000, 36000, 220),
        (36000, 36200, 220),
        (36000, 52000, 220),
        (36000, 92000, 220),
    ],
)
def test_P_sampling_covers_predicted_D_visible_boundary_without_rebasing(window, history, qlen):
    args = dict(history=history, window=window, pin=128, block_size=16)
    source = set(probe.prefix_positions(**args).tolist())
    source |= set(probe.p_next_boundary_positions(**args, query_length=qlen).tolist())
    source |= set(range(history, history + qlen))
    d_samples = set(probe.prefix_positions(**dict(args, history=history + qlen)).tolist())
    assert d_samples <= source
    if history == 8200:
        assert set(range(421, 453)) <= source
        assert not set(range(421, 453)) <= set(probe.prefix_positions(**args).tolist())


def test_P_before_is_noop_and_after_includes_next_D_boundary(tmp_path, monkeypatch):
    model, rows, positions, cache, logical = _pre_capture_fixture(tmp_path, monkeypatch, stage="p", qlen=220)
    probe.capture_forward(model, None, positions, None, None, rows=rows, phase="before_backbone_forward")
    assert not _pre_capture_files(tmp_path)
    inputs = torch.ones(len(positions), 4)
    probe.capture_forward(model, positions, positions, inputs, inputs, rows=rows)
    record = _pre_capture_files(tmp_path)[0]
    assert set(range(421, 453)) <= set(record["prefix_kv"]["layers.0.self_attn.attn"]["positions"].tolist())
    assert len(record["kv"]["layers.0.self_attn.attn"]["positions"]) == 220


def test_unselected_unit_adds_no_capture_or_seen_state(tmp_path, monkeypatch):
    model, rows, positions, cache, logical = _pre_capture_fixture(tmp_path, monkeypatch)
    rows[0]["seq"] = 52
    probe.capture_forward(model, None, positions, None, None, rows=rows, phase="before_backbone_forward")
    assert not _pre_capture_files(tmp_path)
    assert not hasattr(model, "_minicpmo45_numerical_probe_pre_seen")


def test_default_36k_capture_and_later_seq_are_taken_from_metadata(tmp_path, monkeypatch):
    model, rows, positions, cache, logical = _pre_capture_fixture(tmp_path, monkeypatch, history=52000, window=36000)
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "160-175,239-241")
    rows[0]["seq"] = 170
    probe.capture_forward(model, None, positions, None, None, rows=rows, phase="before_backbone_forward")
    record = _pre_capture_files(tmp_path)[0]
    assert record["window_tokens"] == 36000 and record["identity"]["seq"] == 170
    samples = set(record["prefix_kv"]["layers.0.self_attn.attn"]["positions"].tolist())
    assert set(range(16001, 16033)) <= samples
    assert not set(range(44001, 44033)) <= samples  # Would be the wrong 8k boundary.


@pytest.mark.parametrize(
    "enabled,is_decode,seq,expected",
    [
        (True, True, 38, ["before_backbone_forward", "model", "after_backbone_forward"]),
        (True, False, 38, ["model", "after_backbone_forward"]),
        (True, True, 52, ["model"]),
        (False, True, 38, ["model"]),
    ],
)
def test_real_model_hook_orders_captures_and_is_opt_in(monkeypatch, enabled, is_decode, seq, expected):
    """Execute the actual model hook, not a test's reimplementation of it."""
    model_path = (
        Path(__file__).resolve().parents[2] / "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py"
    )
    tree = ast.parse(model_path.read_text())
    fragment = None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        start = next(
            (
                i
                for i, statement in enumerate(body)
                if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "probe_embeds" for target in statement.targets)
            ),
            None,
        )
        if start is None:
            continue
        end = next(
            i
            for i in range(start, len(body))
            if isinstance(body[i], ast.If) and ast.unparse(body[i].test) == "probe_embeds is not None"
        )
        fragment = ast.Module(body=body[start : end + 1], type_ignores=[])
        break
    assert fragment is not None
    events = []

    def capture(_model, _ids, _positions, embeds, _hidden, *, rows, phase="after_backbone_forward"):
        events.append(phase)
        assert rows[0]["seq"] == seq
        if phase == "before_backbone_forward":
            assert embeds is None and _hidden is None
        else:
            assert torch.equal(embeds, torch.ones(1, 2))  # Before in-place model update.

    def thinker(**kwargs):
        events.append("model")
        kwargs["inputs_embeds"].add_(100)
        return torch.zeros(1, 2)

    monkeypatch.setattr(probe, "capture_forward", capture)
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "37-39")
    monkeypatch.delenv("MINICPMO45_NUMERICAL_PROBE_SESSIONS", raising=False)
    rows = [dict(row_index=0, request_id="r", session_id="s", incarnation=0, epoch=0, seq=seq)]
    namespace = {
        "os": __import__("os"),
        "self": SimpleNamespace(
            _minicpmo_pd_thinker=True,
            _minicpmo_pd_decode=is_decode,
            _minicpmo45_numerical_probe_dir="enabled" if enabled else None,
            thinker=thinker,
        ),
        "kwargs": {"minicpmo_numerical_probe_rows": rows},
        "thinker_input_ids": torch.tensor([1]),
        "thinker_positions": torch.tensor([8200]),
        "thinker_inputs_embeds": torch.ones(1, 2),
        "intermediate_tensors": None,
        "added_batch_dim": False,
    }
    exec(compile(fragment, str(model_path), "exec"), namespace)
    assert events == expected


@pytest.mark.parametrize("enabled,seq", [(True, 2), (False, 2), (True, 52)])
def test_layer_io_real_module_hooks_are_bounded_immutable_and_removed(tmp_path, monkeypatch, enabled, seq):
    model, rows, positions, _, _ = _pre_capture_fixture(tmp_path, monkeypatch, history=279, qlen=2)
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "1-3,50-60")
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_LAYER_IO", "1" if enabled else "0")
    rows[0]["seq"] = seq

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.weight_scale = torch.nn.Parameter(torch.tensor(0.125))

        def forward(self, x):
            return x + 1, None

    class Attention(torch.nn.Module):
        impl = SimpleNamespace(supports_quant_query_input=False, _omni_triton_q_quantization_disabled=True)

        def forward(self, q, k, v):
            return q + k + v

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = Projection()
            self.down_proj = Projection()

        def forward(self, x):
            x, _ = self.gate_up_proj(x)
            return self.down_proj(x)[0]

    qkv, attention, output, mlp = Projection(), Attention(), Projection(), MLP()
    modules = {
        "layers.0.self_attn.qkv_proj": qkv,
        "layers.0.self_attn.attn": attention,
        "layers.0.self_attn.o_proj": output,
        "layers.0.mlp": mlp,
        "layers.0.mlp.gate_up_proj": mlp.gate_up_proj,
        "layers.0.mlp.down_proj": mlp.down_proj,
    }
    model.thinker.llm.model.named_modules = lambda: list(modules.items())
    x = torch.zeros(2, 4)
    with probe.capture_layer0_io(model, positions, rows=rows):
        projected = qkv(x)[0]
        result = mlp(output(attention(projected, projected + 1, projected + 2))[0])
    assert torch.equal(result, torch.full_like(x, 9))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in modules.values())
    files = list((tmp_path / "layer-io").glob("*.pt"))
    if not enabled or seq != 2:
        assert not files
    else:
        assert len(files) == 1
        record = torch.load(files[0], weights_only=True)
        assert record["identity"] == rows[0]
        assert record["Q_quantization_disabled"] is True
        assert record["supports_quant_query_input"] is False
        import json

        weights = json.loads(Path(record["weight_fingerprint_file"]).read_text())
        assert weights["layers"]["qkv_proj"]["weight"]["shape"] == [2, 2]
        assert weights["layers"]["qkv_proj"]["weight_scale"]["shape"] == []
        assert len(weights["layers"]["qkv_proj"]["weight"]["sha256"]) == 64
        assert len(record["tensors"]) == 14
        assert torch.equal(record["tensors"]["attn.Q"], torch.ones(2, 4))
        result.zero_()
        assert torch.equal(record["tensors"]["mlp.output"], torch.full_like(x, 9))
