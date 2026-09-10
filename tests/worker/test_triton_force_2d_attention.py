"""CPU execution of the installed metadata path; not whole-model parity tests."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VLLM = Path(importlib.util.find_spec("vllm").origin).parent
PATCH = ROOT / "vllm_omni/patch.py"
BACKEND = VLLM / "v1/attention/backends/triton_attn.py"
PINNED = ROOT / "vllm_omni/engine/pinned_prefix_window.py"


def execute(nodes, source, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)


def named_node(source, name):
    return next(node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) == name)


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(init_calls=0)

    class Base:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            state.init_calls += 1
            self.kv_cache_spec, self.vllm_config, self.device = kv_cache_spec, vllm_config, device

    namespace = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "ClassVar": ClassVar,
        "AttentionMetadataBuilder": Base,
        "AttentionCGSupport": SimpleNamespace(ALWAYS="always"),
        "CUDAGraphMode": SimpleNamespace(FULL_AND_PIECEWISE="mixed", FULL_DECODE_ONLY="decode", FULL="full"),
        "MIN_LAUNCH_GRID_SIZE_2D": 128,
        "NUM_PAR_SOFTMAX_SEGMENTS": 16,
        "next_power_of_2": lambda value: 1 << (value - 1).bit_length(),
        "get_num_attention_heads_from_layers": lambda config, layers: 32,
    }
    execute(
        [named_node(BACKEND, "TritonAttentionMetadata"), named_node(BACKEND, "TritonAttentionMetadataBuilder")],
        BACKEND,
        namespace,
    )
    backend = ModuleType("vllm.v1.attention.backends.triton_attn")
    backend.TritonAttentionMetadataBuilder = namespace["TritonAttentionMetadataBuilder"]
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    hook_namespace = {"_PATCH_LOGGER": logging.getLogger("test.force_2d")}
    execute([named_node(PATCH, "_patch_triton_attention_execution_path")], PATCH, hook_namespace)
    state.install = hook_namespace["_patch_triton_attention_execution_path"]
    state.install()
    state.Builder = backend.TritonAttentionMetadataBuilder
    state.backend, state.namespace = backend, namespace
    return state


def config(additional=None, *, graph=False):
    return SimpleNamespace(
        additional_config=additional,
        compilation_config=SimpleNamespace(
            cudagraph_mode="full" if graph else "none", cudagraph_capture_sizes=[1, 8, 16]
        ),
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel: 8, get_head_size=lambda: 128, rswa_window=None, max_model_len=50000
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )


def spec(*, pinned=False):
    return SimpleNamespace(block_size=16, pinned_prefix_tokens=128 if pinned else 0, sliding_window=36000)


def common(*, history=279, queries=1):
    lengths = torch.tensor([history + queries], dtype=torch.int32)
    starts = torch.tensor([0, queries], dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=queries,
        max_query_len=queries,
        max_seq_len=history + queries,
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        block_table_tensor=torch.arange(4000, dtype=torch.int32)[None],
        slot_mapping=torch.arange(history, history + queries, dtype=torch.int64),
        causal=True,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )


@pytest.mark.parametrize("additional", [None, {}, {"other": True}, {"triton_force_2d_attention": False}])
def test_default_preserves_installed_builder_and_buffers(runtime, additional):
    builder = runtime.Builder(spec(), [], config(additional), torch.device("cpu"))
    result = builder.build(0, common())
    assert result.seq_threshold_3D == builder.seq_threshold_3D == 16
    assert result.softmax_segm_output is builder.softmax_segm_output
    assert result.softmax_segm_output.shape == (16, 32, 16, 128)
    assert result.num_par_softmax_segments == 16


@pytest.mark.parametrize("graph", [False, True])
def test_split_k_threshold_resizes_owned_buffers_before_capture(runtime, graph):
    cfg = config({"triton_decode_split_k_threshold": 32}, graph=graph)
    cfg.scheduler_config.max_num_seqs = 64
    cfg.compilation_config.cudagraph_capture_sizes.append(32)
    builder = runtime.Builder(spec(), [], cfg, torch.device("cpu"))
    result = builder.build(0, common())
    assert result.seq_threshold_3D == 32
    assert result.softmax_segm_output.shape == (32, 32, 16, 128)
    assert result.softmax_segm_max.shape == result.softmax_segm_expsum.shape == (32, 32, 16)
    for name in ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"):
        assert getattr(result, name) is getattr(builder, name)
    again = builder.build(0, common(queries=8))
    assert again.max_query_len == 8  # Upstream still excludes prefills from split-K.
    assert again.softmax_segm_output is result.softmax_segm_output


@pytest.mark.parametrize("value", [0, -1, 65, True, 1.5, "32"])
def test_invalid_split_k_threshold(runtime, value):
    cfg = config({"triton_decode_split_k_threshold": value})
    cfg.scheduler_config.max_num_seqs = 64
    with pytest.raises(ValueError, match="integer"):
        runtime.Builder(spec(), [], cfg, torch.device("cpu"))
    assert runtime.init_calls == 0


def test_split_k_threshold_rejects_conflicting_or_uncaptured_path(runtime):
    cfg = config({"triton_decode_split_k_threshold": 32, "triton_force_2d_attention": True})
    cfg.scheduler_config.max_num_seqs = 64
    with pytest.raises(ValueError, match="conflicts"):
        runtime.Builder(spec(), [], cfg, torch.device("cpu"))
    cfg.additional_config["triton_force_2d_attention"] = False
    cfg.compilation_config.cudagraph_mode = "full"
    with pytest.raises(ValueError, match="capture size"):
        runtime.Builder(spec(), [], cfg, torch.device("cpu"))


@pytest.mark.parametrize("bad", [None, 0, 1, "true", "false", [], {}])
def test_strict_boolean_rejected_before_upstream_allocation(runtime, bad):
    with pytest.raises(ValueError, match="triton_force_2d_attention must be a boolean"):
        runtime.Builder(spec(), [], config({"triton_force_2d_attention": bad}), torch.device("cpu"))
    assert runtime.init_calls == 0


def test_per_instance_config_does_not_change_quantization_or_other_builders(runtime):
    cfg = config({"triton_force_2d_attention": True, "triton_disable_q_quantization": True})
    enabled = runtime.Builder(spec(), [], cfg, torch.device("cpu"))
    cfg.additional_config["triton_force_2d_attention"] = False
    default = runtime.Builder(spec(), [], cfg, torch.device("cpu"))
    inputs = common()
    result = enabled.build(0, inputs)
    assert result.seq_threshold_3D == 0
    assert enabled.seq_threshold_3D == default.build(0, inputs).seq_threshold_3D == 16
    assert result.block_table is inputs.block_table_tensor
    assert result.slot_mapping is inputs.slot_mapping
    assert result.seq_lens is inputs.seq_lens
    assert result.softmax_segm_output is enabled.softmax_segm_output
    assert cfg.additional_config == {"triton_force_2d_attention": False, "triton_disable_q_quantization": True}


def test_idempotent_patch_preserves_signature(runtime):
    init, build = runtime.Builder.__init__, runtime.Builder.build
    runtime.install()
    assert runtime.Builder.__init__ is init and runtime.Builder.build is build
    assert "vllm_config" in inspect.signature(init).parameters
    assert "fast_build" in inspect.signature(build).parameters


def test_does_not_mutate_underlying_shared_metadata(runtime):
    shared = SimpleNamespace(seq_threshold_3D=16, tensor=torch.tensor([1]))

    class SharedBuilder:
        def __init__(self, *args):
            pass

        def build(self):
            return shared

    runtime.backend.TritonAttentionMetadataBuilder = SharedBuilder
    runtime.install()
    enabled = SharedBuilder(spec(), [], config({"triton_force_2d_attention": True}), torch.device("cpu"))
    result = enabled.build()
    assert result is not shared and shared.seq_threshold_3D == 16
    assert result.seq_threshold_3D == 0 and result.tensor is shared.tensor


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("history", [279, 38000])
def test_real_pinned_subclass_order_preserves_compaction_and_mask(runtime, enabled, history):
    # Execute the unchanged nested production subclass against the patched
    # installed base, exactly as install_pinned_prefix_window binds it.
    namespace = runtime.namespace
    namespace.update(copy=copy, cdiv=lambda a, b: (a + b - 1) // b)
    nested = next(
        node
        for node in named_node(PINNED, "install_pinned_prefix_window").body
        if isinstance(node, ast.ClassDef) and node.name == "PinnedPrefixMetadataBuilder"
    )
    execute([named_node(PINNED, "compact_window_view"), nested], PINNED, namespace)
    builder = namespace["PinnedPrefixMetadataBuilder"](
        spec(pinned=True), [], config({"triton_force_2d_attention": enabled}), torch.device("cpu")
    )
    inputs = common(history=history)
    before_table, before_lengths = inputs.block_table_tensor.clone(), inputs.seq_lens.clone()
    result = builder.build(0, inputs)
    gap = max(0, max(history - 36000 + 1, 0) // 16 - 8)
    assert result.seq_threshold_3D == (0 if enabled else 16)
    assert result.rswa_window == 36000 and result.rswa_prefix_lens.tolist() == [128]
    assert result.seq_lens.tolist() == [history + 1 - gap * 16]
    assert result.block_table[0, :8].tolist() == list(range(8))
    assert result.block_table[0, 8].item() == 8 + gap
    assert torch.equal(inputs.block_table_tensor, before_table)
    assert torch.equal(inputs.seq_lens, before_lengths)
    assert result.slot_mapping is inputs.slot_mapping


def test_upstream_cuda_graph_capture_routes_through_configured_build(runtime):
    builder = runtime.Builder(spec(), [], config({"triton_force_2d_attention": True}, graph=True), torch.device("cpu"))
    result = builder.build_for_cudagraph_capture(common())
    assert result.seq_threshold_3D == 0
    assert result.softmax_segm_output.shape[0] == builder.seq_threshold_3D == 16


def test_installed_kernel_condition_selects_2d_with_threshold_zero():
    path = VLLM / "v1/attention/ops/triton_unified_attention.py"
    method = named_node(path, "unified_attention")
    choice = next(
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "use_3d" for target in node.targets)
    )
    for threshold, expected in ((16, True), (0, False)):
        namespace = dict(
            seq_threshold_3D=threshold,
            num_par_softmax_segments=16,
            softmax_segm_output=object(),
            softmax_segm_max=object(),
            softmax_segm_expsum=object(),
            max_seqlen_q=1,
            num_seqs=1,
            is_batch_invariant=False,
        )
        execute([choice], path, namespace)
        assert namespace["use_3d"] is expected


def test_actual_compile_hash_separates_attention_only_setting():
    path = VLLM / "config/vllm.py"
    cls = named_node(path, "VllmConfig")
    method = next(node for node in cls.body if getattr(node, "name", None) == "compute_hash")
    namespace = {"Any": object, "json": json, "safe_hash": hashlib.sha256}
    execute([method], path, namespace)
    names = {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    owner = SimpleNamespace(**dict.fromkeys(names))
    owner.observability_config = SimpleNamespace(compute_hash=lambda: "same")
    results = []
    for value in (False, True):
        owner.additional_config = {"triton_force_2d_attention": value}
        results.append(namespace["compute_hash"](owner))
    assert results[0] != results[1]
