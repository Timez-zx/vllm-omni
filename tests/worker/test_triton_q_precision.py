"""CPU contract tests, NOT GPU kernel/numerical parity certification."""

import ast
import hashlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VLLM = Path(importlib.util.find_spec("vllm").origin).parent
PATCH_SOURCE = ROOT / "vllm_omni/patch.py"
patch_module = ast.parse(PATCH_SOURCE.read_text())
hook = next(
    node
    for node in patch_module.body
    if isinstance(node, ast.FunctionDef) and node.name == "_patch_triton_query_quantization"
)
namespace = {"_PATCH_LOGGER": logging.getLogger("vllm_omni.patch")}
exec(compile(ast.Module(body=[hook], type_ignores=[]), str(PATCH_SOURCE), "exec"), namespace)
candidate = SimpleNamespace(**namespace)


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(config=None, reads=0, original_calls=0)

    def get_config():
        state.reads += 1
        return state.config

    class Impl:
        def __init__(self, supported=True, kv_cache_dtype="fp8"):
            state.original_calls += 1
            self.supports_quant_query_input = supported
            self.kv_cache_dtype = kv_cache_dtype

    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config_or_none = get_config
    backend_module = ModuleType("vllm.v1.attention.backends.triton_attn")
    backend_module.TritonAttentionImpl = Impl
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.triton_attn", backend_module)
    candidate._patch_triton_query_quantization()
    state.Impl = Impl
    return state


@pytest.mark.parametrize("additional", [None, {}, {"unrelated": True}, {"triton_disable_q_quantization": False}])
@pytest.mark.parametrize("supported", [False, True])
def test_default_preserves_original(runtime, additional, supported):
    runtime.config = SimpleNamespace(additional_config=additional)
    obj = runtime.Impl(supported)
    assert obj.supports_quant_query_input is supported
    assert obj.kv_cache_dtype == "fp8"
    assert runtime.original_calls == runtime.reads == 1


def test_explicit_mode_is_per_instance_and_read_only_once(runtime):
    runtime.config = SimpleNamespace(additional_config={"triton_disable_q_quantization": True})
    disabled = runtime.Impl()
    assert disabled.supports_quant_query_input is False
    assert disabled.kv_cache_dtype == "fp8"
    runtime.config.additional_config["triton_disable_q_quantization"] = False
    default = runtime.Impl()
    assert default.supports_quant_query_input is True
    assert disabled.supports_quant_query_input is False
    assert runtime.original_calls == runtime.reads == 2


@pytest.mark.parametrize("bad", ["true", "false", 0, 1, None, [], {}])
def test_non_boolean_config_fails_before_original_init(runtime, bad):
    runtime.config = SimpleNamespace(additional_config={"triton_disable_q_quantization": bad})
    with pytest.raises(ValueError, match="must be a boolean"):
        runtime.Impl()
    assert runtime.original_calls == 0


def test_idempotent_install_and_preserved_signature(runtime):
    before = runtime.Impl.__init__
    candidate._patch_triton_query_quantization()
    assert runtime.Impl.__init__ is before
    assert "kv_cache_dtype" in inspect.signature(before).parameters
    runtime.Impl()
    assert runtime.original_calls == runtime.reads == 1


def load_method(source, class_name, method_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def test_real_attention_constructor_omits_q_quant_but_keeps_fp8_storage(runtime):
    source = (VLLM / "model_executor/layers/attention/attention.py").read_text()
    init = load_method(source, "Attention", "__init__")
    # Execute the unmodified upstream query_quant construction statements,
    # with a CPU constructor spy instead of CUDA QuantFP8.
    start = next(
        i
        for i, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "query_quant" for t in node.targets)
    )
    statements = init.body[start : start + 2]
    assert isinstance(statements[1], ast.If)
    created = []
    for disabled in (False, True):
        runtime.config = SimpleNamespace(additional_config={"triton_disable_q_quantization": disabled})
        impl = runtime.Impl()
        attention = SimpleNamespace(impl=impl, kv_cache_dtype="fp8", head_size=128, num_heads=32, num_kv_heads=8)
        env = {
            "self": attention,
            "QuantFP8": lambda **kw: created.append(kw) or object(),
            "GroupShape": SimpleNamespace(PER_TENSOR="per_tensor"),
        }
        exec(
            compile(
                ast.Module(body=statements, type_ignores=[]),
                str(VLLM / "model_executor/layers/attention/attention.py"),
                "exec",
            ),
            env,
        )
        assert (attention.query_quant is None) is disabled
        assert attention.kv_cache_dtype == "fp8"
    assert len(created) == 1


def test_real_vllm_compute_hash_contains_additional_config():
    source = (VLLM / "config/vllm.py").read_text()
    method = load_method(source, "VllmConfig", "compute_hash")
    # Run upstream compute_hash itself, with non-target config fields as None.
    env = {"Any": object, "json": json, "safe_hash": hashlib.sha256}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<upstream compute_hash>", "exec"), env)
    names = {
        n.attr
        for n in ast.walk(method)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    }
    owner = SimpleNamespace(**{name: None for name in names})
    owner.observability_config = SimpleNamespace(compute_hash=lambda: "same")
    results = []
    for value in (False, True):
        owner.additional_config = {"triton_disable_q_quantization": value}
        results.append(env["compute_hash"](owner))
    assert results[0] != results[1]


def test_real_kernel_existing_dtype_routing():
    source = (VLLM / "v1/attention/ops/triton_unified_attention.py").read_text()
    tree = ast.parse(source)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cast_kv_tile")
    func.decorator_list = []

    # CPU symbolic execution of the exact dtype/dequantization helper. The
    # full Triton kernel and dot products still require the later GPU test.
    class DType:
        def __init__(self, dtype):
            self.dtype = dtype

        def is_fp8(self):
            return self.dtype == torch.float8_e4m3fn

    class Tile:
        def __init__(self, tensor):
            self.tensor, self.dtype = tensor, DType(tensor.dtype)

        def to(self, dtype):
            return Tile(self.tensor.to(dtype.dtype if isinstance(dtype, DType) else dtype))

        def __mul__(self, scale):
            return Tile(self.tensor * scale)

    env = {"tl": SimpleNamespace(constexpr=object, float32=torch.float32, load=lambda value: value)}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<upstream _cast_kv_tile>", "exec"), env)
    stored = torch.tensor([0.3, -0.4, 2.5]).to(torch.float8_e4m3fn)
    before = stored.view(torch.uint8).clone()
    for q_dtype in (torch.bfloat16, torch.float16):
        result = env["_cast_kv_tile"](Tile(stored), Tile(torch.empty(0, dtype=q_dtype)), 0.5, 1)
        assert result.tensor.dtype == q_dtype
        torch.testing.assert_close(result.tensor, (stored.float() * 0.5).to(q_dtype), rtol=0, atol=0)
        torch.testing.assert_close(stored.view(torch.uint8), before, rtol=0, atol=0)
    quantized = env["_cast_kv_tile"](Tile(stored), Tile(torch.empty(0, dtype=torch.float8_e4m3fn)), 0.5, 1)
    assert quantized.tensor.dtype == torch.float8_e4m3fn
    # The existing attention probability operand follows loaded V's dtype.
    assert "acc += tl.dot(P.to(V.dtype), V)" in source
    assert "K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)" in source
    assert "V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)" in source
