import ast
import json
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.engine.kv_transfer_probe import (
    WIRE_KEY,
    TransferProbe,
    cache_sync_request_identity,
    registration_identity,
)


def request(seq=38, session="s0"):
    return SimpleNamespace(request_id=f"decode-{session}-{seq}", model_intermediate_buffer={
        "duplex": {"data_plane": True, "session_id": session,
                   "incarnation": 3, "epoch": 4, "seq": seq}})


ENV = {"MINICPMO45_NUMERICAL_PROBE_DIR": "/diagnostic-only",
       "MINICPMO45_NUMERICAL_PROBE_SEQS": "37-39,239-241"}


def registration(seq=38, indices=None):
    req = request(seq)
    return {
        WIRE_KEY: registration_identity(req, layer_names=[["layer.0", "layer.1"]], environ=ENV),
        "request_id": req.request_id,
        "local_block_ids": ([21, 22],), "source_block_offset": 100,
        "decode_block_size": 16, "remote_prompt_tokens": 1630,
        "matched_prefix_tokens": 1600,
        **({"source_block_indices": indices} if indices is not None else {}),
    }


def probe():
    rows = []
    return TransferProbe("unused", sink=rows.append), rows


def test_disabled_does_not_inspect_request():
    assert registration_identity(object(), layer_names=[], environ={}) is None


@pytest.mark.parametrize("seq,expected", [(0, False), (36, False), (37, True),
                                          (39, True), (40, False), (240, True), (242, False)])
def test_selection(seq, expected):
    assert bool(registration_identity(request(seq), layer_names=[[]], environ=ENV)) == expected


def test_explicit_session_and_no_request_id_guess():
    env = {**ENV, "MINICPMO45_NUMERICAL_PROBE_SESSIONS": '["s1"]'}
    assert registration_identity(request(), layer_names=[[]], environ=env) is None
    row = registration_identity(request(session="s1"), layer_names=[[]], environ=env)
    assert row["session_id"] == "s1" and row["seq"] == 38
    bad = request()
    bad.model_intermediate_buffer["duplex"].pop("seq")
    with pytest.raises(ValueError, match="seq"):
        registration_identity(bad, layer_names=[[]], environ=ENV)


@pytest.mark.parametrize("buffer", [None, {}, {"meta": {"unrelated": True}}])
def test_actual_serialized_carrier_fallback(buffer):
    from vllm_omni.engine.serialization import serialize_additional_information

    req = request()
    req.additional_information = serialize_additional_information(req.model_intermediate_buffer)
    req.model_intermediate_buffer = buffer
    identity = registration_identity(req, layer_names=[[]], environ=ENV)
    assert identity["session_id"] == "s0" and identity["seq"] == 38


def test_flattened_and_singleton_cpu_metadata():
    req = request()
    req.model_intermediate_buffer = {"duplex." + key: [value]
                                     for key, value in req.model_intermediate_buffer["duplex"].items()}
    identity = registration_identity(req, layer_names=[[]], environ=ENV)
    assert identity["session_id"] == "s0" and identity["seq"] == 38


def test_missing_native_identity_is_loud_and_observed():
    req = SimpleNamespace(request_id="duplex-unknown", model_intermediate_buffer={},
                          additional_information={"global_request_id": ["s0"]})
    rows = []
    obj = TransferProbe("unused", sink=rows.append, role="scheduler", environ=ENV)
    with pytest.raises(ValueError, match="no usable identity"):
        obj.registration(req, layer_names=[[]])
    assert rows[0]["event"] == "kv_probe_initialized"
    assert rows[-1]["decision"] == "missing_native_identity"
    assert [carrier["raw_type"] for carrier in rows[-1]["carriers"]] == ["dict", "dict"]


def test_selector_reason_is_bounded_but_selected_always_recorded():
    rows = []
    obj = TransferProbe("unused", sink=rows.append, role="scheduler", environ=ENV)
    for seq in range(20):
        obj.registration(request(seq), layer_names=[[]])
    assert len(rows) == 9  # One init plus first8 unselected observations.
    assert rows[-1]["decision"] == "seq_not_selected"
    identity = obj.registration(request(38), layer_names=[[]])
    assert rows[-1]["decision"] == "selected"
    assert rows[-1]["identity"]["seq"] == identity["seq"] == 38


def test_probe_uses_initialization_environment_not_import_constant(monkeypatch):
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "38")
    rows = []
    obj = TransferProbe("unused", sink=rows.append, role="worker")
    monkeypatch.setenv("MINICPMO45_NUMERICAL_PROBE_SEQS", "99")
    assert obj.registration(request(38), layer_names=[[]])["seq"] == 38


def physical_id(session="s0", seq=38, incarnation=3, epoch=4):
    from vllm_omni.experimental.fullduplex.engine.contracts import duplex_resource_request_id
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

    return duplex_resource_request_id(DuplexFence(session_id=session, incarnation=incarnation, epoch=epoch),
                                      "stage0") + f"-{seq:08x}"


@pytest.mark.parametrize("session", ["s0", "用户.a/b+🤖"])
def test_canonical_cache_sync_identity_roundtrip(session):
    identity = cache_sync_request_identity(physical_id(session=session))
    assert identity == {"session_id": session, "incarnation": 3, "epoch": 4, "seq": 38}


@pytest.mark.parametrize("change", [
    lambda value: value.replace(".i.3.", ".i.03."),
    lambda value: value.replace(".e.4.", ".e.-4."),
    lambda value: value.replace(".r.stage0-", ".r.stage1-"),
    lambda value: value[:-8] + "0000002A",
    lambda value: value + "0",
    lambda value: value.replace("czA.i", "czA=.i"),
    lambda value: value.replace("czA.i", "czB.i"),  # Same decoded bytes, nonzero pad bits.
    lambda value: value.replace("czA.i", "_w.i"),  # Invalid UTF8 byte0xff.
])
def test_noncanonical_cache_sync_identity_rejected(change):
    with pytest.raises(ValueError):
        cache_sync_request_identity(change(physical_id()))


@pytest.mark.parametrize("field,value", [("seq", 39), ("epoch", 5), ("incarnation", 4), ("session_id", "other")])
def test_real_metadata_cross_checks_physical_identity(field, value):
    req = request()
    req.request_id = physical_id()
    req.model_intermediate_buffer["duplex"][field] = value
    with pytest.raises(ValueError, match="disagrees"):
        registration_identity(req, layer_names=[[]], environ=ENV)


def test_actual_orchestrator_early_request_without_media_metadata():
    from vllm import SamplingParams

    from vllm_omni.engine.orchestrator import Orchestrator

    orchestrator = object.__new__(Orchestrator)
    orchestrator._pd_pair = (0, 1)
    orchestrator._pd_prefill_remote = {"remote_engine_id": "p", "remote_host": "127.0.0.1",
                                       "remote_port": 1234, "tp_size": 1}
    orchestrator.stage_pools = {1: SimpleNamespace(stage_vllm_config=SimpleNamespace(model_config=None))}
    orchestrator._build_pd_mrope_features = lambda metadata: None
    state = SimpleNamespace(
        prompt={"cache_salt": "native-session-a", "model_intermediate_buffer": request().model_intermediate_buffer},
        sampling_params_list=[None, SamplingParams(max_tokens=1)], pd_mrope_feature_metadata=None,
    )
    # Use the actual production builder and actual OmniEngineCoreRequest, not
    # a mocked request that helpfully carries metadata absent from production.
    early = orchestrator._build_pd_early_cache_request(
        physical_id().rsplit("-", 1)[0], state,
        engine_request_id=physical_id(), prompt_token_ids=[0] * 16,
    )
    assert early.model_intermediate_buffer is None
    assert early.additional_information is None
    assert early.cache_salt == "native-session-a" and early.pd_cache_sync_retain
    identity = registration_identity(early, layer_names=[["layer.0"]], environ=ENV)
    assert identity["identity_source"] == "request_id_for_cache_sync"
    assert identity["session_id"] == "s0" and identity["seq"] == 38
    # The subsequent real D request carries metadata; it must agree exactly.
    early.model_intermediate_buffer = state.prompt["model_intermediate_buffer"]
    formal = registration_identity(early, layer_names=[["layer.0"]], environ=ENV)
    assert formal["identity_source"] == "model_intermediate_buffer"
    for key in ("session_id", "incarnation", "epoch", "seq", "d_request_id"):
        assert identity[key] == formal[key]


def test_selected_intent_is_not_success():
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    assert rows[0]["position_ranges_by_group"] == [[[1600, 1616], [1616, 1630]]]
    assert rows[0]["source_allocator_block_ids"] == [[11, 12]]
    assert rows[0]["destination_allocator_block_ids"] == [[21, 22]]
    assert rows[0]["complete"] is False and "success" not in rows[0]


def test_window_holes_use_actual_indices_not_suffix_offset():
    obj, rows = probe()
    obj.begin("prefill", registration(indices=[[0, 101]]), ([11, 12],), block_size=16)
    assert rows[0]["source_logical_block_indices"] == [[0, 101]]
    assert rows[0]["position_ranges_by_group"] == [[[0, 16], [1616, 1630]]]


@pytest.mark.parametrize("poll_before_seal", [False, True])
def test_actual_completion_and_writer_core_race(poll_before_seal):
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.submitted("prefill", 51)
    obj.submitted("prefill", 52)
    assert not any(row.get("complete") for row in rows)
    if poll_before_seal:
        obj.completed({"prefill"})
        assert not any(row.get("complete") for row in rows)
    obj.seal("prefill")
    if not poll_before_seal:
        assert not any(row.get("complete") for row in rows)
        obj.completed({"prefill"})
    assert rows[-1]["event"] == "kv_write_completed"
    assert rows[-1]["success"] and rows[-1]["handle_count"] == 2
    assert not obj._states


def test_actual_nixl_opaque_handle_contract_and_lifetime():
    # The installed Python NIXL wrapper can be constructed with a CPU-only
    # fake agent. This is the actual handle class returned in production,
    # including its non-int API and automatic lifetime release behavior.
    import gc

    from nixl import nixl_xfer_handle

    released = []
    agent = SimpleNamespace(releaseXferReq=released.append)
    handle = nixl_xfer_handle(agent, 71)
    with pytest.raises(TypeError):
        int(handle)
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.submitted("prefill", handle)
    obj.submitted("prefill", handle)  # A duplicate observation is not a second handle.
    submitted = [row for row in rows if row["event"] == "kv_write_submitted"]
    assert submitted[0]["probe_handle_id"] == submitted[1]["probe_handle_id"]
    assert submitted[0]["handle_identity_source"] == "probe_local_object"
    assert submitted[0]["handle_type"].endswith(".nixl_xfer_handle")
    assert "handle" not in submitted[0]  # No fictional native integer ID.
    del handle
    gc.collect()
    assert released == []  # Probe owns a strong reference, never releases NIXL.
    obj.seal("prefill")
    obj.completed({"prefill"})
    gc.collect()
    assert released == [71]  # Last reference expires only after completion.
    assert rows[-1]["success"] and rows[-1]["handle_count"] == 1
    json.dumps(rows)  # The opaque object itself never enters the JSON record.


def test_probe_handle_identity_never_coerces_or_compares_object():
    class Opaque:
        def __int__(self):
            raise AssertionError("not an integer")

        def __eq__(self, other):
            raise AssertionError("only object identity is allowed")

    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.submitted("prefill", Opaque())
    obj.submitted("prefill", Opaque())
    obj.seal("prefill")
    obj.completed({"prefill"})
    submitted = [row for row in rows if row["event"] == "kv_write_submitted"]
    assert len({row["probe_handle_id"] for row in submitted}) == 2
    assert rows[-1]["success"] and rows[-1]["handle_count"] == 2


@pytest.mark.parametrize("failure", ["failed_state", "missing_submission"])
def test_partial_failure_cannot_be_success(failure):
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.submitted("prefill", 51)
    if failure == "failed_state":
        obj.failed("prefill")
    else:
        obj.submitted("prefill", None)
    obj.seal("prefill")
    obj.completed({"prefill"})
    assert rows[-1]["complete"] is True and rows[-1]["success"] is False


def test_handshake_failure_no_false_submission():
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.seal("prefill")
    obj.completed({"prefill"})  # E.g. eventual lease cleanup cannot bless it.
    assert rows[-1]["event"] == "kv_write_not_submitted"
    assert not rows[-1]["complete"] and not rows[-1]["success"]


def test_full_hit_has_no_write():
    obj, rows = probe()
    reg = registration()
    reg["local_block_ids"] = ()
    obj.begin("prefill", reg, (), block_size=16)
    obj.seal("prefill", no_write=True)
    assert rows[-1]["event"] == "kv_write_not_needed" and rows[-1]["no_write"]
    assert rows[-1]["source_logical_block_indices"] == []


def test_missing_identity_is_unobservable_not_fabricated():
    obj, rows = probe()
    reg = registration()
    del reg[WIRE_KEY]
    assert not obj.begin("prefill", reg, ([11, 12],), block_size=16)
    obj.submitted("prefill", 123)
    obj.seal("prefill")
    obj.completed({"prefill"})
    assert rows == []


def test_two_sessions_do_not_share_state():
    obj, rows = probe()
    for index in (0, 1):
        reg = registration()
        reg[WIRE_KEY]["session_id"] = f"s{index}"
        obj.begin(f"prefill-{index}", reg, ([11, 12],), block_size=16)
        obj.submitted(f"prefill-{index}", 50 + index)
        obj.seal(f"prefill-{index}")
    obj.failed("prefill-0")
    obj.completed({"prefill-0", "prefill-1"})
    completed = {row["session_id"]: row["success"] for row in rows if row["event"] == "kv_write_completed"}
    assert completed == {"s0": False, "s1": True}


def test_jsonl_and_identity_mismatch(tmp_path):
    obj = TransferProbe(tmp_path)
    reg = registration()
    reg[WIRE_KEY]["d_request_id"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        obj.begin("prefill", reg, ([11, 12],), block_size=16)
    reg = registration()
    obj.begin("prefill", reg, ([11, 12],), block_size=16)
    saved = [json.loads(line) for line in next(tmp_path.glob("kv-transfer-*.jsonl")).read_text().splitlines()]
    assert saved[0]["epoch"] == 4 and saved[0]["seq"] == 38


def test_actual_upstream_failed_handle_done_set_is_not_success():
    # Execute the installed vLLM completion routine, not a guessed contract.
    path = Path(sysconfig.get_paths()["purelib"]) / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_pop_done_transfers")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    obj, rows = probe()
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    obj.submitted("prefill", 51)
    obj.seal("prefill")
    worker = SimpleNamespace(
        nixl_wrapper=SimpleNamespace(check_xfer_state=lambda handle: "ERR"),
        _log_failure=lambda **kwargs: None,
        _handle_failed_transfer=lambda req, handle: obj.failed(req),
    )
    transfers = {"prefill": [51]}
    done = namespace["_pop_done_transfers"](worker, transfers)
    assert done == {"prefill"}  # Upstream includes failed requests here.
    obj.completed(done)
    assert rows[-1]["complete"] and not rows[-1]["success"]


@pytest.mark.parametrize("failure", [False, True])
def test_d_ready_is_separate_from_p_poll(failure):
    obj, rows = probe()
    reg = registration()
    obj.register_receiver(reg)
    if failure:
        obj.failed(reg["request_id"])
    obj.receive_ready({reg["request_id"]})
    assert rows[-1]["event"] == "kv_receive_ready"
    assert rows[-1]["success"] is not failure
    assert rows[-1]["write_id"] == reg[WIRE_KEY]["write_id"]
    assert rows[-1]["monotonic_ns"] >= rows[0]["monotonic_ns"]
    assert not any(row["event"] == "kv_write_completed" for row in rows)


def candidate_source():
    path = Path(__file__).resolve().parents[2] / "vllm_omni/engine/nixl_delta_push_connector.py"
    return path.read_text()


def test_connector_has_receive_ready_hook():
    assert "probe.receive_ready(done_recving)" in candidate_source()


@pytest.mark.parametrize("final_state,expected", [("DONE", True), ("ERR", False)])
def test_real_completion_method_through_candidate_hooks(final_state, expected):
    from nixl import nixl_xfer_handle

    handle = nixl_xfer_handle(SimpleNamespace(releaseXferReq=lambda value: None), 71)
    upstream = Path(sysconfig.get_paths()["purelib"]) / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"
    actual = next(node for node in ast.walk(ast.parse(upstream.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "_pop_done_transfers")
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[actual], type_ignores=[])), "actual", "exec"), ns)
    class Base:
        _pop_done_transfers = ns["_pop_done_transfers"]

        def _xfer_blocks(self, *, request_id):
            return handle

        def _handle_failed_transfer(self, req_id, handle):
            self.failures.append((req_id, handle))

    tree = ast.parse(candidate_source())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "NixlDeltaPushConnectorWorker")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [method for method in cls.body if isinstance(method, ast.FunctionDef)
                and method.name in ("_xfer_blocks", "_handle_failed_transfer", "_pop_done_transfers")]
    mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {"Base": Base}
    exec(compile(ast.fix_missing_locations(mod), "hooks", "exec"), namespace)
    worker = namespace[cls.name]()
    obj, rows = probe()
    worker._numerical_transfer_probe = obj
    worker.failures = []
    state = {handle: "PROC"}
    worker.nixl_wrapper = SimpleNamespace(check_xfer_state=lambda h: state[h],
        get_xfer_telemetry=lambda h: None, release_xfer_handle=lambda h: None)
    worker.xfer_stats = SimpleNamespace(record_transfer=lambda item: None)
    worker._log_failure = lambda **kwargs: None
    obj.begin("prefill", registration(), ([11, 12],), block_size=16)
    handle = worker._xfer_blocks(request_id="prefill")
    obj.seal("prefill")
    worker._sending_transfers = {"prefill": [handle]}
    assert worker._pop_done_transfers(worker._sending_transfers) == set()
    assert not any(row.get("complete") for row in rows)
    state[handle] = final_state
    assert worker._pop_done_transfers(worker._sending_transfers) == {"prefill"}
    assert rows[-1]["success"] is expected
    assert bool(worker.failures) is not expected
