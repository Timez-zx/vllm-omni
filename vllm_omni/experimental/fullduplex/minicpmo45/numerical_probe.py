# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, row-labelled MiniCPM KV correctness capture.

Diagnostic-only: CPU copies synchronize CUDA. The launcher must force eager
execution and reject capacity claims. No activation when the probe is disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def capture_layer0_io(model, positions, *, rows):
    """Bounded eager-only probe; hooks never change/replace module outputs.

    Main numerical selector must select the row, and this more expensive
    probe additionally restricts itself to seq1-3. Normal execution installs
    no hooks. Copies intentionally synchronize and invalidate capacity timing.
    """
    directory = getattr(model, "_minicpmo45_numerical_probe_dir", None)
    chosen = selected_rows(rows) if directory and os.environ.get("MINICPMO45_NUMERICAL_PROBE_LAYER_IO") == "1" else []
    chosen = [row for row in chosen if 1 <= row["seq"] <= 3]
    if not chosen:
        yield
        return
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Layer I/O probe requires real current attention metadata")
    meta = next(iter(metadata.values()))
    starts = meta.query_start_loc.detach().cpu().long()
    positions_cpu = positions.reshape(-1).detach().cpu().long()
    ordinal = getattr(model, "_minicpmo45_layer_io_counter", 0)
    model._minicpmo45_layer_io_counter = ordinal + 1
    modules = dict(model.thinker.llm.model.named_modules())
    targets = {
        "qkv_proj": "layers.0.self_attn.qkv_proj",
        "attn": "layers.0.self_attn.attn",
        "o_proj": "layers.0.self_attn.o_proj",
        "mlp": "layers.0.mlp",
        "gate_up_proj": "layers.0.mlp.gate_up_proj",
        "down_proj": "layers.0.mlp.down_proj",
    }
    if any(path not in modules for path in targets.values()):
        raise ValueError("Layer I/O probe cannot find the actual layer0 modules")
    fingerprint_path = getattr(model, "_minicpmo45_layer_io_fingerprint_path", None)
    if fingerprint_path is None:
        fingerprints = {}
        for label in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            module = modules[targets[label]]
            fingerprints[label] = {"quant_method": type(getattr(module, "quant_method", None)).__name__}
            for field in ("weight", "weight_scale", "weight_scale_inv", "input_scale"):
                value = getattr(module, field, None)
                if isinstance(value, torch.Tensor):
                    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
                    fingerprints[label][field] = {
                        "dtype": str(value.dtype),
                        "shape": tuple(value.shape),
                        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                    }
        root = Path(directory) / "layer-io"
        root.mkdir(parents=True, exist_ok=True)
        stage = "d" if model._minicpmo_pd_decode else "p"
        fingerprint_path = str(root / f"weights-{stage}-{os.getpid()}.json")
        Path(fingerprint_path).write_text(json.dumps({"schema_version": 1, "stage": stage, "layers": fingerprints}))
        model._minicpmo45_layer_io_fingerprint_path = fingerprint_path
    impl = getattr(modules[targets["attn"]], "impl", None)
    qkv_method = getattr(modules[targets["qkv_proj"]], "quant_method", None)
    records = []
    for identity in chosen:
        row = identity["row_index"]
        start, end = int(starts[row]), int(starts[row + 1])
        if not 0 <= start < end <= int(meta.num_actual_tokens):
            raise ValueError("Layer I/O probe row outside current token batch")
        records.append(
            {
                "schema_version": 1,
                "capture_kind": "layer0_io",
                "identity": dict(identity),
                "stage": "d" if model._minicpmo_pd_decode else "p",
                "counter": ordinal,
                "positions": positions_cpu[start:end],
                "query_token_span": (start, end),
                "capture_begin_ns": time.monotonic_ns(),
                "layer_io_sequence_policy": "intersection of selected rows and seq1-3",
                "attention_impl": type(impl).__name__,
                "weight_fingerprint_file": fingerprint_path,
                "supports_quant_query_input": getattr(impl, "supports_quant_query_input", None),
                "Q_quantization_disabled": getattr(impl, "_omni_triton_q_quantization_disabled", None),
                "qkv_quant_method": type(qkv_method).__name__,
                "tensors": {},
            }
        )

    def save_tensor(label, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise ValueError(f"Layer I/O probe expected token-leading tensor for {label}")
        for record in records:
            start, end = record["query_token_span"]
            if end > tensor.shape[0] or label in record["tensors"]:
                raise ValueError(f"Layer I/O probe ambiguous token layout/repeated call: {label}")
            record["tensors"][label] = tensor[start:end].detach().to(device="cpu", copy=True)

    def pre(label):
        def hook(_module, args, kwargs):
            if label == "attn":
                values = args[:3] if len(args) >= 3 else tuple(kwargs.get(k) for k in ("query", "key", "value"))
                for suffix, value in zip(("Q", "K", "V"), values):
                    save_tensor(f"attn.{suffix}", value)
            else:
                value = args[0] if args else kwargs.get("x", kwargs.get("hidden_states", kwargs.get("input_")))
                save_tensor(f"{label}.input", value)

        return hook

    def post(label):
        def hook(_module, _args, _kwargs, output):
            save_tensor(f"{label}.output", output[0] if isinstance(output, tuple) else output)

        return hook

    handles = []
    try:
        for label, path in targets.items():
            handles.append(modules[path].register_forward_pre_hook(pre(label), with_kwargs=True))
            handles.append(modules[path].register_forward_hook(post(label), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
    required = {f"{name}.{suffix}" for name in targets for suffix in ("input", "output") if name != "attn"}
    required.update(("attn.Q", "attn.K", "attn.V", "attn.output"))
    root = Path(directory) / "layer-io"
    root.mkdir(parents=True, exist_ok=True)
    for record in records:
        if set(record["tensors"]) != required:
            raise ValueError("Layer I/O probe did not observe every requested module")
        record["capture_end_ns"] = time.monotonic_ns()
        torch.save(
            record, root / f"{record['stage']}-{os.getpid()}-{ordinal:07d}-row{record['identity']['row_index']}.pt"
        )


def parse_seq_selection(value):
    if not value:
        return None
    selected = set()
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            start = end = int(bounds[0])
        elif len(bounds) == 2:
            start, end = map(int, bounds)
        else:
            raise ValueError("Expected sequences such as 37-39,239-241")
        if not 0 <= start <= end or end - start > 1000:
            raise ValueError("Invalid or unbounded numerical-probe sequence selection")
        selected.update(range(start, end + 1))
    return frozenset(selected)


def snapshot_runner_rows(runner):
    """Snapshot current forward ownership; never read previous sample() rows.

    Called only by the opt-in diagnostic branch in _build_model_kwargs_extra,
    after the current batch and intermediate payloads have been prepared.
    No model state is changed and no tensor data is copied here.
    """
    result = []
    for row_index, request_id in enumerate(runner.input_batch.req_ids):
        buffers = getattr(runner, "model_intermediate_buffer", {})
        info = buffers.get(request_id) if isinstance(buffers, dict) else None
        if not isinstance(info, dict):
            request = getattr(runner, "requests", {}).get(request_id)
            info = getattr(request, "additional_information_cpu", None)
        info = info if isinstance(info, dict) else {}
        duplex = info.get("duplex", {})
        if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
            result.append(None)  # Dummy/non-native rows are never mislabelled.
            continue
        result.append(
            {
                "row_index": row_index,
                "request_id": str(request_id),
                "session_id": duplex.get("session_id"),
                "incarnation": duplex.get("incarnation"),
                "epoch": duplex.get("epoch"),
                "seq": duplex.get("seq"),
            }
        )
    return result


def selected_rows(rows, environ=None):
    environ = os.environ if environ is None else environ
    seqs = parse_seq_selection(environ.get("MINICPMO45_NUMERICAL_PROBE_SEQS"))
    sessions = json.loads(environ.get("MINICPMO45_NUMERICAL_PROBE_SESSIONS", "null"))
    if sessions is not None and (
        not isinstance(sessions, list) or any(not isinstance(value, str) for value in sessions)
    ):
        raise ValueError("NUMERICAL_PROBE_SESSIONS must be a JSON string array")
    selected = []
    for row in rows or []:
        if row is None:
            continue
        if not isinstance(row.get("session_id"), str) or not row["session_id"]:
            raise ValueError("Missing native numerical-probe session identity")
        if not isinstance(row.get("request_id"), str) or not row["request_id"]:
            raise ValueError("Missing native numerical-probe request identity")
        for key in ("incarnation", "epoch", "seq", "row_index"):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(f"Missing/invalid numerical-probe {key}")
        if seqs is not None and row["seq"] not in seqs:
            continue
        if sessions is not None and row["session_id"] not in sessions:
            continue
        selected.append(row)
    return selected


def gap_blocks(*, history, window, pin, block_size):
    if history < 0 or block_size <= 0 or pin < 0 or pin % block_size:
        raise ValueError("Invalid pinned-window mapping arguments")
    if not pin:
        # Ordinary attention's logical table was not compacted.
        return 0
    if window <= 0:
        raise ValueError("Pinned attention must specify a positive window")
    return max(0, max(history - window + 1, 0) // block_size - pin // block_size)


def map_absolute_slots(table_row, positions, *, history, window, pin, block_size):
    """Map original token positions through the ACTUAL compact attention view.

    history is the first query's original absolute position, never the
    compacted metadata.seq_lens. Middle-gap tokens and null blocks are rejected.
    Block offsets remain absolute-position modulo block_size, including partial
    tail blocks. The mapping includes allocated current-query positions.
    """
    positions = torch.as_tensor(positions, dtype=torch.long, device=table_row.device)
    if positions.numel() and bool((positions < 0).any()):
        raise ValueError("Negative KV position")
    gap = gap_blocks(history=history, window=window, pin=pin, block_size=block_size)
    blocks = positions // block_size
    prefix_blocks = pin // block_size
    in_gap = (blocks >= prefix_blocks) & (blocks < prefix_blocks + gap)
    if bool(in_gap.any()):
        raise ValueError("Cannot capture evicted middle-gap KV")
    columns = blocks - torch.where(blocks >= prefix_blocks, gap, 0)
    if columns.numel() and int(columns.max()) >= table_row.numel():
        raise ValueError("Probe compact block-table index out of range")
    physical_blocks = table_row[columns].long()
    if bool((physical_blocks <= 0).any()):
        raise ValueError("Probe selected the null/invalid KV block")
    return physical_blocks * block_size + positions % block_size


def prefix_positions(*, history, window, pin, block_size, span=256):
    """Sample pinned head, recent tail, and first retained partial-block edge.

    This excludes keys not visible to the first query. It is a bounded sample,
    not an assertion that all resident KV has been compared. A matching P→D
    check must intersect absolute positions (their visible windows may differ).
    """
    if span < 1:
        raise ValueError("Prefix sample span must be positive")
    if history <= 0:
        return torch.empty(0, dtype=torch.long)
    head = list(range(min(pin if pin else 128, history)))
    tail_start = max(pin, history - window + 1) if pin else 0
    boundary_end = min(history, tail_start + 2 * block_size)
    tail = list(range(max(tail_start, history - span), history))
    return torch.tensor(sorted(set(head + list(range(tail_start, boundary_end)) + tail)), dtype=torch.long)


def raw_kv(cache, selected_slots):
    """Preserve storage bytes without FP8→FP32 round-trips or requantization."""
    if cache.ndim != 4:
        raise ValueError("Probe requires Triton [block, head, token, 2*head_dim] KV")
    slots = selected_slots.to(cache.device)
    block_size = cache.shape[2]
    values = cache[slots // block_size, :, slots % block_size, :].contiguous()
    return {
        "storage_dtype": str(values.dtype),
        "shape": tuple(values.shape),
        "raw_bytes": values.view(torch.uint8).detach().cpu(),
    }


def p_next_boundary_positions(*, history, query_length, window, pin, block_size):
    """Old P KV needed at predicted next D first-query window boundary.

    Current query KV is captured separately. Use last_query_position + 1,
    not this P chunk's first query, to predict D's first query. For chunked P,
    sample this small edge on each forward, so the last chunk covers D's edge.
    """
    next_history = history + query_length
    boundary = max(pin, next_history - window + 1) if pin else 0
    end = min(history, boundary + 2 * block_size)
    return torch.arange(boundary, end, dtype=torch.long) if boundary < end else torch.empty(0, dtype=torch.long)


def capture_forward(model, input_ids, positions, inputs_embeds, hidden, *, rows, phase="after_backbone_forward"):
    """Immutable selected-row snapshots; diagnostic only, no added KV waits.

    D before: prefix only, once per request/seq, before model forward reads KV.
    P after: newly computed KV and bounded existing prefix samples.
    D after remains useful for output numerics, NOT proof of what D read.
    CPU copies themselves synchronize and can perturb a ready race. Correlate
    actual connector ready timestamps; never use these captures as capacity.
    """
    if phase not in {"before_backbone_forward", "after_backbone_forward"}:
        raise ValueError("Unknown numerical probe capture phase")
    before = phase == "before_backbone_forward"
    if before and not model._minicpmo_pd_decode:
        return
    directory = getattr(model, "_minicpmo45_numerical_probe_dir", None)
    chosen = selected_rows(rows) if directory else []
    if not chosen:
        return
    seen_attr = "_minicpmo45_numerical_probe_pre_seen" if before else "_minicpmo45_numerical_probe_seen"
    seen = getattr(model, seen_attr, None)
    if seen is None:
        seen = set()
        setattr(model, seen_attr, seen)
    if before:
        chosen = [
            r
            for r in chosen
            if tuple(r[k] for k in ("request_id", "session_id", "incarnation", "epoch", "seq")) not in seen
        ]
        if not chosen:
            return
    capture_begin_ns = time.monotonic_ns()
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return
    meta = next(iter(metadata.values()))
    count = int(meta.num_actual_tokens)
    starts = meta.query_start_loc.detach().cpu().long()
    positions_cpu = positions.reshape(-1)[:count].detach().cpu().long()
    ordinal = getattr(model, "_minicpmo45_numerical_probe_counter", 0)
    model._minicpmo45_numerical_probe_counter = ordinal + 1
    stage = "d" if model._minicpmo_pd_decode else "p"
    layers_option = os.environ.get("MINICPMO45_NUMERICAL_PROBE_LAYERS", "all")
    layers = (
        None if layers_option == "all"
        else frozenset() if layers_option == "none"
        else parse_seq_selection(layers_option)
    )
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for identity in chosen:
        row = identity["row_index"]
        if row + 1 >= starts.numel():
            raise ValueError("Probe identity row outside current attention batch")
        start, end = int(starts[row]), int(starts[row + 1])
        if not 0 <= start < end <= count:
            raise ValueError("Probe identity/query-start mismatch")
        pos = positions_cpu[start:end]
        history = int(pos[0])
        if not torch.equal(pos, torch.arange(history, history + end - start)):
            raise ValueError("Numerical probe expects contiguous absolute positions per request")
        pin = int(meta.rswa_prefix_lens[row]) if getattr(meta, "rswa_prefix_lens", None) is not None else 0
        window = int(getattr(meta, "rswa_window", 0) or 0)
        local_key = tuple(identity[k] for k in ("request_id", "session_id", "incarnation", "epoch", "seq"))
        first = local_key not in seen
        seen.add(local_key)
        record = {
            "schema_version": 2,
            "stage": stage,
            "counter": ordinal,
            "capture_phase": phase,
            "capture_begin_ns": capture_begin_ns,
            "identity": dict(identity),
            "first_forward_for_request_seq": first,
            "input_capture": "prefix_only_before_backbone" if before else "before_forward_clone",
            "positions": pos,
            "query_token_span": (start, end),
            "query_start_loc": starts,
            "absolute_seq_len": history + end - start,
            "compact_seq_len": int(meta.seq_lens[row]),
            "pin_tokens": pin,
            "window_tokens": window,
            "kv": {},
            "prefix_kv": {},
            "kv_layer_selection": layers_option,
            "capture_scope": (
                "D before: first prefix only; after: query KV + first prefix + P predicted next-D boundary"
            ),
        }
        if not before:
            record.update(
                input_ids=input_ids.reshape(-1)[start:end].detach().cpu(),
                inputs_embeds=inputs_embeds.reshape(-1, inputs_embeds.shape[-1])[start:end].detach().cpu(),
                hidden=hidden.reshape(-1, hidden.shape[-1])[start:end].detach().cpu(),
            )
        layer_meta_checked = set()
        for name, module in model.thinker.llm.model.named_modules():
            if not name.endswith(".self_attn.attn"):
                continue
            layer_index = int(name.split(".")[1])
            if layers is not None and layer_index not in layers:
                continue
            layer_meta = metadata.get(getattr(module, "layer_name", ""), meta)
            cache = module.kv_cache
            block_size = cache.shape[2]
            table = layer_meta.block_table[row]
            slots = layer_meta.slot_mapping.reshape(-1)[start:end].long()
            arguments = dict(history=history, window=window, pin=pin, block_size=block_size)
            mapped = map_absolute_slots(table, pos, **arguments)
            if not torch.equal(mapped.to(slots.device), slots):
                raise ValueError("Probe mapping disagrees with actual attention query slot_mapping")
            gap = gap_blocks(**arguments)
            if record["compact_seq_len"] != record["absolute_seq_len"] - gap * block_size:
                raise ValueError("Probe mapping disagrees with actual compact seq length")
            if id(layer_meta) not in layer_meta_checked:
                record.setdefault("mapping_witnesses", []).append(
                    {
                        "layer": name,
                        "block_size": block_size,
                        "gap_blocks": gap,
                        "compact_block_table": table.detach().cpu(),
                        "query_slots": slots.detach().cpu(),
                    }
                )
                layer_meta_checked.add(id(layer_meta))
            scale = {
                "logical_kv_dtype": str(module.kv_cache_dtype),
                "k_scale": float(module._k_scale.item()),
                "v_scale": float(module._v_scale.item()),
            }
            if not before:
                record["kv"][name] = {"positions": pos, **raw_kv(cache, slots), **scale}
            prefix = prefix_positions(**arguments) if first else torch.empty(0, dtype=torch.long)
            if stage == "p":
                edge = p_next_boundary_positions(**arguments, query_length=end - start)
                prefix = torch.cat((prefix, edge)).unique(sorted=True)
            if prefix.numel():
                prefix_slots = map_absolute_slots(table, prefix, **arguments)
                record["prefix_kv"][name] = {"positions": prefix, **raw_kv(cache, prefix_slots), **scale}
        record["capture_end_ns"] = time.monotonic_ns()
        torch.save(record, root / f"{stage}-{os.getpid()}-{ordinal:07d}-row{row}.pt")
