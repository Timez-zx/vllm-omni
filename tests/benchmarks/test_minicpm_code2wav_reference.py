"""CPU tests of the diagnostic harness, not real-weight parity evidence."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.minicpmo.check_code2wav_reference import (
    RandomTape,
    compare_states,
    compatible_streams,
    load_payloads,
    smoke_chunks,
    tensor_difference,
)


def test_smoke_has_distinct_users_and_two_clean_turns():
    streams = smoke_chunks(chunks=8)
    assert compatible_streams(streams)
    assert streams[0][0].codes != streams[1][0].codes
    assert [chunk.chunk_seq for chunk in streams[0]].count(0) == 2
    assert sum(chunk.last_chunk for chunk in streams[0]) == 2
    assert streams[0][1].codes[:3] == streams[0][0].codes[-3:]


def test_random_tape_replays_per_user_noise_without_disabling_it():
    tapes, values = [], []
    for seed in (2, 5):
        tape = RandomTape()
        torch.manual_seed(seed)
        with tape.record():
            phase = torch.rand(1, 4)
            noise = torch.randn_like(torch.zeros(1, 9, 4))
        tapes.append(tape)
        values.append((phase, noise))
    with RandomTape.replay(tapes):
        phase = torch.rand(2, 4)
        noise = torch.randn_like(torch.zeros(2, 9, 4))
    assert torch.equal(phase, torch.cat([row[0] for row in values]))
    assert torch.equal(noise, torch.cat([row[1] for row in values]))
    assert not torch.equal(noise[0], noise[1])


def test_random_tape_rejects_missing_or_reordered_draw():
    tape = RandomTape()
    with tape.record():
        torch.rand(1, 4)
    with pytest.raises(ValueError, match="skipped"):
        with RandomTape.replay([tape]):
            pass
    with pytest.raises(ValueError, match="order"):
        with RandomTape.replay([tape]):
            torch.randn_like(torch.zeros(1, 4))


def test_tensor_comparison_reports_shape_nonfinite_and_numerical_error():
    assert tensor_difference(torch.ones(3), torch.ones(3), atol=0, rtol=0)["passed"]
    assert tensor_difference(torch.ones(2), torch.ones(3), atol=0, rtol=0)["reason"] == "shape_mismatch"
    assert tensor_difference(torch.tensor([float("nan")]), torch.ones(1), atol=1, rtol=1)["reason"] == "nonfinite"
    diff = tensor_difference(torch.tensor([2.0]), torch.tensor([1.0]), atol=0, rtol=0)
    assert not diff["passed"] and diff["max_abs"] == 1.0


def test_compare_cache_ignores_only_official_unused_timestep_storage():
    actual = SimpleNamespace(flow_cache={"estimator_cnn_cache": torch.ones(10, 2)}, hift_cache={})
    reference = SimpleNamespace(
        flow_cache={"estimator_cnn_cache": torch.cat([torch.ones(10, 2), torch.full((6, 2), 999.0)])},
        hift_cache={},
    )
    result = compare_states(actual, reference, atol=0, rtol=0, n_timesteps=10)
    assert result["flow_cache.estimator_cnn_cache"]["passed"]
    reference.flow_cache["estimator_cnn_cache"][9, 0] = 2
    assert not compare_states(actual, reference, atol=0, rtol=0, n_timesteps=10)["flow_cache.estimator_cnn_cache"][
        "passed"
    ]


def test_payload_parser_preserves_full_codes_and_strips_control_placeholder(tmp_path):
    path = tmp_path / "speech-1.jsonl"
    rows = [
        {
            "event": "codec_payload",
            "request_id": "a",
            "cache_epoch": 0,
            "chunk_seq": 0,
            "codes": [4218, 4218, 4218, 12, 13, 14, 15],
            "last_chunk": False,
        },
        {
            "event": "codec_payload",
            "request_id": "a",
            "cache_epoch": 0,
            "chunk_seq": 1,
            "codes": [0],
            "code_flat_numel": 0,
            "last_chunk": True,
        },
        {
            "event": "codec_payload",
            "request_id": "a",
            "cache_epoch": 1,
            "chunk_seq": 0,
            "codes": [4218, 4218, 4218, 16],
            "last_chunk": True,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    stream = load_payloads(path)[0]
    assert stream[0].codes == (4218, 4218, 4218, 12, 13, 14, 15)
    assert stream[1].codes == ()
    assert stream[2].cache_epoch == 1


def test_payload_parser_rejects_partial_capture(tmp_path):
    path = tmp_path / "speech-1.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "codec_payload",
                "request_id": "a",
                "cache_epoch": 3,
                "chunk_seq": 9,
                "codes": [1, 2, 3, 4],
                "last_chunk": True,
            }
        )
    )
    with pytest.raises(ValueError, match="clean cache boundary"):
        load_payloads(path)
