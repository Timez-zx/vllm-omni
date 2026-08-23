from pathlib import Path

from benchmarks.live_agent.analysis.stage_stats_v2 import derive, parse


def test_parse_accepts_dynamic_stats_callsite(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(
        "\n".join(
            [
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] [StageRequestStats [request_id=video-test]]",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | num_tokens_in | 100 | 0 | 0 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | num_tokens_out | 4 | 8 | 0 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | vllm_ttft_ms | 10 | 30 | 35 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | serving_time_to_first_output_ms | 10 | 30 | 35 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:1000] [Overall Summary]",
            ]
        )
    )

    records = parse(log)
    rows = derive(records)

    assert len(rows) == 1
    assert rows[0]["request_id"] == "video-test"
    assert rows[0]["talker_add_ms"] == 20
    assert rows[0]["code2wav_add_ms"] == 5


def test_parse_uses_per_session_request_order_across_history_compaction(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    lines = [
        "(API pid=1) INFO 08-21 18:00:00 [video.py:1] "
        "[finite-request] session=user-0 request=video-first turn=10 prompt_tokens=30000",
        "(API pid=1) INFO 08-21 18:00:01 [video.py:1] "
        "[finite-request] session=user-0 request=video-second turn=6 prompt_tokens=16000",
        "(P pid=2) INFO 08-21 18:00:02 [connector.py:1] "
        "[nixl-delta-push] request=video-second-suffix prefix_tokens=2560 "
        "source_blocks=1000 delta_blocks=840",
    ]
    # Complete in reverse order to prove completion interleaving is irrelevant.
    for request_id in ("video-second-suffix", "video-first-suffix"):
        lines.extend(
            [
                f"(API pid=1) INFO 08-21 18:00:03 [stats.py:999] [StageRequestStats [request_id={request_id}]]",
                "(API pid=1) INFO 08-21 18:00:03 [stats.py:999] | num_tokens_in | 100 | 100 |",
                "(API pid=1) INFO 08-21 18:00:03 [stats.py:999] | vllm_ttft_ms | 10 | 20 |",
                "(API pid=1) INFO 08-21 18:00:03 [stats.py:999] | serving_time_to_first_output_ms | 10 | 20 |",
                "(API pid=1) INFO 08-21 18:00:03 [stats.py:1000] [Overall Summary]",
            ]
        )
    log.write_text("\n".join(lines))

    records = {record["request_id"]: record for record in parse(log)}

    assert records["video-first-suffix"]["turn"] == 0
    assert records["video-second-suffix"]["turn"] == 1
    assert records["video-second-suffix"]["kv_prefix_tokens"] == 2560
    assert records["video-second-suffix"]["kv_delta_blocks"] == 840


def test_parse_preserves_non_contiguous_stage_ids(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(
        "\n".join(
            [
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] [StageRequestStats [request_id=video-missing-talker]]",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | Field | 0 | 1 | 3 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | num_tokens_in | 100 | 100 | 8 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | vllm_ttft_ms | 10 | 20 | 40 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] | serving_time_to_first_output_ms | 10 | 20 | 40 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:1000] [Overall Summary]",
            ]
        )
    )

    records = parse(log)
    rows = derive(records)

    assert set(records[0]["stages"]) == {0, 1, 3}
    assert rows[0]["is_pd"] is True
    assert rows[0]["prefill_ttft_ms"] == 10
    assert rows[0]["thinker_add_ms"] == 10
    assert rows[0]["talker_add_ms"] is None
    assert rows[0]["code2wav_add_ms"] is None
    assert rows[0]["post_thinker_add_ms"] == 20
