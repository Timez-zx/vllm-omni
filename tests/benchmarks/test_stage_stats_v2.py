from pathlib import Path

from benchmarks.live_agent.analysis.stage_stats_v2 import derive, parse


def test_parse_accepts_dynamic_stats_callsite(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(
        "\n".join(
            [
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] "
                "[StageRequestStats [request_id=video-test]]",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] "
                "| num_tokens_in | 100 | 0 | 0 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] "
                "| num_tokens_out | 4 | 8 | 0 |",
                "(API pid=1) INFO 08-21 18:00:00 [stats.py:999] "
                "| vllm_ttft_ms | 10 | 30 | 35 |",
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
