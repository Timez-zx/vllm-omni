# SPDX-License-Identifier: Apache-2.0

from benchmarks.live_agent.analysis.p99_attribution import resource_attribution


def test_resource_samples_align_with_ttfa_and_stall_tail_windows() -> None:
    records = [
        {
            "t_q": 1.0,
            "t_fa": 1.2,
            "t_done": 2.0,
            "ttfa_ms": 200.0,
            "playback_start_ms": 300.0,
            "stall_max_ms": 0.0,
        },
        {
            "t_q": 3.0,
            "t_fa": 4.0,
            "t_done": 5.0,
            "ttfa_ms": 1000.0,
            "playback_start_ms": 1500.0,
            "stall_max_ms": 60.0,
        },
    ]
    samples = []
    for gpu in range(3):
        for timestamp, sm in ((1.1, 20.0), (3.5, 90.0), (4.5, 70.0)):
            samples.append(
                {
                    "gpu": gpu,
                    "monotonic_s": timestamp,
                    "sm_active_pct": sm,
                    "sm_occupancy_pct": sm / 2,
                    "power_w": 300.0,
                    "memory_used_mib": 50.0,
                }
            )
    cell = {
        "gpu_samples": samples,
        "gpu_meta": {
            "devices": [{"index": gpu, "total_memory_mib": 100.0, "power_limit_w": 600.0} for gpu in range(3)]
        },
    }

    result = resource_attribution(cell, records, ttfa_p95=900.0, playback_p95=1400.0, stall_p95=50.0)

    assert result["0"]["stage"] == "thinker"
    assert result["1"]["stage"] == "talker"
    assert result["2"]["stage"] == "code2wav"
    assert result["0"]["overall"]["n"] == 3
    assert result["0"]["ttfa_tail95"]["sm_active_pct"]["p50"] == 90.0
    assert result["0"]["playback_tail95"]["sm_active_pct"]["p50"] == 80.0
    assert result["0"]["stall_tail95"]["sm_active_pct"]["p50"] == 70.0
    assert result["0"]["overall"]["power_limit_pct_p95"] == 50.0
    assert result["0"]["overall"]["memory_used_pct_p95"] == 50.0
