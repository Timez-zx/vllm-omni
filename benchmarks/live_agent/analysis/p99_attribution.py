#!/usr/bin/env python3
"""Where does p99 TTFA increase as continuous-AV sessions scale?

Input: cell directories from run_av_session_ladder.sh (turns.jsonl per cell, with the
absolute per-turn stamps t_q/t_ft/t_fa/t_done that mu_bench.py records).

Four decompositions, each aimed at one hypothesis:

  1. STAGE SPLIT -- per turn, thinker side = ttft (query -> first text) and
     speech side = ttfa - ttft (first text -> first audio). For the turns in
     the tail (>= p95 / p99 of the cell), the excess over the cell median is
     split within each turn and only then averaged (never median-minus-median;
     the estimator rule from stage_stats_v2.py).

  2. ARRIVAL STATE -- at each turn's t_q, count OTHER turns that were mid-TTFA
     (t_q' <= t_q < t_fa': the engine owes them a first sound) and mid-DELIVERY
     (t_fa' <= t_q < t_done': the speech stages are streaming for them). TTFA
     bucketed by those counts separates "queued behind others" from
     "everything got slower". Valid because every user shares one driver
     process and therefore one monotonic clock.

  3. TURN INDEX -- canonical history grows between compactions, so a
     context-cost story predicts late turns slower than early ones at the
     same user count. Buckets of 10.

  4. HYGIENE -- timeouts, history compactions, fatal probes, and self-repaired
     warnings per cell (from summary.json), so a pathological cell cannot
     masquerade as a scaling law without confusing a recorded warning with a
     user-visible capacity boundary.

    p99_attribution.py --cells '/tmp/vllm-omni-results/avsession_*_u*'
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

DEFAULT_GPU_STAGES = {0: "thinker", 1: "talker", 2: "code2wav"}
PD_GPU_STAGES = {
    0: "thinker-prefill",
    1: "thinker-decode",
    2: "talker",
    3: "code2wav",
}
RESOURCE_METRICS = (
    "gpu_busy_pct",
    "sm_active_pct",
    "sm_occupancy_pct",
    "tensor_active_pct",
    "dram_active_pct",
    "fp16_active_pct",
    "fp32_active_pct",
    "pcie_rx_mib_s",
    "pcie_tx_mib_s",
    "power_w",
    "memory_used_mib",
)


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_cell(d: pathlib.Path) -> dict:
    recs = []
    with (d / "turns.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    gpu_meta = {}
    gpu_samples = []
    gpu_path = d / "gpu_samples.jsonl"
    if gpu_path.exists():
        with gpu_path.open() as fh:
            for line in fh:
                item = json.loads(line)
                if item.get("k") == "meta":
                    gpu_meta = item
                elif item.get("k") == "sample":
                    gpu_samples.append(item)
    m = re.search(r"_u(\d+)$", d.name)
    return {
        "dir": d,
        "users": int(m.group(1)) if m else summary.get("users", 0),
        "recs": recs,
        "summary": summary,
        "gpu_meta": gpu_meta,
        "gpu_samples": gpu_samples,
    }


def in_intervals(timestamp: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def resource_summary(samples: list[dict], *, total_memory_mib: float | None, power_limit_w: float | None) -> dict:
    output: dict = {"n": len(samples)}
    for metric in RESOURCE_METRICS:
        values = [sample[metric] for sample in samples if sample.get(metric) is not None]
        if values:
            output[metric] = {
                "p50": round(pct(values, 50), 3),
                "p95": round(pct(values, 95), 3),
                "max": round(max(values), 3),
            }
    memory = output.get("memory_used_mib")
    if memory and total_memory_mib:
        output["memory_used_pct_p95"] = round(100 * memory["p95"] / total_memory_mib, 3)
    power = output.get("power_w")
    if power and power_limit_w:
        output["power_limit_pct_p95"] = round(100 * power["p95"] / power_limit_w, 3)
    return output


def resource_attribution(
    cell: dict,
    ok: list[dict],
    ttfa_p95: float,
    playback_p95: float,
    stall_p95: float,
) -> dict:
    samples = cell["gpu_samples"]
    if not samples or not ok:
        return {}
    benchmark_start = min(record["t_q"] for record in ok)
    benchmark_end = max(record.get("t_done") or record["t_q"] for record in ok)
    ttfa_intervals = [
        (record["t_q"], record["t_fa"])
        for record in ok
        if record["ttfa_ms"] >= ttfa_p95 and record.get("t_fa") is not None
    ]
    stall_intervals = [
        (record["t_fa"], record["t_done"])
        for record in ok
        if record.get("stall_max_ms", 0) >= stall_p95
        and record.get("stall_max_ms", 0) > 0
        and record.get("t_fa") is not None
        and record.get("t_done") is not None
    ]
    playback_intervals = [
        (record["t_q"], record["t_q"] + record["playback_start_ms"] / 1000.0)
        for record in ok
        if record.get("playback_start_ms") is not None and record["playback_start_ms"] >= playback_p95
    ]
    device_meta = {device["index"]: device for device in cell["gpu_meta"].get("devices", [])}
    deploy_name = pathlib.Path(str(cell["summary"].get("deploy_config") or "")).name
    gpu_stages = PD_GPU_STAGES if deploy_name == "pd_deploy_4gpu.yaml" else DEFAULT_GPU_STAGES
    output = {}
    for gpu in sorted({sample["gpu"] for sample in samples}):
        meta = device_meta.get(gpu, {})
        gpu_samples = [
            sample
            for sample in samples
            if sample["gpu"] == gpu and benchmark_start <= sample["monotonic_s"] <= benchmark_end
        ]
        output[str(gpu)] = {
            "stage": gpu_stages.get(gpu, f"gpu{gpu}"),
            "overall": resource_summary(
                gpu_samples,
                total_memory_mib=meta.get("total_memory_mib"),
                power_limit_w=meta.get("power_limit_w"),
            ),
            "ttfa_tail95": resource_summary(
                [sample for sample in gpu_samples if in_intervals(sample["monotonic_s"], ttfa_intervals)],
                total_memory_mib=meta.get("total_memory_mib"),
                power_limit_w=meta.get("power_limit_w"),
            ),
            "playback_tail95": resource_summary(
                [sample for sample in gpu_samples if in_intervals(sample["monotonic_s"], playback_intervals)],
                total_memory_mib=meta.get("total_memory_mib"),
                power_limit_w=meta.get("power_limit_w"),
            ),
            "stall_tail95": resource_summary(
                [sample for sample in gpu_samples if in_intervals(sample["monotonic_s"], stall_intervals)],
                total_memory_mib=meta.get("total_memory_mib"),
                power_limit_w=meta.get("power_limit_w"),
            ),
        }
    return output


def arrival_state(recs: list[dict]) -> None:
    """Annotate each ok/timeout record with the system state at its arrival."""
    stamped = [r for r in recs if r.get("t_q") is not None]
    for r in stamped:
        n_wait = n_stream = 0
        for o in stamped:
            if o is r or o["user"] == r["user"]:
                continue
            tq, tfa, tdone = o["t_q"], o.get("t_fa"), o.get("t_done")
            # a timed-out turn with no first audio occupied the engine until
            # its timeout; treat its whole window as mid-TTFA
            if tq <= r["t_q"] and (tfa is None or r["t_q"] < tfa):
                if o.get("status") == "ok" or tfa is None:
                    n_wait += 1
            elif tfa is not None and tfa <= r["t_q"] and (tdone is None or r["t_q"] < tdone):
                n_stream += 1
        r["n_wait"] = n_wait
        r["n_stream"] = n_stream


def cell_report(cell: dict) -> dict:
    warmup_turns = int(cell["summary"].get("warmup_turns") or 0)
    ok = [
        r
        for r in cell["recs"]
        if r.get("status") == "ok"
        and r.get("turn", 0) > warmup_turns
        and r.get("ttfa_ms") is not None
        and r.get("ttft_ms") is not None
    ]
    for r in ok:
        r["speech_ms"] = r["ttfa_ms"] - r["ttft_ms"]
    ttfa = [r["ttfa_ms"] for r in ok]
    playback = [r["playback_start_ms"] for r in ok if r.get("playback_start_ms") is not None]
    buffer_wait = [r["playback_start_ms"] - r["ttfa_ms"] for r in ok if r.get("playback_start_ms") is not None]
    stalls = [r.get("stall_max_ms", 0.0) for r in ok]
    out = {
        "users": cell["users"],
        "n_ok": len(ok),
        "n_timeout": cell["summary"].get("n_timeout"),
        "compactions": (cell["summary"].get("engine_probes") or {}).get("history_compactions"),
        "ttfa_p50": pct(ttfa, 50),
        "ttfa_p95": pct(ttfa, 95),
        "ttfa_p99": pct(ttfa, 99),
        "ttfa_max": max(ttfa) if ttfa else float("nan"),
        "playback_p50": pct(playback, 50),
        "playback_p95": pct(playback, 95),
        "playback_p99": pct(playback, 99),
        "playback_max": max(playback) if playback else float("nan"),
        "buffer_wait_p50": pct(buffer_wait, 50),
        "buffer_wait_p99": pct(buffer_wait, 99),
        "ttft_p50": pct([r["ttft_ms"] for r in ok], 50),
        "ttft_p99": pct([r["ttft_ms"] for r in ok], 99),
        "speech_p50": pct([r["speech_ms"] for r in ok], 50),
        "speech_p99": pct([r["speech_ms"] for r in ok], 99),
        "stall_p95": pct(stalls, 95),
        "stall_p99": pct(stalls, 99),
    }

    # 1. stage split of the tail's excess, within-turn then averaged
    med_ttfa, med_ttft = out["ttfa_p50"], out["ttft_p50"]
    med_speech = out["speech_p50"]
    for tail_name, thresh in (("p95", out["ttfa_p95"]), ("p99", out["ttfa_p99"])):
        tail = [r for r in ok if r["ttfa_ms"] >= thresh]
        if not tail:
            continue
        exc_tot = st.mean([r["ttfa_ms"] - med_ttfa for r in tail])
        exc_thk = st.mean([r["ttft_ms"] - med_ttft for r in tail])
        exc_spc = st.mean([r["speech_ms"] - med_speech for r in tail])
        component_excess = exc_thk + exc_spc
        out[f"tail_{tail_name}"] = {
            "n": len(tail),
            "excess_ms": round(exc_tot, 1),
            "thinker_share": round(exc_thk / component_excess, 3) if component_excess else None,
            "speech_share": round(exc_spc / component_excess, 3) if component_excess else None,
        }

    med_playback = out["playback_p50"]
    med_buffer_wait = out["buffer_wait_p50"]
    for tail_name, thresh in (("p95", out["playback_p95"]), ("p99", out["playback_p99"])):
        tail = [r for r in ok if r.get("playback_start_ms") is not None and r["playback_start_ms"] >= thresh]
        if not tail:
            continue
        exc_total = st.mean([r["playback_start_ms"] - med_playback for r in tail])
        exc_service = st.mean([r["ttfa_ms"] - med_ttfa for r in tail])
        exc_buffer = st.mean([r["playback_start_ms"] - r["ttfa_ms"] - med_buffer_wait for r in tail])
        component_excess = exc_service + exc_buffer
        out[f"playback_tail_{tail_name}"] = {
            "n": len(tail),
            "excess_ms": round(exc_total, 1),
            "service_ttfa_share": round(exc_service / component_excess, 3) if component_excess else None,
            "post_first_audio_share": round(exc_buffer / component_excess, 3) if component_excess else None,
        }

    # 2. arrival state
    arrival_state(cell["recs"])
    by_wait: dict[int, list[float]] = {}
    for r in ok:
        by_wait.setdefault(r["n_wait"], []).append(r["ttfa_ms"])
    out["ttfa_by_n_wait"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)} for k, v in sorted(by_wait.items())
    }
    by_stream: dict[int, list[float]] = {}
    for r in ok:
        if r["n_wait"] == 0:  # isolate the streaming effect from the queueing one
            by_stream.setdefault(min(r["n_stream"], 8), []).append(r["ttfa_ms"])
    out["ttfa_by_n_stream_at_wait0"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)} for k, v in sorted(by_stream.items())
    }
    tail95 = [r for r in ok if r["ttfa_ms"] >= out["ttfa_p95"]]
    out["n_wait_mean_all"] = round(st.mean([r["n_wait"] for r in ok]), 2) if ok else None
    out["n_wait_mean_tail95"] = round(st.mean([r["n_wait"] for r in tail95]), 2) if tail95 else None
    out["n_stream_mean_all"] = round(st.mean([r["n_stream"] for r in ok]), 2) if ok else None
    out["n_stream_mean_tail95"] = round(st.mean([r["n_stream"] for r in tail95]), 2) if tail95 else None

    # 3. turn-index buckets (context growth inside the long session)
    buckets: dict[str, list[float]] = {}
    for r in ok:
        b = f"{((r['turn'] - 1) // 10) * 10 + 1}-{((r['turn'] - 1) // 10) * 10 + 10}"
        buckets.setdefault(b, []).append(r["ttfa_ms"])
    out["ttfa_by_turn_bucket"] = {
        k: {"n": len(v), "p50": round(pct(v, 50), 1), "p99": round(pct(v, 99), 1)}
        for k, v in sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }

    by_mic: dict[str, list[float]] = {}
    for r in ok:
        by_mic.setdefault(r.get("mic_condition") or "unspecified", []).append(r["ttfa_ms"])
    out["ttfa_by_mic_condition"] = {
        key: {"n": len(values), "p50": round(pct(values, 50), 1), "p99": round(pct(values, 99), 1)}
        for key, values in sorted(by_mic.items())
    }

    # 4. hygiene
    probes = cell["summary"].get("engine_probes", {})
    out["bad_probes"] = {
        k: v
        for k, v in probes.items()
        if v
        and k
        in (
            "unowned_audio",
            "torch_cat_error",
            "zero_output_wedge",
            "negative_slice",
            "preempted_reqs",
        )
    }
    out["warning_probes"] = {key: value for key, value in probes.items() if value and key == "counter_leak_clamped"}
    out["resources"] = resource_attribution(
        cell,
        ok,
        out["ttfa_p95"],
        out["playback_p95"],
        out["stall_p95"],
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="glob of cell directories")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cells = [
        load_cell(pathlib.Path(p)) for p in sorted(glob.glob(args.cells)) if (pathlib.Path(p) / "turns.jsonl").exists()
    ]
    cells.sort(key=lambda c: c["users"])
    reports = [cell_report(c) for c in cells]

    hdr = (
        f"{'users':>5} {'n':>4} {'p50':>6} {'p95':>7} {'p99':>7} "
        f"{'thk p50/p99':>12} {'spc p50/p99':>12} "
        f"{'tail95 thk/spc':>14} {'wait all/tail':>13} {'to':>3} {'cmp':>4}"
    )
    print(hdr)
    for r in reports:
        t95 = r.get("tail_p95", {})
        share = (
            (f"{t95.get('thinker_share', float('nan')):.2f}/{t95.get('speech_share', float('nan')):.2f}")
            if t95
            else "-"
        )
        print(
            f"{r['users']:>5} {r['n_ok']:>4} {r['ttfa_p50']:>6.0f} {r['ttfa_p95']:>7.0f} "
            f"{r['ttfa_p99']:>7.0f} "
            f"{r['ttft_p50']:>5.0f}/{r['ttft_p99']:>5.0f} "
            f"{r['speech_p50']:>5.0f}/{r['speech_p99']:>5.0f} "
            f"{share:>14} "
            f"{r['n_wait_mean_all']:>5.2f}/{r['n_wait_mean_tail95']:>5.2f} "
            f"{r['n_timeout'] or 0:>3} {r['compactions'] or 0:>4}"
        )
        if r["bad_probes"]:
            print(f"      !! bad probes: {r['bad_probes']}")
        if r["warning_probes"]:
            print(f"      -- engine warnings: {r['warning_probes']}")
        playback_tail = r.get("playback_tail_p95", {})
        print(
            f"      playback p50/p95/p99={r['playback_p50']:.0f}/{r['playback_p95']:.0f}/"
            f"{r['playback_p99']:.0f} ms; post-first-audio p50/p99="
            f"{r['buffer_wait_p50']:.0f}/{r['buffer_wait_p99']:.0f} ms; "
            f"tail95 service/post={playback_tail.get('service_ttfa_share', float('nan')):.2f}/"
            f"{playback_tail.get('post_first_audio_share', float('nan')):.2f}"
        )
        for resource in r["resources"].values():
            overall = resource["overall"]
            tail = resource["playback_tail95"]

            def pair(stats: dict, name: str) -> str:
                metric = stats.get(name)
                return f"{metric['p50']:.0f}/{metric['p95']:.0f}" if metric else "-"

            print(
                f"      {resource['stage']:<8} overall sm/occ/tensor/dram="
                f"{pair(overall, 'sm_active_pct')}/{pair(overall, 'sm_occupancy_pct')}/"
                f"{pair(overall, 'tensor_active_pct')}/{pair(overall, 'dram_active_pct')} "
                f"play-tail95={pair(tail, 'sm_active_pct')}/{pair(tail, 'sm_occupancy_pct')}/"
                f"{pair(tail, 'tensor_active_pct')}/{pair(tail, 'dram_active_pct')} "
                f"power={overall.get('power_limit_pct_p95', float('nan')):.0f}% "
                f"mem={overall.get('memory_used_pct_p95', float('nan')):.0f}%"
            )

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(reports, indent=1))
        print(f"\nfull detail -> {args.json_out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
