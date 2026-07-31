#!/usr/bin/env python3
"""Fine-grained resource and latency breakdown of a whole camera session.

`ttfa_decompose.py` attributes work inside the TTFA window. `contention_summary.py`
compares user counts. Neither answers "over the whole session, what fraction of
the GPU did each component consume", which is the number that decides how many
sessions fit on a card.

Components attributed here:

    vision encoder      exact -- CUDA events, never enters a CUDA graph
    audio encoder       exact -- same
    memory-note call    bracketed by the [PA_MEM] w0/w1 epoch window
    thinker LLM         RESIDUAL: stage-0 busy minus the three above
    talker (stage 1)    NVML per-PID
    code2wav (stage 2)  NVML per-PID

Three measurement facts that shape how these numbers may be read:

1. **The LLM is a residual, not a measurement.** stage 0 captures `decode, FULL`
   CUDA graphs and vLLM's `@support_torch_compile` replaces the module `__call__`
   with a compiled callable, so `register_forward_hook` never fires on it
   (`llm_backbone` records 0 calls). Everything unexplained inside stage 0 lands
   in this row, including any attribution error from the other three.

2. **Per-process SM% is an NVML sampled estimate**, and SM utilization means "at
   least one warp resident", not occupancy. So `sum(stages)` can exceed device
   busy when stages genuinely overlap. That ratio is reported as a concurrency
   indicator rather than normalised away -- forcing it to 100% would hide the
   overlap, which is the interesting part under contention.

3. **Bracketing the note call is only clean when nothing else runs on stage 0
   during it.** With one user that holds. With several, user A's note overlaps
   user B's turn in the same process, and NVML cannot separate them. The overlap
   fraction is computed and reported; above a few percent, the per-component
   split inside stage 0 is not trustworthy and the honest comparison is A/B
   across policies at the same user count (which this script also prints).

    session_breakdown.py --arms tag=label ... [--users 1,2,4]
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAMEM_W = re.compile(r"\[PA_MEM\] w0=(?P<w0>[\d.]+) w1=(?P<w1>[\d.]+) dur_ms=(?P<dur>[\d.]+)")


def load_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def load_gpu(p: pathlib.Path) -> tuple[dict, dict[str, str], list[dict]]:
    """-> (meta, pid->stage-name, samples)."""
    recs = load_jsonl(p)
    meta: dict = {}
    pid2stage: dict[str, str] = {}
    samples: list[dict] = []
    for r in recs:
        k = r.get("k")
        if k == "meta":
            meta = r
        elif k == "stage_map":
            # later remaps win; stage pids do not change mid-run in practice
            pid2stage.update(r.get("map") or {})
        elif k == "s":
            samples.append(r)
    return meta, pid2stage, samples


def device_busy_in(samples: list[dict], wall0: float, t0: float, t1: float) -> float:
    """Integrate DEVICE SM utilisation over [t0,t1) -> GPU-busy seconds.

    Safe to integrate because the device counter is present in every sample
    (100% of them, 20 ms apart at hz=50). Same left-hand integration as
    ttfa_decompose.device_busy_in, kept identical so the two analyses agree.
    """
    pts = [(wall0 + s["t"], s["sm"]) for s in samples
           if "sm" in s and t0 <= wall0 + s["t"] < t1]
    if len(pts) < 2:
        return 0.0
    return sum((b[0] - a[0]) * a[1] / 100.0 for a, b in zip(pts, pts[1:]))


def stage_shares_in(samples: list[dict], pid2stage: dict[str, str],
                    wall0: float, t0: float, t1: float) -> dict[str, float]:
    """Relative GPU share per stage in [t0,t1), from summed per-PID SM samples.

    Per-PID SM must NOT be integrated over time the way the device counter can
    be. Two reasons, both measured on this data:

      * it is present in only ~7.7% of samples, spaced ~200 ms (up to 10.5 s),
        so multiplying a sample by its gap invents busy time. Doing that
        produced a 92% duty cycle for a single user, against the 3-12% Phase 0
        measured -- an artifact, not a result.
      * the value is NVML's MAX over its internal window (p50 = 95 here), not a
        mean, so it is biased high even where the spacing is regular.

    What survives is the RATIO between stages: every stage is sampled by the same
    biased mechanism, so the proportions are meaningful even though the absolute
    values are not. This is the method the Phase 0 study used
    (`analyze_trace.per_stage_sm_share`), and it reproduces its result: talker
    80.6% / thinker 17.6% / code2wav 1.8% here versus 73.8 / 22.8 / 3.3 there.

    Absolute per-stage seconds are then device_busy x share, with the total
    coming from the dense device counter.

    Several PIDs can map to one stage (the stage process plus its engine-core
    child), so shares are accumulated by stage NAME, not by PID.
    """
    acc: dict[str, float] = {}
    for s in samples:
        w = wall0 + s["t"]
        if not (t0 <= w < t1):
            continue
        for pid, v in (s.get("proc_sm") or {}).items():
            name = pid2stage.get(pid)
            if name is None:
                continue
            acc[name] = acc.get(name, 0.0) + float(v)
    tot = sum(acc.values())
    return {k: v / tot for k, v in acc.items()} if tot else {}


def note_windows(log: pathlib.Path) -> list[tuple[float, float]]:
    if not log.exists():
        return []
    text = ANSI.sub("", log.read_text(errors="ignore"))
    return [(float(m["w0"]), float(m["w1"])) for m in PAMEM_W.finditer(text)]


def merge_windows(ws: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of possibly-overlapping windows, so time is never double counted."""
    if not ws:
        return []
    ws = sorted(ws)
    out = [list(ws[0])]
    for a, b in ws[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def turn_windows(traces: list[pathlib.Path]) -> list[tuple[float, float]]:
    """Per-turn [query, audio_done] windows across all users, for overlap checks."""
    out = []
    for t in traces:
        ev = load_jsonl(t)
        q = None
        for e in ev:
            if e.get("k") == "tx_query":
                q = e["w"]
            elif e.get("k") == "rep_end" and q is not None:
                out.append((q, e["w"]))
                q = None
    return out


def overlap_frac(ws: list[tuple[float, float]],
                 others: list[tuple[float, float]]) -> float:
    """Fraction of total window time that also lies inside `others`."""
    tot = sum(b - a for a, b in ws)
    if tot <= 0:
        return 0.0
    ov = 0.0
    others = merge_windows(others)
    for a, b in ws:
        for c, d in others:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                ov += hi - lo
    return ov / tot


def session_bounds(traces: list[pathlib.Path]) -> tuple[float, float] | None:
    lo, hi = None, None
    for t in traces:
        ev = load_jsonl(t)
        ws = [e["w"] for e in ev if e.get("k") in ("rep_begin", "rep_end")]
        if not ws:
            continue
        lo = min(ws) if lo is None else min(lo, min(ws))
        hi = max(ws) if hi is None else max(hi, max(ws))
    return (lo, hi) if lo is not None else None


def analyse(res: pathlib.Path, tag: str, outdir_name: str,
            events_name: str | None = None,
            log_name: str | None = None) -> dict | None:
    """One arm. events/log may be shared across arms of the same server run.

    When several user counts run against ONE server instance -- which is the
    right design, since it makes user count the only variable -- they share the
    stage-0 event stream and the server log. Windowing by the arm's own session
    bounds separates them, so the shared files are passed in explicitly rather
    than derived from the arm tag.
    """
    gpu_p = res / f"gpu_{tag}.jsonl"
    ev_p = res / (events_name or f"stage0_events_{tag}.jsonl")
    log_p = res / (log_name or f"server_{tag}.log")
    traces = sorted((res / outdir_name).glob("ttfa_user*.jsonl"))
    if not traces:
        return None
    bounds = session_bounds(traces)
    meta, pid2stage, samples = load_gpu(gpu_p)
    if not bounds or not samples or "wall_start" not in meta:
        return None
    t0, t1 = bounds
    wall0 = meta["wall_start"]
    wall_s = t1 - t0

    dev_busy = device_busy_in(samples, wall0, t0, t1)
    shares = stage_shares_in(samples, pid2stage, wall0, t0, t1)
    # absolute seconds = dense device total x relative per-stage share
    stage_busy = {name: dev_busy * f for name, f in shares.items()}

    # exact encoder time from CUDA events inside the session window
    enc = {"vision_encoder": 0.0, "audio_encoder": 0.0}
    enc_calls = {"vision_encoder": 0, "audio_encoder": 0}
    enc_tokens = {"vision_encoder": 0, "audio_encoder": 0}
    for e in load_jsonl(ev_p):
        m = e.get("module")
        ls = e.get("launch_start")
        if m in enc and ls is not None and t0 <= ls < t1:
            enc[m] += e.get("gpu_ms", 0.0) / 1000.0
            enc_calls[m] += 1
            enc_tokens[m] += int(e.get("ntok") or 0)

    # memory-note call: integrate the DEVICE counter inside its windows. Using
    # per-PID here would hit the same sparse/max bias; the device counter is
    # dense. With one user and nothing else running, device busy in the window IS
    # the note's GPU time; with other users active it is an upper bound, which is
    # what `note_overlap_with_turns` warns about.
    nws_raw = [(a, b) for a, b in note_windows(log_p) if t0 <= a < t1]
    nws = merge_windows(nws_raw)
    note_busy = sum(device_busy_in(samples, wall0, a, b) for a, b in nws)
    ov = overlap_frac(nws, turn_windows(traces)) if nws else 0.0

    def stage(*names: str) -> float:
        for n in names:
            if n in stage_busy:
                return stage_busy[n]
        return 0.0

    s0 = stage("stage0", "stage 0")
    s1 = stage("stage1", "stage 1")
    s2 = stage("stage2", "stage 2")
    llm_resid = s0 - enc["vision_encoder"] - enc["audio_encoder"] - note_busy

    comps = {
        "vision_encoder": enc["vision_encoder"],
        "audio_encoder": enc["audio_encoder"],
        "memory_note": note_busy,
        "thinker_llm_residual": llm_resid,
        "talker": s1,
        "code2wav": s2,
    }
    total_comp = sum(max(0.0, v) for v in comps.values())

    return {
        "tag": tag,
        "session_wall_s": round(wall_s, 2),
        "device_busy_s": round(dev_busy, 3),
        "device_duty": round(dev_busy / wall_s, 4) if wall_s else None,
        "stage_share": {k: round(v, 4) for k, v in shares.items()},
        "stage_busy_s": {k: round(v, 3) for k, v in stage_busy.items()},
        "components_s": {k: round(v, 3) for k, v in comps.items()},
        "components_share_of_sum": {
            k: round(max(0.0, v) / total_comp, 4) for k, v in comps.items()
        } if total_comp else {},
        "components_duty": {
            k: round(max(0.0, v) / wall_s, 5) for k, v in comps.items()
        } if wall_s else {},
        "encoder_calls": enc_calls,
        "encoder_tokens": enc_tokens,
        "n_note_calls": len(nws_raw),
        "note_window_s": round(sum(b - a for a, b in nws), 3),
        "note_overlap_with_turns": round(ov, 4),
        "note_attribution_trustworthy": ov < 0.05,
        "n_users": len(traces),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag[=label[:events_file[:log_file]]] -- events/log may be "
                         "shared across arms from one server run")
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--outdir-tpl", default="mu_{tag}",
                    help="directory under --results holding ttfa_user*.jsonl")
    ap.add_argument("--out", default="/data/zx/results/session_breakdown.json")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    summary: dict = {"arms": {}}
    for spec in args.arms:
        tag, _, rest = spec.partition("=")
        parts = rest.split(":") if rest else []
        label = parts[0] if parts else tag
        events = parts[1] if len(parts) > 1 else None
        logf = parts[2] if len(parts) > 2 else None
        a = analyse(res, tag, args.outdir_tpl.format(tag=tag), events, logf)
        if a is None:
            print(f"  [{tag}] missing inputs, skipped")
            continue
        a["label"] = label
        summary["arms"][tag] = a

    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    arms = summary["arms"]
    if not arms:
        print("no arms analysed")
        return 1

    ORDER = ["vision_encoder", "audio_encoder", "memory_note",
             "thinker_llm_residual", "talker", "code2wav"]
    PRETTY = {"vision_encoder": "vision encoder", "audio_encoder": "audio encoder",
              "memory_note": "memory-note call", "thinker_llm_residual": "thinker LLM (resid)",
              "talker": "talker", "code2wav": "code2wav"}

    print("\n=== SESSION SHAPE ===")
    print(f"{'arm':<22}{'users':>6}{'wall s':>9}{'device busy s':>15}{'device duty':>13}")
    for tag, a in arms.items():
        print(f"{a['label']:<22}{a['n_users']:>6}{a['session_wall_s']:>9}"
              f"{a['device_busy_s']:>15}{(a['device_duty'] or 0):>12.1%}")

    print("\n=== PER-STAGE RELATIVE SHARE (per-PID NVML; ratios only) ===")
    names = sorted({n for a in arms.values() for n in (a.get('stage_share') or {})})
    print(f"{'arm':<22}" + "".join(f"{n:>12}" for n in names))
    for tag, a in arms.items():
        print(f"{a['label']:<22}"
              + "".join(f"{(a.get('stage_share') or {}).get(n, 0):>11.1%} " for n in names))
    print("  Absolute per-stage seconds below are device_busy x this share.")
    print("  Per-PID SM is sampled sparsely (~7.7% of samples) and is a MAX over")
    print("  NVML's window, so it is trustworthy as a ratio and NOT as a rate.")

    print("\n=== RESOURCE SHARE PER COMPONENT (share of summed component time) ===")
    print(f"{'arm':<22}" + "".join(f"{PRETTY[c]:>21}" for c in ORDER))
    for tag, a in arms.items():
        row = "".join(f"{a['components_share_of_sum'].get(c, 0):>20.1%} " for c in ORDER)
        print(f"{a['label']:<22}{row}")

    print("\n=== SAME, AS GPU SECONDS ===")
    print(f"{'arm':<22}" + "".join(f"{PRETTY[c]:>21}" for c in ORDER))
    for tag, a in arms.items():
        row = "".join(f"{a['components_s'].get(c, 0):>20.2f} " for c in ORDER)
        print(f"{a['label']:<22}{row}")

    print("\n=== DUTY CYCLE PER COMPONENT (share of session wall clock) ===")
    print(f"{'arm':<22}" + "".join(f"{PRETTY[c]:>21}" for c in ORDER))
    for tag, a in arms.items():
        row = "".join(f"{a['components_duty'].get(c, 0):>20.2%} " for c in ORDER)
        print(f"{a['label']:<22}{row}")

    print("\n=== MEMORY-NOTE ATTRIBUTION QUALITY ===")
    for tag, a in arms.items():
        if not a["n_note_calls"]:
            continue
        flag = "OK" if a["note_attribution_trustworthy"] else "NOT TRUSTWORTHY"
        print(f"  {a['label']:<20} calls={a['n_note_calls']:>3} "
              f"window={a['note_window_s']:>6.2f}s  stage-0 busy in window="
              f"{a['components_s']['memory_note']:>6.2f}s")
        print(f"  {'':<20} overlap with somebody's turn = "
              f"{a['note_overlap_with_turns']:.1%}  -> {flag}")
    print("  Under contention the note shares stage 0 with other users' turns and")
    print("  NVML cannot separate them. When overlap is high, use the A/B")
    print("  difference between policies at equal user count instead.")

    print("\n=== ENCODER WORK (exact, CUDA events) ===")
    print(f"{'arm':<22}{'vis calls':>11}{'vis tok':>11}{'aud calls':>11}{'aud tok':>11}")
    for tag, a in arms.items():
        print(f"{a['label']:<22}{a['encoder_calls']['vision_encoder']:>11}"
              f"{a['encoder_tokens']['vision_encoder']:>11}"
              f"{a['encoder_calls']['audio_encoder']:>11}"
              f"{a['encoder_tokens']['audio_encoder']:>11}")

    print(f"\n-> {args.out}")
    print("\nCaveats that travel with these numbers:")
    print("  * thinker LLM is a RESIDUAL (stage-0 busy minus encoders minus note).")
    print("    It absorbs any attribution error in the other three rows.")
    print("  * per-process SM%% is an NVML sampled estimate, not exact kernel time,")
    print("    and SM util means 'a warp resident', not occupancy.")
    print("  * encoder rows are exact: they never enter a CUDA graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
