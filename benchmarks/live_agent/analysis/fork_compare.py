#!/usr/bin/env python3
"""Does the fork reproduce the numbers the patch-overlay build produced?

WHY THIS IS NOT A FORMALITY. Three of the six commits are on the path this exercises, and
each one changed HOW a setting reaches the code, not just where it lives:

  * The overlay was five files copied over site-packages, driven by env vars
    (PA_SESSION / PA_APPEND_ONLY / PA_EVS_*). The fork reads everything from session.config,
    so the code that resolves the settings is different code.
  * The overlay never downscaled anything at run time: the 640x352 stimulus was produced
    OFFLINE by a script and the server only ever saw small frames. The fork downscales on
    arrival, so this run feeds the ORIGINAL 1280x720 frames and makes the server do it.
  * The EVS gap bound now goes through a public filter method (force_next_retain) instead
    of assigning to the filter's private attribute.

ONE KNOWN, INTENDED DIFFERENCE IN THE STIMULUS. The offline script squashed 1280x720 (16:9)
into exactly 640x352 (20:11), which is 220 tokens/frame. The fork preserves aspect ratio, so
fitting 1280x720 inside 640x352 gives 625x352 = 209 tokens/frame -- 5% fewer. That shifts
per-turn TTFA down slightly at a given turn index, and it is why the metric of record here is
the SLOPE in ms per 1,000 ACCUMULATED PROMPT TOKENS: normalising by token count removes the
difference instead of hiding it. The per-turn table is shown too, with the caveat attached.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

RES = pathlib.Path("/data/zx/results")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
DELTA_RX = re.compile(r"turn=(\d+) queue delta: (\d+) new frames, (\d+) tokens, "
                      r"cum=(\d+), talker_placeholder=(-?\d+)")

# The overlay build's measured result, for comparison. 2 sessions x 50 turns, n=100.
REF = {
    "slope_ms_per_1k": 2.13,
    "ttfa_p50": 360, "ttfa_p95": 477, "ttfa_p99": 494, "ttfa_max": 514,
    "placeholder_median": 598,
    "cum_max": 33772,
}


def client_turns(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RES / d / "ttfa_user*.jsonl"))):
        tq, ft, fa = {}, {}, {}
        for line in open(f):
            r = json.loads(line)
            rep = r.get("rep", -1)
            if rep < 1:            # drop turn 0 as cold, as every other arm does
                continue
            if r["k"] == "tx_query":
                tq[rep] = r["w"]
            elif r["k"] == "rx_first_text":
                ft[rep] = r["w"]
            elif r["k"] == "rx_first_audio":
                fa[rep] = r["w"]
        for rep in sorted(tq):
            if rep in ft and rep in fa:
                out.append({"rep": rep,
                            "thinker": (ft[rep] - tq[rep]) * 1000,
                            "talker": (fa[rep] - ft[rep]) * 1000,
                            "ttfa": (fa[rep] - tq[rep]) * 1000})
    return out


def session_deltas(log: str, marker: str) -> dict[int, tuple[int, int, int]]:
    """turn -> (delta tokens, cum tokens, talker placeholder) for the largest boot.

    Ties go to the LAST such boot, which is why the comparison below is `>=` and not `>`.
    The server log is appended across runs on purpose, so re-running the verification leaves
    two arms of the same length in it, while the client trace directory has been recreated
    and holds only the new one. Keeping the first of the two would join one run's server-side
    token counts onto the other run's client-side latencies -- turn numbers line up, nothing
    raises, and the x-axis silently belongs to a different experiment.
    """
    p = RES / log
    if not p.exists():
        return {}
    lines = [ANSI.sub("", l) for l in p.open(errors="replace")]
    marks = [i for i, l in enumerate(lines) if l.startswith(marker)] + [len(lines)]
    best: dict[int, tuple[int, int, int]] = {}
    for a, b in zip(marks, marks[1:]):
        ds: dict[int, tuple[int, int, int]] = {}
        for l in lines[a:b]:
            m = DELTA_RX.search(l)
            if m:
                ds[int(m.group(1))] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
        if len(ds) >= len(best):
            best = ds
    return best


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def ols(pts):
    n = len(pts)
    mx = sum(a for a, _ in pts) / n
    my = sum(b for _, b in pts) / n
    sxx = sum((a - mx) ** 2 for a, _ in pts)
    sxy = sum((a - mx) * (b - my) for a, b in pts)
    syy = sum((b - my) ** 2 for _, b in pts)
    if sxx == 0:
        return None
    sl = sxy / sxx
    return sl, my - sl * mx, ((sxy ** 2) / (sxx * syy) if syy else float("nan"))


def main() -> int:
    # The names are options rather than constants because more than one runner produces
    # this shape of data (run_fork_verify.sh writes vt_u1_FORK_high + server_fork.log,
    # run_wedge_probe.sh writes vt_u1_WEDGE + server_wedge.log). Copying one run's files
    # onto the other's names would leave two directories holding the same session, which
    # is exactly the kind of duplicate that later gets read as two independent runs.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="vt_u1_FORK_high",
                    help="client trace directory under /data/zx/results")
    ap.add_argument("--log", default="server_fork.log",
                    help="server log under /data/zx/results")
    ap.add_argument("--marker", default="===== FORK",
                    help="boot-marker PREFIX; the boot with the most turns is used")
    a = ap.parse_args()
    rows = client_turns(a.dir)
    if not rows:
        print("the fork arm has produced no client trace yet")
        return 1
    deltas = session_deltas(a.log, a.marker)
    for r in rows:
        d = deltas.get(r["rep"])
        r["delta_tok"], r["prompt"], r["placeholder"] = d if d else (None, None, None)
    joined = [r for r in rows if r.get("prompt")]

    print("=" * 92)
    print("ARM IDENTITY -- did the fork actually do what it claims?")
    print("=" * 92)
    print(f"  turns scored: {len(rows)}, with a joined x-axis: {len(joined)}")
    if joined:
        dts = [r["delta_tok"] for r in joined]
        phs = [r["placeholder"] for r in joined]
        print(f"  delta tokens per turn : median {st.median(dts):.0f}  range {min(dts)}-{max(dts)}")
        print(f"  talker placeholder    : median {st.median(phs):.0f}  range {min(phs)}-{max(phs)}"
              f"   (overlay build: {REF['placeholder_median']})")
        print(f"  accumulated prompt    : {joined[0]['prompt']} -> {joined[-1]['prompt']}"
              f"   (overlay build: -> {REF['cum_max']})")
        print(f"  placeholder / prompt at the end: "
              f"{joined[-1]['placeholder'] / joined[-1]['prompt']:.4f}   <- must be small")

    print()
    print("=" * 92)
    print("THE METRIC OF RECORD -- talker+code2wav ms per 1,000 accumulated prompt tokens")
    print("=" * 92)
    pts = [(r["prompt"], r["talker"]) for r in joined]
    if len(pts) >= 10:
        sl, icpt, r2 = ols(pts)
        got = 1000 * sl
        print(f"  fork    : {got:7.2f} ms/1k   floor {icpt:6.0f} ms   R2 {r2:.4f}   n={len(pts)}")
        print(f"  overlay : {REF['slope_ms_per_1k']:7.2f} ms/1k                              (n=100)")
        print(f"  ratio   : {got / REF['slope_ms_per_1k']:.2f}x")
        print()
        print("  For reference the per-turn builds measured on the same stimulus were")
        print("  36.5 (talker sized to the delta), 59.1 and 84.5 ms/1k. The claim being")
        print("  reproduced is that session mode is an order of magnitude below those, not")
        print("  that it hits 2.13 exactly -- a slope this close to zero is dominated by")
        print("  noise, which is what its low R2 says.")
    else:
        print(f"  only {len(pts)} joined points, need >= 10 for a slope")

    print()
    print("=" * 92)
    print("TTFA DISTRIBUTION")
    print("=" * 92)
    tt = [r["ttfa"] for r in rows]
    print(f"  {'build':10s} {'n':>4s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'max':>6s} {'>600ms':>8s}")
    print(f"  {'fork':10s} {len(tt):4d} {st.median(tt):6.0f} {q(tt,95):6.0f} {q(tt,99):6.0f} "
          f"{max(tt):6.0f} {100*sum(1 for x in tt if x>600)/len(tt):7.1f}%")
    print(f"  {'overlay':10s} {100:4d} {REF['ttfa_p50']:6d} {REF['ttfa_p95']:6d} "
          f"{REF['ttfa_p99']:6d} {REF['ttfa_max']:6d} {0.0:7.1f}%")
    print()
    print("  The fork's frames are 209 tokens each against the overlay's 220 (aspect ratio")
    print("  preserved vs squashed), so at a given turn index the fork carries ~5% less")
    print("  context. Expect its TTFA slightly lower, not identical.")

    print()
    print("=" * 92)
    print("PER-TURN")
    print("=" * 92)
    print(f"  {'turns':>9s} {'prompt':>8s} {'placeholder':>12s} {'thinker':>8s} "
          f"{'talker+c2w':>11s} {'TTFA':>7s}")
    n = len(joined)
    step = max(1, n // 8)
    for a in range(0, n, step):
        g = joined[a:a + step]
        f = lambda k: st.median([x[k] for x in g if x.get(k) is not None])  # noqa: E731
        lbl = f"{g[0]['rep']}-{g[-1]['rep']}"
        print(f"  {lbl:>9s} "
              f"{f('prompt'):8.0f} {f('placeholder'):12.0f} {f('thinker'):7.0f}ms "
              f"{f('talker'):10.0f}ms {f('ttfa'):6.0f}ms")

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    if len(pts) >= 10:
        got = 1000 * ols(pts)[0]
        if got < 8.0:
            print(f"  REPRODUCED. {got:.2f} ms/1k against the overlay's "
                  f"{REF['slope_ms_per_1k']:.2f}, both far below the 36.5 of the best")
            print("  per-turn build. The six commits carry the behaviour the overlay had.")
        else:
            print(f"  NOT REPRODUCED. {got:.2f} ms/1k is well above the overlay's "
                  f"{REF['slope_ms_per_1k']:.2f}.")
            print("  Check the gate output first: the likeliest cause is that the")
            print("  delta-shipping branch did not fire, which the placeholder ratio above")
            print("  would show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
