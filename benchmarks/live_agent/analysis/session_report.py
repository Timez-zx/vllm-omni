#!/usr/bin/env python3
"""Did one request per session remove the stage-0 -> stage-1 payload copy?

CLIENT-SIDE THROUGHOUT, on purpose. The session arm produces no StageRequestStats at all --
those tables print when a request FINISHES and a resumable session request never does -- so
the per-stage server split that the rest of this study uses is unavailable for S640. Rather
than compare S640's client numbers against other arms' server numbers, every arm here is
measured the same way, from the client trace:

    thinker      tx_query      -> rx_first_text
    talker+c2w   rx_first_text -> rx_first_audio
    TTFA         tx_query      -> rx_first_audio

That split was cross-validated against the server-side one on four arms and agreed to
within ~2% on talker+code2wav (the server sees 10-40 ms less on the thinker side, being
inside the websocket). Ample here: the question is whether the talker segment is ~200 ms or
~1,900 ms.

THE X-AXIS differs by necessity and the difference is stated rather than hidden:
  * per-turn arms: stage-0 `num_tokens_in`, the real prompt length, from the server log.
  * S640: the running total of submitted deltas, logged by the patch itself because nothing
    else reports it. This UNDERSTATES the true accumulated prompt, because the engine also
    folds each turn's generated text back in (~150 tokens/turn, so ~7.5k over 50 turns).
    The bias is CONSERVATIVE for the conclusion: S640's true prompt at a given plotted x is
    larger than plotted, so it is achieving its latency at more context than credited.

THE COPY MODEL, for reference: two [L, 2048] bf16 tensors per prompt position = 8 KB/token,
at a bandwidth of 221.8 MB/s with an 18 ms floor, fitted on P640 (R^2 0.9963) and confirmed
out-of-sample on F640 to within 10% at four of six bands.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage_stats_v2 import derive, parse  # noqa: E402

RES = pathlib.Path("/data/zx/results")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
H, B = 2048, 2
XFER_MBPS, XFER_FLOOR = 221.8, 18.0
DELTA_RX = re.compile(r"turn=(\d+) queue delta: \d+ new frames, (\d+) tokens, "
                      r"cum=(\d+), talker_placeholder=(-?\d+)")


def copy_model_ms(L: float) -> float:
    return XFER_FLOOR + (2 * L * H * B / 1e6) / XFER_MBPS * 1000


def client_turns(d: str) -> list[dict]:
    """Per-turn client-side split, one dict per scored turn."""
    out = []
    for f in sorted(glob.glob(str(RES / d / "ttfa_user*.jsonl"))):
        tq, ft, fa = {}, {}, {}
        for line in open(f):
            r = json.loads(line)
            rep = r.get("rep", -1)
            if rep < 0:
                continue
            if r["k"] == "tx_query":
                tq[rep] = r["w"]
            elif r["k"] == "rx_first_text":
                ft[rep] = r["w"]
            elif r["k"] == "rx_first_audio":
                fa[rep] = r["w"]
        for rep in sorted(tq):
            if rep in ft and rep in fa:
                out.append({"rep": rep, "file": f,
                            "thinker": (ft[rep] - tq[rep]) * 1000,
                            "talker": (fa[rep] - ft[rep]) * 1000,
                            "ttfa": (fa[rep] - tq[rep]) * 1000})
    return out


def session_cum(log: str, n_expect: int | None = None) -> dict[int, tuple[int, int]]:
    """turn -> (cum tokens, talker placeholder), from the boot(s) of a session log."""
    lines = [ANSI.sub("", l) for l in (RES / log).open(errors="replace")]
    marks = [i for i, l in enumerate(lines) if l.startswith("===== PA_SESSION")] + [len(lines)]
    best: dict[int, tuple[int, int]] = {}
    for a, b in zip(marks, marks[1:]):
        ds: dict[int, tuple[int, int]] = {}
        for l in lines[a:b]:
            m = DELTA_RX.search(l)
            if m:
                ds[int(m.group(1))] = (int(m.group(3)), int(m.group(4)))
        if n_expect is None:
            if len(ds) > len(best):
                best = ds
        elif len(ds) == n_expect:
            best = ds
    return best


def server_prompt(log: str, boot: int, skip: int) -> dict[int, float]:
    p = RES / log
    if not p.exists():
        return {}
    rs = [r for r in derive(parse(p)) if r["boot"] == boot][skip:]
    return {i: r["prompt_tokens"] for i, r in enumerate(rs) if r["prompt_tokens"]}


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main() -> int:
    # per-turn arms: client trace dir, and the server log giving the real prompt length
    PER_TURN = [
        ("C640", "vt_u1_C640_high", "server_ft.log", 1, 0),
        ("T640", "vt_u1_T640_high", "server_ft.log", 2, 0),
        ("E640", "vt_u1_E640_high", "server_td.log", 1, 6),
        ("F640", "vt_u1_F640_high", "server_td.log", 2, 6),
    ]
    arms: dict[str, list[dict]] = {}
    for label, cdir, log, boot, skip in PER_TURN:
        rows = client_turns(cdir)
        if not rows:
            continue
        sp = server_prompt(log, boot, skip)
        # both are ordered per session/boot; the client drops turn 0 as cold
        for i, r in enumerate(rows):
            r["prompt"] = sp.get(r["rep"] - 1)
        arms[label] = [r for r in rows if r.get("prompt")]

    srows = client_turns("vt_u1_S640_high")
    if not srows:
        print("S640 has produced no client trace yet.")
        return 1
    cum = session_cum("server_ss.log")
    for r in srows:
        d = cum.get(r["rep"])
        r["prompt"] = d[0] if d else None
        r["placeholder"] = d[1] if d else None
    arms["S640"] = [r for r in srows if r.get("prompt")]

    print("=" * 94)
    print("S640 ARM IDENTITY -- the talker's prompt must stay delta-sized as the session grows")
    print("=" * 94)
    s = arms["S640"]
    print(f"  turns with a joined x-axis: {len(s)} of {len(srows)}")
    print(f"  {'turn':>5s} {'cum tok':>9s} {'talker ph':>10s} {'ratio':>7s}")
    for r in s[::max(1, len(s) // 12)]:
        print(f"  {r['rep']:5d} {r['prompt']:9.0f} {r['placeholder']:10.0f} "
              f"{r['placeholder']/r['prompt']:7.3f}")
    phs = [r["placeholder"] for r in s]
    print(f"  placeholder across the whole arm: median {med(phs):.0f}, "
          f"range {min(phs):.0f}-{max(phs):.0f}  <-- must be FLAT")

    print()
    print("=" * 94)
    print("THE COMPARISON -- talker+code2wav at matched accumulated prompt, client-side")
    print("=" * 94)
    BANDS = [(2000, 5000), (8000, 12000), (14000, 18000), (19000, 24000),
             (26000, 32000), (33000, 40000), (41000, 50000)]
    order = [k for k in ("C640", "T640", "E640", "F640", "S640") if k in arms]
    print(f"  {'band':>10s} " + "".join(f"{k:>13s}" for k in order) + f"{'copy model':>12s}")
    for lo, hi in BANDS:
        cells = ""
        for k in order:
            g = [r["talker"] for r in arms[k] if lo <= r["prompt"] <= hi]
            cells += f"{st.median(g):9.0f}({len(g):2d})" if len(g) >= 4 else f"{'-':>13s}"
        print(f"  {f'{lo//1000}-{hi//1000}k':>10s} {cells}{copy_model_ms((lo+hi)/2):12.0f}")

    print()
    print("  Every per-turn arm ships the WHOLE payload every turn, so its talker segment")
    print("  should track the copy model. S640 ships a delta, so it should not.")

    print()
    print("=" * 94)
    print("S640 PER-TURN -- is it flat?")
    print("=" * 94)
    print(f"  {'turns':>9s} {'cum tok':>9s} {'talker ph':>10s} {'thinker':>8s} "
          f"{'talker+c2w':>11s} {'TTFA':>7s} {'copy model':>11s}")
    n = len(s)
    step = max(1, n // 8)
    for a in range(0, n, step):
        g = s[a:a + step]
        cp = med(r["prompt"] for r in g)
        lbl = f"{g[0]['rep']}-{g[-1]['rep']}"
        print(f"  {lbl:>9s} "
              f"{cp:9.0f} {med(r['placeholder'] for r in g):10.0f} "
              f"{med(r['thinker'] for r in g):7.0f}ms {med(r['talker'] for r in g):10.0f}ms "
              f"{med(r['ttfa'] for r in g):6.0f}ms {copy_model_ms(cp):10.0f}ms")

    print()
    print("=" * 94)
    print("SLOPES -- talker+code2wav ms per 1,000 accumulated prompt tokens")
    print("=" * 94)
    for k in order:
        pts = [(r["prompt"], r["talker"]) for r in arms[k]]
        if len(pts) < 10:
            continue
        m = len(pts)
        mx = sum(p[0] for p in pts) / m
        my = sum(p[1] for p in pts) / m
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        if sxx == 0:
            print(f"  {k:6s} n={m:4d}  prompt does not vary (pinned)")
            continue
        sl = sxy / sxx
        print(f"  {k:6s} n={m:4d}  {1000*sl:8.2f} ms/1k  floor {my - sl*mx:7.0f} ms  "
              f"R2 {(sxy**2)/(sxx*syy) if syy else float('nan'):.4f}  "
              f"prompt {min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f}")
    print()
    print("  The copy model implies 36.9 ms/1k for an arm that ships everything. A slope near")
    print("  zero means the speech stage has stopped charging for context.")
    print()
    print("  CAVEAT, restated: S640's x-axis counts submitted deltas only and excludes the")
    print("  generated text the engine folds back in (~150 tokens/turn). Its true prompt is")
    print("  therefore LARGER than plotted, which understates the result rather than")
    print("  flattering it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
