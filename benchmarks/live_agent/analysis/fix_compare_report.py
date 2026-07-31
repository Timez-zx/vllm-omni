#!/usr/bin/env python3
"""Did the frame-handling fix work, and which part of it did the work?

Four arms, each one variable apart, all on high-motion content at 1 user with the
same protocol as the baseline (4 sequential sessions x 60 turns, n=236, turn indices
2-60):

    S_high     shipped frame handling, 1280x720          <- must reproduce the baseline
    N720_high  append-only + retained-gap in [4,10]      <- frame handling only
    N640_high  same at 640x352 (220 tok/frame not 880)   <- adds the resolution cut
    P640_high  same plus stage-1 prefix caching          <- adds that config bit

WHAT EACH NUMBER IS FOR

`ms per 1,000 prompt tokens` on the prefill segment is the prefix-cache detector, and
it is the single most informative column. The baseline measured 79.8 ms/1k on high
motion (a full recompute of ~14,400 tokens every turn) against 11.8 ms/1k on low
motion, whose frame list happened to be append-only and whose prompt therefore grew
1,374 -> 9,668 tokens with prefill flat at 73-85 ms. If the fix works, high motion
should move from ~80 to ~12.

`biggest pause` is the other half, and it may move the WRONG WAY. That pause is stage 1
re-prefilling its whole prompt, measured at 17 ms per 1,000 prompt tokens with no
cache. Append-only makes the prompt grow over a session, so the pause should grow with
it unless stage-1 prefix caching is on. Reporting TTFA alone would hide that trade,
which is why the pause and its turn-index trend are broken out.

TURN-INDEX TRENDS ARE NOT OPTIONAL HERE. The whole design changes what happens as a
session ages: the shipped path re-picks 16 frames forever, so its prompt is constant,
while append-only accumulates. A p50 over 236 turns averages the cheap early turns
with the expensive late ones and can make a growing cost look like a flat win. Every
headline number is therefore also reported per 10-turn block.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGTS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")
TOKROW = re.compile(r"\|\s*num_tokens_in\s*\|(.+)\|\s*$")

ARMS = [
    ("S_high", "shipped frame handling, 1280x720"),
    ("N720_high", "append-only + gaps[4,10], 1280x720"),
    ("N640_high", "append-only + gaps[4,10], 640x352"),
    ("P640_high", "N640 + stage-1 prefix caching"),
]
BASELINE = ("vt_u1_high", "BASELINE measured earlier tonight")

# Each arm's prompt lengths live in the log of the server that served it. The baseline
# ran under a different server instance, so pointing every arm at one log would silently
# return nothing for it (the wall-clock window would not overlap) and print "-" as if
# the data did not exist.
LOG_FOR = {"vt_u1_high": "server_u1t.log"}
DEFAULT_LOG = "server_fx.log"


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def ms(x, w=9):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.0f}"


def rows_of(res: pathlib.Path, tag: str) -> list[dict]:
    p = res / f"decomp_{tag}.json" if tag.startswith("vt_") else res / f"decomp_vt_u1_{tag}.json"
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text()).get("rows", [])
            if r.get("ttfa_end_s") is not None]


def trace_dir(res: pathlib.Path, tag: str) -> pathlib.Path:
    return res / (tag if tag.startswith("vt_") else f"vt_u1_{tag}")


def pauses(res: pathlib.Path, tag: str) -> dict:
    """Per-turn biggest inter-token pause before the first sound, by session and turn.

    Only meaningful where --trace-deltas was on. Returns {} otherwise rather than
    silently reporting zeros.
    """
    out: dict = {}
    for p in sorted(glob.glob(str(trace_dir(res, tag) / "ttfa_user*.jsonl"))):
        by: dict = {}
        for line in open(p):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k, r = e.get("k"), e.get("rep")
            if k == "rx_text_delta":
                by.setdefault(r, {}).setdefault("t", []).append(e["w"])
            elif k == "rx_first_audio":
                by.setdefault(r, {})["fa"] = e["w"]
        for r, d in by.items():
            if not r or r < 1 or "fa" not in d or "t" not in d:
                continue
            b = sorted(w for w in d["t"] if w <= d["fa"])
            if len(b) >= 2:
                out.setdefault(r, []).append(
                    max((y - x) * 1000 for x, y in zip(b, b[1:])))
    return out


def prompt_tokens(res: pathlib.Path, tag: str, log: pathlib.Path) -> dict:
    """Stage-0 prompt length, windowed to this arm and bucketed by turn block.

    Order-matched rather than turn-matched: the log has one table per turn in
    sequence, so the k-th table in the window is the k-th turn of the arm. That is
    exact for a single-user run because there is no interleaving.
    """
    d = trace_dir(res, tag)
    ws = []
    for p in d.glob("ttfa_user*.jsonl"):
        for line in p.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    ws.append(json.loads(line)["w"])
                except Exception:
                    pass
    if not ws or not log.exists():
        return {}
    lo, hi = min(ws) - 3, max(ws) + 3
    keep, vals = False, []
    for ln in ANSI.sub("", log.read_text(errors="ignore")).splitlines():
        m = LOGTS.search(ln)
        if m:
            mo, da, hh, mi, ss = (int(x) for x in m.groups())
            try:
                t = datetime.datetime(2026, mo, da, hh, mi, ss).timestamp()
            except ValueError:
                t = None
            if t is not None:
                keep = lo <= t <= hi
        if not keep:
            continue
        r = TOKROW.search(ln)
        if r:
            c = [x.strip().replace(",", "") for x in r.group(1).split("|")]
            if c and c[0].isdigit() and int(c[0]):
                vals.append(int(c[0]))
    return {"all": vals, "p50": pct(vals, .5), "n": len(vals)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--out", default="/data/zx/results/fix_compare.json")
    a = ap.parse_args()
    res = pathlib.Path(a.results)

    have = []
    for tag, label in [BASELINE] + ARMS:
        r = rows_of(res, tag)
        if r:
            have.append((tag, label, r))
    if not have:
        print("no arms found yet")
        return 1

    print("\n" + "=" * 104)
    print("HIGH MOTION, 1 USER -- DID THE FIX WORK?")
    print("=" * 104)
    print(f"{'arm':<12}{'n':>5}{'TTFA p50':>10}{'p95':>9}{'admit':>8}"
          f"{'prefill':>9}{'ramp':>8}{'prompt tok':>12}{'ms/1k tok':>11}{'>1s':>6}")
    summary = {}
    for tag, label, r in have:
        t = [x["ttfa_end_s"] * 1000 for x in r]
        adm = [x["admit_s"] * 1000 for x in r if x.get("admit_s") is not None]
        pf = [x["to_first_token_s"] * 1000 for x in r if x.get("to_first_token_s") is not None]
        rp = [x["to_first_audio_s"] * 1000 for x in r if x.get("to_first_audio_s") is not None]
        ptk = prompt_tokens(res, tag, res / LOG_FOR.get(tag, DEFAULT_LOG))
        p50pf, p50tok = pct(pf, .5), ptk.get("p50")
        per1k = (1000.0 * p50pf / p50tok) if (p50pf and p50tok) else None
        summary[tag] = {"label": label, "n": len(t), "ttfa_p50": pct(t, .5),
                        "ttfa_p95": pct(t, .95), "admit_p50": pct(adm, .5),
                        "prefill_p50": p50pf, "ramp_p50": pct(rp, .5),
                        "prompt_p50": p50tok, "prefill_ms_per_1k": per1k,
                        "over1s": sum(1 for x in t if x > 1000) / len(t)}
        s = summary[tag]
        print(f"{tag:<12}{s['n']:>5}{ms(s['ttfa_p50'], 10)}{ms(s['ttfa_p95'])}"
              f"{ms(s['admit_p50'], 8)}{ms(s['prefill_p50'])}{ms(s['ramp_p50'], 8)}"
              f"{(f'{p50tok:,.0f}' if p50tok else '-'):>12}"
              f"{(f'{per1k:.1f}' if per1k else '-'):>11}{s['over1s']:>6.0%}")
    print("  'ms/1k tok' is the prefix-cache detector. Baseline high motion measured 79.8")
    print("  (full recompute every turn); low motion, which was accidentally append-only,")
    print("  measured 11.8. A working fix moves high motion toward 12.")

    print("\n--- the per-turn blocking pause, and whether it GROWS over a session ---")
    print("  (stage 1 re-prefills its whole prompt at ~17 ms/1k with no cache, so under")
    print("   append-only this is the number that can move the wrong way)")
    print(f"{'arm':<12}{'n':>5}{'pause p50':>11}" +
          "".join(f"{f'turns {a_}-{b_}':>12}" for a_, b_ in
                  ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59))))
    for tag, label, r in have:
        pz = pauses(res, tag)
        if not pz:
            print(f"{tag:<12}{'-':>5}{'no delta traces':>11}")
            continue
        allp = [v for vs in pz.values() for v in vs]
        row = f"{tag:<12}{len(allp):>5}{pct(allp, .5):>11.0f}"
        for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59)):
            blk = [v for rr, vs in pz.items() if lo <= rr <= hi for v in vs]
            row += f"{(pct(blk, .5) if blk else float('nan')):>12.0f}"
        print(row)
        summary.setdefault(tag, {})["pause_p50"] = pct(allp, .5)

    print("\n--- prompt growth over the session (append-only should climb, shipped flat) ---")
    print(f"{'arm':<12}" + "".join(f"{f'turns {a_}-{b_}':>12}" for a_, b_ in
                                   ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59))))
    for tag, label, r in have:
        ptk = prompt_tokens(res, tag, res / LOG_FOR.get(tag, DEFAULT_LOG))
        v = ptk.get("all") or []
        if not v:
            print(f"{tag:<12}{'(no log window)':>12}")
            continue
        # 4 sessions x 59-60 turns land in sequence; bucket by position within a session
        per = max(1, len(v) // 4)
        row = f"{tag:<12}"
        for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59)):
            idx = [s * per + k for s in range(4) for k in range(lo - 1, min(hi, per))]
            blk = [v[i] for i in idx if 0 <= i < len(v)]
            row += f"{(pct(blk, .5) if blk else float('nan')):>12,.0f}"
        print(row)

    print("\n--- TTFA over the session, same buckets ---")
    print(f"{'arm':<12}" + "".join(f"{f'turns {a_}-{b_}':>12}" for a_, b_ in
                                   ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59))))
    for tag, label, r in have:
        row = f"{tag:<12}"
        for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 59)):
            blk = [x["ttfa_end_s"] * 1000 for x in r if lo <= x.get("rep", 0) <= hi]
            row += f"{(pct(blk, .5) if blk else float('nan')):>12.0f}"
        print(row)

    # Two ways a "faster" result could be an artifact rather than a win, both checked
    # rather than assumed away.
    print("\n--- artifact check 1: did the engine start thrashing? ---")
    print("  Append-only grows the prompt, and max_frames was raised to 512 so eviction")
    print("  would not break the prefix. If that overruns stage-0's 106,880-token KV the")
    print("  engine preempts and recomputes, which would corrupt every number above.")
    print(f"{'arm':<12}{'preempt/recompute lines':>26}{'cache-full':>12}{'prompt too long':>18}")
    for tag, label, r in have:
        lg = res / LOG_FOR.get(tag, DEFAULT_LOG)
        if not lg.exists():
            print(f"{tag:<12}{'(no log)':>26}")
            continue
        txt = ANSI.sub("", lg.read_text(errors="ignore"))
        n_pre = len(re.findall(r"[Pp]reempt|recompute", txt))
        n_full = len(re.findall(r"[Cc]annot allocate|out of.*blocks|KV cache is full", txt))
        n_long = len(re.findall(r"longer than the maximum", txt))
        print(f"{tag:<12}{n_pre:>26}{n_full:>12}{n_long:>18}")
    print("  Counts are per SERVER LOG, not per arm, because arms sharing a server share")
    print("  a log; a nonzero count is a flag to investigate, not proof of which arm.")

    print("\n--- artifact check 2: did the model just answer with less? ---")
    print("  A shorter reply lowers stage-1 and stage-2 work and would shorten the turn")
    print("  cycle, so a resolution cut that made answers terser could look like a")
    print("  latency win. chars is the text present when the sound started; audio")
    print("  deltas and speech seconds describe the reply that was actually produced.")
    print(f"{'arm':<12}{'chars p50':>11}{'reply audio s p50':>19}{'turn cycle s p50':>18}")
    for tag, label, r in have:
        ch = [x["chars"] for x in r if x.get("chars") is not None]
        aud = [(x["t_end"] - x["t_first_audio"]) for x in r
               if x.get("t_end") and x.get("t_first_audio")]
        cyc = [(x["t_end"] - x["t_begin"]) for x in r
               if x.get("t_end") and x.get("t_begin")]
        print(f"{tag:<12}{(pct(ch, .5) or 0):>11.0f}"
              f"{(pct(aud, .5) or 0):>19.2f}{(pct(cyc, .5) or 0):>18.2f}")

    print("\n--- verdict against the baseline ---")
    b = summary.get(BASELINE[0])
    if b:
        for tag, _ in ARMS:
            s = summary.get(tag)
            if not s or not s.get("ttfa_p50"):
                continue
            r = s["ttfa_p50"] / b["ttfa_p50"]
            print(f"  {tag:<12} TTFA {s['ttfa_p50']:.0f} vs baseline {b['ttfa_p50']:.0f} ms"
                  f"  = {r:.2f}x"
                  + (f"   ({1/r:.1f}x faster)" if r < 1 else "   (SLOWER)"))
        s = summary.get("S_high")
        if s and s.get("ttfa_p50"):
            d = abs(s["ttfa_p50"] - b["ttfa_p50"]) / b["ttfa_p50"]
            print(f"\n  patch sanity: S_high is {d:.1%} from the baseline "
                  f"({'OK, the off-path reproduces upstream' if d < 0.10 else 'PROBLEM: the patched off-path does not reproduce upstream'})")

    pathlib.Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
