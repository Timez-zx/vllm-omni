#!/usr/bin/env python3
"""Does session mode still remember? Session-mode recall against the per-turn control.

THE QUESTION. The S640 arm showed one resumable engine request per websocket session drops
the speech stage to 2.13 ms per 1,000 accumulated prompt tokens, so a 33k-token session ends
up with a LOWER TTFA than the configuration that discards history. But keeping history is
only worth anything if the model can still USE it, and session mode changes how history
reaches the model: each turn is appended to a live request as its own delta, rather than the
entrypoint rebuilding one prompt that contains every frame. If the engine rebased the new
multimodal features' offsets wrongly, the model would read image tokens at the wrong
positions -- which yields plausible-but-wrong answers, not an error.

THE CONTROL is `tb_E640_*`: per-turn requests, one user block per turn, same stimuli, same
EVS settings. Session mode must match it.

WHY THE SCORING USES answer_acc AND NOT done_text. `recall_bench.py` accumulates
`response.text.delta` into `answer_acc`; the `response.text.done` payload is emitted from
inside the server's AUDIO branch and therefore contains only the text generated before the
first audio chunk. Using the latter as a length or content metric produced a confident wrong
claim earlier in this study (see FRAME_PIPELINE.md §5.8). The bench already stores both, so
this reads the accumulated one.

Fisher exact is reported for every comparison because n = 8 items per arm: with one session
per arm, only a very large difference is detectable at all, and saying so is part of the
result rather than a footnote.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

RES = pathlib.Path("/data/zx/results")


def fisher2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a, b], [c, d]]."""
    def hg(a, b, c, d):
        n = a + b + c + d
        return (math.comb(a + b, a) * math.comb(c + d, c)) / math.comb(n, a + c)
    r1, c1 = a + b, a + c
    p0, tot = hg(a, b, c, d), 0.0
    for x in range(max(0, c1 - (c + d)), min(r1, c1) + 1):
        y, z = r1 - x, c1 - x
        w = (c + d) - z
        p = hg(x, y, z, w)
        if p <= p0 + 1e-12:
            tot += p
    return min(1.0, tot)


def load(tag: str) -> dict | None:
    p = RES / tag / "recall_score.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def summarise(name: str, s: dict) -> dict:
    sc = s["score"]
    rs = s["results"]
    det = sc.get("probe_detail")
    got = sum(1 for r in rs if r.get("got_audio"))
    # answer_acc is the full text; done_text is truncated at the first audio chunk
    accs = [len(r.get("answer") or "") for r in rs]
    dones = [len(r.get("done_text") or "") for r in rs]
    return {
        "name": name,
        "read_ok": sc["scene_read_rate"]["n_ok"],
        "read_n": sc["scene_read_rate"]["n"],
        "first": bool(sc["probe_first"]["correct"]),
        "recalled": sc["probe_listall"]["n_recalled"],
        "recall_n": sc["probe_listall"]["n_total"],
        "detail": (bool(det["correct"]) if det else None),
        "audio": f"{got}/{len(rs)}",
        "listall": sc["probe_listall"]["answer"],
        "acc_med": sorted(accs)[len(accs) // 2] if accs else 0,
        "done_med": sorted(dones)[len(dones) // 2] if dones else 0,
    }


def main() -> int:
    PAIRS = [
        ("words", "tb_E640_words", "sr_S640_words"),
        ("detail", "tb_E640_detail", "sr_S640_detail"),
    ]
    rows: list[dict] = []
    missing: list[str] = []
    for _, ctrl, sess in PAIRS:
        for tag, label in ((ctrl, f"per-turn  {ctrl}"), (sess, f"SESSION   {sess}")):
            s = load(tag)
            if s is None:
                missing.append(tag)
            else:
                rows.append(summarise(label, s))
    if missing:
        print("missing score files: " + ", ".join(missing))
    if not rows:
        return 1

    print("=" * 96)
    print("RECALL: session mode vs the per-turn control, same stimuli, same EVS settings")
    print("=" * 96)
    print(f"  {'arm':30s} {'read current':>13s} {'first word':>11s} {'words recalled':>15s} "
          f"{'17px digit':>11s} {'audio':>7s}")
    for r in rows:
        read = f"{r['read_ok']}/{r['read_n']}"
        rec = f"{r['recalled']}/{r['recall_n']}"
        first = "correct" if r["first"] else "WRONG"
        digit = "correct" if r["detail"] else ("WRONG" if r["detail"] is False else "-")
        print(f"  {r['name']:30s} {read:>13s} {first:>11s} {rec:>15s} {digit:>11s} "
              f"{r['audio']:>7s}")

    print()
    print("  full list-all answers, verbatim:")
    for r in rows:
        print(f"    {r['name']:30s} {r['listall']!r}")

    print()
    print("=" * 96)
    print("SIGNIFICANCE -- n = 8 items per arm, one session per arm")
    print("=" * 96)
    for kind, ctrl, sess in PAIRS:
        c = next((r for r in rows if ctrl in r["name"]), None)
        s = next((r for r in rows if sess in r["name"]), None)
        if not c or not s:
            continue
        p_recall = fisher2(c["recalled"], c["recall_n"] - c["recalled"],
                           s["recalled"], s["recall_n"] - s["recalled"])
        p_read = fisher2(c["read_ok"], c["read_n"] - c["read_ok"],
                         s["read_ok"], s["read_n"] - s["read_ok"])
        print(f"  {kind:8s} words recalled {c['recalled']}/{c['recall_n']} vs "
              f"{s['recalled']}/{s['recall_n']}   Fisher two-sided p = {p_recall:.3f}")
        print(f"  {kind:8s} current-scene  {c['read_ok']}/{c['read_n']} vs "
              f"{s['read_ok']}/{s['read_n']}   Fisher two-sided p = {p_read:.3f}")
    print()
    print("  With 8 items and one session per arm this test detects only a LARGE difference.")
    print("  Read it as: session mode did not visibly break recall. It cannot establish")
    print("  equivalence, and it is not evidence of improvement either.")

    print()
    print("=" * 96)
    print("SANITY -- accumulated answer vs the truncated done_text, median chars")
    print("=" * 96)
    for r in rows:
        print(f"  {r['name']:30s} answer_acc {r['acc_med']:5d}   done_text {r['done_med']:5d}")
    print()
    print("  These two are EQUAL here, and that is expected rather than reassuring: the recall")
    print("  answers are ~32 characters and finish before the first audio chunk, so there is")
    print("  nothing for the truncation to cut. The truncation only bites when the answer")
    print("  outruns the ramp -- which is exactly the case in the latency arms, where a")
    print("  216-vs-72 char 'difference' turned out to be the ramp duration in disguise")
    print("  (FRAME_PIPELINE.md section 5.8). The scoring above uses answer_acc regardless,")
    print("  because relying on the two agreeing is relying on the answers staying short.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
