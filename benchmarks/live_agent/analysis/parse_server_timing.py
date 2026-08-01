#!/usr/bin/env python3
"""Parse the engine's own [TIMING] lines as an independent cross-check.

vllm-omni's streaming-video handler logs, per turn:

    [TIMING] mode=on total=9.96s first_text=0.10s first_audio=0.34s audio_chunks=37

These are measured inside the server process, so they are independent of the
client-side trace. Agreement between the two is the check that the client
harness is not inventing or mis-anchoring latency.

audio_chunks counts engine output steps that carried audio. Comparing it with
the number of response.audio.delta events the client actually received
quantifies the streaming-audio serialization defect directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics as st

PAT = re.compile(
    r"\[TIMING\] mode=(?P<mode>\w+) total=(?P<total>[-\d.]+)s "
    r"first_text=(?P<ft>[-\d.]+)s first_audio=(?P<fa>[-\d.]+)s "
    r"audio_chunks=(?P<n>\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--traces", default=None,
                    help="glob of client traces, to compare delivered vs generated audio")
    ap.add_argument("--skip", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for line in pathlib.Path(args.log).read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if m:
            rows.append({"mode": m["mode"], "total": float(m["total"]),
                         "ft": float(m["ft"]), "fa": float(m["fa"]),
                         "n": int(m["n"])})
    if not rows:
        print(f"no [TIMING] lines in {args.log}")
        return 1
    rows = rows[args.skip:]
    print(f"\n=== server-side [TIMING], {len(rows)} turns "
          f"(first {args.skip} dropped) ===")
    print(f"{'turn':>5}{'first_text s':>14}{'first_audio s':>15}"
          f"{'total s':>10}{'audio steps':>13}")
    for i, r in enumerate(rows):
        print(f"{i+args.skip:>5}{r['ft']:>14.3f}{r['fa']:>15.3f}"
              f"{r['total']:>10.2f}{r['n']:>13}")
    print(f"\n{'metric':<22}{'p50':>10}{'min':>10}{'max':>10}")
    for k, lab in (("ft", "first_text (server)"), ("fa", "first_audio (server)"),
                   ("total", "turn total"), ("n", "audio steps")):
        v = [r[k] for r in rows]
        print(f"{lab:<22}{st.median(v):>10.3f}{min(v):>10.3f}{max(v):>10.3f}")

    if args.traces:
        delivered = []
        for f in sorted(glob.glob(args.traces)):
            per = {}
            for line in pathlib.Path(f).read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("k") == "rx_audio_delta" or e.get("k") == "rx_first_audio":
                    per[e.get("rep")] = per.get(e.get("rep"), 0) + 1
            delivered.extend(v for k, v in sorted(per.items()) if k is not None and k >= args.skip)
        gen = [r["n"] for r in rows]
        if delivered and gen:
            print(f"\n=== generated vs delivered audio outputs ===")
            print(f"  engine produced (p50):   {st.median(gen):.0f} steps/turn")
            print(f"  client received (p50):   {st.median(delivered):.0f} deltas/turn")
            lost = 1 - st.median(delivered) / st.median(gen)
            print(f"  -> discarded by the serialization defect: {lost*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
