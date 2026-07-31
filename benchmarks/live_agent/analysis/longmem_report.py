#!/usr/bin/env python3
"""Compare history policies on both axes: does it remember, and what does it cost.

Reads, per policy arm:
  * ``recall_score.json``  from recall_bench.py   -> memory quality
  * ``server_<tag>.log``                          -> cost and context growth

Context growth comes from the engine's own ``num_tokens_in`` stats row, which is
the per-request prompt length. That was validated against an independent
instrumented measurement (a probe inside
``kv_cache_manager.get_computed_blocks``): both report 14,293 / 28,852 / 43,390 /
57,964 for the same run, digit for digit. So the stock log is sufficient and no
patch is needed to measure context.

Under ``text_memory`` two kinds of request hit the engine per turn:

  main turn  prompt = system + all history notes + current frames/audio
             -> GROWS with turn index
  aux call   prompt = current frames/audio + fixed instruction, NO history
             -> FLAT, by construction

They are separated here by that signature rather than by request id, which the
stats table does not carry. The flat series being flat is itself a result worth
checking: it is what makes the memory-note cost O(1) per turn instead of O(turn).
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TOKENS_IN = re.compile(r"num_tokens_in\s*\|([^\n]*)")
# Log lines are stamped "MM-DD HH:MM:SS" with no year and no sub-second field.
LOGTS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")
TIMING = re.compile(
    r"\[TIMING\] mode=(?P<mode>\w+) total=(?P<total>[-\d.]+)s "
    r"first_text=(?P<ft>[-\d.]+)s first_audio=(?P<fa>[-\d.]+)s "
    r"audio_chunks=(?P<n>\d+)"
)
# Tolerates both formats: the original `dur_ms=... chars=... note=...` and the
# later one that prepends `w0=<epoch> w1=<epoch>` so the note call's GPU work can
# be bracketed out of stage 0. Old runs must stay parseable.
PAMEM = re.compile(
    r"\[PA_MEM\]"
    r"(?: w0=(?P<w0>[\d.]+) w1=(?P<w1>[\d.]+))?"
    r" dur_ms=(?P<dur>[\d.]+) chars=(?P<chars>\d+)"
    r"(?: failed=(?P<failed>\d+))?"
    r" note=(?P<note>.*)"
)


def trace_bounds(path: pathlib.Path) -> tuple[float, float] | None:
    """Wall-clock [first, last] of a client trace, for windowing the server log."""
    if not path.exists():
        return None
    ws = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            ws.append(json.loads(line)["w"])
        except Exception:
            continue
    return (min(ws), max(ws)) if ws else None


def parse_log(path: pathlib.Path, window: tuple[float, float] | None = None) -> dict:
    """Parse the server log, optionally restricted to a wall-clock window.

    Windowing is REQUIRED when more than one bench ran against the same server
    process, which is the case here: the matrix runs recall (EVS off, 8 frames,
    ~7.3k tokens/turn) and then latency (EVS on, talking-head, ~1.3k tokens/turn)
    in one session. Parsing the whole log concatenates the two workloads, and the
    step down between them shows up as a *negative* context slope
    (-540 tokens/turn on the first attempt) -- an artifact, not a measurement.
    """
    if not path.exists():
        return {}
    lines = ANSI.sub("", path.read_text(errors="ignore")).splitlines()

    if window is not None:
        lo, hi = window[0] - 3.0, window[1] + 3.0
        ref = datetime.datetime.fromtimestamp(window[0])
        kept = []
        for ln in lines:
            m = LOGTS.search(ln)
            if m is None:
                kept.append(ln)          # continuation lines (e.g. stats rows)
                continue
            mo, da, hh, mm, ss = (int(x) for x in m.groups())
            try:
                t = datetime.datetime(ref.year, mo, da, hh, mm, ss).timestamp()
            except ValueError:
                kept.append(ln)
                continue
            if lo <= t <= hi:
                kept.append(ln)
        lines = kept
    text = "\n".join(lines)

    tokens: list[int] = []
    for m in TOKENS_IN.finditer(text):
        for cell in m.group(1).split("|"):
            c = cell.strip().replace(",", "")
            if c.isdigit() and int(c) > 0:
                tokens.append(int(c))
                break

    timings = [
        {"total": float(m["total"]), "ft": float(m["ft"]),
         "fa": float(m["fa"]), "n": int(m["n"])}
        for m in TIMING.finditer(text)
    ]
    notes = [
        {"dur_ms": float(m["dur"]), "chars": int(m["chars"]),
         "note": m["note"][:400],
         "w0": float(m["w0"]) if m["w0"] else None,
         "w1": float(m["w1"]) if m["w1"] else None,
         "failed": bool(int(m["failed"])) if m["failed"] else False}
        for m in PAMEM.finditer(text)
    ]
    errors = [
        ln.strip()[:200] for ln in text.splitlines()
        if re.search(r"Query processing failed|longer than the maximum|PA_MEM.*failed", ln)
    ]
    return {"tokens_in": tokens, "timings": timings, "notes": notes, "errors": errors}


def split_series(tokens: list[int], n_notes: int) -> tuple[list[int], list[int], str]:
    """Separate main-turn prompts from memory-note (aux) prompts.

    Under text_memory each turn issues exactly two requests, main first (the
    turn's own generation) and aux second (the note, scheduled from
    on_turn_complete), so the stats rows strictly alternate. Split by parity and
    then *verify* the assumption: the main series must be non-decreasing, since
    history only ever grows. If it is not, the alternation broke (e.g. an aux
    call finished after the next turn's main call) and the split is reported as
    unreliable rather than silently trusted.

    A first attempt classified by "is this value continuing an ascent", which
    failed: in an early run the aux prompts grew too, because the frame buffer
    was accumulating frames, so both series ascended and everything landed in
    one bucket. Parity is structural; ascent is not.
    """
    if n_notes == 0:
        return list(tokens), [], "no aux calls (single request per turn)"
    if len(tokens) != 2 * n_notes:
        return list(tokens), [], (
            f"UNRELIABLE: {len(tokens)} stats rows but {n_notes} notes "
            f"(expected {2 * n_notes}); not split"
        )
    main, aux = tokens[0::2], tokens[1::2]
    if any(b < a for a, b in zip(main, main[1:])):
        return list(tokens), [], "UNRELIABLE: main series not monotonic; not split"
    return main, aux, "split by parity, main series monotonic (verified)"


def slope(vals: list[int]) -> float:
    if len(vals) < 2:
        return 0.0
    n = len(vals)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(vals) / n
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, vals)) / den) if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag=policy pairs, e.g. lm_shipped=shipped lm_tm=text_memory")
    ap.add_argument("--results", default="/data/zx/results")
    ap.add_argument("--out", default="/data/zx/results/longmem_summary.json")
    ap.add_argument("--window", type=int, default=65536,
                    help="model context window, for the survivable-turns estimate")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    summary: dict = {"window": args.window, "arms": {}}

    for spec in args.arms:
        tag, _, policy = spec.partition("=")
        policy = policy or tag
        logp = res / f"server_{tag}.log"
        # The two benches share one server process, so each is parsed inside its
        # own client-trace time window. See parse_log().
        w_recall = trace_bounds(res / f"longmem_{tag}" / "recall.jsonl")
        w_lat = trace_bounds(res / f"longmem_lat_{tag}" / "ttfa_user0.jsonl")
        log = parse_log(logp, w_recall) if w_recall else parse_log(logp)
        log_lat = parse_log(logp, w_lat) if w_lat else {}
        arm: dict = {"tag": tag, "policy": policy,
                     "windowed": {"recall": bool(w_recall), "latency": bool(w_lat)}}

        sc_path = res / f"longmem_{tag}" / "recall_score.json"
        if sc_path.exists():
            blob = json.loads(sc_path.read_text())
            arm["recall"] = blob.get("score", {})
            # The TTFA study found a second-order effect: more context makes the
            # model talk MORE (88 -> 424 chars), which lengthens response-path
            # occupancy, which is exactly what steals other users' latency. So
            # answer length is a cost signal, not a curiosity.
            lens = [len(r.get("answer") or "") for r in blob.get("results", [])
                    if r.get("kind") == "describe"]
            trunc = [len(r.get("done_text") or "") for r in blob.get("results", [])
                     if r.get("kind") == "describe"]
            if lens:
                arm["answer_chars_p50"] = int(st.median(lens))
                arm["answer_chars_max"] = max(lens)
            if trunc and lens:
                # How much of the answer the text.done payload would have lost.
                arm["text_done_kept_frac"] = round(
                    st.median(trunc) / st.median(lens), 3) if st.median(lens) else None

        # --- context + TTFA come from the LATENCY phase -----------------------
        # Same stimulus and knobs as the earlier TTFA study, so these numbers are
        # directly comparable to its L0/L1/L2/L3 rows. Reading them from the
        # recall phase instead would compare against a different workload.
        src = log_lat if log_lat.get("tokens_in") else log
        arm["cost_phase"] = "latency" if src is log_lat else "recall(fallback)"
        if src:
            grow, flat, how = split_series(src["tokens_in"], len(src["notes"]))
            arm["ctx_split_method"] = how
            arm["ctx_first"] = grow[0] if grow else None
            arm["ctx_last"] = grow[-1] if grow else None
            arm["ctx_slope_per_turn"] = round(slope(grow), 1)
            arm["ctx_n_turns_measured"] = len(grow)
            if grow and len(grow) >= 2 and slope(grow) > 0:
                arm["survivable_turns_est"] = round(
                    (args.window - grow[0]) / slope(grow), 1)
            else:
                arm["survivable_turns_est"] = None
            arm["aux_prompt_tokens_p50"] = int(st.median(flat)) if flat else None
            arm["aux_prompt_flat"] = (
                (max(flat) - min(flat)) / st.median(flat) < 0.25 if len(flat) >= 3 else None
            )
            if src["timings"]:
                t = src["timings"][1:] or src["timings"]     # drop warmup turn
                fa = [x["fa"] * 1000 for x in t]
                arm["ttfa_first"] = round(fa[0], 1)
                arm["ttfa_last"] = round(fa[-1], 1)
                arm["ttfa_p50_ms"] = round(st.median(fa), 1)
                arm["ttfa_mean_ms"] = round(st.mean(fa), 1)
                # p50 is a poor summary here: every arm has a STEP partway
                # through (EVS starts retaining a 2nd and 3rd frame), and the
                # step lands one turn earlier in some arms than others, so the
                # median flips from one side of it to the other and manufactures
                # a difference that is not there. The end-of-session plateau is
                # the comparable number.
                arm["ttfa_plateau_ms"] = round(st.mean(fa[-3:]), 1)
                arm["ttfa_slope_ms_per_turn"] = round(slope([int(x) for x in fa]), 1)
                arm["turn_total_p50_s"] = round(st.median([x["total"] for x in t]), 2)
                arm["ttfa_n_turns"] = len(t)
                # Audio steps per turn = how long the response path is occupied.
                # This is the cost that matters for OTHER users: the response
                # path is 76-84% of GPU time (Phase 0), so a model that talks
                # more directly reduces how many sessions fit on the card.
                ch = [x["n"] for x in t]
                arm["audio_steps_first"] = ch[0]
                arm["audio_steps_last"] = ch[-1]
                arm["audio_steps_growth"] = (
                    round(ch[-1] / ch[0], 2) if ch[0] else None)
                arm["turn_total_first_s"] = round(t[0]["total"], 2)
                arm["turn_total_last_s"] = round(t[-1]["total"], 2)

        # --- the CLEAN context cost, from the recall phase --------------------
        # The recall bench runs with EVS off and max_frames == num_frames == 8,
        # so the frame count per turn is constant and the whole slope is history.
        # Verified: the aux-call prompts there vary by 0.2%, which is what a
        # constant frame count looks like. On the latency workload EVS admits a
        # 2nd and 3rd frame partway through, so its slope is not history alone.
        if log.get("tokens_in"):
            g2, a2, how2 = split_series(log["tokens_in"], len(log["notes"]))
            if g2 and len(g2) >= 3:
                arm["ctx_clean_first"] = g2[0]
                arm["ctx_clean_last"] = g2[-1]
                arm["ctx_clean_slope"] = round(slope(g2), 1)
                arm["ctx_clean_split"] = how2
                if slope(g2) > 0:
                    arm["turns_to_window_clean"] = round(
                        (args.window - g2[0]) / slope(g2))
            if a2 and len(a2) >= 3:
                arm["aux_clean_p50"] = int(st.median(a2))
                arm["aux_clean_spread"] = round(
                    (max(a2) - min(a2)) / st.median(a2), 4)

        # --- note cost: whichever phase produced notes ------------------------
        notes = (log.get("notes") or []) + (log_lat.get("notes") or [])
        if notes:
            d = [x["dur_ms"] for x in notes]
            arm["note_n"] = len(d)
            arm["note_dur_p50_ms"] = round(st.median(d), 1)
            arm["note_dur_max_ms"] = round(max(d), 1)
            arm["note_chars_p50"] = int(st.median([x["chars"] for x in notes]))
            arm["note_sample"] = notes[0]["note"]
            # What fraction of a turn's server-side wall time the note adds. It
            # is off the critical path, but it is not free to OTHER users.
            if arm.get("turn_total_p50_s"):
                arm["note_frac_of_turn"] = round(
                    st.median(d) / (arm["turn_total_p50_s"] * 1000), 4)

        arm["errors"] = (log.get("errors") or [])[:2] + (log_lat.get("errors") or [])[:2]
        summary["arms"][tag] = arm

    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- tables --
    arms = summary["arms"]
    print("\n=== DOES IT REMEMBER ===")
    print(f"{'policy':<14}{'scene read':>12}{'word leak':>11}{'first word':>12}"
          f"{'list-all recall':>18}")
    for tag, a in arms.items():
        r = a.get("recall") or {}
        rr = r.get("scene_read_rate") or {}
        wl = r.get("word_leaked_into_answers") or {}
        pf = r.get("probe_first") or {}
        pl = r.get("probe_listall") or {}
        read = f"{rr.get('n_ok','-')}/{rr.get('n','-')}"
        leak = f"{wl.get('n','-')}/{wl.get('of','-')}" if wl else "-"
        first = ("CORRECT" if pf.get("correct") else "wrong") if pf else "-"
        lst = (f"{pl.get('n_recalled')}/{pl.get('n_total')} ({pl.get('frac',0):.0%})"
               if pl else "-")
        print(f"{a['policy']:<14}{read:>12}{leak:>11}{first:>12}{lst:>18}")
    print("  word leak: how often the describe answers named the word anyway. "
          "Nonzero\n  would mean history text carries it and the probe cannot "
          "isolate the note.")

    print("\n=== WHAT IT COSTS: context (latency-phase workload) ===")
    print(f"{'policy':<14}{'ctx first':>10}{'ctx last':>10}{'tok/turn':>10}"
          f"{'turns left':>12}")
    for tag, a in arms.items():
        print(f"{a['policy']:<14}"
              f"{a.get('ctx_first','-') or '-':>10}"
              f"{a.get('ctx_last','-') or '-':>10}"
              f"{a.get('ctx_slope_per_turn','-') or '-':>10}"
              f"{a.get('survivable_turns_est','-') or '-':>12}")
    print("  NOTE: on this workload EVS admits more frames as the session runs,")
    print("  so tok/turn mixes history growth with frame growth. The clean")
    print("  history-only slope is in the recall phase (constant frame count).")

    print("\n=== WHAT IT COSTS: latency on the critical path ===")
    print(f"{'policy':<14}{'turn1':>8}{'turn10':>8}{'plateau':>9}{'p50':>8}"
          f"{'mean':>8}{'ms/turn':>9}")
    for tag, a in arms.items():
        print(f"{a['policy']:<14}"
              f"{a.get('ttfa_first','-') or '-':>8}"
              f"{a.get('ttfa_last','-') or '-':>8}"
              f"{a.get('ttfa_plateau_ms','-') or '-':>9}"
              f"{a.get('ttfa_p50_ms','-') or '-':>8}"
              f"{a.get('ttfa_mean_ms','-') or '-':>8}"
              f"{a.get('ttfa_slope_ms_per_turn','-') or '-':>9}")
    print("  Use plateau, not p50: each arm steps up when EVS admits a 2nd/3rd")
    print("  frame, and the step lands a turn earlier in some arms, so the median")
    print("  flips sides and invents a difference.")

    print("\n=== WHAT IT COSTS: response-path occupancy (hurts OTHER users) ===")
    print(f"{'policy':<14}{'audio steps 1':>15}{'-> 10':>7}{'growth':>9}"
          f"{'turn s 1':>10}{'-> 10':>8}")
    for tag, a in arms.items():
        print(f"{a['policy']:<14}"
              f"{a.get('audio_steps_first','-') or '-':>15}"
              f"{a.get('audio_steps_last','-') or '-':>7}"
              f"{(str(a.get('audio_steps_growth','-')) + 'x'):>9}"
              f"{a.get('turn_total_first_s','-') or '-':>10}"
              f"{a.get('turn_total_last_s','-') or '-':>8}")
    print("  The response path is 76-84% of GPU time, so a model that talks more")
    print("  directly reduces how many sessions fit on the card.")

    print("\n=== WHAT IT COSTS: history alone (recall phase, frame count fixed) ===")
    print(f"{'policy':<14}{'ctx first':>10}{'ctx last':>10}{'tok/turn':>10}"
          f"{'turns to 64k':>14}{'aux spread':>12}")
    for tag, a in arms.items():
        sp = a.get("aux_clean_spread")
        print(f"{a['policy']:<14}"
              f"{a.get('ctx_clean_first','-') or '-':>10}"
              f"{a.get('ctx_clean_last','-') or '-':>10}"
              f"{a.get('ctx_clean_slope','-') or '-':>10}"
              f"{a.get('turns_to_window_clean','-') or '-':>14}"
              f"{(f'{sp:.1%}' if sp is not None else '-'):>12}")
    print("  This is the memory mechanism's own cost. aux spread ~0 confirms the")
    print("  note-generation prompt is O(1) per turn, not O(turn).")

    print("\n=== MEMORY-NOTE COST (text_memory only) ===")
    for tag, a in arms.items():
        if not a.get("note_n"):
            continue
        print(f"  {a['policy']}: n={a['note_n']}  dur p50={a['note_dur_p50_ms']} ms  "
              f"max={a['note_dur_max_ms']} ms  chars p50={a['note_chars_p50']}")
        # Report the CLEAN (fixed-frame-count) aux prompt. The latency-phase
        # figure is not a flatness test: EVS admits more frames partway through,
        # so the aux prompt grows there for a reason unrelated to history.
        sp = a.get("aux_clean_spread")
        print(f"    aux prompt (fixed frames) p50={a.get('aux_clean_p50')} tok, "
              f"spread={sp:.1%}" if sp is not None else
              f"    aux prompt p50={a.get('aux_prompt_tokens_p50')} tok")
        if sp is not None:
            print(f"      -> O(1) per turn: the note call never sees history")
        if a.get("note_frac_of_turn"):
            print(f"    adds {a['note_frac_of_turn']:.1%} to server-side turn time, "
                  f"thinker only (skips the 76-84% response path)")
        print(f"    ctx split: {a.get('ctx_split_method')}")
        print(f"    sample note: {a.get('note_sample')}")

    bad = {t: a["errors"] for t, a in arms.items() if a.get("errors")}
    if bad:
        print("\n=== ERRORS ===")
        for t, e in bad.items():
            print(f"  {t}: {e}")

    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
