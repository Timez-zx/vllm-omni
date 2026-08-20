#!/usr/bin/env python3
"""Pure playback-timeline metrics shared by live-agent benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlaybackReport:
    start_s: float | None
    stalls_s: tuple[float, ...]

    @property
    def stall_total_s(self) -> float:
        return sum(self.stalls_s)

    @property
    def stall_max_s(self) -> float:
        return max(self.stalls_s, default=0.0)


def simulate_playback(
    deltas: list[tuple[float, int]],
    *,
    sample_rate: int = 24000,
    prebuffer_s: float = 0.06,
    release_at_s: float | None = None,
) -> PlaybackReport:
    """Replay a 1x player against ``(arrival_s, samples)`` deltas.

    Playback begins at the arrival that first takes queued audio over the
    prebuffer threshold. ``release_at_s`` models ``response.audio.done``:
    the browser releases a short reply even when its total audio never reaches
    the smooth-start threshold. A later delta creates a listener-visible stall
    only when it arrives after all previously delivered samples would have played.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if prebuffer_s < 0:
        raise ValueError("prebuffer_s must be non-negative")
    if any(samples < 0 for _, samples in deltas):
        raise ValueError("sample counts must be non-negative")
    if any(b[0] < a[0] for a, b in zip(deltas, deltas[1:])):
        raise ValueError("delta arrivals must be monotonic")

    queued_s = 0.0
    start_index: int | None = None
    for index, (_, samples) in enumerate(deltas):
        queued_s += samples / sample_rate
        if queued_s >= prebuffer_s:
            start_index = index
            break
    if start_index is None:
        if not deltas or release_at_s is None:
            return PlaybackReport(start_s=None, stalls_s=())
        if release_at_s < deltas[-1][0]:
            raise ValueError("release must not precede the final delta")
        return PlaybackReport(start_s=release_at_s, stalls_s=())

    start_s = deltas[start_index][0]
    covered_until = start_s + queued_s
    stalls: list[float] = []
    for arrival_s, samples in deltas[start_index + 1 :]:
        if arrival_s > covered_until:
            stalls.append(arrival_s - covered_until)
            covered_until = arrival_s
        covered_until += samples / sample_rate
    return PlaybackReport(start_s=start_s, stalls_s=tuple(stalls))
