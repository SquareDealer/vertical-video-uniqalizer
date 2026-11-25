"""Sliding windows over an audio of a given length."""

from typing import List, Tuple

MIN_LAST_WINDOW = 0.5   # a shorter tail window is dropped (unless it is the only one)


def window_bounds(duration: float, window: float, hop: float) -> List[Tuple[float, float]]:
    """``[k * hop, min(k * hop + window, duration)]`` for every ``k * hop < duration``."""
    if duration <= 0:
        return []
    bounds: List[Tuple[float, float]] = []
    k = 0
    while k * hop < duration:
        start = k * hop
        bounds.append((start, min(start + window, duration)))
        k += 1
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_LAST_WINDOW:
        bounds.pop()
    return bounds
