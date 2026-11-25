"""The funny moment edit of a whole clip: lead-in, the joke, the reaction, and no dead weight.

Pieces are ``(source_start, source_end, speed)``. This module plans them; ``Renderer`` plays them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .audio import AudioEnvelope
from .glossary import is_laughter
from .moments import normalize_text
from .timeline import Piece, Range, merge_ranges, subtract_ranges

POSITIVE_THRESHOLD = 0.5   # P(positive) from which a second counts as laughter
WORD_PAD = 0.1             # seconds kept around every word
CUT_MARGIN = 0.15          # seconds kept at the edges of a cut
MIN_PIECE = 0.25           # shorter kept ranges are dropped
MIN_FAST = 0.5             # shorter parts of a wordless passage are not sped up


def _span(word) -> Tuple[float, float]:
    if isinstance(word, (list, tuple)):
        return float(word[0]), float(word[1])
    return float(word.start), float(word.end)


def _text(word) -> str:
    if isinstance(word, (list, tuple)):
        return str(word[2]) if len(word) > 2 else ""
    return str(word.text)


def laughter_ranges(words: Sequence) -> List[Range]:
    """The seconds ``build_curve`` marks around every laughter word, merged."""
    found: List[Range] = []
    for word in words:
        if is_laughter(normalize_text(_text(word))):
            start, end = _span(word)
            found.append((float(max(0, int(start) - 1)), float(int(end) + 3)))
    return merge_ranges(found)


def protected_ranges(words: Sequence, positive: Optional[np.ndarray]) -> List[Range]:
    """Laughter plus every second with ``P(positive) >= POSITIVE_THRESHOLD``: never cut, never sped up."""
    found = list(laughter_ranges(words))
    if positive is not None:
        found += [(float(i), float(i + 1)) for i in np.flatnonzero(np.asarray(positive) >= POSITIVE_THRESHOLD)]
    return merge_ranges(found)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """``[first, end)`` frame indices of the runs of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def plan_pieces(start: float, end: float, env: AudioEnvelope, words: Sequence, min_silence: float,
                threshold: Optional[float] = None, *, protected: Sequence[Range] = (), speech_rule: bool = False,
                quiet_cut_max: float = 3.0, quiet_speed: float = 2.0) -> List[Piece]:
    """Cut dead air and short wordless passages, speed up long ones.

    Without ``speech_rule`` and ``protected`` this is ``compute_keep_ranges`` (speed 1.0 everywhere).
    Words keep ``WORD_PAD`` around them and ``protected`` ranges (laughter, positive voice) are untouched.
    """
    threshold = env.silence_threshold() if threshold is None else threshold
    hop = env.hop
    a, b = env.frame(start), env.frame(end)
    if b - a < 2:
        return [(start, end, 1.0)]
    count = b - a
    quiet = env.rms_db[a:b] < threshold
    speech = np.zeros(count, dtype=bool)
    for word in words:
        ws, we = _span(word)
        if we < start or ws > end:
            continue
        speech[max(0, env.frame(ws - WORD_PAD) - a): max(0, env.frame(we + WORD_PAD) - a + 1)] = True
    guard = np.zeros(count, dtype=bool)
    for ps, pe in protected:
        first = int(np.clip(np.floor((ps - start) / hop + 1e-9), 0, count))
        last = int(np.clip(np.ceil((pe - start) / hop - 1e-9), 0, count))
        guard[first:last] = True
    free = ~speech & ~guard

    def frame_time(index: int) -> float:
        return start + index * hop

    def cut_of(t0: float, t1: float) -> Optional[Range]:
        cut_start = t0 + (CUT_MARGIN if t0 > start + hop else 0.0)
        cut_end = t1 - (CUT_MARGIN if t1 < end - hop else 0.0)
        return (cut_start, cut_end) if cut_end - cut_start > 0.1 else None

    # Dead air: quiet and free, longer than min_silence: the old rule.
    dead: List[Tuple[float, float, Optional[Range]]] = []
    for first, last in _runs(quiet & free):
        t0, t1 = frame_time(first), min(end, frame_time(last))
        if t1 - t0 >= min_silence:
            dead.append((t0, t1, cut_of(t0, t1)))
    cuts: List[Range] = [c for _, _, c in dead if c is not None]
    fast: List[Range] = []

    if speech_rule:
        cuts = []
        for first, last in _runs(free):
            g0, g1 = frame_time(first), min(end, frame_time(last))
            inside = [d for d in dead if d[0] >= g0 - 1e-9 and d[1] <= g1 + 1e-9]
            live = (g1 - g0) - sum(d[1] - d[0] for d in inside)
            if min_silence - 1e-6 <= live <= quiet_cut_max + 1e-6:
                whole = cut_of(g0, g1)
                if whole is not None:
                    cuts.append(whole)
                continue
            cuts += [d[2] for d in inside if d[2] is not None]
            if live > quiet_cut_max + 1e-6:
                fast.append((g0, g1))
        # Dead runs that are not part of any free gap cannot exist (quiet & free is inside free).

    kept = [r for r in subtract_ranges((start, end), cuts) if r[1] - r[0] >= MIN_PIECE]
    kept = merge_ranges(kept) or [(start, end)]
    if not fast:
        return [(s, e, 1.0) for s, e in kept]

    pieces: List[Piece] = []
    for s, e in kept:
        points = sorted({s, e, *[p for gap in fast for p in gap if s < p < e]})
        for lo, hi in zip(points, points[1:]):
            mid = (lo + hi) / 2.0
            in_fast = any(g0 <= mid <= g1 for g0, g1 in fast)
            speed = quiet_speed if in_fast and hi - lo >= MIN_FAST else 1.0
            if pieces and pieces[-1][2] == speed and abs(pieces[-1][1] - lo) < 1e-9:
                pieces[-1] = (pieces[-1][0], hi, speed)
            else:
                pieces.append((lo, hi, speed))
    return pieces
