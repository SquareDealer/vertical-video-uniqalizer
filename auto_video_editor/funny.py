"""The funny moment edit of a whole clip: lead-in, the joke, the reaction, and no dead weight.

Pieces are ``(source_start, source_end, speed)``. This module plans them; ``Renderer`` plays them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .audio import AudioEnvelope
from .glossary import is_laughter
from .moments import normalize_text
from .timeline import Piece, Range, merge_ranges, pieces_duration, subtract_ranges
from .transcribe import Transcript

POSITIVE_THRESHOLD = 0.5   # P(positive) from which a second counts as laughter
WORD_PAD = 0.1             # seconds kept around every word
CUT_MARGIN = 0.15          # seconds kept at the edges of a cut
MIN_PIECE = 0.25           # shorter kept ranges are dropped
MIN_FAST = 0.5             # shorter parts of a wordless passage are not sped up
SEARCH_BEFORE = 30.0       # the chat reacts late: the joke is up to this long before the trigger...
SEARCH_AFTER = 3.0         # ...and at most this long after it
ANCHOR_REACH = 3.0         # a phrase ending this close before the peak is the joke itself
ANCHOR_FALLBACK = 2.0      # no speech near the peak: the joke starts this long before it
MIN_OUTPUT = 5.0           # less than this is left after the cuts: play the clip uncut


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


# -- peak, anchor, lead-in ---------------------------------------------------------------------

def search_window(duration: float, trigger: Optional[float]) -> Tuple[float, float]:
    """Where the funny moment is searched: before the chat trigger, or in the whole clip without one."""
    if trigger is None:
        return 0.0, float(duration)
    return max(0.0, trigger - SEARCH_BEFORE), min(float(duration), trigger + SEARCH_AFTER)


def find_peak(score: np.ndarray, window: Tuple[float, float], duration: float) -> float:
    """Middle of the best second inside ``window`` (the earliest on a tie)."""
    n = int(np.asarray(score).size)
    if n == 0:
        return duration / 2.0
    lo = min(n - 1, max(0, int(np.floor(window[0]))))
    hi = min(n, max(lo + 1, int(np.ceil(window[1]))))
    best = lo + int(np.argmax(np.asarray(score)[lo:hi]))
    return min(best + 0.5, float(duration))


def find_anchor(peak: float, segments: Sequence) -> float:
    """Start of the joke: the phrase around the peak, or the one that ended just before it."""
    containing = [seg for seg in segments if seg.start <= peak <= seg.end]
    if containing:
        return float(max(seg.start for seg in containing))
    before = [seg for seg in segments if seg.end < peak and peak - seg.end <= ANCHOR_REACH]
    if before:
        return float(max(before, key=lambda seg: seg.end).start)
    return max(0.0, peak - ANCHOR_FALLBACK)


def output_between(pieces: Sequence[Piece], a: float, b: float) -> float:
    """Seconds of finished video the pieces give between source times ``a`` and ``b``."""
    total = 0.0
    for start, end, speed in pieces:
        lo, hi = max(start, a), min(end, b)
        if hi > lo:
            total += (hi - lo) / speed
    return total


def clip_pieces(pieces: Sequence[Piece], a: float, b: float) -> List[Piece]:
    result: List[Piece] = []
    for start, end, speed in pieces:
        lo, hi = max(start, a), min(end, b)
        if hi - lo > 1e-9:
            result.append((lo, hi, speed))
    return result


def choose_start(pieces: Sequence[Piece], anchor: float, segments: Sequence, words: Sequence,
                 lead_in_min: float, lead_in_max: float) -> Tuple[float, bool]:
    """Where the video starts: ``lead_in_min`` seconds of finished video before the joke, on a phrase.

    Returns ``(start, lead_in_short)``; ``lead_in_short`` when the file has less lead-in than that.
    """
    first = pieces[0][0]
    if output_between(pieces, first, anchor) < lead_in_min:
        return first, True
    remaining, t = lead_in_min, first
    for start, end, speed in reversed(pieces):
        top = min(end, anchor)
        if top <= start:
            continue
        seconds = (top - start) / speed
        if seconds >= remaining:
            t = top - remaining * speed
            break
        remaining -= seconds
    phrase = next((seg for seg in segments if seg.start < t < seg.end), None)
    if phrase is not None and output_between(pieces, phrase.start, anchor) <= lead_in_max:
        t = float(phrase.start)
    else:
        word = next((w for w in words if _span(w)[0] < t < _span(w)[1]), None)
        if word is not None:
            t = _span(word)[0]
    return max(first, t), False


@dataclass
class EditPlan:
    start: float
    end: float
    peak: float
    anchor: float
    pieces: List[Piece] = field(default_factory=list)     # empty = play start..end uncut
    lead_in: float = 0.0
    lead_in_short: bool = False
    search_window: Tuple[float, float] = (0.0, 0.0)
    trigger_offset: Optional[float] = None
    cuts: int = 0
    speedups: List[Piece] = field(default_factory=list)
    emotion: Optional[dict] = None

    def to_dict(self) -> dict:
        """The ``edit`` field of ``manifest.json``: why the video starts and ends where it does."""
        return {
            "peak": round(self.peak, 2), "anchor": round(self.anchor, 2),
            "start": round(self.start, 2), "end": round(self.end, 2),
            "lead_in": round(self.lead_in, 2), "lead_in_short": bool(self.lead_in_short),
            "search_window": [round(self.search_window[0], 2), round(self.search_window[1], 2)],
            "trigger_offset": None if self.trigger_offset is None else round(self.trigger_offset, 2),
            "cuts": int(self.cuts),
            "speedups": [[round(a, 2), round(b, 2), round(sp, 2)] for a, b, sp in self.speedups],
            "emotion": self.emotion,
        }


def _snap_end(end: float, words: Sequence) -> float:
    """An end that lands inside a word moves to the start of that word."""
    for word in words:
        ws, we = _span(word)
        if ws < end < we:
            return ws
    return end


def plan_edit(*, duration: float, score: np.ndarray, env: AudioEnvelope, transcript: Optional[Transcript],
              positive: Optional[np.ndarray] = None, emotion_model: str = "", trigger: Optional[float] = None,
              jump_cuts: bool = True, min_silence: float = 0.8, threshold: Optional[float] = None,
              lead_in_min: float = 10.0, lead_in_max: float = 20.0, quiet_cut_max: float = 3.0,
              quiet_speed: float = 2.0, long_after: float = 180.0, long_cut: float = 90.0) -> EditPlan:
    """Plan the whole-clip edit: lead-in -> joke -> reaction, with dead weight cut or sped up."""
    if trigger is not None and not 0.0 <= trigger <= duration:
        trigger = None
    window = search_window(duration, trigger)
    peak = find_peak(score, window, duration)
    segments = list(transcript.segments) if transcript else []
    words = list(transcript.words()) if transcript else []
    anchor = find_anchor(peak, segments)

    protected = protected_ranges(words, positive)
    # The reaction stays even when it has no recognized words (laughter is rarely written down).
    tail = peak + 3.0
    for ps, pe in protected:
        if ps <= tail <= pe:
            tail = pe
    end0 = min(duration, max(max(_span(w)[1] for w in words) + 0.5, tail)) if words else float(duration)

    uncut = [(0.0, end0, 1.0)]
    if jump_cuts:
        all_pieces = plan_pieces(0.0, end0, env, words, min_silence, threshold, protected=protected,
                                 speech_rule=bool(words), quiet_cut_max=quiet_cut_max, quiet_speed=quiet_speed)
    else:
        all_pieces = uncut

    def trim(source: List[Piece]) -> Tuple[float, float, bool]:
        start, short = choose_start(source, anchor, segments, words, lead_in_min, lead_in_max)
        end = end0
        if end - start > long_after:
            end = _snap_end(min(end0, max(start + long_cut, peak + 3.0)), words)
        return round(start, 3), round(end, 3), short   # milliseconds, as project.json stores them

    def clipped(source: List[Piece], a: float, b: float) -> List[Piece]:
        rounded = [(round(lo, 3), round(hi, 3), speed) for lo, hi, speed in clip_pieces(source, a, b)]
        return [p for p in rounded if p[1] > p[0]]

    start, end, short = trim(all_pieces)
    pieces = clipped(all_pieces, start, end)
    if pieces_duration(pieces) < min(MIN_OUTPUT, end0):
        start, end, short = trim(uncut)    # too little left after the cuts: play the clip uncut
        pieces = []
    elif pieces and pieces[0][0] > start:
        start = pieces[0][0]               # the phrase start fell into a cut: the video starts at its sound

    lead_in = output_between(pieces or [(start, end, 1.0)], start, anchor)
    cuts = sum(1 for prev, nxt in zip(pieces, pieces[1:]) if nxt[0] - prev[1] > 1e-6)
    emotion = None
    if positive is not None:
        index = int(peak)
        value = float(positive[index]) if 0 <= index < len(positive) else 0.0
        emotion = {"model": emotion_model, "peak_positive": round(value, 2)}
    return EditPlan(start=start, end=end, peak=peak, anchor=anchor, pieces=pieces, lead_in=lead_in,
                    lead_in_short=short, search_window=window, trigger_offset=trigger, cuts=cuts,
                    speedups=[p for p in pieces if p[2] != 1.0], emotion=emotion)
