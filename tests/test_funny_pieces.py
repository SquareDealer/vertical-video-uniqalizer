"""Pieces of a clip: dead air, wordless passages and what is protected from both."""

import numpy as np
import pytest
from conftest import make_envelope

from auto_video_editor.funny import laughter_ranges, plan_pieces, protected_ranges
from auto_video_editor.moments import compute_keep_ranges

WORDS = [[5.0, 9.9, "hello there"], [12.0, 14.0, "again"], [20.0, 25.0, "the end"]]   # gaps: 2 s and 6 s


def test_same_as_the_old_rule_when_the_wordless_rule_is_off():
    env = make_envelope(duration=40, base=-20, quiet=[(10, 13), (16, 16.4), (30, 38)])
    words = [[11.0, 12.0, "hey"], [20.0, 22.0, "more words"]]
    old = compute_keep_ranges(5, 40, env, words, 0.8)
    assert [(a, b) for a, b, _ in plan_pieces(5, 40, env, words, 0.8)] == old
    assert all(speed == 1.0 for _, _, speed in plan_pieces(5, 40, env, words, 0.8))


def test_two_second_wordless_gap_is_cut_and_six_second_gap_is_sped_up():
    env = make_envelope(duration=30, base=-20)                      # loud everywhere, never quiet
    pieces = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    assert pieces == [(5, pytest.approx(10.2, abs=0.06), 1.0), (pytest.approx(11.75, abs=0.06), pytest.approx(14.1, abs=0.06), 1.0),
                      (pytest.approx(14.1, abs=0.06), pytest.approx(19.9, abs=0.06), 2.0), (pytest.approx(19.9, abs=0.06), 25, 1.0)]
    assert plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=False) == [(5, 25, 1.0)], "old rule: loud passages stay"


def test_words_are_never_cut_or_sped_up():
    env = make_envelope(duration=30, base=-20, quiet=[(14.5, 19.5)])
    for a, b, speed in plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True):
        if speed != 1.0:
            assert all(we + 0.1 <= a + 0.051 or ws - 0.1 >= b - 0.051 for ws, we, _ in WORDS)
    kept = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    for ws, we, _ in WORDS:
        assert any(a <= ws and b >= we and s == 1.0 for a, b, s in kept), "every word whole, at normal speed"


def test_laughter_and_positive_are_protected_from_both_rules():
    env = make_envelope(duration=30, base=-20, quiet=[(15, 19)])    # dead air inside the 6 s gap
    free = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    assert not any(a <= 15.5 and b >= 18.5 for a, b, _ in free), "dead air is cut"
    guarded = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True, protected=[(14.0, 20.0)])
    assert any(a <= 14.1 and b >= 19.9 and s == 1.0 for a, b, s in guarded), "kept whole, at normal speed"
    assert guarded[1][0] == pytest.approx(11.75, abs=0.06), "the 2 s gap outside the protection is still cut"


def test_dead_air_inside_a_long_gap_is_cut_and_the_rest_sped_up():
    env = make_envelope(duration=40, base=-20, quiet=[(18, 22)])
    words = [[5.0, 9.9, "start"], [30.0, 35.0, "finish"]]            # 20 s gap: 16 s live + 4 s dead
    pieces = plan_pieces(5, 35, env, words, 0.8, speech_rule=True)
    assert [s for _, _, s in pieces] == [1.0, 2.0, 2.0, 1.0]
    assert pieces[1][1] == pytest.approx(18.15, abs=0.06) and pieces[2][0] == pytest.approx(21.85, abs=0.06)


def test_gap_shorter_than_min_silence_is_untouched_and_gaps_at_the_edges_have_no_margin():
    env = make_envelope(duration=30, base=-20)
    words = [[7.0, 10.0, "a"], [10.6, 20.0, "b"]]                    # 0.4 s between words; 2 s before, 5 s after
    pieces = plan_pieces(5, 25, env, words, 0.8, speech_rule=True)
    assert pieces == [(pytest.approx(6.75, abs=0.06), pytest.approx(20.1, abs=0.06), 1.0),
                      (pytest.approx(20.1, abs=0.06), 25, 2.0)]


def test_protected_ranges_from_laughter_and_positive():
    words = [[3.2, 3.9, "hahaha"], [10.0, 11.0, "ok"]]
    positive = np.zeros(20, np.float32)
    positive[14:16] = 0.7
    positive[16] = 0.49
    assert laughter_ranges(words) == [(2, 6)]
    assert protected_ranges(words, positive) == [(2, 6), (14, 16)]
    assert protected_ranges(words, None) == [(2, 6)]
