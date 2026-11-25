"""Peak, joke anchor and lead-in: where the funny-moment video starts and ends."""

import json

import numpy as np
import pytest
from conftest import make_envelope, make_transcript

from auto_video_editor.funny import find_anchor, find_peak, output_between, plan_edit, search_window


def score_with(n, **peaks):
    score = np.zeros(n, np.float32)
    for second, value in peaks.items():
        score[int(second)] = value
    return score


TALK = [(0, 9.5, "so we are walking here and nothing happens at all right now"),
        (10, 19.5, "i think we should go left because the right side is closed"),
        (20, 27.5, "wait wait what is that thing over there guys"),
        (28, 31.5, "it is a chicken with a gun"),                                   # the joke
        (32, 36, "hahaha no way")]


def plan(score, transcript=None, **kw):
    env = make_envelope(duration=60, base=-20)
    return plan_edit(duration=60.0, score=score, env=env, transcript=transcript, **kw)


def test_peak_is_searched_before_the_trigger():
    score = score_with(60, **{"12": 0.9, "30": 0.6, "50": 1.0})
    assert find_peak(score, search_window(60, None), 60) == 50.5
    assert search_window(60, 37.0) == (7.0, 40.0)
    assert find_peak(score, search_window(60, 37.0), 60) == 12.5
    assert find_peak(score_with(60, **{"20": 0.5, "30": 0.5}), (0, 60), 60) == 20.5, "the earliest on a tie"
    assert search_window(60, 10.0) == (0.0, 13.0) and search_window(60, 59.0) == (29.0, 60.0)


def test_anchor_is_the_start_of_the_joke_phrase():
    segments = make_transcript(TALK).segments
    assert find_anchor(30.5, segments) == 28           # inside the joke
    assert find_anchor(33.5, segments) == 32           # inside the next phrase
    assert find_anchor(31.8, segments) == 28           # just after it, within 3 s
    assert find_anchor(50.5, segments) == 48.5         # no speech near: peak - 2
    assert find_anchor(1.0, []) == 0.0


def test_lead_in_is_at_least_ten_seconds_of_finished_video_and_starts_on_a_phrase():
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(TALK))
    assert (p.peak, p.anchor) == (30.5, 28.0)
    assert p.start == 10.0, "pulled back to the start of the phrase"
    assert 10.0 <= p.lead_in <= 20.0 and not p.lead_in_short
    assert p.lead_in == pytest.approx(output_between(p.pieces, p.start, p.anchor), abs=0.01)
    assert p.end == pytest.approx(36.45, abs=0.01), "last word + 0.5 s: the reaction stays"
    assert p.pieces[0][0] == p.start


def test_lead_in_counts_finished_video_not_source_seconds():
    talk = [(0, 6, "one two three four five six"), (16, 22, "seven eight nine ten eleven twelve"),
            (24, 27.5, "it is a chicken with a gun")]
    env = make_envelope(duration=60, base=-20, quiet=[(6.3, 15.7)])      # 9.4 s of dead air inside the lead-in
    p = plan_edit(duration=60.0, score=score_with(60, **{"25": 1.0}), env=env, transcript=make_transcript(talk))
    assert p.anchor == 24.0 and p.start == 0.0, "14 source seconds before the joke are not 10 s of video"
    assert p.lead_in == pytest.approx(13.35, abs=0.2) and not p.lead_in_short   # 6.45 + 6.65 + 0.25 s of pieces


def test_wordless_laugh_after_the_last_word_stays_in_the_video():
    positive = np.zeros(60, np.float32)
    positive[32:36] = 0.9
    p = plan(score_with(60, **{"33": 1.0}), make_transcript(TALK[:4]), positive=positive)
    assert p.anchor == 28.0 and p.end == pytest.approx(36.5, abs=0.06), "the reaction is not cut off after the last word"
    assert p.start <= p.peak <= p.end
    assert not any(s != 1.0 and a < 36 and b > 32 for a, b, s in p.pieces)


def test_long_phrase_start_lands_on_a_word():
    talk = [(0, 27.5, " ".join(f"w{i}" for i in range(55))), (28, 31.5, "it is a chicken with a gun")]
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(talk), lead_in_max=20.0)
    words = [w.start for w in make_transcript(talk).words()]
    assert p.start in [pytest.approx(w) for w in words], "the phrase is longer than lead_in_max: a word border"
    assert 10.0 <= p.lead_in <= 20.0


def test_joke_at_the_very_beginning_gives_a_short_lead_in():
    talk = [(1, 4.5, "it is a chicken with a gun"), (5, 9, "hahaha no way")]
    p = plan(score_with(60, **{"3": 1.0}), make_transcript(talk))
    assert p.lead_in_short and p.start == pytest.approx(p.pieces[0][0]) and p.start <= 1.0
    assert p.lead_in < 10.0


def test_trigger_outside_the_clip_is_ignored():
    score = score_with(60, **{"12": 0.9, "50": 1.0})
    assert plan(score, trigger=400.0).trigger_offset is None and plan(score, trigger=400.0).peak == 50.5
    assert plan(score, trigger=-5.0).search_window == (0.0, 60.0)
    assert plan(score, trigger=37.0).peak == 12.5 and plan(score, trigger=37.0).trigger_offset == 37.0


def test_no_words_means_only_dead_air_is_cut():
    env = make_envelope(duration=60, base=-20, quiet=[(40, 45)])
    p = plan_edit(duration=60.0, score=score_with(60, **{"30": 1.0}), env=env, transcript=make_transcript([]))
    assert p.speedups == [] and p.cuts == 1, "a recognizer that heard nothing must not speed the clip up"
    assert p.anchor == 28.5 and p.end == 60.0


def test_long_clip_is_checked_after_trimming_the_start():
    env = make_envelope(duration=200, base=-20)
    late = plan_edit(duration=200.0, score=score_with(200, **{"120": 1.0}), env=env, transcript=None)
    assert late.start == pytest.approx(108.5) and late.end == 200.0, "92 s left: not a long clip any more"
    env = make_envelope(duration=400, base=-20)
    long = plan_edit(duration=400.0, score=score_with(400, **{"120": 1.0}), env=env, transcript=None)
    assert long.start == pytest.approx(108.5) and long.end == pytest.approx(198.5), "at most 90 s of source"


def test_less_than_five_seconds_left_plays_uncut():
    env = make_envelope(duration=20, base=-20, quiet=[(0, 17)])
    p = plan_edit(duration=20.0, score=score_with(20, **{"18": 1.0}), env=env, transcript=None)
    assert p.pieces == [] and p.cuts == 0 and p.speedups == []
    assert (p.start, p.end) == (6.5, 20.0), "the lead-in is counted on the uncut clip then"


def test_plan_dict_has_the_manifest_fields():
    positive = np.zeros(60, np.float32)
    positive[30] = 0.82
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(TALK), positive=positive, emotion_model="gigaam-emo", trigger=37.0)
    data = p.to_dict()
    assert set(data) == {"peak", "anchor", "start", "end", "lead_in", "lead_in_short", "search_window",
                         "trigger_offset", "cuts", "speedups", "emotion"}
    assert data["search_window"] == [7.0, 40.0] and data["trigger_offset"] == 37.0
    assert data["emotion"] == {"model": "gigaam-emo", "peak_positive": 0.82}
    assert plan(score_with(60, **{"30": 1.0})).to_dict()["emotion"] is None
    json.dumps(data)
