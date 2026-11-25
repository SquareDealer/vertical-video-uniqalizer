import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _fake_emotion import FakeEmotionServer  # noqa: E402

from auto_video_editor.emotion import (EmotionServer, EmotionServerUnavailable, EmotionTrack,  # noqa: E402
                                       EmotionWindow, load_cached, save_cached)


def test_url_is_normalised():
    for url in ("localhost:8092", "http://localhost:8092/", "http://localhost:8092/health",
                "http://localhost:8092/v1/audio/emotions"):
        assert EmotionServer(url).url == "http://localhost:8092"


def test_analyze_through_the_server():
    happy = lambda a, b: {"angry": 0.1, "sad": 0.0, "neutral": 0.3, "positive": 0.6}
    with FakeEmotionServer(probs=happy) as fake:
        server = EmotionServer(fake.url)
        assert server.check() == "gigaam-emo (angry, sad, neutral, positive)"
        track = server.analyze(np.zeros(16000 * 6, np.float32))
    assert track.model == "gigaam-emo" and len(track.windows) == 4      # 0-3, 1.5-4.5, 3-6, 4.5-6
    assert track.windows[1].start == 1.5 and track.windows[-1].end == 6.0
    method, path, body = fake.requests[-1]
    assert (method, path) == ("POST", "/v1/audio/emotions")
    assert b'name="window"\r\n\r\n3' in body and b'name="hop"\r\n\r\n1.5' in body and b"RIFF" in body
    assert track.score(6) == pytest.approx([0.65] * 6)


def test_labels_are_matched_without_case_and_required():
    with FakeEmotionServer(labels=("Angry", "Sad", "Neutral", "Positive"),
                           probs=lambda a, b: {"Angry": 0.0, "Sad": 0.0, "Neutral": 0.2, "Positive": 0.8}) as fake:
        track = EmotionServer(fake.url).analyze(np.zeros(16000 * 3, np.float32))
        assert track.per_second("positive", 3) == pytest.approx([0.8] * 3)
    with FakeEmotionServer(labels=("happy", "calm")) as fake:
        with pytest.raises(RuntimeError, match="positive"):
            EmotionServer(fake.url).check()


def test_per_second_weights_by_overlap_and_leaves_uncovered_seconds_at_zero():
    track = EmotionTrack("m", ["angry", "positive"], [
        EmotionWindow(0.0, 1.5, {"positive": 1.0, "angry": 0.0}),
        EmotionWindow(1.5, 3.0, {"positive": 0.0, "angry": 1.0})])
    assert track.per_second("positive", 5) == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0])
    assert track.score(5) == pytest.approx([1.0, 0.75, 0.5, 0.0, 0.0])
    assert track.shifted(10).windows[0].start == 10.0
    assert EmotionTrack("m", ["angry", "positive"], []).score(3) == pytest.approx([0, 0, 0])


def test_down_or_loading_server_is_unavailable_and_other_errors_are_not():
    with pytest.raises(EmotionServerUnavailable, match="Cannot reach the emotion server"):
        EmotionServer("http://127.0.0.1:9", timeout=2).check()
    with FakeEmotionServer(health_status=503, post_status=503) as fake:
        with pytest.raises(EmotionServerUnavailable):
            EmotionServer(fake.url).check()
        with pytest.raises(EmotionServerUnavailable):
            EmotionServer(fake.url).analyze(np.zeros(16000, np.float32))
    with FakeEmotionServer(post_status=400) as fake:
        with pytest.raises(RuntimeError, match="400") as caught:
            EmotionServer(fake.url).analyze(np.zeros(16000, np.float32))
        assert not isinstance(caught.value, EmotionServerUnavailable)


def test_cache_round_trip(tmp_path):
    track = EmotionTrack("gigaam-emo", ["angry", "positive"], [EmotionWindow(0.0, 3.0, {"angry": 0.1, "positive": 0.9})])
    save_cached(tmp_path / "e" / "x.json", track)
    assert load_cached(tmp_path / "e" / "x.json") == track
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert load_cached(tmp_path / "bad.json") is None and load_cached(tmp_path / "missing.json") is None
