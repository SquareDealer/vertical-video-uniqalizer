"""The funny moment edit through the pipeline: emotions, lead-in, speed-ups, rendering."""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _clips import _pipeline, _video  # noqa: E402
from _fake_emotion import FakeEmotionServer  # noqa: E402
from _fake_speech import FakeSpeechServer  # noqa: E402

from auto_video_editor.emotion import EmotionServerUnavailable  # noqa: E402
from auto_video_editor.ffmpeg import probe  # noqa: E402
from auto_video_editor.pipeline import Project  # noqa: E402
from auto_video_editor.variants import variant_presets  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SEGMENTS = [(1.0, 9.0, "so we are walking here and nothing happens at all"),
            (10.0, 19.0, "i think we should go left because the right side is closed"),
            (20.0, 27.0, "wait wait what is that thing over there guys"),
            (28.0, 31.0, "it is a chicken with a gun"),
            (40.0, 43.0, "okay that was funny")]            # 9 s without words after the joke


def test_whole_clip_gets_a_lead_in_emotions_and_speed_ups(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 46, bursts=[(0, 46)])          # loud everywhere
    positive = lambda a, b: ({"angry": 0, "sad": 0, "neutral": 0.1, "positive": 0.9} if 31 <= a < 34
                             else {"angry": 0, "sad": 0, "neutral": 1.0, "positive": 0})
    with FakeSpeechServer(SEGMENTS) as stt, FakeEmotionServer(probs=positive) as emo:
        pipeline = _pipeline(video, tmp_path, monkeypatch, transcriber="http", stt_url=stt.url,
                             emotion_url=emo.url, trigger_offset=38.0, language="ru", diarize=False,
                             auto_vocabulary=False)
        project = pipeline.analyze()
        short = project.shorts[0]
        edit = short.edit
        assert edit["trigger_offset"] == 38.0 and edit["search_window"] == [8.0, 41.0]
        assert 28 <= edit["peak"] <= 36 and edit["anchor"] in (28.0, pytest.approx(edit["peak"] - 2, abs=0.01))
        assert edit["lead_in"] >= 10.0 and not edit["lead_in_short"] and short.start == pytest.approx(edit["start"], abs=0.01)
        assert edit["emotion"]["model"] == "gigaam-emo" and edit["emotion"]["peak_positive"] > 0.5
        assert edit["speedups"], "the wordless passage after the reaction is sped up"
        assert all(a >= 34.9 for a, b, s in edit["speedups"]), "seconds with a positive voice are not sped up"
        assert short.hook == []
        posts = [r for r in emo.requests if r[0] == "POST"]
        project2 = _pipeline(video, tmp_path, monkeypatch, transcriber="http", stt_url=stt.url,
                             emotion_url=emo.url, trigger_offset=38.0, use_cache=True, language="ru",
                             diarize=False, auto_vocabulary=False).analyze()
        assert len([r for r in emo.requests if r[0] == "POST"]) == len(posts), "the second run reads the cache"
        assert project2.shorts[0].edit == edit
        out, captions = pipeline.render_variant(project, variant_presets(1, False)[0], tmp_path / "v1.mp4")
    assert captions and probe(str(out)).duration == pytest.approx(short.edited_duration(True), abs=0.4)
    assert Project.load(project.path).shorts[0].pieces(True) == short.pieces(True)


def test_without_an_emotion_address_nothing_is_sent(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 20, bursts=[(14, 17)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()            # transcriber none, emotion_url ""
    assert project.shorts[0].edit["emotion"] is None and project.shorts[0].edit["speedups"] == []


def test_emotion_server_down_stops_the_analysis(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 8, bursts=[(0, 8)])
    with pytest.raises(EmotionServerUnavailable):
        _pipeline(video, tmp_path, monkeypatch, emotion_url="http://127.0.0.1:9").analyze()
