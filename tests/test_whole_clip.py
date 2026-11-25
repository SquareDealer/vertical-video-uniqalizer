"""Whole-clip plan: one short = the clip without dead air (auto mode for TikTok)."""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _clips import _pipeline, _video  # noqa: E402

from auto_video_editor.ffmpeg import probe  # noqa: E402
from auto_video_editor.pipeline import Pipeline  # noqa: E402
from auto_video_editor.timeline import total_duration  # noqa: E402
from auto_video_editor.transcribe import SpeechServer, SpeechServerUnavailable

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_whole_clip_gives_one_short_without_dead_air(tmp_path, monkeypatch):
    video = _video(tmp_path / "talk.mp4", 30, bursts=[(0, 4), (9, 20)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert project.highlight == []
    assert len(project.shorts) == 1
    short = project.shorts[0]
    assert short.keep, "dead air must be cut"
    assert total_duration(short.keep) < 22
    assert short.edit["lead_in"] == pytest.approx(10.0, abs=0.1), "the video starts a lead-in before the joke"
    assert short.keep[0][0] == pytest.approx(short.edit["start"], abs=0.01)


def test_whole_clip_without_audio(tmp_path, monkeypatch):
    video = _video(tmp_path / "mute.mp4", 8, audio=False)
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert len(project.shorts) == 1
    assert project.shorts[0].start == 0.0
    assert project.shorts[0].end == pytest.approx(probe(str(video)).duration, abs=0.1)


def test_whole_clip_tiny_clip_renders(tmp_path, monkeypatch):
    video = _video(tmp_path / "tiny.mp4", 5, bursts=[(0, 5)])
    pipeline = _pipeline(video, tmp_path, monkeypatch, shorts_count=1)
    project = pipeline.analyze()
    assert len(project.shorts) == 1
    result = pipeline.render(project)
    assert result.shorts and result.shorts[0].exists()


def test_long_clip_is_cut_to_best_fragment(tmp_path, monkeypatch):
    video = _video(tmp_path / "long.mp4", 400, bursts=[(120, 126)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert len(project.shorts) == 1
    short = project.shorts[0]
    assert short.duration <= 90.5
    assert short.start <= 123 <= short.end, "the fragment holds the loud part"
    assert short.edit["lead_in"] >= 0


def test_clip_that_is_short_after_trimming_keeps_its_end(tmp_path, monkeypatch):
    video = _video(tmp_path / "late.mp4", 200, bursts=[(120, 126)])
    duration = probe(str(video)).duration
    short = _pipeline(video, tmp_path, monkeypatch).analyze().shorts[0]
    assert short.end == pytest.approx(duration, abs=0.1)
    assert short.duration < 180


def test_long_clip_without_peaks_falls_back_to_its_start(tmp_path, monkeypatch):
    video = _video(tmp_path / "flat.mp4", 190, audio=False)
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert len(project.shorts) == 1
    assert project.shorts[0].start == 0.0 and project.shorts[0].duration == pytest.approx(90.0)


def test_speech_server_down_raises_unavailable():
    with pytest.raises(SpeechServerUnavailable):
        SpeechServer("http://127.0.0.1:9", timeout=2).check()


def test_whole_clip_peak_is_the_loudest_part(tmp_path, monkeypatch):
    video = _video(tmp_path / "peak.mp4", 20, bursts=[(14, 17)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    short = project.shorts[0]
    assert 13 <= short.peak <= 18, "the peak is the best second, not the middle"


def test_full_mode_keeps_the_whole_long_clip(tmp_path, monkeypatch):
    video = _video(tmp_path / "long.mp4", 24, bursts=[(2, 5), (15, 18)])
    duration = probe(str(video)).duration
    project = _pipeline(video, tmp_path, monkeypatch, whole_clip_max=10, whole_clip_cut=6,
                        whole_clip_full=True).analyze()
    short = project.shorts[0]
    assert (short.start, short.end) == (0.0, pytest.approx(duration, abs=0.1)), "montage-mode full: never trimmed"
    assert short.edit is None


def test_full_mode_does_not_trim_around_the_words():
    from auto_video_editor.captions import TimedWord

    class Words:
        def words(self):
            return [TimedWord(3.0, 4.0, "а"), TimedWord(10.0, 11.0, "б")]

    full = Pipeline._whole_clip_moment(Words(), 20.0, full=True)
    assert (full.start, full.end) == (0.0, 20.0)
    cut = Pipeline._whole_clip_moment(Words(), 20.0)
    assert (cut.start, cut.end) == (pytest.approx(2.7), pytest.approx(11.5))


def test_render_variant_uses_the_stream_area(tmp_path, monkeypatch):
    from auto_video_editor.variants import variant_presets

    video = _video(tmp_path / "clip.mp4", 6, bursts=[(0, 6)])
    pipeline = _pipeline(video, tmp_path, monkeypatch, facecam="0.7,0.65,0.275,0.3056",
                         stream_area="0,0,1,0.62")
    project = pipeline.analyze()
    seen = []
    monkeypatch.setattr(Pipeline, "_render_one_short",
                        lambda self, *a, **k: seen.append(k) or ("split", None, []))
    pipeline.render_variant(project, variant_presets(1, True)[0], tmp_path / "v1.mp4")
    assert seen[0]["game_area"] == (0.0, 0.0, 1.0, 0.62)
