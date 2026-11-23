"""Whole-clip plan: one short = the clip without dead air (auto mode for TikTok)."""

import shutil
import subprocess

import pytest

from auto_video_editor.ffmpeg import probe
from auto_video_editor.pipeline import Pipeline
from auto_video_editor.reporting import Reporter
from auto_video_editor.settings import Settings
from auto_video_editor.timeline import total_duration
from auto_video_editor.transcribe import SpeechServer, SpeechServerUnavailable

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _video(path, duration, bursts=(), audio=True):
    volume = "+".join(f"between(t,{a},{b})" for a, b in bursts) or "0"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={duration}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={duration}",
                "-filter_complex", f"[1:a]volume='if({volume},1.0,0.02)':eval=frame[a]", "-map", "0:v", "-map", "[a]",
                "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


def _pipeline(path, tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr("auto_video_editor.pipeline.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    settings = Settings(transcriber="none", whole_clip=True, shorts_jump_cuts=True, shorts_layout="crop",
                        encoder="libx264", quality="draft", use_cache=False, **overrides)
    return Pipeline(str(path), settings, str(tmp_path / "out"), Reporter(lambda _m: None, lambda *_: None))


def test_whole_clip_gives_one_short_without_dead_air(tmp_path, monkeypatch):
    video = _video(tmp_path / "talk.mp4", 30, bursts=[(0, 4), (9, 20)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert project.highlight == []
    assert len(project.shorts) == 1
    short = project.shorts[0]
    assert short.keep, "dead air must be cut"
    assert total_duration(short.keep) < 22
    assert short.keep[0][0] < 0.5


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
    video = _video(tmp_path / "long.mp4", 200, bursts=[(120, 126)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()
    assert len(project.shorts) == 1
    short = project.shorts[0]
    assert short.duration <= 90.5
    assert short.start <= 123 <= short.end, "the fragment holds the loud part"


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
    assert 13 <= short.peak <= 18, "the hook must come from the best seconds, not the middle"
