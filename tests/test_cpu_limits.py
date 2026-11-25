"""CPU limits of ffmpeg processes: thread count and a lower process priority (auto mode)."""

import os

from auto_video_editor import ffmpeg
from auto_video_editor.ffmpeg import limit_command, resource_limits, spawn_options

CMD = ["C:/tools/ffmpeg.exe", "-y", "-ss", "1", "-i", "a.mp4", "-i", "b.wav", "-c:v", "libx264", "out.mp4"]


def test_no_limits_by_default():
    assert limit_command(CMD) == CMD
    assert spawn_options().get("creationflags", 0) & 0x4000 == 0


def test_threads_reach_decoders_filters_and_encoder():
    with resource_limits(4, False):
        cmd = limit_command(CMD)
    assert cmd[:5] == [CMD[0], "-filter_threads", "4", "-filter_complex_threads", "4"]
    assert cmd.count("-threads") == 3
    for i, arg in enumerate(cmd):
        if arg == "-i":
            assert cmd[i - 2:i] == ["-threads", "4"], "each input decoded with 4 threads"
    assert cmd[-3:] == ["-threads", "4", "out.mp4"], "the encoder too"
    assert limit_command(["ffprobe", "-i", "a.mp4"]) == ["ffprobe", "-i", "a.mp4"], "only ffmpeg"


def test_low_priority_and_restore():
    with resource_limits(2, True):
        options = spawn_options()
        if os.name == "nt":
            assert options["creationflags"] & 0x4000, "BELOW_NORMAL_PRIORITY_CLASS"
        else:
            assert callable(options["preexec_fn"])
    assert ffmpeg._LIMITS == {"threads": 0, "low_priority": False}


def test_run_ffmpeg_applies_the_limits(monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise OSError("not started")

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", fake_popen)
    with resource_limits(2, True):
        try:
            ffmpeg.run_ffmpeg(["-i", "a.mp4", "out.mp4"])
        except OSError:
            pass
        try:
            ffmpeg.popen_binary([ffmpeg.ffmpeg_bin(), "-i", "a.mp4", "-"])
        except OSError:
            pass
    assert len(calls) == 2
    for cmd, kwargs in calls:
        assert "-threads" in cmd
        if os.name == "nt":
            assert kwargs["creationflags"] & 0x4000
