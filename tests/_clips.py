"""Synthetic clips and pipelines shared by the whole-clip tests."""

import subprocess

from auto_video_editor.pipeline import Pipeline
from auto_video_editor.reporting import Reporter
from auto_video_editor.settings import Settings


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


def _pipeline(path, tmp_path, monkeypatch, use_cache=False, **overrides):
    monkeypatch.setattr("auto_video_editor.pipeline.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    options = dict(transcriber="none", whole_clip=True, shorts_jump_cuts=True, shorts_layout="crop",
                   encoder="libx264", quality="draft", use_cache=use_cache)
    options.update(overrides)
    settings = Settings(**options)
    return Pipeline(str(path), settings, str(tmp_path / "out"), Reporter(lambda _m: None, lambda *_: None))
