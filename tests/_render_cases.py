"""Shared render cases: the snapshot in tests/data/render_snapshot.json was built from these
with the renderer as it was before the variation knobs, so default arguments must not change."""

from auto_video_editor.ffmpeg import MediaInfo
from auto_video_editor.render import Renderer

INFO = MediaInfo(path="in.mp4", duration=60.0, has_video=True, has_audio=True, width=1920, height=1080,
                 fps=59.94, audio_tracks=["1"])
CASES = {
    "horizontal": dict(ranges=[(1.0, 5.0), (7.0, 12.0)], kwargs=dict(fade=0.3)),
    "blur": dict(ranges=[(1.0, 5.0)], kwargs=dict(layout="blur", captions_file="c.ass", audio_filter="volume=1")),
    "split": dict(ranges=[(2.0, 4.0), (5.0, 9.0)],
                  kwargs=dict(layout="split", facecam=(0.0, 0.0, 0.25, 0.3), captions_file="c.ass")),
    "smart": dict(ranges=[(20.0, 23.0), (1.0, 4.0)], kwargs=dict(layout="smart", subject_x=0.3)),
}


def build(name, **extra):
    case = CASES[name]
    renderer = Renderer(INFO, "libx264", "standard")
    return renderer.build_segment_command(case["ranges"], "out.mp4", **case["kwargs"], **extra)
