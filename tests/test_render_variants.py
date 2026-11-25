"""Variation knobs of the renderer: zoom/shift of the game area, speed, effects, extra output args."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _render_cases import CASES, INFO, build  # noqa: E402

from auto_video_editor.ffmpeg import probe  # noqa: E402
from auto_video_editor.render import Renderer, zoom_window  # noqa: E402

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "render_snapshot.json").read_text())


@pytest.mark.parametrize("name", sorted(CASES))
def test_defaults_unchanged(name):
    assert build(name) == SNAPSHOT[name]


def test_zoom_window_size_and_shift():
    w, h, x, y = zoom_window(1920, 1080, 1.05, 0.0)
    assert (w, h) == (1828, 1028) and w % 2 == 0 and h % 2 == 0
    assert x == (1920 - w) // 2 // 2 * 2 or abs(x - (1920 - w) / 2) <= 2
    assert zoom_window(1920, 1080, 1.05, -1.0)[2] == 0
    assert zoom_window(1920, 1080, 1.05, 1.0)[2] == 1920 - w
    assert zoom_window(1920, 1080, 1.0, 0.7) == (1920, 1080, 0, 0)


def test_zoom_window_keeps_facecam():
    cam = (0.0, 0.0, 0.25, 0.3)  # top-left webcam
    for shift in (-1.0, 0.0, 1.0):
        w, h, x, y = zoom_window(1920, 1080, 1.05, shift, keep=cam)
        assert x <= 0 and y <= 0, f"webcam cut with shift {shift}"
    right_cam = (0.75, 0.7, 0.25, 0.3)  # bottom-right webcam
    w, h, x, y = zoom_window(1920, 1080, 1.05, -1.0, keep=right_cam)
    assert x + w >= 1920 and y + h >= 1080


def test_effects_and_extra_args_present():
    args = build("blur", video_effects="noise=alls=3:allf=t", extra_output_args=["-g", "48"])
    graph = args[args.index("-filter_complex") + 1]
    assert graph.index("noise=alls=3:allf=t") < graph.index("ass=c.ass")
    assert args[-3:] == ["-g", "48", "out.mp4"]


def test_speed_and_zoom_reach_the_graph():
    args = build("split", speed=1.03, game_zoom=1.04, game_shift=0.5, split_top=820)
    graph = args[args.index("-filter_complex") + 1]
    assert "setpts=PTS/1.03" in graph and "atempo=1.03" in graph
    assert ":820" in graph, "camera height follows split_top"
    assert args[args.index("-t", len(args) - 12) + 1] == f"{6.0 / 1.03:.3f}"


def _lavfi_source(tmp_path):
    """A 6 s test clip (video + 440 Hz tone) and its media info."""
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-preset",
                    "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    return probe(str(src))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_speed_changes_duration(tmp_path):
    info = _lavfi_source(tmp_path)
    out = tmp_path / "out.mp4"
    Renderer(info, "libx264", "draft").render_segment([(0.0, 6.0)], out, layout="blur", speed=1.03,
                                                      game_zoom=1.03, game_shift=-0.6)
    result = probe(str(out))
    assert (result.width, result.height) == (1080, 1920)
    assert result.duration == pytest.approx(6.0 / 1.03, abs=0.12)


def test_stream_area_is_the_game_picture_of_the_split_layout():
    # INFO is 1920x1080; the Swiper drew the game in the top 62 % of the frame
    args = build("split", game_area=(0.0, 0.0, 1.0, 0.62))
    graph = args[args.index("-filter_complex") + 1]
    game = graph[graph.index(";[gamesrc]") + 1:]
    assert game.startswith("[gamesrc]crop=1920:670:0:0,"), "the game is cut from the stream area first"
    cam = graph[graph.index(";[camsrc]"):graph.index(";[gamesrc]")]
    assert "670" not in cam, "the webcam band is not affected"
    zoomed = build("split", game_area=(0.1, 0.1, 0.5, 0.5), game_zoom=1.05)
    graph = zoomed[zoomed.index("-filter_complex") + 1]
    assert "[gamesrc]crop=960:540:192:108,crop=" in graph, "the zoom works inside the stream area"


def test_stream_area_in_the_blur_layout():
    args = build("blur", game_area=(0.0, 0.0, 1.0, 0.62))
    graph = args[args.index("-filter_complex") + 1]
    assert "[fgsrc]crop=1920:670:0:0," in graph


def test_piece_speed_reaches_the_graph():
    renderer = Renderer(INFO, "libx264", "standard")
    args = renderer.build_segment_command([(1.0, 5.0), (7.0, 13.0, 2.0)], "out.mp4")
    graph = args[args.index("-filter_complex") + 1]
    assert "trim=start=1.000:end=5.000,setpts=PTS-STARTPTS[v0]" in graph, "normal pieces as before"
    assert "setpts=(PTS-STARTPTS)/2[v1]" in graph
    assert "asetpts=PTS-STARTPTS,atempo=2,afade=t=in" in graph
    assert f"afade=t=out:st={3.0 - 0.02:.3f}" in graph, "the fade is placed in sped-up time"
    assert args[args.index("-t", len(args) - 12) + 1] == "7.000"
    single = renderer.build_segment_command([(1.0, 5.0, 2.0)], "out.mp4")
    graph = single[single.index("-filter_complex") + 1]
    assert "setpts=(PTS-STARTPTS)/2[vcat]" in graph and "asetpts=PTS-STARTPTS,atempo=2[acat]" in graph
    with pytest.raises(ValueError):
        renderer.build_segment_command([(1.0, 5.0, 8.0)], "out.mp4")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_sped_up_piece_shortens_the_file(tmp_path):
    info = _lavfi_source(tmp_path)
    out = tmp_path / "out.mp4"
    Renderer(info, "libx264", "draft").render_segment([(0.0, 2.0), (2.0, 6.0, 2.0)], out, layout="blur")
    assert probe(str(out)).duration == pytest.approx(4.0, abs=0.15)
