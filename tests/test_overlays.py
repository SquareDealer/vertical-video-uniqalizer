"""Branding overlays: a different divider or banner in each TikTok version."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _render_cases import INFO, build  # noqa: E402

from auto_video_editor import overlays  # noqa: E402
from auto_video_editor.ffmpeg import MediaInfo  # noqa: E402
from auto_video_editor.render import Renderer, caption_top_margin, seam_y  # noqa: E402
from auto_video_editor.settings import Settings  # noqa: E402
from auto_video_editor.variants import variant_presets  # noqa: E402


def test_each_version_gets_its_own_overlay():
    for has_facecam in (True, False):
        assert [s.overlay for s in variant_presets(3, has_facecam)] == ["neon", "ticker", "banner"]
    assert [s.overlay for s in variant_presets(5, True)] == ["neon", "ticker", "banner", "neon", "ticker"]
    assert {s.overlay for s in variant_presets(3, True, overlays=False)} == {""}


def test_dividers_sit_on_the_seam():
    neon = overlays.place("neon", 640, animated=True)
    ticker = overlays.place("ticker", 680, animated=True)
    assert (neon.kind, neon.y, neon.height, neon.loop) == ("neon", 570, 140, True)
    assert (ticker.kind, ticker.y, ticker.height, ticker.loop) == ("ticker", 630, 100, True)
    assert neon.path.endswith("neon.webm") and neon.animated


def test_banner_has_a_fixed_place_and_plays_once():
    banner = overlays.place("banner", 720, animated=True)
    assert (banner.y, banner.loop, banner.start) == (1130, False, overlays.BANNER_START)
    assert overlays.place("banner", None, animated=True).y == 1130


def test_divider_without_a_seam_becomes_the_banner():
    assert overlays.place("neon", None, animated=True).kind == "banner"
    assert overlays.place("ticker", 120, animated=True).kind == "banner", "seam under the TikTok header"
    assert overlays.place("", 640) is None
    with pytest.raises(ValueError):
        overlays.place("confetti", 640)


def test_still_png_without_libvpx():
    neon = overlays.place("neon", 640, animated=False)
    assert neon.path.endswith("neon.png") and not neon.animated
    assert neon.input_args()[:4] == ["-loop", "1", "-framerate", "30"]
    banner = overlays.place("banner", None, animated=False)
    assert "enable='between(t,2,7)'" in banner.filter("[a]", 3, "[b]")


def test_input_args_decode_alpha():
    neon = overlays.place("neon", 640, animated=True)
    args = neon.input_args()
    assert args[:2] == ["-stream_loop", "-1"]
    assert args[args.index("-i") - 2:args.index("-i")] == ["-c:v", "libvpx-vp9"], "alpha needs libvpx-vp9"
    banner = overlays.place("banner", None, animated=True).input_args()
    assert "-stream_loop" not in banner and banner[banner.index("-itsoffset") + 1] == "2"


def test_seam_of_each_layout():
    assert seam_y("split", INFO, 680) == 680
    assert seam_y("blur", INFO) == (1920 - 608) // 2, "16:9 picture centered in the blur layout"
    assert seam_y("smart", INFO) is None
    tall = MediaInfo(path="in.mp4", duration=10.0, has_video=True, has_audio=True, width=1080, height=1920,
                     fps=30.0, audio_tracks=["1"])
    assert seam_y("blur", tall) is None, "a vertical source fills the frame"


def test_overlay_is_drawn_under_the_captions():
    neon = overlays.place("neon", 640, animated=True)
    args = build("split", overlay=neon)
    graph = args[args.index("-filter_complex") + 1]
    inputs = [i for i, a in enumerate(args) if a == "-i"]
    assert args[inputs[-1] + 1] == neon.path
    index = len(inputs) - 1
    assert f"[vlay][{index}:v]overlay=0:570:shortest=1[vovr]" in graph
    assert graph.index("[vovr]") < graph.index("ass=c.ass")
    assert "-map" in args and args[args.index("-map") + 1] == "[vout]"


def test_overlay_input_index_without_audio():
    silent = MediaInfo(path="in.mp4", duration=60.0, has_video=True, has_audio=False, width=1920, height=1080,
                       fps=30.0, audio_tracks=[])
    banner = overlays.place("banner", None, animated=True)
    args = Renderer(silent, "libx264", "standard").build_segment_command(
        [(1.0, 5.0)], "out.mp4", layout="blur", overlay=banner)
    graph = args[args.index("-filter_complex") + 1]
    assert "[vlay][2:v]overlay=0:1130:eof_action=pass[vovr]" in graph, "after the clip and the silent track"


def test_top_captions_clear_a_divider():
    assert caption_top_margin("split", 720) == 760
    assert caption_top_margin("split", 720, overlays.place("neon", 720, animated=True)) == 790
    assert caption_top_margin("split", 720, overlays.place("banner", 720, animated=True)) == 760
    assert caption_top_margin("smart", 720, overlays.place("banner", None, animated=True)) is None


def test_custom_overlay_folder_is_checked(tmp_path):
    assert overlays.check_dir(str(overlays.default_dir())) is None
    settings = Settings(watch_overlay_dir=str(tmp_path))
    assert any("overlay.json" in e for e in settings.validate())
    assert not any("overlay" in e for e in Settings(watch_overlay_dir=str(tmp_path), watch_overlays=False).validate())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("animated", [True, False])
def test_banner_shows_only_while_it_plays(tmp_path, animated):
    if animated and not overlays.alpha_decoder():
        pytest.skip("ffmpeg has no libvpx-vp9")
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=green:size=640x360:rate=30:duration=10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)], check=True)
    info = MediaInfo(path=str(src), duration=10.0, has_video=True, has_audio=False, width=640, height=360,
                     fps=30.0, audio_tracks=[])
    out = tmp_path / "out.mp4"
    Renderer(info, "libx264", "draft").render_segment(
        [(0.0, 10.0)], out, layout="smart", overlay=overlays.place("banner", None, animated=animated))

    def band(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                              "-vf", "crop=1080:240:0:1130,format=gray", "-f", "rawvideo", "-"],
                             capture_output=True, check=True).stdout
        return raw

    def difference(a, b):  # mean absolute difference of two gray pictures
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    clean, shown, gone = band(0.5), band(4.5), band(9.0)
    assert difference(clean, gone) < 2, "nothing is left after the banner"
    assert difference(clean, shown) > 10, "the banner is on screen in the middle of its play"
