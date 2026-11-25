"""Three TikTok versions of one clip: same cuts, different looks."""

import hashlib
import json
import shutil
import subprocess

import pytest

from auto_video_editor.captions import TimedWord
from auto_video_editor.ffmpeg import probe
from auto_video_editor.pipeline import Pipeline
from auto_video_editor.reporting import Reporter
from auto_video_editor.settings import Settings
from auto_video_editor.variants import VariantSpec, speed_words, variant_presets


def test_presets_never_crop_with_facecam():
    with_cam = variant_presets(3, True)
    assert [s.index for s in with_cam] == [1, 2, 3]
    assert {s.layout for s in with_cam} <= {"split", "blur"}
    assert with_cam[2].layout == "split" and with_cam[2].split_top == 720
    # a typical 0.25 x 0.30 webcam of a 16:9 frame (480x324, aspect 1.48) loses < 5 % in a 1080x720 band
    band_aspect, cam_aspect = 1080 / with_cam[2].split_top, 480 / 324
    assert abs(1 - cam_aspect / band_aspect) < 0.05
    assert variant_presets(3, False)[2].layout == "smart"
    assert variant_presets(3, False)[0].layout == "blur"


def test_every_version_is_split_with_a_webcam():
    specs = variant_presets(3, True)
    assert [s.layout for s in specs] == ["split", "split", "split"]
    assert len({s.split_top for s in specs}) == 3, "the webcam band differs between versions"


def test_every_version_has_its_own_caption_color():
    specs = variant_presets(3, True)
    colors = [s.caption_text_color.upper() for s in specs]
    assert len(set(colors)) == 3 and "#FFFFFF" not in colors
    assert [s.caption_text_color for s in specs] == [s.caption_text_color for s in variant_presets(3, False)]


def test_presets_follow_the_spec_table():
    v1, v2, v3 = variant_presets(3, True)
    assert all("hook" not in s.to_dict() for s in (v1, v2, v3)), "no version opens with a teaser"
    assert (v1.speed, v2.speed, v3.speed) == (1.0, 1.02, 0.98)
    assert (v1.zoom, v2.zoom, v3.zoom) == (1.0, 1.03, 1.05)
    assert v2.shift < 0 < v3.shift
    assert (v1.caption_position, v2.caption_position, v3.caption_position) == ("bottom", "middle", "top")
    assert (v1.caption_style, v3.caption_style) == ("karaoke", "simple")
    assert len({s.effects for s in (v1, v2, v3)}) == 3 and len({s.gop for s in (v1, v2, v3)}) == 3
    json.dumps(v2.to_dict())


def test_more_than_three_presets_stay_distinct():
    specs = variant_presets(5, False)
    assert [s.index for s in specs] == [1, 2, 3, 4, 5]
    assert len({(s.layout, s.shift, s.effects) for s in specs}) == 5


def test_speed_words_divides_times():
    words = [TimedWord(1.02, 2.04, "привет")]
    assert speed_words(words, 1.02)[0].start == pytest.approx(1.0)
    assert speed_words(words, 1.02)[0].end == pytest.approx(2.0)
    assert speed_words(words, 1.0) == words


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_three_variants_differ(tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=20",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
        "-filter_complex", "[0:v]drawbox=x=0:y=0:w=160:h=108:color=red:t=fill[v];"
                           "[1:a]volume='if(between(t,0,6)+between(t,10,18),1.0,0.02)':eval=frame[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-metadata", "creation_time=2020-01-01T00:00:00Z", "-shortest", str(src)], check=True)
    monkeypatch.setattr("auto_video_editor.pipeline.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    settings = Settings(transcriber="none", whole_clip=True, shorts_jump_cuts=True, shorts_hook=True,
                        facecam="0,0,0.25,0.3", encoder="libx264", quality="draft", use_cache=False)
    pipeline = Pipeline(str(src), settings, str(tmp_path / "out"), Reporter(lambda _m: None, lambda *_: None))
    project = pipeline.analyze()
    outputs = []
    specs = variant_presets(3, True)
    seen = []
    original = Pipeline._render_one_short
    short = project.shorts[0]
    short.hook = [(12.0, 15.0)]

    def spy(self, project, m, path, renderer, work, ranges, captions_name, layout, *args, **kwargs):
        seen.append((layout, self.settings.caption_text_color, self.settings.caption_speaker_colors))
        assert ranges[0][0] == pytest.approx(short.pieces(True)[0][0]), "no version opens with a teaser"
        return original(self, project, m, path, renderer, work, ranges, captions_name, layout, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "_render_one_short", spy)
    edited = short.edited_duration(True)
    expected = [edited / spec.speed for spec in specs]
    for spec in specs:
        path, captions = pipeline.render_variant(project, spec, tmp_path / "out" / f"v{spec.index}.mp4")
        assert captions is False, "no transcript, no captions"
        outputs.append(path)
    durations = []
    for path in outputs:
        info = probe(str(path))
        assert (info.width, info.height) == (1080, 1920)
        durations.append(info.duration)
        tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
                              capture_output=True, text=True, check=True).stdout
        assert "2020-01-01" not in tags, "source metadata must not leak into a version"
    for got, want in zip(durations, expected):
        assert got == pytest.approx(want, abs=0.2), "same cuts in every version, only the speed differs"
    hashes = {hashlib.md5(p.read_bytes()).hexdigest() for p in outputs}
    assert len(hashes) == 3
    assert [row[0] for row in seen] == ["split"] * 3
    assert [row[1] for row in seen] == [spec.caption_text_color for spec in specs]
    assert not any(row[2] for row in seen), "speaker colors would override the version color"
    assert pipeline.settings.caption_text_color == "#FFFFFF", "user settings restored"


def test_top_captions_stay_below_the_webcam_band():
    from auto_video_editor.captions import build_ass
    from auto_video_editor.render import caption_top_margin

    assert caption_top_margin("split", 720) == 760
    assert caption_top_margin("blur", 720) is None
    words = [TimedWord(0.0, 1.0, "рот")]
    ass = build_ass(words, position="top", top_margin=760)
    style = next(line for line in ass.splitlines() if line.startswith("Style:"))
    assert style.split(",")[-2] == "760" and style.split(",")[18] == "8", "top-aligned, 760 px from the top"
    default = next(line for line in build_ass(words, position="top").splitlines() if line.startswith("Style:"))
    assert default.split(",")[-2] == str(int(1920 * 0.14)), "other layouts keep the usual top margin"


def test_caption_text_color():
    from auto_video_editor.captions import build_ass

    words = [TimedWord(0.0, 1.0, "рот"), TimedWord(1.0, 2.0, "кот")]
    for style in ("karaoke", "simple"):
        ass = build_ass(words, style=style, text_color="#39FF88", highlight="#FFFFFF")
        style_line = next(line for line in ass.splitlines() if line.startswith("Style:"))
        assert style_line.split(",")[3] == "&H0088FF39", style
    default = next(line for line in build_ass(words).splitlines() if line.startswith("Style:"))
    assert default.split(",")[3] == "&H00FFFFFF"
