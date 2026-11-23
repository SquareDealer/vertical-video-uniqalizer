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
from auto_video_editor.timeline import total_duration
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


def test_presets_follow_the_spec_table():
    v1, v2, v3 = variant_presets(3, True)
    assert (v1.hook, v2.hook, v3.hook) == (False, True, False)
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
    settings = Settings(transcriber="none", whole_clip=True, shorts_jump_cuts=True, facecam="0,0,0.25,0.3",
                        encoder="libx264", quality="draft", use_cache=False)
    pipeline = Pipeline(str(src), settings, str(tmp_path / "out"), Reporter(lambda _m: None, lambda *_: None))
    project = pipeline.analyze()
    outputs = []
    specs = variant_presets(3, True)
    short = project.shorts[0]
    edited = total_duration(short.ranges(True))
    expected = [(edited + (total_duration(short.hook) if spec.hook else 0.0)) / spec.speed for spec in specs]
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
        assert got == pytest.approx(want, abs=0.2), "same cuts in every version, only hook and speed differ"
    hashes = {hashlib.md5(p.read_bytes()).hexdigest() for p in outputs}
    assert len(hashes) == 3
