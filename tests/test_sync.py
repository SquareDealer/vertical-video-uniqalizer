"""Audio/video sync of rendered clips on recordings that are hard to cut: variable frame rate video and
audio starting later than the video. A white flash and a beep happen together every 2 seconds."""

import shutil
import subprocess

import numpy as np
import pytest

from auto_video_editor.audio import AudioEnvelope, analyze_audio, media_fingerprint
from auto_video_editor.ffmpeg import probe
from auto_video_editor.captions import remap_words_sequence
from auto_video_editor.pipeline import Pipeline
from auto_video_editor.render import Renderer
from auto_video_editor.reporting import Reporter
from auto_video_editor.settings import Settings
from auto_video_editor.transcribe import SAMPLE_RATE, _decode_pcm

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FLASH = "color=c=black:s=160x90:r=60:d=40,drawbox=c=white:t=fill:enable='lt(mod(t\\,2)\\,0.1)'"
BEEP = "sine=f=1000:r=48000:d=40,volume='if(lt(mod(t\\,2)\\,0.1)\\,1\\,0)':eval=frame"
ENCODE = ["-c:v", "libx264", "-preset", "ultrafast", "-g", "120", "-c:a", "aac"]


def _make(path, video_filter="", audio_delay=0.0, audio_filter=""):
    video = FLASH + video_filter
    audio = BEEP + (f",atrim=start={audio_delay},asetpts=PTS-STARTPTS" if audio_delay else "") + audio_filter
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", video]
    if audio_delay:
        cmd += ["-itsoffset", str(audio_delay)]
    cmd += ["-f", "lavfi", "-i", audio, "-fps_mode", "passthrough", *ENCODE, str(path)]
    subprocess.run(cmd, check=True)
    return path


def _onsets(values, times, threshold):
    above = values > threshold
    return [times[i] for i in range(len(values)) if above[i] and (i == 0 or not above[i - 1])]


def _sync_errors(path):
    info = probe(str(path))
    frames = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=8:8,format=gray",
                             "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    luma = np.frombuffer(frames, np.uint8).reshape(-1, 64).mean(axis=1)
    flashes = _onsets(luma, np.arange(len(luma)) / info.fps, 128)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    samples = np.frombuffer(pcm, np.float32)
    window = 240  # 5 ms
    rms = np.sqrt((samples[: len(samples) // window * window].reshape(-1, window) ** 2).mean(axis=1))
    beeps = _onsets(rms, np.arange(len(rms)) * window / 48000, 0.05)
    assert len(flashes) >= 3 and abs(len(flashes) - len(beeps)) <= 1, (flashes, beeps)
    return [min(abs(b - f) for b in beeps) for f in flashes]


@pytest.mark.parametrize("kind", ["vfr", "late_audio"])
def test_clips_stay_in_sync(tmp_path, kind):
    if kind == "vfr":  # dropped frames, as in OBS recordings under load or phone videos
        source = _make(tmp_path / "vfr.mp4", video_filter=",select='not(between(mod(n\\,113)\\,3\\,40))"
                                                              "+lt(mod(t\\,2)\\,0.1)'")
    else:  # audio stream starting 0.4 s after the video, as in Twitch VODs (MPEG-TS)
        source = _make(tmp_path / "late.ts", audio_delay=0.4)
    renderer = Renderer(probe(str(source)), "libx264", "draft")
    # A clip starting just before a flash, and one with a jump cut and a teaser going back in time.
    for ranges in ([(7.3, 13.2)], [(30.3, 34.9), (12.2, 15.5), (17.1, 21.9)]):
        out = tmp_path / "clip.mp4"
        renderer.render_segment(ranges, out)
        errors = _sync_errors(out)
        assert max(errors) < 0.06, f"{kind} {ranges}: audio/video offsets {[round(e, 3) for e in errors]}"


def test_captions_follow_sped_up_pieces():
    words = [[1.0, 1.4, "раз"], [9.0, 9.4, "два"]]
    timed = remap_words_sequence(words, [(0.0, 2.0), (2.0, 8.0, 2.0), (8.0, 10.0)])
    assert [(round(w.start, 2), round(w.end, 2)) for w in timed] == [(1.0, 1.4), (6.0, 6.4)]
    inside = remap_words_sequence([[4.0, 5.0, "быстро"]], [(0.0, 2.0), (2.0, 8.0, 2.0)])
    assert (inside[0].start, inside[0].end) == (pytest.approx(3.0), pytest.approx(3.5))


# Twitch captures lose pieces of audio: the timestamps keep running, the samples are missing.
HOLES = {"one_hole": ",aselect='not(between(t,2.3,3.5))'", "many_holes": ",aselect='mod(n,25)'"}


def _beep_errors(onsets):
    """Distance of every beep from the even second it was recorded at."""
    assert len(onsets) >= 15, onsets
    return [abs(t - 2 * round(t / 2)) for t in onsets]


@pytest.mark.parametrize("kind", ["one_hole", "many_holes", "late_audio"])
def test_analysis_audio_is_on_the_render_clock(tmp_path, kind):
    """Words, loudness and emotions are timed from this audio; the cuts and captions use the render clock."""
    if kind == "late_audio":
        source = _make(tmp_path / "late.ts", audio_delay=0.4)
    else:
        source = _make(tmp_path / f"{kind}.mp4", audio_filter=HOLES[kind])
    info = probe(str(source))
    for start in (0.0, 11.0):   # the whole file, and a chunk that starts in the middle (long recordings)
        samples = _decode_pcm(info.path, start, 40.0 - start, None)
        window = SAMPLE_RATE // 200
        rms = np.sqrt((samples[: len(samples) // window * window].reshape(-1, window) ** 2).mean(axis=1))
        onsets = [start + t for t in _onsets(rms, np.arange(len(rms)) * window / SAMPLE_RATE, 0.05)]
        errors = _beep_errors(onsets) if start == 0.0 else [abs(t - 2 * round(t / 2)) for t in onsets]
        assert max(errors) < 0.1, f"{kind} from {start}: offsets {[round(e, 2) for e in errors]}"
    env = analyze_audio(info)
    loud = _onsets(env.rms_db, np.arange(len(env.rms_db)) * env.hop, -30.0)
    assert max(_beep_errors(loud)) < 0.15, f"{kind} envelope: {[round(t, 2) for t in loud]}"


def test_audio_cache_of_the_old_clock_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr("auto_video_editor.pipeline.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    source = _make(tmp_path / "clip.mp4")
    pipeline = Pipeline(str(source), Settings(transcriber="none", use_cache=True), str(tmp_path / "out"),
                        Reporter(lambda _m: None, lambda *_: None))
    info = probe(str(source))
    old = tmp_path / "cache" / "audio" / f"{media_fingerprint(info.path)}-v1.npz"
    old.parent.mkdir(parents=True)
    AudioEnvelope(0.05, np.zeros(10, np.float32), np.zeros(10, np.float32)).save(old)
    assert len(pipeline._load_envelope(info).rms_db) > 700, "a 40 s clip has about 800 frames, not the old 10"
