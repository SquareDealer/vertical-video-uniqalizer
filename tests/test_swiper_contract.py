"""Contract with the Swiper (2026-11-24): metadata of a clip in montage/, status.json and manifest.json."""

import threading

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from test_s3watch import SOURCE, FakeRender, make_watcher, put_source, store  # noqa: E402,F401

from auto_video_editor.s3watch import ClipOptions, clip_options, output_keys, watch_settings  # noqa: E402
from auto_video_editor.settings import Settings  # noqa: E402

BROADCAST = "324159870123_2026-11-20_18-02"
NESTED = f"montage/buster/{BROADCAST}/2026-11-20_18-30-05_chat+audio_6f4a.mp4"
SWIPER_META = {"clip-id": "8d7f3a2c9c1e6f4a", "stream-id": "324159870123", "montage-mode": "full",
               "webcam-region": "0.7000,0.6500,0.2750,0.3056", "stream-region": "0.0000,0.0000,1.0000,0.6200",
               "swiper-source": "x"}


def put(store, key=NESTED, metadata=None):
    store.client.put_object(Bucket="watch-bucket", Key=key, Body=b"source-video",
                            Metadata=SWIPER_META if metadata is None else metadata)
    return store.head(key)


class Recorder(FakeRender):
    def __init__(self, store=None, keys=None, **kwargs):
        super().__init__(**kwargs)
        self.options, self.status_seen, self.store, self.keys = [], [], store, keys

    def __call__(self, source, work, count, cancel, options=None):
        self.options.append(options)
        if self.store is not None:
            self.status_seen.append(self.store.get_json(self.keys.status))
        return super().__call__(source, work, count, cancel)


def test_clip_options_from_swiper_metadata():
    options = clip_options(SWIPER_META)
    assert options.facecam == "0.7000,0.6500,0.2750,0.3056" and options.stream_area == "0.0000,0.0000,1.0000,0.6200"
    assert options.mode == "full" and options.stream_id == "324159870123" and not options.notes
    assert clip_options({}) == ClipOptions(), "no regions: the layout is detected, mode cut"
    assert clip_options({"montage-mode": "cut"}).mode == "cut"
    bad = clip_options({"webcam-region": "0.9,0.9,0.5,0.5", "stream-region": "abc", "montage-mode": "weird"})
    assert (bad.facecam, bad.stream_area, bad.mode) == ("", "", "cut")
    assert len(bad.notes) == 3, "every ignored value is reported"


def test_watch_settings_follow_the_clip():
    user = Settings(facecam="0.1,0.1,0.2,0.2", shorts_jump_cuts=False)
    cut = watch_settings(user, ClipOptions())
    assert cut.facecam == "", "a webcam set by hand for manual edits never leaks into a clip"
    assert cut.whole_clip and cut.shorts_jump_cuts and not cut.whole_clip_full and cut.transcriber == "http"
    full = watch_settings(user, clip_options(SWIPER_META))
    assert full.facecam == "0.7000,0.6500,0.2750,0.3056" and full.stream_area == "0.0000,0.0000,1.0000,0.6200"
    assert full.whole_clip_full and not full.shorts_jump_cuts, "full: whole length, pauses kept"
    assert user.facecam == "0.1,0.1,0.2,0.2", "user settings untouched"


def test_regions_and_mode_reach_the_render_and_the_manifest(store, tmp_path):
    info = put(store)
    keys = output_keys(NESTED, "uniqalized", 3, prefix="montage/")
    render = Recorder(store, keys)
    watcher, _ = make_watcher(store, tmp_path, render)
    assert watcher.run_once(threading.Event()).done == 1
    assert render.options[0].facecam == "0.7000,0.6500,0.2750,0.3056" and render.options[0].mode == "full"
    started = render.status_seen[0]
    assert started["status"] == "processing" and started["clipId"] == "8d7f3a2c9c1e6f4a"
    assert started["attempts"] == 1
    manifest = store.get_json(keys.manifest)
    assert manifest["version"] == 1 and manifest["clipId"] == "8d7f3a2c9c1e6f4a"
    assert manifest["source"]["key"] == NESTED and manifest["source"]["etag"] == info.etag
    assert [v["key"] for v in manifest["variants"]] == keys.final
    assert all(v["key"].startswith(keys.folder + "/") for v in manifest["variants"])
    assert manifest["montage_mode"] == "full" and manifest["stream_id"] == "324159870123"
    assert manifest["regions"] == {"webcam": "0.7000,0.6500,0.2750,0.3056", "stream": "0.0000,0.0000,1.0000,0.6200",
                                   "source": "swiper"}
    done = store.get_json(keys.status)
    assert done["status"] == "done" and done["clipId"] == "8d7f3a2c9c1e6f4a"


def test_failed_status_has_reason_and_attempts(store, tmp_path):
    put(store)
    keys = output_keys(NESTED, "uniqalized", 3, prefix="montage/")
    watcher, _ = make_watcher(store, tmp_path, Recorder(error=ValueError("broken audio")))
    watcher.run_once(threading.Event())
    first = store.get_json(keys.status)
    assert first["status"] == "failed" and "broken audio" in first["reason"] and first["attempts"] == 1
    render = Recorder(store, keys, error=ValueError("broken audio"))
    watcher.render = render
    watcher.run_once(threading.Event())
    assert render.status_seen[0]["attempts"] == 2, "the retry is the second attempt"
    assert store.get_json(keys.status)["attempts"] == 2


def test_source_taken_back_during_render_is_dropped_before_upload(store, tmp_path, monkeypatch):
    put(store)
    keys = output_keys(NESTED, "uniqalized", 3, prefix="montage/")
    uploads = []
    monkeypatch.setattr(store, "upload", lambda *a, **k: uploads.append(a) or (_ for _ in ()).throw(AssertionError))
    render = Recorder(on_call=lambda: store.client.delete_object(Bucket="watch-bucket", Key=NESTED))
    watcher, _ = make_watcher(store, tmp_path, render)
    summary = watcher.run_once(threading.Event())
    assert summary.failed == 0 and not uploads, "the Swiper took the clip back: nothing is uploaded"
    assert store.get_json(keys.status) is None and store.get_json(keys.manifest) is None


def test_old_layout_without_regions(store, tmp_path):
    put_source(store)
    render = Recorder()
    watcher, _ = make_watcher(store, tmp_path, render)
    assert watcher.run_once(threading.Event()).done == 1
    assert render.options[0] == ClipOptions(), "no regions: the layout is detected as before"
    manifest = store.get_json(output_keys(SOURCE, "uniqalized", 3, prefix="montage/").manifest)
    assert manifest["regions"]["source"] == "auto" and manifest["montage_mode"] == "cut"


def test_no_regions_with_split_saved_in_the_app_falls_back_to_detection():
    """The app may have shorts_layout=split saved for manual edits: a clip without regions must not fail."""
    user = Settings(shorts_layout="split", facecam="0.003,0.008,0.247,0.236")
    s = watch_settings(user, ClipOptions())
    assert s.shorts_layout == "auto" and s.facecam == ""
    assert s.validate() == [], "the webcam is detected, or the versions fall back to blur/smart"
    with_regions = watch_settings(user, clip_options(SWIPER_META))
    assert with_regions.shorts_layout == "auto" and with_regions.facecam == "0.7000,0.6500,0.2750,0.3056"


def test_watch_limits_cpu_during_render(store, tmp_path, monkeypatch):
    from auto_video_editor import ffmpeg, s3watch

    seen = []

    class FakePipeline:
        def __init__(self, *a, **k):
            seen.append(dict(ffmpeg._LIMITS))
            raise RuntimeError("stop here")

    monkeypatch.setattr("auto_video_editor.pipeline.Pipeline", FakePipeline)
    render = s3watch.default_render_fn(Settings(watch_threads=3), lambda _m: None)
    with pytest.raises(RuntimeError):
        render(tmp_path / "a.mp4", tmp_path, 3, threading.Event(), ClipOptions())
    assert seen == [{"threads": 3, "low_priority": True}]
    assert ffmpeg._LIMITS == {"threads": 0, "low_priority": False}, "limits end with the clip"


def test_trigger_offset_from_the_autoclipper():
    assert clip_options({"trigger-offset-ms": "37000"}).trigger_offset == 37.0
    assert clip_options({}).trigger_offset is None
    for bad in ("soon", "-5", "12.5"):
        options = clip_options({"trigger-offset-ms": bad})
        assert options.trigger_offset is None and len(options.notes) == 1
    s = watch_settings(Settings(watch_emotion_url="http://emo:8092", emotion_url="http://manual"), clip_options({"trigger-offset-ms": "37000"}))
    assert (s.emotion_url, s.trigger_offset) == ("http://emo:8092", 37.0)
    assert watch_settings(Settings(), ClipOptions()).trigger_offset == -1.0
    assert watch_settings(Settings(), clip_options({"montage-mode": "full"})).emotion_url == "", "full: no emotions needed"


class Planned(FakeRender):
    """A render whose versions carry the edit plan of the clip."""

    def __call__(self, *args, **kwargs):
        variants = super().__call__(*args, **kwargs)
        for variant in variants:
            variant.edit = {"peak": 31.5, "lead_in": 11.3}
        return variants


def test_edit_plan_reaches_the_manifest(store, tmp_path):
    put_source(store)
    k = output_keys(SOURCE, "uniqalized", 3)
    make_watcher(store, tmp_path, Planned())[0].run_once(threading.Event())
    assert store.get_json(k.manifest)["edit"] == {"peak": 31.5, "lead_in": 11.3}
