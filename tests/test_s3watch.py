"""Watch mode: S3 polling, publishing protocol and integrity of the versions of a clip."""

import json
import os
import threading
from pathlib import Path
from urllib.parse import unquote

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from auto_video_editor.emotion import EmotionServerUnavailable  # noqa: E402
from auto_video_editor.s3store import S3Config, S3Store, file_md5  # noqa: E402
from auto_video_editor.s3watch import (RenderedVariant, Watcher, WatchState, acquire_lock, is_ready,  # noqa: E402
                                       is_source_key, output_keys)
from auto_video_editor.settings import Settings  # noqa: E402
from auto_video_editor.transcribe import SpeechServerUnavailable  # noqa: E402

BUCKET = "watch-bucket"
SOURCE = "montage/buster/2026-11-20_reaction.mp4"


@pytest.fixture()
def store(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
        client.create_bucket(Bucket=BUCKET)
        yield S3Store(S3Config("https://s3.test", BUCKET, "k", "s", region="us-east-1"), client=client)


def put_source(store, key=SOURCE, body=b"source-video", clip_id="clip-123"):
    metadata = {"clip-id": clip_id} if clip_id else {}
    store.client.put_object(Bucket=BUCKET, Key=key, Body=body, Metadata=metadata)
    return store.head(key)


class FakeRender:
    """Writes ``count`` small distinct files instead of rendering video."""

    def __init__(self, error=None, on_call=None):
        self.calls, self.error, self.on_call = 0, error, on_call
        self.cancels = []

    def __call__(self, source: Path, work: Path, count: int, cancel: threading.Event, options=None):
        self.calls += 1
        self.cancels.append(cancel)
        assert source.exists() and source.read_bytes().startswith(b"source")
        if self.on_call:
            self.on_call()
        if self.error:
            raise self.error
        result = []
        for i in range(1, count + 1):
            path = work / f"v{i}.mp4"
            path.write_bytes(b"v%d-" % i + os.urandom(64))
            result.append(RenderedVariant(path, {"index": i}, captions=True))
        return result


def make_watcher(store, tmp_path, render, speech_check=None, emotion_check=None, **settings):
    logs = []
    watcher = Watcher(store, Settings(**settings), render, tmp_path / "work", WatchState(tmp_path / "state.json"),
                      logs.append, speech_check=speech_check, emotion_check=emotion_check)
    return watcher, logs


def keys(store, prefix=""):
    return sorted(o.key for o in store.list(prefix))


def state(store, k):
    return (store.get_json(k.status) or {}).get("state")


def test_source_key_filter():
    assert is_source_key("montage/buster/a.MP4", "montage/")
    assert is_source_key("montage/buster/clip.mkv", "montage/")
    assert not is_source_key("montage/a.mp4", "montage/")
    assert not is_source_key("montage/buster/uniqalized/a/v1.mp4", "montage/")
    assert not is_source_key("montage/buster/a.txt", "montage/")
    assert not is_source_key("to_sort/buster/a.mp4", "montage/")


def test_output_keys_layout():
    k = output_keys(SOURCE, "uniqalized", 3)
    folder = "montage/buster/uniqalized/2026-11-20_reaction"
    assert k.folder == folder, "one folder per source, named after it"
    assert k.final == [f"{folder}/v{i}.mp4" for i in (1, 2, 3)]
    assert k.partial[0] == f"{folder}/.partial/v1.mp4"
    assert k.manifest == f"{folder}/manifest.json"
    assert k.status == f"{folder}/status.json"
    assert output_keys("montage/buster/clip.mov", "uniqalized", 3, unique=True).folder ==         "montage/buster/uniqalized/clip.mov"


def test_happy_path_publishes_versions_then_manifest(store, tmp_path):
    source = put_source(store)
    k = output_keys(SOURCE, "uniqalized", 3)
    seen = []
    render = FakeRender(on_call=lambda: seen.append(store.get_json(k.status)))
    watcher, _ = make_watcher(store, tmp_path, render)
    summary = watcher.run_once(threading.Event())
    assert (summary.done, summary.failed) == (1, 0)
    assert seen[0]["state"] == "processing" and seen[0]["clip_id"] == "clip-123"
    assert seen[0]["source"] == {"key": SOURCE, "etag": source.etag}
    done = store.get_json(k.status)
    assert done["state"] == "done" and done["manifest"] == k.manifest and done["versions"] == k.final
    manifest = store.get_json(k.manifest)
    assert manifest["clip_id"] == "clip-123"
    assert manifest["source"]["key"] == SOURCE and manifest["source"]["etag"] == source.etag
    assert manifest["captions"] is True
    for i, entry in enumerate(manifest["variants"], 1):
        head = store.head(entry["key"])
        assert entry["key"] == k.final[i - 1]
        assert head.size == entry["size"] and head.etag == entry["md5"]
        assert head.metadata["clip-id"] == "clip-123" and head.metadata["variant"] == str(i)
        assert head.metadata["source-etag"] == source.etag
    assert not keys(store, f"{k.folder}/.partial/")
    assert is_ready(store, store.head(SOURCE), k)
    assert store.head(SOURCE).etag == source.etag, "the source is never touched"

    summary = watcher.run_once(threading.Event())
    assert (summary.done, summary.skipped) == (0, 1) and render.calls == 1


def test_unicode_keys(store, tmp_path):
    key = "montage/бустер/клип 1+2.mp4"
    put_source(store, key=key)
    watcher, _ = make_watcher(store, tmp_path, FakeRender())
    assert watcher.run_once(threading.Event()).done == 1
    k = output_keys(key, "uniqalized", 3)
    assert unquote(store.head(k.final[0]).metadata["source-key"]) == key
    assert store.get_json(k.manifest)["source"]["key"] == key


def test_crash_after_partial_upload_leaves_no_manifest(store, tmp_path, monkeypatch):
    put_source(store)
    real_upload, calls = store.upload, []

    def flaky_upload(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ConnectionError("network down")
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(store, "upload", flaky_upload)
    watcher, _ = make_watcher(store, tmp_path, FakeRender())
    summary = watcher.run_once(threading.Event())
    k = output_keys(SOURCE, "uniqalized", 3)
    assert (summary.failed, summary.pending) == (0, 1), "a network error is retried, not counted"
    assert not any(store.head(key) for key in k.final) and store.get_json(k.manifest) is None
    assert WatchState(tmp_path / "state.json").data.get(SOURCE, {}).get("failures", 0) == 0
    assert state(store, k) == "waiting"

    monkeypatch.setattr(store, "upload", real_upload)
    assert watcher.run_once(threading.Event()).done == 1
    assert is_ready(store, store.head(SOURCE), k)


def test_source_changed_triggers_rebuild(store, tmp_path):
    put_source(store)
    render = FakeRender()
    watcher, _ = make_watcher(store, tmp_path, render)
    watcher.run_once(threading.Event())
    k = output_keys(SOURCE, "uniqalized", 3)
    old_md5 = store.head(k.final[0]).etag
    new = put_source(store, body=b"source-video-v2")
    assert not is_ready(store, new, k)
    assert watcher.run_once(threading.Event()).done == 1
    assert store.get_json(k.manifest)["source"]["etag"] == new.etag
    assert store.head(k.final[0]).etag != old_md5 and render.calls == 2


def test_fewer_versions_removes_stale_ones(store, tmp_path):
    put_source(store)
    make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    put_source(store, body=b"source-video-v2")
    watcher, _ = make_watcher(store, tmp_path, FakeRender(), watch_variants=2)
    assert watcher.run_once(threading.Event()).done == 1
    assert store.head(output_keys(SOURCE, "uniqalized", 3).final[2]) is None


def test_source_vanished_mid_render(store, tmp_path):
    put_source(store)
    render = FakeRender(on_call=lambda: store.delete(SOURCE))
    watcher, _ = make_watcher(store, tmp_path, render)
    summary = watcher.run_once(threading.Event())
    assert summary.failed == 0
    assert not keys(store, "montage/buster/uniqalized/")
    assert WatchState(tmp_path / "state.json").data.get(SOURCE, {}).get("failures", 0) == 0


def test_speech_server_down_keeps_clip_pending(store, tmp_path):
    put_source(store)
    watcher, _ = make_watcher(store, tmp_path, FakeRender(error=SpeechServerUnavailable("Kamerton is down")))
    summary = watcher.run_once(threading.Event())
    assert (summary.pending, summary.failed) == (1, 0)
    status = store.get_json(output_keys(SOURCE, "uniqalized", 3).status)
    assert status["state"] == "waiting" and "Kamerton is down" in status["reason"]


def test_parked_after_three_failures(store, tmp_path):
    put_source(store)
    render = FakeRender(error=ValueError("broken file"))
    watcher, _ = make_watcher(store, tmp_path, render)
    for _ in range(3):
        assert watcher.run_once(threading.Event()).failed == 1
    k = output_keys(SOURCE, "uniqalized", 3)
    error = store.get_json(k.status)
    assert error["state"] == "failed" and error["failures"] == 3 and error["parked"] is True
    assert "broken file" in error["error"]
    summary = watcher.run_once(threading.Event())
    assert render.calls == 3 and summary.skipped == 1
    put_source(store, body=b"source-fixed")
    render.error = None
    assert watcher.run_once(threading.Event()).done == 1 and render.calls == 4
    assert state(store, k) == "done", "a successful run replaces the error report"


def test_source_without_clip_id(store, tmp_path):
    put_source(store, clip_id=None)
    make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    manifest = store.get_json(output_keys(SOURCE, "uniqalized", 3).manifest)
    assert manifest["clip_id"] is None


def test_lock_rejects_second_watcher(tmp_path):
    lock = tmp_path / "watch.lock"
    with acquire_lock(lock):
        with pytest.raises(RuntimeError, match="another watcher"):
            with acquire_lock(lock):
                pass
    with acquire_lock(lock):  # released, can be taken again
        pass


def test_stale_lock_is_replaced(tmp_path):
    lock = tmp_path / "watch.lock"
    lock.write_text("999999999")
    with acquire_lock(lock):
        assert lock.read_text().strip() == str(os.getpid())


def test_rebuild_failure_leaves_no_stale_versions(store, tmp_path):
    put_source(store)
    make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    put_source(store, body=b"source-video-v2")
    watcher, _ = make_watcher(store, tmp_path, FakeRender(error=ValueError("bad frame")))
    assert watcher.run_once(threading.Event()).failed == 1
    k = output_keys(SOURCE, "uniqalized", 3)
    assert not any(store.head(key) for key in k.final), "versions of the old file must not stay under final names"
    assert store.get_json(k.manifest) is None


def test_copy_failure_removes_copied_finals(store, tmp_path, monkeypatch):
    put_source(store)
    real_copy, calls = store.copy, []

    def flaky_copy(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise RuntimeError("copy refused")
        return real_copy(src, dst)

    monkeypatch.setattr(store, "copy", flaky_copy)
    make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    k = output_keys(SOURCE, "uniqalized", 3)
    assert not any(store.head(key) for key in k.final), "no half-published set"


def test_same_stem_sources_do_not_collide(store, tmp_path):
    put_source(store, key="montage/buster/clip.mp4", body=b"source-a")
    put_source(store, key="montage/buster/clip.mov", body=b"source-b")
    render = FakeRender()
    watcher, _ = make_watcher(store, tmp_path, render)
    assert watcher.run_once(threading.Event()).done == 2
    finals = keys(store, "montage/buster/uniqalized/")
    assert len([k for k in finals if k.endswith(".mp4")]) == 6
    assert watcher.run_once(threading.Event()).skipped == 2 and render.calls == 2


def test_one_bad_clip_does_not_block_the_queue(store, tmp_path):
    put_source(store, key="montage/buster/a.mp4")
    put_source(store, key="montage/buster/b.mp4")

    class OneBad(FakeRender):
        def __call__(self, source, work, count, cancel, options=None):
            if self.calls == 0:
                self.calls += 1
                raise SpeechServerUnavailable("read timed out")
            return super().__call__(source, work, count, cancel)

    watcher, _ = make_watcher(store, tmp_path, OneBad(), speech_check=lambda: True)
    summary = watcher.run_once(threading.Event())
    assert (summary.failed, summary.done) == (1, 1), "the server answers, so it was this clip, not the server"


def test_network_error_is_retried_not_parked(store, tmp_path):
    from botocore.exceptions import EndpointConnectionError

    put_source(store)
    render = FakeRender(error=EndpointConnectionError(endpoint_url="https://s3.test"))
    watcher, _ = make_watcher(store, tmp_path, render)
    for _ in range(4):
        summary = watcher.run_once(threading.Event())
        assert (summary.pending, summary.failed) == (1, 0)
    assert render.calls == 4
    assert state(store, output_keys(SOURCE, "uniqalized", 3)) != "failed"


def test_source_deleted_during_download_is_a_quiet_skip(store, tmp_path, monkeypatch):
    put_source(store)
    real_download = store.download

    def vanishing_download(info, dest):
        store.delete(info.key)
        return real_download(info, dest)

    monkeypatch.setattr(store, "download", vanishing_download)
    summary = make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    assert (summary.failed, summary.skipped) == (0, 1)
    assert state(store, output_keys(SOURCE, "uniqalized", 3)) != "failed"


def test_stop_lets_the_current_clip_finish(store, tmp_path):
    put_source(store, key="montage/buster/a.mp4")
    put_source(store, key="montage/buster/b.mp4")
    stop = threading.Event()
    render = FakeRender(on_call=stop.set)
    summary = make_watcher(store, tmp_path, render)[0].run_once(stop)
    assert summary.done == 1 and render.calls == 1, "the clip in progress is published, then the watcher stops"
    assert not render.cancels[0].is_set()


def test_dropped_download_is_retried_not_counted(store, tmp_path, monkeypatch):
    from botocore.exceptions import IncompleteReadError

    put_source(store)

    def dropping_download(info, dest):
        raise IncompleteReadError(actual_bytes=5637240, expected_bytes=63611757)

    monkeypatch.setattr(store, "download", dropping_download)
    watcher, _ = make_watcher(store, tmp_path, FakeRender())
    summary = watcher.run_once(threading.Event())
    assert (summary.pending, summary.failed) == (1, 0)
    assert state(store, output_keys(SOURCE, "uniqalized", 3)) != "failed"


def test_webhook_after_publish(store, tmp_path):
    from test_webhook import CRM  # noqa: E402
    from auto_video_editor.webhook import WebhookConfig, WebhookOutbox

    crm = CRM()
    try:
        put_source(store)
        watcher, logs = make_watcher(store, tmp_path, FakeRender())
        watcher.webhook = WebhookOutbox(tmp_path / "webhook.json", WebhookConfig(crm.url), logs.append)
        assert watcher.run_once(threading.Event()).done == 1
        k = output_keys(SOURCE, "uniqalized", 3)
        manifest = store.get_json(k.manifest)
        event = json.loads(crm.requests[0]["body"])
        assert event["event"] == "published" and event["clip_id"] == "clip-123"
        assert event["versions"] == [v["key"] for v in manifest["variants"]] == k.final
        assert event["manifest"] == manifest and event["folder"] == k.folder
    finally:
        crm.close()


def test_webhook_failure_does_not_undo_publish(store, tmp_path):
    from test_webhook import CRM  # noqa: E402
    from auto_video_editor.webhook import WebhookConfig, WebhookOutbox

    crm = CRM(codes=[500])
    try:
        put_source(store)
        watcher, logs = make_watcher(store, tmp_path, FakeRender())
        watcher.webhook = WebhookOutbox(tmp_path / "webhook.json", WebhookConfig(crm.url), logs.append)
        summary = watcher.run_once(threading.Event())
        k = output_keys(SOURCE, "uniqalized", 3)
        assert summary.done == 1 and summary.failed == 0
        assert is_ready(store, store.head(SOURCE), k) and state(store, k) == "done"
        assert len(watcher.webhook.pending()) == 1
    finally:
        crm.close()


# Broadcast folders (S3 layout contract of the Autoclipper and the Swiper, 2026-11-24):
#   montage/<streamer>/<broadcast>/<clip>.mp4          source (swipe right)
#   montage/<streamer>/<broadcast>/<clip>/uniqalized/  written by this service
BROADCAST = "324159870123_2026-11-20_18-02"
NESTED = f"montage/buster/{BROADCAST}/2026-11-20_18-30-12_laugh_ab12.mp4"


def test_source_key_in_a_broadcast_folder():
    assert is_source_key(NESTED, "montage/")
    assert not is_source_key(f"montage/buster/{BROADCAST}/2026-11-20_18-30-12_laugh_ab12/uniqalized/v1.mp4",
                             "montage/")
    assert not is_source_key(f"montage/buster/{BROADCAST}/clip/uniqalized/.partial/v1.mp4", "montage/")
    assert not is_source_key("montage/buster/uniqalized/clip.mp4", "montage/"), "never the versions folder"
    assert not is_source_key("montage/buster/.partial/clip.mp4", "montage/")
    assert not is_source_key(f"montage/buster/{BROADCAST}/x/y.mp4", "montage/"), "no deeper than the broadcast"


def test_output_keys_next_to_a_clip_in_a_broadcast_folder():
    k = output_keys(NESTED, "uniqalized", 3, prefix="montage/")
    folder = f"montage/buster/{BROADCAST}/2026-11-20_18-30-12_laugh_ab12/uniqalized"
    assert k.folder == folder
    assert k.final == [f"{folder}/v{i}.mp4" for i in (1, 2, 3)]
    assert k.partial[0] == f"{folder}/.partial/v1.mp4"
    assert (k.manifest, k.status) == (f"{folder}/manifest.json", f"{folder}/status.json")
    legacy = output_keys(SOURCE, "uniqalized", 3, prefix="montage/")
    assert legacy.folder == "montage/buster/uniqalized/2026-11-20_reaction", "old clips keep their folder"
    assert output_keys(NESTED.replace(".mp4", ".mov"), "uniqalized", 3, unique=True, prefix="montage/").folder \
        == f"montage/buster/{BROADCAST}/2026-11-20_18-30-12_laugh_ab12.mov/uniqalized"


def test_watcher_publishes_a_broadcast_clip_and_the_old_layout(store, tmp_path):
    put_source(store, NESTED, clip_id="nested-1")
    put_source(store, SOURCE, clip_id="legacy-1")
    watcher, _logs = make_watcher(store, tmp_path, FakeRender())
    assert watcher.run_once(threading.Event()).done == 2
    nested = output_keys(NESTED, "uniqalized", 3, prefix="montage/")
    legacy = output_keys(SOURCE, "uniqalized", 3, prefix="montage/")
    assert store.get_json(nested.manifest)["clip_id"] == "nested-1" and state(store, nested) == "done"
    assert store.get_json(legacy.manifest)["clip_id"] == "legacy-1"
    assert all(store.head(key) for key in nested.final)
    fresh, _ = make_watcher(store, tmp_path / "again", FakeRender())
    summary = fresh.run_once(threading.Event())
    assert (summary.done, summary.skipped) == (0, 2), "versions are never taken for new clips"


def test_emotion_server_down_keeps_clip_pending(store, tmp_path):
    put_source(store)
    k = output_keys(SOURCE, "uniqalized", 3)
    watcher, logs = make_watcher(store, tmp_path, FakeRender(error=EmotionServerUnavailable("connection refused")),
                                 emotion_check=lambda: False)
    summary = watcher.run_once(threading.Event())
    assert (summary.pending, summary.failed) == (1, 0)
    status = store.get_json(k.status)
    assert status["state"] == "waiting" and status["status"] == "processing"
    assert status["reason"].startswith("emotion server unavailable")
    assert watcher.state.failures(store.head(SOURCE)) == 0


def test_emotion_error_on_a_healthy_server_counts_against_the_clip(store, tmp_path):
    put_source(store)
    watcher, _ = make_watcher(store, tmp_path, FakeRender(error=EmotionServerUnavailable("503")), emotion_check=lambda: True)
    assert watcher.run_once(threading.Event()).failed == 1


def test_manifest_edit_is_null_without_a_plan(store, tmp_path):   # FakeRender as is
    put_source(store)
    k = output_keys(SOURCE, "uniqalized", 3)
    make_watcher(store, tmp_path, FakeRender())[0].run_once(threading.Event())
    assert store.get_json(k.manifest)["edit"] is None
