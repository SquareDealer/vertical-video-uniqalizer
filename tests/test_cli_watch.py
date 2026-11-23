"""``python main.py --watch``: one cycle against a moto bucket, and clean errors without credentials."""

import threading

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from auto_video_editor import cli  # noqa: E402
from auto_video_editor.s3store import S3Config, S3Store  # noqa: E402
from auto_video_editor.s3watch import RenderedVariant, output_keys  # noqa: E402

BUCKET = "cli-bucket"
SOURCE = "montage/buster/clip.mp4"
ENV = ("AVE_S3_ENDPOINT", "AVE_S3_BUCKET", "AVE_S3_ACCESS_KEY", "AVE_S3_SECRET_KEY")


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    for name in ENV + ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.cli.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.settings_path", lambda: tmp_path / "settings.json")
    return tmp_path


def fake_render_fn(settings, log):
    def render(source, work, count, cancel: threading.Event):
        out = []
        for i in range(1, count + 1):
            path = work / f"v{i}.mp4"
            path.write_bytes(b"version %d" % i)
            out.append(RenderedVariant(path, {"index": i}, True))
        return out
    return render


def test_watch_once_against_moto(isolated, monkeypatch, capsys):
    env = isolated / ".env"
    env.write_text(f"AVE_S3_ENDPOINT=https://s3.test\nAVE_S3_BUCKET={BUCKET}\n"
                   "AVE_S3_ACCESS_KEY=AKIATEST\nAVE_S3_SECRET_KEY=topsecret\n", encoding="utf-8")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
        client.create_bucket(Bucket=BUCKET)
        client.put_object(Bucket=BUCKET, Key=SOURCE, Body=b"source", Metadata={"clip-id": "c1"})
        store = S3Store(S3Config("https://s3.test", BUCKET, "k", "s", region="us-east-1"), client=client)
        monkeypatch.setattr("auto_video_editor.cli.open_store", lambda config: store)
        monkeypatch.setattr("auto_video_editor.s3watch.default_render_fn", fake_render_fn)
        assert cli.main(["--watch", "--once", "--env-file", str(env)]) == 0
        assert store.get_json(output_keys(SOURCE, "uniqalized", 3).manifest)["clip_id"] == "c1"
    out = capsys.readouterr()
    assert "topsecret" not in out.out + out.err and "AKIATEST" not in out.out + out.err


def test_watch_requires_credentials(isolated, capsys):
    code = cli.main(["--watch", "--once", "--env-file", str(isolated / "missing.env")])
    assert code == 2
    assert "AVE_S3_ENDPOINT" in capsys.readouterr().err


def test_watch_once_sends_crm_webhook(isolated, monkeypatch, capsys):
    from test_webhook import CRM

    monkeypatch.delenv("AVE_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AVE_WEBHOOK_SECRET", raising=False)
    crm = CRM()
    env = isolated / ".env"
    env.write_text(f"AVE_S3_ENDPOINT=https://s3.test\nAVE_S3_BUCKET={BUCKET}\n"
                   "AVE_S3_ACCESS_KEY=AKIATEST\nAVE_S3_SECRET_KEY=topsecret\n"
                   f"AVE_WEBHOOK_URL={crm.url}\nAVE_WEBHOOK_SECRET=hooksecret\n", encoding="utf-8")
    try:
        with moto.mock_aws():
            client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
            client.create_bucket(Bucket=BUCKET)
            client.put_object(Bucket=BUCKET, Key=SOURCE, Body=b"source", Metadata={"clip-id": "c1"})
            store = S3Store(S3Config("https://s3.test", BUCKET, "k", "s", region="us-east-1"), client=client)
            monkeypatch.setattr("auto_video_editor.cli.open_store", lambda config: store)
            monkeypatch.setattr("auto_video_editor.s3watch.default_render_fn", fake_render_fn)
            assert cli.main(["--watch", "--once", "--env-file", str(env)]) == 0
        assert len(crm.requests) == 1
        request = crm.requests[0]
        assert b'"clip_id": "c1"' in request["body"]
        assert request["headers"]["X-AVE-Signature"].startswith("sha256=")
    finally:
        crm.close()
    out = capsys.readouterr()
    assert "hooksecret" not in out.out + out.err
