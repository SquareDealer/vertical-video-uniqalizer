"""S3 store: verified transfers against a moto bucket."""

import dataclasses
import os

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from auto_video_editor.s3store import S3Config, S3Store, TransferError  # noqa: E402

BUCKET = "test-bucket"


@pytest.fixture()
def store(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    with moto.mock_aws():
        config = S3Config("https://s3.example.test", BUCKET, "AKIAEXAMPLE", "secret-value", region="us-east-1")
        client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")
        client.create_bucket(Bucket=BUCKET)
        yield S3Store(config, client=client)


def test_from_env_reads_dotenv_and_masks_keys(tmp_path, monkeypatch):
    for name in ("AVE_S3_ENDPOINT", "AVE_S3_BUCKET", "AVE_S3_ACCESS_KEY", "AVE_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("AVE_S3_ENDPOINT=https://s3.example.test\nAVE_S3_BUCKET=b\n"
                   "AVE_S3_ACCESS_KEY=AKIAEXAMPLE\nAVE_S3_SECRET_KEY=secret-value\n", encoding="utf-8")
    cfg = S3Config.from_env(env)
    assert (cfg.endpoint, cfg.bucket, cfg.access_key, cfg.secret_key) == (
        "https://s3.example.test", "b", "AKIAEXAMPLE", "secret-value")
    assert "secret-value" not in repr(cfg) and "AKIAEXAMPLE" not in repr(cfg)
    monkeypatch.setenv("AVE_S3_BUCKET", "from-env")
    assert S3Config.from_env(env).bucket == "from-env", "environment wins over the file"
    env.write_text("AVE_S3_ENDPOINT=x\nAVE_S3_BUCKET=b\nAVE_S3_ACCESS_KEY=k\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="AVE_S3_SECRET_KEY"):
        S3Config.from_env(env)


def test_upload_download_roundtrip_keeps_metadata(store, tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(os.urandom(1024 * 1024))
    info = store.upload(src, "montage/buster/clip.mp4", {"clip-id": "abc"})
    assert info.size == src.stat().st_size
    head = store.head("montage/buster/clip.mp4")
    assert head.metadata["clip-id"] == "abc" and head.size == info.size
    dest = tmp_path / "out" / "clip.mp4"
    store.download(head, dest)
    assert dest.read_bytes() == src.read_bytes()


def test_download_detects_size_mismatch(store, tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x" * 1000)
    info = store.upload(src, "a/b/clip.mp4", {})
    dest = tmp_path / "dl.mp4"
    with pytest.raises(TransferError):
        store.download(dataclasses.replace(info, size=info.size + 1), dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("dl.mp4*")), "no temporary file is left behind"


def test_download_detects_md5_mismatch(store, tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"y" * 1000)
    info = store.upload(src, "a/b/clip.mp4", {})
    with pytest.raises(TransferError):
        store.download(dataclasses.replace(info, etag="0" * 32), tmp_path / "dl.mp4")


def test_head_missing_returns_none(store):
    assert store.head("nope/missing.mp4") is None


def test_copy_keeps_metadata(store, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"v1")
    store.upload(src, "p/.partial/v.mp4", {"variant": "1"})
    store.copy("p/.partial/v.mp4", "p/v.mp4")
    assert store.head("p/v.mp4").metadata["variant"] == "1"
    store.delete("p/.partial/v.mp4")
    assert store.head("p/.partial/v.mp4") is None


def test_list_is_paginated(store):
    client = store.client
    for i in range(1005):
        client.put_object(Bucket=BUCKET, Key=f"montage/s/{i:04d}.mp4", Body=b"")
    assert len(store.list("montage/")) == 1005


def test_json_roundtrip_and_missing(store):
    assert store.get_json("x/missing.json") is None
    store.put_json("x/m.json", {"a": "клип"})
    assert store.get_json("x/m.json") == {"a": "клип"}
