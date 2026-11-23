"""CRM webhook: signed POST after the versions of a clip are published, with a durable retry queue."""

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from auto_video_editor.webhook import WebhookConfig, WebhookOutbox, published_event, sign


class CRM:
    """A local HTTP server standing in for the CRM: records requests, answers with ``codes`` in turn."""

    def __init__(self, codes=(200,)):
        self.requests, self.codes = [], list(codes)
        crm = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                crm.requests.append({"path": self.path, "headers": self.headers, "body": body})
                code = crm.codes.pop(0) if len(crm.codes) > 1 else crm.codes[0]
                self.send_response(code)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/hooks/ave"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def crm():
    server = CRM()
    yield server
    server.close()


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


EVENT = {"event": "published", "event_id": "clip-1:etag-1", "clip_id": "clip-1"}


def outbox(tmp_path, url, secret="", clock=None, logs=None):
    return WebhookOutbox(tmp_path / "webhook.json", WebhookConfig(url, secret), (logs if logs is not None else []).append,
                         clock=clock or Clock())


def test_sign_is_hmac_sha256_of_the_body():
    body = b'{"a": 1}'
    assert sign(body, "s3cret") == "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()


def test_config_from_env_file_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("AVE_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AVE_WEBHOOK_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text("AVE_WEBHOOK_URL=https://crm.test/hook\nAVE_WEBHOOK_SECRET=topsecret\n", encoding="utf-8")
    config = WebhookConfig.from_env(env)
    assert (config.url, config.secret) == ("https://crm.test/hook", "topsecret")
    assert "topsecret" not in repr(config)
    monkeypatch.setenv("AVE_WEBHOOK_URL", "https://other.test/hook")
    assert WebhookConfig.from_env(env).url == "https://other.test/hook"


def test_config_is_off_without_url(tmp_path, monkeypatch):
    monkeypatch.delenv("AVE_WEBHOOK_URL", raising=False)
    assert WebhookConfig.from_env(tmp_path / "missing.env") is None


def test_delivers_signed_event(tmp_path, crm):
    box = outbox(tmp_path, crm.url, secret="s3cret")
    box.add(EVENT)
    assert box.flush() == (1, 0)
    request = crm.requests[0]
    assert request["path"] == "/hooks/ave"
    assert json.loads(request["body"]) == EVENT
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["X-AVE-Event"] == "published"
    assert request["headers"]["X-AVE-Event-Id"] == "clip-1:etag-1"
    assert request["headers"]["X-AVE-Signature"] == sign(request["body"], "s3cret")
    assert box.pending() == []
    assert WebhookOutbox(tmp_path / "webhook.json", WebhookConfig(crm.url), print).pending() == []


def test_no_signature_header_without_secret(tmp_path, crm):
    box = outbox(tmp_path, crm.url)
    box.add(EVENT)
    box.flush()
    assert "X-AVE-Signature" not in crm.requests[0]["headers"]


def test_server_error_is_retried_after_restart(tmp_path, crm):
    crm.codes = [500, 200]
    clock = Clock()
    box = outbox(tmp_path, crm.url, clock=clock)
    box.add(EVENT)
    assert box.flush() == (0, 1)
    assert box.flush() == (0, 1) and len(crm.requests) == 1  # backoff: not before the next try time
    clock.now += 3600
    again = outbox(tmp_path, crm.url, clock=clock)  # the queue survives a restart
    assert again.flush() == (1, 0)
    assert len(crm.requests) == 2 and again.pending() == []


def test_unreachable_crm_keeps_the_event(tmp_path):
    box = outbox(tmp_path, "http://127.0.0.1:9/hook")
    box.add(EVENT)
    assert box.flush() == (0, 1)
    assert box.pending()[0]["attempts"] == 1 and box.pending()[0]["last_error"]


def test_client_error_is_not_retried(tmp_path, crm):
    crm.codes = [400]
    clock, logs = Clock(), []
    box = outbox(tmp_path, crm.url, clock=clock, logs=logs)
    box.add(EVENT)
    assert box.flush() == (0, 0)
    clock.now += 86400
    assert box.flush() == (0, 0) and len(crm.requests) == 1
    assert box.pending() == [] and box.rejected()[0]["event"]["event_id"] == "clip-1:etag-1"
    assert any("400" in line for line in logs)


def test_too_many_requests_is_retried(tmp_path, crm):
    crm.codes = [429, 200]
    clock = Clock()
    box = outbox(tmp_path, crm.url, clock=clock)
    box.add(EVENT)
    assert box.flush() == (0, 1)
    clock.now += 3600
    assert box.flush() == (1, 0)


def test_same_event_is_queued_once(tmp_path, crm):
    box = outbox(tmp_path, crm.url)
    box.add(EVENT)
    box.add(dict(EVENT))
    box.flush()
    assert len(crm.requests) == 1


def test_published_event_from_manifest():
    manifest = {"manifest_version": 1, "clip_id": "clip-1",
                "source": {"key": "montage/buster/a.mp4", "etag": "e1", "size": 10, "last_modified": "t"},
                "variants": [{"key": "montage/buster/uniqalized/a/v1.mp4"},
                             {"key": "montage/buster/uniqalized/a/v2.mp4"}], "created": "c"}
    event = published_event(manifest, "montage/buster/uniqalized/a", "montage/buster/uniqalized/a/manifest.json",
                            "montage/buster/uniqalized/a/status.json")
    assert event["event"] == "published" and event["event_id"] == "clip-1:e1"
    assert event["clip_id"] == "clip-1" and event["source"] == manifest["source"]
    assert event["folder"] == "montage/buster/uniqalized/a"
    assert event["versions"] == ["montage/buster/uniqalized/a/v1.mp4", "montage/buster/uniqalized/a/v2.mp4"]
    assert event["manifest_key"].endswith("manifest.json") and event["status_key"].endswith("status.json")
    assert event["manifest"] == manifest


def test_event_id_without_clip_id_uses_source_key():
    manifest = {"clip_id": None, "source": {"key": "montage/x/a.mp4", "etag": "e1"}, "variants": []}
    assert published_event(manifest, "f", "m", "s")["event_id"] == "montage/x/a.mp4:e1"
