"""CRM webhook: a signed ``POST`` after the versions of a clip are published.

The bucket stays the source of truth: a webhook that does not get through never undoes a publish. Events wait in
a local queue (``webhook.json``) and are retried with a growing pause until the CRM answers 2xx, also after a
restart. A 4xx answer other than 408/429 means the CRM refuses the event itself: it is set aside, not retried.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import __version__
from .s3store import DEFAULT_ENV_FILE, _mask, _read_env_file

URL_ENV, SECRET_ENV = "AVE_WEBHOOK_URL", "AVE_WEBHOOK_SECRET"
FIRST_PAUSE, MAX_PAUSE = 30.0, 3600.0  # seconds between attempts: 30, 60, 120 ... up to an hour
RETRY_CODES = {408, 429}


@dataclass
class WebhookConfig:
    url: str
    secret: str = ""
    timeout: float = 15.0

    def __repr__(self) -> str:
        return f"WebhookConfig(url={self.url!r}, secret={_mask(self.secret) if self.secret else 'none'})"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> Optional["WebhookConfig"]:
        """``AVE_WEBHOOK_URL``/``AVE_WEBHOOK_SECRET`` from the environment, then ``env_file``. None when no URL."""
        values = _read_env_file(Path(env_file) if env_file else DEFAULT_ENV_FILE)
        url = (os.environ.get(URL_ENV) or values.get(URL_ENV, "")).strip()
        if not url:
            return None
        secret = (os.environ.get(SECRET_ENV) or values.get(SECRET_ENV, "")).strip()
        return cls(url, secret)


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def published_event(manifest: dict, folder: str, manifest_key: str, status_key: str) -> dict:
    """The ``published`` event of one clip; ``event_id`` is the same for every resend of the same publish."""
    source = manifest.get("source") or {}
    clip_id = manifest.get("clip_id")
    return {"event": "published", "event_id": f"{clip_id or source.get('key')}:{source.get('etag')}",
            "clip_id": clip_id, "source": source, "folder": folder,
            "versions": [v["key"] for v in manifest.get("variants", [])],
            "manifest_key": manifest_key, "status_key": status_key, "manifest": manifest}


class WebhookOutbox:
    """Durable queue of events for the CRM."""

    def __init__(self, path: Path, config: WebhookConfig, log: Callable[[str], None],
                 clock: Callable[[], float] = time.time) -> None:
        self.path, self.config, self.log, self.clock = Path(path), config, log, clock
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self._pending: List[dict] = list(data.get("pending", [])) if isinstance(data, dict) else []
        self._rejected: List[dict] = list(data.get("rejected", [])) if isinstance(data, dict) else []

    def pending(self) -> List[dict]:
        return list(self._pending)

    def rejected(self) -> List[dict]:
        return list(self._rejected)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"pending": self._pending, "rejected": self._rejected[-100:]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, event: dict) -> None:
        """Queue an event; a newer event with the same id replaces the waiting one."""
        self._pending = [e for e in self._pending if e["event"].get("event_id") != event.get("event_id")]
        self._pending.append({"event": event, "attempts": 0, "next_try": 0.0, "last_error": ""})
        self._save()

    def flush(self) -> Tuple[int, int]:
        """Send what is due. Returns (sent, still waiting). Never raises."""
        sent, now, keep = 0, self.clock(), []
        for entry in self._pending:
            if entry.get("next_try", 0) > now:
                keep.append(entry)
                continue
            event = entry["event"]
            try:
                code = self._post(event)
            except Exception as exc:  # noqa: BLE001 - network: try again later
                code, error = None, f"{type(exc).__name__}: {exc}"
            else:
                error = f"HTTP {code}"
            if code is not None and 200 <= code < 300:
                sent += 1
                self.log(f"CRM webhook: {event.get('event')} {event.get('event_id')} delivered.")
                continue
            if code is not None and 400 <= code < 500 and code not in RETRY_CODES:
                self._rejected.append({"event": event, "error": error, "at": now})
                self.log(f"CRM webhook: {event.get('event_id')} refused by the CRM ({error}), not retried.")
                continue
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_error"] = error[:500]
            entry["next_try"] = now + min(FIRST_PAUSE * 2 ** (entry["attempts"] - 1), MAX_PAUSE)
            self.log(f"CRM webhook: {event.get('event_id')} not delivered ({error}), will retry.")
            keep.append(entry)
        self._pending = keep
        try:
            self._save()
        except OSError as exc:
            self.log(f"CRM webhook: cannot save the queue: {exc}")
        return sent, len(keep)

    def _post(self, event: dict) -> int:
        body = json.dumps(event, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": f"auto-video-editor/{__version__}",
                   "X-AVE-Event": str(event.get("event")), "X-AVE-Event-Id": str(event.get("event_id"))}
        if self.config.secret:
            headers["X-AVE-Signature"] = sign(body, self.config.secret)
        request = urllib.request.Request(self.config.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code
