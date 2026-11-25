"""Client of the emotion server (services/emotion): GigaAM-Emo probabilities per window, with a cache."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .transcribe import _multipart, _wav_bytes

EMOTION_WINDOW = 3.0       # seconds of audio the model looks at once
EMOTION_HOP = 1.5          # step between windows
EMOTION_TIMEOUT = 600.0    # one chunk of audio on a CPU can take a while


class EmotionServerUnavailable(RuntimeError):
    """The emotion server cannot be reached, timed out or is still loading its model."""


def check_labels(labels: Sequence[str]) -> None:
    """The edit needs ``positive`` and ``angry``; the server gives its labels as they are (any case)."""
    have = {str(label).lower() for label in labels}
    if not {"positive", "angry"} <= have:
        raise RuntimeError("The emotion server has no 'positive'/'angry' label "
                           f"(it gives: {', '.join(str(label) for label in labels) or 'nothing'})")


@dataclass
class EmotionWindow:
    start: float
    end: float
    probs: Dict[str, float]    # keys lower-cased


@dataclass
class EmotionTrack:
    model: str
    labels: List[str]
    windows: List[EmotionWindow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"model": self.model, "labels": list(self.labels),
                "windows": [{"start": w.start, "end": w.end, "probs": w.probs} for w in self.windows]}

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionTrack":
        return cls(str(data["model"]), [str(x) for x in data["labels"]],
                   [EmotionWindow(float(w["start"]), float(w["end"]),
                                  {str(k).lower(): float(v) for k, v in w["probs"].items()})
                    for w in data["windows"]])

    def per_second(self, label: str, seconds: int) -> np.ndarray:
        """Overlap-weighted mean of ``label`` over the windows touching each second; 0 where none does."""
        label = label.lower()
        total = np.zeros(max(0, seconds), np.float64)
        weight = np.zeros_like(total)
        for window in self.windows:
            value = window.probs.get(label, 0.0)
            first = max(0, int(np.floor(window.start)))
            last = min(seconds, int(np.ceil(window.end)))
            for i in range(first, last):
                share = min(window.end, i + 1.0) - max(window.start, float(i))
                if share > 0:
                    total[i] += share * value
                    weight[i] += share
        out = np.zeros_like(total)
        np.divide(total, weight, out=out, where=weight > 0)
        return out.astype(np.float32)

    def score(self, seconds: int) -> np.ndarray:
        """``P(positive) + 0.5 * P(angry)`` per second, 0..1."""
        return np.clip(self.per_second("positive", seconds) + 0.5 * self.per_second("angry", seconds),
                       0.0, 1.0).astype(np.float32)

    def shifted(self, offset: float) -> "EmotionTrack":
        return EmotionTrack(self.model, list(self.labels),
                            [EmotionWindow(w.start + offset, w.end + offset, dict(w.probs)) for w in self.windows])


@dataclass
class EmotionServer:
    url: str
    timeout: float = EMOTION_TIMEOUT

    def __post_init__(self) -> None:
        url = (self.url or "").strip().rstrip("/")
        if url and "://" not in url:
            url = "http://" + url
        for suffix in ("/v1/audio/emotions", "/health"):
            if url.endswith(suffix):
                url = url[: -len(suffix)].rstrip("/")
        self.url = url

    def describe(self) -> str:
        return self.url

    def _request(self, path: str, data: Optional[bytes] = None, content_type: Optional[str] = None) -> dict:
        import urllib.error
        import urllib.request

        headers = {"Accept": "application/json", "User-Agent": "auto-video-editor"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.url + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("detail", parsed.get("error", detail)))
            except ValueError:
                pass
            error = EmotionServerUnavailable if exc.code in (502, 503) else RuntimeError
            raise error(f"Emotion server error {exc.code} at {self.url}{path}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise EmotionServerUnavailable(
                f"Cannot reach the emotion server at {self.url} ({reason}). "
                "Is it running? docker compose up -d in services/emotion.") from exc
        try:
            data = json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"The emotion server at {self.url} did not answer with JSON.") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"The emotion server at {self.url} answered with something unexpected.")
        return data

    def check(self) -> str:
        """``GET /health``; returns e.g. ``gigaam-emo (angry, sad, neutral, positive)``."""
        data = self._request("/health")
        if data.get("status") != "ok":
            raise EmotionServerUnavailable(f"The emotion server at {self.url} is not ready ({data.get('status')}).")
        labels = [str(x) for x in data.get("labels") or []]
        check_labels(labels)
        return f"{data.get('model', 'emotion model')} ({', '.join(labels)})"

    def analyze(self, audio: np.ndarray, window: float = EMOTION_WINDOW, hop: float = EMOTION_HOP) -> EmotionTrack:
        body, content_type = _multipart([("window", f"{window:g}"), ("hop", f"{hop:g}")], "audio.wav",
                                        _wav_bytes(audio))
        data = self._request("/v1/audio/emotions", body, content_type)
        raw = data.get("windows")
        if not isinstance(raw, list):
            raise RuntimeError(f"The emotion server at {self.url} sent no 'windows' list.")
        windows: List[EmotionWindow] = []
        for item in raw:
            try:
                start, end, probs = float(item["start"]), float(item["end"]), item["probs"]
                windows.append(EmotionWindow(start, end, {str(k).lower(): float(v) for k, v in probs.items()}))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise RuntimeError(f"The emotion server at {self.url} sent a malformed window: {item!r}") from exc
        labels = [str(x) for x in data.get("labels") or (windows[0].probs if windows else [])]
        check_labels(labels)
        return EmotionTrack(str(data.get("model", "emotion model")), [x.lower() for x in labels], windows)


# -- cache --------------------------------------------------------------------------------------

def load_cached(path: Path) -> Optional[EmotionTrack]:
    try:
        return EmotionTrack.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def save_cached(path: Path, track: EmotionTrack) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(track.to_dict(), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
