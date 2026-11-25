"""A stand-in for the emotion server (services/emotion) for tests: same HTTP contract, no model."""

import io
import json
import re
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _neutral(labels):
    return lambda start, end: {name: (1.0 if name.lower() == "neutral" else 0.0) for name in labels}


class FakeEmotionServer:
    """Context manager around ``http.server`` on ``127.0.0.1:<free port>``.

    ``.url`` is the address, ``.requests`` a list of ``(method, path, body_bytes)``. ``probs(start, end)``
    answers the probabilities of one window, keyed like ``labels``.
    """

    def __init__(self, labels=("angry", "sad", "neutral", "positive"), health_status=200, post_status=200,
                 probs=None, model="gigaam-emo"):
        self.labels = tuple(labels)
        self.health_status = health_status
        self.post_status = post_status
        self.probs = probs or _neutral(self.labels)
        self.model = model
        self.requests = []
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the test output clean
                pass

            def _send(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.requests.append(("GET", self.path, b""))
                if self.path != "/health":
                    return self._send(404, {"detail": "not found"})
                if fake.health_status != 200:
                    return self._send(fake.health_status, {"status": "loading"})
                self._send(200, {"status": "ok", "model": fake.model, "labels": list(fake.labels)})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.requests.append(("POST", self.path, body))
                if self.path != "/v1/audio/emotions":
                    return self._send(404, {"detail": "not found"})
                if fake.post_status != 200:
                    return self._send(fake.post_status, {"detail": f"fake error {fake.post_status}"})
                window = float(_field(body, "window") or 3.0)
                hop = float(_field(body, "hop") or 1.5)
                duration = _wav_seconds(body)
                windows, k = [], 0
                while k * hop < duration:
                    start, end = k * hop, min(k * hop + window, duration)
                    windows.append({"start": round(start, 3), "end": round(end, 3), "probs": fake.probs(start, end)})
                    k += 1
                self._send(200, {"model": fake.model, "labels": list(fake.labels), "windows": windows})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _field(body: bytes, name: str):
    match = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n([^\r]*)\r\n', body)
    return match.group(1).decode() if match else None


def _wav_seconds(body: bytes) -> float:
    start = body.find(b"RIFF")
    end = body.rfind(b"\r\n--")
    with wave.open(io.BytesIO(body[start:end]), "rb") as wav:
        return wav.getnframes() / wav.getframerate()
