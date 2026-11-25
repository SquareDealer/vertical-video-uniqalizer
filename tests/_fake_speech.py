"""A stand-in for the speech server (Kamerton): OpenAI-style models and transcriptions endpoints."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeSpeechServer:
    """Answers every transcription with the given ``(start, end, text)`` segments (words spread evenly)."""

    def __init__(self, segments):
        self.segments = list(segments)
        self.requests = []
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _answer(self) -> dict:
        segments, words = [], []
        for start, end, text in self.segments:
            tokens = text.split()
            step = (end - start) / len(tokens)
            segments.append({"start": start, "end": end, "text": text})
            words += [{"word": token, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step - 0.05, 3)}
                      for i, token in enumerate(tokens)]
        return {"language": "russian", "text": " ".join(t for _, _, t in self.segments),
                "segments": segments, "words": words}

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.requests.append(("GET", self.path))
                self._send({"object": "list", "data": [{"id": "kamerton", "variant": "fake", "deployed": True}]})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.requests.append(("POST", self.path))
                self._send(fake._answer())

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
