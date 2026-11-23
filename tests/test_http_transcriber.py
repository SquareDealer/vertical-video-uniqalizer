"""The "http" transcriber: an OpenAI-compatible speech server such as Kamerton."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from auto_video_editor.transcribe import (SpeechServer, Transcriber, _guess_language, _segments_from_response,
                                          resolve_backend)


def test_resolve_backend_needs_no_package():
    assert resolve_backend("http") == "http"


@pytest.mark.parametrize("url", ["localhost:8091", "http://localhost:8091/", "http://localhost:8091/v1",
                                 "http://localhost:8091/v1/audio/transcriptions"])
def test_server_url_is_normalised(url):
    assert SpeechServer(url).url == "http://localhost:8091/v1"


def test_openai_top_level_words_go_to_their_segment():
    data = {"language": "russian", "text": "привет мир. как дела",
            "segments": [{"start": 0.0, "end": 1.5, "text": "Привет мир."},
                         {"start": 2.0, "end": 3.0, "text": "Как дела"}],
            "words": [{"word": "Привет", "start": 0.1, "end": 0.6}, {"word": "мир.", "start": 0.7, "end": 1.4},
                      {"word": "Как", "start": 2.0, "end": 2.3}, {"word": "дела", "start": 2.4, "end": 2.9}]}
    segments = _segments_from_response(data, 3.0)
    assert [len(s.words) for s in segments] == [2, 2]
    assert segments[1].words[0].text == " Как"


def test_words_only_are_grouped_into_sentences():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate(["Да.", "Нет", "конечно"])]
    segments = _segments_from_response({"words": words}, 2.0)
    assert [s.text for s in segments] == ["Да.", "Нет конечно"]


def test_segments_without_words_get_spread_timings():
    segments = _segments_from_response({"segments": [{"start": 1.0, "end": 3.0, "text": "раз два"}]}, 4.0)
    words = segments[0].words
    assert len(words) == 2 and words[0].start == 1.0 and words[-1].end == pytest.approx(3.0)


def test_plain_text_answer_still_gives_a_segment():
    segments = _segments_from_response({"text": "просто текст"}, 5.0)
    assert segments[0].start == 0.0 and segments[0].end == 5.0 and len(segments[0].words) == 2


def test_language_guess_from_script():
    assert _guess_language("Ну что, погнали играть дальше") == "ru"
    assert _guess_language("let's go, that was insane") is None


class _FakeServer(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply(200, {"object": "list", "data": [{"id": "kamerton", "variant": "e2e_rnnt", "deployed": True}]})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).requests.append((self.path, self.headers.get("Authorization"), body))
        if self.headers.get("Authorization") != "Bearer secret":
            self._reply(401, {"error": {"message": "bad key"}})
            return
        self._reply(200, {"text": "Погнали дальше", "duration": 2.0,
                          "segments": [{"start": 0.2, "end": 1.6, "text": "Погнали дальше"}],
                          "words": [{"word": "Погнали", "start": 0.2, "end": 0.8},
                                    {"word": "дальше", "start": 0.9, "end": 1.6}]})


@pytest.fixture()
def fake_server():
    _FakeServer.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_transcribe_through_the_server(fake_server):
    server = SpeechServer(fake_server, "secret", "kamerton")
    assert server.check() == "kamerton, variant e2e_rnnt"
    transcriber = Transcriber("http", "kamerton", "", log=lambda _m: None)
    transcriber.server = server
    transcriber._load()
    audio = (0.1 * np.sin(np.linspace(0, 2000, 32000))).astype(np.float32)
    segments, language = transcriber._transcribe_array(audio, None, lambda _t: None)
    assert language == "ru"
    assert [w.text for w in segments[0].words] == [" Погнали", " дальше"]
    path, auth, body = _FakeServer.requests[-1]
    assert path == "/v1/audio/transcriptions" and auth == "Bearer secret"
    assert b'name="response_format"\r\n\r\nverbose_json' in body
    assert b'name="timestamp_granularities[]"\r\n\r\nword' in body
    assert b"RIFF" in body and b"WAVE" in body


def test_wrong_key_is_reported(fake_server):
    with pytest.raises(RuntimeError, match="401.*bad key.*API key"):
        SpeechServer(fake_server, "nope").transcribe(np.zeros(16000, np.float32), None)


def test_unreachable_server_is_reported():
    with pytest.raises(RuntimeError, match="Cannot reach the speech server"):
        SpeechServer("http://127.0.0.1:9", timeout=2).check()


def test_zero_length_words_get_a_short_duration():
    data = {"segments": [{"start": 0.0, "end": 1.0, "text": "в дом"}],
            "words": [{"word": "в", "start": 0.2, "end": 0.2}, {"word": "дом", "start": 0.3, "end": 0.8}]}
    words = _segments_from_response(data, 1.0)[0].words
    assert words[0].end == pytest.approx(0.3)
