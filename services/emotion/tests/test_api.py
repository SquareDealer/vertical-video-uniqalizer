import io
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ave_emotion.app import create_app


class FakeModel:
    labels = ["angry", "sad", "neutral", "positive"]

    def probs(self, samples):
        positive = float(np.count_nonzero(samples)) / max(1, samples.size)
        return {"angry": 0.0, "sad": 0.0, "neutral": 1.0 - positive, "positive": positive}


def wav_bytes(seconds, amplitude=0.3, rate=16000):
    samples = (np.full(int(seconds * rate), amplitude) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


def ready(client, status=200, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/health")
        if response.status_code == status and response.json().get("status") != "loading":
            return response
        time.sleep(0.02)
    raise AssertionError(f"/health never reached {status}: {client.get('/health').json()}")


def post(client, data, **fields):
    return client.post("/v1/audio/emotions", files={"file": ("a.wav", data, "audio/wav")}, data=fields)


@pytest.fixture()
def client():
    with TestClient(create_app(lambda: FakeModel())) as client:
        ready(client)
        yield client


def test_health_reports_loading_then_ok():
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return FakeModel()

    with TestClient(create_app(slow)) as client:
        loading = client.get("/health")
        assert loading.status_code == 503 and loading.json() == {"status": "loading"}
        gate.set()
        ok = ready(client)
        assert ok.json() == {"status": "ok", "model": "gigaam-emo", "labels": ["angry", "sad", "neutral", "positive"]}


def test_emotions_contract_on_a_short_wav(client):
    response = post(client, wav_bytes(6))
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gigaam-emo" and body["labels"] == ["angry", "sad", "neutral", "positive"]
    assert [(w["start"], w["end"]) for w in body["windows"]] == [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0), (4.5, 6.0)]
    assert set(body["windows"][0]["probs"]) == set(body["labels"])
    assert body["windows"][0]["probs"]["positive"] == pytest.approx(1.0)


def test_window_and_hop_fields_are_used(client):
    body = post(client, wav_bytes(6), window="2", hop="2").json()
    assert [(w["start"], w["end"]) for w in body["windows"]] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]


def test_bad_file_is_400_and_bad_params_are_400(client):
    bad = post(client, b"not audio")
    assert bad.status_code == 400 and "detail" in bad.json()
    assert post(client, wav_bytes(3), window="0").status_code == 400
    assert post(client, wav_bytes(3), window="2", hop="3").status_code == 400
    assert post(client, wav_bytes(3), window="2", hop="0.01").status_code == 400


def test_503_while_the_model_is_loading():
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return FakeModel()

    with TestClient(create_app(slow)) as client:
        response = post(client, wav_bytes(3))
        assert response.status_code == 503
        gate.set()


def test_failed_load_is_reported():
    def broken():
        raise RuntimeError("boom")

    with TestClient(create_app(broken)) as client:
        response = ready(client, status=503)
        assert response.json()["status"] == "error" and "boom" in response.json()["detail"]
        assert post(client, wav_bytes(3)).status_code == 503
