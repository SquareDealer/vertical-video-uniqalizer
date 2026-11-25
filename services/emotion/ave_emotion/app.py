"""HTTP API of the emotion service: ``GET /health`` and ``POST /v1/audio/emotions``."""

import threading
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .audio import SAMPLE_RATE, decode
from .model import EmotionModel, load_gigaam
from .windows import window_bounds

MODEL_NAME = "gigaam-emo"
MAX_SECONDS = 900.0


class _State:
    def __init__(self) -> None:
        self.model: Optional[EmotionModel] = None
        self.error: Optional[str] = None
        self.lock = threading.Lock()   # one request at a time uses the CPUs


def create_app(loader: Callable[[], EmotionModel] = load_gigaam) -> FastAPI:
    state = _State()

    def load() -> None:
        try:
            state.model = loader()
        except Exception as exc:  # noqa: BLE001 - reported by /health
            state.error = f"{type(exc).__name__}: {exc}"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=load, name="emotion-model-loader", daemon=True).start()
        yield

    app = FastAPI(title="ave-emotion", lifespan=lifespan)

    @app.get("/health")
    def health():
        if state.model is not None:
            return {"status": "ok", "model": MODEL_NAME, "labels": list(state.model.labels)}
        if state.error:
            return JSONResponse({"status": "error", "detail": state.error}, status_code=503)
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.post("/v1/audio/emotions")
    def emotions(file: UploadFile = File(...), window: float = Form(3.0), hop: float = Form(1.5)):
        model = state.model
        if model is None:
            raise HTTPException(503, detail=state.error or "the model is still loading")
        if not 0.5 <= window <= 30:
            raise HTTPException(400, detail="window must be between 0.5 and 30 seconds")
        if not 0.1 <= hop <= window:
            raise HTTPException(400, detail="hop must be between 0.1 seconds and window")
        try:
            samples = decode(file.file.read())
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc))
        duration = samples.size / SAMPLE_RATE
        if duration > MAX_SECONDS:
            raise HTTPException(400, detail=f"the audio is {duration:.0f} s long, at most {MAX_SECONDS:.0f} s")
        windows = []
        with state.lock:
            for start, end in window_bounds(duration, window, hop):
                piece = samples[int(round(start * SAMPLE_RATE)): int(round(end * SAMPLE_RATE))]
                windows.append({"start": round(start, 3), "end": round(end, 3), "probs": model.probs(piece)})
        return {"model": MODEL_NAME, "labels": list(model.labels), "windows": windows}

    return app


app = create_app()
