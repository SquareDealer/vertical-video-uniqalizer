"""Decoding an uploaded file to 16 kHz mono float samples."""

import io
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 16000


def decode(data: bytes) -> np.ndarray:
    """16 kHz mono float32 in -1..1. A PCM16 WAV of that format is read directly, anything else via ffmpeg.

    Raises ``ValueError`` when the bytes are not audio or hold no samples.
    """
    if not data:
        raise ValueError("the file is empty")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == 1 and wav.getsampwidth() == 2:
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
                if samples.size == 0:
                    raise ValueError("the audio has no samples")
                return samples
    except (wave.Error, EOFError):
        pass  # not a plain WAV: let ffmpeg try
    try:
        proc = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                               "-ar", str(SAMPLE_RATE), "-"], input=data, capture_output=True, timeout=300)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed in the service image") from exc
    out = proc.stdout[: len(proc.stdout) - len(proc.stdout) % 4]
    if proc.returncode != 0 or not out:
        raise ValueError("the file is not audio that ffmpeg can read")
    return np.frombuffer(out, dtype=np.float32).copy()
