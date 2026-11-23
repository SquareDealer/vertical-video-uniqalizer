"""Speech transcription with word timestamps (faster-whisper or openai-whisper)."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .audio import SAMPLE_RATE
from .ffmpeg import FFmpegError, pcm_reader_command, popen_binary
from .reporting import Cancelled
from .speakers import voice_features

CHUNK_SECONDS = 600.0   # transcribe long recordings in 10 minute windows
OVERLAP_SECONDS = 30.0  # extra audio so sentences at a window edge are not cut



# CUDA runtime libraries CTranslate2 loads on Windows; without them GPU inference fails or hangs.
_WINDOWS_CUDA_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")


@lru_cache(maxsize=1)
def cuda_compute_type() -> Optional[str]:
    """Best compute type for faster-whisper on the GPU, or None when the GPU cannot be used.

    Older cards (Pascal, e.g. GTX 10xx) have no efficient float16, so ``int8`` is used there.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return None
        supported = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:  # noqa: BLE001
        return None
    if os.name == "nt":
        import ctypes

        for dll in _WINDOWS_CUDA_DLLS:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                return None
    for compute_type in ("float16", "int8_float16", "int8"):
        if compute_type in supported:
            return compute_type
    return None


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0
    speaker: int = 0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: int = 0
    embedding: Optional[List[Optional[float]]] = None  # voice signature, see speakers.py

    def set_speaker(self, speaker: int) -> None:
        self.speaker = speaker
        for word in self.words:
            word.speaker = speaker


@dataclass
class Transcript:
    segments: List[Segment]
    language: str = ""
    backend: str = ""
    model: str = ""

    def words(self) -> Iterator[Word]:
        for segment in self.segments:
            yield from segment.words

    def words_between(self, start: float, end: float) -> List[Word]:
        return [w for w in self.words() if w.end > start and w.start < end]

    @property
    def speaker_count(self) -> int:
        return max((s.speaker for s in self.segments), default=0) + 1 if self.segments else 0

    def text_between(self, start: float, end: float) -> str:
        """Text spoken between two times; speakers are labelled (A:, B:) when there are several."""
        words = self.words_between(start, end)
        if not words:
            return " ".join(s.text.strip() for s in self.segments if s.end > start and s.start < end)
        if self.speaker_count <= 1:
            return _join_words(words)
        parts, current, run = [], None, []
        for w in words:
            if w.speaker != current and run:
                parts.append(f"{speaker_label(current)}: {_join_words(run)}")
                run = []
            current = w.speaker
            run.append(w)
        parts.append(f"{speaker_label(current)}: {_join_words(run)}")
        return "  ".join(parts)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker, "embedding": s.embedding,
                 "words": [[round(w.start, 3), round(w.end, 3), w.text, round(w.probability, 3)] for w in s.words]}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        segments = []
        for s in data.get("segments", []):
            segment = Segment(s["start"], s["end"], s["text"], [Word(*w[:4]) for w in s.get("words", [])],
                              embedding=s.get("embedding"))
            segment.set_speaker(int(s.get("speaker", 0)))
            segments.append(segment)
        return cls(segments, data.get("language", ""), data.get("backend", ""), data.get("model", ""))


def speaker_label(speaker: int) -> str:
    """0 -> A, 1 -> B, ..."""
    return chr(ord("A") + speaker) if 0 <= speaker < 26 else f"S{speaker + 1}"


def _join_words(words: List[Word]) -> str:
    # Whisper word tokens carry their own leading spaces.
    return "".join(w.text for w in words).strip()


# -- prompt -----------------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "fr": "On est en live, on discute et on réagit en direct.",
    "en": "We're live, chatting and reacting in real time.",
    "es": "Estamos en directo, charlando y reaccionando.",
    "de": "Wir sind live, wir quatschen und reagieren.",
    "it": "Siamo in diretta, chiacchieriamo e reagiamo.",
    "pt": "Estamos ao vivo, conversando e reagindo.",
}
VOCABULARY_LABELS = {"fr": "Vocabulaire", "es": "Vocabulario", "de": "Wortschatz", "it": "Vocabolario",
                     "pt": "Vocabulário"}


def build_prompt(language: Optional[str], vocabulary: Sequence[str] = (), profanity: bool = True) -> str:
    """Text the speech model is conditioned on: it nudges spelling towards the given vocabulary
    (game names, jargon, foreign words) and stops the model from silently skipping swear words."""
    from .censor import PROMPT_HINTS

    lang = (language or "en").lower()
    parts = [PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])]
    words = [w.strip() for w in vocabulary if w.strip()]
    if words:
        parts.append(f"{VOCABULARY_LABELS.get(lang, 'Vocabulary')}: {', '.join(words)}.")
    if profanity:
        parts.append(PROMPT_HINTS.get(lang, PROMPT_HINTS["en"]))
    return " ".join(parts)


# -- backends ---------------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> List[str]:
    backends = []
    if _has_module("faster_whisper"):
        backends.append("faster-whisper")
    if _has_module("whisper"):
        backends.append("openai-whisper")
    return backends


def resolve_backend(preference: str) -> Optional[str]:
    """Return the backend to use, or ``None`` when transcription is disabled/unavailable."""
    if preference == "none":
        return None
    if preference == HTTP_BACKEND:
        return HTTP_BACKEND
    available = available_backends()
    if preference == "auto":
        return available[0] if available else None
    if preference not in available:
        package = "faster-whisper" if preference == "faster-whisper" else "openai-whisper"
        raise RuntimeError(f"The '{preference}' transcriber is not installed. Run: pip install {package}")
    return preference


class Transcriber:
    def __init__(self, backend: str, model: str = "base", language: str = "",
                 log: Callable[[str], None] = print,
                 cancel_event: Optional[threading.Event] = None, beam_size: int = 5,
                 vocabulary: Sequence[str] = (), profanity: bool = True) -> None:
        self.beam_size = max(1, int(beam_size))
        self.vocabulary = list(vocabulary)
        self.profanity = profanity
        self.backend = backend
        self.model_name = model
        self.language = language.strip() or None
        self.log = log
        self.cancel_event = cancel_event or threading.Event()
        self.audio_track, self.track_count = 1, 1  # track to listen to (1-based, 0 = all mixed)
        self.server: Optional[SpeechServer] = None  # set for the "http" backend
        self._model = None

    # -- model loading --------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend != HTTP_BACKEND:
            self.log(f"Loading the '{self.model_name}' speech model ({self.backend}). "
                     "The first run downloads it, please be patient...")
        if self.backend == HTTP_BACKEND:
            if self.server is None:
                raise RuntimeError("The http transcriber needs a server address (stt_url).")
            self._model = self.server
            self.log(f"Speech recognition server: {self.server.describe()}")
            return
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            device, compute_type = "cpu", "int8"
            gpu_type = cuda_compute_type()
            if gpu_type:
                device, compute_type = "cuda", gpu_type
            self.log(f"Speech model runs on {device.upper()} ({compute_type}).")
            threads = max(1, (os.cpu_count() or 2) - 1)
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                                       cpu_threads=threads)
        else:
            import whisper

            name = "large-v3" if self.model_name == "large-v3" else self.model_name
            self._model = whisper.load_model(name)

    # -- transcription --------------------------------------------------------------------
    def transcribe(self, path: str, duration: float,
                   on_progress: Optional[Callable[[float], None]] = None) -> Transcript:
        self._load()
        segments: List[Segment] = []
        language = self.language
        if language is None:
            language = self._detect_language(path, duration)
            if language:
                self.log(f"Detected language: {language}")
        start = 0.0
        while start < duration - 0.5:
            self._check_cancel()
            window = min(CHUNK_SECONDS + OVERLAP_SECONDS, duration - start)
            is_last = start + window >= duration - 0.5
            audio = self._decode(path, start, window)

            def chunk_progress(t: float, _start=start) -> None:
                if on_progress and duration:
                    on_progress(min(1.0, (_start + min(t, CHUNK_SECONDS)) / duration))

            chunk, detected = self._transcribe_array(audio, language, chunk_progress)
            if language is None and detected:
                language = detected  # lock the language after the first window
                self.log(f"Detected language: {language}")

            limit = CHUNK_SECONDS
            kept_end = None
            for seg in chunk:
                if not is_last and seg.start >= limit:
                    break
                seg.embedding = voice_features(audio, seg.start, seg.end)
                segments.append(_offset_segment(seg, start))
                kept_end = seg.end
            if is_last:
                break
            next_start = start + CHUNK_SECONDS
            if kept_end is not None:
                next_start = max(next_start, start + kept_end)
            start = next_start
            if on_progress:
                on_progress(min(1.0, start / duration))

        if on_progress:
            on_progress(1.0)
        return Transcript(_clean_segments(segments), language or "", self.backend, self.model_name)

    def _transcribe_array(self, audio: np.ndarray, language: Optional[str],
                          on_progress: Callable[[float], None]):
        if audio.size < SAMPLE_RATE // 2:
            return [], None
        if self.backend == HTTP_BACKEND:
            self._check_cancel()
            result, detected = self.server.transcribe(audio, language, words=True)
            self._check_cancel()
            if result:
                on_progress(result[-1].end)
            return result, detected
        prompt = build_prompt(language, self.vocabulary, self.profanity)
        if self.backend == "faster-whisper":
            # No VAD: Silero drops real speech over game audio/music. Hallucinations in
            # silent parts are filtered afterwards with the loudness envelope instead.
            # "hotwords" is applied to every 30 s window (initial_prompt only to the first one).
            options = dict(language=language, word_timestamps=True, vad_filter=False,
                           condition_on_previous_text=False, beam_size=self.beam_size)
            try:
                raw_segments, info = self._model.transcribe(audio, hotwords=prompt, **options)
            except TypeError:  # faster-whisper < 1.0.2
                raw_segments, info = self._model.transcribe(audio, initial_prompt=prompt, **options)
            result = []
            for seg in raw_segments:  # lazy generator: decoding happens here
                self._check_cancel()
                if getattr(seg, "no_speech_prob", 0) > 0.8 and getattr(seg, "avg_logprob", 0) < -1.0:
                    continue
                words = [Word(w.start, w.end, w.word, w.probability) for w in (seg.words or [])]
                result.append(Segment(seg.start, seg.end, seg.text.strip(), words))
                on_progress(seg.end)
            return result, info.language

        import torch

        # openai-whisper only keeps initial_prompt in context when it conditions on previous text.
        output = self._model.transcribe(
            audio, language=language, word_timestamps=True, verbose=None, initial_prompt=prompt,
            condition_on_previous_text=True, fp16=torch.cuda.is_available(), beam_size=self.beam_size,
        )
        result = []
        for seg in output.get("segments", []):
            if seg.get("no_speech_prob", 0) > 0.8 and seg.get("avg_logprob", 0) < -1.0:
                continue
            words = [Word(w["start"], w["end"], w["word"], w.get("probability", 1.0))
                     for w in seg.get("words", [])]
            result.append(Segment(seg["start"], seg["end"], seg["text"].strip(), words))
        return result, output.get("language")

    def preview(self, path: str, duration: float, window: float = 30.0,
                on_progress: Optional[Callable[[float], None]] = None) -> Tuple[Optional[str], str]:
        """Quickly transcribe three short windows spread over the recording.

        Returns the dominant language (a vote, more robust than one sample) and the text heard,
        used to recognise the game or topic before the full transcription.
        """
        self._load()
        if duration <= window * 1.5:
            positions = [0.0]
        else:
            positions = [max(0.0, min(duration - window, duration * f)) for f in (0.2, 0.5, 0.8)]
        votes: Dict[str, float] = {}
        texts: List[str] = []
        for index, start in enumerate(positions):
            self._check_cancel()
            audio = self._decode(path, start, min(window, duration))
            if audio.size >= SAMPLE_RATE:
                try:
                    text, language, confidence = self._quick_transcribe(audio)
                except Exception as exc:  # noqa: BLE001 - the preview is optional
                    self.log(f"Speech preview skipped ({exc}).")
                    text, language, confidence = "", None, 0.0
                if text:
                    texts.append(text)
                if language:
                    votes[language] = votes.get(language, 0.0) + confidence * max(1, len(text))
            if on_progress:
                on_progress((index + 1) / len(positions))
        return (max(votes, key=votes.get) if votes else None), " ".join(texts)

    def _quick_transcribe(self, audio: np.ndarray) -> Tuple[str, Optional[str], float]:
        if self.backend == HTTP_BACKEND:
            segments, language = self.server.transcribe(audio, self.language, words=False)
            return " ".join(seg.text for seg in segments).strip(), language, 1.0
        if self.backend == "faster-whisper":
            segments, info = self._model.transcribe(audio, language=self.language, beam_size=1, vad_filter=False,
                                                    condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(seg.text.strip() for seg in segments)
            return text, info.language, float(getattr(info, "language_probability", 1.0) or 1.0)
        import torch

        output = self._model.transcribe(audio, language=self.language, fp16=torch.cuda.is_available(),
                                        condition_on_previous_text=False, verbose=None)
        return output.get("text", "").strip(), output.get("language"), 1.0

    def _detect_language(self, path: str, duration: float) -> Optional[str]:
        """Detect the language on 30 s taken from the first quarter of the recording (intros are
        often silent or music only)."""
        start = max(0.0, min(duration * 0.25, duration - 30.0))
        audio = self._decode(path, start, min(30.0, duration))
        if audio.size < SAMPLE_RATE:
            return None
        try:
            if self.backend == HTTP_BACKEND:
                return self._quick_transcribe(audio)[1]
            if self.backend == "faster-whisper":
                detect = getattr(self._model, "detect_language", None)
                if detect is not None:
                    language, _prob, _all = detect(audio=audio)
                    return language
                _segments, info = self._model.transcribe(audio, beam_size=1)
                return info.language
            import whisper

            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=self._model.dims.n_mels)
            mel = mel.to(self._model.device)
            _tokens, probs = self._model.detect_language(mel)
            return max(probs, key=probs.get)
        except Exception as exc:  # noqa: BLE001 - fall back to per-chunk detection
            self.log(f"Language detection skipped ({exc}).")
            return None

    def _decode(self, path: str, start: float, duration: float) -> np.ndarray:
        return _decode_pcm(path, start, duration, self.cancel_event, self.audio_track, self.track_count)

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled by user")


# -- remote speech server ----------------------------------------------------------------------

HTTP_BACKEND = "http"


class SpeechServerUnavailable(RuntimeError):
    """The speech server cannot be reached or its engine is not ready (502/503). Worth retrying later."""
HTTP_TIMEOUT = 900.0  # one 10 minute window on a slow CPU server


@dataclass
class SpeechServer:
    """An OpenAI-compatible speech-to-text server (``POST {url}/audio/transcriptions``), e.g. Kamerton.

    The audio is sent as 16 kHz mono WAV, one transcription window at a time, and the answer is read
    as ``verbose_json`` with word timestamps.
    """

    url: str = "http://localhost:8091/v1"
    api_key: str = ""
    model: str = "kamerton"
    timeout: float = HTTP_TIMEOUT

    def __post_init__(self) -> None:
        url = (self.url or "").strip().rstrip("/")
        if url and "://" not in url:
            url = "http://" + url
        if url.endswith("/audio/transcriptions"):
            url = url[: -len("/audio/transcriptions")]
        if url and not url.endswith("/v1"):
            url += "/v1"
        self.url = url

    def describe(self) -> str:
        return f"{self.url} (model {self.model or 'default'})"

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "auto-video-editor"}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        return headers

    def _request(self, path: str, data: Optional[bytes] = None,
                 content_type: Optional[str] = None, timeout: Optional[float] = None) -> dict:
        import urllib.error
        import urllib.request

        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.url + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            try:
                parsed = json.loads(detail)
                error = parsed.get("error", parsed)
                detail = error.get("message", detail) if isinstance(error, dict) else str(error)
            except ValueError:
                pass
            hint = " Check the API key." if exc.code in (401, 403) else ""
            error = SpeechServerUnavailable if exc.code in (502, 503) else RuntimeError
            raise error(f"Speech server error {exc.code} at {self.url}{path}: {detail}{hint}") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise SpeechServerUnavailable(f"Cannot reach the speech server at {self.url} ({reason}). "
                               "Is it running? For Kamerton: docker compose up -d in its folder.") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"The speech server at {self.url} did not answer with JSON.") from exc

    def check(self) -> str:
        """Ping ``GET /models``; returns a short description of what the server serves."""
        data = self._request("/models", timeout=10)
        models = data.get("data") or []
        if not models:
            return "reachable"
        first = models[0]
        parts = [str(first.get("id", "?"))]
        if first.get("variant"):
            parts.append(f"variant {first['variant']}")
        if first.get("deployed") is False:
            parts.append("no model deployed yet")
        return ", ".join(parts)

    def transcribe(self, audio: np.ndarray, language: Optional[str],
                   words: bool = True) -> Tuple[List[Segment], Optional[str]]:
        fields: List[Tuple[str, str]] = [("model", self.model or "whisper-1"),
                                         ("response_format", "verbose_json")]
        if words:
            fields += [("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
        if language:
            fields.append(("language", language))
        body, content_type = _multipart(fields, "audio.wav", _wav_bytes(audio))
        data = self._request("/audio/transcriptions", body, content_type)
        segments = _segments_from_response(data, len(audio) / SAMPLE_RATE)
        detected = _language_code(data.get("language")) or language or _guess_language(
            " ".join(seg.text for seg in segments))
        return segments, detected


def _wav_bytes(audio: np.ndarray) -> bytes:
    import io
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def _multipart(fields: Sequence[Tuple[str, str]], filename: str, payload: bytes) -> Tuple[bytes, str]:
    import uuid

    boundary = f"----auto-video-editor-{uuid.uuid4().hex}"
    parts: List[bytes] = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
                     .encode("utf-8"))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode("utf-8"))
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


_LANGUAGE_NAMES = {"russian": "ru", "english": "en", "french": "fr", "german": "de", "spanish": "es",
                   "italian": "it", "portuguese": "pt", "kazakh": "kk", "kyrgyz": "ky", "uzbek": "uz",
                   "ukrainian": "uk"}


def _language_code(value) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip().lower()
    return _LANGUAGE_NAMES.get(value, value if len(value) <= 3 else None)


def _guess_language(text: str) -> Optional[str]:
    """Servers that do not report the language (GigaAM): tell Cyrillic from Latin script."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return None
    cyrillic = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    return "ru" if cyrillic > len(letters) / 2 else None


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _word_from_json(item: dict) -> Optional[Word]:
    text = str(item.get("word", item.get("text", ""))).strip()
    if not text:
        return None
    start, end = _float(item.get("start")), _float(item.get("end"))
    probability = _float(item.get("probability", item.get("confidence", 1.0)), 1.0)
    # Whisper tokens carry their own leading space, _join_words relies on it.
    return Word(start, max(end, start), " " + text, probability)


def _segments_from_response(data: dict, duration: float) -> List[Segment]:
    """Turn an OpenAI ``verbose_json`` answer into segments that all have word timings."""
    raw_segments = [s for s in (data.get("segments") or []) if isinstance(s, dict)]
    words = [w for w in (_word_from_json(item) for item in (data.get("words") or []) if isinstance(item, dict)) if w]
    segments: List[Segment] = []
    for raw in raw_segments:
        text = str(raw.get("text", "")).strip()
        start, end = _float(raw.get("start")), _float(raw.get("end"))
        inner = [w for w in (_word_from_json(i) for i in (raw.get("words") or []) if isinstance(i, dict)) if w]
        segments.append(Segment(start, max(end, start), text, inner))
    if words:
        if segments and not any(seg.words for seg in segments):
            for word in words:  # top-level words: give each one to the segment it falls in
                middle = (word.start + word.end) / 2
                target = min(segments, key=lambda seg: 0 if seg.start <= middle <= seg.end
                             else min(abs(middle - seg.start), abs(middle - seg.end)))
                target.words.append(word)
        elif not segments:
            segments = _sentences_from_words(words)
    if not segments and str(data.get("text", "")).strip():
        segments = [Segment(0.0, duration, str(data["text"]).strip())]
    for seg in segments:
        if not seg.words and seg.text:
            seg.words = _spread_words(seg.text, seg.start, seg.end)
        for word, following in zip(seg.words, seg.words[1:] + [None]):
            if word.end - word.start < 0.05:  # GigaAM gives short words no length: captions would flash
                limit = following.start if following is not None else seg.end
                word.end = max(word.end, min(word.start + 0.15, limit))
        if seg.words:
            seg.start = min(seg.start, seg.words[0].start) if seg.end > seg.start else seg.words[0].start
            seg.end = max(seg.end, seg.words[-1].end)
    return [seg for seg in segments if seg.text or seg.words]


def _sentences_from_words(words: List[Word], max_gap: float = 1.2, max_length: float = 15.0) -> List[Segment]:
    segments: List[Segment] = []
    run: List[Word] = []
    for word in words:
        if run and (word.start - run[-1].end > max_gap or word.end - run[0].start > max_length):
            segments.append(Segment(run[0].start, run[-1].end, _join_words(run), run))
            run = []
        run.append(word)
        if word.text.rstrip()[-1:] in ".!?…":
            segments.append(Segment(run[0].start, run[-1].end, _join_words(run), run))
            run = []
    if run:
        segments.append(Segment(run[0].start, run[-1].end, _join_words(run), run))
    return segments


def _spread_words(text: str, start: float, end: float) -> List[Word]:
    """No word timings from the server: share the segment time in proportion to word length."""
    tokens = text.split()
    if not tokens or end <= start:
        return []
    total = sum(len(t) + 1 for t in tokens)
    words, cursor = [], start
    for token in tokens:
        step = (end - start) * (len(token) + 1) / total
        words.append(Word(cursor, cursor + step, " " + token, 1.0))
        cursor += step
    return words


def _offset_segment(seg: Segment, offset: float) -> Segment:
    return Segment(
        seg.start + offset, seg.end + offset, seg.text,
        [Word(w.start + offset, w.end + offset, w.text, w.probability) for w in seg.words],
        embedding=seg.embedding,
    )


def _clean_segments(segments: List[Segment]) -> List[Segment]:
    """Drop empty segments and the repeated-phrase loops Whisper sometimes hallucinates."""
    cleaned: List[Segment] = []
    repeats = 0
    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.text.strip() or seg.end <= seg.start:
            continue
        if cleaned and seg.text.strip().lower() == cleaned[-1].text.strip().lower():
            repeats += 1
            if repeats >= 2:
                continue
        else:
            repeats = 0
        if cleaned and seg.start < cleaned[-1].end - 0.5:
            # Overlap from a chunk boundary: skip exact duplicates.
            if seg.text.strip() == cleaned[-1].text.strip():
                continue
        seg.words = _drop_stutter(w for w in seg.words if w.text.strip() and w.end >= w.start)
        if segments and not seg.words:
            continue
        seg.text = _join_words(seg.words) if seg.words else seg.text
        cleaned.append(seg)
    return cleaned


def _drop_stutter(words) -> List[Word]:
    """Remove hallucinated repetitions: a word repeated 3+ times whose first copy has ~zero
    confidence ("fuck, fuck, fuck, fuck" over a shout), and cap real ones at 6 in a row."""
    words = list(words)
    result: List[Word] = []
    i = 0
    while i < len(words):
        key = _normalize_token(words[i].text)
        j = i
        while j < len(words) and _normalize_token(words[j].text) == key:
            j += 1
        run = words[i:j]
        if len(run) >= 3 and run[0].probability < 0.2:
            run = []
        result.extend(run[:6])
        i = j
    return result


def _normalize_token(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _decode_pcm(path: str, start: float, duration: float, cancel_event: Optional[threading.Event],
                track: int = 1, track_count: int = 1) -> np.ndarray:
    proc = popen_binary(pcm_reader_command(path, SAMPLE_RATE, start, duration, track, track_count))
    try:
        data, err = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    if proc.returncode != 0:
        raise FFmpegError(f"Audio decoding failed: {err.decode('utf-8', 'replace')[-800:]}")
    return np.frombuffer(data[: len(data) - len(data) % 4], dtype=np.float32).copy()


# -- cache --------------------------------------------------------------------------------------

def load_cached(path: Path) -> Optional[Transcript]:
    try:
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
