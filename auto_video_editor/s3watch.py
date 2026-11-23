"""Auto mode: poll ``montage/<streamer>/`` in S3 and publish TikTok versions of every new clip.

Integrity protocol (docs/superpowers/specs/2026-11-23-s3-watch-uniqualize-design.md):

* a clip is ready only when its manifest exists, names the current source ETag and size, and every
  version it lists exists with the listed size;
* versions are uploaded to ``uniqalized/.partial/`` and checked there, the source is checked again,
  then they are copied to their final names and the manifest is written last;
* before reprocessing, the old manifest is deleted, so a stale set is never taken for a ready one;
* nothing outside ``<streamer>/uniqalized/`` is ever written or deleted, sources are read-only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import posixpath
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional
from urllib.parse import quote

from . import __version__
from .reporting import Cancelled
from .s3store import ObjectInfo, S3Store, file_md5
from .settings import Settings
from .transcribe import SpeechServerUnavailable

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv")
MANIFEST_VERSION = 1


@dataclass
class RenderedVariant:
    path: Path
    spec: dict
    captions: bool
    duration: float = 0.0          # seconds, after cuts and speed
    source_duration: float = 0.0


RenderFn = Callable[[Path, Path, int, threading.Event], List[RenderedVariant]]


@dataclass
class OutputKeys:
    final: List[str]
    partial: List[str]
    manifest: str
    error: str
    folder: str
    stem: str


@dataclass
class Summary:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    pending: int = 0


def is_source_key(key: str, prefix: str) -> bool:
    """A clip lies exactly at ``<prefix><streamer>/<file>`` and has a video extension."""
    if not key.startswith(prefix):
        return False
    parts = key[len(prefix):].split("/")
    return (len(parts) == 2 and all(p.strip() for p in parts)
            and parts[1].lower().endswith(VIDEO_EXTENSIONS) and not parts[1].startswith("."))


def output_keys(source_key: str, subdir: str, count: int, unique: bool = False) -> OutputKeys:
    """Keys of the versions of a clip. ``unique`` keeps the extension in the name, for two sources that
    share a stem in one folder (``clip.mp4`` and ``clip.mov``)."""
    folder, name = posixpath.split(source_key)
    stem = name if unique else posixpath.splitext(name)[0]
    base = f"{folder}/{subdir.strip('/')}"
    return OutputKeys(
        final=[f"{base}/{stem}__v{i}.mp4" for i in range(1, count + 1)],
        partial=[f"{base}/.partial/{stem}__v{i}.mp4" for i in range(1, count + 1)],
        manifest=f"{base}/{stem}.manifest.json", error=f"{base}/{stem}.error.json", folder=base, stem=stem,
    )


def is_ready(store: S3Store, source: ObjectInfo, keys: OutputKeys) -> bool:
    manifest = store.get_json(keys.manifest)
    if not manifest:
        return False
    src = manifest.get("source") or {}
    if src.get("etag") != source.etag or src.get("size") != source.size:
        return False
    variants = manifest.get("variants") or []
    if [v.get("key") for v in variants] != keys.final:
        return False
    for variant in variants:
        head = store.head(variant["key"])
        if head is None or head.size != variant.get("size"):
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_transient(exc: BaseException) -> bool:
    """Network and server-side hiccups: retried on the next cycle, never counted against the clip."""
    from .s3store import TransferError

    if isinstance(exc, (ConnectionError, TimeoutError, TransferError)):
        return True
    try:
        from botocore.exceptions import BotoCoreError, ClientError, ConnectionError as BotoConnectionError
    except ImportError:  # pragma: no cover - boto3 is required by the watch mode
        return False
    if isinstance(exc, BotoConnectionError):
        return True
    if isinstance(exc, ClientError):
        response = exc.response or {}
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
        code = str((response.get("Error") or {}).get("Code", ""))
        return status >= 500 or code in ("SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable")
    return isinstance(exc, BotoCoreError) and "timeout" in type(exc).__name__.lower()


def _is_missing(exc: BaseException) -> bool:
    try:
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, ClientError) and str((exc.response.get("Error") or {}).get("Code", "")) in (
        "NoSuchKey", "404", "NotFound")


def _finals_of(store: S3Store, keys: OutputKeys) -> List[str]:
    pattern = re.compile(re.escape(f"{keys.folder}/{keys.stem}__v") + r"\d+\.mp4$")
    return [o.key for o in store.list(f"{keys.folder}/{keys.stem}__v") if pattern.match(o.key)]


def _delete_quietly(store: S3Store, keys) -> None:
    for key in keys:
        try:
            store.delete(key)
        except Exception:  # noqa: BLE001 - cleanup must not hide the real error
            pass


def process_clip(store: S3Store, source: ObjectInfo, settings: Settings, render: RenderFn, work: Path,
                 log: Callable[[str], None], cancel: Optional[threading.Event] = None,
                 unique: bool = False) -> Optional[dict]:
    """Render and publish the versions of one clip. Returns the manifest, or None when the source moved."""
    cancel = cancel or threading.Event()
    count = settings.watch_variants
    keys = output_keys(source.key, settings.watch_subdir, count, unique)
    # From now on the old set is not "ready" any more, and no version of an older file can be taken
    # for a current one: the manifest and every published version of this clip go first.
    store.delete(keys.manifest)
    for key in _finals_of(store, keys):
        store.delete(key)
    info = store.head(source.key)
    if info is None:
        log(f"  {source.key}: gone before download, skipped.")
        return None
    work.mkdir(parents=True, exist_ok=True)
    local = work / f"source{posixpath.splitext(info.key)[1].lower()}"
    try:
        store.download(info, local)
    except Exception as exc:  # noqa: BLE001
        if _is_missing(exc):
            log(f"  {source.key}: moved or deleted during download, skipped.")
            return None
        raise
    log(f"  downloaded {info.size / 1e6:.1f} MB, rendering {count} versions...")
    variants = render(local, work, count, cancel)
    if len(variants) != count:
        raise RuntimeError(f"rendered {len(variants)} versions instead of {count}")

    clip_id = info.metadata.get("clip-id") or None
    entries: List[dict] = []
    copied: List[str] = []
    published = False
    try:
        for i, variant in enumerate(variants):
            metadata = {"source-key": quote(info.key, safe="/"), "source-etag": info.etag,
                        "source-size": str(info.size), "variant": str(i + 1),
                        "manifest-version": str(MANIFEST_VERSION)}
            if clip_id:
                metadata["clip-id"] = clip_id
            uploaded = store.upload(variant.path, keys.partial[i], metadata)
            entries.append({"key": keys.final[i], "size": uploaded.size, "md5": file_md5(variant.path),
                            "sha256": _sha256(variant.path), "duration": round(variant.duration, 3),
                            "captions": variant.captions, "spec": variant.spec})
        current = store.head(info.key)
        if current is None or current.etag != info.etag or current.size != info.size:
            log(f"  {info.key}: the source changed or disappeared during processing, versions discarded.")
            return None
        for partial, entry in zip(keys.partial, entries):
            copied.append(entry["key"])
            store.copy(partial, entry["key"])
            final = store.head(entry["key"])
            if final is None or final.size != entry["size"]:
                raise RuntimeError(f"{entry['key']}: copy check failed")
        manifest = {
            "manifest_version": MANIFEST_VERSION, "clip_id": clip_id,
            "source": {"key": info.key, "etag": info.etag, "size": info.size,
                       "last_modified": info.last_modified.isoformat() if info.last_modified else None},
            "durations": {"source": round(variants[0].source_duration, 3),
                          "versions": [e["duration"] for e in entries]},
            "captions": all(v.captions for v in variants),
            "variants": entries, "app_version": __version__, "created": _now(),
            "stt": {"url": settings.stt_url, "model": settings.stt_model},
        }
        store.put_json(keys.manifest, manifest)
        published = True
        store.delete(keys.error)
        return manifest
    finally:
        _delete_quietly(store, keys.partial)
        if not published:
            _delete_quietly(store, copied)  # never leave a half-published set under the final names
            log(f"  {info.key}: not published.")


class WatchState:
    """Local memory of failures, so a broken clip is not retried every minute forever."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            self.data: Dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def record_failure(self, info: ObjectInfo, error: BaseException) -> int:
        entry = self.data.get(info.key)
        if entry is None or entry.get("etag") != info.etag:
            entry = {"etag": info.etag, "failures": 0}
        entry["failures"] = int(entry.get("failures", 0)) + 1
        entry["last_error"] = f"{type(error).__name__}: {error}"[:500]
        self.data[info.key] = entry
        self._save()
        return entry["failures"]

    def clear(self, key: str) -> None:
        if self.data.pop(key, None) is not None:
            self._save()

    def parked(self, info: ObjectInfo, max_failures: int) -> bool:
        entry = self.data.get(info.key)
        return bool(entry) and entry.get("etag") == info.etag and int(entry.get("failures", 0)) >= max_failures


class Watcher:
    def __init__(self, store: S3Store, settings: Settings, render: RenderFn, work_dir: Path, state: WatchState,
                 log: Callable[[str], None], speech_check: Optional[Callable[[], bool]] = None) -> None:
        self.store, self.settings, self.render = store, settings, render
        self.speech_check = speech_check
        self.work_dir, self.state, self.log = Path(work_dir), state, log
        self._ready: set = set()  # (key, etag) known to be published, saves S3 requests every cycle
        self._parked_logged: set = set()

    def sources(self) -> List[ObjectInfo]:
        prefix = self.settings.watch_prefix
        found = [o for o in self.store.list(prefix) if is_source_key(o.key, prefix)]
        return sorted(found, key=lambda o: (o.last_modified or datetime.min.replace(tzinfo=timezone.utc), o.key))

    def run_once(self, stop: threading.Event) -> Summary:
        summary = Summary()
        s = self.settings
        sources = self.sources()
        stems: Dict[tuple, int] = {}
        for source in sources:
            folder, name = posixpath.split(source.key)
            stems[(folder, posixpath.splitext(name)[0])] = stems.get((folder, posixpath.splitext(name)[0]), 0) + 1
        for source in sources:
            if stop.is_set():
                break
            folder, name = posixpath.split(source.key)
            unique = stems[(folder, posixpath.splitext(name)[0])] > 1
            keys = output_keys(source.key, s.watch_subdir, s.watch_variants, unique)
            marker = (source.key, source.etag, s.watch_variants, unique)
            if marker in self._ready:
                summary.skipped += 1
                continue
            if is_ready(self.store, source, keys):
                self._ready.add(marker)
                summary.skipped += 1
                continue
            if self.state.parked(source, s.watch_max_failures):
                if marker not in self._parked_logged:
                    self.log(f"{source.key}: parked after {s.watch_max_failures} failures, waiting for a new file.")
                    self._parked_logged.add(marker)
                summary.skipped += 1
                continue
            self.log(f"New clip: {source.key}")
            work = self.work_dir / hashlib.sha1(source.key.encode("utf-8")).hexdigest()[:12]
            shutil.rmtree(work, ignore_errors=True)
            try:
                # Stopping waits for the clip in progress: its render is never cancelled half way.
                manifest = process_clip(self.store, source, s, self.render, work, self.log, threading.Event(),
                                        unique)
            except Cancelled:
                summary.pending += 1
                break
            except Exception as exc:  # noqa: BLE001 - one broken clip must not stop the watcher
                if isinstance(exc, SpeechServerUnavailable) and not self._speech_server_answers():
                    summary.pending += 1
                    self.log(f"Speech server unavailable, {source.key} stays in the queue: {exc}")
                    break
                if is_transient(exc):
                    summary.pending += 1
                    self.log(f"Network or storage error, {source.key} will be retried: {type(exc).__name__}: {exc}")
                    break
                failures = self.state.record_failure(source, exc)
                summary.failed += 1
                self.log(f"FAILED {source.key} ({failures}/{s.watch_max_failures}): {type(exc).__name__}: {exc}")
                try:
                    self.store.put_json(keys.error, {"source": source.key, "etag": source.etag,
                                                     "error": f"{type(exc).__name__}: {exc}"[:2000],
                                                     "failures": failures, "at": _now()})
                except Exception:  # noqa: BLE001
                    pass
            else:
                if manifest is None:
                    summary.skipped += 1
                else:
                    summary.done += 1
                    self.state.clear(source.key)
                    self._ready.add(marker)
                    self.log(f"Published {len(manifest['variants'])} versions of {source.key}.")
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return summary

    def _speech_server_answers(self) -> bool:
        """After a speech error: is the server itself fine (so this clip is the problem)?"""
        if self.speech_check is None:
            return False
        try:
            return bool(self.speech_check())
        except Exception:  # noqa: BLE001
            return False

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                summary = self.run_once(stop)
                if summary.done or summary.failed:
                    self.log(f"Cycle: {summary.done} published, {summary.failed} failed, "
                             f"{summary.pending} waiting.")
            except Exception as exc:  # noqa: BLE001 - network hiccups: try again next cycle
                self.log(f"Watch cycle failed: {type(exc).__name__}: {exc}")
            stop.wait(self.settings.watch_interval)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def acquire_lock(path: Path) -> Iterator[None]:
    """One watcher per machine. A lock left by a process that died is taken over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if _pid_alive(pid):
                raise RuntimeError(f"another watcher is running (pid {pid})") from None
            with contextlib.suppress(OSError):
                path.unlink()
    else:
        raise RuntimeError(f"cannot create the lock file {path}")
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def speech_check_for(settings: Settings) -> Callable[[], bool]:
    """Probe used by the watcher to tell a down speech server from a clip the server cannot handle."""
    from .transcribe import SpeechServer

    def check() -> bool:
        SpeechServer(settings.stt_url, settings.stt_api_key, settings.stt_model, timeout=10).check()
        return True

    return check


def default_render_fn(settings: Settings, log: Callable[[str], None]) -> RenderFn:
    """Analyze the clip once (Kamerton captions, cuts without pauses) and render the versions."""
    from .ffmpeg import probe
    from .pipeline import Pipeline
    from .reporting import Reporter
    from .variants import variant_presets

    def render(source: Path, work: Path, count: int, cancel: threading.Event) -> List[RenderedVariant]:
        s = Settings.from_dict(settings.to_dict())
        s.whole_clip, s.transcriber, s.shorts_jump_cuts, s.captions = True, "http", True, True
        pipeline = Pipeline(str(source), s, str(work / "out"), Reporter(log, None, cancel))
        project = pipeline.analyze()
        has_facecam = bool(project.effective_settings(s).facecam.strip())
        result = []
        for spec in variant_presets(count, has_facecam):
            path, captions = pipeline.render_variant(project, spec, work / f"v{spec.index}.mp4")
            result.append(RenderedVariant(path, spec.to_dict(), captions, probe(str(path)).duration,
                                          project.media.duration))
        return result

    return render
