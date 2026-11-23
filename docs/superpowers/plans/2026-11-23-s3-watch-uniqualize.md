# S3 Watch & Uniqualize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `--watch` mode that polls `montage/<streamer>/` in S3, renders 3 distinct vertical TikTok versions of each new clip (jump cuts, Kamerton captions) and publishes them to `montage/<streamer>/uniqalized/` with a manifest that guarantees source↔versions integrity.

**Architecture:** `s3store.py` wraps boto3 with verified download/upload. `Pipeline` gains a "whole clip" plan (one short = the clip minus dead air) and `render_variant()`; `variants.py` defines the 3 presets; `Renderer` gains zoom/shift/speed/effects knobs. `s3watch.py` owns keys, manifest, readiness, the partial→final publish protocol, failure state and the poll loop. CLI `--watch [--once]` and a GUI page drive it.

**Tech Stack:** Python 3.10 venv, FFmpeg, boto3 (optional extra `s3`), moto (tests), existing Kamerton HTTP transcriber.

**Spec:** `docs/superpowers/specs/2026-11-23-s3-watch-uniqualize-design.md` (Russian). Read it before Task 1.

**Branch:** work on a local branch `feature/s3-watch` in `D:\proj\auto-video-editor`; never push.

## Global Constraints

- Bucket `twitch-highlights-storage`, endpoint `https://s3.twcstorage.ru`, region `ru-1`; credentials only from env / `.env`: `AVE_S3_ENDPOINT`, `AVE_S3_BUCKET`, `AVE_S3_ACCESS_KEY`, `AVE_S3_SECRET_KEY`. `.env` is git-ignored; keys never logged (log `***`).
- Sources: objects exactly at `montage/<streamer>/<file>` with extension `.mp4`, `.mov`, `.mkv` (case-insensitive); nothing deeper, nothing directly in `montage/`.
- Outputs: `montage/<s>/uniqalized/<stem>__v{N}.mp4`, manifest `montage/<s>/uniqalized/<stem>.manifest.json`, errors `…/<stem>.error.json`, staging `montage/<s>/uniqalized/.partial/<stem>__v{N}.mp4`.
- Never write, copy or delete outside `montage/<s>/uniqalized/` (including `.partial/`). Sources are read-only.
- Readiness = manifest exists AND manifest `source.etag`/`source.size` equal the current source AND every listed version exists with the listed size.
- Per-version object metadata: `clip-id`, `source-key` (URL-encoded, S3 metadata is ASCII), `source-etag`, `source-size`, `variant`, `manifest-version` (= `1`).
- Defaults: 3 versions, poll every 60 s, whole clip up to 180 s, longer clips → best fragment ≤ 90 s, 3 consecutive failures → parked until the source ETag changes.
- Kamerton unavailable → clip stays pending, not a failure, no output without captions. Kamerton answered with no words → versions without captions, manifest `captions: false`.
- The webcam is never cropped: with a detected facecam no variant uses `crop`/`smart`, and zoom/shift never move the facecam out of frame.
- No mirroring. Speed changes keep pitch (`atempo`), captions follow the speed.
- One watcher per machine (lock file).

## Review Focus

1. Cyrillic / spaces / `+` in streamer folder or file name (`montage/бустер/клип 1+2.mp4`) → keys round-trip, metadata URL-encoded — test in Task 5.
2. Source moved or deleted between listing and download (Swiper / human) → clip skipped quietly, not a failure, nothing written — test in Task 5.
3. Clip without audio track, or a silent one → whole clip rendered, no captions, `captions: false` — test in Task 2.
4. Very short clip (≤ 6 s) or jump cuts leaving < `shorts_min` → whole clip played without cuts, render still succeeds — test in Task 2.
5. Network/S3 failure after some versions reached `.partial/` → no final keys or manifest change; next cycle completes — test in Task 5.

---

### Task 1: S3 store with verified transfers

**Files:**
- Create: `auto_video_editor/s3store.py`, `tests/test_s3store.py`, `.env.example`
- Modify: `requirements.txt` (add `boto3>=1.34`), `pyproject.toml` (extra `s3 = ["boto3>=1.34"]`, add to `all`; `dev` += `moto[s3]>=5`), `.gitignore` (add `.env`)

**Interfaces:**
- Produces:
  - `@dataclass S3Config(endpoint: str, bucket: str, access_key: str, secret_key: str, region: str = "ru-1")`; `S3Config.from_env(env_file: Optional[Path] = None) -> S3Config` (reads `os.environ`, then `KEY=VALUE` lines of `env_file`, default `<repo>/.env`; env wins; raises `RuntimeError` naming the missing variable); `__repr__` masks keys.
  - `@dataclass ObjectInfo(key: str, etag: str, size: int, last_modified: datetime, metadata: Dict[str, str])` (etag without quotes).
  - `class S3Store(config: S3Config, client=None)` with: `list(prefix: str) -> List[ObjectInfo]` (paginated, no metadata), `head(key) -> Optional[ObjectInfo]` (None on 404), `download(info: ObjectInfo, dest: Path) -> None` (temp file + rename; size must equal `info.size`; if etag has no `-` the MD5 must equal it; else raise `TransferError`), `upload(path: Path, key: str, metadata: Dict[str, str], content_type="video/mp4") -> ObjectInfo` (`put_object` with `ContentMD5`, then `head` and size check), `copy(src: str, dst: str) -> None` (server-side, `MetadataDirective="COPY"`), `delete(key) -> None`, `put_json(key, data: dict) -> None`, `get_json(key) -> Optional[dict]`.
  - `class TransferError(RuntimeError)`.

- [ ] **Step 1: Write failing tests** in `tests/test_s3store.py` using `moto.mock_aws` and a bucket created in a fixture:
  - `test_from_env_reads_dotenv_and_masks_keys`: `.env` with the four vars → fields set; `"secret" not in repr(cfg)`; missing var → `RuntimeError` mentioning `AVE_S3_SECRET_KEY`.
  - `test_upload_download_roundtrip_keeps_metadata`: upload 1 MB file with `{"clip-id": "abc"}` → `head(key).metadata["clip-id"] == "abc"`, size equal; `download` → bytes equal.
  - `test_download_detects_size_mismatch`: `download(replace(info, size=info.size + 1), dest)` raises `TransferError`, `dest` absent.
  - `test_head_missing_returns_none`, `test_copy_keeps_metadata`, `test_list_is_paginated` (1005 keys → 1005 results), `test_get_json_missing_returns_none`.
- [ ] **Step 2:** `.venv/Scripts/python -m pip install boto3 "moto[s3]"` then `.venv/Scripts/python -m pytest tests/test_s3store.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement `s3store.py` per Interfaces; client `boto3.client("s3", endpoint_url=…, region_name=…, config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}))`. `.env.example` lists the four vars with empty values and endpoint/bucket filled.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat(s3): verified S3 store`.

### Task 2: Whole-clip plan in the pipeline

**Files:**
- Modify: `auto_video_editor/settings.py` (fields + ANALYSIS_KEYS), `auto_video_editor/pipeline.py` (`_analyze`), `auto_video_editor/transcribe.py`
- Test: `tests/test_whole_clip.py`

**Interfaces:**
- Produces:
  - Settings: `whole_clip: bool = False`, `whole_clip_max: float = 180.0`, `whole_clip_cut: float = 90.0` (add the three to `pipeline.ANALYSIS_KEYS`).
  - With `whole_clip=True`, `Pipeline.analyze()` returns a `Project` with `highlight == []` and exactly one short: if duration ≤ `whole_clip_max` → `Moment(start=max(0, first_word.start − 0.3), end=min(duration, last_word.end + 0.5))` (no words → `0..duration`), score 100; otherwise `plan_shorts(..., count=1, min_len=20, max_len=whole_clip_cut)`. Details attached with `_attach_details(is_short=True)` and `hook = compute_hook(...)`. The "No usable moments" error must not fire for ≤ `whole_clip_max`.
  - `transcribe.SpeechServerUnavailable(RuntimeError)`: raised by `SpeechServer._request` on `URLError`/`OSError` and on HTTP 502/503 (engine not ready); other HTTP errors stay `RuntimeError`. The pipeline must let it propagate unchanged.
- [ ] **Step 1: Write failing tests** (generated 30 s video like `tests/test_integration.py::sample_video`, but speech-like bursts: tone 0–4 s, silence 4–9 s, tone 9–20 s, silence 20–30 s; `transcriber="none"`, `jump_cuts`, `shorts_jump_cuts=True`, cache in `tmp_path`):
  - `test_whole_clip_gives_one_short_without_dead_air`: one short, `highlight == []`, `total_duration(short.keep) < 22`, `short.keep[0][0] < 0.5`.
  - `test_whole_clip_without_audio` (video only input): one short `0..duration`, no exception.
  - `test_whole_clip_tiny_clip` (5 s): one short, render via `pipeline.render(project)` succeeds and output exists.
  - `test_long_clip_is_cut_to_best_fragment` (200 s generated): short duration ≤ 90.
  - `test_speech_server_down_raises_unavailable`: `SpeechServer("http://127.0.0.1:9", timeout=2).check()` raises `SpeechServerUnavailable`.
- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest tests/test_whole_clip.py -q` → FAIL.
- [ ] **Step 3:** Implement: branch in `_analyze` right before `find_candidates` (keep content classification, layout and facecam resolution above it); `SpeechServerUnavailable` in `transcribe.py`; update the existing `test_unreachable_server_is_reported` regex if wording changes.
- [ ] **Step 4:** Run new tests + `tests/test_http_transcriber.py tests/test_integration.py` → PASS.
- [ ] **Step 5:** Commit `feat(pipeline): whole-clip short plan`.

### Task 3: Renderer variation knobs

**Files:**
- Modify: `auto_video_editor/render.py`
- Test: `tests/test_render_variants.py`

**Interfaces:**
- Produces:
  - `zoom_window(width: int, height: int, zoom: float, shift: float, keep: Optional[Tuple[float,float,float,float]] = None) -> Tuple[int, int, int, int]` — even crop `(w, h, x, y)` of size `/zoom`, horizontally offset by `shift` ∈ [−1, 1] of the free margin; if `keep` (normalized rect) is given the window is clamped to contain it fully.
  - `vertical_filter(..., game_zoom: float = 1.0, game_shift: float = 0.0, split_top: int = SPLIT_TOP_H)`: zoom applies to the game branch in `split`, to the foreground in `blur` (with `keep=facecam` when one is passed), and to the crop window in `crop`/`smart`; `split_top` replaces the 640 px camera height.
  - `Renderer.build_segment_command(..., speed: float = 1.0, game_zoom: float = 1.0, game_shift: float = 0.0, split_top: int = SPLIT_TOP_H, video_effects: Optional[str] = None, extra_output_args: Sequence[str] = ())`: `setpts=PTS/{speed}` right after `[vfps]`, `atempo={speed}` first in audio steps, `out_duration /= speed` before fades/`-t`; `video_effects` inserted before `ass=` in the tail; `extra_output_args` appended just before the output path. Defaults produce byte-identical arguments to today.
- [ ] **Step 1: Write failing tests:**
  - `test_defaults_unchanged`: `build_segment_command` with old kwargs equals a snapshot built before the change (capture with the current code first and paste the list into the test).
  - `test_zoom_window_keeps_facecam`: `zoom_window(1920, 1080, 1.05, -1.0, keep=(0.0, 0.0, 0.25, 0.3))` → `x == 0`; with `shift=1.0` the window still contains `x ≤ 0` (clamped).
  - `test_speed_changes_duration` (real ffmpeg, 6 s testsrc2 + sine): render `[(0, 6)]` layout `blur`, `speed=1.03` → probed duration ≈ 6/1.03 ± 0.1, size 1080×1920.
  - `test_effects_and_extra_args_present`: `video_effects="noise=alls=3:allf=t"`, `extra_output_args=["-g","48"]` appear in the args; `ass=` comes after `noise=`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run this file + `tests/test_outputs.py tests/test_sync.py` → PASS (except the known Windows path failure `test_edl_and_xml`).
- [ ] **Step 5:** Commit `feat(render): zoom, speed and effects knobs`.

### Task 4: Variant presets and `render_variant`

**Files:**
- Create: `auto_video_editor/variants.py`
- Modify: `auto_video_editor/pipeline.py` (extract the per-short body of `_render_shorts` into `_render_one_short`, add `render_variant`)
- Test: `tests/test_variants.py`

**Interfaces:**
- Consumes: Task 3 knobs; Task 2 `whole_clip` project.
- Produces:
  - `@dataclass VariantSpec(index: int, layout: str, split_top: int, caption_style: str, caption_color: str, caption_position: str, hook: bool, zoom: float, shift: float, speed: float, effects: str, gop: int, quality: str)`; `to_dict()`.
  - `variant_presets(count: int, has_facecam: bool) -> List[VariantSpec]` — values from the spec table: v1 `split`(640)/`blur`, karaoke `#FFE600` bottom, no hook, zoom 1.00, speed 1.00; v2 `blur`, karaoke `#39FF88` middle, hook, zoom 1.03 shift −0.6, saturation +, warmer, speed 1.02; v3 facecam → `split` with `split_top=820`, else `smart`; simple `#FFFFFF` top, no hook, zoom 1.05 shift +0.6, contrast +, cooler, speed 0.98. All: `noise=alls=2..4:allf=t` (different per variant), gop 48/60/72, quality high/standard/high. count > 3 cycles presets with the index changing shift/noise. Effects strings are ffmpeg `eq`/`colorbalance`/`noise` chains (exact values are the builder's call within these directions).
  - `Pipeline.render_variant(project: Project, spec: VariantSpec, output: Path) -> Tuple[Path, bool]` — renders `project.shorts[0]` with the spec (layout overrides, caption overrides, hook on/off, word times divided by `spec.speed`), extra output args `["-g", str(gop), "-map_metadata", "-1", "-metadata", f"comment={uuid4().hex}", "-metadata", f"creation_time={now_iso}"]`; returns `(path, captions_burned)`. `_render_shorts` keeps its current behaviour through `_render_one_short`.
- [ ] **Step 1: Write failing tests** (generated 20 s clip with a static colored box top-left as "facecam", `facecam="0,0,0.25,0.3"`, `whole_clip=True`, `transcriber="none"`, encoder `libx264`, quality `draft`):
  - `test_presets_never_crop_with_facecam`: `{s.layout for s in variant_presets(3, True)} <= {"split", "blur"}`; `variant_presets(3, False)[2].layout == "smart"`.
  - `test_three_variants_differ`: render 3 → all 1080×1920, durations differ by < 6 %, MD5 of the three files distinct, no file has `creation_time` from the source.
  - `test_existing_shorts_render_unchanged`: `tests/test_integration.py::test_full_pipeline` still passes.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run `tests/test_variants.py tests/test_integration.py` → PASS.
- [ ] **Step 5:** Commit `feat(variants): three TikTok presets`.

### Task 5: Watcher — keys, manifest, publish protocol, failures

**Files:**
- Create: `auto_video_editor/s3watch.py`, `tests/test_s3watch.py`
- Modify: `auto_video_editor/settings.py` (fields `watch_prefix: str = "montage/"`, `watch_subdir: str = "uniqalized"`, `watch_variants: int = 3`, `watch_interval: int = 60`, `watch_max_failures: int = 3`; validate ranges `watch_variants` 1–10, `watch_interval` 10–3600)

**Interfaces:**
- Consumes: Task 1 `S3Store`/`ObjectInfo`; Task 2 `SpeechServerUnavailable`; Task 4 `variant_presets`, `render_variant`.
- Produces:
  - `is_source_key(key: str, prefix: str) -> bool` (exactly `prefix/<s>/<file>` with a video extension).
  - `output_keys(source_key: str, subdir: str, count: int) -> OutputKeys` with fields `final: List[str]`, `partial: List[str]`, `manifest: str`, `error: str`.
  - `RenderedVariant(path: Path, spec: dict, captions: bool)`; type `RenderFn = Callable[[Path, Path, int, threading.Event], List[RenderedVariant]]` (source file, work dir, count, cancel). `default_render_fn(settings: Settings, log) -> RenderFn` builds `Settings` from the user's settings with `whole_clip=True`, `transcriber="http"`, `shorts_jump_cuts=True`, `captions=True`, then analyze + `render_variant` per preset.
  - `is_ready(store, source: ObjectInfo, keys: OutputKeys) -> bool` (Global Constraints rule).
  - `process_clip(store, source: ObjectInfo, settings, render: RenderFn, work: Path, log) -> Optional[dict]` — protocol, in order: delete old manifest → download (if `head` is None → return None, not a failure) → render → upload each to `partial[i]` with metadata → `head(source)` again (changed/missing → delete this clip's partial keys, return None) → `copy` partial→final → delete stale finals beyond `count` → `put_json(manifest)` → delete partials → return manifest. Manifest fields: `manifest_version: 1`, `clip_id`, `source {key, etag, size, last_modified}`, `durations {source, edited}`, `captions`, `variants [{key, size, md5, sha256, spec}]`, `app_version`, `created`, `stt {url, model}`.
  - `class WatchState(path: Path)` — JSON `{source_key: {"etag", "failures", "last_error"}}`; `record_failure(info, error) -> int`, `clear(key)`, `parked(info, max_failures) -> bool` (true while failures ≥ max and etag unchanged).
  - `class Watcher(store, settings, render: RenderFn, work_dir: Path, state: WatchState, log)`; `run_once(stop: threading.Event) -> Summary(done: int, skipped: int, failed: int, pending: int)`; `run_forever(stop)` sleeps `watch_interval` between cycles via `stop.wait`. `SpeechServerUnavailable` ends the cycle with the clip pending (not a failure). Any other exception → `record_failure`, write `error.json` `{source, etag, error, failures, at}`, continue with next clip.
  - `acquire_lock(path: Path) -> ContextManager` — `O_CREAT|O_EXCL` file with PID; stale (dead PID) lock is replaced; live lock → `RuntimeError("another watcher is running (pid N)")`.
- [ ] **Step 1: Write failing tests** with moto bucket and a fake `RenderFn` that writes N small distinct files (`b"v%d"` + random bytes):
  - `test_source_key_filter`: `montage/buster/a.MP4` ✓, `montage/a.mp4` ✗, `montage/buster/uniqalized/a__v1.mp4` ✗, `montage/buster/a.txt` ✗.
  - `test_happy_path_publishes_versions_then_manifest`: one source with `clip-id` → 3 finals with metadata (`clip-id`, `variant`, `source-etag`), manifest lists md5/size matching objects, no `.partial/` keys left, `is_ready` true; second `run_once` → `done == 0, skipped == 1`, fake render not called.
  - `test_unicode_keys`: source `montage/бустер/клип 1+2.mp4` → `head(final[0]).metadata["source-key"]` URL-decodes to the source key.
  - `test_crash_after_partial_upload_leaves_no_manifest`: fake `store.upload` raising on the 2nd call → no final keys, no manifest, failures == 1; next run (upload fixed) → ready.
  - `test_source_changed_triggers_rebuild`: overwrite source → not ready → run → manifest etag updated, finals replaced.
  - `test_source_vanished_mid_render`: fake render deletes the source → nothing final, partials deleted, failures == 0.
  - `test_speech_server_down_keeps_clip_pending`: render raises `SpeechServerUnavailable` → `pending == 1`, failures == 0, no `error.json`.
  - `test_parked_after_three_failures`: render raises `ValueError` 3 times → `error.json` exists, 4th cycle doesn't call render; new source etag → render called again.
  - `test_lock_rejects_second_watcher`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run `tests/test_s3watch.py` → PASS.
- [ ] **Step 5:** Commit `feat(watch): S3 watcher with manifest integrity`.

### Task 6: CLI, GUI page, docs

**Files:**
- Modify: `auto_video_editor/cli.py`, `auto_video_editor/gui.py`, `README.md`
- Test: `tests/test_cli_watch.py`

**Interfaces:**
- Consumes: Task 5 `Watcher`, `default_render_fn`, `acquire_lock`, `WatchState`; Task 1 `S3Config.from_env`.
- Produces:
  - CLI: `--watch` (loop), `--once` (with `--watch`: single cycle; exit code 0 if no failures else 1), `--env-file PATH`. Lock at `cache_dir()/s3watch/watch.lock`, state `cache_dir()/s3watch/state.json`, work dir `cache_dir()/s3watch/work` (cleared per clip). Ctrl+C sets the stop event; the current clip finishes, then exit 0. On start logs bucket, prefix, Kamerton URL and `S3Config` repr (masked).
  - GUI: sidebar page "Auto (S3)": Start/Stop button (thread + stop event), status chips S3 ✓/✗ and Kamerton ✓/✗ (checked on start and each cycle), last 20 outcomes list (`stem — done/failed/pending`), versions count and interval fields bound to settings. Logs go to the existing Activity log.
  - README section "Auto mode (S3)" in English: `.env` setup, `python main.py --watch`, bucket layout, manifest meaning, Docker/Kamerton must be running.
- [ ] **Step 1: Write failing tests:** `test_watch_once_against_moto` (moto bucket, monkeypatched `default_render_fn` returning fake variants, env vars set) → `main(["--watch", "--once"]) == 0` and manifest exists; `test_watch_requires_credentials` → missing env → exit code 2 with message naming the variable.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement CLI, GUI, README. **Step 4:** Run full suite `.venv/Scripts/python -m pytest -q` → only the known `test_edl_and_xml` failure. Launch the GUI, open "Auto (S3)", check the page renders and Start/Stop toggles (manual).
- [ ] **Step 5:** Commit `feat(watch): CLI and GUI auto mode`.

### Task 7: Live verification (needs the user's OK before writing to the bucket)

- [ ] **Step 1:** Ask the user to start Docker Desktop (Kamerton) and confirm a test write. Proposed: run with `watch_prefix="ave-test/montage/"` after copying the existing `to_sort/buster/2026-11-20_21-59-44_reaction_6f4a.mp4` to `ave-test/montage/buster/` (server-side copy, source untouched).
- [ ] **Step 2:** `.venv/Scripts/python main.py --watch --once` with that prefix → exit 0; check 3 finals + manifest, `is_ready` true, download v1–v3 and view a frame of each (captions, webcam intact, visible differences).
- [ ] **Step 3:** Re-run → nothing re-rendered. Report results; delete the `ave-test/` objects only if the user agrees.
