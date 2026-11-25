# Funny Moment Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In auto mode every TikTok version tells "lead-in → joke → reaction": the funny moment is found with GigaAM-Emo emotions, the video starts ≥ 10 s (of finished video) before it, wordless passages are cut or sped up, and the v2 hook is gone.

**Architecture:** A separate Docker HTTP service `services/emotion/` (FastAPI + `gigaam` `emo`, CPU) returns emotion probabilities per window; `auto_video_editor/emotion.py` is its client with a cache. `build_curve` gets an optional emotion signal. A new module `auto_video_editor/funny.py` plans the whole-clip edit (peak → anchor → lead-in start → pieces `(start, end, speed)`), stored in `Moment.keep` and `Moment.edit`. `Renderer` and captions understand a per-piece speed. The watcher reads `trigger-offset-ms`, waits while the emotion server is down and writes `edit` into `manifest.json`.

**Tech Stack:** Python 3.10 (`.venv` of the main checkout), numpy, FFmpeg, pytest + moto; service: FastAPI, uvicorn, python-multipart, torch CPU, `gigaam` 0.2.0 (GitHub), Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-25-funny-moment-edit-design.md` (Russian). Read it before your task.

**Branch / worktree:** branch `feature/funny-moment-edit`, worktree `D:\proj\.claude\worktrees\auto-video-editor-setup-cc83a0\auto-video-editor`. The main checkout `D:\proj\auto-video-editor` runs the live watcher: never edit, checkout or run anything there. Never push.

**Running tests** (from the worktree root; the worktree's package is the one imported):
`D:/proj/auto-video-editor/.venv/Scripts/python.exe -m pytest tests/<file> -q`. The whole suite takes ~3 min. Baseline before this plan: 195 passed, 1 failed (`tests/test_outputs.py::test_edl_and_xml`, a Windows path issue that is not ours: leave it).

## Global Constraints

- Manual mode (long streams) must not change: with `emotion_url` empty the scoring weights are the old ones, `tests/data/render_snapshot.json` and `tests/test_render_variants.py::test_defaults_unchanged` stay untouched and green. The `--hook` / `shorts_hook` setting of manual shorts stays.
- `montage-mode: full`: nothing is cut, sped up or trimmed; `edit` in the manifest is `null`.
- Old pause rules stay: a quiet AND wordless run longer than `min_silence` is cut with 0.15 s margins; 0.1 s is kept around every word; if less than 5 s is left after the cuts the clip plays uncut.
- Laughter seconds and seconds with `P(positive) ≥ 0.5` are never cut and never sped up, by either rule. Speech is never cut or sped up.
- Defaults (exact): `watch_emotion_url = "http://localhost:8092"`, `emotion_url = ""`, `lead_in_min = 10`, `lead_in_max = 20`, `quiet_cut_max = 3.0`, `quiet_speed = 2.0` (allowed 1.25–2.0). Peak search window with a trigger: `trigger − 30 s … trigger + 3 s`. Anchor reach: a phrase ending ≤ 3 s before the peak; no speech → `peak − 2 s`. Long clip: checked after trimming the start, `> 180 s` → at most 90 s of source.
- Emotion signal: `emotion(t) = P(positive) + 0.5 × P(angry)`. Weights with emotions: emotion 0.30, loudness 0.25, spikes 0.15, speech 0.05, keywords 0.20, exclamations 0.05; without: 0.35 / 0.20 / 0.15 / 0.20 / 0.10.
- Emotion service: port **8092**, `restart: unless-stopped`, `cpus: 2`, `OMP_NUM_THREADS=2` and `torch.set_num_threads(2)`, weights downloaded at image build. `GET /health`, `POST /v1/audio/emotions` (multipart `file`, `window` = 3, `hop` = 1.5). 400 = bad file, 503 = model still loading. Labels are passed as the model gives them; AVE matches them case-insensitively and refuses to work when `positive` or `angry` is missing.
- Emotion server unreachable → the clip **waits** (`status: processing`, `state: waiting`, `reason: emotion server unavailable: …`); the server answers but fails on this clip → a clip failure (attempt counter).
- AVE works fully without `trigger-offset-ms` (peak searched in the whole clip).
- Subtitles must match the sound after speed-ups. CPU limits of the watcher (`watch_threads`, low ffmpeg priority) stay.
- Code, comments, log lines and commit messages in English, like the rest of the code base; README and `docs/crm-integration.md` in Russian. Commit after every task; end commit messages with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. `git add` only the files of your task.

## Decisions the spec left to us

- The service lives in `services/emotion/` of this repository, FastAPI.
- Emotion cache key = `media_fingerprint(path)` + server URL hash + audio track + window/hop (the same scheme as the transcript cache; the server URL stands for the model).
- Per-second emotion = overlap-weighted mean of the windows touching that second; no extra smoothing (`build_curve` already smooths the score).
- Wordless passages are judged per **gap** (a maximal run without words, laughter or `positive`): dead air inside it is cut by the old rule; what is left ("live" seconds) decides: `< min_silence` untouched, `≤ quiet_cut_max` the whole gap is cut, longer → the rest of the gap plays at `quiet_speed`.
- **Deviation from spec §6, needs the user's OK:** the wordless rule needs a transcript with at least one word in the clip. No transcript, or a recognizer that heard nothing, means we cannot tell speech from silence: only the old dead-air rule applies then (protects the anti-goal "speech is sped up" when Kamerton misses a language). By the letter of the spec such a clip would be sped up whole. The log line of Task 9 and the README say when it happened.
- The end of the video is the last word + 0.5 s as before, but never earlier than 3 s after the peak (or the end of the laughter / `positive` range there): a wordless reaction is not cut off.
- A `full` clip is not sent to the emotion server.
- The lead-in edit applies to every non-`full` whole clip, with or without emotions. `VariantSpec.hook` is removed; `edit` gets an extra `end` field.
- `event-start-offset-ms` is not read by AVE.

## Review Focus

1. A clip where the recognizer heard no words (music, English speech) → nothing is sped up, only dead air is cut — test in Task 7 (`speech_rule` off) and Task 8.
2. `trigger-offset-ms` that is garbage, negative or beyond the end of the clip → ignored with a log note, peak searched in the whole clip — tests in Task 8 and Task 10.
3. The joke in the first seconds of the file (old clips with a 15 s pre-roll) → video starts at the first sound, `lead_in_short: true`, no crash — test in Task 8.
4. Emotion server answers with other label case (`Positive`), without `positive`, or with windows that do not cover the end of the clip → case-insensitive match / clear refusal / zeros for uncovered seconds — tests in Task 5.
5. `project.json` written before this change (`keep` as `[a, b]`, a `hook`, no `edit`) → loads, plays at speed 1.0 — test in Task 1.

---

### Task 1: Pieces with speed in the timeline and in `Moment`

**Files:**
- Modify: `auto_video_editor/timeline.py`, `auto_video_editor/moments.py` (class `Moment` only)
- Test: `tests/test_timeline.py`, `tests/test_moments.py`

**Interfaces:**
- Produces (`timeline.py`): `Piece = Tuple[float, float, float]` (source start, source end, speed); `as_piece(item: Sequence[float]) -> Piece` (a 2-item range gets speed 1.0); `pieces_duration(items: Iterable[Sequence[float]]) -> float` = Σ `(end − start) / speed` (items may be 2- or 3-tuples); `total_duration(ranges)` keeps returning **source** seconds and now also accepts 3-tuples.
- Produces (`Moment`): `keep: List[Piece]`; `edit: Optional[dict] = None` (the edit plan of a whole clip, see Task 8); `pieces(jump_cuts: bool) -> List[Piece]`; `ranges(jump_cuts: bool) -> List[Range]` (2-tuples, speed dropped — manual code and exporters keep using it); `edited_duration(jump_cuts)` = output seconds (speed applied). `ranges()`/`pieces()` normalize on read with `as_piece`, so a `keep` assigned as 2-tuples works.
- Serialization: `to_dict()` writes a keep item as `[a, b]` when speed is 1.0 and `[a, b, speed]` otherwise; `"edit"` only when not None. `from_dict()` reads both forms and a missing `edit`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_timeline.py
def test_pieces_carry_a_speed():
    assert as_piece((1, 3)) == (1.0, 3.0, 1.0)
    assert as_piece([1, 3, 2]) == (1.0, 3.0, 2.0)
    assert pieces_duration([(0, 2), (2, 6, 2.0)]) == pytest.approx(4.0)
    assert total_duration([(0, 2), (2, 6, 2.0)]) == 6, "source seconds, whatever the speed"

# tests/test_moments.py
def test_moment_keep_has_speeds_and_old_projects_still_load():
    m = Moment(10, 30, keep=[(10, 14), (16, 24, 2.0)])
    assert m.pieces(True) == [(10.0, 14.0, 1.0), (16.0, 24.0, 2.0)]
    assert m.ranges(True) == [(10.0, 14.0), (16.0, 24.0)]
    assert m.pieces(False) == [(10, 30, 1.0)] and m.ranges(False) == [(10, 30)]
    assert m.edited_duration(True) == pytest.approx(8.0) and m.edited_duration(False) == 20
    data = m.to_dict()
    assert data["keep"] == [[10, 14], [16, 24, 2.0]] and "edit" not in data
    again = Moment.from_dict(data)
    assert again.pieces(True) == m.pieces(True) and again.edit is None
    old = Moment.from_dict({"start": 0, "end": 9, "keep": [[0, 4], [5, 9]], "hook": [[6, 8]]})
    assert old.pieces(True) == [(0.0, 4.0, 1.0), (5.0, 9.0, 1.0)] and old.hook == [(6.0, 8.0)]
    planned = Moment(0, 9, edit={"peak": 5.5})
    assert Moment.from_dict(planned.to_dict()).edit == {"peak": 5.5}
```

- [ ] **Step 2:** Run `pytest tests/test_timeline.py tests/test_moments.py -q` — FAIL (`as_piece` not defined).
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run the two files, then `pytest tests/test_whole_clip.py tests/test_variants.py tests/test_integration.py -q` — PASS (nothing else may change yet).
- [ ] **Step 5:** Commit `feat(timeline): pieces with a speed in Moment.keep`.

---

### Task 2: Renderer and captions follow a per-piece speed

**Files:**
- Modify: `auto_video_editor/render.py` (`build_segment_command`, `render_segment`, `measure_segment_loudness`), `auto_video_editor/captions.py` (`remap_words_sequence`)
- Test: `tests/test_render_variants.py`, `tests/test_sync.py`

**Interfaces:**
- Consumes: `as_piece`, `pieces_duration` (Task 1).
- Produces: `Renderer.build_segment_command(ranges, …)` and `render_segment(ranges, …)` accept items `(a, b)` or `(a, b, speed)`; `remap_words_sequence(words, ranges)` accepts the same items and returns caption times of the finished video (before the global version speed, which `speed_words` still applies).

Rules for `build_segment_command`:
- Normalize with `as_piece`; drop pieces with `b − a ≤ 0.04`; a piece speed outside 0.5–4.0 raises `ValueError`. `out_duration = pieces_duration(pieces)` (the global `speed` still divides it afterwards). Runs/inputs are grouped by source times exactly as now.
- A piece with speed 1.0 produces **exactly** the filter text of today (snapshot!). A piece with speed `s ≠ 1`: video `trim=…,setpts=(PTS-STARTPTS)/{s:g}`; audio `atrim=…,asetpts=PTS-STARTPTS,atempo={s:g}` and, in the multi-piece branch, the micro fades come after `atempo` with `st = max(0, (b − a)/s − MICRO_FADE)`.
- `render_segment` reports progress against `pieces_duration(ranges)`; `measure_segment_loudness` reads `r[0]`/`r[1]`.

`remap_words_sequence`: for each piece map the words as now (`TimeMap([(a, b)])`), divide the local times by the piece speed, add the offset; the offset grows by `(b − a) / speed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render_variants.py
def test_piece_speed_reaches_the_graph():
    renderer = Renderer(INFO, "libx264", "standard")
    args = renderer.build_segment_command([(1.0, 5.0), (7.0, 13.0, 2.0)], "out.mp4")
    graph = args[args.index("-filter_complex") + 1]
    assert "trim=start=1.000:end=5.000,setpts=PTS-STARTPTS[v0]" in graph, "normal pieces as before"
    assert "setpts=(PTS-STARTPTS)/2[v1]" in graph
    assert "asetpts=PTS-STARTPTS,atempo=2,afade=t=in" in graph
    assert f"afade=t=out:st={3.0 - 0.02:.3f}" in graph, "the fade is placed in sped-up time"
    assert args[args.index("-t", len(args) - 12) + 1] == "7.000"
    single = renderer.build_segment_command([(1.0, 5.0, 2.0)], "out.mp4")
    graph = single[single.index("-filter_complex") + 1]
    assert "setpts=(PTS-STARTPTS)/2[vcat]" in graph and "asetpts=PTS-STARTPTS,atempo=2[acat]" in graph
    with pytest.raises(ValueError):
        renderer.build_segment_command([(1.0, 5.0, 8.0)], "out.mp4")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_sped_up_piece_shortens_the_file(tmp_path):
    # the 6 s lavfi source of test_speed_changes_duration
    ...
    Renderer(info, "libx264", "draft").render_segment([(0.0, 2.0), (2.0, 6.0, 2.0)], out, layout="blur")
    assert probe(str(out)).duration == pytest.approx(4.0, abs=0.15)

# tests/test_sync.py
def test_captions_follow_sped_up_pieces():
    words = [[1.0, 1.4, "раз"], [9.0, 9.4, "два"]]
    timed = remap_words_sequence(words, [(0.0, 2.0), (2.0, 8.0, 2.0), (8.0, 10.0)])
    assert [(round(w.start, 2), round(w.end, 2)) for w in timed] == [(1.0, 1.4), (6.0, 6.4)]
    inside = remap_words_sequence([[4.0, 5.0, "быстро"]], [(0.0, 2.0), (2.0, 8.0, 2.0)])
    assert (inside[0].start, inside[0].end) == (pytest.approx(3.0), pytest.approx(3.5))
```

Factor the lavfi source of `test_speed_changes_duration` into a helper used by both render tests.

- [ ] **Step 2:** Run both files — the new tests FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest tests/test_render_variants.py tests/test_sync.py tests/test_auto.py -q` — PASS, `test_defaults_unchanged` included.
- [ ] **Step 5:** Commit `feat(render): per-piece speed in the filter graph and in caption times`.

---

### Task 3: No hook in the versions

**Files:**
- Modify: `auto_video_editor/variants.py`, `auto_video_editor/pipeline.py` (`render_variant` only)
- Test: `tests/test_variants.py`

**Interfaces:**
- Produces: `VariantSpec` without the `hook` field (also gone from `_LOOKS` and `to_dict()`); `Pipeline.render_variant` plays `m.pieces(s.shorts_jump_cuts)` and never a hook, even when the user's settings have `shorts_hook=True` and the moment has a `hook` (set `s.shorts_hook = False` there; drop `hook` from its log line). `compute_hook`, `Settings.shorts_hook`, `--hook` and `_render_shorts` stay: manual mode.

- [ ] **Step 1: Tests.** In `test_presets_follow_the_spec_table` replace the hook assertion with `assert all("hook" not in s.to_dict() for s in (v1, v2, v3))`. In `test_three_variants_differ`: build the settings with `shorts_hook=True`, set `short.hook = [(12.0, 15.0)]` before rendering, expect `[edited / spec.speed for spec in specs]` and add to the spy `assert ranges[0][0] == pytest.approx(short.pieces(True)[0][0])` ("no version opens with a teaser").
- [ ] **Step 2:** Run `pytest tests/test_variants.py -q` — FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest tests/test_variants.py tests/test_whole_clip.py tests/test_auto.py -q` — PASS.
- [ ] **Step 5:** Commit `feat(variants): no teaser hook in any version`.

---

### Task 4: Settings

**Files:**
- Modify: `auto_video_editor/settings.py`, `auto_video_editor/pipeline.py` (`ANALYSIS_KEYS` only)
- Test: `tests/test_advanced.py`

**Interfaces:**
- Produces: `Settings` fields `watch_emotion_url: str = "http://localhost:8092"`, `emotion_url: str = ""`, `lead_in_min: float = 10.0`, `lead_in_max: float = 20.0`, `quiet_cut_max: float = 3.0`, `quiet_speed: float = 2.0`, `trigger_offset: float = -1.0` (seconds from the start of the clip where the chat fired, set per clip by the watcher like `facecam`; negative = unknown). All seven names join `ANALYSIS_KEYS` except `watch_emotion_url`.
- `validate()`: `lead_in_min` 0–60, `lead_in_max` 0–120 and `≥ lead_in_min` (error text `lead_in_max must not be below lead_in_min`), `quiet_cut_max` 0–30, `quiet_speed` 1.25–2.0.

- [ ] **Step 1: Test**

```python
def test_funny_edit_settings_defaults_and_limits():
    s = Settings()
    assert (s.watch_emotion_url, s.emotion_url) == ("http://localhost:8092", "")
    assert (s.lead_in_min, s.lead_in_max, s.quiet_cut_max, s.quiet_speed, s.trigger_offset) == (10.0, 20.0, 3.0, 2.0, -1.0)
    assert s.validate() == []
    assert any("quiet_speed" in e for e in s.with_value("quiet_speed", "3").validate())
    assert any("quiet_speed" in e for e in s.with_value("quiet_speed", "1.1").validate())
    assert any("lead_in_max" in e for e in s.with_value("lead_in_max", "5").validate())
    assert Settings.from_dict({"quiet_cut_max": "2.5"}).quiet_cut_max == 2.5
    from auto_video_editor.pipeline import ANALYSIS_KEYS
    assert {"emotion_url", "lead_in_min", "lead_in_max", "quiet_cut_max", "quiet_speed", "trigger_offset"} <= set(ANALYSIS_KEYS)
```

- [ ] **Step 2–4:** Run `pytest tests/test_advanced.py -q` (FAIL), implement, run `pytest tests/test_advanced.py tests/test_auto.py tests/test_swiper_contract.py -q` (PASS).
- [ ] **Step 5:** Commit `feat(settings): emotion server address, lead-in and wordless passage settings`.

---

### Task 5: Emotion server client and cache

**Files:**
- Create: `auto_video_editor/emotion.py`, `tests/_fake_emotion.py`, `tests/test_emotion.py`

**Interfaces:**
- Consumes: `_wav_bytes`, `_multipart`, `SAMPLE_RATE` from `transcribe.py`.
- Produces (`emotion.py`):
  - `EMOTION_WINDOW = 3.0`, `EMOTION_HOP = 1.5`, `EMOTION_TIMEOUT = 600.0`
  - `class EmotionServerUnavailable(RuntimeError)` — unreachable, timed out, 502 or 503.
  - `@dataclass EmotionWindow(start: float, end: float, probs: Dict[str, float])` — keys lower-cased.
  - `@dataclass EmotionTrack(model: str, labels: List[str], windows: List[EmotionWindow])` with `to_dict()`, `from_dict(data)`, `per_second(label: str, seconds: int) -> np.ndarray` (float32, overlap-weighted mean of `probs[label]` of the windows touching each second, 0 where no window does), `score(seconds: int) -> np.ndarray` (`clip(positive + 0.5 × angry, 0, 1)`), `shifted(offset: float) -> EmotionTrack`.
  - `check_labels(labels: Sequence[str]) -> None` — `RuntimeError("The emotion server has no 'positive'/'angry' label (it gives: …)")` when one is missing, case-insensitive.
  - `@dataclass EmotionServer(url: str, timeout: float = EMOTION_TIMEOUT)`: the URL is normalized (`localhost:8092`, a trailing `/`, `/health` or `/v1/audio/emotions` all give `http://localhost:8092`); `describe() -> str`; `check() -> str` (`GET /health`, validates labels, returns e.g. `"gigaam-emo (angry, sad, neutral, positive)"`; a body whose `status` is not `"ok"` → `EmotionServerUnavailable`); `analyze(audio: np.ndarray, window: float = EMOTION_WINDOW, hop: float = EMOTION_HOP) -> EmotionTrack` (`POST /v1/audio/emotions`, multipart `file` = 16 kHz mono WAV, fields `window`, `hop`; validates labels; a body without a `windows` list or a window without numeric `start`/`end`/`probs` → `RuntimeError`).
  - HTTP errors: 502/503 and `URLError`/`OSError` → `EmotionServerUnavailable` (message for the unreachable case: `Cannot reach the emotion server at {url} ({reason}). Is it running? docker compose up -d in services/emotion.`); any other status → `RuntimeError(f"Emotion server error {code} at {url}{path}: {detail}")`, `detail` from the JSON `detail`/`error` field when present.
  - `load_cached(path: Path) -> Optional[EmotionTrack]`, `save_cached(path: Path, track: EmotionTrack) -> None` (atomic, like the transcript cache; an unreadable file → `None`).
- Produces (`tests/_fake_emotion.py`): `class FakeEmotionServer` — a context manager around `http.server` on `127.0.0.1:0` with `.url`, `.requests` (list of `(method, path, body_bytes)`), and constructor knobs `labels=("angry", "sad", "neutral", "positive")`, `health_status=200`, `post_status=200`, `probs=lambda start, end: {...}` (default: neutral 1.0), `model="gigaam-emo"`. It reads `window`/`hop` and the WAV length from the multipart body and answers windows `[k·hop, min(k·hop + window, duration)]` for `k·hop < duration`, in the response format of the spec (§2). Later tasks reuse it.

- [ ] **Step 1: Write the failing tests** (`tests/test_emotion.py`)

```python
def test_url_is_normalised():
    for url in ("localhost:8092", "http://localhost:8092/", "http://localhost:8092/health",
                "http://localhost:8092/v1/audio/emotions"):
        assert EmotionServer(url).url == "http://localhost:8092"

def test_analyze_through_the_server():
    happy = lambda a, b: {"angry": 0.1, "sad": 0.0, "neutral": 0.3, "positive": 0.6}
    with FakeEmotionServer(probs=happy) as fake:
        server = EmotionServer(fake.url)
        assert server.check() == "gigaam-emo (angry, sad, neutral, positive)"
        track = server.analyze(np.zeros(16000 * 6, np.float32))
    assert track.model == "gigaam-emo" and len(track.windows) == 4      # 0-3, 1.5-4.5, 3-6, 4.5-6
    assert track.windows[1].start == 1.5 and track.windows[-1].end == 6.0
    method, path, body = fake.requests[-1]
    assert (method, path) == ("POST", "/v1/audio/emotions")
    assert b'name="window"\r\n\r\n3' in body and b'name="hop"\r\n\r\n1.5' in body and b"RIFF" in body
    assert track.score(6) == pytest.approx([0.65] * 6)

def test_labels_are_matched_without_case_and_required():
    with FakeEmotionServer(labels=("Angry", "Sad", "Neutral", "Positive"),
                           probs=lambda a, b: {"Angry": 0.0, "Sad": 0.0, "Neutral": 0.2, "Positive": 0.8}) as fake:
        track = EmotionServer(fake.url).analyze(np.zeros(16000 * 3, np.float32))
        assert track.per_second("positive", 3) == pytest.approx([0.8] * 3)
    with FakeEmotionServer(labels=("happy", "calm")) as fake:
        with pytest.raises(RuntimeError, match="positive"):
            EmotionServer(fake.url).check()

def test_per_second_weights_by_overlap_and_leaves_uncovered_seconds_at_zero():
    track = EmotionTrack("m", ["angry", "positive"], [
        EmotionWindow(0.0, 1.5, {"positive": 1.0, "angry": 0.0}),
        EmotionWindow(1.5, 3.0, {"positive": 0.0, "angry": 1.0})])
    assert track.per_second("positive", 5) == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0])
    assert track.score(5) == pytest.approx([1.0, 0.75, 0.5, 0.0, 0.0])
    assert track.shifted(10).windows[0].start == 10.0
    assert EmotionTrack("m", ["angry", "positive"], []).score(3) == pytest.approx([0, 0, 0])

def test_down_or_loading_server_is_unavailable_and_other_errors_are_not():
    with pytest.raises(EmotionServerUnavailable, match="Cannot reach the emotion server"):
        EmotionServer("http://127.0.0.1:9", timeout=2).check()
    with FakeEmotionServer(health_status=503, post_status=503) as fake:
        with pytest.raises(EmotionServerUnavailable):
            EmotionServer(fake.url).check()
        with pytest.raises(EmotionServerUnavailable):
            EmotionServer(fake.url).analyze(np.zeros(16000, np.float32))
    with FakeEmotionServer(post_status=400) as fake:
        with pytest.raises(RuntimeError, match="400") as caught:
            EmotionServer(fake.url).analyze(np.zeros(16000, np.float32))
        assert not isinstance(caught.value, EmotionServerUnavailable)

def test_cache_round_trip(tmp_path):
    track = EmotionTrack("gigaam-emo", ["angry", "positive"], [EmotionWindow(0.0, 3.0, {"angry": 0.1, "positive": 0.9})])
    save_cached(tmp_path / "e" / "x.json", track)
    assert load_cached(tmp_path / "e" / "x.json") == track
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert load_cached(tmp_path / "bad.json") is None and load_cached(tmp_path / "missing.json") is None
```

- [ ] **Step 2–4:** Run `pytest tests/test_emotion.py -q` (FAIL), implement, run again (PASS).
- [ ] **Step 5:** Commit `feat(emotion): client of the emotion server with a cache`.

---

### Task 6: Emotion signal in the score curve

**Files:**
- Modify: `auto_video_editor/moments.py` (`ScoreCurve`, `build_curve`)
- Test: `tests/test_moments.py`

**Interfaces:**
- Produces: `build_curve(env, transcript, duration, keywords=(), weights=None, skip=(0.0, 0.0), emotion: Optional[np.ndarray] = None)`; `ScoreCurve.emotion: Optional[np.ndarray] = None` (new last field with a default). `emotion` is a per-second array (0–1); it is cut or zero-padded to `n` seconds. `gain` gets `"emotion": 1.0`.
- With `emotion=None` the code path and the numbers are exactly the old ones. With an emotion array: with words `raw = 0.30·emotion + 0.25·excitement + 0.15·burst + 0.05·speech + 0.20·keywords + 0.05·exclaim`; without words `raw = 0.30·emotion + 0.42·excitement + 0.28·burst` (each term times its gain, then the existing `(0.4 + 0.6·active)` factor and skips).

- [ ] **Step 1: Write the failing tests**

```python
def _active(env, n):
    return (env.per_second(n)[0] > env.silence_threshold()).astype(np.float32)

def test_emotion_reweights_the_curve_and_no_emotion_keeps_the_old_weights(envelope_factory, transcript_factory):
    env = envelope_factory(duration=60)       # no loud burst: loudness + spikes (0.40) would outweigh emotion (0.30)
    transcript = transcript_factory([(10, 14, "just walking around here"), (30, 34, "oh my god no way")])
    old = build_curve(env, transcript, 60, keywords=["oh my god"])
    factor = 0.4 + 0.6 * _active(env, 60)
    assert old.emotion is None
    assert old.raw == pytest.approx((0.35 * old.excitement + 0.20 * old.burst + 0.15 * old.speech
                                     + 0.20 * old.keywords) * factor, abs=1e-6)
    emotion = np.zeros(60, np.float32)
    emotion[40:44] = 1.0
    new = build_curve(env, transcript, 60, keywords=["oh my god"], emotion=emotion)
    assert new.raw == pytest.approx((0.30 * emotion + 0.25 * new.excitement + 0.15 * new.burst + 0.05 * new.speech
                                     + 0.20 * new.keywords) * factor, abs=1e-6)
    assert int(np.argmax(new.score)) in range(40, 44), "a laughing voice beats a reaction phrase"
    assert int(np.argmax(old.score)) not in range(40, 44)

def test_emotion_without_words_and_with_a_short_array(envelope_factory):
    env = envelope_factory(duration=30)
    curve = build_curve(env, None, 30, emotion=np.ones(10, np.float32))
    factor = 0.4 + 0.6 * _active(env, 30)
    padded = np.concatenate([np.ones(10), np.zeros(20)])
    assert curve.raw == pytest.approx((0.30 * padded + 0.42 * curve.excitement + 0.28 * curve.burst) * factor, abs=1e-6)
```

- [ ] **Step 2–4:** Run `pytest tests/test_moments.py -q` (FAIL), implement, run `pytest tests/test_moments.py tests/test_auto.py tests/test_advanced.py -q` (PASS).
- [ ] **Step 5:** Commit `feat(moments): emotion signal in the score curve`.

---

### Task 7: Pieces of a clip — dead air, wordless passages, protection

**Files:**
- Create: `auto_video_editor/funny.py`, `tests/test_funny_pieces.py`

**Interfaces:**
- Consumes: `Piece`, `merge_ranges`, `subtract_ranges` (timeline), `compute_keep_ranges`, `normalize_text` (moments), `is_laughter` (glossary), `AudioEnvelope`.
- Produces (`funny.py`):
  - constants `POSITIVE_THRESHOLD = 0.5`, `WORD_PAD = 0.1`, `CUT_MARGIN = 0.15`, `MIN_PIECE = 0.25`, `MIN_FAST = 0.5`
  - `laughter_ranges(words: Sequence) -> List[Range]` — for every word that `is_laughter(normalize_text(text))`: `(max(0, int(start) − 1), int(end) + 3)` (the seconds `build_curve` marks as laughter), merged. Words are `[start, end, text, …]` lists or objects with `.start/.end/.text`.
  - `protected_ranges(words: Sequence, positive: Optional[np.ndarray]) -> List[Range]` — laughter ranges plus `(i, i + 1)` for every second with `positive[i] ≥ POSITIVE_THRESHOLD`, merged.
  - `plan_pieces(start: float, end: float, env: AudioEnvelope, words: Sequence, min_silence: float, threshold: Optional[float] = None, *, protected: Sequence[Range] = (), speech_rule: bool = False, quiet_cut_max: float = 3.0, quiet_speed: float = 2.0) -> List[Piece]`

`plan_pieces` algorithm (frames of `env.hop`, as in `compute_keep_ranges`):
1. `quiet` = frames below the threshold; `speech` = frames within `WORD_PAD` of a word; `guard` = frames inside `protected`; `free = ~speech & ~guard`.
2. Dead runs = runs of `quiet & free` lasting `≥ min_silence`. Each gives the old cut: `[t0 + margin, t1 − margin]`, no margin on a side that touches the start or the end of the clip (the exact conditions of `compute_keep_ranges`), kept only if longer than 0.1 s.
3. With `speech_rule`, for every gap (maximal run of `free` frames) `live` = its length minus its dead runs. `min_silence ≤ live ≤ quiet_cut_max` → instead of the dead cuts of that gap, one cut over the whole gap with `CUT_MARGIN` on the sides that do not touch the clip edge. `live > quiet_cut_max` → the gap is "fast".
4. Kept ranges = clip minus cuts, ranges shorter than `MIN_PIECE` dropped, merged (if nothing is left: the whole clip). Split them at the borders of fast gaps: the parts inside a fast gap get `quiet_speed` unless shorter than `MIN_FAST` (those stay at 1.0); touching pieces of equal speed are merged.
5. With `speech_rule=False` and no `protected` the result must equal `compute_keep_ranges` with speed 1.0.

- [ ] **Step 1: Write the failing tests** (`tests/test_funny_pieces.py`; envelopes from `conftest.make_envelope`, `base=-20` is "loud", `quiet=[…]` is −70 dB)

```python
WORDS = [[5.0, 9.9, "hello there"], [12.0, 14.0, "again"], [20.0, 25.0, "the end"]]   # gaps: 2 s and 6 s

def test_same_as_the_old_rule_when_the_wordless_rule_is_off():
    env = make_envelope(duration=40, base=-20, quiet=[(10, 13), (16, 16.4), (30, 38)])
    words = [[11.0, 12.0, "hey"], [20.0, 22.0, "more words"]]
    old = compute_keep_ranges(5, 40, env, words, 0.8)
    assert [(a, b) for a, b, _ in plan_pieces(5, 40, env, words, 0.8)] == old
    assert all(speed == 1.0 for _, _, speed in plan_pieces(5, 40, env, words, 0.8))

def test_two_second_wordless_gap_is_cut_and_six_second_gap_is_sped_up():
    env = make_envelope(duration=30, base=-20)                      # loud everywhere, never quiet
    pieces = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    assert pieces == [(5, pytest.approx(10.2, abs=0.06), 1.0), (pytest.approx(11.75, abs=0.06), pytest.approx(14.1, abs=0.06), 1.0),
                      (pytest.approx(14.1, abs=0.06), pytest.approx(19.9, abs=0.06), 2.0), (pytest.approx(19.9, abs=0.06), 25, 1.0)]
    assert plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=False) == [(5, 25, 1.0)], "old rule: loud passages stay"

def test_words_are_never_cut_or_sped_up():
    env = make_envelope(duration=30, base=-20, quiet=[(14.5, 19.5)])
    for a, b, speed in plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True):
        if speed != 1.0:
            assert all(we + 0.1 <= a + 0.051 or ws - 0.1 >= b - 0.051 for ws, we, _ in WORDS)
    kept = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    for ws, we, _ in WORDS:
        assert any(a <= ws and b >= we and s == 1.0 for a, b, s in kept), "every word whole, at normal speed"

def test_laughter_and_positive_are_protected_from_both_rules():
    env = make_envelope(duration=30, base=-20, quiet=[(15, 19)])    # dead air inside the 6 s gap
    free = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True)
    assert not any(a <= 15.5 and b >= 18.5 for a, b, _ in free), "dead air is cut"
    guarded = plan_pieces(5, 25, env, WORDS, 0.8, speech_rule=True, protected=[(14.0, 20.0)])
    assert any(a <= 14.1 and b >= 19.9 and s == 1.0 for a, b, s in guarded), "kept whole, at normal speed"
    assert guarded[1][0] == pytest.approx(11.75, abs=0.06), "the 2 s gap outside the protection is still cut"

def test_dead_air_inside_a_long_gap_is_cut_and_the_rest_sped_up():
    env = make_envelope(duration=40, base=-20, quiet=[(18, 22)])
    words = [[5.0, 9.9, "start"], [30.0, 35.0, "finish"]]            # 20 s gap: 16 s live + 4 s dead
    pieces = plan_pieces(5, 35, env, words, 0.8, speech_rule=True)
    assert [s for _, _, s in pieces] == [1.0, 2.0, 2.0, 1.0]
    assert pieces[1][1] == pytest.approx(18.15, abs=0.06) and pieces[2][0] == pytest.approx(21.85, abs=0.06)

def test_gap_shorter_than_min_silence_is_untouched_and_gaps_at_the_edges_have_no_margin():
    env = make_envelope(duration=30, base=-20)
    words = [[7.0, 10.0, "a"], [10.6, 20.0, "b"]]                    # 0.4 s between words; 2 s before, 5 s after
    pieces = plan_pieces(5, 25, env, words, 0.8, speech_rule=True)
    assert pieces == [(pytest.approx(6.75, abs=0.06), pytest.approx(20.1, abs=0.06), 1.0),
                      (pytest.approx(20.1, abs=0.06), 25, 2.0)]

def test_protected_ranges_from_laughter_and_positive():
    words = [[3.2, 3.9, "hahaha"], [10.0, 11.0, "ok"]]
    positive = np.zeros(20, np.float32)
    positive[14:16] = 0.7
    positive[16] = 0.49
    assert laughter_ranges(words) == [(2, 6)]
    assert protected_ranges(words, positive) == [(2, 6), (14, 16)]
    assert protected_ranges(words, None) == [(2, 6)]
```

- [ ] **Step 2–4:** Run `pytest tests/test_funny_pieces.py -q` (FAIL), implement, run again (PASS).
- [ ] **Step 5:** Commit `feat(funny): cut short wordless passages, speed up long ones, protect laughter`.

---

### Task 8: Peak, anchor, lead-in → `EditPlan`

**Files:**
- Modify: `auto_video_editor/funny.py`
- Create: `tests/test_funny_plan.py`

**Interfaces:**
- Consumes: Task 7 functions, `pieces_duration`, `Transcript`.
- Produces (`funny.py`):
  - constants `SEARCH_BEFORE = 30.0`, `SEARCH_AFTER = 3.0`, `ANCHOR_REACH = 3.0`, `ANCHOR_FALLBACK = 2.0`, `MIN_OUTPUT = 5.0`
  - `search_window(duration: float, trigger: Optional[float]) -> Tuple[float, float]` — `(max(0, trigger − 30), min(duration, trigger + 3))`, or `(0.0, duration)` without a trigger.
  - `find_peak(score: np.ndarray, window: Tuple[float, float], duration: float) -> float` — the middle (`i + 0.5`, at most `duration`) of the best second `i` with `floor(window[0]) ≤ i < ceil(window[1])`; the earliest on a tie; an empty score → `duration / 2`.
  - `find_anchor(peak: float, segments: Sequence) -> float` — the start of the phrase with `start ≤ peak ≤ end`; else of the phrase that ended last within `ANCHOR_REACH` before the peak; else `max(0, peak − ANCHOR_FALLBACK)`.
  - `output_between(pieces: Sequence[Piece], a: float, b: float) -> float` — seconds of finished video the pieces give between source times `a` and `b`.
  - `clip_pieces(pieces: Sequence[Piece], a: float, b: float) -> List[Piece]`.
  - `choose_start(pieces, anchor: float, segments: Sequence, words: Sequence, lead_in_min: float, lead_in_max: float) -> Tuple[float, bool]` — returns `(start, lead_in_short)`:
    1. `first = pieces[0][0]`. If `output_between(pieces, first, anchor) < lead_in_min` → `(first, True)`.
    2. `t` = the latest source time with `output_between(pieces, t, anchor) ≥ lead_in_min` (walk the pieces backwards from the anchor).
    3. If `t` is inside a phrase (`seg.start < t < seg.end`) and `output_between(pieces, seg.start, anchor) ≤ lead_in_max` → `t = seg.start`; otherwise, if `t` is inside a word → `t = word.start`.
    4. `(max(first, t), False)`.
  - `@dataclass EditPlan`: `start, end, peak, anchor: float`; `pieces: List[Piece]` (empty = play `start…end` uncut); `lead_in: float`; `lead_in_short: bool`; `search_window: Tuple[float, float]`; `trigger_offset: Optional[float]`; `cuts: int`; `speedups: List[Piece]`; `emotion: Optional[dict]`; `to_dict() -> dict` with exactly the keys `peak, anchor, start, end, lead_in, lead_in_short, search_window, trigger_offset, cuts, speedups, emotion` (floats rounded to 2 decimals, `speedups` as `[[a, b, speed], …]`).
  - `plan_edit(*, duration: float, score: np.ndarray, env: AudioEnvelope, transcript: Optional[Transcript], positive: Optional[np.ndarray] = None, emotion_model: str = "", trigger: Optional[float] = None, jump_cuts: bool = True, min_silence: float = 0.8, threshold: Optional[float] = None, lead_in_min: float = 10.0, lead_in_max: float = 20.0, quiet_cut_max: float = 3.0, quiet_speed: float = 2.0, long_after: float = 180.0, long_cut: float = 90.0) -> EditPlan`

`plan_edit` algorithm:
1. A trigger outside `0…duration` is dropped (`trigger_offset` becomes None). `window = search_window(…)`, `peak = find_peak(…)`, `anchor = find_anchor(peak, segments)`.
2. `words` = all words of the transcript; `protected = protected_ranges(words, positive)`. The reaction must stay even when it has no recognized words (GigaAM rarely writes laughter as text): `tail = peak + 3.0`, moved to the end of a protected range that covers it; `end0 = min(duration, max(last word end + 0.5, tail))` with words, else `duration`.
3. `speech_rule = bool(words)`; `all_pieces = plan_pieces(0.0, end0, …, protected=protected, speech_rule=speech_rule, …)` when `jump_cuts`, else `[(0.0, end0, 1.0)]`.
4. `start, short = choose_start(all_pieces, anchor, segments, words, lead_in_min, lead_in_max)`.
5. `end = end0`; if `end − start > long_after`: `end = min(end0, max(start + long_cut, peak + 3.0))`, and if that lands inside a word, `end = word.start`.
6. `pieces = clip_pieces(all_pieces, start, end)`. The 5-second rule: if `pieces_duration(pieces) < min(MIN_OUTPUT, end0)` the cuts are dropped — repeat steps 4–5 with `all_pieces = [(0.0, end0, 1.0)]` and return `pieces = []` (the clip plays `start…end` uncut).
7. `lead_in = output_between(pieces or [(start, end, 1.0)], start, anchor)`; `cuts` = number of holes between consecutive pieces (touching pieces are not a cut); `speedups` = pieces with speed ≠ 1; `emotion = {"model": emotion_model, "peak_positive": positive at the peak second}` when `positive` is given, else None.

- [ ] **Step 1: Write the failing tests** (`tests/test_funny_plan.py`; use `make_envelope`, `make_transcript` from `conftest`; a helper `score_with(n, **peaks)` builds a zero array with given seconds set)

```python
TALK = [(0, 9.5, "so we are walking here and nothing happens at all right now"),
        (10, 19.5, "i think we should go left because the right side is closed"),
        (20, 27.5, "wait wait what is that thing over there guys"),
        (28, 31.5, "it is a chicken with a gun"),                                   # the joke
        (32, 36, "hahaha no way")]

def plan(score, transcript=None, **kw):
    env = make_envelope(duration=60, base=-20)
    return plan_edit(duration=60.0, score=score, env=env, transcript=transcript, **kw)

def test_peak_is_searched_before_the_trigger():
    score = score_with(60, **{"12": 0.9, "30": 0.6, "50": 1.0})
    assert find_peak(score, search_window(60, None), 60) == 50.5
    assert search_window(60, 37.0) == (7.0, 40.0)
    assert find_peak(score, search_window(60, 37.0), 60) == 12.5
    assert find_peak(score_with(60, **{"20": 0.5, "30": 0.5}), (0, 60), 60) == 20.5, "the earliest on a tie"
    assert search_window(60, 10.0) == (0.0, 13.0) and search_window(60, 59.0) == (29.0, 60.0)

def test_anchor_is_the_start_of_the_joke_phrase():
    segments = make_transcript(TALK).segments
    assert find_anchor(30.5, segments) == 28           # inside the joke
    assert find_anchor(33.5, segments) == 32           # inside the next phrase
    assert find_anchor(31.8, segments) == 28           # just after it, within 3 s
    assert find_anchor(50.5, segments) == 48.5         # no speech near: peak - 2
    assert find_anchor(1.0, []) == 0.0

def test_lead_in_is_at_least_ten_seconds_of_finished_video_and_starts_on_a_phrase():
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(TALK))
    assert (p.peak, p.anchor) == (30.5, 28.0)
    assert p.start == 10.0, "pulled back to the start of the phrase"
    assert 10.0 <= p.lead_in <= 20.0 and not p.lead_in_short
    assert p.lead_in == pytest.approx(output_between(p.pieces, p.start, p.anchor), abs=0.01)
    assert p.end == pytest.approx(36.45, abs=0.01), "last word + 0.5 s: the reaction stays"
    assert p.pieces[0][0] == p.start

def test_lead_in_counts_finished_video_not_source_seconds():
    talk = [(0, 6, "one two three four five six"), (16, 22, "seven eight nine ten eleven twelve"),
            (24, 27.5, "it is a chicken with a gun")]
    env = make_envelope(duration=60, base=-20, quiet=[(6.3, 15.7)])      # 9.4 s of dead air inside the lead-in
    p = plan_edit(duration=60.0, score=score_with(60, **{"25": 1.0}), env=env, transcript=make_transcript(talk))
    assert p.anchor == 24.0 and p.start == 0.0, "14 source seconds before the joke are not 10 s of video"
    assert p.lead_in == pytest.approx(13.35, abs=0.2) and not p.lead_in_short   # 6.45 + 6.65 + 0.25 s of pieces

def test_wordless_laugh_after_the_last_word_stays_in_the_video():
    positive = np.zeros(60, np.float32)
    positive[32:36] = 0.9
    p = plan(score_with(60, **{"33": 1.0}), make_transcript(TALK[:4]), positive=positive)
    assert p.anchor == 28.0 and p.end == pytest.approx(36.5, abs=0.06), "the reaction is not cut off after the last word"
    assert p.start <= p.peak <= p.end
    assert not any(s != 1.0 and a < 36 and b > 32 for a, b, s in p.pieces)

def test_long_phrase_start_lands_on_a_word():
    talk = [(0, 27.5, " ".join(f"w{i}" for i in range(55))), (28, 31.5, "it is a chicken with a gun")]
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(talk), lead_in_max=20.0)
    words = [w.start for w in make_transcript(talk).words()]
    assert p.start in [pytest.approx(w) for w in words], "the phrase is longer than lead_in_max: a word border"
    assert 10.0 <= p.lead_in <= 20.0

def test_joke_at_the_very_beginning_gives_a_short_lead_in():
    talk = [(1, 4.5, "it is a chicken with a gun"), (5, 9, "hahaha no way")]
    p = plan(score_with(60, **{"3": 1.0}), make_transcript(talk))
    assert p.lead_in_short and p.start == pytest.approx(p.pieces[0][0]) and p.start <= 1.0
    assert p.lead_in < 10.0

def test_trigger_outside_the_clip_is_ignored():
    score = score_with(60, **{"12": 0.9, "50": 1.0})
    assert plan(score, trigger=400.0).trigger_offset is None and plan(score, trigger=400.0).peak == 50.5
    assert plan(score, trigger=-5.0).search_window == (0.0, 60.0)
    assert plan(score, trigger=37.0).peak == 12.5 and plan(score, trigger=37.0).trigger_offset == 37.0

def test_no_words_means_only_dead_air_is_cut():
    env = make_envelope(duration=60, base=-20, quiet=[(40, 45)])
    p = plan_edit(duration=60.0, score=score_with(60, **{"30": 1.0}), env=env, transcript=make_transcript([]))
    assert p.speedups == [] and p.cuts == 1, "a recognizer that heard nothing must not speed the clip up"
    assert p.anchor == 28.5 and p.end == 60.0

def test_long_clip_is_checked_after_trimming_the_start():
    env = make_envelope(duration=200, base=-20)
    late = plan_edit(duration=200.0, score=score_with(200, **{"120": 1.0}), env=env, transcript=None)
    assert late.start == pytest.approx(108.5) and late.end == 200.0, "92 s left: not a long clip any more"
    env = make_envelope(duration=400, base=-20)
    long = plan_edit(duration=400.0, score=score_with(400, **{"120": 1.0}), env=env, transcript=None)
    assert long.start == pytest.approx(108.5) and long.end == pytest.approx(198.5), "at most 90 s of source"

def test_less_than_five_seconds_left_plays_uncut():
    env = make_envelope(duration=20, base=-20, quiet=[(0, 17)])
    p = plan_edit(duration=20.0, score=score_with(20, **{"18": 1.0}), env=env, transcript=None)
    assert p.pieces == [] and p.cuts == 0 and p.speedups == []
    assert (p.start, p.end) == (6.5, 20.0), "the lead-in is counted on the uncut clip then"

def test_plan_dict_has_the_manifest_fields():
    positive = np.zeros(60, np.float32)
    positive[30] = 0.82
    p = plan(score_with(60, **{"30": 1.0}), make_transcript(TALK), positive=positive, emotion_model="gigaam-emo", trigger=37.0)
    data = p.to_dict()
    assert set(data) == {"peak", "anchor", "start", "end", "lead_in", "lead_in_short", "search_window",
                         "trigger_offset", "cuts", "speedups", "emotion"}
    assert data["search_window"] == [7.0, 40.0] and data["trigger_offset"] == 37.0
    assert data["emotion"] == {"model": "gigaam-emo", "peak_positive": 0.82}
    assert plan(score_with(60, **{"30": 1.0})).to_dict()["emotion"] is None
    json.dumps(data)
```

If an exact number above disagrees with the algorithm as specified by a frame (0.05 s), adjust the tolerance, not the algorithm; if it disagrees by more, stop and report.

- [ ] **Step 2–4:** Run `pytest tests/test_funny_plan.py -q` (FAIL), implement, run `pytest tests/test_funny_plan.py tests/test_funny_pieces.py -q` (PASS).
- [ ] **Step 5:** Commit `feat(funny): peak, joke anchor and a lead-in of at least 10 s of finished video`.

---

### Task 9: Pipeline — emotions, the edit plan of a whole clip, rendering pieces

**Files:**
- Modify: `auto_video_editor/pipeline.py`
- Modify tests: `tests/test_whole_clip.py`; Create: `tests/test_funny_pipeline.py`

**Interfaces:**
- Consumes: `EmotionServer`, `EmotionTrack`, `load_cached`/`save_cached` (Task 5); `build_curve(emotion=…)` (Task 6); `plan_edit`, `EditPlan` (Task 8); `Moment.pieces/edit` (Task 1); settings (Task 4); `_decode_pcm` (transcribe).
- Produces:
  - `Pipeline._load_emotions(info: MediaInfo, s: Settings) -> EmotionTrack` — cache file `cache_dir()/emotions/{media_fingerprint}-{sha1(server.url)[:8]}{-t<track> unless 1}-w{window:g}-h{hop:g}-v1.json`, read when `s.use_cache`, always written. On a miss: log `Emotion server: {describe} — {check()}.`, decode the listened audio track in chunks of 600 s (`_decode_pcm`), `analyze` each, shift the windows by the chunk offset, join. Log `Emotions: {n} windows ({model}).` Exceptions of the client pass through.
  - `_analyze`: emotions are loaded when `s.emotion_url.strip()` and the file has audio (stage `("emotion", 6)` in `r.plan`, 0 when off); `build_curve(..., emotion=track.score(n) if track is not None else None)` (`track` is None when emotions are off). In whole-clip mode candidates are no longer searched (`candidates = []` when `s.whole_clip`). Whole clip, not `whole_clip_full`: `plan = plan_edit(duration=info.duration, score=curve.score, env=env, transcript=transcript, positive=track.per_second("positive", n) if track is not None else None, emotion_model=track.model if track is not None else "", trigger=s.trigger_offset if ≥ 0 else None, jump_cuts=s.shorts_jump_cuts, min_silence=s.min_silence, threshold=<as _jump_cut_ranges>, lead_in_min=…, lead_in_max=…, quiet_cut_max=…, quiet_speed=…, long_after=s.whole_clip_max, long_cut=s.whole_clip_cut)`; the short is `Moment(plan.start, plan.end, 100.0, plan.peak, reasons=["whole clip"], keep=plan.pieces, edit=plan.to_dict())`. Log one line: `Funny moment: peak {peak:.1f} s, joke from {anchor:.1f} s, video from {start:.1f} s (lead-in {lead_in:.1f} s[, short]), {cuts} cuts, {k} sped-up passages.` — with `, no words recognized: wordless passages left as they are` appended when the transcript has no words. When `s.trigger_offset >= 0` and `plan.trigger_offset is None`, also log `Chat trigger at {t:.1f} s is outside the clip ({duration:.1f} s): ignored, the whole clip is searched.` `whole_clip_full` keeps using `_whole_clip_moment(..., full=True)` and gets no `edit`.
  - A moment with `edit` is never re-cut: `_attach_details`, `_render` and `render_variant` skip `_jump_cut_ranges` when `moment.edit is not None`; its `hook` is `[]`.
  - `_render_shorts` and `render_variant` pass `m.pieces(s.shorts_jump_cuts)` to `_render_one_short` (durations via `pieces_duration`).

- [ ] **Step 1: Update `tests/test_whole_clip.py`** for the new long-clip rule: `test_long_clip_is_cut_to_best_fragment` uses a 400 s video with bursts `[(120, 126)]` and asserts `short.duration <= 90.5`, `short.start <= 123 <= short.end` and `short.edit["lead_in"] >= 0`; add `test_clip_that_is_short_after_trimming_keeps_its_end` (200 s, burst at 120–126 → `short.end == pytest.approx(duration, abs=0.1)` and `short.duration < 180`). `test_whole_clip_peak_is_the_loudest_part`: change the message to `"the peak is the best second, not the middle"`. Add to `test_full_mode_keeps_the_whole_long_clip`: `assert short.edit is None`.

- [ ] **Step 2: Write `tests/test_funny_pipeline.py`** (ffmpeg required; a fake Kamerton — copy the handler pattern of `tests/test_http_transcriber.py` into a small `FakeSpeechServer` context manager in `tests/_fake_speech.py` that answers `/v1/models` and `/v1/audio/transcriptions` with given segments/words — and `FakeEmotionServer`):

```python
SEGMENTS = [(1.0, 9.0, "so we are walking here and nothing happens at all"),
            (10.0, 19.0, "i think we should go left because the right side is closed"),
            (20.0, 27.0, "wait wait what is that thing over there guys"),
            (28.0, 31.0, "it is a chicken with a gun"),
            (40.0, 43.0, "okay that was funny")]            # 9 s without words after the joke

def test_whole_clip_gets_a_lead_in_emotions_and_speed_ups(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 46, bursts=[(0, 46)])          # loud everywhere
    positive = lambda a, b: ({"angry": 0, "sad": 0, "neutral": 0.1, "positive": 0.9} if 31 <= a < 34
                             else {"angry": 0, "sad": 0, "neutral": 1.0, "positive": 0})
    with FakeSpeechServer(SEGMENTS) as stt, FakeEmotionServer(probs=positive) as emo:
        pipeline = _pipeline(video, tmp_path, monkeypatch, transcriber="http", stt_url=stt.url,
                             emotion_url=emo.url, trigger_offset=38.0)
        project = pipeline.analyze()
        short = project.shorts[0]
        edit = short.edit
        assert edit["trigger_offset"] == 38.0 and edit["search_window"] == [8.0, 41.0]
        assert 28 <= edit["peak"] <= 36 and edit["anchor"] in (28.0, pytest.approx(edit["peak"] - 2, abs=0.01))
        assert edit["lead_in"] >= 10.0 and not edit["lead_in_short"] and short.start == pytest.approx(edit["start"], abs=0.01)
        assert edit["emotion"]["model"] == "gigaam-emo" and edit["emotion"]["peak_positive"] > 0.5
        assert edit["speedups"], "the wordless passage after the reaction is sped up"
        assert all(a >= 34.9 for a, b, s in edit["speedups"]), "seconds with a positive voice are not sped up"
        assert short.hook == []
        posts = [r for r in emo.requests if r[0] == "POST"]
        project2 = _pipeline(video, tmp_path, monkeypatch, transcriber="http", stt_url=stt.url,
                             emotion_url=emo.url, trigger_offset=38.0, use_cache=True).analyze()
        assert len([r for r in emo.requests if r[0] == "POST"]) == len(posts), "the second run reads the cache"
        assert project2.shorts[0].edit == edit
        out, captions = pipeline.render_variant(project, variant_presets(1, False)[0], tmp_path / "v1.mp4")
    assert captions and probe(str(out)).duration == pytest.approx(short.edited_duration(True), abs=0.4)
    assert Project.load(project.path).shorts[0].pieces(True) == short.pieces(True)

def test_without_an_emotion_address_nothing_is_sent(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 20, bursts=[(14, 17)])
    project = _pipeline(video, tmp_path, monkeypatch).analyze()            # transcriber none, emotion_url ""
    assert project.shorts[0].edit["emotion"] is None and project.shorts[0].edit["speedups"] == []

def test_emotion_server_down_stops_the_analysis(tmp_path, monkeypatch):
    video = _video(tmp_path / "clip.mp4", 8, bursts=[(0, 8)])
    with pytest.raises(EmotionServerUnavailable):
        _pipeline(video, tmp_path, monkeypatch, emotion_url="http://127.0.0.1:9").analyze()
```

Move `_video` and `_pipeline` of `tests/test_whole_clip.py` to `tests/_clips.py` and import them in both files; `_pipeline` gets a `use_cache=False` parameter. In the first test pass `language="ru"`, `diarize=False`, `auto_vocabulary=False` so the fake Kamerton only serves the transcription itself. The emotion cache is always written and read only with `use_cache=True`: the assertion is about the number of POSTs to the emotion server.

- [ ] **Step 3:** Run `pytest tests/test_whole_clip.py tests/test_funny_pipeline.py -q` — FAIL.
- [ ] **Step 4:** Implement.
- [ ] **Step 5:** Run `pytest tests/test_whole_clip.py tests/test_funny_pipeline.py tests/test_variants.py tests/test_integration.py tests/test_auto.py tests/test_advanced.py -q` — PASS.
- [ ] **Step 6:** Commit `feat(pipeline): funny moment edit of a whole clip with emotions`.

---

### Task 10: Watcher — trigger metadata, waiting for the emotion server, `edit` in the manifest

**Files:**
- Modify: `auto_video_editor/s3watch.py`, `auto_video_editor/cli.py` (`_watch`), `auto_video_editor/gui.py` (the watch thread that builds `s3watch.Watcher`, near line 2349)
- Test: `tests/test_swiper_contract.py`, `tests/test_s3watch.py`, `tests/test_cli_watch.py`

**Interfaces:**
- Consumes: `EmotionServer`, `EmotionServerUnavailable` (Task 5), settings (Task 4), `Moment.edit` (Task 9).
- Produces:
  - `ClipOptions.trigger_offset: Optional[float] = None` (seconds). `clip_options` reads `trigger-offset-ms`: digits only → `int / 1000`; anything else → note `trigger-offset-ms='…' is not a whole number of milliseconds, ignored`.
  - `watch_settings`: `s.emotion_url = "" if s.whole_clip_full else settings.watch_emotion_url` (a `full` clip gets no edit plan, so it must not wait for the emotion server); `s.trigger_offset = options.trigger_offset` or `-1.0`.
  - `RenderedVariant.edit: Optional[dict] = None`; `default_render_fn` fills it from `project.shorts[0].edit` and logs `  chat trigger from the Autoclipper at {t:.1f} s` only when that `edit` has a non-null `trigger_offset`; the manifest gets `"edit": variants[0].edit`.
  - `Watcher.__init__(..., emotion_check: Optional[Callable[[], bool]] = None)`; in `run_once`, before the transient check: `EmotionServerUnavailable` while `emotion_check` fails (or is None) → `summary.pending += 1`, log `Emotion server unavailable, {key} stays in the queue: {exc}`, `write_status(…, "waiting", …, reason=f"emotion server unavailable: {exc}")`, `break`. When the check passes, the exception is a clip failure like any other.
  - `emotion_check_for(settings: Settings) -> Callable[[], bool]` — `EmotionServer(settings.watch_emotion_url, timeout=10).check()`; always True when the address is empty.
  - `cli._watch` passes `emotion_check=s3watch.emotion_check_for(settings)` and names the emotion server in its start-up log line (`emotion server {url}` or `no emotion server`).
  - The GUI watcher passes the same `emotion_check` (without it an emotion error on a healthy server would keep a clip waiting forever and block the queue) and logs `Emotion server: {check()}` or `Emotion server not ready, clips will wait: {exc}` next to its speech server line; nothing when `watch_emotion_url` is empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_swiper_contract.py
def test_trigger_offset_from_the_autoclipper():
    assert clip_options({"trigger-offset-ms": "37000"}).trigger_offset == 37.0
    assert clip_options({}).trigger_offset is None
    for bad in ("soon", "-5", "12.5"):
        options = clip_options({"trigger-offset-ms": bad})
        assert options.trigger_offset is None and len(options.notes) == 1
    s = watch_settings(Settings(watch_emotion_url="http://emo:8092", emotion_url="http://manual"), clip_options({"trigger-offset-ms": "37000"}))
    assert (s.emotion_url, s.trigger_offset) == ("http://emo:8092", 37.0)
    assert watch_settings(Settings(), ClipOptions()).trigger_offset == -1.0
    assert watch_settings(Settings(), clip_options({"montage-mode": "full"})).emotion_url == "", "full: no emotions needed"

def test_edit_plan_reaches_the_manifest(store, tmp_path):
    # a FakeRender subclass whose variants carry edit={"peak": 31.5, "lead_in": 11.3}
    ...
    assert store.get_json(k.manifest)["edit"] == {"peak": 31.5, "lead_in": 11.3}

# tests/test_s3watch.py
def test_emotion_server_down_keeps_clip_pending(store, tmp_path):
    put_source(store)
    k = output_keys(SOURCE, "uniqalized", 3)
    watcher, logs = make_watcher(store, tmp_path, FakeRender(error=EmotionServerUnavailable("connection refused")),
                                 emotion_check=lambda: False)
    summary = watcher.run_once(threading.Event())
    assert (summary.pending, summary.failed) == (1, 0)
    status = store.get_json(k.status)
    assert status["state"] == "waiting" and status["status"] == "processing"
    assert status["reason"].startswith("emotion server unavailable")
    assert watcher.state.failures(store.head(SOURCE)) == 0

def test_emotion_error_on_a_healthy_server_counts_against_the_clip(store, tmp_path):
    put_source(store)
    watcher, _ = make_watcher(store, tmp_path, FakeRender(error=EmotionServerUnavailable("503")), emotion_check=lambda: True)
    assert watcher.run_once(threading.Event()).failed == 1

def test_manifest_edit_is_null_without_a_plan(store, tmp_path):   # FakeRender as is
    ...
    assert store.get_json(k.manifest)["edit"] is None
```

`make_watcher` gets an `emotion_check=None` parameter. In `tests/test_cli_watch.py` assert that the start-up output names `emotion server http://localhost:8092`.

- [ ] **Step 2–4:** Run the three files (FAIL), implement, run `pytest tests/test_s3watch.py tests/test_swiper_contract.py tests/test_cli_watch.py -q` (PASS).
- [ ] **Step 5:** Commit `feat(watch): chat trigger offset, waiting for the emotion server, edit plan in the manifest`.

---

### Task 11: Emotion service `services/emotion/`

Independent of Tasks 1–10 (touches only `services/emotion/`).

**Files:**
- Create: `services/emotion/ave_emotion/__init__.py`, `windows.py`, `audio.py`, `model.py`, `app.py`, `selftest.py`; `services/emotion/tests/test_api.py`, `tests/test_windows.py`; `services/emotion/Dockerfile`, `docker-compose.yml`, `requirements.txt`, `requirements-dev.txt`, `.dockerignore`, `README.md` (Russian, short)

**Interfaces:**
- `windows.py`: `window_bounds(duration: float, window: float, hop: float) -> List[Tuple[float, float]]` — `[k·hop, min(k·hop + window, duration)]` for `k·hop < duration`; a last window shorter than 0.5 s is dropped unless it is the only one; `duration ≤ 0` → `[]`.
- `audio.py`: `decode(data: bytes) -> np.ndarray` — 16 kHz mono float32 in −1…1. A PCM16 WAV that is already 16 kHz mono is read with `wave`; anything else goes through `ffmpeg -i pipe:0 -f f32le -ac 1 -ar 16000 -`. Not audio / empty → `ValueError`.
- `model.py`: `class EmotionModel(Protocol)`: `labels: List[str]`, `probs(samples: np.ndarray) -> Dict[str, float]`; `load_gigaam() -> EmotionModel` — sets `torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))`, `gigaam.load_model("emo", device="cpu")`, labels from `model.id2name` as they are. `probs` mirrors `GigaAMEmo.get_probs` on a tensor (upstream `get_probs` only takes a file path):

```python
wav = torch.from_numpy(samples).to(model._device).to(model._dtype).unsqueeze(0)
length = torch.full([1], wav.shape[-1], device=model._device)
with torch.inference_mode():
    encoded, _ = model.forward(wav, length)
    pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
    values = torch.nn.functional.softmax(model.head(pooled)[0], dim=-1).tolist()
```

  `torch` and `gigaam` are imported inside `load_gigaam` only, so the tests need neither.
- `app.py`: `create_app(loader: Callable[[], EmotionModel] = load_gigaam) -> FastAPI`. The model loads in a background thread at start-up. `GET /health` → 200 `{"status": "ok", "model": "gigaam-emo", "labels": [...]}`; while loading → 503 `{"status": "loading"}`; if loading failed → 503 `{"status": "error", "detail": "…"}`. `POST /v1/audio/emotions` (multipart `file`, `window: float = 3.0`, `hop: float = 1.5`) → `{"model": "gigaam-emo", "labels": [...], "windows": [{"start", "end", "probs"}]}` with times rounded to 3 decimals; 503 while loading; 400 `{"detail": …}` for a bad file, `window` outside 0.5–30, `hop` outside 0.1–`window`, or audio longer than 900 s. Windows are classified one at a time under a lock (one request at a time uses the 2 CPUs). `app = create_app()` at module level for uvicorn.
- `selftest.py`: `python -m ave_emotion.selftest` loads the real model, classifies 3 s of noise, checks that the probabilities sum to 1 ± 0.01 and that the labels contain `positive` and `angry` (case-insensitive); exit code 1 otherwise. The Docker build runs it, so a wrong adapter or changed labels break the build, not production.
- `Dockerfile`: `python:3.11-slim`; apt `ffmpeg`; `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`; `pip install -r requirements.txt` (fastapi, uvicorn[standard], python-multipart, numpy, and `gigaam @ https://github.com/salute-developers/GigaAM/archive/refs/heads/main.tar.gz` — build arg `GIGAAM_REF=main`); `ENV OMP_NUM_THREADS=2`; copy the package; `RUN python -m ave_emotion.selftest` (downloads the weights into the image, `~/.cache/gigaam`); `EXPOSE 8092`; `CMD uvicorn ave_emotion.app:app --host 0.0.0.0 --port 8092 --workers 1`.
- `docker-compose.yml`: service `ave-emotion`, `build: .`, `ports: ["8092:8092"]`, `restart: unless-stopped`, `cpus: 2`, `environment: OMP_NUM_THREADS=2`, a healthcheck on `/health` (python `urllib`, start period 120 s).

- [ ] **Step 1: Write the failing tests** (`services/emotion/tests/`, a `FakeModel` with `labels = ["angry", "sad", "neutral", "positive"]` whose `probs` returns positive = fraction of non-zero samples; a helper builds a WAV with `wave`)

```python
def test_health_reports_loading_then_ok():      # loader blocks on an Event: 503 loading; released: 200 ok + labels
def test_emotions_contract_on_a_short_wav():    # 6 s WAV → 4 windows (0–3, 1.5–4.5, 3–6, 4.5–6), probs keyed by the labels, model "gigaam-emo"
def test_window_and_hop_fields_are_used():      # window=2, hop=2 on 6 s → 3 windows
def test_bad_file_is_400_and_bad_params_are_400():   # b"not audio"; window=0; hop > window
def test_503_while_the_model_is_loading():      # POST before the loader returns
def test_failed_load_is_reported():             # loader raises → /health 503 {"status": "error"}
def test_window_bounds():                        # (6, 3, 1.5) → 4 windows; (3.2, 3, 1.5) → [(0, 3), (1.5, 3.2)]; (0.3, 3, 1.5) → [(0, 0.3)]; (0, 3, 1.5) → []
```

Use `fastapi.testclient.TestClient` as a context manager so start-up runs.

- [ ] **Step 2:** Create the venv and run the tests: `C:/Python310/python.exe -m venv services/emotion/.venv`, `services/emotion/.venv/Scripts/python.exe -m pip install -r services/emotion/requirements-dev.txt` (fastapi, uvicorn, python-multipart, numpy, httpx, pytest — no torch), then from `services/emotion/`: `.venv/Scripts/python.exe -m pytest tests -q` — FAIL.
- [ ] **Step 3:** Implement. Before writing `model.py`, read the current upstream `gigaam/model.py`, `gigaam/__init__.py` and `pyproject.toml` (https://github.com/salute-developers/GigaAM) and adjust the adapter if the API differs from the snippet above.
- [ ] **Step 4:** Tests PASS; `docker compose -f services/emotion/docker-compose.yml config` prints the resolved file without errors. Do **not** build the image (several GB on a nearly full disk; the user decides when).
- [ ] **Step 5:** Commit `feat(emotion-service): GigaAM-Emo HTTP service in Docker`.

---

### Task 12: Documentation and CI

**Files:**
- Modify: `README.md`, `docs/crm-integration.md`, `.github/workflows/tests.yml`

- [ ] **Step 1: README (Russian).** "Как это устроено": the edit is now "подводка → шутка → реакция" (one short paragraph: emotions, lead-in ≥ 10 s, wordless passages cut/sped up; one sentence: если в клипе не распознано ни одного слова, куски без речи не трогаются, вырезается только тишина). "Чем отличаются версии": remove the hook of v2. "Установка": a step "Сервис эмоций" next to Kamerton — `cd services\emotion`, `docker compose up -d --build` (first build downloads ~1 GB of weights), check `http://localhost:8092/health`. "Настройки": the six rows of the spec table (§3). "Что появляется в бакете" / manifest description: the `edit` field with the example of the spec and one line per key. "Частые проблемы": `state: waiting`, `reason: emotion server unavailable` → start the container. "Разработка": `emotion.py`, `funny.py`, `services/emotion/` in the code map and how to run the service tests.
- [ ] **Step 2: `docs/crm-integration.md` (Russian).** Metadata table: `trigger-offset-ms` (read; whole milliseconds from the start of the clip file; absent or invalid → the funny moment is searched in the whole clip), `event-start-offset-ms` (ignored). Fix the `montage-mode full` row (no hook) and rewrite the `cut` row: начало обрезается до подводки ≥ 10 с перед смешным моментом; паузы вырезаются, короткие куски без речи вырезаются, длинные ускоряются; если после обрезки начала клип длиннее 3 минут — остаётся до 90 с исходника от начала подводки. States: waiting also for the emotion server. Manifest example: add `edit` (and `"edit": null` for `full`); explain each key in a small table.
- [ ] **Step 3: CI.** Add a second job `emotion-service` (ubuntu, Python 3.11) with `defaults.run.working-directory: services/emotion` and the steps `python -m pip install -r requirements-dev.txt`, `python -m pytest tests -q`.
- [ ] **Step 4:** Run the whole suite: `pytest -q` → only `test_edl_and_xml` may fail. Commit `docs: funny moment edit, emotion service and the edit field of the manifest`.

---

## Live check (after all tasks, with the user)

Build and start `ave-emotion`, take 3–5 clips from `montage/` into a test prefix (`--set watch_prefix=…` on a scratch copy), look at the first seconds of the versions and at `edit` in `manifest.json`, compare durations with the old versions. Not part of the automated plan: it needs the Docker image (disk space) and the user's bucket.
