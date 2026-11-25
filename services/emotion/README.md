# Сервис эмоций `ave-emotion`

Небольшой HTTP-сервис вокруг модели GigaAM-Emo (4 эмоции: angry, sad, neutral, positive).
Auto Video Editor спрашивает у него, где в клипе смеются и радуются, и по этому находит смешной момент.

## Запуск

```powershell
cd services\emotion
docker compose up -d --build
```

Первая сборка скачивает веса модели (~1 ГБ) прямо в образ, поэтому запуск потом быстрый.
Проверка: <http://localhost:8092/health> отвечает `{"status": "ok", ...}`. Пока модель грузится, ответ — 503.

Сервис ограничен двумя ядрами процессора (`cpus: 2`, `OMP_NUM_THREADS=2`), чтобы не занимать весь компьютер,
и перезапускается сам (`restart: unless-stopped`).

## API

- `GET /health` — `{"status": "ok", "model": "gigaam-emo", "labels": [...]}`; 503, пока модель грузится.
- `POST /v1/audio/emotions` — multipart: `file` (аудио), `window` (по умолчанию 3 с), `hop` (по умолчанию 1.5 с).
  Ответ: `{"model": "gigaam-emo", "labels": [...], "windows": [{"start", "end", "probs": {...}}]}`.
  400 — неверный файл или параметры (аудио не длиннее 900 с), 503 — модель ещё грузится.

## Тесты

Модель для тестов не нужна (подставляется поддельная):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
```
