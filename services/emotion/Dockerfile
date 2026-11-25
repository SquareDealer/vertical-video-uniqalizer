FROM python:3.11-slim

# ffmpeg decodes anything that is not a plain 16 kHz WAV.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small (no CUDA libraries).
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GigaAM is not on PyPI under a pinned name for us: take it from GitHub (a build arg picks the branch or tag).
ARG GIGAAM_REF=main
RUN pip install --no-cache-dir "gigaam @ https://github.com/salute-developers/GigaAM/archive/refs/heads/${GIGAAM_REF}.tar.gz"

ENV OMP_NUM_THREADS=2
COPY ave_emotion ave_emotion

# Downloads the weights into the image (~/.cache/gigaam) and checks the adapter and the labels:
# a wrong adapter or other labels break the build, not production.
RUN python -m ave_emotion.selftest

EXPOSE 8092
CMD ["uvicorn", "ave_emotion.app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
