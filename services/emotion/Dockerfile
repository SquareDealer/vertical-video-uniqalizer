FROM python:3.11-slim

# ffmpeg decodes anything that is not a plain 16 kHz WAV.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small (no CUDA libraries).
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GigaAM comes from GitHub; the build arg is a branch, a tag or a commit (pin it once the image has been verified).
ARG GIGAAM_REF=main
RUN pip install --no-cache-dir "gigaam @ https://github.com/salute-developers/GigaAM/archive/${GIGAAM_REF}.tar.gz"

ENV OMP_NUM_THREADS=2
COPY ave_emotion ave_emotion

# Downloads the weights into the image (~/.cache/gigaam) and checks the adapter and the labels:
# a wrong adapter or other labels break the build, not production.
RUN python -m ave_emotion.selftest

EXPOSE 8092
CMD ["uvicorn", "ave_emotion.app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
