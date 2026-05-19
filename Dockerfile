FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir librosa torch torchaudio demucs faster-whisper --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
RUN pip freeze

COPY ./docker /app

WORKDIR /workspace