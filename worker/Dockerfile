FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only pyproject.toml first — deps layer cached until deps change
COPY worker/pyproject.toml ./pyproject.toml

# Install dependencies WITHOUT the project itself (heavy layer: torch+CUDA ~7GB)
# This layer is cached as long as pyproject.toml doesn't change.
RUN uv sync --no-install-project --no-dev

# Now copy application code (small, changes frequently)
COPY worker/tts_api.py ./

# Voice files mounted at /data/voices
VOLUME /data/voices

# HuggingFace cache for model weights
VOLUME /root/.cache/huggingface

EXPOSE 8100

ENV TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
ENV TTS_DEVICE="cuda"

CMD ["sh", "-c", "uv run python tts_api.py --model \"$TTS_MODEL\" --device \"$TTS_DEVICE\" --host 0.0.0.0 --port 8100"]
