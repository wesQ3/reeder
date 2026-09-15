FROM ghcr.io/0xshug0/audio.cpp:full-cuda12

USER root

# Install uv for fast Python dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PATH="/app:$PATH"

# Copy dependency manifests first for caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies without virtual environment overhead
RUN uv sync --no-install-project --no-dev

# Copy application configuration and source code
COPY server.json run.sh tts_api.py ./
RUN chmod +x /app/run.sh

# Voice files and GGUF models mounted as volumes
VOLUME /data/voices
VOLUME /models

# 8100: Reeder compatibility gateway
# 8080: audio.cpp native server & Arena WebUI
EXPOSE 8100 8080

ENV ACTIVE_TTS_MODEL="qwen3-tts"
ENV AUDIOCPP_BACKEND="cuda"
ENV AUDIOCPP_URL="http://127.0.0.1:8080"
ENV VOICES_DIR="/data/voices"
ENV MODELS_DIR="/models"
ENV SERVER_CONFIG_PATH="/app/server.json"

ENTRYPOINT ["/app/run.sh"]
