from __future__ import annotations

from pathlib import Path

import pytest


def write_config(tmp_path: Path, *, mode: str = "pull") -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[paths]
base_dir = "{tmp_path}"
inbox = "inbox"
processing = "processing"
done = "done"
audio = "www/audio"
www = "www"
voices = "voices"
status_file = "var/status.txt"
db = "var/reeder.db"
artifacts = "var/artifacts"
logs = "var/logs"

[jobs]
mode = "{mode}"
lease_seconds = 180
heartbeat_seconds = 15
claim_wait_seconds = 1
max_attempts = 2

[jobs.worker_api]
token_env = "REEDER_WORKER_TOKEN"
max_artifact_mb = 4096

[tts]
default_voice = "default.wav"
model = "fake-qwen3"
temperature = 0.8
device = "cpu"
audio_format = "opus"
opus_bitrate = 48
max_tokens_per_chunk = 100

[feed]
title = "Reeder Test"
description = "Test feed"
author = "Tests"
language = "en-us"
base_url = "http://test"
image = "cover.jpg"

[extractors]
default = "trafilatura"
""",
        encoding="utf-8",
    )
    return config


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path)


@pytest.fixture
def config(config_path: Path) -> dict:
    from reeder.config import load_config

    return load_config(config_path)
