"""Remote TTS dispatch — call a remote GPU worker API.

Tries the remote worker first; falls back to local generation if unavailable.
"""

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


def check_remote_health(url: str, timeout: float = 5.0) -> bool:
    """Check if the remote TTS worker is reachable and ready."""
    health_url = f"{url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def generate_audio_remote(
    text: str,
    output_path: Path,
    job: dict,
    config: dict,
    paths: dict,
    status_callback: Callable[[str], None] | None = None,
) -> bool:
    """Try to generate audio via the remote TTS worker.

    Returns True if successful, False if remote is unavailable (caller should fallback).
    Raises on remote errors that indicate a real problem (not just unavailability).
    """
    import json

    remote_config = config.get("tts", {}).get("remote", {})
    if not remote_config.get("enabled", False):
        return False

    url = remote_config["url"].rstrip("/")
    timeout = remote_config.get("timeout", 300)

    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"  [remote] {msg}", flush=True)

    # Check health
    status("Checking remote worker...")
    if not check_remote_health(url):
        status("Remote worker unreachable, falling back to local")
        return False

    status("Remote worker available, dispatching...")

    # Build request
    voice = job.get("voice", "default")
    temperature = job.get("temperature", config["tts"].get("temperature", 0.8))
    language = job.get("language", "Auto")
    max_tokens = config["tts"].get("max_tokens_per_chunk", 100)

    payload = json.dumps({
        "text": text,
        "voice": voice,
        "temperature": temperature,
        "language": language,
        "max_tokens_per_chunk": max_tokens,
    }).encode()

    generate_url = f"{url}/generate"
    req = urllib.request.Request(
        generate_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                status(f"Remote returned status {resp.status}, falling back")
                return False

            # Stream response to output file
            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)

        elapsed = time.monotonic() - start
        duration = resp.headers.get("X-Duration-Seconds", "?")
        rtf = resp.headers.get("X-RTF", "?")
        chunks = resp.headers.get("X-Chunks-Generated", "?")
        status(f"Done: {duration}s audio, {chunks} chunks, RTF={rtf}, took {elapsed:.1f}s")
        return True

    except (urllib.error.URLError, OSError, TimeoutError) as e:
        status(f"Remote request failed: {e}, falling back to local")
        return False


def generate_audio_with_fallback(
    text: str,
    output_path: Path,
    job: dict,
    config: dict,
    paths: dict,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    """Generate audio, trying remote first then falling back to local.

    Returns the backend used: "remote" or "local".
    """
    # Try remote first
    if generate_audio_remote(text, output_path, job, config, paths, status_callback):
        return "remote"

    # Fall back to local generation
    from reeder.tts import generate_audio
    generate_audio(text, output_path, job, config, paths, status_callback)
    return "local"
