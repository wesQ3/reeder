"""Remote TTS dispatch — call a remote GPU worker API.

Tries the remote worker first; falls back to local generation if unavailable.
"""

import socket
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


def send_wowlan_magic_packet(
    mac_address: str,
    broadcast: str = "255.255.255.255",
    port: int = 9,
    retries: int = 3,
    interval: float = 1.0,
):
    """Send a WoWLAN magic packet to a target MAC address."""
    normalized = mac_address.replace(":", "").replace("-", "").strip().lower()
    if len(normalized) != 12 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"Invalid MAC address: {mac_address!r}")

    payload = bytes.fromhex("ff" * 6 + normalized * 16)
    for attempt in range(max(1, int(retries))):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(payload, (broadcast, int(port)))
        finally:
            sock.close()
        if attempt + 1 < retries:
            time.sleep(max(0.0, interval))


def wake_remote_if_configured(
    url: str,
    remote_config: dict,
    status: Callable[[str], None],
) -> bool:
    """Attempt to wake the remote host and wait for worker health."""
    if not remote_config.get("wake_enabled", False):
        return False

    wake_mac = remote_config.get("wake_mac", "").strip()
    if not wake_mac:
        status("Wake is enabled but wake_mac is missing; skipping wake")
        return False

    wake_broadcast = remote_config.get("wake_broadcast", "255.255.255.255")
    wake_port = int(remote_config.get("wake_port", 9))
    wake_retries = int(remote_config.get("wake_retries", 5))
    wake_interval = float(remote_config.get("wake_retry_interval", 1.0))
    wake_wait = float(remote_config.get("wake_wait", 60))
    wake_poll = float(remote_config.get("wake_poll_interval", 3))

    status(f"Sending WoWLAN magic packet to {wake_mac} via {wake_broadcast}:{wake_port}")
    send_wowlan_magic_packet(
        wake_mac,
        broadcast=wake_broadcast,
        port=wake_port,
        retries=wake_retries,
        interval=wake_interval,
    )

    status(f"Waiting up to {wake_wait:.0f}s for remote worker...")
    deadline = time.monotonic() + wake_wait
    while time.monotonic() < deadline:
        if check_remote_health(url, timeout=5.0):
            status("Remote worker is reachable after wake")
            return True
        time.sleep(max(0.5, wake_poll))

    status("Remote worker did not come online in time")
    return False


def _prechunk_text(text: str, config: dict) -> list[dict]:
    """Split text into chunks with real token counts for the remote worker.

    The remote audio.cpp worker has no tokenizer; the job processor does.
    Splitting here (with reeder.tts's sentence-boundary chunker and the actual
    Qwen tokenizer) keeps the worker's CUDA graph allocations bounded —
    audio.cpp's qwen3_tts decoder sizes its graph to the input.
    """
    from reeder.tts import get_tts_tokenizer, split_text_into_chunks

    model_name = config["tts"].get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    max_tokens = config["tts"].get("max_tokens_per_chunk", 100)
    tokenizer = get_tts_tokenizer(model_name)
    pieces = split_text_into_chunks(text, tokenizer, max_tokens=max_tokens)
    return [
        {"text": piece, "tokens": len(tokenizer(piece, return_tensors="pt").input_ids[0])}
        for piece in pieces
    ]


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
        status("Remote worker unreachable")
        woke = wake_remote_if_configured(url, remote_config, status)
        if not woke and not check_remote_health(url):
            status("Falling back to local")
            return False

    status("Remote worker available, dispatching...")

    # Split into chunks with the real tokenizer before dispatching. The worker
    # accepts pre-split chunks so it never has to size its CUDA graph to a
    # full article.
    try:
        chunks = _prechunk_text(text, config)
    except Exception as e:
        status(f"Tokenizer unavailable for pre-splitting ({e}); falling back to local")
        return False
    if len(chunks) > 1:
        status(f"Pre-split text into {len(chunks)} chunks (max {config['tts'].get('max_tokens_per_chunk', 100)} tokens each)")

    # Build request
    voice = job.get("voice", "default")
    temperature = job.get("temperature", config["tts"].get("temperature", 0.8))
    language = job.get("language", "Auto")

    payload = json.dumps({
        "chunks": chunks,
        "voice": voice,
        "temperature": temperature,
        "language": language,
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
