"""Shared job submission helper for the Reeder CLI scripts."""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def find_config() -> Path | None:
    env = os.environ.get("REEDER_CONFIG")
    if env:
        return Path(env)
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "config.toml",
        Path("/etc/reeder/config.toml"),
        Path.home() / ".config/reeder/config.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_config() -> dict:
    path = find_config()
    if path is None:
        return {}
    with open(path, "rb") as f:
        config = tomllib.load(f)
    override = path.with_name(f"{path.stem}.override{path.suffix}")
    if override.exists():
        with open(override, "rb") as f:
            override_data = tomllib.load(f)
        # One-level-deep merge is enough for the sections used by CLI scripts.
        for key, value in override_data.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                merged = dict(config[key])
                merged.update(value)
                config[key] = merged
            else:
                config[key] = value
    config["_config_path"] = path
    return config


def base_dir(config: dict) -> Path:
    raw = config.get("paths", {}).get("base_dir", "/var/lib/reeder")
    base = Path(raw)
    if not base.is_absolute():
        config_path = config.get("_config_path")
        base = (config_path.parent / base).resolve() if config_path else base.resolve()
    return base


def api_url(config: dict) -> str:
    explicit = os.environ.get("REEDER_API_URL") or os.environ.get("REEDER_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("REEDER_WEB_PORT", "8081")
    return f"http://127.0.0.1:{port}"


def slugify(text: str, max_len: int = 50) -> str:
    text = re.sub(r"https?://", "", text or "")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text)
    return text[:max_len].strip("-") or "job"


def write_legacy_inbox(config: dict, request: dict) -> dict:
    paths = config.get("paths", {})
    inbox = base_dir(config) / paths.get("inbox", "inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    stamp = int(datetime.now(timezone.utc).timestamp())
    filename = f"{stamp}-{slugify(str(request.get('title') or request.get('url') or 'text'))}.json"
    with open(inbox / filename, "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)
    return {"id": Path(filename).stem, "legacy_inbox_file": str(inbox / filename)}


def submit(request: dict) -> dict:
    config = load_config()
    mode = config.get("jobs", {}).get("mode", "push")
    if mode == "push":
        return write_legacy_inbox(config, request)

    url = f"{api_url(config)}/api/v1/jobs"
    body = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Host rejected job ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not reach Reeder API at {url}: {exc}. "
            "Set REEDER_API_URL if the web service is elsewhere."
        ) from exc


def fail(message: str, code: int = 1) -> None:
    sys.stderr.write(message + "\n")
    raise SystemExit(code)
