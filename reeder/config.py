"""Configuration loading and path resolution."""

import os
import tomllib
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override values into base config."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from config.toml.

    Checks REEDER_CONFIG env var first, then standard locations.
    The resolved config file path is stored in the returned dict as '_config_path'.
    """
    if config_path is None:
        env_path = os.environ.get("REEDER_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            candidates = [
                Path(__file__).parent.parent / "config.toml",
                Path("/etc/reeder/config.toml"),
                Path.home() / ".config/reeder/config.toml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
            else:
                raise FileNotFoundError("No config.toml found")

    config_path = config_path.resolve()
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    override_path = config_path.with_name(f"{config_path.stem}.override{config_path.suffix}")
    if override_path.exists():
        with open(override_path, "rb") as f:
            override = tomllib.load(f)
        config = _deep_merge(config, override)

    config["_config_path"] = config_path
    return config


def get_paths(config: dict) -> dict[str, Path]:
    """Resolve all paths from config.

    If base_dir is relative, it's resolved against the config file's directory.
    """
    base = Path(config["paths"]["base_dir"])
    if not base.is_absolute():
        config_dir = config.get("_config_path", Path.cwd()).parent
        base = (config_dir / base).resolve()

    return {
        "base": base,
        "inbox": base / config["paths"]["inbox"],
        "processing": base / config["paths"]["processing"],
        "done": base / config["paths"]["done"],
        "audio": base / config["paths"]["audio"],
        "www": base / config["paths"]["www"],
        "voices": base / config["paths"]["voices"],
        "status": base / config["paths"]["status_file"],
    }
