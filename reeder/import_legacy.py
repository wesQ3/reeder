"""One-time importer for legacy inbox jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from reeder.config import get_paths, load_config
from reeder.store import StoreError, store_from_config


def import_inbox(*, dry_run: bool = False) -> dict:
    config = load_config()
    paths = get_paths(config)
    store = store_from_config(config)
    imported_dir = paths["base"] / "var" / "imported"
    imported_dir.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    errors: list[dict] = []

    for path in sorted(paths["inbox"].glob("*.json")):
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                raise ValueError("job file is not a JSON object")
            if dry_run:
                imported.append(path.name)
                continue
            job = store.create_job(request)
            destination = imported_dir / path.name
            if destination.exists():
                destination = imported_dir / f"{path.stem}-{job['id']}.json"
            shutil.move(str(path), destination)
            imported.append(job["id"])
        except (OSError, ValueError, StoreError) as exc:
            errors.append({"file": path.name, "error": str(exc)})

    return {"imported": imported, "errors": errors, "dry_run": dry_run}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import pending legacy inbox/*.json files into the pull queue"
    )
    parser.add_argument("--config", help="Path to config.toml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        os.environ["REEDER_CONFIG"] = str(Path(args.config).resolve())
    result = import_inbox(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
