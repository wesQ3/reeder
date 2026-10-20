import json

from reeder.import_legacy import import_inbox
from reeder.store import store_from_config


def test_import_inbox(config_path, config, monkeypatch):
    monkeypatch.setenv("REEDER_CONFIG", str(config_path))
    from reeder.config import get_paths

    paths = get_paths(config)
    paths["inbox"].mkdir(parents=True, exist_ok=True)
    (paths["inbox"] / "123-test.json").write_text(
        json.dumps({"type": "text", "text": "Legacy text", "title": "Legacy"})
    )

    result = import_inbox()
    assert result["errors"] == []
    assert len(result["imported"]) == 1
    jobs = store_from_config(config).list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["request"]["text"] == "Legacy text"
    assert not list(paths["inbox"].glob("*.json"))
    assert (paths["base"] / "var" / "imported" / "123-test.json").exists()


def test_import_inbox_dry_run(config_path, config, monkeypatch):
    monkeypatch.setenv("REEDER_CONFIG", str(config_path))
    from reeder.config import get_paths

    paths = get_paths(config)
    paths["inbox"].mkdir(parents=True, exist_ok=True)
    (paths["inbox"] / "123-test.json").write_text(
        json.dumps({"type": "text", "text": "Legacy text", "title": "Legacy"})
    )
    result = import_inbox(dry_run=True)
    assert result["imported"] == ["123-test.json"]
    assert (paths["inbox"] / "123-test.json").exists()
    assert store_from_config(config).list_jobs() == []
