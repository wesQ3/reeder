from __future__ import annotations

import json
from types import SimpleNamespace

from reeder.orchestrator import Orchestrator
from reeder.store import store_from_config


class FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=[[0] * max(1, len(text.split()))])


def patch_tts(monkeypatch, chunks=None):
    import reeder.tts

    chunks = chunks or ["First sentence.", "Second sentence."]
    monkeypatch.setattr(reeder.tts, "get_tts_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr(
        reeder.tts,
        "split_text_into_chunks",
        lambda text, tokenizer, max_tokens=100: list(chunks),
    )


def test_url_pipeline_end_to_end_with_fake_worker(config, monkeypatch):
    import reeder.extract
    import reeder.tts

    patch_tts(monkeypatch)
    monkeypatch.setattr(
        reeder.extract, "extract_text", lambda job, cfg: "First sentence. Second sentence."
    )

    def fake_convert(wav_path, output_path, cfg):
        output_path.write_bytes(b"opus-data")
        return 12

    monkeypatch.setattr(reeder.tts, "convert_audio", fake_convert)

    store = store_from_config(config)
    orchestrator = Orchestrator(config, store=store, status=lambda message: None)
    job = store.create_job(
        {"type": "url", "url": "https://example.com/article", "title": "Example"}
    )

    # Host extracts text.
    assert orchestrator.tick()["host_tasks"] == 1
    fetch_task = store.get_job(job["id"])["tasks"][0]
    assert fetch_task["status"] == "completed"

    # Orchestrator prepares the tokenizer-dependent tts task.
    orchestrator.materialize_all()
    tts_task = store.task_by_type(job["id"], "tts")
    assert tts_task is not None
    assert tts_task["status"] == "queued"
    input_artifact = store.get_artifact(tts_task["payload"]["input_artifact"])
    payload = json.loads(store.artifact_content_path(input_artifact["id"]).read_text())
    assert payload["engine"] == "audio.cpp:qwen3-tts"
    assert len(payload["chunks"]) == 2

    # Fake worker claims and uploads a WAV artifact.
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    assert claimed["id"] == tts_task["id"]
    assert store.task_input_artifact(claimed["id"])["filename"] == "input.json"
    wav = b"RIFFfake"
    import hashlib

    artifact = store.init_artifact(
        claimed["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="output.wav",
        size=len(wav),
        sha256=hashlib.sha256(wav).hexdigest(),
        content_type="audio/wav",
        kind="audio",
    )
    store.upload_artifact(artifact["id"], offset=0, data=wav)
    store.complete_artifact(artifact["id"])
    store.complete_task(
        claimed["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        result={
            "output_artifact": artifact["id"],
            "engine": "audio.cpp:qwen3-tts",
            "model": "fake-qwen3",
            "metrics": {"rtf": 0.1},
        },
    )

    # Finalize runs on the host and publishes the projection.
    orchestrator.tick()
    final_job = store.get_job(job["id"])
    assert final_job["status"] == "completed"
    assert [task["type"] for task in final_job["tasks"]] == [
        "fetch_extract",
        "tts",
        "finalize_audio",
    ]
    done_path = orchestrator.paths["done"] / f"{job['id']}.json"
    assert done_path.exists()
    projection = json.loads(done_path.read_text())
    assert projection["_completed"]["audio_file"] == f"{job['id']}.opus"
    assert (orchestrator.paths["audio"] / f"{job['id']}.opus").exists()
    assert (orchestrator.paths["www"] / "feed.xml").exists()


def test_text_pipeline_prepares_tts_without_fetch(config, monkeypatch):
    patch_tts(monkeypatch, chunks=["Only chunk."])
    store = store_from_config(config)
    orchestrator = Orchestrator(config, store=store, status=lambda message: None)
    job = store.create_job({"type": "text", "text": "Only chunk.", "title": "Text"})
    assert store.get_job(job["id"])["tasks"] == []
    assert orchestrator.materialize_all() == 1
    tts = store.task_by_type(job["id"], "tts")
    assert tts["depends_on"] == []
    assert tts["payload"]["chunks"][0]["text"] == "Only chunk."


def test_fetch_failure_fails_job_without_worker(config, monkeypatch):
    import reeder.extract

    def boom(job, cfg):
        raise ValueError("extract failed")

    monkeypatch.setattr(reeder.extract, "extract_text", boom)
    store = store_from_config(config)
    orchestrator = Orchestrator(config, store=store, status=lambda message: None)
    job = store.create_job({"type": "url", "url": "https://bad.example", "title": "Bad"})
    orchestrator.tick()
    assert store.get_job(job["id"])["status"] == "failed"
    assert store.get_task(store.get_job(job["id"])["tasks"][0]["id"])["status"] == "failed"
