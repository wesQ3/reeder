import asyncio
import sys
from pathlib import Path

import pytest
import httpx


@pytest.fixture
def anyio_backend():
    return "asyncio"

sys.path.insert(0, str(Path(__file__).parent.parent))

import tts_api


@pytest.fixture
async def runtime_client(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_api, "WORK_DIR", tmp_path)
    tts_api.RUNTIME_TASKS.clear()
    transport = httpx.ASGITransport(app=tts_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, tmp_path


@pytest.mark.anyio
async def test_runtime_task_progress_logs_and_artifact(runtime_client, monkeypatch):
    client, work_dir = runtime_client

    async def fake_synthesize(chunk_specs, *, progress_cb=None, log_cb=None, **kwargs):
        log_cb("info", "first chunk")
        progress_cb(1, len(chunk_specs), {"chunks_done": 1, "chunks_total": len(chunk_specs)})
        await asyncio.sleep(0.01)
        return {
            "wav": b"RIFFruntime",
            "duration_seconds": 1.5,
            "sample_rate": 24000,
            "chunks_generated": len(chunk_specs),
            "retried_chunks": 0,
            "generation_time": 0.2,
            "rtf": 0.13,
            "model": "qwen3-tts",
        }

    monkeypatch.setattr(tts_api, "synthesize_chunks", fake_synthesize)
    response = await client.post(
        "/runtime/tasks",
        json={
            "kind": "tts",
            "payload": {
                "engine": "audio.cpp:qwen3-tts",
                "chunks": [{"text": "Hello.", "tokens": 2}],
                "voice": "default",
            },
            "output_path": "tasks/remote-task/output.wav",
        },
    )
    assert response.status_code == 202
    task_id = response.json()["id"]

    for _ in range(100):
        status = (await client.get(f"/runtime/tasks/{task_id}")).json()
        if status["state"] in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(0.01)
    assert status["state"] == "completed", status
    assert status["progress"]["percent"] == 100.0
    assert status["result"]["output_path"] == "tasks/remote-task/output.wav"
    assert status["result"]["metrics"]["rtf"] == 0.13

    logs = (await client.get(f"/runtime/tasks/{task_id}/logs?since=0")).json()
    assert logs["last_seq"] > 0
    messages = [entry["message"] for entry in logs["entries"]]
    assert any("first chunk" in message for message in messages)
    assert (work_dir / "tasks/remote-task/output.wav").read_bytes() == b"RIFFruntime"
    assert (work_dir / "tasks/remote-task/metrics.json").exists()

    artifact = await client.get(f"/runtime/tasks/{task_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.content == b"RIFFruntime"


@pytest.mark.anyio
async def test_runtime_rejects_output_path_traversal(runtime_client, monkeypatch):
    client, _ = runtime_client

    async def fake_synthesize(*args, **kwargs):
        return {
            "wav": b"x",
            "duration_seconds": 1,
            "sample_rate": 24000,
            "chunks_generated": 1,
            "retried_chunks": 0,
            "generation_time": 0.1,
            "rtf": 0.1,
            "model": "fake",
        }

    monkeypatch.setattr(tts_api, "synthesize_chunks", fake_synthesize)
    response = await client.post(
        "/runtime/tasks",
        json={"payload": {"chunks": [{"text": "Hi"}]}, "output_path": "../escape.wav"},
    )
    task_id = response.json()["id"]
    for _ in range(100):
        status = (await client.get(f"/runtime/tasks/{task_id}")).json()
        if status["state"] in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(0.01)
    assert status["state"] == "failed"
    assert "escapes WORK_DIR" in status["error"]


@pytest.mark.anyio
async def test_runtime_cancel(runtime_client, monkeypatch):
    client, _ = runtime_client

    async def slow_synthesize(chunk_specs, *, cancel_check=None, **kwargs):
        for _ in range(500):
            if cancel_check and cancel_check():
                raise tts_api.GenerationCancelled("canceled")
            await asyncio.sleep(0.005)
        raise AssertionError("cancel was not observed")

    monkeypatch.setattr(tts_api, "synthesize_chunks", slow_synthesize)
    response = await client.post(
        "/runtime/tasks", json={"payload": {"chunks": [{"text": "Hi"}]}}
    )
    task_id = response.json()["id"]
    await asyncio.sleep(0.02)
    cancel = await client.post(f"/runtime/tasks/{task_id}/cancel")
    assert cancel.status_code == 200
    for _ in range(100):
        status = (await client.get(f"/runtime/tasks/{task_id}")).json()
        if status["state"] in ("completed", "failed", "canceled"):
            break
        await asyncio.sleep(0.01)
    assert status["state"] == "canceled"
