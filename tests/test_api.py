from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from reeder.config import load_config
from reeder.web import create_app

TOKEN = "test-worker-token"


@pytest.fixture
def client(config_path, monkeypatch):
    monkeypatch.setenv("REEDER_WORKER_TOKEN", TOKEN)
    app = create_app(config_path)
    with TestClient(app) as test_client:
        yield test_client


def auth_headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def add_worker_task(client, text="Hello world"):
    store = client.app.state.store
    job = store.create_job({"type": "text", "text": text, "title": "API test"})
    task = store.create_task(
        job["id"],
        "tts",
        executor="worker",
        required_caps=["tts", "tts:qwen3"],
        payload={"chunks": [{"text": text, "tokens": 2}]},
    )
    return job, task


def test_form_submission_creates_db_job(client):
    response = client.post(
        "/add",
        data={"type": "text", "title": "Form job", "text": "Some words", "voice": "default"},
        follow_redirects=False,
    )
    assert response.status_code == 200
    jobs = client.app.state.store.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["request"]["text"] == "Some words"


def test_worker_auth_is_required(client):
    assert client.post("/api/v1/tasks/claim", json={"worker_id": "rig"}).status_code == 401
    response = client.post(
        "/api/v1/tasks/claim",
        json={"worker_id": "rig", "wait_seconds": 0},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_worker_claim_heartbeat_logs_artifact_and_complete(client):
    job, task = add_worker_task(client)
    register = client.post(
        "/api/v1/workers/register",
        headers=auth_headers(),
        json={
            "worker_id": "rig",
            "name": "Test rig",
            "capabilities": ["tts", "tts:qwen3"],
            "inventory": {"voices": ["default"]},
        },
    )
    assert register.status_code == 200

    claim = client.post(
        "/api/v1/tasks/claim",
        headers=auth_headers(),
        json={
            "worker_id": "rig",
            "capabilities": ["tts", "tts:qwen3"],
            "wait_seconds": 0,
        },
    )
    assert claim.status_code == 200
    claimed = claim.json()["task"]
    assert claimed["id"] == task["id"]
    base = {
        "worker_id": "rig",
        "attempt": claimed["attempt"],
        "lease_token": claimed["lease_token"],
    }

    heartbeat = client.post(
        f"/api/v1/tasks/{task['id']}/heartbeat",
        headers=auth_headers(),
        json={**base, "state": "running", "progress": {"current": 1, "total": 2}},
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["cancel_requested"] is False

    logs = client.post(
        f"/api/v1/tasks/{task['id']}/logs",
        headers=auth_headers(),
        json={**base, "entries": [{"seq": 1, "message": "started"}]},
    )
    assert logs.status_code == 200
    assert logs.json()["accepted"] == 1

    wav = b"RIFFtest-wav"
    init = client.post(
        f"/api/v1/tasks/{task['id']}/artifacts",
        headers=auth_headers(),
        json={
            **base,
            "filename": "output.wav",
            "size": len(wav),
            "sha256": hashlib.sha256(wav).hexdigest(),
            "content_type": "audio/wav",
            "kind": "audio",
        },
    )
    assert init.status_code == 200
    artifact_id = init.json()["artifact_id"]
    upload = client.put(
        f"/api/v1/artifacts/{artifact_id}?offset=0",
        headers=auth_headers(),
        content=wav,
    )
    assert upload.status_code == 200
    completed_artifact = client.post(
        f"/api/v1/artifacts/{artifact_id}/complete", headers=auth_headers()
    )
    assert completed_artifact.status_code == 200
    assert completed_artifact.json()["sha256"] == hashlib.sha256(wav).hexdigest()

    complete = client.post(
        f"/api/v1/tasks/{task['id']}/complete",
        headers=auth_headers(),
        json={**base, "result": {"output_artifact": artifact_id}},
    )
    assert complete.status_code == 200
    assert complete.json()["status"] == "completed"

    # Duplicate completion is accepted idempotently.
    duplicate = client.post(
        f"/api/v1/tasks/{task['id']}/complete",
        headers=auth_headers(),
        json={**base, "result": {"output_artifact": artifact_id}},
    )
    assert duplicate.status_code == 200

    detail = client.get(f"/api/v1/jobs/{job['id']}")
    assert detail.status_code == 200
    assert detail.json()["tasks"][0]["status"] == "completed"
    logs_response = client.get(f"/api/v1/jobs/{job['id']}/logs")
    assert "started" in logs_response.text


def test_worker_cannot_claim_without_capabilities(client):
    add_worker_task(client)
    response = client.post(
        "/api/v1/tasks/claim",
        headers=auth_headers(),
        json={"worker_id": "rig", "capabilities": ["tts"], "wait_seconds": 0},
    )
    assert response.status_code == 204


def test_stale_attempt_is_conflict(client):
    _, task = add_worker_task(client)
    claim = client.post(
        "/api/v1/tasks/claim",
        headers=auth_headers(),
        json={"worker_id": "rig", "capabilities": ["tts", "tts:qwen3"], "wait_seconds": 0},
    ).json()["task"]
    response = client.post(
        f"/api/v1/tasks/{task['id']}/heartbeat",
        headers=auth_headers(),
        json={
            "worker_id": "rig",
            "attempt": claim["attempt"] + 1,
            "lease_token": claim["lease_token"],
        },
    )
    assert response.status_code == 409


def test_status_and_workers_api(client):
    add_worker_task(client)
    status = client.get("/api/v1/status")
    assert status.status_code == 200
    assert status.json()["jobs"]["queued"] == 1
    assert client.get("/api/v1/workers").status_code == 200
    assert client.get("/status").status_code == 200
