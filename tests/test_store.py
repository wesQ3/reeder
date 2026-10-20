from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest

from reeder.store import ConflictError, Store, ValidationError, store_from_config


def make_store(config: dict) -> Store:
    return store_from_config(config)


def make_text_job(store: Store):
    job = store.create_job({"type": "text", "text": "Hello world", "title": "Test"})
    task = store.create_task(
        job["id"],
        "tts",
        executor="worker",
        required_caps=["tts", "tts:qwen3"],
        payload={"chunks": [{"text": "Hello world", "tokens": 2}]},
    )
    return job, task


def test_capability_matching_and_claim(config):
    store = make_store(config)
    job, task = make_text_job(store)

    assert store.claim_task("wrong", capabilities=["gpu:cuda"]) is None
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    assert claimed["id"] == task["id"]
    assert claimed["attempt"] == 1
    assert claimed["lease_token"]
    assert store.get_job(job["id"])["status"] == "running"


def test_claim_has_single_winner(config):
    store = make_store(config)
    _, task = make_text_job(store)

    def claim(index: int):
        local = make_store(config)
        return local.claim_task(f"worker-{index}", capabilities=["tts", "tts:qwen3"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0]["id"] == task["id"]


def test_lease_expiry_retries_then_fails(config):
    store = make_store(config)
    job, task = make_text_job(store)
    store.claim_task("rig", capabilities=["tts", "tts:qwen3"], lease_seconds=60)

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (task["id"],),
        )
        conn.commit()

    reaped = store.reap_expired()
    assert len(reaped) == 1
    assert store.get_task(task["id"])["status"] == "queued"
    assert store.get_job(job["id"])["status"] == "running"

    claimed2 = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    assert claimed2["attempt"] == 2
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (task["id"],),
        )
        conn.commit()
    store.reap_expired()
    assert store.get_task(task["id"])["status"] == "failed"
    assert store.get_job(job["id"])["status"] == "failed"


def test_stale_lease_cannot_mutate(config):
    store = make_store(config)
    _, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE tasks SET lease_token='new-token' WHERE id=?", (task["id"],))
        conn.commit()
    with pytest.raises(ConflictError):
        store.heartbeat_task(
            task["id"],
            worker_id="rig",
            attempt=claimed["attempt"],
            lease_token=claimed["lease_token"],
        )


def test_artifact_upload_checksum_and_sanitized_name(config):
    store = make_store(config)
    _, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    data = b"RIFFfake-wave-data"
    digest = hashlib.sha256(data).hexdigest()

    artifact = store.init_artifact(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="../../output.wav",
        size=len(data),
        sha256=digest,
        content_type="audio/wav",
        kind="audio",
    )
    assert artifact["filename"] == "output.wav"
    # A retried chunk at the same offset is safe.
    assert store.upload_artifact(artifact["id"], offset=0, data=data[:5])[
        "next_offset"
    ] == 5
    assert store.upload_artifact(artifact["id"], offset=0, data=data[:5])[
        "next_offset"
    ] == 5
    store.upload_artifact(artifact["id"], offset=5, data=data[5:])
    completed = store.complete_artifact(artifact["id"])
    assert completed["sha256"] == digest
    assert store.artifact_content_path(artifact["id"]).read_bytes() == data


def test_artifact_rejects_bad_checksum(config):
    store = make_store(config)
    _, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    artifact = store.init_artifact(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="output.wav",
        size=4,
        sha256="0" * 64,
    )
    store.upload_artifact(artifact["id"], offset=0, data=b"abcd")
    with pytest.raises(ValidationError):
        store.complete_artifact(artifact["id"])


def test_log_sequence_high_water_mark(config):
    store = make_store(config)
    job, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    first = store.append_logs(
        task["id"],
        attempt=claimed["attempt"],
        worker_id="rig",
        lease_token=claimed["lease_token"],
        entries=[
            {"seq": 2, "message": "two"},
            {"seq": 1, "message": "one"},
        ],
    )
    assert first == {"last_seq": 2, "accepted": 2}
    second = store.append_logs(
        task["id"],
        attempt=claimed["attempt"],
        worker_id="rig",
        lease_token=claimed["lease_token"],
        entries=[{"seq": 2, "message": "duplicate"}, {"seq": 3, "message": "three"}],
    )
    assert second == {"last_seq": 3, "accepted": 1}
    text = store.read_log(job["id"], task["id"], claimed["attempt"])
    assert '"one"' in text and '"three"' in text
    assert "duplicate" not in text


def test_complete_is_idempotent(config):
    store = make_store(config)
    job, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    first = store.complete_task(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        result={"ok": True},
    )
    second = store.complete_task(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        result={"ok": True},
    )
    assert first["status"] == second["status"] == "completed"
    assert store.get_job(job["id"])["tasks"][0]["attempt"] == 1


def test_artifact_reinit_reports_resume_offset_and_completed_size(config):
    store = make_store(config)
    _, task = make_text_job(store)
    claimed = store.claim_task("rig", capabilities=["tts", "tts:qwen3"])
    artifact = store.init_artifact(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="output.wav",
        size=10,
        sha256=hashlib.sha256(b"0123456789").hexdigest(),
    )
    store.upload_artifact(artifact["id"], offset=0, data=b"01234")
    resumed = store.init_artifact(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="output.wav",
        size=10,
        sha256=hashlib.sha256(b"0123456789").hexdigest(),
    )
    assert resumed["offset"] == 5
    store.upload_artifact(artifact["id"], offset=5, data=b"56789")
    store.complete_artifact(artifact["id"])
    completed = store.init_artifact(
        task["id"],
        worker_id="rig",
        attempt=claimed["attempt"],
        lease_token=claimed["lease_token"],
        filename="output.wav",
        size=10,
        sha256=hashlib.sha256(b"0123456789").hexdigest(),
    )
    assert completed["offset"] == 10
