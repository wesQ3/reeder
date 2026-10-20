import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import Agent, AgentConfig
from agent.client import ApiError


class FakeHost:
    def __init__(self, work_dir: Path, task: dict | None = None):
        self.work_dir = work_dir
        self.task = task
        self.registered = None
        self.completed = None
        self.failed = None
        self.released = None
        self.logs = []
        self.artifacts = []
        self.task_state = dict(task or {})

    def healthz(self):
        return True

    def register(self, registration):
        self.registered = registration
        return registration

    def worker_heartbeat(self, *args, **kwargs):
        return {"ok": True}

    def claim(self, **kwargs):
        return {"task": self.task} if self.task else None

    def get_task(self, task_id):
        return self.task_state

    def download_task_input(self, task_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.task["payload"]))
        return destination

    def heartbeat(self, *args, **kwargs):
        return {"cancel_requested": False, "lease_expires_at": "later"}

    def upload_logs(self, task_id, *, entries, **kwargs):
        self.logs.extend(entries)
        return {"accepted": len(entries), "last_seq": entries[-1]["seq"]}

    def upload_artifact(self, task, *, path, filename, content_type, kind, **kwargs):
        self.artifacts.append({"path": path, "filename": filename, "kind": kind})
        return {"id": f"art_{len(self.artifacts)}", "filename": filename}

    def complete_task(self, task_id, **kwargs):
        self.completed = (task_id, kwargs)
        return {"status": "completed"}

    def fail_task(self, task_id, **kwargs):
        self.failed = (task_id, kwargs)
        return {"status": "failed"}

    def release_task(self, task_id, **kwargs):
        self.released = (task_id, kwargs)
        return {"status": "queued"}


class FakeRuntime:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.started = None
        self.canceled = False
        self.log_seq = 0
        self._wrote_output = False

    def healthy(self):
        return True

    def start(self, payload, output_path):
        self.started = {"payload": payload, "output_path": output_path}
        return {"id": "rt_1", "state": "queued"}

    def status(self, local_task_id):
        if not self._wrote_output:
            output = self.work_dir / self.started["output_path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"RIFFagent")
            output.with_name("metrics.json").write_text(
                json.dumps({"rtf": 0.2, "chunks_generated": 1})
            )
            self._wrote_output = True
        return {
            "state": "completed",
            "progress": {"current": 1, "total": 1, "percent": 100.0},
            "metrics": {"rtf": 0.2},
            "result": {
                "output_path": self.started["output_path"],
                "size": 9,
                "metrics": {"rtf": 0.2, "chunks_generated": 1},
            },
        }

    def logs(self, local_task_id, since=0):
        if since == 0:
            return {
                "entries": [{"seq": 1, "level": "info", "message": "started"}],
                "last_seq": 1,
            }
        return {"entries": [], "last_seq": since}

    def cancel(self, local_task_id):
        self.canceled = True
        return {"state": "canceling"}


def make_task():
    return {
        "id": "task_abc",
        "job_id": "job_1",
        "type": "tts",
        "attempt": 1,
        "lease_token": "lease-token",
        "heartbeat_interval_seconds": 15,
        "payload": {
            "engine": "audio.cpp:qwen3-tts",
            "model": "qwen3-tts",
            "chunks": [{"text": "Hello.", "tokens": 2}],
            "voice": "default",
        },
    }


@pytest.fixture
def agent_setup(tmp_path):
    work_dir = tmp_path / "work"
    config = AgentConfig(
        host_url="http://host",
        worker_id="rig",
        name="Rig",
        capabilities=["tts", "tts:qwen3"],
        work_dir=work_dir,
        compose_file="",
        start_container=False,
        stop_container=False,
        on_idle="none",
        poll_wait_seconds=0,
        heartbeat_seconds=15,
        max_task_runtime_seconds=20,
    )
    task = make_task()
    host = FakeHost(work_dir, task)
    runtime = FakeRuntime(work_dir)
    agent = Agent(config, host=host, runtime=runtime)
    return agent, host, runtime, task


def test_agent_executes_task_and_uploads_result(agent_setup):
    agent, host, runtime, task = agent_setup
    assert agent.execute(task) is True
    assert runtime.started["payload"]["chunks"][0]["text"] == "Hello."
    assert host.completed[0] == task["id"]
    result = host.completed[1]["result"]
    assert result["output_artifact"] == "art_1"
    assert result["metrics_artifact"] == "art_2"
    assert result["metrics"]["rtf"] == 0.2
    assert host.logs and host.logs[0]["message"] == "started"
    assert not (agent.config.outbox_dir / f"{task['id']}.json").exists()
    assert (agent.config.work_dir / "tasks" / task["id"] / "output.wav").exists()


def test_agent_run_loops_and_registers(agent_setup):
    agent, host, runtime, task = agent_setup
    assert agent.run() == 1
    assert host.registered["worker_id"] == "rig"
    assert host.completed is not None


def test_agent_keeps_outbox_when_completion_upload_fails(agent_setup, monkeypatch):
    agent, host, runtime, task = agent_setup

    def fail_complete(*args, **kwargs):
        raise ApiError(0, "network down")

    monkeypatch.setattr(host, "complete_task", fail_complete)
    assert agent.execute(task) is False
    assert (agent.config.outbox_dir / f"{task['id']}.json").exists()


def test_agent_reconciles_outbox_result(agent_setup):
    agent, host, runtime, task = agent_setup
    output_rel = f"tasks/{task['id']}/output.wav"
    output = agent.config.work_dir / output_rel
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"RIFFoutbox")
    metrics_path = output.with_name("metrics.json")
    metrics_path.write_text(json.dumps({"rtf": 0.3}))
    host.task_state = {
        **task,
        "status": "running",
        "attempt": task["attempt"],
        "lease_token": task["lease_token"],
    }
    record = {
        "task": task,
        "local_task_id": "rt_old",
        "output_rel": output_rel,
        "metrics_rel": str(metrics_path.relative_to(agent.config.work_dir)),
        "metrics": {"rtf": 0.3},
    }
    agent._write_outbox(record)

    agent.reconcile_outbox()
    assert host.completed is not None
    assert host.completed[1]["result"]["metrics"]["rtf"] == 0.3
    assert not (agent.config.outbox_dir / f"{task['id']}.json").exists()


def test_agent_purges_stale_outbox(agent_setup):
    agent, host, runtime, task = agent_setup
    host.task_state = {**task, "status": "queued", "attempt": 2, "lease_token": None}
    record = {
        "task": task,
        "local_task_id": "rt_old",
        "output_rel": "tasks/x/output.wav",
        "metrics_rel": "tasks/x/metrics.json",
        "metrics": {},
    }
    agent._write_outbox(record)
    agent.reconcile_outbox()
    assert not (agent.config.outbox_dir / f"{task['id']}.json").exists()
    assert host.completed is None
