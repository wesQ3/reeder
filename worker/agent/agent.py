"""Pull-worker agent for the GPU rig.

The agent runs on the rig host (not inside the GPU container). It claims one
task at a time from the Reeder host, starts the local container/runtime,
forwards progress and logs, uploads artifacts, and optionally suspends the rig
after an idle drain.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import ApiError, HostClient, RuntimeClient, env_token

AGENT_VERSION = "0.1.0"


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} [agent] {message}", flush=True)


class LeaseLost(Exception):
    pass


@dataclass
class AgentConfig:
    host_url: str = "http://127.0.0.1:8081"
    token_env: str = "REEDER_WORKER_TOKEN"
    worker_id: str = "nvidia-rig"
    name: str = "GPU rig"
    capabilities: list[str] = field(
        default_factory=lambda: ["tts", "tts:qwen3", "gpu:cuda"]
    )
    inventory: dict = field(default_factory=dict)
    poll_wait_seconds: int = 30
    heartbeat_seconds: int = 15
    lease_seconds: int = 180
    max_tasks_per_wake: int = 1
    max_runtime_seconds: int = 6 * 3600
    max_task_runtime_seconds: int = 5 * 3600
    idle_grace_seconds: int = 0
    on_idle: str = "none"
    suspend_command: list[str] = field(default_factory=lambda: ["systemctl", "suspend"])
    work_dir: Path = Path("/var/lib/reeder-worker")
    compose_file: str = ""
    compose_command: list[str] = field(default_factory=lambda: ["docker", "compose"])
    start_container: bool = True
    stop_container: bool = True
    container_start_timeout: int = 120
    runtime_url: str = "http://127.0.0.1:8100"
    host_wait_seconds: int = 300

    @property
    def token(self) -> str | None:
        return env_token(self.token_env)

    @property
    def outbox_dir(self) -> Path:
        return self.work_dir / "outbox"


def _as_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return list(default or [])


def load_config(path: Path | str) -> AgentConfig:
    raw: dict = {}
    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

    host = raw.get("host", {})
    worker = raw.get("worker", {})
    paths = raw.get("paths", {})
    runtime = raw.get("runtime", {})
    container = raw.get("container", {})
    config = AgentConfig(
        host_url=host.get("url", AgentConfig.host_url),
        token_env=host.get("token_env", AgentConfig.token_env),
        worker_id=worker.get("id", AgentConfig.worker_id),
        name=worker.get("name", AgentConfig.name),
        capabilities=_as_list(worker.get("capabilities"), AgentConfig().capabilities),
        inventory=worker.get("inventory", {}),
        poll_wait_seconds=int(worker.get("poll_wait_seconds", 30)),
        heartbeat_seconds=int(worker.get("heartbeat_seconds", 15)),
        lease_seconds=int(worker.get("lease_seconds", 180)),
        max_tasks_per_wake=int(worker.get("max_tasks_per_wake", 1)),
        max_runtime_seconds=int(worker.get("max_runtime_seconds", 6 * 3600)),
        max_task_runtime_seconds=int(
            worker.get("max_task_runtime_seconds", 5 * 3600)
        ),
        idle_grace_seconds=int(worker.get("idle_grace_seconds", 0)),
        on_idle=str(worker.get("on_idle", "none")),
        suspend_command=_as_list(
            worker.get("suspend_command"), ["systemctl", "suspend"]
        ),
        work_dir=Path(paths.get("work_dir", "/var/lib/reeder-worker")),
        compose_file=str(paths.get("compose_file", "")),
        compose_command=_as_list(
            container.get("compose_command"), ["docker", "compose"]
        ),
        start_container=bool(container.get("start", True)),
        stop_container=bool(container.get("stop", True)),
        container_start_timeout=int(container.get("start_timeout_seconds", 120)),
        runtime_url=runtime.get("url", "http://127.0.0.1:8100"),
        host_wait_seconds=int(host.get("wait_seconds", 300)),
    )
    return config


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        host: HostClient | None = None,
        runtime: RuntimeClient | None = None,
        sleep=time.sleep,
    ):
        self.config = config
        self.host = host or HostClient(config.host_url, token=config.token)
        self.runtime = runtime or RuntimeClient(config.runtime_url)
        self.sleep = sleep
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.config.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event: threading.Event | None = None
        self._lease_lost = threading.Event()
        self._last_log_seq = 0

    # ------------------------------------------------------------------
    # lifecycle helpers
    # ------------------------------------------------------------------
    def _registration(self) -> dict:
        return {
            "worker_id": self.config.worker_id,
            "name": self.config.name,
            "capabilities": self.config.capabilities,
            "inventory": self.config.inventory,
            "version": AGENT_VERSION,
        }

    def wait_for_host(self) -> bool:
        deadline = time.monotonic() + max(0, self.config.host_wait_seconds)
        while True:
            if self.host.healthz():
                return True
            if time.monotonic() >= deadline:
                return False
            _log(f"waiting for host {self.config.host_url}")
            self.sleep(min(10, max(1, deadline - time.monotonic())))

    def ensure_container(self) -> bool:
        if self.config.start_container and self.config.compose_file:
            command = [
                *self.config.compose_command,
                "-f",
                self.config.compose_file,
                "up",
                "-d",
            ]
            _log("starting worker container: " + " ".join(command))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                _log(
                    "container start failed: "
                    + (result.stderr or result.stdout or "unknown error").strip()
                )
                return False

        deadline = time.monotonic() + max(1, self.config.container_start_timeout)
        while time.monotonic() < deadline:
            if self.runtime.healthy():
                _log("worker runtime is healthy")
                return True
            self.sleep(2)
        _log("worker runtime did not become healthy")
        return False

    def stop_container(self) -> None:
        if not (self.config.stop_container and self.config.compose_file):
            return
        command = [
            *self.config.compose_command,
            "-f",
            self.config.compose_file,
            "stop",
        ]
        _log("stopping worker container")
        subprocess.run(command, capture_output=True, text=True)

    def run_idle_hook(self) -> None:
        if self.config.idle_grace_seconds:
            self.sleep(self.config.idle_grace_seconds)
        if self.config.on_idle in ("none", "", "exit"):
            return
        if self.config.on_idle == "suspend":
            command = self.config.suspend_command
        else:
            command = _as_list(self.config.on_idle)
        if not command:
            return
        _log("running idle hook: " + " ".join(command))
        subprocess.run(command, check=False)

    # ------------------------------------------------------------------
    # outbox
    # ------------------------------------------------------------------
    def _outbox_path(self, task_id: str) -> Path:
        return self.config.outbox_dir / f"{task_id}.json"

    def _write_outbox(self, record: dict) -> None:
        path = self._outbox_path(record["task"]["id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _clear_outbox(self, task_id: str) -> None:
        path = self._outbox_path(task_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def reconcile_outbox(self) -> None:
        """Reconcile durable local results after a crash or reboot."""
        for path in sorted(self.config.outbox_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                task = record["task"]
                remote = self.host.get_task(task["id"])
            except ApiError as exc:
                if exc.status == 404:
                    path.unlink(missing_ok=True)
                else:
                    _log(f"outbox reconcile deferred for {path.stem}: {exc}")
                continue
            except Exception as exc:
                _log(f"ignoring invalid outbox entry {path}: {exc}")
                path.unlink(missing_ok=True)
                continue

            status = remote.get("status")
            if status in ("completed", "failed", "canceled"):
                _log(f"outbox {task['id']} already {status}; purging")
                path.unlink(missing_ok=True)
                continue
            if (
                int(remote.get("attempt", 0)) != int(task["attempt"])
                or remote.get("lease_token") != task["lease_token"]
            ):
                _log(f"outbox {task['id']} is stale; purging")
                path.unlink(missing_ok=True)
                continue

            output_path = self.config.work_dir / record.get("output_rel", "")
            metrics_path = self.config.work_dir / record.get("metrics_rel", "")
            if output_path.is_file():
                try:
                    result = self._upload_result(
                        task, output_path, metrics_path, record.get("metrics", {})
                    )
                    self.host.complete_task(
                        task["id"],
                        worker_id=self.config.worker_id,
                        attempt=task["attempt"],
                        lease_token=task["lease_token"],
                        result=result,
                    )
                    _log(f"reconciled completed outbox {task['id']}")
                    path.unlink(missing_ok=True)
                except ApiError as exc:
                    _log(f"outbox upload deferred for {task['id']}: {exc}")
                continue

            try:
                self.host.release_task(
                    task["id"],
                    worker_id=self.config.worker_id,
                    attempt=task["attempt"],
                    lease_token=task["lease_token"],
                    reason="Agent restarted before local output was available",
                )
                path.unlink(missing_ok=True)
            except ApiError as exc:
                _log(f"outbox release failed for {task['id']}: {exc}")

    # ------------------------------------------------------------------
    # runtime forwarding
    # ------------------------------------------------------------------
    def _heartbeat_loop(self, task: dict, local_task_id: str) -> None:
        interval = max(3, min(self.config.heartbeat_seconds, 30))
        while not self._stop_event.wait(interval):
            try:
                status = self.runtime.status(local_task_id)
                response = self.host.heartbeat(
                    task["id"],
                    worker_id=self.config.worker_id,
                    attempt=task["attempt"],
                    lease_token=task["lease_token"],
                    state="running",
                    progress=status.get("progress"),
                    message=status.get("state"),
                    metrics=status.get("metrics"),
                    lease_seconds=self.config.lease_seconds,
                )
                if response.get("cancel_requested"):
                    _log(f"host requested cancellation of {task['id']}")
                    try:
                        self.runtime.cancel(local_task_id)
                    except ApiError:
                        pass
            except ApiError as exc:
                if exc.status in (404, 409):
                    _log(f"lease lost for {task['id']}: {exc}")
                    self._lease_lost.set()
                    try:
                        self.runtime.cancel(local_task_id)
                    except ApiError:
                        pass
                    return
                _log(f"heartbeat error for {task['id']}: {exc}")

    def _flush_logs(self, task: dict, local_task_id: str) -> None:
        try:
            batch = self.runtime.logs(local_task_id, self._last_log_seq)
        except ApiError:
            return
        entries = batch.get("entries") or []
        if not entries:
            return
        try:
            self.host.upload_logs(
                task["id"],
                worker_id=self.config.worker_id,
                attempt=task["attempt"],
                lease_token=task["lease_token"],
                entries=entries,
            )
            self._last_log_seq = int(batch.get("last_seq", self._last_log_seq))
        except ApiError as exc:
            if exc.status in (404, 409):
                self._lease_lost.set()
            else:
                _log(f"log upload error for {task['id']}: {exc}")

    def _log_loop(self, task: dict, local_task_id: str) -> None:
        while not self._stop_event.wait(1.0):
            self._flush_logs(task, local_task_id)

    # ------------------------------------------------------------------
    # task execution
    # ------------------------------------------------------------------
    def _upload_result(
        self,
        task: dict,
        output_path: Path,
        metrics_path: Path,
        metrics: dict,
    ) -> dict:
        output_artifact = self.host.upload_artifact(
            task,
            worker_id=self.config.worker_id,
            path=output_path,
            filename="output.wav",
            content_type="audio/wav",
            kind="audio",
        )
        metrics_artifact = None
        if metrics_path.is_file():
            metrics_artifact = self.host.upload_artifact(
                task,
                worker_id=self.config.worker_id,
                path=metrics_path,
                filename="metrics.json",
                content_type="application/json",
                kind="metrics",
            )
        return {
            "output_artifact": output_artifact["id"],
            "metrics_artifact": metrics_artifact["id"] if metrics_artifact else None,
            "metrics": metrics,
            "engine": task.get("payload", {}).get("engine"),
            "model": task.get("payload", {}).get("model"),
        }

    def execute(self, task: dict) -> bool:
        task_id = task["id"]
        self._lease_lost.clear()
        self._last_log_seq = 0
        local_dir = self.config.work_dir / "tasks" / task_id
        local_dir.mkdir(parents=True, exist_ok=True)
        input_path = local_dir / "input.json"

        try:
            self.host.download_task_input(task_id, input_path)
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log(f"could not download task input for {task_id}: {exc}")
            try:
                self.host.fail_task(
                    task_id,
                    worker_id=self.config.worker_id,
                    attempt=task["attempt"],
                    lease_token=task["lease_token"],
                    error=f"Input download failed: {exc}",
                    retryable=True,
                )
            except ApiError:
                pass
            return False

        output_rel = f"tasks/{task_id}/output.wav"
        output_path = self.config.work_dir / output_rel
        metrics_path = output_path.with_name("metrics.json")
        record = {
            "task": task,
            "local_task_id": None,
            "output_rel": output_rel,
            "metrics_rel": str(metrics_path.relative_to(self.config.work_dir)),
            "metrics": {},
        }
        self._write_outbox(record)

        try:
            started = self.runtime.start(payload, output_rel)
        except ApiError as exc:
            _log(f"runtime start failed for {task_id}: {exc}")
            self._fail(task, f"Runtime start failed: {exc}", retryable=True)
            self._clear_outbox(task_id)
            return False

        local_task_id = started["id"]
        record["local_task_id"] = local_task_id
        self._write_outbox(record)
        self._stop_event = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(task, local_task_id),
            daemon=True,
            name=f"heartbeat-{task_id}",
        )
        log_thread = threading.Thread(
            target=self._log_loop,
            args=(task, local_task_id),
            daemon=True,
            name=f"logs-{task_id}",
        )
        heartbeat.start()
        log_thread.start()

        deadline = time.monotonic() + self.config.max_task_runtime_seconds
        status: dict = {}
        timed_out = False
        try:
            while True:
                if self._lease_lost.is_set():
                    break
                try:
                    status = self.runtime.status(local_task_id)
                except ApiError as exc:
                    _log(f"runtime status failed for {task_id}: {exc}")
                    self.sleep(2)
                    continue
                if status.get("state") in ("completed", "failed", "canceled"):
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _log(f"local task {task_id} exceeded max runtime; canceling")
                    try:
                        self.runtime.cancel(local_task_id)
                    except ApiError:
                        pass
                self.sleep(1)
        finally:
            self._stop_event.set()
            heartbeat.join(timeout=5)
            log_thread.join(timeout=5)
            self._flush_logs(task, local_task_id)

        if self._lease_lost.is_set():
            _log(f"not reporting task {task_id}; lease was lost")
            return False
        if timed_out:
            self._fail(task, "Local task exceeded maximum runtime", retryable=True)
            self._clear_outbox(task_id)
            return False
        if status.get("state") != "completed":
            error = status.get("error") or f"Local task ended in state {status.get('state')}"
            self._fail(task, error, retryable=True)
            self._clear_outbox(task_id)
            return False

        metrics = (status.get("result") or {}).get("metrics") or status.get("metrics") or {}
        if not output_path.is_file():
            self._fail(task, "Runtime completed without an output WAV", retryable=True)
            self._clear_outbox(task_id)
            return False

        try:
            result = self._upload_result(task, output_path, metrics_path, metrics)
            self.host.complete_task(
                task_id,
                worker_id=self.config.worker_id,
                attempt=task["attempt"],
                lease_token=task["lease_token"],
                result=result,
            )
        except ApiError as exc:
            if exc.status in (404, 409):
                _log(f"completion rejected for {task_id}: {exc}")
            else:
                _log(f"completion upload failed for {task_id}; kept in outbox: {exc}")
            return False

        self._clear_outbox(task_id)
        _log(f"completed task {task_id} ({status.get('result', {}).get('size', 0)} bytes)")
        return True

    def _fail(self, task: dict, error: str, *, retryable: bool) -> None:
        try:
            self.host.fail_task(
                task["id"],
                worker_id=self.config.worker_id,
                attempt=task["attempt"],
                lease_token=task["lease_token"],
                error=error,
                retryable=retryable,
            )
        except ApiError as exc:
            _log(f"could not report failure for {task['id']}: {exc}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        started = time.monotonic()
        if not self.wait_for_host():
            _log("host unreachable; exiting without claiming work")
            return 0
        self.host.register(self._registration())
        _log(f"registered as {self.config.worker_id}")
        self.reconcile_outbox()

        if not self.ensure_container():
            _log("worker container is unavailable; exiting")
            return 0

        completed = 0
        attempted = 0
        try:
            while attempted < self.config.max_tasks_per_wake:
                if time.monotonic() - started >= self.config.max_runtime_seconds:
                    _log("maximum wake runtime reached")
                    break
                task = self.host.claim(
                    worker_id=self.config.worker_id,
                    name=self.config.name,
                    capabilities=self.config.capabilities,
                    inventory=self.config.inventory,
                    lease_seconds=self.config.lease_seconds,
                    wait_seconds=self.config.poll_wait_seconds,
                    version=AGENT_VERSION,
                )
                if task is None:
                    _log("no compatible task; idle")
                    break
                task = task["task"]
                attempted += 1
                _log(f"claimed {task['type']} {task['id']} attempt {task['attempt']}")
                if self.execute(task):
                    completed += 1
                self.host.worker_heartbeat(
                    self.config.worker_id, state="idle", status={"tasks_done": completed}
                )
        finally:
            self.stop_container()
        _log(f"drained {completed} task(s) in {attempted} attempt(s)")
        self.run_idle_hook()
        return completed


def _acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another agent instance holds {path}")
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Reeder pull-worker agent")
    parser.add_argument(
        "--config", default=os.environ.get("REEDER_AGENT_CONFIG", "/etc/reeder-worker/agent.toml")
    )
    parser.add_argument("--once", action="store_true", help="Claim at most one task")
    parser.add_argument("--no-container", action="store_true", help="Do not manage compose")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.once:
        config.max_tasks_per_wake = 1
        config.on_idle = "none"
    if args.no_container:
        config.start_container = False
        config.stop_container = False
    if not config.token:
        sys.stderr.write(
            f"Missing worker token in ${config.token_env}; set it in the agent environment.\n"
        )
        raise SystemExit(2)

    lock = _acquire_lock(config.work_dir / "agent.lock")
    try:
        agent = Agent(config)
        agent.run()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
