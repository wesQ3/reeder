"""Small stdlib HTTP clients for the host API and local worker runtime.

Keeping the agent on urllib avoids pulling the container's GPU dependency set
onto the rig host. Only the Python standard library is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ApiError(Exception):
    def __init__(self, status: int, body: str, url: str = ""):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} for {url}: {body[:500]}")


class HttpClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        raw: bool = False,
    ) -> Any:
        headers = dict(headers or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                req, timeout=timeout if timeout is not None else self.timeout
            ) as response:
                body = response.read()
                if raw:
                    return body
                if not body:
                    return None
                try:
                    return json.loads(body)
                except ValueError:
                    return body.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ApiError(exc.code, body, req.full_url) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ApiError(0, str(exc), req.full_url) from exc


class HostClient(HttpClient):
    """Client for the authenticated Reeder host worker API."""

    def healthz(self) -> bool:
        try:
            result = self.request("GET", "/healthz", timeout=5)
            return isinstance(result, dict) and bool(result.get("ok"))
        except ApiError:
            return False

    def register(self, registration: dict) -> dict:
        return self.request("POST", "/api/v1/workers/register", json_body=registration)

    def worker_heartbeat(
        self,
        worker_id: str,
        *,
        state: str = "idle",
        current_task_id: str | None = None,
        status: dict | None = None,
    ) -> dict:
        return self.request(
            "POST",
            "/api/v1/workers/heartbeat",
            json_body={
                "worker_id": worker_id,
                "state": state,
                "current_task_id": current_task_id,
                "status": status or {},
            },
        )

    def claim(
        self,
        *,
        worker_id: str,
        name: str,
        capabilities: list[str],
        inventory: dict,
        lease_seconds: int,
        wait_seconds: int,
        version: str | None = None,
    ) -> dict | None:
        try:
            return self.request(
                "POST",
                "/api/v1/tasks/claim",
                json_body={
                    "worker_id": worker_id,
                    "name": name,
                    "capabilities": capabilities,
                    "inventory": inventory,
                    "lease_seconds": lease_seconds,
                    "wait_seconds": wait_seconds,
                    "max_tasks": 1,
                    "version": version,
                },
                timeout=max(30, wait_seconds + 20),
            )
        except ApiError as exc:
            if exc.status == 204:
                return None
            raise

    def get_task(self, task_id: str) -> dict:
        return self.request("GET", f"/api/v1/tasks/{task_id}")

    def heartbeat(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        state: str = "running",
        progress: dict | None = None,
        message: str | None = None,
        metrics: dict | None = None,
        lease_seconds: int = 180,
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/heartbeat",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "state": state,
                "progress": progress,
                "message": message,
                "metrics": metrics,
                "lease_seconds": lease_seconds,
            },
        )

    def upload_logs(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        entries: list[dict],
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/logs",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "entries": entries,
            },
        )

    def init_artifact(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        filename: str,
        size: int,
        sha256: str,
        content_type: str,
        kind: str,
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/artifacts",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "filename": filename,
                "name": filename,
                "size": size,
                "sha256": sha256,
                "content_type": content_type,
                "kind": kind,
            },
        )

    def upload_artifact_chunk(self, artifact_id: str, offset: int, data: bytes) -> dict:
        return self.request(
            "PUT",
            f"/api/v1/artifacts/{artifact_id}?offset={offset}",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )

    def complete_artifact(self, artifact_id: str) -> dict:
        return self.request(
            "POST", f"/api/v1/artifacts/{artifact_id}/complete", json_body={}
        )

    def upload_artifact(
        self,
        task: dict,
        *,
        worker_id: str,
        path: Path,
        filename: str,
        content_type: str,
        kind: str,
        chunk_size: int = 512 * 1024,
    ) -> dict:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        init = self.init_artifact(
            task["id"],
            worker_id=worker_id,
            attempt=task["attempt"],
            lease_token=task["lease_token"],
            filename=filename,
            size=size,
            sha256=digest.hexdigest(),
            content_type=content_type,
            kind=kind,
        )
        offset = int(init.get("offset", 0))
        with open(path, "rb") as f:
            while offset < size:
                f.seek(offset)
                data = f.read(min(chunk_size, size - offset))
                result = self.upload_artifact_chunk(init["artifact_id"], offset, data)
                offset = int(result["next_offset"])
        return self.complete_artifact(init["artifact_id"])

    def download_task_input(self, task_id: str, destination: Path) -> Path:
        data = self.request("GET", f"/api/v1/tasks/{task_id}/input", raw=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def complete_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        result: dict,
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/complete",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "result": result,
            },
        )

    def fail_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        error: str,
        retryable: bool = True,
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/fail",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "error": error,
                "retryable": retryable,
            },
        )

    def release_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        reason: str,
    ) -> dict:
        return self.request(
            "POST",
            f"/api/v1/tasks/{task_id}/release",
            json_body={
                "worker_id": worker_id,
                "attempt": attempt,
                "lease_token": lease_token,
                "reason": reason,
            },
        )


class RuntimeClient(HttpClient):
    """Client for the worker container's local /runtime API."""

    def healthy(self) -> bool:
        try:
            result = self.request("GET", "/health", timeout=5)
            return isinstance(result, dict) and result.get("status") == "ready"
        except ApiError:
            # Older gateway builds may not have audio.cpp ready yet.
            return False

    def start(self, payload: dict, output_path: str) -> dict:
        return self.request(
            "POST",
            "/runtime/tasks",
            json_body={"kind": "tts", "payload": payload, "output_path": output_path},
        )

    def status(self, local_task_id: str) -> dict:
        return self.request("GET", f"/runtime/tasks/{local_task_id}")

    def logs(self, local_task_id: str, since: int = 0) -> dict:
        return self.request(
            "GET", f"/runtime/tasks/{local_task_id}/logs?since={int(since)}"
        )

    def cancel(self, local_task_id: str) -> dict:
        return self.request(
            "POST", f"/runtime/tasks/{local_task_id}/cancel", json_body={}
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_url(path: Path) -> str:
    return urllib.parse.urljoin("file:", path.as_uri())


def env_token(name: str) -> str | None:
    return os.environ.get(name) or None
