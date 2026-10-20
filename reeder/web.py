"""FastAPI web and worker API for Reeder.

The public form/feed behavior is preserved, while the worker-facing endpoints
implement the pull protocol described in ``PULL_WORKER.PLAN.md``.
"""

from __future__ import annotations

import html
import hmac
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from reeder.config import get_paths, load_config
from reeder.store import (
    ConflictError,
    NotFoundError,
    Store,
    StoreError,
    ValidationError,
    store_from_config,
    utcnow,
)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _enumerate_voices(voices_dir: Path) -> list[str]:
    if not voices_dir.exists():
        return ["default"]
    voices: list[str] = []
    for ext in ("*.safetensors", "*.wav"):
        for path in voices_dir.glob(ext):
            if path.stem not in voices:
                voices.append(path.stem)
    voices.sort(key=lambda value: (value != "default", value))
    return voices or ["default"]


def _render_form(config: dict, message: str = "") -> str:
    paths = get_paths(config)
    voices = _enumerate_voices(paths["voices"])
    options = "\n".join(
        f'        <option value="{html.escape(v)}">{html.escape(v)}</option>'
        for v in voices
    )
    template = _load_template("index.html")
    return template.replace("{{MESSAGE}}", message).replace("{{VOICE_OPTIONS}}", options)


def _legacy_create_job(paths: dict, request: dict) -> str:
    """Write a legacy inbox job file (push mode)."""
    from datetime import datetime, timezone
    import re

    stamp = int(datetime.now(timezone.utc).timestamp())
    title = str(request.get("title") or request.get("url") or "text")
    slug = re.sub(r"https?://", "", title)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug)[:50].strip("-")
    filename = f"{stamp}-{slug}.json"

    job = {"type": request.get("type", "url")}
    if job["type"] == "text":
        job["text"] = request.get("text", "")
    else:
        job["url"] = request.get("url", "")
    for key in ("title", "voice", "temperature", "language", "priority", "tts_profile"):
        if request.get(key) not in (None, ""):
            job[key] = request[key]

    inbox = paths["inbox"]
    inbox.mkdir(parents=True, exist_ok=True)
    with open(inbox / filename, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    return filename


def _submit(config: dict, store: Store, request: dict) -> dict:
    mode = config.get("jobs", {}).get("mode", "push")
    if mode == "push":
        filename = _legacy_create_job(get_paths(config), request)
        return {
            "id": Path(filename).stem,
            "status": "queued",
            "detail_url": "/status",
            "legacy_inbox_file": filename,
        }
    job = store.create_job(request)
    if mode == "dual" and config.get("jobs", {}).get("mirror_inbox", False):
        _legacy_create_job(get_paths(config), request)
    return {
        "id": job["id"],
        "status": job["status"],
        "detail_url": f"/jobs/{job['id']}",
    }


def _task_lease(body: dict) -> tuple[str, int, str, str]:
    try:
        worker_id = str(body["worker_id"])
        attempt = int(body["attempt"])
        lease_token = str(body["lease_token"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "worker_id, attempt, and lease_token are required")
    return worker_id, attempt, lease_token, str(body.get("reason") or "")


def create_app(config_path: Path | str | None = None) -> FastAPI:
    config = load_config(Path(config_path) if config_path else None)
    paths = get_paths(config)
    store = store_from_config(config)
    jobs_config = config.get("jobs", {})
    worker_api = jobs_config.get("worker_api", {})
    token_env = worker_api.get("token_env", "REEDER_WORKER_TOKEN")
    lease_seconds = int(jobs_config.get("lease_seconds", 180))
    claim_wait = int(jobs_config.get("claim_wait_seconds", 30))

    app = FastAPI(title="Reeder", version="0.2.0")
    app.state.config = config
    app.state.paths = paths
    app.state.store = store
    app.state.token_env = token_env

    def require_worker(request: Request) -> None:
        expected = os.environ.get(token_env, "")
        if not expected:
            raise HTTPException(
                503,
                f"Worker API token is not configured; set ${token_env} on the host",
            )
        header = request.headers.get("Authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
            raise HTTPException(401, "Invalid worker token")

    def worker_body(body: Any) -> dict:
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object")
        return body

    def handle_store_error(exc: Exception) -> HTTPException:
        if isinstance(exc, NotFoundError):
            return HTTPException(404, str(exc))
        if isinstance(exc, ConflictError):
            return HTTPException(409, str(exc))
        if isinstance(exc, ValidationError):
            return HTTPException(400, str(exc))
        return HTTPException(500, str(exc))

    # ------------------------------------------------------------------
    # Public UI and compatibility endpoints
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return _render_form(config)

    @app.get("/add")
    def add_get(
        request: Request,
        url: str = "",
        title: str = "",
        voice: str = "default",
    ):
        if not url:
            raise HTTPException(400, "Missing 'url' parameter")
        try:
            result = _submit(
                config, store, {"type": "url", "url": url, "title": title, "voice": voice}
            )
        except (StoreError, ValueError) as exc:
            raise handle_store_error(exc)
        accept = request.headers.get("Accept", "")
        if "text/html" in accept:
            message = f'<p class="success">✅ Queued: {html.escape(result["id"])}</p>'
            return HTMLResponse(_render_form(config, message))
        return PlainTextResponse(f"Queued: {result['id']}")

    @app.post("/add")
    async def add_post(request: Request):
        body = (await request.body()).decode("utf-8", "replace")
        params = parse_qs(body)
        job_type = params.get("type", ["url"])[0]
        url = params.get("url", [""])[0]
        title = params.get("title", [""])[0]
        voice = params.get("voice", ["default"])[0]
        text = params.get("text", [""])[0]

        if job_type == "text":
            if not text:
                return HTMLResponse(
                    _render_form(config, '<p class="error">❌ Error: Text is required</p>'),
                    status_code=400,
                )
            if not title:
                return HTMLResponse(
                    _render_form(
                        config, '<p class="error">❌ Error: Title is required for text jobs</p>'
                    ),
                    status_code=400,
                )
        elif not url:
            return HTMLResponse(
                _render_form(config, '<p class="error">❌ Error: URL is required</p>'),
                status_code=400,
            )

        payload = {
            "type": job_type,
            "url": url,
            "title": title,
            "voice": voice,
            "text": text,
        }
        try:
            result = _submit(config, store, payload)
            message = f'<p class="success">✅ Queued: {html.escape(result["id"])}</p>'
            return HTMLResponse(_render_form(config, message))
        except Exception as exc:  # form should always render an error
            message = f'<p class="error">❌ Error: {html.escape(str(exc))}</p>'
            return HTMLResponse(_render_form(config, message), status_code=500)

    @app.get("/bookmarklet", response_class=HTMLResponse)
    def bookmarklet(request: Request):
        host = request.headers.get("Host", "localhost:8081")
        scheme = "https" if ("443" in host or request.url.scheme == "https") else "http"
        content = _load_template("bookmarklet.html").replace(
            "{{REEDER_URL}}", f"{scheme}://{host}"
        )
        return HTMLResponse(content)

    @app.get("/status", response_class=PlainTextResponse)
    def legacy_status():
        if config.get("jobs", {}).get("mode", "push") == "push":
            inbox_count = len(list(paths["inbox"].glob("*.json")))
            processing_count = len(list(paths["processing"].glob("*.json")))
            done_count = len(list(paths["done"].glob("*.json")))
            failed_count = len(list(paths["done"].glob("FAILED-*.json")))
            lines = [f"Inbox: {inbox_count} job(s)"]
            if processing_count:
                lines.append(f"Processing: {processing_count} job(s)")
            status_file = paths["status"]
            if status_file.exists():
                status_lines = status_file.read_text().strip().split("\n")
                if len(status_lines) >= 2:
                    lines.extend([f"Current: {status_lines[1]}", f"Updated: {status_lines[0]}"])
            lines.append(f"Completed: {done_count - failed_count}")
            if failed_count:
                lines.append(f"Failed: {failed_count}")
            lines.append("OK")
            return PlainTextResponse("\n".join(lines))

        status = store.status()
        lines = [
            f"Queued: {status['queued']} job(s)",
            f"Running: {status['running']} job(s)",
            f"Completed: {status['completed']}",
            f"Failed: {status['failed']}",
        ]
        active = status.get("active_task")
        if active:
            lines.append(f"Active: {active['type']} {active['id']}")
        lines.append(f"Workers online: {len(status['online_workers'])}")
        lines.append("OK")
        return PlainTextResponse("\n".join(lines))

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(limit: int = 50):
        jobs = store.list_jobs(limit=limit)
        rows = "".join(
            "<tr>"
            f'<td><a href="/jobs/{html.escape(job["id"])}">{html.escape(job["id"])}</a></td>'
            f"<td>{html.escape(job['status'])}</td>"
            f"<td>{html.escape(job.get('title') or '')}</td>"
            f"<td>{html.escape(job['created_at'])}</td>"
            "</tr>"
            for job in jobs
        )
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>Reeder jobs</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}"
            "table{border-collapse:collapse;width:100%}td,th{padding:.4rem;"
            "border-bottom:1px solid #ddd;text-align:left}</style>"
            "<h1>Reeder jobs</h1><p><a href='/'>Submit a job</a> · "
            "<a href='/api/v1/status'>JSON status</a></p>"
            "<table><tr><th>Job</th><th>Status</th><th>Title</th><th>Created</th></tr>"
            f"{rows}</table>"
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(job_id: str):
        try:
            job = store.get_job(job_id)
        except NotFoundError as exc:
            raise handle_store_error(exc)
        task_rows = "".join(
            "<tr>"
            f"<td>{task['sequence']}. {html.escape(task['type'])}</td>"
            f"<td>{html.escape(task['status'])}</td>"
            f"<td>{html.escape(task['executor'])}</td>"
            f"<td>{task['attempt']}/{task['max_attempts']}</td>"
            f"<td>{html.escape(task.get('lease_owner') or '')}</td>"
            f"<td>{html.escape(task.get('error', {}).get('message', '') if task.get('error') else '')}</td>"
            "</tr>"
            for task in job["tasks"]
        )
        artifacts = "".join(
            f'<li>{html.escape(a["kind"])}: <a href="{html.escape(a["url"])}">'
            f"{html.escape(a['filename'])}</a> ({a['size']} bytes)</li>"
            for a in job["artifacts"]
        )
        events = "".join(
            f"<li>{html.escape(e['ts'])} [{html.escape(e['level'])}] "
            f"{html.escape(e['event'])}: {html.escape(e.get('message') or '')}</li>"
            for e in job["events"][-50:]
        )
        logs = store.job_logs(job_id, tail=200)
        body = (
            "<!doctype html><meta charset=utf-8>"
            f"<title>Reeder job {html.escape(job_id)}</title>"
            "<meta http-equiv='refresh' content='10'>"
            "<style>body{font-family:sans-serif;max-width:1200px;margin:2rem auto}"
            "table{border-collapse:collapse;width:100%}td,th{padding:.4rem;"
            "border-bottom:1px solid #ddd;text-align:left;vertical-align:top}"
            "pre{background:#111;color:#eee;padding:1rem;overflow:auto;max-height:34rem}</style>"
            f"<h1>{html.escape(job.get('title') or job_id)}</h1>"
            f"<p>Status: <strong>{html.escape(job['status'])}</strong> · "
            f"Created: {html.escape(job['created_at'])} · "
            f"<a href='/api/v1/jobs/{html.escape(job_id)}'>JSON</a> · "
            "<a href='/jobs'>All jobs</a></p>"
            f"<p>Source: {html.escape(job.get('source_url') or '(text job)')}</p>"
            "<h2>Tasks</h2><table><tr><th>Task</th><th>Status</th><th>Executor</th>"
            "<th>Attempt</th><th>Worker</th><th>Error</th></tr>"
            f"{task_rows}</table><h2>Artifacts</h2><ul>{artifacts or '<li>None</li>'}</ul>"
            f"<h2>Recent events</h2><ul>{events or '<li>None</li>'}</ul>"
            f"<h2>Log tail</h2><pre>{html.escape(logs or '(no logs yet)')}</pre>"
        )
        return HTMLResponse(body)

    # ------------------------------------------------------------------
    # Public JSON API
    # ------------------------------------------------------------------
    @app.post("/api/v1/jobs")
    async def create_job(request: Request):
        body = await request.json()
        try:
            job = await run_in_threadpool(store.create_job, body)
        except Exception as exc:
            raise handle_store_error(exc)
        return JSONResponse(
            {
                "id": job["id"],
                "status": job["status"],
                "detail_url": f"/jobs/{job['id']}",
            },
            status_code=201,
        )

    @app.get("/api/v1/jobs")
    def list_jobs(limit: int = 50, status: str | None = None):
        return {"jobs": store.list_jobs(limit=limit, status=status)}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return store.get_job(job_id)
        except Exception as exc:
            raise handle_store_error(exc)

    @app.get("/api/v1/jobs/{job_id}/events")
    def get_job_events(job_id: str, limit: int = 500):
        try:
            store.get_job(job_id)
        except Exception as exc:
            raise handle_store_error(exc)
        return {"events": store.get_events(job_id, limit=limit)}

    @app.get("/api/v1/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def get_job_logs(job_id: str, tail: int = 200):
        try:
            store.get_job(job_id)
        except Exception as exc:
            raise handle_store_error(exc)
        return PlainTextResponse(store.job_logs(job_id, tail=tail))

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return store.cancel_job(job_id)
        except Exception as exc:
            raise handle_store_error(exc)

    @app.get("/api/v1/status")
    def api_status():
        return store.status()

    @app.get("/api/v1/workers")
    def list_workers():
        return {"workers": store.list_workers()}

    # ------------------------------------------------------------------
    # Worker protocol
    # ------------------------------------------------------------------
    @app.post("/api/v1/workers/register")
    async def register_worker(request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        try:
            worker_id = str(body["worker_id"])
            name = str(body.get("name") or worker_id)
        except (KeyError, TypeError):
            raise HTTPException(400, "worker_id is required")
        return store.register_worker(
            worker_id=worker_id,
            name=name,
            capabilities=body.get("capabilities", []),
            inventory=body.get("inventory"),
            version=body.get("version"),
        )

    @app.post("/api/v1/workers/heartbeat")
    async def heartbeat_worker(request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        try:
            worker_id = str(body["worker_id"])
        except (KeyError, TypeError):
            raise HTTPException(400, "worker_id is required")
        try:
            return store.worker_heartbeat(
                worker_id=worker_id,
                state=str(body.get("state", "idle")),
                current_task_id=body.get("current_task_id"),
                status=body.get("status"),
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/claim")
    async def claim_task(request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        try:
            worker_id = str(body["worker_id"])
        except (KeyError, TypeError):
            raise HTTPException(400, "worker_id is required")
        name = str(body.get("name") or worker_id)
        capabilities = body.get("capabilities", [])
        wait_seconds = min(int(body.get("wait_seconds", claim_wait)), 60)
        requested_lease = int(body.get("lease_seconds", lease_seconds))
        if store.get_worker(worker_id) is None:
            store.register_worker(
                worker_id=worker_id,
                name=name,
                capabilities=capabilities,
                inventory=body.get("inventory"),
                version=body.get("version"),
            )
        task = await run_in_threadpool(
            store.wait_for_task,
            worker_id,
            capabilities=capabilities,
            lease_seconds=requested_lease,
            wait_seconds=wait_seconds,
        )
        if task is None:
            return Response(status_code=204)
        return {"task": task}

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str, request: Request):
        require_worker(request)
        try:
            return store.get_task(task_id)
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/{task_id}/heartbeat")
    async def task_heartbeat(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, _ = _task_lease(body)
        try:
            return store.heartbeat_task(
                task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_token=lease_token,
                state=body.get("state"),
                progress=body.get("progress"),
                message=body.get("message"),
                metrics=body.get("metrics"),
                lease_seconds=int(body.get("lease_seconds", lease_seconds)),
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/{task_id}/logs")
    async def task_logs(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, _ = _task_lease(body)
        entries = body.get("entries")
        if entries is None and "lines" in body:
            base = int(body.get("seq", 1))
            entries = [
                {"seq": base + index, "message": str(line)}
                for index, line in enumerate(body["lines"])
            ]
        if entries is None:
            raise HTTPException(400, "entries or lines are required")
        try:
            return store.append_logs(
                task_id,
                attempt=attempt,
                entries=entries,
                worker_id=worker_id,
                lease_token=lease_token,
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/{task_id}/artifacts")
    async def init_artifact(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, _ = _task_lease(body)
        try:
            artifact = store.init_artifact(
                task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_token=lease_token,
                filename=body.get("filename") or body.get("name") or "artifact.bin",
                size=int(body.get("size", 0)),
                sha256=body.get("sha256"),
                content_type=body.get("content_type"),
                kind=body.get("kind", "other"),
            )
            return {"artifact_id": artifact["id"], "offset": artifact["offset"]}
        except Exception as exc:
            raise handle_store_error(exc)

    @app.put("/api/v1/artifacts/{artifact_id}")
    async def upload_artifact(artifact_id: str, request: Request, offset: int = 0):
        require_worker(request)
        data = await request.body()
        try:
            return await run_in_threadpool(
                store.upload_artifact, artifact_id, offset=offset, data=data
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/artifacts/{artifact_id}/complete")
    def complete_artifact(artifact_id: str, request: Request):
        require_worker(request)
        try:
            return store.complete_artifact(artifact_id)
        except Exception as exc:
            raise handle_store_error(exc)

    @app.get("/api/v1/artifacts/{artifact_id}/content")
    def artifact_content(artifact_id: str):
        try:
            artifact = store.get_artifact(artifact_id)
            path = store.artifact_content_path(artifact_id)
        except Exception as exc:
            raise handle_store_error(exc)
        return FileResponse(
            path,
            media_type=artifact.get("content_type") or "application/octet-stream",
            filename=artifact["filename"],
        )

    @app.get("/api/v1/tasks/{task_id}/input")
    def task_input(task_id: str, request: Request):
        require_worker(request)
        try:
            artifact = store.task_input_artifact(task_id)
            path = store.artifact_content_path(artifact["id"])
        except Exception as exc:
            raise handle_store_error(exc)
        return FileResponse(
            path,
            media_type=artifact.get("content_type") or "application/octet-stream",
            filename=artifact["filename"],
        )

    @app.post("/api/v1/tasks/{task_id}/complete")
    async def complete_task(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, _ = _task_lease(body)
        try:
            return store.complete_task(
                task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_token=lease_token,
                result=body.get("result"),
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/{task_id}/fail")
    async def fail_task(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, _ = _task_lease(body)
        try:
            return store.fail_task(
                task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_token=lease_token,
                error=body.get("error") or body.get("message") or "Worker reported failure",
                retryable=bool(body.get("retryable", True)),
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.post("/api/v1/tasks/{task_id}/release")
    async def release_task(task_id: str, request: Request):
        require_worker(request)
        body = worker_body(await request.json())
        worker_id, attempt, lease_token, reason = _task_lease(body)
        try:
            return store.release_task(
                task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_token=lease_token,
                reason=reason or None,
            )
        except Exception as exc:
            raise handle_store_error(exc)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "time": utcnow(), "mode": jobs_config.get("mode", "push")}

    return app


# Uvicorn can import this if REEDER_CONFIG is set. The bin/reeder-web wrapper
# is preferred because it supports a --config argument.
def app_factory() -> FastAPI:
    return create_app()
