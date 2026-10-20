"""Host-side orchestrator for the pull-worker pipeline.

The orchestrator materializes typed tasks, executes host tasks, reaps expired
worker leases, and writes the legacy ``done/*.json`` + RSS feed projection.
Worker tasks are left in the queue for the agent to claim over HTTP.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from reeder.config import get_paths, load_config
from reeder.store import Store, store_from_config


ENGINE_CAPABILITIES = {
    "qwen3": ["tts", "tts:qwen3"],
    "qwen3-tts": ["tts", "tts:qwen3"],
    "kokoro": ["tts", "tts:kokoro"],
}


def _engine_for_job(job: dict, config: dict) -> tuple[str, list[str]]:
    request = job.get("request", {})
    profile = str(request.get("tts_profile") or "").lower()
    model = str(config.get("tts", {}).get("model", "qwen3")).lower()
    key = profile or ("kokoro" if "kokoro" in model else "qwen3")
    for name, caps in ENGINE_CAPABILITIES.items():
        if name in key:
            engine = "audio.cpp:kokoro" if name == "kokoro" else "audio.cpp:qwen3-tts"
            return engine, caps
    return "audio.cpp:qwen3-tts", ["tts", "tts:qwen3"]


class Orchestrator:
    def __init__(
        self,
        config: dict | None = None,
        *,
        store: Store | None = None,
        status: Callable[[str], None] | None = None,
    ):
        self.config = config or load_config()
        self.paths = get_paths(self.config)
        self.store = store or store_from_config(self.config)
        self.status = status or (lambda message: print(f"[orchestrator] {message}", flush=True))
        self.jobs_config = self.config.get("jobs", {})
        for key in ("inbox", "processing", "done", "audio", "www", "voices"):
            self.paths[key].mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # task materialization
    # ------------------------------------------------------------------
    def materialize_all(self) -> int:
        created = 0
        for job in self.store.list_jobs(limit=500):
            if job["status"] not in ("queued", "running"):
                continue
            try:
                if self.materialize_job(job["id"]):
                    created += 1
            except Exception as exc:
                self.status(f"materialize {job['id']} failed: {exc}")
                self._fail_job(job["id"], str(exc))
        return created

    def materialize_job(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        if job["status"] not in ("queued", "running"):
            return False
        tasks = job["tasks"]
        if any(task["status"] in ("queued", "claimed", "running", "uploading") for task in tasks):
            return False
        if any(task["status"] == "failed" for task in tasks):
            return False

        by_type = {task["type"]: task for task in tasks}
        if "tts" not in by_type:
            if job["pipeline"] == "url" and "fetch_extract" not in by_type:
                self.store.create_task(
                    job_id,
                    "fetch_extract",
                    executor="host",
                    payload={"url": job["request"].get("url")},
                    sequence=1,
                )
                return True
            fetch = by_type.get("fetch_extract")
            if job["pipeline"] == "url" and (
                fetch is None or fetch["status"] != "completed"
            ):
                return False
            self._prepare_tts(job, fetch)
            return True

        tts = by_type["tts"]
        if "finalize_audio" not in by_type and tts["status"] == "completed":
            self.store.create_task(
                job_id,
                "finalize_audio",
                executor="host",
                depends_on=[tts["id"]],
                payload={"tts_task_id": tts["id"]},
            )
            return True
        return False

    def _artifact_bytes(self, artifact_id: str) -> bytes:
        path = self.store.artifact_content_path(artifact_id)
        return path.read_bytes()

    def _prepare_tts(self, job: dict, fetch_task: dict | None) -> None:
        request = job["request"]
        if job["pipeline"] == "url":
            if not fetch_task or not fetch_task.get("result"):
                raise ValueError("fetch_extract result is missing")
            text_artifact_id = fetch_task["result"].get("text_artifact")
            if not text_artifact_id:
                raise ValueError("fetch_extract did not produce a text artifact")
            text = self._artifact_bytes(text_artifact_id).decode("utf-8")
        else:
            text = request.get("text", "")
        if not text.strip():
            raise ValueError("No text to synthesize")

        engine, required_caps = _engine_for_job(job, self.config)
        model = self.config.get("tts", {}).get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        max_tokens = int(self.config.get("tts", {}).get("max_tokens_per_chunk", 100))

        # Import lazily so hosts that only run web/orchestrator tests do not
        # need torch/qwen-tts installed.
        from reeder.tts import get_tts_tokenizer, split_text_into_chunks

        tokenizer = get_tts_tokenizer(model)
        pieces = split_text_into_chunks(text, tokenizer, max_tokens=max_tokens)
        chunks = [
            {
                "text": piece,
                "tokens": len(tokenizer(piece, return_tensors="pt").input_ids[0]),
            }
            for piece in pieces
        ]
        payload = {
            "kind": "tts",
            "engine": engine,
            "model": model,
            "voice": request.get("voice", "default"),
            "temperature": request.get(
                "temperature", self.config.get("tts", {}).get("temperature", 0.8)
            ),
            "language": request.get("language", "Auto"),
            "chunks": chunks,
            "output": {"kind": "audio", "format": "wav", "filename": "output.wav"},
            "input_artifact": None,
        }
        depends = [fetch_task["id"]] if fetch_task else []
        task = self.store.create_task(
            job["id"],
            "tts",
            executor="worker",
            required_caps=required_caps,
            depends_on=depends,
            payload=payload,
        )
        input_json = dict(payload)
        input_json["job_id"] = job["id"]
        input_json["title"] = request.get("title")
        artifact = self.store.create_artifact_bytes(
            task["id"],
            filename="input.json",
            data=json.dumps(input_json, indent=2).encode("utf-8"),
            kind="input",
            content_type="application/json",
        )
        payload["input_artifact"] = artifact["id"]
        self.store.update_task_payload(task["id"], payload)
        self.store.add_event(
            "tts_prepared",
            job_id=job["id"],
            task_id=task["id"],
            message=f"Prepared {len(chunks)} chunk(s) for {engine}",
            data={
                "chunks": len(chunks),
                "tokens": sum(chunk["tokens"] for chunk in chunks),
                "input_artifact": artifact["id"],
            },
        )
        self.status(f"prepared tts for {job['id']}: {len(chunks)} chunk(s)")

    # ------------------------------------------------------------------
    # host task execution
    # ------------------------------------------------------------------
    def run_host_tasks(self, max_tasks: int = 10) -> int:
        completed = 0
        for _ in range(max_tasks):
            task = self.store.claim_next_host_task()
            if task is None:
                break
            self.status(f"claimed host task {task['type']} {task['id']}")
            try:
                if task["type"] == "fetch_extract":
                    self._run_fetch_extract(task)
                    completed += 1
                elif task["type"] == "finalize_audio":
                    self._run_finalize_audio(task)
                    completed += 1
                else:
                    raise ValueError(f"Unknown host task type: {task['type']}")
            except Exception as exc:
                retryable = task["type"] != "fetch_extract"
                self.store.fail_task(
                    task["id"],
                    worker_id="host",
                    attempt=task["attempt"],
                    lease_token=task["lease_token"],
                    error=str(exc),
                    retryable=retryable,
                )
                self.status(f"host task {task['id']} failed: {exc}")
        return completed

    def _run_fetch_extract(self, task: dict) -> None:
        from reeder.extract import extract_text

        job = self.store.get_job(task["job_id"])
        text = extract_text(job["request"], self.config)
        artifact = self.store.create_artifact_bytes(
            task["id"],
            filename="text.txt",
            data=text.encode("utf-8"),
            kind="input",
            content_type="text/plain; charset=utf-8",
        )
        result = {
            "text_artifact": artifact["id"],
            "characters": len(text),
            "extractor": self.config.get("extractors", {}).get("default", "trafilatura"),
        }
        self.store.complete_task(
            task["id"],
            worker_id="host",
            attempt=task["attempt"],
            lease_token=task["lease_token"],
            result=result,
        )
        self.store.add_event(
            "text_extracted",
            job_id=task["job_id"],
            task_id=task["id"],
            message=f"Extracted {len(text)} characters",
            data=result,
        )

    def _run_finalize_audio(self, task: dict) -> None:
        from reeder.feed import update_feed
        from reeder.tts import convert_audio

        job = self.store.get_job(task["job_id"])
        tts_task_id = task["payload"].get("tts_task_id")
        tts_task = self.store.get_task(tts_task_id)
        result = tts_task.get("result") or {}
        output_artifact_id = result.get("output_artifact")
        if not output_artifact_id:
            raise ValueError("tts task did not produce output_artifact")

        audio_format = self.config.get("tts", {}).get("audio_format", "opus")
        audio_ext = "opus" if audio_format == "opus" else "mp3"
        audio_filename = f"{job['id']}.{audio_ext}"
        audio_output = self.paths["audio"] / audio_filename
        audio_output.parent.mkdir(parents=True, exist_ok=True)

        wav_source = self.store.artifact_content_path(output_artifact_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "output.wav"
            shutil.copyfile(wav_source, wav_path)
            temporary_output = audio_output.with_suffix(f".{audio_ext}.tmp")
            duration = convert_audio(wav_path, temporary_output, self.config)
            os.replace(temporary_output, audio_output)

        metrics = result.get("metrics") or {}
        completed = {
            "started": tts_task.get("started_at") or job["created_at"],
            "completed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": result.get("elapsed_seconds")
            or metrics.get("elapsed_seconds"),
            "audio_file": audio_filename,
            "audio_size": audio_output.stat().st_size,
            "duration_seconds": duration,
            "guid": job["id"],
            "tts_model": result.get("model")
            or self.config.get("tts", {}).get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            "engine": result.get("engine"),
            "worker_id": tts_task.get("lease_owner"),
            "metrics": metrics,
            "log_path": str(
                self.store.log_path(job["id"], tts_task["id"], max(1, tts_task["attempt"]))
            ),
        }
        projection = dict(job["request"])
        projection.setdefault("title", job.get("title") or job["id"])
        projection["_completed"] = completed
        done_path = self.paths["done"] / f"{job['id']}.json"
        done_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_done = done_path.with_suffix(".json.tmp")
        with open(tmp_done, "w", encoding="utf-8") as f:
            json.dump(projection, f, indent=2)
        os.replace(tmp_done, done_path)

        update_feed(self.config)
        self.store.complete_task(
            task["id"],
            worker_id="host",
            attempt=task["attempt"],
            lease_token=task["lease_token"],
            result={"audio_file": audio_filename, "duration_seconds": duration},
        )
        self.store.update_job_status(
            job["id"], "completed", completed=completed
        )
        self.store.add_event(
            "job_completed",
            job_id=job["id"],
            task_id=task["id"],
            message=f"Published {audio_filename}",
            data=completed,
        )
        self.status(f"completed {job['id']} -> {audio_filename}")

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def _fail_job(self, job_id: str, error: str) -> None:
        try:
            self.store.update_job_status(job_id, "failed", error={"message": error})
            self.store.add_event(
                "job_failed", job_id=job_id, level="error", message=error
            )
        except Exception:
            pass

    def tick(self) -> dict:
        reaped = self.store.reap_expired()
        if reaped:
            self.status(f"reaped {len(reaped)} expired task(s)")
        created = self.materialize_all()
        completed = self.run_host_tasks()
        return {"reaped": len(reaped), "materialized": created, "host_tasks": completed}

    def run_forever(self, interval: float = 2.0) -> None:
        self.status("orchestrator started")
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.status("stopping")
                return
            except Exception as exc:
                self.status(f"tick failed: {exc}")
            time.sleep(max(0.5, interval))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Reeder host orchestrator")
    parser.add_argument("--config", help="Path to config.toml")
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.config:
        os.environ["REEDER_CONFIG"] = str(Path(args.config).resolve())
    config = load_config()
    orchestrator = Orchestrator(config)
    if args.once:
        print(json.dumps(orchestrator.tick(), indent=2))
    else:
        orchestrator.run_forever(interval=args.interval)


if __name__ == "__main__":
    main()
