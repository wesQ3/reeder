"""CLI entry points for reeder commands."""

import os
from pathlib import Path

from reeder.config import load_config
from reeder.feed import update_feed as _update_feed


def process_main():
    """Entry point for reeder-process."""
    # Import here to avoid heavy deps on feed-only invocations
    import json
    import shutil
    import tempfile
    import time
    from datetime import datetime, timezone

    from reeder.config import get_paths
    from reeder.extract import extract_text
    from reeder.tts import convert_audio, generate_audio

    config_path = os.environ.get("REEDER_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    paths = get_paths(config)

    # Ensure directories exist
    for name, path in paths.items():
        if name != "status":
            path.mkdir(parents=True, exist_ok=True)

    def write_status(message: str):
        status_path = paths["status"]
        status_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(status_path, "w") as f:
            f.write(f"{timestamp}\n{message}\n")
        print(f"[STATUS] {message}", flush=True)

    def get_oldest_job():
        jobs = sorted(paths["inbox"].glob("*.json"))
        return jobs[0] if jobs else None

    processed = 0
    while True:
        job_path = get_oldest_job()
        if not job_path:
            break

        job_name = job_path.stem
        start_time = datetime.now(timezone.utc)
        start_mono = time.monotonic()

        print(f"\n{'='*60}")
        print(f"Processing job: {job_name}")
        print(f"{'='*60}")
        write_status(f"Processing: {job_name}")

        with open(job_path) as f:
            job = json.load(f)

        processing_path = paths["processing"] / job_path.name
        shutil.move(job_path, processing_path)

        try:
            write_status(f"Extracting text: {job_name}")
            text = extract_text(job, config)
            print(f"  Extracted {len(text)} characters")

            audio_format = config["tts"].get("audio_format", "opus")
            audio_ext = "opus" if audio_format == "opus" else "mp3"
            audio_filename = f"{job_name}.{audio_ext}"
            audio_output = paths["audio"] / audio_filename

            write_status(f"Generating audio: {job_name}")
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "output.wav"
                generate_audio(
                    text, wav_path, job, config, paths,
                    status_callback=write_status,
                )
                write_status(f"Converting audio: {job_name}")
                duration = convert_audio(wav_path, audio_output, config)

            end_time = datetime.now(timezone.utc)
            elapsed_seconds = round(time.monotonic() - start_mono, 2)

            job["_completed"] = {
                "started": start_time.isoformat(),
                "completed": end_time.isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "audio_file": audio_filename,
                "audio_size": audio_output.stat().st_size,
                "duration_seconds": duration,
                "guid": job_name,
                "tts_model": config["tts"].get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            }

            print(f"\n{'='*60}")
            print(f"Job completed: {job_name}")
            print(f"  Elapsed: {elapsed_seconds}s")
            print(f"  Audio duration: {duration}s")
            if duration > 0:
                print(f"  RTF: {elapsed_seconds / duration:.2f}x")
            print(f"{'='*60}")

            if "title" not in job:
                job["title"] = job.get("url", "Untitled") if job["type"] == "url" else "Untitled"

            done_path = paths["done"] / job_path.name
            with open(done_path, "w") as f:
                json.dump(job, f, indent=2)
            processing_path.unlink()

            write_status(f"Updating feed: {job_name}")
            _update_feed(config)
            write_status(f"Completed: {job_name}")

        except Exception as e:
            job["_error"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": str(e),
            }
            failed_path = paths["done"] / f"FAILED-{job_path.name}"
            with open(failed_path, "w") as f:
                json.dump(job, f, indent=2)
            processing_path.unlink()
            write_status(f"FAILED: {job_name} - {e}")
            print(f"Job failed: {e}", flush=True)

        processed += 1

    if processed == 0:
        write_status("Idle - no jobs in queue")
    else:
        write_status(f"Idle - processed {processed} job(s)")


def feed_main():
    """Entry point for reeder-feed."""
    config_path = os.environ.get("REEDER_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    _update_feed(config)
