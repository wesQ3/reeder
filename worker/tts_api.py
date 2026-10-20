"""Reeder TTS Worker — audio.cpp Compatibility Gateway.

Provides a FastAPI server that translates Reeder's TTS worker protocol
to audio.cpp's OpenAI-compatible speech API (/v1/audio/speech).
Enables running multiple GGUF-based TTS models (Qwen3-TTS, PocketTTS,
Kokoro, Supertonic, etc.) with high performance and zero changes to the
main Reeder service.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import statistics
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("reeder-audiocpp-gateway")

app = FastAPI(
    title="Reeder audio.cpp Compatibility Gateway",
    description="Translates Reeder TTS requests to native audio.cpp C++ inference",
    version="0.2.0",
)

# Global Configuration & State
AUDIOCPP_URL: str = os.getenv("AUDIOCPP_URL", "http://127.0.0.1:8080").rstrip("/")
ACTIVE_TTS_MODEL: str = os.getenv("ACTIVE_TTS_MODEL", "qwen3-tts")
AUDIOCPP_BACKEND: str = os.getenv("AUDIOCPP_BACKEND", "audio.cpp (cuda)")
DEVICE: str = os.getenv("TTS_DEVICE", "cuda:0")
VOICES_DIR: Path = Path(os.getenv("VOICES_DIR", "/data/voices"))
SERVER_CONFIG_PATH: Path = Path(os.getenv("SERVER_CONFIG_PATH", "/app/server.json"))

# Chunking & runaway-generation detection settings.
# audio.cpp's qwen3_tts speech decoder allocates its CUDA graph sized to the
# input, so long texts must arrive as small pre-split chunks (the job
# processor splits with the real Qwen tokenizer before dispatching).
# This worker synthesizes chunks sequentially and retries chunks whose
# audio length is a statistical outlier.
OUTLIER_Z_LIMIT = float(os.getenv("OUTLIER_Z_LIMIT", "3.0"))
MAX_ATTEMPTS_PER_CHUNK = int(os.getenv("MAX_ATTEMPTS_PER_CHUNK", "3"))
CHUNK_HTTP_TIMEOUT = float(os.getenv("CHUNK_HTTP_TIMEOUT", "300.0"))
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "4.0"))
# Preseeded samples-per-token history from past runs (see reeder.tts).
# Recalibrate via SAMPLES_PER_TOKEN_SEED="v1,v2,..." if a voice/model yields
# a different scale.
_DEFAULT_SPT_SEED = "5110,4793,5234,5889,5130,5941,5877,5138,5607,6370,6260,6381,5894,5538,6027,5280"
_seed_env = os.getenv("SAMPLES_PER_TOKEN_SEED", "").strip() or _DEFAULT_SPT_SEED
SAMPLES_PER_TOKEN_SEED: list[float] = [float(v) for v in _seed_env.split(",") if v.strip()]
if len(SAMPLES_PER_TOKEN_SEED) < 2:
    # statistics.stdev needs at least two points for outlier detection to work
    SAMPLES_PER_TOKEN_SEED = [float(v) for v in _DEFAULT_SPT_SEED.split(",")]


def estimate_tokens(text: str) -> int:
    """Fallback token estimate for chunks dispatched without a token count.

    The job processor measures real token counts with the Qwen tokenizer;
    this heuristic (~4 chars/token) only covers ad-hoc callers that send
    raw `text` (e.g. curl demos) and is used solely for the s/t statistics.
    """
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def parse_wav_frames(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Parse a WAV payload into (pcm_frames, sample_rate, sampwidth, channels)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return (
            wf.readframes(wf.getnframes()),
            wf.getframerate(),
            wf.getsampwidth(),
            wf.getnchannels(),
        )


def load_configured_models() -> list[str]:
    """Read configured model IDs from server.json if available."""
    default_models = [ACTIVE_TTS_MODEL]
    config_paths = [
        SERVER_CONFIG_PATH,
        Path(__file__).parent / "server.json",
        Path("server.json"),
    ]
    for p in config_paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                models = [m.get("id") for m in data.get("models", []) if m.get("id")]
                if models:
                    return models
            except Exception as e:
                logger.warning(f"Could not read models from {p}: {e}")
    return default_models


class ChunkSpec(BaseModel):
    """A single pre-split chunk of text with its measured token count."""
    text: str = Field(..., min_length=1, description="Chunk text")
    tokens: Optional[int] = Field(
        default=None, ge=1, description="Token count measured by the caller's Qwen tokenizer"
    )


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint matching Reeder contract.

    Send either `chunks` (pre-split with the real tokenizer on the job
    processor — required for article-length text, since audio.cpp's qwen3_tts
    decoder graph scales with input size) or a single `text` (convenience for
    demo-sized requests).
    """
    text: Optional[str] = Field(default=None, description="Single text to synthesize (short texts only)")
    chunks: Optional[list[ChunkSpec]] = Field(default=None, description="Pre-split chunks")
    voice: str = Field(default="default", description="Voice name (maps to voices dir or preset)")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    language: str = Field(default="Auto")


class HealthResponse(BaseModel):
    """Health check response maintaining backward compatibility."""
    status: str
    model: str
    backend: str = "audio.cpp (cuda)"
    device: str = "cuda:0"
    active_model: str
    available_models: list[str] = Field(default_factory=list)
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verifies audio.cpp backend is online and models are loaded/available."""
    available_models = load_configured_models()
    audiocpp_online = False
    models_from_api: list[str] = []

    # Check audio.cpp health endpoint
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{AUDIOCPP_URL}/health")
            if resp.status_code == 200:
                audiocpp_online = True
        except Exception as e:
            logger.debug(f"audio.cpp /health not reachable: {e}")

        # Try to query /v1/models if available
        try:
            resp_models = await client.get(f"{AUDIOCPP_URL}/v1/models")
            if resp_models.status_code == 200:
                audiocpp_online = True
                data = resp_models.json()
                for item in data.get("data", []):
                    m_id = item.get("id")
                    if m_id and m_id not in models_from_api:
                        models_from_api.append(m_id)
        except Exception as e:
            logger.debug(f"audio.cpp /v1/models query failed: {e}")

    if models_from_api:
        available_models = models_from_api

    if not audiocpp_online:
        logger.warning(f"audio.cpp server unreachable at {AUDIOCPP_URL}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": f"audio.cpp server unreachable at {AUDIOCPP_URL}",
                "model": ACTIVE_TTS_MODEL,
                "backend": AUDIOCPP_BACKEND,
                "device": DEVICE,
                "active_model": ACTIVE_TTS_MODEL,
                "available_models": available_models,
            },
        )

    return HealthResponse(
        status="ready",
        model=ACTIVE_TTS_MODEL,
        backend=AUDIOCPP_BACKEND,
        device=DEVICE,
        active_model=ACTIVE_TTS_MODEL,
        available_models=available_models,
    )


def resolve_voice_and_model(
    voice: str,
    requested_language: str = "Auto",
) -> tuple[str, dict[str, Any]]:
    """Resolve requested voice name to model ID and audio.cpp payload parameters.

    Supports:
    1. Voice alias metadata JSON (/data/voices/<voice>.json)
    2. Voice cloning WAV + transcript (/data/voices/<voice>.wav + .txt)
    3. Preset speaker voices (e.g. Kokoro preset names)
    4. Default fallback resolution
    """
    clean_voice = Path(voice).stem if not Path(voice).is_absolute() else Path(voice).stem
    target_model = ACTIVE_TTS_MODEL
    extra_params: dict[str, Any] = {}

    # Check for JSON metadata alias first (e.g. kokoro-bella.json)
    json_path = VOICES_DIR / f"{clean_voice}.json"
    if json_path.is_file():
        try:
            alias_data = json.loads(json_path.read_text(encoding="utf-8"))
            if "model" in alias_data:
                target_model = alias_data["model"]

            # Speaker preset ID
            voice_id = alias_data.get("voice_id") or alias_data.get("voice") or alias_data.get("speaker")
            if voice_id:
                extra_params["voice"] = voice_id

            # Voice ref audio for cloning models
            voice_ref = alias_data.get("voice_ref") or alias_data.get("ref_audio")
            if voice_ref:
                extra_params["voice_ref"] = str(voice_ref)

            # Reference transcript
            ref_text = alias_data.get("reference_text") or alias_data.get("ref_text")
            if ref_text:
                extra_params["reference_text"] = ref_text

            if "language" in alias_data:
                extra_params["language"] = alias_data["language"]

            logger.info(f"Resolved voice alias '{voice}' from {json_path} -> model={target_model}")
            return target_model, extra_params
        except Exception as e:
            logger.warning(f"Error reading voice alias {json_path}: {e}")

    # Check for direct WAV voice sample for cloning
    wav_path = VOICES_DIR / f"{clean_voice}.wav"
    if not wav_path.is_file() and clean_voice == "default":
        # Look for default.wav or any .wav in voices directory
        wav_files = sorted(VOICES_DIR.glob("*.wav"))
        if wav_files:
            wav_path = wav_files[0]
            clean_voice = wav_path.stem
            logger.info(f"Default voice using first available sample: {wav_path.name}")

    if wav_path.is_file():
        extra_params["voice_ref"] = str(wav_path)
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.is_file():
            extra_params["reference_text"] = txt_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"Voice transcript not found at {txt_path}; proceed without reference_text")
        return target_model, extra_params

    # If voice looks like a preset identifier (e.g. af_bella, am_adam) or model doesn't require audio cloning
    if "kokoro" in target_model.lower() or "supertonic" in target_model.lower():
        # Pass voice name as preset speaker ID
        extra_params["voice"] = clean_voice if clean_voice != "default" else "af_bella"
        return target_model, extra_params

    # If neither wav nor json was found and target model is cloning-based
    if clean_voice != "default":
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice}' not found in {VOICES_DIR} (expected {wav_path.name} or {json_path.name})",
        )

    # Fallback for default when no wav files exist
    logger.warning(f"No voice files found in {VOICES_DIR} for '{voice}', delegating to model default")
    return target_model, extra_params


class GenerationCancelled(Exception):
    """Raised when an operator/agent cancels a synthesis run."""


async def synthesize_chunks(
    chunk_specs: list[tuple[str, Optional[int]]],
    *,
    voice: str,
    temperature: float,
    language: str,
    client_presplit: bool,
    progress_cb=None,
    log_cb=None,
    cancel_check=None,
) -> dict[str, Any]:
    """Synthesize pre-split chunks into one WAV.

    This is the reusable core behind both the synchronous ``/generate``
    compatibility endpoint and the asynchronous local runtime API. Callbacks
    let the runtime report per-chunk progress and structured logs.
    """
    start_time = time.monotonic()

    def log(level: str, message: str, data: Any = None) -> None:
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
        if log_cb:
            log_cb(level, message, data)

    def check_cancel() -> None:
        if cancel_check and cancel_check():
            raise GenerationCancelled("Generation canceled")

    target_model, voice_params = resolve_voice_and_model(voice, language)

    base_payload: dict[str, Any] = {
        "model": target_model,
        "response_format": "wav",
        "temperature": temperature,
    }
    base_payload.update(voice_params)
    if language and language.lower() != "auto" and "language" not in base_payload:
        base_payload["language"] = language

    total_chars = sum(len(t) for t, _ in chunk_specs)
    total_chunks = len(chunk_specs)
    source = "client pre-split" if client_presplit else "single text"
    log(
        "info",
        f"Generating: {total_chars} chars in {total_chunks} chunk(s) ({source}), "
        f"voice={voice}, model={target_model}, temp={temperature}",
        {"chunks_total": total_chunks, "chars": total_chars, "model": target_model},
    )

    samples_per_token_history: list[float] = list(SAMPLES_PER_TOKEN_SEED)
    std_dev_limit = OUTLIER_Z_LIMIT
    max_attempts = MAX_ATTEMPTS_PER_CHUNK

    sample_rate: Optional[int] = None
    sampwidth: Optional[int] = None
    nchannels: Optional[int] = None
    total_frames = 0
    retried_chunks = 0
    chunks_done = 0

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as pcm_tmp:
        pcm_path = Path(pcm_tmp.name)

    try:
        async with httpx.AsyncClient(timeout=CHUNK_HTTP_TIMEOUT) as client:
            for i, (chunk, measured_tokens) in enumerate(chunk_specs):
                check_cancel()
                chunk_tokens = measured_tokens if measured_tokens else estimate_tokens(chunk)
                chunk_payload = dict(base_payload, input=chunk)

                attempt, is_outlier = 1, True
                while is_outlier and attempt <= max_attempts:
                    check_cancel()
                    try:
                        resp = await client.post(f"{AUDIOCPP_URL}/v1/audio/speech", json=chunk_payload)
                    except httpx.RequestError as exc:
                        log(
                            "error",
                            f"audio.cpp request failed on chunk {i + 1}/{total_chunks}: {exc}",
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=f"Failed to communicate with audio.cpp backend: {exc}",
                        ) from exc

                    if resp.status_code != 200:
                        log(
                            "error",
                            f"audio.cpp returned error {resp.status_code}: {resp.text}",
                        )
                        raise HTTPException(
                            status_code=resp.status_code,
                            detail=f"audio.cpp generation error: {resp.text}",
                        )

                    try:
                        frames, sr, sw, ch = parse_wav_frames(resp.content)
                    except Exception as exc:
                        log(
                            "error",
                            f"Could not parse WAV from audio.cpp response (chunk {i + 1}/{total_chunks}): {exc}",
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=f"audio.cpp returned unparsable audio for chunk {i + 1}/{total_chunks}",
                        ) from exc
                    if not frames:
                        raise HTTPException(
                            status_code=502,
                            detail=f"audio.cpp returned empty audio for chunk {i + 1}/{total_chunks}",
                        )

                    if sample_rate is None:
                        sample_rate, sampwidth, nchannels = sr, sw, ch
                    elif (sr, sw, ch) != (sample_rate, sampwidth, nchannels):
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "audio.cpp returned inconsistent audio format across chunks "
                                f"({sr}Hz/{sw * 8}bit/{ch}ch vs {sample_rate}Hz/{sampwidth * 8}bit/{nchannels}ch)"
                            ),
                        )

                    n_samples = len(frames) // (sw * ch)
                    samples_per_token = n_samples / chunk_tokens if chunk_tokens > 0 else 0.0
                    running_avg = statistics.mean(samples_per_token_history)
                    std_dev = statistics.stdev(samples_per_token_history)
                    z_score = abs(samples_per_token - running_avg) / std_dev if std_dev > 0 else 0.0
                    is_outlier = z_score > std_dev_limit

                    if attempt > 1:
                        retried_chunks += 1
                    flag = " !!! RETRY" if is_outlier else ""
                    log(
                        "info",
                        f"  Chunk {i + 1:>3}/{total_chunks:<3} | {len(chunk):>3}c | {chunk_tokens:>3}t | "
                        f"{samples_per_token:>9,.2f} s/t | {n_samples / sr:>5.2f}s | "
                        f"{running_avg:>9,.2f}avg | {std_dev:>6,.2f}\u03c3 | {z_score:>4.2f}z{flag}",
                        {
                            "chunk": i + 1,
                            "chunks_total": total_chunks,
                            "attempt": attempt,
                            "retry": is_outlier,
                            "z_score": z_score,
                        },
                    )
                    attempt += 1

                samples_per_token_history.append(samples_per_token)
                total_frames += n_samples
                chunks_done += 1
                with open(pcm_path, "ab") as pcm_file:
                    pcm_file.write(frames)
                if progress_cb:
                    progress_cb(
                        chunks_done,
                        total_chunks,
                        {
                            "chunks_done": chunks_done,
                            "chunks_total": total_chunks,
                            "elapsed_seconds": round(time.monotonic() - start_time, 2),
                        },
                    )

        if sample_rate is None or sampwidth is None or nchannels is None or total_frames == 0:
            raise HTTPException(status_code=500, detail="No audio generated")

        wav_buf = io.BytesIO()
        with open(pcm_path, "rb") as pcm_file, wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            while True:
                block = pcm_file.read(1024 * 1024)
                if not block:
                    break
                wf.writeframesraw(block)
        wav_bytes = wav_buf.getvalue()
    finally:
        if pcm_path.exists():
            pcm_path.unlink()

    duration = total_frames / (sample_rate * nchannels) if sample_rate else 0.0
    generation_time = time.monotonic() - start_time
    rtf = generation_time / duration if duration > 0 else 0.0
    log(
        "info",
        f"Done: {duration:.2f}s audio in {generation_time:.2f}s (RTF={rtf:.3f}x, "
        f"{total_chunks} chunks, {retried_chunks} retried)",
    )
    return {
        "wav": wav_bytes,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "chunks_generated": total_chunks,
        "retried_chunks": retried_chunks,
        "generation_time": generation_time,
        "rtf": rtf,
        "model": target_model,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Synchronous compatibility endpoint returning WAV bytes."""
    if request.text is not None and request.chunks is not None:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'chunks', not both")

    if request.chunks is not None:
        chunk_specs = [(c.text.strip(), c.tokens) for c in request.chunks]
        chunk_specs = [(t, n) for t, n in chunk_specs if t]
        if not chunk_specs:
            raise HTTPException(status_code=400, detail="'chunks' must contain at least one non-empty chunk")
        client_presplit = True
    elif request.text is not None:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        chunk_specs = [(text, None)]
        client_presplit = False
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'chunks'")

    result = await synthesize_chunks(
        chunk_specs,
        voice=request.voice,
        temperature=request.temperature,
        language=request.language,
        client_presplit=client_presplit,
    )
    return Response(
        content=result["wav"],
        media_type="audio/wav",
        headers={
            "X-Duration-Seconds": f"{result['duration_seconds']:.2f}",
            "X-Sample-Rate": str(result["sample_rate"]),
            "X-Chunks-Generated": str(result["chunks_generated"]),
            "X-Retried-Chunks": str(result["retried_chunks"]),
            "X-Generation-Time": f"{result['generation_time']:.2f}",
            "X-RTF": f"{result['rtf']:.3f}",
        },
    )


# ----------------------------------------------------------------------
# Local runtime API (used by reeder-worker-agent inside the container)
# ----------------------------------------------------------------------
WORK_DIR = Path(os.getenv("WORK_DIR", "/work"))
RUNTIME_TASKS: dict[str, dict[str, Any]] = {}
_RUNTIME_SEQ = 0


class RuntimeStartRequest(BaseModel):
    kind: str = Field(default="tts")
    payload: dict[str, Any] = Field(default_factory=dict)
    output_path: Optional[str] = Field(
        default=None, description="Path relative to WORK_DIR; defaults to runtime/<id>/output.wav"
    )


def _runtime_task_public(task: dict) -> dict:
    return {key: value for key, value in task.items() if key != "logs"}


def _runtime_log(task: dict, level: str, message: str, data: Any = None) -> None:
    global _RUNTIME_SEQ
    _RUNTIME_SEQ += 1
    task["logs"].append(
        {
            "seq": _RUNTIME_SEQ,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "message": message,
            **({"data": data} if data is not None else {}),
        }
    )


def _resolve_runtime_output(task_id: str, output_path: Optional[str]) -> tuple[Path, str]:
    relative = output_path or f"runtime/{task_id}/output.wav"
    relative_path = Path(relative)
    if relative_path.is_absolute():
        try:
            relative_path = relative_path.relative_to(WORK_DIR)
        except ValueError:
            raise HTTPException(400, "output_path must live under WORK_DIR")
    root = WORK_DIR.resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents:
        raise HTTPException(400, "output_path escapes WORK_DIR")
    return path, str(relative_path)


async def _run_runtime_task(task_id: str) -> None:
    task = RUNTIME_TASKS[task_id]
    task["state"] = "running"
    task["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _runtime_log(task, "info", f"Runtime task {task_id} started")
    start = time.monotonic()
    try:
        payload = task["payload"]
        if payload.get("chunks") is not None:
            chunk_specs = [
                (str(chunk.get("text", "")).strip(), chunk.get("tokens"))
                for chunk in payload["chunks"]
            ]
            chunk_specs = [(text, tokens) for text, tokens in chunk_specs if text]
            presplit = True
        else:
            text = str(payload.get("text", "")).strip()
            chunk_specs = [(text, None)] if text else []
            presplit = False
        if not chunk_specs:
            raise ValueError("Runtime tts payload must include chunks or text")

        def progress(current, total, metrics):
            task["progress"] = {
                "phase": "tts",
                "current": current,
                "total": total,
                "unit": "chunk",
                "percent": round((current / total) * 100, 2) if total else 0.0,
            }
            task["metrics"] = metrics

        def log_cb(level, message, data=None):
            _runtime_log(task, level, message, data)

        result = await synthesize_chunks(
            chunk_specs,
            voice=str(payload.get("voice", "default")),
            temperature=float(payload.get("temperature", 0.8)),
            language=str(payload.get("language", "Auto")),
            client_presplit=presplit,
            progress_cb=progress,
            log_cb=log_cb,
            cancel_check=lambda: bool(task.get("cancel_requested")),
        )

        output_path, relative = _resolve_runtime_output(task_id, task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(result["wav"])
        metrics = {
            "duration_seconds": round(result["duration_seconds"], 3),
            "sample_rate": result["sample_rate"],
            "chunks_generated": result["chunks_generated"],
            "retried_chunks": result["retried_chunks"],
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "rtf": round(result["rtf"], 4),
            "model": result["model"],
            "engine": payload.get("engine"),
        }
        metrics_path = output_path.with_name("metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        task["result"] = {
            "output_path": relative,
            "metrics_path": str(metrics_path.relative_to(WORK_DIR.resolve())),
            "size": output_path.stat().st_size,
            "metrics": metrics,
        }
        task["state"] = "completed"
        task["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _runtime_log(task, "info", "Runtime task completed", task["result"])
    except GenerationCancelled:
        task["state"] = "canceled"
        task["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _runtime_log(task, "warning", "Runtime task canceled")
    except HTTPException as exc:
        task["state"] = "failed"
        task["error"] = str(exc.detail)
        task["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _runtime_log(task, "error", f"Runtime task failed: {exc.detail}")
    except Exception as exc:
        task["state"] = "failed"
        task["error"] = str(exc)
        task["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _runtime_log(task, "error", f"Runtime task failed: {exc}")


@app.post("/runtime/tasks", status_code=202)
async def start_runtime_task(request: RuntimeStartRequest):
    if request.kind != "tts":
        raise HTTPException(400, f"Unsupported runtime task kind: {request.kind}")
    task_id = f"rt_{int(time.time() * 1000)}_{len(RUNTIME_TASKS) + 1}"
    task = {
        "id": task_id,
        "kind": request.kind,
        "state": "queued",
        "payload": request.payload,
        "output_path": request.output_path,
        "progress": {"phase": "queued", "current": 0, "total": 0, "percent": 0.0},
        "metrics": {},
        "error": None,
        "result": None,
        "logs": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RUNTIME_TASKS[task_id] = task
    _runtime_log(task, "info", "Runtime task accepted")
    asyncio.create_task(_run_runtime_task(task_id))
    return {"id": task_id, "state": task["state"]}


@app.get("/runtime/tasks/{task_id}")
async def get_runtime_task(task_id: str):
    task = RUNTIME_TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Runtime task not found")
    return _runtime_task_public(task)


@app.get("/runtime/tasks/{task_id}/logs")
async def get_runtime_logs(task_id: str, since: int = 0):
    task = RUNTIME_TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Runtime task not found")
    entries = [entry for entry in task["logs"] if entry["seq"] > int(since)]
    return {"entries": entries, "last_seq": entries[-1]["seq"] if entries else int(since)}


@app.post("/runtime/tasks/{task_id}/cancel")
async def cancel_runtime_task(task_id: str):
    task = RUNTIME_TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "Runtime task not found")
    task["cancel_requested"] = True
    if task["state"] in ("queued", "running"):
        task["state"] = "canceling"
    _runtime_log(task, "warning", "Cancellation requested")
    return _runtime_task_public(task)


@app.get("/runtime/tasks/{task_id}/artifact")
async def get_runtime_artifact(task_id: str):
    task = RUNTIME_TASKS.get(task_id)
    if task is None or not task.get("result"):
        raise HTTPException(404, "Runtime artifact not available")
    path = (WORK_DIR / task["result"]["output_path"]).resolve()
    if not path.exists():
        raise HTTPException(404, "Runtime artifact file missing")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def main():
    """CLI entry point for the worker gateway server."""
    global AUDIOCPP_URL, ACTIVE_TTS_MODEL, DEVICE, VOICES_DIR

    parser = argparse.ArgumentParser(description="Reeder audio.cpp Compatibility Gateway")
    parser.add_argument("--audiocpp-url", default=AUDIOCPP_URL, help="URL of audio.cpp server")
    parser.add_argument("--model", default=ACTIVE_TTS_MODEL, help="Default active model ID")
    parser.add_argument("--device", default=DEVICE, help="Target device (e.g. cuda:0)")
    parser.add_argument("--voices-dir", default=str(VOICES_DIR), help="Path to voices directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port to listen on")
    args = parser.parse_args()

    AUDIOCPP_URL = args.audiocpp_url.rstrip("/")
    ACTIVE_TTS_MODEL = args.model
    DEVICE = args.device
    VOICES_DIR = Path(args.voices_dir)

    logger.info(
        f"Starting gateway on {args.host}:{args.port} -> audio.cpp at {AUDIOCPP_URL} "
        f"(model: {ACTIVE_TTS_MODEL}, voices: {VOICES_DIR})"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
