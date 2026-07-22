"""Reeder TTS Worker — GPU-accelerated Qwen3-TTS API.

Provides a FastAPI server that generates voice-cloned audio using CUDA.
Designed to run on a GPU-equipped machine, called by the main reeder
service on nuc0.
"""

import logging
import tempfile
import time
import wave
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from reeder.tts import generate_audio, get_tts_model, split_text_into_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

app = FastAPI(
    title="Reeder TTS Worker",
    description="GPU-accelerated TTS generation via Qwen3-TTS",
    version="0.1.0",
)

# Global model state
_model_loaded: bool = False
_model_name: str = ""
_device: str = "cuda"


def load_model(model_name: str, device: str = "cuda"):
    """Load the Qwen3-TTS model."""
    global _model_loaded, _model_name, _device
    _device = device
    _model_name = model_name

    logger.info(f"Loading model: {model_name} on {device}")
    get_tts_model(model_name, device)
    _model_loaded = True
    logger.info("Model loaded successfully")


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice name (maps to voices dir)")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    language: str = Field(default="Auto")
    max_tokens_per_chunk: int = Field(default=100, ge=10, le=500)


class GenerateResponse(BaseModel):
    """Response metadata for /generate endpoint."""
    duration_seconds: float
    sample_rate: int
    chunks_generated: int
    generation_time_seconds: float
    rtf: float


class HealthResponse(BaseModel):
    model: str
    device: str
    status: str
    gpu_memory_used_mb: int | None = None
    gpu_memory_total_mb: int | None = None


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms model is loaded and GPU is available."""
    gpu_used = None
    gpu_total = None
    if torch.cuda.is_available():
        gpu_used = int(torch.cuda.memory_allocated() / 1024 / 1024)
        gpu_total = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)

    return HealthResponse(
        model=_model_name,
        device=_device,
        status="ready" if _model_loaded else "not_loaded",
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate voice-cloned audio from text.

    Returns the audio as a WAV file response with metadata headers.
    """
    from fastapi.responses import Response

    tts = get_tts_model(_model_name, _device)
    start_time = time.monotonic()
    tokenizer = tts.processor.tokenizer
    chunks = split_text_into_chunks(request.text, tokenizer, max_tokens=request.max_tokens_per_chunk)
    total_chunks = len(chunks)
    logger.info(
        f"Generating: {len(request.text)} chars, voice={request.voice}, "
        f"temp={request.temperature}, chunks={total_chunks}"
    )

    # Reuse shared inference path from reeder.tts to keep behavior consistent.
    config = {
        "tts": {
            "default_voice": "default.safetensors",
            "model": _model_name,
            "device": _device,
            "temperature": request.temperature,
            "max_tokens_per_chunk": request.max_tokens_per_chunk,
        }
    }
    paths = {"voices": Path("/data/voices")}
    job = {
        "voice": request.voice,
        "temperature": request.temperature,
        "language": request.language,
        "title": "remote-request",
    }

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            generate_audio(request.text, tmp_path, job, config, paths)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with wave.open(str(tmp_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        duration = frame_count / sample_rate if sample_rate > 0 else 0.0
        generation_time = time.monotonic() - start_time
        rtf = generation_time / duration if duration > 0 else 0.0

        logger.info(f"  Done: {duration:.1f}s audio in {generation_time:.1f}s (RTF={rtf:.2f}x)")
        wav_bytes = tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Duration-Seconds": f"{duration:.2f}",
            "X-Sample-Rate": str(sample_rate),
            "X-Chunks-Generated": str(total_chunks),
            "X-Generation-Time": f"{generation_time:.2f}",
            "X-RTF": f"{rtf:.3f}",
        },
    )


def main():
    """Entry point for the worker server."""
    import argparse

    parser = argparse.ArgumentParser(description="Reeder TTS Worker")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load_model(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
