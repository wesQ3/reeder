"""Reeder TTS Worker — GPU-accelerated Qwen3-TTS API.

Provides a FastAPI server that generates voice-cloned audio using CUDA.
Designed to run on a GPU-equipped machine, called by the main reeder
service on nuc0.
"""

import logging
import re
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Reeder TTS Worker",
    description="GPU-accelerated TTS generation via Qwen3-TTS",
    version="0.1.0",
)

# Global model state
_model: Qwen3TTSModel | None = None
_model_name: str = ""
_device: str = "cuda"


def get_model() -> Qwen3TTSModel:
    """Get the loaded TTS model (must call load_model first)."""
    if _model is None:
        raise RuntimeError("Model not loaded. Server not initialized properly.")
    return _model


def load_model(model_name: str, device: str = "cuda"):
    """Load the Qwen3-TTS model."""
    global _model, _model_name, _device
    _device = device
    _model_name = model_name

    logger.info(f"Loading model: {model_name} on {device}")
    # Keep float32 to match the known-stable local path in reeder.tts.
    dtype = torch.float32
    _model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
    )
    logger.info("Model loaded successfully")


def split_text_into_chunks(text: str, tokenizer, max_tokens: int = 50) -> list[str]:
    """Split text into chunks at sentence boundaries using the tokenizer."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    total_tokens = len(tokenizer(text, return_tensors="pt").input_ids[0])
    if total_tokens <= max_tokens:
        return [text]

    abbreviations = [
        (r"\bMr\.", "Mr\x00"), (r"\bMrs\.", "Mrs\x00"),
        (r"\bMs\.", "Ms\x00"), (r"\bDr\.", "Dr\x00"),
        (r"\bProf\.", "Prof\x00"), (r"\bSr\.", "Sr\x00"),
        (r"\bJr\.", "Jr\x00"), (r"\bSt\.", "St\x00"),
        (r"\bvs\.", "vs\x00"), (r"\betc\.", "etc\x00"),
        (r"\be\.g\.", "eg\x00"), (r"\bi\.e\.", "ie\x00"),
        (r"\bU\.S\.", "US\x00"), (r"\bU\.K\.", "UK\x00"),
        (r"\bNo\.", "No\x00"), (r"\bCo\.", "Co\x00"),
        (r"\bInc\.", "Inc\x00"), (r"\bLtd\.", "Ltd\x00"),
        (r"\bCorp\.", "Corp\x00"),
    ]

    protected = text
    for pattern, replacement in abbreviations:
        protected = re.sub(pattern, replacement, protected)

    sentences_raw = re.split(r"(?<=[.!?])\s+(?=[A-Z]|$)", protected)
    sentences = []
    for sentence in sentences_raw:
        restored = sentence.replace("\x00", ".")
        if restored.strip():
            sentences.append(restored.strip())

    if not sentences:
        sentences = [text]

    chunks = []
    current_chunk = ""
    current_tokens = 0
    for sentence in sentences:
        num_tokens = len(tokenizer(sentence, return_tensors="pt").input_ids[0])
        if not current_chunk:
            current_chunk = sentence
            current_tokens = num_tokens
        elif current_tokens + num_tokens <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += num_tokens
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_tokens = num_tokens
    if current_chunk:
        chunks.append(current_chunk)

    final_chunks = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer(chunk, return_tensors="pt").input_ids[0])
        if chunk_tokens <= max_tokens:
            final_chunks.append(chunk)
            continue

        parts = re.split(r"(?<=[,;:])\s+", chunk)
        sub_chunk = ""
        sub_tokens = 0
        for part in parts:
            part_tokens = len(tokenizer(part, return_tensors="pt").input_ids[0])
            if not sub_chunk:
                sub_chunk = part
                sub_tokens = part_tokens
            elif sub_tokens + part_tokens <= max_tokens:
                sub_chunk += " " + part
                sub_tokens += part_tokens
            else:
                final_chunks.append(sub_chunk)
                sub_chunk = part
                sub_tokens = part_tokens
        if sub_chunk:
            final_chunks.append(sub_chunk)

    return final_chunks


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
        status="ready" if _model is not None else "not_loaded",
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate voice-cloned audio from text.

    Returns the audio as a WAV file response with metadata headers.
    """
    from fastapi.responses import Response

    model = get_model()
    voices_dir = Path("/data/voices")

    # Resolve voice files
    voice_name = request.voice
    if voice_name == "default":
        # Find any .wav in voices dir
        wav_files = list(voices_dir.glob("*.wav"))
        if not wav_files:
            raise HTTPException(status_code=500, detail="No voice files found in /data/voices")
        voice_wav = wav_files[0]
    else:
        voice_wav = voices_dir / f"{voice_name}.wav"

    voice_txt = voice_wav.with_suffix(".txt")
    if not voice_wav.exists():
        raise HTTPException(status_code=400, detail=f"Voice audio not found: {voice_wav.name}")
    if not voice_txt.exists():
        raise HTTPException(status_code=400, detail=f"Voice transcript not found: {voice_txt.name}")

    ref_text = voice_txt.read_text().strip()
    logger.info(f"Generating: {len(request.text)} chars, voice={voice_name}, temp={request.temperature}")

    start_time = time.monotonic()

    tokenizer = model.processor.tokenizer
    chunks = split_text_into_chunks(request.text, tokenizer, max_tokens=request.max_tokens_per_chunk)
    total_chunks = len(chunks)
    logger.info(f"  Split into {total_chunks} chunks")

    # Create voice clone prompt
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=str(voice_wav),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    gen_kwargs = dict(
        pad_token_id=model.processor.tokenizer.eos_token_id,
        max_new_tokens=4096,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=request.temperature,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=request.temperature,
    )

    # Generate audio chunks with outlier detection
    sample_per_token_history = [
        5110, 4793, 5234, 5889, 5130, 5941, 5877, 5138,
        5607, 6370, 6260, 6381, 5894, 5538, 6027, 5280,
    ]
    std_dev_limit = 3.0
    all_audio = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        attempt, max_attempts, is_outlier = 1, 3, True
        while is_outlier and attempt <= max_attempts:
            chunk_tokens = len(tokenizer(chunk, return_tensors="pt").input_ids[0])
            wavs, sr = model.generate_voice_clone(
                text=chunk,
                language=request.language,
                voice_clone_prompt=voice_clone_prompt,
                **gen_kwargs,
            )
            segment_samples = len(wavs[0])
            samples_per_token = segment_samples / chunk_tokens if chunk_tokens > 0 else 0.0

            running_avg = statistics.mean(sample_per_token_history)
            std_dev = statistics.stdev(sample_per_token_history)
            z_score = abs(samples_per_token - running_avg) / std_dev if std_dev > 0 else 0
            is_outlier = z_score > std_dev_limit

            flag = " RETRY" if is_outlier else ""
            logger.info(
                f"  Chunk {i+1}/{total_chunks} | {chunk_tokens}t | "
                f"{samples_per_token:.0f} s/t | z={z_score:.2f}{flag}"
            )
            attempt += 1

        sample_per_token_history.append(samples_per_token)
        all_audio.append(wavs[0])
        sample_rate = sr

    if sample_rate is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    # Concatenate all chunks
    full_audio = np.concatenate(all_audio)
    generation_time = time.monotonic() - start_time
    duration = len(full_audio) / sample_rate
    rtf = generation_time / duration if duration > 0 else 0

    logger.info(f"  Done: {duration:.1f}s audio in {generation_time:.1f}s (RTF={rtf:.2f}x)")

    # Write WAV to buffer
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, full_audio, sample_rate)
        tmp_path = Path(tmp.name)

    wav_bytes = tmp_path.read_bytes()
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
