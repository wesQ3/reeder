"""TTS generation using Qwen3-TTS with voice cloning.

Handles text chunking, audio generation with outlier detection/retry,
and PCM-to-WAV conversion.
"""

import re
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np


# Global TTS model (lazy-loaded)
_tts_model = None


def get_tts_model(model_name: str, device: str = "cpu"):
    """Get or load the Qwen3-TTS model (singleton)."""
    global _tts_model
    if _tts_model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        print(f"Loading TTS model: {model_name}", flush=True)
        print(f"  Device: {device}", flush=True)
        _tts_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
        print("  Model loaded successfully.", flush=True)
    return _tts_model


def split_text_into_chunks(text: str, tokenizer, max_tokens: int = 50) -> list[str]:
    """Split text into chunks at sentence boundaries using the tokenizer.

    Uses actual token counts for accurate chunking. Avoids splitting on
    common abbreviations (Mr., Dr., etc.).
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    total_tokens = len(tokenizer(text, return_tensors='pt').input_ids[0])
    if total_tokens <= max_tokens:
        return [text]

    # Common abbreviations that shouldn't trigger sentence splits
    abbreviations = [
        (r'\bMr\.', 'Mr\x00'), (r'\bMrs\.', 'Mrs\x00'),
        (r'\bMs\.', 'Ms\x00'), (r'\bDr\.', 'Dr\x00'),
        (r'\bProf\.', 'Prof\x00'), (r'\bSr\.', 'Sr\x00'),
        (r'\bJr\.', 'Jr\x00'), (r'\bSt\.', 'St\x00'),
        (r'\bvs\.', 'vs\x00'), (r'\betc\.', 'etc\x00'),
        (r'\be\.g\.', 'eg\x00'), (r'\bi\.e\.', 'ie\x00'),
        (r'\bU\.S\.', 'US\x00'), (r'\bU\.K\.', 'UK\x00'),
        (r'\bNo\.', 'No\x00'), (r'\bCo\.', 'Co\x00'),
        (r'\bInc\.', 'Inc\x00'), (r'\bLtd\.', 'Ltd\x00'),
        (r'\bCorp\.', 'Corp\x00'),
    ]

    protected = text
    for pattern, replacement in abbreviations:
        protected = re.sub(pattern, replacement, protected)

    # Split into sentences at .!? followed by space and capital letter
    sentences_raw = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected)

    sentences = []
    for s in sentences_raw:
        restored = s.replace('\x00', '.')
        if restored.strip():
            sentences.append(restored.strip())

    if not sentences:
        sentences = [text]

    # Count tokens for each sentence
    sentence_tokens = []
    for sentence in sentences:
        num_tokens = len(tokenizer(sentence, return_tensors='pt').input_ids[0])
        sentence_tokens.append((sentence, num_tokens))

    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence, num_tokens in sentence_tokens:
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

    # Handle chunks that are still too long (single long sentence)
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer(chunk, return_tensors='pt').input_ids[0])
        if chunk_tokens <= max_tokens:
            final_chunks.append(chunk)
        else:
            # Split on clause boundaries
            parts = re.split(r'(?<=[,;:])\s+', chunk)
            sub_chunk = ""
            sub_tokens = 0
            for part in parts:
                part_tokens = len(tokenizer(part, return_tensors='pt').input_ids[0])
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


def resolve_voice(voice: str, config: dict, voices_dir: Path) -> tuple[Path, Path, str]:
    """Resolve voice name to (wav_path, txt_path, voice_name).

    Raises ValueError if voice files are missing.
    """
    if voice == "default":
        default_voice = config["tts"]["default_voice"]
        voice_name = Path(default_voice).stem
        voice_wav = voices_dir / f"{voice_name}.wav"
    elif not Path(voice).is_absolute():
        voice_name = Path(voice).stem
        voice_wav = voices_dir / f"{voice_name}.wav"
    else:
        voice_wav = Path(voice)
        voice_name = voice_wav.stem

    voice_txt = voice_wav.with_suffix(".txt")
    if not voice_txt.exists():
        raise ValueError(f"Voice transcript not found: {voice_txt}")
    if not voice_wav.exists():
        raise ValueError(f"Voice audio not found: {voice_wav}")

    return voice_wav, voice_txt, voice_name


def generate_audio(
    text: str,
    output_path: Path,
    job: dict,
    config: dict,
    paths: dict,
    status_callback: Callable[[str], None] | None = None,
):
    """Generate audio using Qwen3-TTS with chunking for long texts.

    Streams each chunk's audio to a raw PCM temp file to avoid holding
    hours of audio in memory. The PCM file is converted to WAV at the end.

    Args:
        text: Full text to synthesize.
        output_path: Where to write the final WAV file.
        job: Job dict (for voice, temperature, language overrides).
        config: Reeder config dict.
        paths: Resolved paths dict.
        status_callback: Optional function called with status messages.
    """
    def status(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"  {msg}", flush=True)

    voice = job.get("voice", "default")
    voice_wav, voice_txt, voice_name = resolve_voice(voice, config, paths["voices"])
    ref_text = voice_txt.read_text().strip()

    device = config["tts"].get("device", "cpu")
    model_name = config["tts"].get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    temperature = job.get("temperature", config["tts"].get("temperature", 0.9))
    language = job.get("language", "Auto")

    print(f"  Voice: {voice_name}", flush=True)
    print(f"  Reference audio: {voice_wav}", flush=True)
    print(f"  Temperature: {temperature}", flush=True)
    print(f"  Language: {language}", flush=True)

    tts = get_tts_model(model_name, device)

    # Split text into chunks
    tokenizer = tts.processor.tokenizer
    max_tokens = config["tts"].get("max_tokens_per_chunk", 100)
    chunks = split_text_into_chunks(text, tokenizer, max_tokens=max_tokens)
    total_chunks = len(chunks)
    job_name = job.get("title", "untitled")
    print(f"  Text split into {total_chunks} chunk(s)", flush=True)

    # Create voice clone prompt once
    print("  Creating voice clone prompt...", flush=True)
    voice_clone_prompt = tts.create_voice_clone_prompt(
        ref_audio=str(voice_wav),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    gen_kwargs = dict(
        pad_token_id=tts.processor.tokenizer.eos_token_id,
        max_new_tokens=4096,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=temperature,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=temperature,
    )

    # Stream chunks to raw PCM temp file
    pcm_path = output_path.with_suffix(".pcm")
    sample_rate = None
    # Preseed history based on past runs
    # TODO: this is possibly significantly different for each voice
    sample_per_token_history = [
        5110, 4793, 5234, 5889, 5130, 5941, 5877, 5138,
        5607, 6370, 6260, 6381, 5894, 5538, 6027, 5280,
    ]
    std_dev_limit = 3.0

    try:
        with open(pcm_path, "wb") as pcm_file:
            for i, chunk in enumerate(chunks):
                attempt, max_attempts, is_outlier = 1, 3, True
                while is_outlier and attempt <= max_attempts:
                    chunk_tokens = len(tokenizer(chunk, return_tensors="pt").input_ids[0])
                    title = "Generating audio:" if attempt == 1 else f"RETRY CHUNK {attempt}/{max_attempts}:"
                    status(f"{title} {job_name} (chunk {i+1:>3}/{total_chunks},{len(chunk):>3}c,{chunk_tokens:>3}t)")
                    wavs, sr = tts.generate_voice_clone(
                        text=chunk,
                        language=language,
                        voice_clone_prompt=voice_clone_prompt,
                        **gen_kwargs,
                    )
                    segment_samples = len(wavs[0])
                    samples_per_token = (
                        segment_samples / chunk_tokens if chunk_tokens > 0 else 0.0
                    )
                    running_avg = statistics.mean(sample_per_token_history)
                    std_dev = statistics.stdev(sample_per_token_history)

                    z_score = abs(samples_per_token - running_avg) / std_dev if std_dev > 0 else 0
                    is_outlier = z_score > std_dev_limit
                    flag = " !!!" if is_outlier else ""

                    print(
                        f"    Chunk {i+1:>3}/{total_chunks:<3} | {len(chunk):>3}c | {chunk_tokens:>3}t | "
                        f"{samples_per_token:>7,.2f} s/t | {segment_samples/sr:>5.2f}s | "
                        f"{running_avg:>7,.2f}avg | {std_dev:>6,.2f}σ | {z_score:>4.2f}z{flag}",
                        flush=True,
                    )
                    attempt += 1

                sample_per_token_history.append(samples_per_token)
                pcm_file.write(wavs[0].astype(np.float32).tobytes())
                sample_rate = sr

        if sample_rate is None:
            raise RuntimeError("No audio generated (empty text?)")

        # Convert raw PCM to WAV
        print("  Writing WAV from raw PCM...", flush=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", str(pcm_path),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        if pcm_path.exists():
            pcm_path.unlink()


def convert_audio(wav_path: Path, output_path: Path, config: dict) -> int:
    """Convert WAV to final format (opus/mp3). Returns duration in seconds."""
    audio_format = config["tts"].get("audio_format", "opus")

    if audio_format == "opus":
        bitrate = config["tts"].get("opus_bitrate", 48)
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-c:a", "libopus", "-b:a", f"{bitrate}k",
            str(output_path),
        ]
    else:  # mp3
        bitrate = config["tts"].get("mp3_bitrate", 64)
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)

    # Get duration via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(output_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = int(float(result.stdout.strip()))

    return duration
