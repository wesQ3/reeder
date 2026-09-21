# audio.cpp Migration & Model Playbook

This document details the migration of the Reeder TTS worker from the Python/PyTorch runtime to [audio.cpp](https://github.com/0xShug0/audio.cpp) (a C++/ggml inference engine with CUDA acceleration and GGUF quantization).

---

## Architecture Overview

```
[User Browser]
       │
       ▼
[reeder-web (nuc0:8081)] ──> writes job.json to inbox/ (NO CODE CHANGES)
       │
[process-job (nuc0 daemon)] ──> reeder/tts_remote.py (NO CODE CHANGES)
       │
       ├─> GET  http://worker:8100/health
       └─> POST http://worker:8100/generate
             │
┌────────────▼────────────────────────────────────────────────────────┐
│ Worker Container (GPU Host)                                         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Reeder Compatibility Gateway (Port 8100)                      │  │
│  │ - Implements GET /health & POST /generate                     │  │
│  │ - Translates voice names to clone refs or preset speaker IDs  │  │
│  │ - Dispatches requests to internal audio.cpp server            │  │
│  │ - Computes audio headers: X-Duration-Seconds, X-RTF, etc.     │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
│                                  │ Internal HTTP (127.0.0.1:8080)   │
│                                  ▼                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ audio.cpp Native Server (audiocpp_server on port 8080)        │  │
│  │ - Native C++ / CUDA inference powered by ggml                 │  │
│  │ - OpenAI-compatible API: POST /v1/audio/speech                │  │
│  │ - GGUF model support (Qwen3-TTS, PocketTTS, Kokoro, etc.)     │  │
│  │ - Embedded WebUI & Arena mode on port 8080                    │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
│                                  │                                  │
│                       /app/models │   /data/voices                   │
└───────────────────────────┬───────┴─────────┬────────────────────────┘
                            ▼                 ▼
                     [GGUF Models]     [Voice Samples & Transcripts]
```

All paths referenced by processes inside the container are internal
(`/app/models`, `/data/voices`). Host paths appear only in the compose file's
volume substitutions.

### Key Highlights
- **Zero changes** to `reeder/`, `bin/reeder-web`, `bin/process-job`, or systemd configurations.
- **Dramatically reduced footprint**: Python CUDA/PyTorch dependencies (~7-8 GB) eliminated in favor of native C++ kernels.
- **Instant container builds**: Builds take seconds instead of 15+ minutes.
- **Lower latency & faster RTF**: 2x–8x faster inference via GGUF Q8_0/FP16 native execution.

---

## Quickstart

### 1. Configure Host Environment
In `worker/` (run all `docker compose` commands from this directory so the
local `.env` is picked up):
```bash
cp .env.example .env
vim .env
```
Ensure `VOICES_DIR` points to your voice files directory and `MODELS_DIR` points to your GGUF models directory:
```bash
VOICES_DIR=/var/lib/reeder/voices
MODELS_DIR=/var/lib/reeder/models
ACTIVE_TTS_MODEL=qwen3-tts
AUDIOCPP_BACKEND=cuda
```

> **Note:** `.env` values are used by docker-compose on the host for volume
> substitution only (`${VOICES_DIR}:/data/voices`, `${MODELS_DIR}:/app/models`).
> They are intentionally **not** loaded into the container environment — the
> container only ever sees the internal paths.

### 2. Launch Worker Container
```bash
docker compose up -d
```
Check container logs:
```bash
docker compose logs -f
```

### 3. Verify Health
```bash
curl -i http://localhost:8100/health
```
Expected response:
```json
{
  "status": "ready",
  "model": "qwen3-tts",
  "backend": "audio.cpp (cuda)",
  "device": "cuda:0",
  "active_model": "qwen3-tts",
  "available_models": ["qwen3-tts", "qwen3-tts-0.6b", "pocket-tts", "kokoro-82m"]
}
```

---

## Multi-Model & Voice Routing Playbook

The compatibility gateway seamlessly routes requests based on the selected voice:

### 1. Voice Cloning (Qwen3-TTS / PocketTTS)
For voices requiring cloning, Reeder looks for:
- `/data/voices/<voice>.wav` (Reference audio sample)
- `/data/voices/<voice>.txt` (Matching reference transcript)

When `voice="default"` or `voice="wes"` is chosen, the gateway provides `voice_ref` and `reference_text` to the active cloning model (`qwen3-tts`).

### 2. Preset Speakers & Model Aliases (e.g. Kokoro, Supertonic)
To use preset models without cloning audio, create a JSON alias in `/data/voices/`:

Example: `/data/voices/kokoro-bella.json`
```json
{
  "model": "kokoro-82m",
  "voice_id": "af_bella",
  "language": "en-us"
}
```

To display `kokoro-bella` in the Reeder web UI dropdown without modifying `reeder-web`, place an empty stub file `/data/voices/kokoro-bella.wav` next to the JSON file. Reeder's `enumerate_voices()` will list it, and the gateway will parse `kokoro-bella.json` and route directly to Kokoro with preset speaker `af_bella`!

### 3. Adding New Models to `server.json`
Edit `worker/server.json`:
```json
{
  "id": "pocket-tts",
  "family": "pocket_tts",
  "path": "/app/models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
  "task": "tts",
  "mode": "offline"
}
```
With `"lazy_load": true`, models are loaded into VRAM only when requested and evicted when idle according to `"max_loaded_models": 2`.

---

## Embedded WebUI & Model Arena

audio.cpp includes an embedded WebUI exposed on port `8080`:

1. Open `http://<worker-ip>:8080` in your web browser.
2. **Model Management**: Search, download, and manage GGUF models directly via HuggingFace or ModelScope catalogs.
3. **Model Arena**: Compare models side-by-side on the same prompt to evaluate voice naturalness, RTF, tokens/sec, and GPU memory usage.

Models downloaded via the WebUI are installed to audio.cpp's models root,
`/app/models` inside the container — which is your host `MODELS_DIR`, so they
persist across container restarts and are usable from the host too.

---

## Chunking & Runaway-Generation Detection

The gateway **never forwards full articles** to audio.cpp. The `qwen3_tts`
speech decoder allocates its CUDA graph proportional to the input length, so a
56,000-character article in a single request exhausts VRAM
(`cudaMalloc failed: out of memory`) even on a 16 GB card. Instead, the
gateway:

1. **Splits text into chunks** at sentence boundaries (with abbreviation
   protection and clause-boundary fallback), bounded by
   `max_tokens_per_chunk` (default 100, from the Reeder request). Token counts
   are estimated at ~4 chars/token — configurable via `CHARS_PER_TOKEN` —
   since the lightweight gateway does not load a tokenizer.
2. **Synthesizes chunks sequentially** against `/v1/audio/speech`, streaming
   each chunk's PCM to disk (articles can be hours of audio) and assembling
   the final WAV at the end. Sequential generation keeps the backend's VRAM
   footprint bounded to a single chunk.
3. **Detects runaway chunks**: each chunk's `samples-per-token` ratio is
   scored against a running history (z-score > 3 → regenerate, up to 3
   attempts). This catches degenerate generations that produce minutes of
   garbage audio for a short chunk.

Tuning environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CHARS_PER_TOKEN` | `4.0` | Token estimate calibration for chunk sizing and s/t stats |
| `OUTLIER_Z_LIMIT` | `3.0` | z-score threshold marking a chunk as runaway |
| `MAX_ATTEMPTS_PER_CHUNK` | `3` | Generation attempts per chunk |
| `CHUNK_HTTP_TIMEOUT` | `300` | HTTP timeout (seconds) per chunk request |
| `SAMPLES_PER_TOKEN_SEED` | preseeded | Comma-separated s/t history for outlier detection |

Note: long articles take proportionally long to generate (hundreds of chunks
× seconds each). Ensure the Reeder-side `tts.remote.timeout` accounts for this.

## Troubleshooting & Verification

### Test Audio Synthesis via cURL
```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Testing audio.cpp migration for Reeder.", "voice": "default"}' \
  --output test.wav -D -
```

Check the returned response headers:
- `X-Duration-Seconds`: Duration of synthesized audio
- `X-Sample-Rate`: Sampling rate (e.g., 24000)
- `X-Chunks-Generated`: Number of chunks processed
- `X-Retried-Chunks`: Chunks that were regenerated as statistical outliers
- `X-Generation-Time`: Processing time in seconds
- `X-RTF`: Real-time factor (`wall_time / audio_duration`)
