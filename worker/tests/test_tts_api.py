import io
import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

import sys
from pathlib import Path

# Add worker directory and workspace root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tts_api


def create_dummy_wav(duration_seconds: float = 1.0, sample_rate: int = 24000) -> bytes:
    """Create a valid in-memory PCM WAV file."""
    buf = io.BytesIO()
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


class TestTTSApiGateway(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.voices_dir = Path(self.tmp_dir.name)
        tts_api.VOICES_DIR = self.voices_dir
        tts_api.ACTIVE_TTS_MODEL = "qwen3-tts"
        tts_api.AUDIOCPP_URL = "http://127.0.0.1:8080"
        self.client = TestClient(tts_api.app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("tts_api.httpx.AsyncClient")
    def test_health_success(self, mock_client_cls):
        """Test health endpoint returns 200 when audio.cpp is reachable."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_health_resp = MagicMock(status_code=200)
        mock_models_resp = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "qwen3-tts"}, {"id": "pocket-tts"}, {"id": "kokoro-82m"}]},
        )
        mock_client.get.side_effect = [mock_health_resp, mock_models_resp]

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["model"], "qwen3-tts")
        self.assertEqual(data["active_model"], "qwen3-tts")
        self.assertIn("pocket-tts", data["available_models"])
        self.assertIn("kokoro-82m", data["available_models"])
        self.assertEqual(data["backend"], "audio.cpp (cuda)")
        self.assertEqual(data["device"], "cuda:0")

    @patch("tts_api.httpx.AsyncClient")
    def test_health_unreachable(self, mock_client_cls):
        """Test health endpoint returns 503 when audio.cpp is unreachable."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)

    @patch("tts_api.httpx.AsyncClient")
    def test_generate_voice_cloning(self, mock_client_cls):
        """Test voice cloning routes reference wav and text to audio.cpp."""
        # Create voice files
        sample_wav = self.voices_dir / "narrator.wav"
        sample_wav.write_bytes(create_dummy_wav(0.5))
        sample_txt = self.voices_dir / "narrator.txt"
        sample_txt.write_text("This is the reference transcript.")

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Calibrate response length to the seed history so the outlier
        # detector doesn't retry: 5500 samples/token, text is 10 est. tokens.
        output_wav = create_dummy_wav(5500 * 10 / 24000, 24000)
        mock_resp = MagicMock(status_code=200, content=output_wav)
        mock_client.post.return_value = mock_resp

        payload = {
            "text": "Hello world from Reeder audio.cpp test",
            "voice": "narrator",
            "temperature": 0.7,
            "language": "en",
        }
        with patch.object(tts_api, "SAMPLES_PER_TOKEN_SEED", [5400.0, 5500.0, 5600.0]):
            resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["Content-Type"], "audio/wav")
        self.assertEqual(resp.headers["X-Sample-Rate"], "24000")
        self.assertEqual(resp.headers["X-Duration-Seconds"], "2.29")
        self.assertIn("X-RTF", resp.headers)
        self.assertIn("X-Generation-Time", resp.headers)
        self.assertEqual(resp.headers["X-Chunks-Generated"], "1")
        self.assertEqual(resp.headers["X-Retried-Chunks"], "0")
        self.assertEqual(resp.content, output_wav)
        self.assertEqual(mock_client.post.call_count, 1)

        # Check call arguments
        mock_client.post.assert_called_once()
        sent_payload = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["model"], "qwen3-tts")
        self.assertEqual(sent_payload["input"], "Hello world from Reeder audio.cpp test")
        self.assertEqual(sent_payload["voice_ref"], str(sample_wav))
        self.assertEqual(sent_payload["reference_text"], "This is the reference transcript.")
        self.assertEqual(sent_payload["temperature"], 0.7)
        self.assertEqual(sent_payload["response_format"], "wav")

    @patch("tts_api.httpx.AsyncClient")
    def test_generate_voice_alias_json(self, mock_client_cls):
        """Test voice alias JSON routes to specified model and preset speaker."""
        # Create alias file: kokoro-bella.json
        alias_file = self.voices_dir / "kokoro-bella.json"
        alias_file.write_text(json.dumps({
            "model": "kokoro-82m",
            "voice_id": "af_bella",
            "language": "en-us"
        }))

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # "Testing Kokoro preset voice" is 7 estimated tokens -> 1.604s of
        # audio looks statistically normal against this seed history.
        output_wav = create_dummy_wav(5500 * 7 / 24000, 24000)
        mock_resp = MagicMock(status_code=200, content=output_wav)
        mock_client.post.return_value = mock_resp

        payload = {
            "text": "Testing Kokoro preset voice",
            "voice": "kokoro-bella",
            "temperature": 0.8,
        }
        with patch.object(tts_api, "SAMPLES_PER_TOKEN_SEED", [5400.0, 5500.0, 5600.0]):
            resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_client.post.call_count, 1)

        sent_payload = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["model"], "kokoro-82m")
        self.assertEqual(sent_payload["voice"], "af_bella")
        self.assertEqual(sent_payload["language"], "en-us")
        self.assertNotIn("voice_ref", sent_payload)

    def test_generate_missing_voice(self):
        """Test 404 returned when requested voice files do not exist."""
        payload = {
            "text": "Missing voice test",
            "voice": "nonexistent_voice_sample",
        }
        resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 404)


    @patch("tts_api.httpx.AsyncClient")
    def test_generate_chunks_long_text(self, mock_client_cls):
        """Long text is split into multiple chunks that are synthesized and concatenated."""
        sample_wav = self.voices_dir / "narrator.wav"
        sample_wav.write_bytes(create_dummy_wav(0.5))
        (self.voices_dir / "narrator.txt").write_text("Reference transcript.")

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        # Narrow seed history so "normal" responses are easy to calibrate:
        # 5500 samples per (estimated) token at 24kHz.
        seed = [5400.0, 5500.0, 5600.0]

        long_text = " ".join(
            f"Sentence number {i} contains enough words to span multiple token chunks."
            for i in range(40)
        )
        with patch.object(tts_api, "SAMPLES_PER_TOKEN_SEED", seed):
            chunks = tts_api.split_text_into_chunks(long_text, max_tokens=100)
            num_chunks = len(chunks)
            self.assertGreater(num_chunks, 1)

            # Each chunk responds with audio sized to look statistically normal
            expected_frames = 0
            responses = []
            for chunk in chunks:
                frames = int(5500 * tts_api.estimate_tokens(chunk))
                expected_frames += frames
                responses.append(MagicMock(
                    status_code=200,
                    content=create_dummy_wav(frames / 24000.0, 24000),
                ))
            mock_client.post.side_effect = responses

            resp = self.client.post("/generate", json={"text": long_text, "voice": "narrator"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(int(resp.headers["X-Chunks-Generated"]), num_chunks)
        self.assertEqual(int(resp.headers["X-Retried-Chunks"]), 0)
        self.assertEqual(mock_client.post.call_count, num_chunks)

        # Every dispatched payload must carry a chunk-sized input, not the full text
        for call in mock_client.post.call_args_list:
            self.assertLess(len(call.kwargs["json"]["input"]), len(long_text))

        # The returned WAV must contain the concatenated PCM of all chunks
        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            self.assertEqual(wf.getframerate(), 24000)
            self.assertEqual(wf.getnframes(), expected_frames)

        # Short single-sentence text must remain a single chunk
        mock_client.post.reset_mock()
        mock_client.post.side_effect = None
        # "One short sentence." is 5 estimated tokens -> 1.146s looks normal
        mock_client.post.return_value = MagicMock(
            status_code=200, content=create_dummy_wav(5500 * 5 / 24000, 24000)
        )
        with patch.object(tts_api, "SAMPLES_PER_TOKEN_SEED", seed):
            resp = self.client.post("/generate", json={"text": "One short sentence.", "voice": "narrator"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["X-Chunks-Generated"], "1")
        self.assertEqual(mock_client.post.call_count, 1)

    @patch("tts_api.httpx.AsyncClient")
    def test_generate_retries_runaway_chunk(self, mock_client_cls):
        """Chunks whose audio length is a statistical outlier are regenerated."""
        sample_wav = self.voices_dir / "narrator.wav"
        sample_wav.write_bytes(create_dummy_wav(0.5))
        (self.voices_dir / "narrator.txt").write_text("Reference transcript.")

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        seed = [5400.0, 5500.0, 5600.0]
        text = "First chunk sentence. Second chunk sentence that is a bit longer here."
        payload = {"text": text, "voice": "narrator", "max_tokens_per_chunk": 10}

        with patch.object(tts_api, "SAMPLES_PER_TOKEN_SEED", seed):
            chunks = tts_api.split_text_into_chunks(text, max_tokens=10)
            self.assertGreaterEqual(len(chunks), 2)

            def normal_resp(chunk):
                frames = int(5500 * tts_api.estimate_tokens(chunk))
                return MagicMock(status_code=200, content=create_dummy_wav(frames / 24000.0, 24000))

            def runaway_resp(chunk):
                frames = int(5500 * tts_api.estimate_tokens(chunk)) * 100
                return MagicMock(status_code=200, content=create_dummy_wav(frames / 24000.0, 24000))

            # Chunk 1 normal; chunk 2 runaway once, then normal
            mock_client.post.side_effect = [
                normal_resp(chunks[0]),
                runaway_resp(chunks[1]),
                normal_resp(chunks[1]),
            ]

            resp = self.client.post("/generate", json=payload)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(int(resp.headers["X-Chunks-Generated"]), len(chunks))
        self.assertEqual(int(resp.headers["X-Retried-Chunks"]), 1)
        self.assertEqual(mock_client.post.call_count, len(chunks) + 1)

    def test_split_text_into_chunks(self):
        """Chunk splitting respects token budget, sentences, and abbreviations."""
        text = "Mr. Smith went to Washington. He arrived at 3 p.m. and spoke. " * 5
        chunks = tts_api.split_text_into_chunks(text, max_tokens=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(tts_api.estimate_tokens(chunk), 20 + tts_api.estimate_tokens(" "))
        # Short text is not split
        self.assertEqual(tts_api.split_text_into_chunks("Hi there.", max_tokens=100), ["Hi there."])
        # Clause fallback for long single sentences
        long_sentence = "word " * 200
        clause_chunks = tts_api.split_text_into_chunks(long_sentence.replace(" ", ", ", 10), max_tokens=10)
        self.assertGreater(len(clause_chunks), 1)

    @patch("tts_api.httpx.AsyncClient")
    def test_reeder_tts_remote_integration(self, mock_client_cls):
        """Test compatibility with reeder/tts_remote.py client functions."""
        import threading
        import uvicorn
        from urllib.request import urlopen, Request
        from reeder.tts_remote import check_remote_health, generate_audio_remote

        # Create dummy voice
        sample_wav = self.voices_dir / "default.wav"
        sample_wav.write_bytes(create_dummy_wav(1.0))
        sample_txt = self.voices_dir / "default.txt"
        sample_txt.write_text("Default reference text.")

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_health_resp = MagicMock(status_code=200)
        mock_models_resp = MagicMock(status_code=200, json=lambda: {"data": [{"id": "qwen3-tts"}]})
        mock_client.get.return_value = mock_health_resp

        output_wav = create_dummy_wav(3.0, 24000)
        mock_resp = MagicMock(status_code=200, content=output_wav)
        mock_client.post.return_value = mock_resp

        # Start gateway in a local test server
        config = uvicorn.Config(tts_api.app, host="127.0.0.1", port=8199, log_level="warning")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Wait for server to bind
        time.sleep(0.5)

        try:
            worker_url = "http://127.0.0.1:8199"
            self.assertTrue(check_remote_health(worker_url, timeout=2.0))

            output_file = Path(self.tmp_dir.name) / "output.wav"
            reeder_config = {
                "tts": {
                    "remote": {"enabled": True, "url": worker_url, "timeout": 10},
                    "temperature": 0.8,
                    "max_tokens_per_chunk": 100,
                }
            }
            job = {"voice": "default", "temperature": 0.8, "language": "Auto"}
            success = generate_audio_remote(
                text="Integration test between reeder and audio.cpp gateway",
                output_path=output_file,
                job=job,
                config=reeder_config,
                paths={"voices": self.voices_dir},
            )
            self.assertTrue(success)
            self.assertTrue(output_file.is_file())
            self.assertEqual(output_file.read_bytes(), output_wav)
        finally:
            server.should_exit = True
            server_thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
