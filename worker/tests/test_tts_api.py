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

        output_wav = create_dummy_wav(2.5, 24000)
        mock_resp = MagicMock(status_code=200, content=output_wav)
        mock_client.post.return_value = mock_resp

        payload = {
            "text": "Hello world from Reeder audio.cpp test",
            "voice": "narrator",
            "temperature": 0.7,
            "language": "en",
        }
        resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["Content-Type"], "audio/wav")
        self.assertEqual(resp.headers["X-Sample-Rate"], "24000")
        self.assertEqual(resp.headers["X-Duration-Seconds"], "2.50")
        self.assertIn("X-RTF", resp.headers)
        self.assertIn("X-Generation-Time", resp.headers)
        self.assertEqual(resp.headers["X-Chunks-Generated"], "1")
        self.assertEqual(resp.content, output_wav)

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

        output_wav = create_dummy_wav(1.0, 24000)
        mock_resp = MagicMock(status_code=200, content=output_wav)
        mock_client.post.return_value = mock_resp

        payload = {
            "text": "Testing Kokoro preset voice",
            "voice": "kokoro-bella",
            "temperature": 0.8,
        }
        resp = self.client.post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200)

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
