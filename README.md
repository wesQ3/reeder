# Reeder

A personal TTS RSS service that converts articles and text to audio, served as a podcast feed.

> **Be advised:** Cobbled together by various LLMs. Claude, take the wheel!

## Overview

```
inbox/           → Drop job files here
  ↓
processing/      → Worker processes jobs one at a time
  ↓
done/            → Completed job metadata
  ↓
www/audio/       → Generated audio files
www/feed.xml     → RSS podcast feed
```

## Quick Start

```bash
# Install
./install.sh

# Configure (edit base_url and voice settings)
sudo vim /var/lib/reeder/config.toml

# Add a voice file
sudo cp your-voice.wav /var/lib/reeder/voices/default.wav
sudo chown reeder:reeder /var/lib/reeder/voices/default.wav

# Convert to safetensors for faster loading (recommended)
# Note: Uses GitHub version for safetensors support
sudo -u reeder uvx --from git+https://github.com/kyutai-labs/pocket-tts.git pocket-tts export-voice /var/lib/reeder/voices/default.wav /var/lib/reeder/voices/default.safetensors --truncate

# Submit a test job
echo '{"type":"text","text":"Hello world","title":"Test"}' | \
  sudo -u reeder tee /var/lib/reeder/inbox/$(date +%s)-test.json

# Monitor
journalctl -u reeder -f
```

## Job Submission

### Via helper scripts

```bash
# URL (fetches and extracts article text)
submit-url https://example.com/article "Optional Title"

# Direct text
submit-text "My Notes" "Text to convert..."
echo "Piped text" | submit-text "From Stdin"
```

### Manual job files

Drop a JSON file in `/var/lib/reeder/inbox/`:

```json
{
  "type": "url",
  "url": "https://example.com/article",
  "title": "Article Title"
}
```

See [docs/job-format.md](docs/job-format.md) for full schema.

## Monitoring

```bash
# Current status
reeder-status

# Live status file
tail -f /var/lib/reeder/var/status.txt

# System logs
journalctl -u reeder -f
```

## Configuration

Edit `/var/lib/reeder/config.toml`:

- **base_url**: Your Tailscale hostname (e.g., `https://myserver.tail1234.ts.net/reeder`)
- **default_voice**: Voice file for TTS (place in `voices/` directory)
- **audio_format**: `opus` (smaller) or `mp3` (more compatible)
- **temperature**: TTS expressiveness (0.0-1.0)

## RSS Feed

Subscribe to `https://your-hostname/feed.xml` in any podcast app:
- Pocket Casts
- Overcast
- AntennaPod
- Apple Podcasts

## Architecture

- **systemd.path**: Watches inbox for new job files
- **systemd.service**: Processes one job at a time
- **pocket-tts**: Generates speech audio
- **trafilatura**: Extracts article text from URLs
- **Caddy**: Serves audio files and RSS feed over HTTPS

## Development

```bash
# Install dependencies
uv sync

# Run locally (from repo directory)
export REEDER_CONFIG=config.dev.toml

# Submit and process a job
bin/submit-url https://example.com/article "Test Article"
uv run bin/process-job

# Update feed manually
uv run bin/update-feed

# Check status
bin/reeder-status
```

## Requirements

- Python 3.11+
- uv (Python package manager)
- ffmpeg
- curl
- pup (for custom CSS selectors)
- Caddy (optional, for HTTPS)

## License

MIT License - see [LICENSE](LICENSE) file for details.
