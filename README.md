# Reeder

A personal TTS RSS service that converts articles and text to audio, served as a podcast feed.

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
sudo nano /var/lib/reeder/config.toml

# Add a voice file
sudo cp your-voice.wav /var/lib/reeder/voices/default.wav
sudo chown reeder:reeder /var/lib/reeder/voices/default.wav

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
# Run locally (from repo directory)
export REEDER_CONFIG=$PWD/config.toml

# Create local directories
mkdir -p inbox processing done www/audio voices var

# Process a job manually
./bin/process-job

# Update feed manually
./bin/update-feed
```

## Requirements

- Python 3.11+
- ffmpeg
- curl
- pup (for custom CSS selectors)
- uv (Python package manager)
- Caddy (optional, for HTTPS)

## License

MIT
