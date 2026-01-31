# Voice Files for Qwen3-TTS

Reeder uses Qwen3-TTS for voice cloning. Each voice requires two files:
- A `.wav` audio file (the voice sample)
- A `.txt` transcript file (exact text spoken in the audio)

## Voice Cloning Requirements

For best results:
- Use a clean audio sample (minimal background noise)
- Keep the sample short (10-30 seconds is ideal)
- The transcript must match the spoken audio exactly
- Use clear, natural speech

## Setting Up Voices

### File Structure

```
voices/
├── default.wav      # Audio sample
├── default.txt      # Transcript of what's spoken
├── narrator.wav
├── narrator.txt
└── ...
```

### Example

1. Record or obtain a voice sample:
```bash
# Example voice sample
cp my-recording.wav voices/my-voice.wav
```

2. Create a transcript file with the exact text spoken:
```bash
echo "Hello, this is my voice sample for the text to speech system." > voices/my-voice.txt
```

3. Update config.toml to use the voice:
```toml
[tts]
default_voice = "my-voice"  # Just the base name, without extension
```

## Using Voices in Jobs

### In config.toml

```toml
[tts]
default_voice = "my-voice"  # relative to voices/ directory
```

### In job files

```json
{
  "type": "url",
  "url": "https://example.com/article",
  "voice": "my-voice"
}
```

Or use an absolute path:
```json
{
  "voice": "/path/to/voices/my-voice"
}
```

## File Locations

- **Development**: `voices/` in the repo
- **Production**: `/var/lib/reeder/voices/`

## Tips

- Use clear recordings with minimal background noise
- Shorter samples (10-30 seconds) work well
- Ensure the transcript is accurate—Qwen3-TTS uses it to learn voice characteristics
- Multiple voices can be used by specifying different voice names in job files
