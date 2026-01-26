# Voice Files and Safetensors

Reeder supports using pre-computed voice embeddings in `.safetensors` format for faster loading.

**Note:** Safetensors support requires the development version of pocket-tts from GitHub (automatically handled by the scripts).

## Why use safetensors?

Processing a WAV file for voice cloning takes time on every generation. Pre-computing the voice embedding and saving it as a `.safetensors` file means:
- **Faster startup** - No need to process the audio every time
- **Consistent results** - Same embedding used every time
- **Smaller files** - Embeddings are compact (~few MB vs potentially large audio files)

## Converting WAV to safetensors

### Quick method

Use the provided conversion script:

```bash
# Place your voice files in voices/ as .wav
cp my-voice.wav voices/my-voice.wav

# Convert all WAV files to safetensors
bin/convert-voices

# Optional: add --truncate to clip audio to 30 seconds
bin/convert-voices --truncate
```

### Manual conversion

```bash
# Single file
uvx pocket-tts export-voice voices/my-voice.wav voices/my-voice.safetensors

# Batch convert a directory
uvx pocket-tts export-voice voices/ voices/ --truncate

# From URL
uvx pocket-tts export-voice https://example.com/voice.wav voices/voice.safetensors
```

## Using safetensors files

### In config.toml

```toml
[tts]
default_voice = "my-voice.safetensors"  # relative to voices/ directory
```

### In job files

```json
{
  "type": "url",
  "url": "https://example.com/article",
  "voice": "my-voice.safetensors"
}
```

Or use an absolute path:
```json
{
  "voice": "/path/to/voices/my-voice.safetensors"
}
```

## File locations

- **Development**: `voices/` in the repo
- **Production**: `/var/lib/reeder/voices/`

## Recommended workflow

1. Record or obtain a clean voice sample (WAV, MP3, etc.)
2. Copy to `voices/` directory
3. Run `bin/convert-voices` to create `.safetensors`
4. Update `config.toml` to use the `.safetensors` file
5. (Optional) Delete the original WAV to save space

## Notes

- Voice embeddings work with any audio format pocket-tts supports (WAV, MP3, etc.)
- The `--truncate` flag automatically clips long audio files to 30 seconds
- Safetensors files are typically 2-5 MB each
- You can keep both WAV and safetensors files if you want
- The system will work with either format, safetensors is just faster
