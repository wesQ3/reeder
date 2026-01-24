# Job File Format

Jobs are submitted as JSON files dropped into the `inbox/` directory. The filename should be unique (timestamp-based recommended).

## Naming Convention

```
{timestamp}-{slug}.json
```

Example: `1737654321-interesting-article.json`

The timestamp ensures FIFO ordering. The slug is optional but helps identify jobs at a glance.

## Job Types

### URL Job (fetch and extract text)

```json
{
  "type": "url",
  "url": "https://example.com/article",
  "title": "Article Title",
  "extractor": "auto",
  "voice": "default",
  "temperature": 0.8
}
```

### Text Job (direct text input)

```json
{
  "type": "text",
  "text": "The full text to be converted to speech...",
  "title": "My Custom Text",
  "source_url": "https://example.com/original",
  "voice": "default",
  "temperature": 0.8
}
```

## Field Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `type` | Yes | - | `"url"` or `"text"` |
| `url` | Yes (url type) | - | URL to fetch and extract text from |
| `text` | Yes (text type) | - | Raw text to convert |
| `title` | No | Extracted or "Untitled" | Episode title in RSS feed |
| `source_url` | No | - | Original source URL (for text type, used in feed) |
| `extractor` | No | `"auto"` | Text extraction method (see below) |
| `voice` | No | `"default"` | Voice name or path (see config.toml) |
| `temperature` | No | `0.8` | TTS temperature (0.0-1.0, higher = more expressive) |
| `description` | No | - | Episode description for RSS feed |

## Extractor Options

| Value | Description |
|-------|-------------|
| `"auto"` | Use trafilatura for smart article extraction |
| `"trafilatura"` | Explicit trafilatura extraction |
| `"readability"` | Use readability-cli (requires Node.js) |
| `{"pup": "selector"}` | Use pup with custom CSS selector |

### Custom pup selector example

```json
{
  "type": "url",
  "url": "https://news.example.com/story/123",
  "title": "Breaking News Story",
  "extractor": {
    "pup": "article div[data-component=\"text-block\"] text{}"
  }
}
```

## Completed Job Metadata

After processing, the job file is moved to `done/` with additional fields:

```json
{
  "type": "url",
  "url": "https://example.com/article",
  "title": "Article Title",
  "extractor": "auto",
  "voice": "default",
  "temperature": 0.8,
  "_completed": {
    "timestamp": "2026-01-23T10:30:00Z",
    "audio_file": "1737654321-interesting-article.opus",
    "audio_size": 1234567,
    "duration_seconds": 342,
    "guid": "1737654321-interesting-article"
  }
}
```

## Quick Examples

### Submit a URL job via command line

```bash
cat > inbox/$(date +%s)-article.json << 'EOF'
{
  "type": "url",
  "url": "https://example.com/interesting-article"
}
EOF
```

### Submit text directly

```bash
cat > inbox/$(date +%s)-custom.json << 'EOF'
{
  "type": "text",
  "title": "My Notes",
  "text": "These are my notes that I want to listen to later..."
}
EOF
```
