# Project Overview

**PocketTTS OpenAI-Compatible Server** wraps [pocket-tts](https://github.com/kyutai-labs/pocket-tts) to provide OpenAI-compatible TTS endpoints. Any OpenAI TTS client can use this for local, CPU-based text-to-speech.

## Why This Exists

The official `pocket-tts` has a FastAPI server with `/tts` endpoint, but it's **not OpenAI API compatible**. This project adds:

- `/v1/audio/speech` matching OpenAI's schema
- `/v1/voices` for voice listing
- Docker deployment with voice mounting
- Windows executable distribution

## Architecture

```
server.py                    # Entry point, CLI, starts Waitress
    └── app/__init__.py      # Flask app factory
        ├── app/routes.py    # API endpoints
        ├── app/config.py    # Environment config
        └── app/services/
            ├── tts.py       # TTSService: model, voice cache
            └── audio.py     # Format conversion, streaming
```

## Key Files

| File                  | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| `server.py`           | Entry point, CLI args, Waitress server                           |
| `app/routes.py`       | HTTP endpoints: `/`, `/health`, `/v1/voices`, `/v1/audio/speech` |
| `app/services/tts.py` | Model loading, voice caching, generation                         |
| `app/config.py`       | Environment variables, path resolution                           |

## API Endpoints

| Endpoint           | Method | Purpose                             |
| ------------------ | ------ | ----------------------------------- |
| `/`                | GET    | Web UI                              |
| `/health`          | GET    | Health check for containers         |
| `/v1/voices`       | GET    | List voices                         |
| `/v1/audio/speech` | POST   | Generate speech (OpenAI-compatible) |

### Speech Request

```json
{
	"model": "tts-1",
	"input": "Text to speak",
	"voice": "alba",
	"response_format": "mp3"
}
```

## Configuration (Environment Variables)

| Variable                    | Default   | Purpose            |
| --------------------------- | --------- | ------------------ |
| `POCKET_TTS_HOST`           | `0.0.0.0` | Bind address       |
| `POCKET_TTS_PORT`           | `49112`   | Port               |
| `POCKET_TTS_VOICES_DIR`     | None      | Custom voices path |
| `POCKET_TTS_LOG_LEVEL`      | `INFO`    | Log verbosity      |

## Voice Resolution Order

1. Built-in names (`alba`, `marius`, etc.) → pass to pocket-tts
2. HuggingFace models (`hf://` URLs) → pass to pocket-tts
3. Files in `POCKET_TTS_VOICES_DIR`
4. Absolute paths
5. Fallback to pocket-tts

**Security Note**: HTTP/HTTPS URLs are blocked to prevent SSRF attacks. Only `hf://` URLs are allowed for remote models.

## Development

```bash
# Install
pip install -r requirements.txt

# Run
python server.py --log-level DEBUG

# Test
curl http://localhost:49112/health
curl -X POST http://localhost:49112/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": "alba"}' -o test.mp3
```

## Code Style

- Linter/formatter: `ruff` (config in `pyproject.toml`)
- Line length: 100
- Single quotes

## Deployment

- **Python**: `python server.py`
- **Docker**: `docker compose up -d`
- **Windows EXE**: Built via GitHub Actions on release tags
