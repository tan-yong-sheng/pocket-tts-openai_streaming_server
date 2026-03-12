# Modal Deployment Guide (Scale-to-Zero)

This guide mirrors the `model2vec-py-api-server` Modal setup and is optimized
for occasional use with scale-to-zero and memory snapshots.

## 1) What You Get

- Single Modal deployment serving the Flask app.
- Scale-to-zero by default (`min_containers=0`).
- Volume-cached model weights to avoid repeated downloads.
- Optional voices volume for custom voice packs.
- Memory snapshots to reduce cold-start latency after the first deploy.

## 2) Files

- `infra/modal/modal_config.py` — Modal entrypoint
- `infra/modal/README.md` — quick entry point
- `app/*`, `templates/*`, `static/*` — Flask app

## 3) One-Time Setup

```bash
# Install Modal CLI (if not installed)
# https://modal.com/docs/guide

# Create a volume for model caching
modal volume create pocket-tts-models
```

Optional voices volume (only if you want to store custom voices in a volume):

```bash
modal volume create pocket-tts-voices
```

## 4) Optional Environment Variables

You can either export variables in your shell **or** create
`infra/modal/.env.modal` (auto-loaded by `modal_config.py`).

Set these in your shell before running Modal commands:

```bash
# Or copy the example file and edit it:
cp infra/modal/.env.modal.example infra/modal/.env.modal

# Override Modal app name
export POCKET_TTS_MODAL_APP=pocket-tts

# Where to cache downloaded model files
export MODEL_VOLUME_NAME=pocket-tts-models

# Optional: mount a voices volume at /voices
export VOICES_VOLUME_NAME=pocket-tts-voices

# Optional: attach Modal secrets
export AUTH_SECRET_NAME=pocket-tts-auth
export HF_SECRET_NAME=pocket-tts-hf

# Optional: force a specific model path or variant
export POCKET_TTS_MODEL_PATH=

# Optional: override voices directory in the container
export POCKET_TTS_VOICES_DIR=

# Optional: source path to sync voices into the voices volume
export POCKET_TTS_VOICES_SRC=/app/voices

# Optional: enable cold-start timing logs
export POCKET_TTS_COLDSTART_LOG=true

# Optional: enable per-request timing logs
export POCKET_TTS_REQUEST_TIMING_LOG=true
export POCKET_TTS_REQUEST_TIMING_LOG_JSON=true

# Optional: disable the web UI at /
export POCKET_TTS_UI_ENABLED=false

# Optional: avoid torch.inference_mode errors in Pocket-TTS
export POCKET_TTS_DISABLE_INFERENCE_MODE=true
```

### Auth tokens (optional, recommended via Modal Secret)

Use a Modal secret so tokens are injected at runtime and not baked into images.
Avoid putting `AUTHENTICATION_ALLOWED_TOKENS` directly in `infra/modal/.env.modal`.

```bash
modal secret create pocket-tts-auth \
  AUTHENTICATION_ALLOWED_TOKENS=token1,token2
```

## 5) Download the Model to the Volume (One-Time)

```bash
modal run infra/modal/modal_config.py::download_models
```

This will download and cache the Pocket-TTS model into the Modal volume.

## 6) Sync Voices to the Volume (Optional, One-Time)

If you set `VOICES_VOLUME_NAME`, you can pre-load custom voices into the
voices volume. By default, this copies from `/app/voices` in the image.

```bash
modal run infra/modal/modal_config.py::sync_voices
```

## 7) Deploy

```bash
modal deploy infra/modal/modal_config.py
```

## 8) Quick Tests

```bash
# Replace with your deployed URL
APP_URL="https://<your-app>.modal.run"

curl -sS "$APP_URL/health"

curl -sS "$APP_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello","voice":"alba","response_format":"wav"}' \
  -o /tmp/tts.wav
```

## 9) Notes

- Scale-to-zero is enabled (`min_containers=0`), so cold starts are expected.
- Memory snapshots reduce cold-start time after the first deploy.
- The voices directory defaults to `/app/voices` (bundled with the image).
- If `VOICES_VOLUME_NAME` is set, `/voices` is mounted and used instead.
