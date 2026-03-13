from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import modal


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export ') :].lstrip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except Exception as exc:
        print(f'Failed to load env file {path}: {exc}', flush=True)


_load_env_file(Path(__file__).with_name('.env.modal'))

APP_NAME = os.environ.get('POCKET_TTS_MODAL_APP', 'pocket-tts')
MODEL_VOLUME_NAME = os.environ.get('MODEL_VOLUME_NAME', 'pocket-tts-models')
MODEL_VOLUME_PATH = '/models'
HF_CACHE_PATH = f'{MODEL_VOLUME_PATH}/hf'

VOICES_VOLUME_NAME = os.environ.get('VOICES_VOLUME_NAME', '').strip()
VOICES_VOLUME_PATH = '/voices'
VOICES_SRC_PATH = os.environ.get('POCKET_TTS_VOICES_SRC', '/app/voices')

COLDSTART_LOG = os.environ.get('POCKET_TTS_COLDSTART_LOG', 'false').lower() == 'true'
_BOOT_T0 = time.monotonic()

AUTH_SECRET_NAME = os.environ.get('AUTH_SECRET_NAME', '').strip()
HF_SECRET_NAME = os.environ.get('HF_SECRET_NAME', '').strip()

app = modal.App(APP_NAME)

image_env = {
    'PYTHONPATH': '/app',
    'HF_HOME': HF_CACHE_PATH,
    'HUGGINGFACE_HUB_CACHE': HF_CACHE_PATH,
    'TRANSFORMERS_CACHE': HF_CACHE_PATH,
    'MODEL_VOLUME_NAME': MODEL_VOLUME_NAME,
    'VOICES_VOLUME_NAME': VOICES_VOLUME_NAME,
    'AUTH_SECRET_NAME': AUTH_SECRET_NAME,
    'HF_SECRET_NAME': HF_SECRET_NAME,
}

_offline = os.environ.get('POCKET_TTS_HF_OFFLINE', '').strip()
if _offline:
    image_env['HF_HUB_OFFLINE'] = _offline
    image_env['TRANSFORMERS_OFFLINE'] = _offline
elif os.environ.get('POCKET_TTS_MODEL_PATH'):
    image_env.setdefault('HF_HUB_OFFLINE', '1')
    image_env.setdefault('TRANSFORMERS_OFFLINE', '1')

# Pass through non-secret POCKET_TTS_* settings from the local environment
_PASSTHROUGH_ENV_KEYS = (
    'POCKET_TTS_HOST',
    'POCKET_TTS_PORT',
    'POCKET_TTS_VOICES_DIR',
    'POCKET_TTS_MODEL_PATH',
    'POCKET_TTS_STREAM_DEFAULT',
    'POCKET_TTS_TEXT_PREPROCESS_DEFAULT',
    'POCKET_TTS_LOG_LEVEL',
    'POCKET_TTS_LOG_DIR',
    'POCKET_TTS_LOG_FILE',
    'POCKET_TTS_LOG_MAX_BYTES',
    'POCKET_TTS_LOG_BACKUP_COUNT',
    'POCKET_TTS_COLDSTART_LOG',
    'POCKET_TTS_REQUEST_TIMING_LOG',
    'POCKET_TTS_REQUEST_TIMING_LOG_JSON',
    'POCKET_TTS_MAX_INPUT_CHARS',
    'POCKET_TTS_REQUEST_ID_HEADER',
    'POCKET_TTS_TORCH_THREADS',
    'POCKET_TTS_TORCH_INTEROP_THREADS',
    'POCKET_TTS_UI_ENABLED',
    'POCKET_TTS_DISABLE_INFERENCE_MODE',
    'POCKET_TTS_HF_OFFLINE',
)
for key in _PASSTHROUGH_ENV_KEYS:
    value = os.environ.get(key)
    if value is not None and value != '':
        image_env[key] = value

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements.txt')
    .env(image_env)
    .add_local_dir('app', remote_path='/app/app')
    .add_local_dir('templates', remote_path='/app/templates')
    .add_local_dir('static', remote_path='/app/static')
)
if not VOICES_VOLUME_NAME:
    image = image.add_local_dir('voices', remote_path='/app/voices')

sync_image = image
if VOICES_VOLUME_NAME:
    sync_image = image.add_local_dir('voices', remote_path='/app/voices')

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
voices_volume: modal.Volume | None = None

volumes: dict[str, modal.Volume] = {MODEL_VOLUME_PATH: model_volume}
if VOICES_VOLUME_NAME:
    voices_volume = modal.Volume.from_name(VOICES_VOLUME_NAME, create_if_missing=True)
    volumes[VOICES_VOLUME_PATH] = voices_volume


def _ensure_env_defaults() -> None:
    if VOICES_VOLUME_NAME:
        os.environ.setdefault('POCKET_TTS_VOICES_DIR', VOICES_VOLUME_PATH)
    else:
        os.environ.setdefault('POCKET_TTS_VOICES_DIR', '/app/voices')


def _get_model_path() -> str | None:
    value = os.environ.get('POCKET_TTS_MODEL_PATH')
    return value or None


def _get_voices_dir() -> str | None:
    value = os.environ.get('POCKET_TTS_VOICES_DIR')
    return value or None


def _log_coldstart(message: str) -> None:
    if not COLDSTART_LOG:
        return
    elapsed = time.monotonic() - _BOOT_T0
    print(f'[coldstart +{elapsed:.3f}s] {message}', flush=True)


def _get_secrets() -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    if AUTH_SECRET_NAME:
        secrets.append(modal.Secret.from_name(AUTH_SECRET_NAME))
    if HF_SECRET_NAME:
        secrets.append(modal.Secret.from_name(HF_SECRET_NAME))
    return secrets


@app.function(
    image=image,
    volumes=volumes,
    cpu=1,
    memory=4096,
    timeout=900,
    secrets=_get_secrets(),
)
def download_models() -> None:
    """One-time model download to the persistent Modal volume."""

    _ensure_env_defaults()
    os.chdir('/app')

    from pocket_tts import TTSModel

    model_path = _get_model_path()
    if model_path:
        TTSModel.load_model(config=model_path)
    else:
        TTSModel.load_model()

    model_volume.commit()


@app.function(
    image=sync_image,
    volumes=volumes,
    cpu=1,
    memory=2048,
    timeout=900,
    secrets=_get_secrets(),
)
def sync_voices() -> None:
    """One-time copy of voices into the voices volume (if configured)."""

    if not voices_volume:
        raise RuntimeError('VOICES_VOLUME_NAME is not set; no voices volume is mounted.')

    _ensure_env_defaults()
    os.chdir('/app')

    src_root = Path(VOICES_SRC_PATH)
    dst_root = Path(VOICES_VOLUME_PATH)

    if not src_root.exists():
        raise RuntimeError(f'Voices source path does not exist: {src_root}')

    for path in src_root.rglob('*'):
        if not path.is_file():
            continue
        rel_path = path.relative_to(src_root)
        target = dst_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    voices_volume.commit()


@app.cls(
    image=image,
    volumes=volumes,
    cpu=2,
    memory=4096,
    timeout=900,
    scaledown_window=60,
    min_containers=0,
    max_containers=10,
    enable_memory_snapshot=True,
    secrets=_get_secrets(),
)
class PocketTTSApp:
    @modal.enter(snap=True)
    def _warmup(self) -> None:
        _log_coldstart('warmup start')
        _ensure_env_defaults()
        os.chdir('/app')

        from app import init_tts_service

        init_tts_service(model_path=_get_model_path(), voices_dir=_get_voices_dir())
        try:
            from app.services.tts import get_tts_service

            tts = get_tts_service()
            warmup_voice = os.environ.get('POCKET_TTS_WARMUP_VOICE', 'alba')
            warmup_text = os.environ.get('POCKET_TTS_WARMUP_TEXT', 'Hello')
            if warmup_text:
                voice_state = tts.get_voice_state(warmup_voice)
                tts.generate_audio(voice_state, warmup_text)
                _log_coldstart('warmup inference complete')
        except Exception as exc:
            _log_coldstart(f'warmup inference failed: {exc}')
        _log_coldstart('warmup complete')

    @modal.wsgi_app()
    def flask_app(self):
        _log_coldstart('flask_app start')
        _ensure_env_defaults()
        os.chdir('/app')

        from app import create_app

        app_instance = create_app()
        _log_coldstart('flask_app ready')
        return app_instance
