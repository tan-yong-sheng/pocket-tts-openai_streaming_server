"""
Configuration management for PocketTTS OpenAI Server.
Loads settings from environment variables with sensible defaults.
"""

import os
import sys
from pathlib import Path


def _int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _csv_env(name: str) -> list[str] | None:
    value = os.environ.get(name, '')
    if not value:
        return None
    tokens = [item.strip() for item in value.split(',') if item.strip()]
    return tokens or None


def get_base_path() -> Path:
    """Get the base path for the application, handling PyInstaller frozen state."""
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            # One-file mode
            return Path(sys._MEIPASS)
        else:
            # One-dir mode
            return Path(sys.executable).parent
    return Path(__file__).parent.parent


class Config:
    """Application configuration loaded from environment variables."""

    # Base paths
    BASE_PATH = get_base_path()
    IS_FROZEN = getattr(sys, 'frozen', False)

    # Server settings
    HOST = os.environ.get('POCKET_TTS_HOST', '0.0.0.0')
    PORT = int(os.environ.get('POCKET_TTS_PORT', '49112'))

    # Model settings
    MODEL_NAME = os.environ.get('POCKET_TTS_MODEL_NAME', 'pocket-tts')
    MODEL_PATH = os.environ.get('POCKET_TTS_MODEL_PATH', None)
    DEFAULT_VOICE = os.environ.get(
        'POCKET_TTS_DEFAULT_VOICE', 'hf://kyutai/tts-voices/alba-mackenna/casual.wav'
    )

    # Voice directory
    VOICES_DIR = os.environ.get('POCKET_TTS_VOICES_DIR', None)

    # Text preprocessing default
    TEXT_PREPROCESS_DEFAULT = (
        os.environ.get('POCKET_TTS_TEXT_PREPROCESS_DEFAULT', 'false').lower() == 'true'
    )

    # Docker detection
    @staticmethod
    def _is_docker() -> bool:
        """Detect if running in a Docker container."""
        # Check for .dockerenv file (most reliable)
        if os.path.exists('/.dockerenv'):
            return True
        # Check cgroup for docker/containerd references
        try:
            with open('/proc/1/cgroup') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (FileNotFoundError, PermissionError):
            return False

    IS_DOCKER = _is_docker.__func__()

    # Logging
    LOG_LEVEL = os.environ.get('POCKET_TTS_LOG_LEVEL', 'INFO')
    LOG_DIR = os.environ.get('POCKET_TTS_LOG_DIR', str(BASE_PATH / 'logs'))
    LOG_FILE = os.environ.get('POCKET_TTS_LOG_FILE', 'pocket_tts.log')
    LOG_MAX_BYTES = int(os.environ.get('POCKET_TTS_LOG_MAX_BYTES', str(10 * 1024 * 1024)))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('POCKET_TTS_LOG_BACKUP_COUNT', '5'))
    COLDSTART_LOG = os.environ.get('POCKET_TTS_COLDSTART_LOG', 'false').lower() == 'true'
    REQUEST_TIMING_LOG = (
        os.environ.get('POCKET_TTS_REQUEST_TIMING_LOG', 'false').lower() == 'true'
    )
    REQUEST_TIMING_LOG_JSON = (
        os.environ.get('POCKET_TTS_REQUEST_TIMING_LOG_JSON', 'false').lower() == 'true'
    )
    TTFA_LOG = os.environ.get('POCKET_TTS_TTFA_LOG', 'false').lower() == 'true'
    UI_ENABLED = os.environ.get('POCKET_TTS_UI_ENABLED', 'true').lower() == 'true'
    DISABLE_INFERENCE_MODE = (
        os.environ.get('POCKET_TTS_DISABLE_INFERENCE_MODE', 'false').lower() == 'true'
    )
    MAX_INPUT_CHARS = _int_env('POCKET_TTS_MAX_INPUT_CHARS', 2000) or 2000
    REQUEST_ID_HEADER = os.environ.get('POCKET_TTS_REQUEST_ID_HEADER', 'X-Request-ID')
    TORCH_NUM_THREADS = _int_env('POCKET_TTS_TORCH_THREADS')
    TORCH_NUM_INTEROP_THREADS = _int_env('POCKET_TTS_TORCH_INTEROP_THREADS')
    AUTHENTICATION_ALLOWED_TOKENS = _csv_env('AUTHENTICATION_ALLOWED_TOKENS')
    CHUNK_MAX_CHARS = _int_env('POCKET_TTS_CHUNK_CHARS', 0) or 0
    CHUNK_CHARS_ALLOW_OVERRIDE = (
        os.environ.get('POCKET_TTS_CHUNK_CHARS_ALLOW_OVERRIDE', 'false').lower() == 'true'
    )

    @classmethod
    def is_auth_enabled(cls) -> bool:
        return bool(cls.AUTHENTICATION_ALLOWED_TOKENS)

    @classmethod
    def is_valid_token(cls, token: str) -> bool:
        if not cls.AUTHENTICATION_ALLOWED_TOKENS:
            return True
        return token in cls.AUTHENTICATION_ALLOWED_TOKENS

    # Built-in voice mappings (these are resolved by pocket-tts internally)
    BUILTIN_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

    ALLOWED_MODELS = _csv_env('POCKET_TTS_ALLOWED_MODELS') or [MODEL_NAME]

    # Supported audio extensions for custom voices
    VOICE_EXTENSIONS = ('.wav', '.mp3', '.flac', '.safetensors')

    @classmethod
    def get_bundle_paths(cls) -> tuple:
        """Get bundled paths for frozen executables."""
        if cls.IS_FROZEN:
            voices_dir = cls.BASE_PATH / 'voices'
            model_path = cls.BASE_PATH / 'model' / 'b6369a24.yaml'
            return (
                str(voices_dir) if voices_dir.is_dir() else None,
                str(model_path) if model_path.is_file() else None,
            )
        return None, None

    @classmethod
    def get_template_folder(cls) -> str:
        """Get the templates folder path."""
        return str(cls.BASE_PATH / 'templates')

    @classmethod
    def get_static_folder(cls) -> str:
        """Get the static files folder path."""
        return str(cls.BASE_PATH / 'static')
