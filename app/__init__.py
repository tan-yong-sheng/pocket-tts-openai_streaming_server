"""
PocketTTS OpenAI-Compatible Server

Flask application factory and initialization.
"""

import uuid

from flask import Flask, jsonify, request

from app.config import Config
from app.logging_config import get_logger, setup_logging


def create_app(config_overrides: dict = None) -> Flask:
    """
    Application factory for creating the Flask app.

    Args:
        config_overrides: Optional dictionary of config values to override

    Returns:
        Configured Flask application
    """
    # Setup logging first
    setup_logging()
    logger = get_logger()

    # Create Flask app with correct paths
    app = Flask(
        __name__,
        template_folder=Config.get_template_folder(),
        static_folder=Config.get_static_folder(),
    )

    # Apply overrides
    if config_overrides:
        app.config.update(config_overrides)

    @app.before_request
    def _auth_guard():
        if request.method == 'OPTIONS':
            return None
        if not Config.is_auth_enabled():
            return None

        auth_header = request.headers.get('Authorization')
        token = _extract_bearer_token(auth_header)
        if not token or not Config.is_valid_token(token):
            request_id = request.headers.get(Config.REQUEST_ID_HEADER) or uuid.uuid4().hex
            logger.warning('Authentication failed for %s %s', request.method, request.path)
            response = jsonify(
                {'error': 'Unauthorized', 'message': 'Invalid or missing authentication token'}
            )
            response.status_code = 401
            response.headers['WWW-Authenticate'] = 'Bearer'
            response.headers[Config.REQUEST_ID_HEADER] = request_id
            response.headers['Cache-Control'] = 'no-store'
            return response
        return None

    # Register blueprints
    from app.routes import api

    app.register_blueprint(api)

    logger.info('Flask application created')

    return app


def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    token = parts[1].strip()
    return token if token else None


def init_tts_service(model_path: str = None, voices_dir: str = None) -> None:
    """
    Initialize the TTS service with model and voices.

    Args:
        model_path: Optional path to model file
        voices_dir: Optional path to voices directory
    """
    from app.services.tts import get_tts_service

    logger = get_logger()
    tts = get_tts_service()

    # Configure torch thread usage if requested
    try:
        import torch

        if Config.TORCH_NUM_THREADS:
            torch.set_num_threads(Config.TORCH_NUM_THREADS)
        if Config.TORCH_NUM_INTEROP_THREADS:
            torch.set_num_interop_threads(Config.TORCH_NUM_INTEROP_THREADS)
    except Exception as exc:
        logger.warning(f'Could not configure torch threads: {exc}')

    # Load model
    tts.load_model(model_path)

    # Set voices directory
    if voices_dir:
        tts.set_voices_dir(voices_dir)
    else:
        # Check for bundled voices
        bundle_voices, _ = Config.get_bundle_paths()
        if bundle_voices:
            tts.set_voices_dir(bundle_voices)

    logger.info('TTS service initialized')
