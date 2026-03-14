import threading
import sys
from pathlib import Path

import pytest
import torch
from werkzeug.serving import make_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.config import Config
import app.routes as routes


@pytest.fixture()
def client():
    Config.AUTHENTICATION_ALLOWED_TOKENS = None
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


class _DummyTTSService:
    sample_rate = 24000

    def validate_voice(self, voice: str):
        return True, None

    def list_voices(self):
        return [{'id': 'alba', 'name': 'alba', 'type': 'builtin'}]

    def get_voice_state(self, voice: str):
        return {'id': voice}

    def generate_audio(self, voice_state, text: str, **_kwargs):
        # 0.1s of silence at 24kHz
        return torch.zeros((1, 2400), dtype=torch.float32)

    def generate_audio_stream(self, voice_state, text: str, **kwargs):
        yield self.generate_audio(voice_state, text, **kwargs)


@pytest.fixture()
def openai_base_url():
    Config.AUTHENTICATION_ALLOWED_TOKENS = None
    dummy = _DummyTTSService()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(routes, 'get_tts_service', lambda: dummy)

    app = create_app()
    app.config.update(TESTING=True)

    server = make_server('127.0.0.1', 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f'http://127.0.0.1:{server.server_port}/v1'
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join()
        monkeypatch.undo()
