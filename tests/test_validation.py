import pytest

from app.config import Config


@pytest.mark.unit
def test_invalid_model_returns_401_with_suggestions(client):
    resp = client.post(
        '/v1/audio/speech',
        json={
            'model': 'Pocket-TTS',
            'input': 'Hello',
            'voice': 'alba',
            'response_format': 'mp3',
        },
    )
    assert resp.status_code == 401
    data = resp.get_json()
    err = data['error']
    assert err['param'] == 'model'
    assert 'allowed_models' in err['details']
    assert 'suggested_defaults' in err['details']
    assert err['details']['suggested_defaults']['model'] == Config.ALLOWED_MODELS[0]


@pytest.mark.unit
def test_invalid_voice_returns_401_with_suggestions(client):
    resp = client.post(
        '/v1/audio/speech',
        json={
            'model': Config.ALLOWED_MODELS[0],
            'input': 'Hello',
            'voice': 'NOPE',
            'response_format': 'mp3',
        },
    )
    assert resp.status_code == 401
    data = resp.get_json()
    err = data['error']
    assert err['param'] == 'voice'
    assert 'available_voices' in err['details']
    assert 'suggested_defaults' in err['details']
