from pathlib import Path

import pytest
from openai import OpenAI


def _write_audio_response(response, output_path: Path) -> None:
    if hasattr(response, 'write_to_file'):
        response.write_to_file(output_path)
        return
    if hasattr(response, 'read'):
        data = response.read()
    elif hasattr(response, 'content'):
        data = response.content
    else:
        data = bytes(response)
    output_path.write_bytes(data)


@pytest.mark.integration
def test_openai_sdk_streaming_speech(openai_base_url, tmp_path):
    client = OpenAI(base_url=openai_base_url, api_key='sk-test')
    output_path = tmp_path / 'speech.wav'

    with client.audio.speech.with_streaming_response.create(
        model='pocket-tts',
        voice='alba',
        input='Hello from the OpenAI SDK.',
        response_format='wav',
        stream_format='audio',
    ) as response:
        response.stream_to_file(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 44


@pytest.mark.integration
def test_openai_sdk_non_streaming_speech(openai_base_url, tmp_path):
    client = OpenAI(base_url=openai_base_url, api_key='sk-test')
    output_path = tmp_path / 'speech.wav'

    response = client.audio.speech.create(
        model='pocket-tts',
        voice='alba',
        input='Hello from the OpenAI SDK (non-stream).',
        response_format='wav',
    )
    _write_audio_response(response, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
