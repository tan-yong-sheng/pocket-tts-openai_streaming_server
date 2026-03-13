"""
Flask routes for the OpenAI-compatible TTS API.
"""

import re
import threading
import time

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

import json
import uuid

from app.config import Config
from app.logging_config import get_logger
from app.services.audio import (
    convert_audio,
    get_mime_type,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.preprocess import TextPreprocessor
from app.services.tts import get_tts_service

logger = get_logger('routes')

# Cold-start request logging (first request only)
_FIRST_REQUEST_LOGGED = False
_FIRST_REQUEST_LOCK = threading.Lock()

# Create blueprint
api = Blueprint('api', __name__)

# Create text preprocessor instance, some options changed from defaults
text_preprocessor = TextPreprocessor(
    remove_urls=False,
    remove_emails=False,
    remove_html=True,
    remove_hashtags=True,
    remove_mentions=False,
    remove_punctuation=False,
    remove_stopwords=False,
    remove_extra_whitespace=False,
)

_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _split_text_for_tts(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _RE_SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ''

    def _flush_current():
        nonlocal current
        if current:
            chunks.append(current)
            current = ''

    for sentence in sentences:
        if len(sentence) > max_chars:
            _flush_current()
            words = sentence.split()
            buf = ''
            for word in words:
                candidate = word if not buf else f'{buf} {word}'
                if len(candidate) <= max_chars:
                    buf = candidate
                    continue
                if buf:
                    chunks.append(buf)
                if len(word) > max_chars:
                    for i in range(0, len(word), max_chars):
                        chunks.append(word[i : i + max_chars])
                    buf = ''
                else:
                    buf = word
            if buf:
                chunks.append(buf)
            continue

        candidate = sentence if not current else f'{current} {sentence}'
        if len(candidate) <= max_chars:
            current = candidate
        else:
            _flush_current()
            current = sentence

    _flush_current()
    return chunks


@api.route('/')
def home():
    """Serve the web interface."""
    from app.config import Config

    if not Config.UI_ENABLED:
        return (
            jsonify(
                {
                    'service': 'pocket-tts',
                    'status': 'ok',
                    'endpoints': {
                        'health': '/health',
                        'voices': '/v1/voices',
                        'speech': '/v1/audio/speech',
                    },
                }
            ),
            200,
        )

    return render_template('index.html', is_docker=Config.IS_DOCKER)


@api.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.

    Returns service status and basic model info.
    """
    tts = get_tts_service()

    # Validate a built-in voice quickly
    voice_valid, voice_msg = tts.validate_voice('alba')

    return jsonify(
        {
            'status': 'healthy' if tts.is_loaded else 'unhealthy',
            'model_loaded': tts.is_loaded,
            'device': tts.device if tts.is_loaded else None,
            'sample_rate': tts.sample_rate if tts.is_loaded else None,
            'voices_dir': tts.voices_dir,
            'voice_check': {'valid': voice_valid, 'message': voice_msg},
        }
    ), 200 if tts.is_loaded else 503


@api.route('/v1/voices', methods=['GET'])
def list_voices():
    """
    List available voices.

    Returns OpenAI-compatible voice list format.
    """
    tts = get_tts_service()
    voices = tts.list_voices()

    return jsonify(
        {
            'object': 'list',
            'data': [
                {
                    'id': v['id'],
                    'name': v['name'],
                    'object': 'voice',
                    'type': v.get('type', 'builtin'),
                }
                for v in voices
            ],
        }
    )


@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (ignored, for compatibility)
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        stream: boolean (optional) - Enable streaming

    Returns:
        Audio file or streaming audio response
    """
    from flask import current_app

    request_start = time.monotonic()
    request_id = request.headers.get(Config.REQUEST_ID_HEADER) or uuid.uuid4().hex
    data = request.get_json(silent=True)

    if not data:
        return _error_response('Missing JSON body', 400, request_id)

    text = data.get('input')
    if not text:
        return _error_response("Missing 'input' text", 400, request_id)
    if not isinstance(text, str):
        return _error_response("'input' must be a string", 400, request_id)
    if len(text) > Config.MAX_INPUT_CHARS:
        return _error_response(
            f"'input' exceeds max length of {Config.MAX_INPUT_CHARS} characters",
            413,
            request_id,
        )

    voice = data.get('voice', 'alba')
    if not isinstance(voice, str):
        return _error_response("'voice' must be a string", 400, request_id)
    stream_request = data.get('stream', False)

    response_format = data.get('response_format', 'mp3')
    if not isinstance(response_format, str):
        return _error_response("'response_format' must be a string", 400, request_id)
    target_format = validate_format(response_format)

    tts = get_tts_service()

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        return _error_response(
            f"Voice '{voice}' not found",
            400,
            request_id,
            {
                'available_voices': available[:10],  # Limit to first 10
                'hint': 'Use /v1/voices to see all available voices',
            },
        )

    try:
        voice_state = tts.get_voice_state(voice)

        # Check if streaming should be used
        use_streaming = stream_request or current_app.config.get('STREAM_DEFAULT', False)

        # Streaming supports only PCM/WAV today; fall back to file for other formats.
        if use_streaming and target_format not in ('pcm', 'wav'):
            logger.warning(
                "Streaming format '%s' is not supported; returning full file instead.",
                target_format,
            )
            use_streaming = False
        # Check if text preprocessing should be used
        use_text_preprocess = current_app.config.get('TEXT_PREPROCESS_DEFAULT', False)
        # Preprocess text
        if use_text_preprocess:
            # logger.info(f'Preprocessing text: {text}')
            text = text_preprocessor.process(text)
            # logger.info(f'Preprocessed text: {text}')
        chunk_chars = Config.CHUNK_MAX_CHARS
        if Config.CHUNK_CHARS_ALLOW_OVERRIDE:
            requested_chunk_chars = data.get('chunk_chars')
            if isinstance(requested_chunk_chars, int):
                if requested_chunk_chars > 0:
                    chunk_chars = min(requested_chunk_chars, Config.MAX_INPUT_CHARS)
                elif requested_chunk_chars == 0:
                    chunk_chars = 0
            elif requested_chunk_chars is not None:
                logger.warning('Ignoring non-integer chunk_chars override: %s', requested_chunk_chars)
        chunks = _split_text_for_tts(text, chunk_chars)
        if len(chunks) > 1:
            logger.info(
                'Chunking input into %s segments (max %s chars each)',
                len(chunks),
                chunk_chars,
            )
        if use_streaming:
            pre_response_s = time.monotonic() - request_start
            if len(chunks) > 1:
                return _stream_audio_chunks(
                    tts,
                    voice_state,
                    chunks,
                    target_format,
                    request_start,
                    pre_response_s,
                    request_id,
                )
            return _stream_audio(
                tts, voice_state, text, target_format, request_start, pre_response_s, request_id
            )
        if len(chunks) > 1:
            return _generate_file_chunked(
                tts, voice_state, chunks, target_format, request_start, request_id
            )
        return _generate_file(tts, voice_state, text, target_format, request_start, request_id)

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return _error_response(str(e), 400, request_id)
    except Exception as e:
        logger.exception('Generation failed')
        return _error_response('Generation failed', 500, request_id)


def _emit_timing_log(event: str, metrics: dict) -> None:
    payload = {'event': event, **metrics}
    if Config.REQUEST_TIMING_LOG_JSON:
        logger.info(json.dumps(payload, separators=(',', ':')))
    else:
        logger.info('%s: %s', event, payload)


def _log_first_request(metrics: dict) -> None:
    if not Config.COLDSTART_LOG:
        return
    global _FIRST_REQUEST_LOGGED
    with _FIRST_REQUEST_LOCK:
        if _FIRST_REQUEST_LOGGED:
            return
        _FIRST_REQUEST_LOGGED = True
    _emit_timing_log('first_request_timing', metrics)


def _error_response(message: str, status_code: int, request_id: str, extra: dict | None = None):
    payload = {'error': message, 'request_id': request_id}
    if extra:
        payload.update(extra)
    response = jsonify(payload)
    response.status_code = status_code
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _generate_file(
    tts, voice_state, text: str, fmt: str, request_start: float, request_id: str
):
    """Generate complete audio and return as file."""
    t0 = time.time()
    audio_tensor = tts.generate_audio(voice_state, text)
    generation_time = time.time() - t0

    logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')

    convert_t0 = time.time()
    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    convert_time = time.time() - convert_t0
    mimetype = get_mime_type(fmt)
    total_s = time.monotonic() - request_start
    _log_first_request(
        {
            'mode': 'non_stream',
            'format': fmt,
            'text_len': len(text),
            'generation_s': round(generation_time, 4),
            'total_s': round(total_s, 4),
            'request_id': request_id,
        }
    )
    if Config.REQUEST_TIMING_LOG:
        _emit_timing_log(
            'request_timing',
            {
                'mode': 'non_stream',
                'format': fmt,
                'text_len': len(text),
                'generation_s': round(generation_time, 4),
                'convert_s': round(convert_time, 4),
                'total_s': round(total_s, 4),
                'request_id': request_id,
            },
        )

    response = send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _generate_file_chunked(
    tts,
    voice_state,
    chunks: list[str],
    fmt: str,
    request_start: float,
    request_id: str,
):
    """Generate complete audio from chunked text and return as file."""
    import torch

    t0 = time.time()
    tensors = []
    total_chars = 0
    for chunk in chunks:
        total_chars += len(chunk)
        tensors.append(tts.generate_audio(voice_state, chunk))
    generation_time = time.time() - t0

    processed = []
    for tensor in tensors:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        processed.append(tensor)
    audio_tensor = processed[0] if len(processed) == 1 else torch.cat(processed, dim=1)

    convert_t0 = time.time()
    audio_buffer = convert_audio(audio_tensor, tts.sample_rate, fmt)
    convert_time = time.time() - convert_t0
    total_s = time.monotonic() - request_start
    _log_first_request(
        {
            'mode': 'non_stream',
            'format': fmt,
            'text_len': total_chars,
            'chunks': len(chunks),
            'generation_s': round(generation_time, 4),
            'total_s': round(total_s, 4),
            'request_id': request_id,
        }
    )
    if Config.REQUEST_TIMING_LOG:
        _emit_timing_log(
            'request_timing',
            {
                'mode': 'non_stream',
                'format': fmt,
                'text_len': total_chars,
                'chunks': len(chunks),
                'generation_s': round(generation_time, 4),
                'convert_s': round(convert_time, 4),
                'total_s': round(total_s, 4),
                'request_id': request_id,
            },
        )

    response = send_file(
        audio_buffer, mimetype=get_mime_type(fmt), as_attachment=True, download_name=f'speech.{fmt}'
    )
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    return response


def _stream_audio(
    tts,
    voice_state,
    text: str,
    fmt: str,
    request_start: float,
    pre_response_s: float,
    request_id: str,
):
    """Stream audio chunks."""
    # Normalize streaming format: we always emit PCM bytes, optionally wrapped
    # in a WAV container. For non-PCM/WAV formats (e.g. mp3, opus), coerce to
    # raw PCM to avoid mismatched content-type vs. payload.
    stream_fmt = fmt
    if stream_fmt not in ('pcm', 'wav'):
        logger.warning(
            "Requested streaming format '%s' is not supported for streaming; "
            "falling back to 'pcm'.",
            stream_fmt,
        )
        stream_fmt = 'pcm'

    first_log = {'done': False}
    counters = {'bytes': 0, 'chunks': 0}

    try:
        stream = tts.generate_audio_stream(voice_state, text)
        first_chunk = next(stream, None)
    except Exception:
        logger.exception('Streaming generation failed before first chunk')
        return _error_response('Streaming generation failed', 500, request_id)

    if first_chunk is None:
        logger.error('Streaming generation produced no audio chunks')
        return _error_response('Streaming generation produced no audio', 500, request_id)

    def _maybe_log_first_bytes():
        if first_log['done']:
            return
        first_log['done'] = True
        first_bytes_s = time.monotonic() - request_start
        metrics = {
            'mode': 'stream',
            'format': stream_fmt,
            'text_len': len(text),
            'pre_response_s': round(pre_response_s, 4),
            'first_bytes_s': round(first_bytes_s, 4),
            'request_id': request_id,
        }
        _log_first_request(metrics)
        if Config.TTFA_LOG:
            _emit_timing_log('ttfa', metrics)

    def stream_with_header():
        stream_start = time.monotonic()
        try:
            # Yield WAV header first if streaming as WAV
            if stream_fmt == 'wav':
                _maybe_log_first_bytes()
                header = write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
                counters['bytes'] += len(header)
                yield header

            _maybe_log_first_bytes()
            chunk_bytes = tensor_to_pcm_bytes(first_chunk)
            counters['chunks'] += 1
            counters['bytes'] += len(chunk_bytes)
            yield chunk_bytes

            for chunk_tensor in stream:
                _maybe_log_first_bytes()
                chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                counters['chunks'] += 1
                counters['bytes'] += len(chunk_bytes)
                yield chunk_bytes
        except Exception:
            logger.exception('Streaming generation failed mid-stream')
        finally:
            if Config.REQUEST_TIMING_LOG:
                total_s = time.monotonic() - stream_start
                _emit_timing_log(
                    'request_timing',
                    {
                        'mode': 'stream',
                        'format': stream_fmt,
                        'text_len': len(text),
                        'pre_response_s': round(pre_response_s, 4),
                        'total_s': round(total_s, 4),
                        'chunks': counters['chunks'],
                        'bytes': counters['bytes'],
                        'request_id': request_id,
                    },
                )

    mimetype = get_mime_type(stream_fmt)

    response = Response(stream_with_context(stream_with_header()), mimetype=mimetype)
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def _stream_audio_chunks(
    tts,
    voice_state,
    chunks: list[str],
    fmt: str,
    request_start: float,
    pre_response_s: float,
    request_id: str,
):
    """Stream audio chunks from chunked text input."""
    stream_fmt = fmt
    if stream_fmt not in ('pcm', 'wav'):
        logger.warning(
            "Requested streaming format '%s' is not supported for streaming; "
            "falling back to 'pcm'.",
            stream_fmt,
        )
        stream_fmt = 'pcm'

    first_log = {'done': False}
    counters = {'bytes': 0, 'chunks': 0}

    try:
        stream = tts.generate_audio_stream(voice_state, chunks[0])
        first_chunk = next(stream, None)
    except Exception:
        logger.exception('Streaming generation failed before first chunk')
        return _error_response('Streaming generation failed', 500, request_id)

    if first_chunk is None:
        logger.error('Streaming generation produced no audio chunks')
        return _error_response('Streaming generation produced no audio', 500, request_id)

    def _maybe_log_first_bytes():
        if first_log['done']:
            return
        first_log['done'] = True
        first_bytes_s = time.monotonic() - request_start
        metrics = {
            'mode': 'stream',
            'format': stream_fmt,
            'text_len': sum(len(chunk) for chunk in chunks),
            'chunks': len(chunks),
            'pre_response_s': round(pre_response_s, 4),
            'first_bytes_s': round(first_bytes_s, 4),
            'request_id': request_id,
        }
        _log_first_request(metrics)
        if Config.TTFA_LOG:
            _emit_timing_log('ttfa', metrics)

    def stream_with_header():
        stream_start = time.monotonic()
        try:
            if stream_fmt == 'wav':
                _maybe_log_first_bytes()
                header = write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
                counters['bytes'] += len(header)
                yield header

            _maybe_log_first_bytes()
            chunk_bytes = tensor_to_pcm_bytes(first_chunk)
            counters['chunks'] += 1
            counters['bytes'] += len(chunk_bytes)
            yield chunk_bytes

            for chunk_tensor in stream:
                _maybe_log_first_bytes()
                chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                counters['chunks'] += 1
                counters['bytes'] += len(chunk_bytes)
                yield chunk_bytes

            for chunk_text in chunks[1:]:
                for chunk_tensor in tts.generate_audio_stream(voice_state, chunk_text):
                    _maybe_log_first_bytes()
                    chunk_bytes = tensor_to_pcm_bytes(chunk_tensor)
                    counters['chunks'] += 1
                    counters['bytes'] += len(chunk_bytes)
                    yield chunk_bytes
        except Exception:
            logger.exception('Streaming generation failed mid-stream')
        finally:
            if Config.REQUEST_TIMING_LOG:
                total_s = time.monotonic() - stream_start
                _emit_timing_log(
                    'request_timing',
                    {
                        'mode': 'stream',
                        'format': stream_fmt,
                        'text_len': sum(len(chunk) for chunk in chunks),
                        'chunks': len(chunks),
                        'pre_response_s': round(pre_response_s, 4),
                        'total_s': round(total_s, 4),
                        'chunks_out': counters['chunks'],
                        'bytes': counters['bytes'],
                        'request_id': request_id,
                    },
                )

    mimetype = get_mime_type(stream_fmt)

    response = Response(stream_with_context(stream_with_header()), mimetype=mimetype)
    response.headers[Config.REQUEST_ID_HEADER] = request_id
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Accel-Buffering'] = 'no'
    return response
