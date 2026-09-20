"""Regression tests for ElevenLabs realtime STT reconnect replay."""

from unittest.mock import Mock

import numpy as np

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT.elevenlabs_realtime_stt_handler import (
    ElevenLabsRealtimeSTTHandler,
    RealtimeSTTConnectionError,
    _TranscriptResult,
)


def test_connection_drop_replays_full_accumulated_turn_once():
    handler = object.__new__(ElevenLabsRealtimeSTTHandler)
    handler._current_turn_id = None
    handler._current_turn_revision = None
    handler._last_sent_samples = 0
    handler._ws = None
    handler._token = None
    handler._session_id = None
    handler.sample_rate = 16_000
    handler.model_id = "scribe_v2_realtime"
    handler.language_code = "en"
    handler.text_output_queue = None
    handler.should_listen = None
    handler.minimum_average_logprob = -0.7
    handler.minimum_word_logprob = -1.5
    handler._ensure_connection = Mock(
        side_effect=[RealtimeSTTConnectionError("received 1000 (OK)"), None]
    )
    handler._disconnect = Mock()
    sent_deltas: list[np.ndarray] = []
    handler._send_audio_and_collect_partials = Mock(
        side_effect=lambda delta: sent_deltas.append(delta.copy()) or []
    )
    handler._commit_and_wait_for_transcript = Mock(return_value=_TranscriptResult(text="I am still here."))
    audio = np.ones(16_000, dtype=np.int16)

    outputs = list(
        handler.process(
            VADAudio(
                audio=audio,
                mode="final",
                turn_id="turn_1",
                turn_revision=0,
            )
        )
    )

    assert handler._ensure_connection.call_count == 2
    handler._disconnect.assert_called_once()
    assert len(sent_deltas) == 1
    assert np.array_equal(sent_deltas[0], audio)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "I am still here."


def test_timestamp_payload_produces_conservative_uncertainty_signal():
    handler = object.__new__(ElevenLabsRealtimeSTTHandler)
    handler.minimum_average_logprob = -0.7
    handler.minimum_word_logprob = -1.5

    result = handler._transcript_result(
        {
            "message_type": "committed_transcript_with_timestamps",
            "text": "Tool call into YouTube.",
            "language_code": "en",
            "words": [
                {"type": "word", "text": "Tool", "logprob": -0.1},
                {"type": "spacing", "text": " "},
                {"type": "word", "text": "YouTube", "logprob": -2.0},
            ],
        }
    )

    assert result.average_logprob == -1.05
    assert result.minimum_logprob == -2.0
    assert handler._uncertainty_reason(result) == "low_average_word_logprob"
