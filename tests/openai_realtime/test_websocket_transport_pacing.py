import asyncio
from time import monotonic
from types import SimpleNamespace

from speech_to_speech.api.openai_realtime.transports import WebSocketTransport


class _RecordingTransport(WebSocketTransport):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace())  # type: ignore[arg-type]
        self.sent: list[object] = []

    async def send_events(self, events):
        self.sent.extend(events)


class _Service:
    def encode_audio_chunk(self, session_id: str, pcm: bytes):
        return [(session_id, len(pcm))]


def test_websocket_audio_is_paced_by_pcm_duration() -> None:
    async def run() -> tuple[float, _RecordingTransport]:
        transport = _RecordingTransport()
        service = _Service()
        pcm_100ms = bytes(3_200)
        started = monotonic()
        await transport.send_audio_chunk(service, "session", pcm_100ms)  # type: ignore[arg-type]
        await transport.send_audio_chunk(service, "session", pcm_100ms)  # type: ignore[arg-type]
        return monotonic() - started, transport

    elapsed, transport = asyncio.run(run())
    assert elapsed >= 0.09
    assert transport.sent == [("session", 3_200), ("session", 3_200)]


def test_discard_resets_websocket_pacing_deadline() -> None:
    async def run() -> float:
        transport = _RecordingTransport()
        service = _Service()
        await transport.send_audio_chunk(service, "session", bytes(3_200))  # type: ignore[arg-type]
        transport.discard_pending_audio()
        started = monotonic()
        await transport.send_audio_chunk(service, "session", bytes(3_200))  # type: ignore[arg-type]
        return monotonic() - started

    assert asyncio.run(run()) < 0.05
