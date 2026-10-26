"""
Provider-Neutral Speech-to-Text Adapter Interface.

Defines the abstract base contract that all STT engine adapters (e.g. Whisper, Google STT,
Deepgram, or local/mock engines) must implement. This decouples the higher-level Voice /
Orchestrator layers from specific third-party provider APIs.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Optional, Union
from sde4_stt.schemas.audio import AudioChunk
from sde4_stt.schemas.transcript import TranscriptEvent


class BaseTranscriberAdapter(ABC):
    """
    Abstract Base Class for Speech-to-Text Transcribers.

    Enforces a consistent lifecycle contract:
    1. `start_session(session_id, language, ...)` -> Emits START event.
    2. `process_chunk(chunk)` -> Emits 0 or more INTERIM / FINAL events.
    3. `finish_session()` -> Emits remaining FINAL and/or COMPLETE events.

    Also provides high-level generator utility `transcribe_stream()`.
    """

    def __init__(self, provider_name: str):
        self._provider_name = provider_name
        self._session_id: Optional[str] = None
        self._is_active: bool = False

    @property
    def provider_name(self) -> str:
        """Name of the underlying STT provider (e.g. 'fake-stt', 'whisper-local', 'deepgram')."""
        return self._provider_name

    @property
    def session_id(self) -> Optional[str]:
        """Currently active session ID, or None if no session is active."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """True if an audio transcription session is currently open."""
        return self._is_active

    @abstractmethod
    def start_session(
        self,
        session_id: str,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> List[TranscriptEvent]:
        """
        Initializes a transcription session.

        Args:
            session_id: Non-empty identifier for this interaction.
            language: Target spoken language code (e.g. 'en', 'en-US').
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            List containing the initial START event (or ERROR event if init fails).
        """
        pass

    @abstractmethod
    def process_chunk(
        self,
        chunk: Union[AudioChunk, bytes],
    ) -> List[TranscriptEvent]:
        """
        Ingests a single audio chunk and returns zero or more transcript events.

        Args:
            chunk: AudioChunk instance or raw PCM/WAV audio bytes.

        Returns:
            List of generated TranscriptEvent objects (e.g. INTERIM or FINAL).
        """
        pass

    @abstractmethod
    def finish_session(self) -> List[TranscriptEvent]:
        """
        Finalizes the active session, flushing any pending interim results to final,
        and emitting a COMPLETE event.

        Returns:
            List of finalization TranscriptEvent objects.
        """
        pass

    def transcribe_stream(
        self,
        chunks: Iterable[Union[AudioChunk, bytes]],
        session_id: str,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> Iterator[TranscriptEvent]:
        """
        Convenience generator that processes a continuous stream of chunks end-to-end.

        Yields:
            TranscriptEvent instances in strict chronological order.
        """
        # 1. Start Session
        for event in self.start_session(session_id, language, sample_rate):
            yield event

        # 2. Feed chunks
        for chunk in chunks:
            events = self.process_chunk(chunk)
            for event in events:
                yield event

        # 3. Finish session
        for event in self.finish_session():
            yield event
