"""
Deterministic Fake Transcriber for Mock Testing and Offline Development.

Implements `BaseTranscriberAdapter` without requiring any external or paid speech APIs.
Key guarantee: Identical inputs (audio chunks + session_id) yield identical, deterministic
`TranscriptEvent` sequences.
"""

import hashlib
from typing import List, Optional, Union
from sde4_stt.adapters.base import BaseTranscriberAdapter
from sde4_stt.schemas.audio import AudioChunk
from sde4_stt.schemas.transcript import (
    TranscriptError,
    TranscriptEvent,
    TranscriptEventType,
    TranscriptSegment,
    WordTiming,
)

# Standard mock vocabulary for deterministic chunk hashing
DETERMINISTIC_VOCABULARY = [
    "why", "is", "this", "application", "showing", "an", "error",
    "explain", "code", "context", "terminal", "module", "not", "found",
    "academia", "copilot", "voice", "assistant", "debugging", "pipeline"
]


class FakeTranscriber(BaseTranscriberAdapter):
    """
    Deterministic Mock Transcriber for the SDE 4 Speech-to-Text Pipeline.

    Features:
    - Zero external network calls, zero API keys required.
    - Deterministic execution: identical chunks always yield identical events and hashes.
    - Supports two operational modes:
      1. `canned_transcript`: An explicit script that is unrolled progressively across chunks.
      2. Hash-derived vocabulary: Generates deterministic words derived from SHA-256 chunk hashes.
    - Supports simulated error injection for negative testing.
    """

    def __init__(
        self,
        canned_transcript: Optional[str] = None,
        base_timestamp: float = 1700000000.0,
        fixed_confidence: float = 0.95,
        fail_at_chunk_index: Optional[int] = None,
        error_code: str = "SIMULATED_STT_ERROR",
        bytes_per_second: int = 32000,  # e.g., 16000Hz * 16-bit mono = 32000 bytes/sec
    ):
        super().__init__(provider_name="fake-deterministic-stt")
        self._canned_transcript = canned_transcript
        self._canned_words = canned_transcript.split() if canned_transcript else []
        self._base_timestamp = base_timestamp
        self._fixed_confidence = fixed_confidence
        self._fail_at_chunk_index = fail_at_chunk_index
        self._error_code = error_code
        self._bytes_per_second = bytes_per_second

        # Session tracking state
        self._language = "en"
        self._chunk_count = 0
        self._accumulated_audio_duration = 0.0
        self._accumulated_words: List[str] = []
        self._has_emitted_final = False

    def start_session(
        self,
        session_id: str,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> List[TranscriptEvent]:
        """Initializes a new deterministic transcription session."""
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be non-empty")

        self._session_id = session_id.strip()
        self._language = language
        self._is_active = True
        self._chunk_count = 0
        self._accumulated_audio_duration = 0.0
        self._accumulated_words = []
        self._has_emitted_final = False

        start_event = TranscriptEvent(
            event_id=f"evt_{self._session_id}_start",
            session_id=self._session_id,
            event_type=TranscriptEventType.START,
            timestamp=self._base_timestamp,
        )
        return [start_event]

    def process_chunk(
        self,
        chunk: Union[AudioChunk, bytes],
    ) -> List[TranscriptEvent]:
        """Processes an audio chunk and deterministically produces transcript events."""
        if not self._is_active or self._session_id is None:
            raise RuntimeError("Cannot process chunks without an active session. Call start_session() first.")

        # Extract bytes, sequence number, and duration
        if isinstance(chunk, AudioChunk):
            chunk_bytes = chunk.data
            seq = chunk.sequence_number
            chunk_duration = (
                chunk.duration_seconds
                if chunk.duration_seconds is not None
                else len(chunk_bytes) / (chunk.sample_rate * chunk.channels * 2)
            )
            is_last = chunk.is_last
        else:
            chunk_bytes = chunk
            seq = self._chunk_count
            chunk_duration = len(chunk_bytes) / float(self._bytes_per_second)
            is_last = False

        current_chunk_idx = self._chunk_count
        self._chunk_count += 1

        # Check for simulated failure injection
        if self._fail_at_chunk_index is not None and current_chunk_idx == self._fail_at_chunk_index:
            self._is_active = False
            return [
                TranscriptEvent(
                    event_id=f"evt_{self._session_id}_err_{current_chunk_idx}",
                    session_id=self._session_id,
                    event_type=TranscriptEventType.ERROR,
                    timestamp=self._base_timestamp + self._accumulated_audio_duration,
                    error=TranscriptError(
                        code=self._error_code,
                        message=f"Simulated transcriber failure triggered at chunk index {current_chunk_idx}",
                        recoverable=False,
                    ),
                )
            ]

        # Calculate time windows
        start_t = self._accumulated_audio_duration
        end_t = start_t + chunk_duration
        self._accumulated_audio_duration = end_t

        # Determine words recognized for this chunk deterministically
        new_words = self._resolve_chunk_words(chunk_bytes, current_chunk_idx, is_last=is_last)
        self._accumulated_words.extend(new_words)

        if not self._accumulated_words:
            # Silence or no tokens recognized in this chunk
            return []

        # If marked as last chunk, emit FINAL event directly
        if is_last:
            self._has_emitted_final = True
            # For canned transcripts, ensure the final segment contains the full transcript
            final_words = self._canned_words if self._canned_transcript else self._accumulated_words
            final_segment = self._build_segment(
                words=final_words,
                start_time=0.0,
                end_time=round(self._accumulated_audio_duration, 3),
                sequence_number=seq,
                is_final=True,
            )
            event_hash = hashlib.sha256(chunk_bytes).hexdigest()[:8]
            final_event = TranscriptEvent(
                event_id=f"evt_{self._session_id}_final_{seq}_{event_hash}",
                session_id=self._session_id,
                event_type=TranscriptEventType.FINAL,
                timestamp=self._base_timestamp + end_t,
                segment=final_segment,
            )
            complete_event = TranscriptEvent(
                event_id=f"evt_{self._session_id}_complete",
                session_id=self._session_id,
                event_type=TranscriptEventType.COMPLETE,
                timestamp=self._base_timestamp + end_t,
            )
            self._is_active = False
            return [final_event, complete_event]

        # Otherwise emit INTERIM event
        interim_segment = self._build_segment(
            words=self._accumulated_words,
            start_time=0.0,
            end_time=round(end_t, 3),
            sequence_number=seq,
            is_final=False,
        )
        event_hash = hashlib.sha256(chunk_bytes).hexdigest()[:8]
        interim_event = TranscriptEvent(
            event_id=f"evt_{self._session_id}_interim_{seq}_{event_hash}",
            session_id=self._session_id,
            event_type=TranscriptEventType.INTERIM,
            timestamp=self._base_timestamp + end_t,
            segment=interim_segment,
        )
        return [interim_event]

    def finish_session(self) -> List[TranscriptEvent]:
        """Closes the session, emitting final segment if pending, and a COMPLETE event."""
        if not self._is_active or self._session_id is None:
            return []

        events: List[TranscriptEvent] = []
        end_t = self._accumulated_audio_duration

        if not self._has_emitted_final:
            text_words = (
                self._canned_words
                if self._canned_transcript
                else (self._accumulated_words or ["<silence>"])
            )
            final_segment = self._build_segment(
                words=text_words,
                start_time=0.0,
                end_time=round(end_t if end_t > 0 else 0.1, 3),
                sequence_number=self._chunk_count,
                is_final=True,
            )
            final_event = TranscriptEvent(
                event_id=f"evt_{self._session_id}_flush_final",
                session_id=self._session_id,
                event_type=TranscriptEventType.FINAL,
                timestamp=self._base_timestamp + end_t,
                segment=final_segment,
            )
            events.append(final_event)
            self._has_emitted_final = True

        complete_event = TranscriptEvent(
            event_id=f"evt_{self._session_id}_complete",
            session_id=self._session_id,
            event_type=TranscriptEventType.COMPLETE,
            timestamp=self._base_timestamp + end_t,
        )
        events.append(complete_event)
        self._is_active = False
        return events

    # --- Internal Deterministic Helper Methods ---

    def _resolve_chunk_words(self, chunk_bytes: bytes, chunk_idx: int, is_last: bool = False) -> List[str]:
        """Resolves words deterministically either from script or chunk content hash."""
        if self._canned_transcript is not None:
            if is_last:
                already_collected = len(self._accumulated_words)
                if already_collected < len(self._canned_words):
                    return self._canned_words[already_collected:]
                return []

            words_per_chunk = max(1, len(self._canned_words) // 3)
            start_word_idx = chunk_idx * words_per_chunk
            if start_word_idx < len(self._canned_words):
                end_word_idx = min(start_word_idx + words_per_chunk, len(self._canned_words))
                return self._canned_words[start_word_idx:end_word_idx]
            return []

        # Hash-based vocabulary mode
        # Hash chunk bytes to pick 1-2 reproducible vocabulary words
        digest = hashlib.sha256(chunk_bytes).digest()
        idx1 = digest[0] % len(DETERMINISTIC_VOCABULARY)
        idx2 = digest[1] % len(DETERMINISTIC_VOCABULARY)
        word1 = DETERMINISTIC_VOCABULARY[idx1]
        word2 = DETERMINISTIC_VOCABULARY[idx2]
        return [word1] if word1 == word2 else [word1, word2]

    def _build_segment(
        self,
        words: List[str],
        start_time: float,
        end_time: float,
        sequence_number: int,
        is_final: bool,
    ) -> TranscriptSegment:
        """Constructs a TranscriptSegment with deterministic word timings."""
        text = " ".join(words)
        word_count = len(words)
        duration = max(0.01, end_time - start_time)

        # Build proportional word timings
        word_timings: List[WordTiming] = []
        if word_count > 0:
            step = duration / word_count
            for i, w in enumerate(words):
                w_start = round(start_time + i * step, 3)
                w_end = round(w_start + step, 3)
                # Ensure boundary clamping
                w_end = min(w_end, end_time)
                word_timings.append(
                    WordTiming(
                        word=w,
                        start_time=w_start,
                        end_time=w_end,
                        confidence=self._fixed_confidence,
                    )
                )

        segment_id = f"seg_{self._session_id}_{sequence_number}"
        return TranscriptSegment(
            segment_id=segment_id,
            sequence_number=sequence_number,
            text=text,
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            is_final=is_final,
            confidence=self._fixed_confidence,
            language=self._language,
            words=word_timings if word_timings else None,
        )
