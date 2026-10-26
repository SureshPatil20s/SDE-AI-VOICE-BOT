"""
Transcript Schemas for SDE 4 Speech-to-Text (W2) Workstream.

This module defines the fundamental data contracts for speech-to-text processing:
- `TranscriptSegment`: An individual transcribed audio time slice.
- `TranscriptError`: Structured error information for failed or degraded events.
- `TranscriptEventType`: Enumeration of lifecycle events emitted by transcribers.
- `TranscriptEvent`: A streamable event emitted during audio transcription.

All schemas enforce strict runtime validation using Pydantic V2, with explicit units,
bounds checks, and descriptive error messages.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptEventType(str, Enum):
    """
    Lifecycle event types emitted across the transcription pipeline.
    
    - START: A new audio stream/session or chunk recognition has begun.
    - INTERIM: Partial, non-final hypothesis (subject to change as more audio arrives).
    - FINAL: Committed, immutable transcription segment with high confidence.
    - COMPLETE: The audio stream or utterance has finished processing cleanly.
    - ERROR: A processing, format, or network error occurred during transcription.
    """
    START = "START"
    INTERIM = "INTERIM"
    FINAL = "FINAL"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class WordTiming(BaseModel):
    """
    Timing information for an individual word inside a segment.
    
    Attributes:
        word: The word token string.
        start_time: Start offset in seconds from audio onset (>= 0.0).
        end_time: End offset in seconds from audio onset (>= start_time).
        confidence: Word-level confidence score in range [0.0, 1.0].
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    word: str = Field(..., min_length=1, description="Word token text")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds from audio start")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds from audio start")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 (lowest) and 1.0 (highest)"
    )

    @model_validator(mode="after")
    def validate_timing_order(self) -> "WordTiming":
        if self.end_time < self.start_time:
            raise ValueError(
                f"Word timing invalid: end_time ({self.end_time}s) cannot precede "
                f"start_time ({self.start_time}s)."
            )
        return self


class TranscriptSegment(BaseModel):
    """
    Represents an individual transcribed segment of speech.

    Fields & Units:
        segment_id: Unique string identifier for the segment (e.g. 'seg_001').
        sequence_number: Monotonically non-decreasing index of the segment within the stream (>= 0).
        text: Transcribed textual content.
        start_time: Start offset in seconds relative to stream start (>= 0.0).
        end_time: End offset in seconds relative to stream start (>= start_time).
        is_final: True if this is an immutable final result; False if interim/partial.
        confidence: Normalized confidence score [0.0, 1.0].
        language: BCP-47 or ISO 639-1 language code (e.g. 'en', 'en-US').
        words: Optional list of word-level timestamps and confidences.
        speaker_id: Optional speaker tag/diarization identifier.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the segment (non-empty string)"
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Zero-based sequence order of this segment in the audio stream"
    )
    text: str = Field(
        ...,
        description="Transcribed text output. Cannot be whitespace-only for final segments."
    )
    start_time: float = Field(
        ...,
        ge=0.0,
        description="Start timestamp in seconds relative to the start of audio (>= 0.0)"
    )
    end_time: float = Field(
        ...,
        ge=0.0,
        description="End timestamp in seconds relative to the start of audio (>= start_time)"
    )
    is_final: bool = Field(
        default=False,
        description="True if segment is committed and final; False if interim/tentative"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score normalized between 0.0 and 1.0"
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=15,
        description="Language code (e.g., 'en', 'en-US', 'es', 'fr')"
    )
    words: Optional[List[WordTiming]] = Field(
        default=None,
        description="Optional list of word-level timings and confidences"
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="Optional speaker identifier (for diarized audio)"
    )

    @field_validator("segment_id", mode="after")
    @classmethod
    def validate_segment_id(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("segment_id must contain non-whitespace characters")
        return trimmed

    @field_validator("language", mode="after")
    @classmethod
    def validate_language_format(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("language must not be empty or whitespace")
        # Allowed chars: alphanumeric and hyphens (e.g., en, en-US, pt-BR)
        for char in trimmed:
            if not (char.isalnum() or char == "-"):
                raise ValueError(
                    f"Invalid character '{char}' in language code '{v}'. "
                    "Expected standard ISO/BCP-47 format (e.g., 'en', 'en-US')."
                )
        return trimmed

    @model_validator(mode="after")
    def validate_segment_integrity(self) -> "TranscriptSegment":
        # Check start and end bounds
        if self.end_time < self.start_time:
            raise ValueError(
                f"Segment timing conflict: end_time ({self.end_time}s) cannot be earlier than "
                f"start_time ({self.start_time}s)."
            )

        # Check final segment text
        if self.is_final and not self.text.strip():
            raise ValueError(
                "A finalized segment (is_final=True) must not contain empty or whitespace-only text."
            )

        # Check word timings if provided
        if self.words:
            for w in self.words:
                if w.start_time < self.start_time or w.end_time > (self.end_time + 1e-4):
                    raise ValueError(
                        f"Word '{w.word}' with range [{w.start_time}s, {w.end_time}s] "
                        f"exceeds segment boundary [{self.start_time}s, {self.end_time}s]."
                    )

        return self

    @property
    def duration(self) -> float:
        """Returns the duration of the segment in seconds."""
        return round(self.end_time - self.start_time, 4)


class TranscriptError(BaseModel):
    """
    Structured error information attached to an ERROR event.

    Attributes:
        code: Machine-readable error code (e.g. 'INVALID_AUDIO_FORMAT', 'STREAM_TIMEOUT').
        message: Human-readable error description.
        details: Optional dictionary containing additional diagnostic data.
        recoverable: Flag indicating if the client can retry or recover without restarting session.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1, description="Machine-readable error identifier")
    message: str = Field(..., min_length=1, description="Human-readable explanation of error")
    details: Optional[dict] = Field(default=None, description="Optional extra diagnostic key-value pairs")
    recoverable: bool = Field(default=False, description="Whether transcription can proceed or retry")


class TranscriptEvent(BaseModel):
    """
    Streamable event emitted during real-time speech transcription.

    Lifecycle:
        START -> (INTERIM)* -> FINAL -> (INTERIM)* -> FINAL ... -> COMPLETE
        (Or ERROR emitted at any point upon failure)

    Attributes:
        event_id: Unique identifier for this event instance.
        session_id: Associated audio session or client interaction ID.
        event_type: Enum value specifying the type of lifecycle event.
        timestamp: Unix epoch timestamp in seconds when the event was generated.
        segment: The transcript segment payload (present for INTERIM and FINAL events).
        error: Structured error payload (required if event_type == ERROR).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(..., min_length=1, description="Unique event identifier")
    session_id: str = Field(..., min_length=1, description="Session or stream identifier")
    event_type: TranscriptEventType = Field(..., description="Lifecycle type of the event")
    timestamp: float = Field(..., ge=0.0, description="Unix timestamp in seconds when event was created")
    segment: Optional[TranscriptSegment] = Field(
        default=None,
        description="Segment data payload. Required for INTERIM and FINAL events."
    )
    error: Optional[TranscriptError] = Field(
        default=None,
        description="Error details. Required if event_type is ERROR."
    )

    @field_validator("event_id", "session_id", mode="after")
    @classmethod
    def validate_non_empty_ids(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("ID strings must not be empty or whitespace-only")
        return trimmed

    @model_validator(mode="after")
    def validate_event_consistency(self) -> "TranscriptEvent":
        # If event_type is ERROR, error field MUST be provided
        if self.event_type == TranscriptEventType.ERROR:
            if self.error is None:
                raise ValueError("TranscriptEvent of type ERROR must have an 'error' payload.")

        # If event_type is INTERIM or FINAL, segment field MUST be provided
        if self.event_type in (TranscriptEventType.INTERIM, TranscriptEventType.FINAL):
            if self.segment is None:
                raise ValueError(
                    f"TranscriptEvent of type {self.event_type.value} must include a 'segment' payload."
                )

            # Ensure consistency between event_type and segment.is_final
            if self.event_type == TranscriptEventType.FINAL and not self.segment.is_final:
                raise ValueError("A FINAL TranscriptEvent must carry a segment with is_final=True.")
            if self.event_type == TranscriptEventType.INTERIM and self.segment.is_final:
                raise ValueError("An INTERIM TranscriptEvent must carry a segment with is_final=False.")

        # For START and COMPLETE events, error must not be populated
        if self.event_type in (TranscriptEventType.START, TranscriptEventType.COMPLETE):
            if self.error is not None:
                raise ValueError(
                    f"TranscriptEvent of type {self.event_type.value} cannot contain an error payload."
                )

        return self
