"""
Audio input schemas for SDE 4 Speech-to-Text pipeline.

Defines standardized audio chunk payloads consumed by transcriber adapters.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioChunk(BaseModel):
    """
    Represents a discrete slice/chunk of audio data passed to a transcriber.

    Attributes:
        chunk_id: Unique identifier for this chunk (e.g., 'chk_001').
        data: Raw audio bytes (PCM, WAV, or Opus encoded).
        sequence_number: Monotonic ordering of the chunk in the stream (>= 0).
        sample_rate: Audio sampling frequency in Hz (e.g. 16000, 44100, 48000).
        channels: Number of audio channels (1 for mono, 2 for stereo).
        is_last: Boolean flag indicating if this chunk marks the end of the utterance/stream.
        duration_seconds: Estimated duration of this audio slice in seconds (>= 0.0).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    data: bytes = Field(..., description="Raw audio bytes")
    sequence_number: int = Field(..., ge=0, description="Sequential position in stream")
    sample_rate: int = Field(default=16000, gt=0, description="Sampling rate in Hz (e.g. 16000)")
    channels: int = Field(default=1, ge=1, le=2, description="Channel count: 1=mono, 2=stereo")
    is_last: bool = Field(default=False, description="True if end of audio stream")
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Duration of this chunk in seconds"
    )

    @field_validator("chunk_id", mode="after")
    @classmethod
    def validate_chunk_id(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("chunk_id must contain non-whitespace characters")
        return trimmed

    @field_validator("data", mode="after")
    @classmethod
    def validate_data_not_empty(cls, v: bytes) -> bytes:
        if len(v) == 0:
            raise ValueError("Audio chunk data cannot be empty bytes")
        return v
