"""
SDE 4 - Speech-to-Text (W2) Core Library
Academia AI Voice Co-Pilot Project
"""

from sde4_stt.schemas.transcript import (
    TranscriptEventType,
    WordTiming,
    TranscriptSegment,
    TranscriptError,
    TranscriptEvent,
)
from sde4_stt.schemas.audio import AudioChunk
from sde4_stt.adapters.base import BaseTranscriberAdapter
from sde4_stt.adapters.fake_transcriber import FakeTranscriber
from sde4_stt.accumulator import SegmentAccumulator

__version__ = "0.1.0"

__all__ = [
    "TranscriptEventType",
    "WordTiming",
    "TranscriptSegment",
    "TranscriptError",
    "TranscriptEvent",
    "AudioChunk",
    "BaseTranscriberAdapter",
    "FakeTranscriber",
    "SegmentAccumulator",
]
