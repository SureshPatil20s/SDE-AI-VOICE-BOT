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

__version__ = "0.1.0"

__all__ = [
    "TranscriptEventType",
    "WordTiming",
    "TranscriptSegment",
    "TranscriptError",
    "TranscriptEvent",
]
