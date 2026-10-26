"""
Ordered Segment Accumulator

Consumes a stream (iterable) of :class:`~sde4_stt.schemas.transcript.TranscriptEvent`
and produces a deterministic, chronologically ordered list of final
:class:`~sde4_stt.schemas.transcript.TranscriptSegment` objects.

The accumulator is deliberately tolerant of out‑of‑order ``FINAL`` events –
segments are stored in a dict keyed by ``sequence_number`` and emitted sorted
once the ``COMPLETE`` lifecycle event arrives.

Typical usage:

```python
from sde4_stt.accumulator import SegmentAccumulator

# `events` could be the output of FakeTranscriber.transcribe_stream(...)
segments = SegmentAccumulator().accumulate(events)
for seg in segments:
    print(seg.text)
```
"""

from typing import Iterable, List, Dict, Optional
from sde4_stt.schemas.transcript import TranscriptEvent, TranscriptEventType, TranscriptSegment


class SegmentAccumulator:
    """Collect ``FINAL`` segments from a transcript event stream.

    The accumulator keeps a small amount of state:
    * ``_final_segments`` – dict mapping ``sequence_number`` → ``TranscriptSegment``
    * ``_interim`` – optional latest interim segment (useful for debugging)
    * ``_started`` – boolean flag indicating a START event has been seen
    * ``_completed`` – flag set once a COMPLETE event is processed
    """

    def __init__(self) -> None:
        self._reset_state()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def accumulate(self, events: Iterable[TranscriptEvent]) -> List[TranscriptSegment]:
        """Consume an iterable of ``TranscriptEvent`` and return ordered finals.

        Parameters
        ----------
        events:
            Any iterable (list, generator, etc.) yielding ``TranscriptEvent``
            objects produced by a transcriber implementation.

        Returns
        -------
        list[TranscriptSegment]
            All ``FINAL`` segments sorted by ``sequence_number`` (which aligns
            with chronological order). Raises ``RuntimeError`` if the stream
            does not contain a ``COMPLETE`` event.
        """
        for ev in events:
            self._process_event(ev)

        if not self._completed:
            raise RuntimeError("Transcript stream ended without a COMPLETE event.")

        # Return segments sorted by sequence_number – this guarantees deterministic order.
        ordered = [self._final_segments[key] for key in sorted(self._final_segments)]
        return ordered

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _reset_state(self) -> None:
        self._final_segments: Dict[int, TranscriptSegment] = {}
        self._interim: Optional[TranscriptSegment] = None
        self._started: bool = False
        self._completed: bool = False

    def _process_event(self, ev: TranscriptEvent) -> None:
        et = ev.event_type
        if et == TranscriptEventType.START:
            self._started = True
            # Reset any previous state in case of re‑use
            self._final_segments.clear()
            self._interim = None
            self._completed = False
            return

        if not self._started:
            # Defensive – any non‑START event before a START is a contract violation.
            raise RuntimeError("Received TranscriptEvent before START event.")

        if et == TranscriptEventType.INTERIM:
            # Store the latest interim (useful for debugging, not needed for final output)
            self._interim = ev.segment
            return

        if et == TranscriptEventType.FINAL:
            seg = ev.segment
            if seg is None:
                raise RuntimeError("FINAL event missing segment payload.")
            # Store by sequence_number – later we will sort.
            self._final_segments[seg.sequence_number] = seg
            # Clear interim for this sequence (if any)
            self._interim = None
            return

        if et == TranscriptEventType.COMPLETE:
            self._completed = True
            return

        if et == TranscriptEventType.ERROR:
            # Propagate error upwards – user code can catch RuntimeError.
            raise RuntimeError(f"Transcriber emitted ERROR: {ev.error.code if ev.error else 'unknown'}")

        # Should never reach here because all enum members are handled.
        raise RuntimeError(f"Unhandled TranscriptEventType: {et}")

    # ---------------------------------------------------------------------
    # Convenience classmethod for one‑shot usage
    # ---------------------------------------------------------------------
    @classmethod
    def from_stream(cls, events: Iterable[TranscriptEvent]) -> List[TranscriptSegment]:
        """Shortcut – ``SegmentAccumulator.from_stream(events)`` returns ordered finals."""
        return cls().accumulate(events)

"""Exported symbols for ``sde4_stt.accumulator`` package."""

__all__ = ["SegmentAccumulator"]
