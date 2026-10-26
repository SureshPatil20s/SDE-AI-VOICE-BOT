"""
Unit tests for the ordered SegmentAccumulator (Day 5‑6).
"""

import pytest
from sde4_stt.accumulator import SegmentAccumulator
from sde4_stt.schemas.transcript import (
    TranscriptEvent,
    TranscriptEventType,
    TranscriptSegment,
)


def make_final_event(session_id: str, seq: int, text: str) -> TranscriptEvent:
    seg = TranscriptSegment(
        segment_id=f"seg_{session_id}_{seq}",
        sequence_number=seq,
        text=text,
        start_time=0.0,
        end_time=1.0,
        is_final=True,
        confidence=0.9,
        language="en",
    )
    return TranscriptEvent(
        event_id=f"evt_{session_id}_final_{seq}",
        session_id=session_id,
        event_type=TranscriptEventType.FINAL,
        timestamp=1000.0 + seq,
        segment=seg,
    )


def make_start_event(session_id: str) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=f"evt_{session_id}_start",
        session_id=session_id,
        event_type=TranscriptEventType.START,
        timestamp=999.0,
    )


def make_complete_event(session_id: str) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=f"evt_{session_id}_complete",
        session_id=session_id,
        event_type=TranscriptEventType.COMPLETE,
        timestamp=2000.0,
    )


def make_error_event(session_id: str, code: str = "TEST_ERROR") -> TranscriptEvent:
    from sde4_stt.schemas.transcript import TranscriptError
    return TranscriptEvent(
        event_id=f"evt_{session_id}_error",
        session_id=session_id,
        event_type=TranscriptEventType.ERROR,
        timestamp=1500.0,
        error=TranscriptError(code=code, message="Simulated error", recoverable=False),
    )


class TestSegmentAccumulator:
    def test_ordered_output_from_out_of_order_finals(self):
        events = [
            make_start_event("sess1"),
            # Emit FINAL with sequence 2 first
            make_final_event("sess1", 2, "third"),
            # Emit FINAL with sequence 0 next
            make_final_event("sess1", 0, "first"),
            # Emit FINAL with sequence 1 next
            make_final_event("sess1", 1, "second"),
            make_complete_event("sess1"),
        ]
        ordered = SegmentAccumulator().accumulate(events)
        assert [seg.text for seg in ordered] == ["first", "second", "third"]
        # Ensure sequence numbers are sorted
        assert [seg.sequence_number for seg in ordered] == [0, 1, 2]

    def test_missing_complete_raises(self):
        events = [make_start_event("sess2"), make_final_event("sess2", 0, "only")]  # No COMPLETE
        with pytest.raises(RuntimeError) as exc:
            SegmentAccumulator().accumulate(events)
        assert "without a COMPLETE" in str(exc.value)

    def test_error_event_propagates(self):
        events = [make_start_event("sess3"), make_error_event("sess3"), make_complete_event("sess3")]
        with pytest.raises(RuntimeError) as exc:
            SegmentAccumulator().accumulate(events)
        assert "Transcriber emitted ERROR" in str(exc.value)

    def test_interim_events_are_ignored(self):
        # Build a simple stream with an INTERIM event between START and FINAL
        from sde4_stt.schemas.transcript import TranscriptEvent, TranscriptEventType, TranscriptSegment
        interim_seg = TranscriptSegment(
            segment_id="seg_interim",
            sequence_number=0,
            text="partial",
            start_time=0.0,
            end_time=0.5,
            is_final=False,
            confidence=0.8,
            language="en",
        )
        interim_event = TranscriptEvent(
            event_id="evt_interim",
            session_id="sess4",
            event_type=TranscriptEventType.INTERIM,
            timestamp=1000.0,
            segment=interim_seg,
        )
        events = [
            make_start_event("sess4"),
            interim_event,
            make_final_event("sess4", 0, "final"),
            make_complete_event("sess4"),
        ]
        ordered = SegmentAccumulator().accumulate(events)
        # Only the FINAL segment should be present
        assert len(ordered) == 1
        assert ordered[0].text == "final"

    def test_from_stream_classmethod(self):
        events = [
            make_start_event("sess5"),
            make_final_event("sess5", 0, "hello"),
            make_complete_event("sess5"),
        ]
        result = SegmentAccumulator.from_stream(events)
        assert isinstance(result, list)
        assert result[0].text == "hello"

