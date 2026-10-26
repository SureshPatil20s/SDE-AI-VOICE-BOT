"""
Tests for Day 1-2: Transcript Schemas (TranscriptSegment, TranscriptEvent, etc.)
"""

import pytest
from pydantic import ValidationError
from sde4_stt.schemas.transcript import (
    TranscriptEventType,
    WordTiming,
    TranscriptSegment,
    TranscriptError,
    TranscriptEvent,
)


class TestTranscriptSegmentSchema:
    def test_valid_interim_segment(self):
        seg = TranscriptSegment(
            segment_id="seg_001",
            sequence_number=0,
            text="hello world",
            start_time=0.0,
            end_time=1.25,
            is_final=False,
            confidence=0.85,
            language="en-US",
        )
        assert seg.segment_id == "seg_001"
        assert seg.sequence_number == 0
        assert seg.text == "hello world"
        assert seg.start_time == 0.0
        assert seg.end_time == 1.25
        assert not seg.is_final
        assert seg.confidence == 0.85
        assert seg.duration == 1.25

    def test_valid_final_segment_with_words(self):
        words = [
            WordTiming(word="hello", start_time=0.0, end_time=0.5, confidence=0.95),
            WordTiming(word="world", start_time=0.6, end_time=1.1, confidence=0.98),
        ]
        seg = TranscriptSegment(
            segment_id="seg_002",
            sequence_number=1,
            text="hello world",
            start_time=0.0,
            end_time=1.2,
            is_final=True,
            confidence=0.96,
            words=words,
        )
        assert seg.is_final is True
        assert len(seg.words) == 2
        assert seg.words[0].word == "hello"

    def test_reject_negative_start_time(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=-0.5,
                end_time=1.0,
            )
        assert "start_time" in str(exc.value)

    def test_reject_inverted_timestamps(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=2.0,
                end_time=1.0,
            )
        assert "cannot be earlier than start_time" in str(exc.value)

    def test_reject_out_of_bounds_confidence(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=0.0,
                end_time=1.0,
                confidence=1.5,
            )

        with pytest.raises(ValidationError):
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=0.0,
                end_time=1.0,
                confidence=-0.1,
            )

    def test_reject_empty_segment_id(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptSegment(
                segment_id="   ",
                sequence_number=0,
                text="hello",
                start_time=0.0,
                end_time=1.0,
            )
        assert "segment_id must contain non-whitespace" in str(exc.value)

    def test_reject_empty_text_on_final_segment(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="   ",
                start_time=0.0,
                end_time=1.0,
                is_final=True,
            )
        assert "must not contain empty or whitespace-only text" in str(exc.value)

    def test_reject_word_timing_exceeding_segment(self):
        words = [
            WordTiming(word="hello", start_time=0.0, end_time=1.5, confidence=0.9),
        ]
        with pytest.raises(ValidationError) as exc:
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=0.0,
                end_time=1.0,
                words=words,
            )
        assert "exceeds segment boundary" in str(exc.value)

    def test_reject_extra_fields(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(
                segment_id="seg_001",
                sequence_number=0,
                text="hello",
                start_time=0.0,
                end_time=1.0,
                unauthorized_extra_field="malicious",
            )


class TestTranscriptEventSchema:
    def test_valid_start_event(self):
        evt = TranscriptEvent(
            event_id="evt_001",
            session_id="sess_123",
            event_type=TranscriptEventType.START,
            timestamp=1700000000.0,
        )
        assert evt.event_type == TranscriptEventType.START
        assert evt.segment is None
        assert evt.error is None

    def test_valid_interim_event(self):
        seg = TranscriptSegment(
            segment_id="seg_001",
            sequence_number=0,
            text="testing",
            start_time=0.0,
            end_time=0.8,
            is_final=False,
        )
        evt = TranscriptEvent(
            event_id="evt_002",
            session_id="sess_123",
            event_type=TranscriptEventType.INTERIM,
            timestamp=1700000001.0,
            segment=seg,
        )
        assert evt.segment.text == "testing"
        assert not evt.segment.is_final

    def test_valid_final_event(self):
        seg = TranscriptSegment(
            segment_id="seg_002",
            sequence_number=1,
            text="testing one two three",
            start_time=0.0,
            end_time=2.1,
            is_final=True,
        )
        evt = TranscriptEvent(
            event_id="evt_003",
            session_id="sess_123",
            event_type=TranscriptEventType.FINAL,
            timestamp=1700000002.5,
            segment=seg,
        )
        assert evt.segment.is_final is True

    def test_reject_final_event_with_interim_segment(self):
        seg = TranscriptSegment(
            segment_id="seg_001",
            sequence_number=0,
            text="partial",
            start_time=0.0,
            end_time=1.0,
            is_final=False,
        )
        with pytest.raises(ValidationError) as exc:
            TranscriptEvent(
                event_id="evt_err",
                session_id="sess_123",
                event_type=TranscriptEventType.FINAL,
                timestamp=1700000000.0,
                segment=seg,
            )
        assert "A FINAL TranscriptEvent must carry a segment with is_final=True" in str(exc.value)

    def test_reject_interim_event_without_segment(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptEvent(
                event_id="evt_err",
                session_id="sess_123",
                event_type=TranscriptEventType.INTERIM,
                timestamp=1700000000.0,
            )
        assert "must include a 'segment' payload" in str(exc.value)

    def test_valid_error_event(self):
        err = TranscriptError(
            code="AUDIO_CORRUPTED",
            message="Unrecognized sample rate or corrupted WAV header",
            recoverable=False,
        )
        evt = TranscriptEvent(
            event_id="evt_err_01",
            session_id="sess_123",
            event_type=TranscriptEventType.ERROR,
            timestamp=1700000005.0,
            error=err,
        )
        assert evt.error.code == "AUDIO_CORRUPTED"
        assert not evt.error.recoverable

    def test_reject_error_event_without_error_payload(self):
        with pytest.raises(ValidationError) as exc:
            TranscriptEvent(
                event_id="evt_err_02",
                session_id="sess_123",
                event_type=TranscriptEventType.ERROR,
                timestamp=1700000005.0,
            )
        assert "must have an 'error' payload" in str(exc.value)
