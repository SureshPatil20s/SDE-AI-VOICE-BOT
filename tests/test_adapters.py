"""
Unit tests for Day 3-4: BaseTranscriberAdapter and Deterministic FakeTranscriber.
"""

import pytest
from sde4_stt.adapters.fake_transcriber import FakeTranscriber
from sde4_stt.schemas.audio import AudioChunk
from sde4_stt.schemas.transcript import TranscriptEventType


class TestFakeTranscriber:
    def test_determinism_identical_runs_produce_identical_events(self):
        """Guarantee: same input chunks -> 100% identical transcript events."""
        chunks = [
            AudioChunk(chunk_id="c1", data=b"chunk_one_pcm_data_001", sequence_number=0, duration_seconds=0.5),
            AudioChunk(chunk_id="c2", data=b"chunk_two_pcm_data_002", sequence_number=1, duration_seconds=0.5),
            AudioChunk(chunk_id="c3", data=b"chunk_three_pcm_data_003", sequence_number=2, duration_seconds=0.5, is_last=True),
        ]

        transcriber_a = FakeTranscriber(canned_transcript="Why is this code failing")
        transcriber_b = FakeTranscriber(canned_transcript="Why is this code failing")

        events_a = list(transcriber_a.transcribe_stream(chunks, session_id="sess_det_123"))
        events_b = list(transcriber_b.transcribe_stream(chunks, session_id="sess_det_123"))

        assert len(events_a) == len(events_b)
        assert len(events_a) > 0

        for ea, eb in zip(events_a, events_b):
            assert ea.event_id == eb.event_id
            assert ea.event_type == eb.event_type
            assert ea.timestamp == eb.timestamp
            if ea.segment:
                assert ea.segment.text == eb.segment.text
                assert ea.segment.is_final == eb.segment.is_final
                assert ea.segment.start_time == eb.segment.start_time
                assert ea.segment.end_time == eb.segment.end_time
                assert ea.segment.confidence == eb.segment.confidence

    def test_lifecycle_events_progression(self):
        """Verifies START -> INTERIM -> FINAL -> COMPLETE lifecycle."""
        transcriber = FakeTranscriber(canned_transcript="Explain this error")
        
        # 1. Start
        start_events = transcriber.start_session("sess_001")
        assert len(start_events) == 1
        assert start_events[0].event_type == TranscriptEventType.START

        # 2. Chunk 1 (interim)
        chunk1 = AudioChunk(chunk_id="c1", data=b"audio_bytes_1", sequence_number=0, duration_seconds=1.0)
        c1_events = transcriber.process_chunk(chunk1)
        assert len(c1_events) == 1
        assert c1_events[0].event_type == TranscriptEventType.INTERIM
        assert not c1_events[0].segment.is_final

        # 3. Chunk 2 (last chunk -> emits FINAL + COMPLETE)
        chunk2 = AudioChunk(chunk_id="c2", data=b"audio_bytes_2", sequence_number=1, duration_seconds=1.0, is_last=True)
        c2_events = transcriber.process_chunk(chunk2)
        assert len(c2_events) == 2
        assert c2_events[0].event_type == TranscriptEventType.FINAL
        assert c2_events[0].segment.is_final is True
        assert c2_events[0].segment.text == "Explain this error"
        assert c2_events[1].event_type == TranscriptEventType.COMPLETE

    def test_finish_session_flushes_pending_transcript(self):
        """When audio stream ends without is_last flag, finish_session flushes final event."""
        transcriber = FakeTranscriber(canned_transcript="Run unit tests")
        transcriber.start_session("sess_002")
        
        chunk = AudioChunk(chunk_id="c1", data=b"audio_slice", sequence_number=0, duration_seconds=0.8, is_last=False)
        interim = transcriber.process_chunk(chunk)
        assert interim[0].event_type == TranscriptEventType.INTERIM

        flush_events = transcriber.finish_session()
        assert len(flush_events) == 2
        assert flush_events[0].event_type == TranscriptEventType.FINAL
        assert flush_events[0].segment.is_final is True
        assert flush_events[1].event_type == TranscriptEventType.COMPLETE
        assert not transcriber.is_active

    def test_raw_bytes_input_support(self):
        """Transcriber accepts raw bytes in addition to AudioChunk."""
        transcriber = FakeTranscriber()
        transcriber.start_session("sess_bytes")
        
        raw_pcm = b"\x00\x01" * 16000  # 1 second of 16kHz 16-bit mono
        events = transcriber.process_chunk(raw_pcm)
        assert len(events) == 1
        assert events[0].event_type == TranscriptEventType.INTERIM
        assert events[0].segment.end_time == 1.0

    def test_simulated_error_injection(self):
        """Transcriber can inject an ERROR event on a designated chunk index."""
        transcriber = FakeTranscriber(fail_at_chunk_index=1, error_code="MIC_DISCONNECTED")
        transcriber.start_session("sess_err")

        chunk0 = AudioChunk(chunk_id="c0", data=b"chunk_0_ok", sequence_number=0, duration_seconds=0.5)
        evts0 = transcriber.process_chunk(chunk0)
        assert evts0[0].event_type == TranscriptEventType.INTERIM

        chunk1 = AudioChunk(chunk_id="c1", data=b"chunk_1_fail", sequence_number=1, duration_seconds=0.5)
        evts1 = transcriber.process_chunk(chunk1)
        assert len(evts1) == 1
        assert evts1[0].event_type == TranscriptEventType.ERROR
        assert evts1[0].error.code == "MIC_DISCONNECTED"
        assert not transcriber.is_active

    def test_process_chunk_without_start_raises_runtime_error(self):
        transcriber = FakeTranscriber()
        with pytest.raises(RuntimeError):
            transcriber.process_chunk(b"some_audio")

    def test_invalid_session_id_raises_value_error(self):
        transcriber = FakeTranscriber()
        with pytest.raises(ValueError):
            transcriber.start_session("   ")
