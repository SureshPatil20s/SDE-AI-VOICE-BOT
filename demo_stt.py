"""
Demonstration script for SDE 4 Speech-to-Text Pipeline (Mock Mode).

Shows:
1. Session start lifecycle.
2. Ingesting continuous audio chunks with incremental interim hypotheses.
3. Final committed segment emission with word-level alignments.
4. Clean session completion.
"""

import sys
from pathlib import Path

# Add src to python path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sde4_stt.adapters.fake_transcriber import FakeTranscriber
from sde4_stt.schemas.audio import AudioChunk


def main():
    print("=" * 65)
    print("  ACADEMIA AI VOICE CO-PILOT — SDE 4 (W2) STT DEMONSTRATION")
    print("=" * 65)

    transcriber = FakeTranscriber(
        canned_transcript="Why is this application showing this error"
    )

    chunks = [
        AudioChunk(chunk_id="chunk_0", data=b"raw_audio_slice_1", sequence_number=0, duration_seconds=0.75),
        AudioChunk(chunk_id="chunk_1", data=b"raw_audio_slice_2", sequence_number=1, duration_seconds=0.75),
        AudioChunk(chunk_id="chunk_2", data=b"raw_audio_slice_3", sequence_number=2, duration_seconds=0.75, is_last=True),
    ]

    print("\nStreaming 3 audio chunks into FakeTranscriber (Session: 'demo_session_101'):\n")

    for event in transcriber.transcribe_stream(chunks, session_id="demo_session_101"):
        seg = event.segment
        if seg:
            status_flag = "FINAL" if seg.is_final else "INTERIM"
            print(f"[{event.event_type.value:<8}] [{status_flag:<7}] Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s | Confidence: {seg.confidence:.2f}")
            print(f"           Text: \"{seg.text}\"")
            if seg.words and seg.is_final:
                print("           Word Timings:")
                for w in seg.words:
                    print(f"             • {w.word:<12} [{w.start_time:.2f}s -> {w.end_time:.2f}s]")
            print()
        else:
            print(f"[{event.event_type.value:<8}] (Session Lifecycle Event: {event.event_id})\n")

    print("=" * 65)
    print("  SIMULATION COMPLETED WITH 100% SUCCESS")
    print("=" * 65)


if __name__ == "__main__":
    main()
