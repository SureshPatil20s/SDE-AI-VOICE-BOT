# SDE 4 – Workstream W2: Speech-to-Text Pipeline (Week 1)

**Student:** Suresh V Patil  
**Team:** SDE 4  
**Workstream:** W2 – Speech-to-text  
**Project:** Academia – AI Voice Co-Pilot  

---

## Architecture & Contract Overview

The speech-to-text pipeline serves as the auditory foundation for the Academia AI Voice Co-Pilot. It consumes raw audio chunks and streams structured, validated transcription events into the system.

Later workstreams (W1 Voice Ingestion, Context Engine, Agent Orchestrator, LLM Reasoning) depend strictly on these contracts.

```
Audio Input (Mic/Stream)
         │
         ▼
[ Provider-Neutral Adapter ]
         │
         ▼
[ TranscriptEvent Stream ]
   ├── START
   ├── INTERIM (is_final=False)
   ├── FINAL   (is_final=True)
   ├── COMPLETE
   └── ERROR
         │
         ▼
[ Segment Accumulator ]
         │
         ▼
Ordered Full Utterance / Transcript
```

---

## Day 1–2 Deliverables: Transcript Schemas

Defined in [`src/sde4_stt/schemas/transcript.py`](src/sde4_stt/schemas/transcript.py):

### 1. `TranscriptEventType` (Enum)
- `START`: Initialization of a recognition session or utterance.
- `INTERIM`: Non-final hypothesis (subject to correction as subsequent audio arrives).
- `FINAL`: Committed, finalized segment.
- `COMPLETE`: Clean termination of audio stream.
- `ERROR`: An operational, audio format, or network failure.

### 2. `WordTiming` (Model)
Word-level alignment details:
| Field | Type | Units / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `word` | `str` | Non-empty string | The transcribed token |
| `start_time` | `float` | Seconds ($\ge 0.0$) | Offset from audio onset |
| `end_time` | `float` | Seconds ($\ge \text{start\_time}$) | Offset from audio onset |
| `confidence` | `float \| None` | $[0.0, 1.0]$ | Confidence score |

### 3. `TranscriptSegment` (Model)
Individual segment of transcribed speech:
| Field | Type | Units / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `segment_id` | `str` | Non-empty, trimmed | Unique segment ID |
| `sequence_number`| `int` | $\ge 0$ | Monotonic segment index |
| `text` | `str` | Cannot be empty if final | Transcribed text |
| `start_time` | `float` | Seconds ($\ge 0.0$) | Segment start offset |
| `end_time` | `float` | Seconds ($\ge \text{start\_time}$) | Segment end offset |
| `is_final` | `bool` | Default `False` | Committed state flag |
| `confidence` | `float` | $[0.0, 1.0]$, Default `1.0` | Segment confidence score |
| `language` | `str` | BCP-47 / ISO 639-1 format | Language code (e.g. `'en'`, `'en-US'`) |
| `words` | `List[WordTiming] \| None`| Boundary validated | Optional word timestamps |
| `speaker_id` | `str \| None` | Optional string | Speaker diarization tag |

### 4. `TranscriptError` (Model)
Structured diagnostic payload attached to `ERROR` events:
| Field | Type | Units / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `code` | `str` | Non-empty string | Standard error code |
| `message` | `str` | Non-empty string | Explanatory message |
| `details` | `dict \| None`| Key-value dictionary | Diagnostic metadata |
| `recoverable`| `bool` | `True` or `False` | Whether client can retry |

### 5. `TranscriptEvent` (Model)
Streamable wrapper combining session metadata with segment or error data:
| Field | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `event_id` | `str` | Non-empty string | Unique event identifier |
| `session_id` | `str` | Non-empty string | Associated session ID |
| `event_type` | `TranscriptEventType` | Valid enum | Lifecycle stage |
| `timestamp` | `float` | Epoch seconds ($\ge 0.0$) | Generation timestamp |
| `segment` | `TranscriptSegment \| None` | Required for `INTERIM` & `FINAL` | Segment payload |
| `error` | `TranscriptError \| None` | Required for `ERROR` | Error payload |

---

## Validation & Edge Cases Handled

- ✅ Negative timestamps are rejected with explicit `ValidationError`.
- ✅ Inverted intervals (`end_time < start_time`) are caught by model validators.
- ✅ Word timings exceeding segment boundaries are strictly forbidden.
- ✅ Empty/whitespace-only text in `is_final=True` segments is rejected.
- ✅ Event type alignment: `FINAL` events must carry `is_final=True` segments; `INTERIM` events must carry `is_final=False` segments.
- ✅ Extra unexpected fields are forbidden (`extra="forbid"`) to prevent schema drift across teams.

---

## Day 3–4 Deliverables: Provider-Neutral Adapter & Deterministic Fake Transcriber

Defined in:
- [`src/sde4_stt/adapters/base.py`](src/sde4_stt/adapters/base.py): Abstract base class `BaseTranscriberAdapter`.
- [`src/sde4_stt/adapters/fake_transcriber.py`](src/sde4_stt/adapters/fake_transcriber.py): Zero-API-key mock transcriber `FakeTranscriber`.

### Key Capabilities:
- **Provider-Neutral Interface:** Clean separation of concerns with `start_session()`, `process_chunk()`, `finish_session()`, and `transcribe_stream()`.
- **100% Deterministic:** Repeated runs with the same audio chunks and session ID generate identical events, hashes, word timings, and confidence scores.
- **Progressive Streaming:** Emits `START` $\to$ `INTERIM` (partial hypotheses) $\to$ `FINAL` (committed segment with word timestamps) $\to$ `COMPLETE`.
- **Flexible Modes:**
  - *Scripted Mode:* Provide a `canned_transcript` to test specific test phrases progressively.
  - *Generative Hash Mode:* Deterministically computes words from chunk SHA-256 digests over a standardized development vocabulary.
- **Chaos / Error Simulation:** Configurable `fail_at_chunk_index` to test downstream failure recovery with standard `TranscriptError` codes.

---

## Running the Unit Tests

```bash
# Run all tests (Schemas + Adapters)
python -m pytest -v

# Run only adapter tests
python -m pytest tests/test_adapters.py -v
```
