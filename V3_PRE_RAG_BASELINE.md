# GovBA-GAR V3 Pre-RAG Engineering Baseline

## Baseline Purpose

This document records the validated GovBA-GAR V3 system state immediately
before authoritative retrieval-augmented generation (RAG) development begins.

The baseline is intended to support software regression control, experimental
reproducibility, and later comparison between the pre-RAG and RAG-enabled
architectures.

## Baseline Architecture

Branch:

`v3-governance-rag`

AI gateway instrumentation milestone:

`1f9c11c - Instrument AI gateway with privacy-safe telemetry`

Earlier V3 milestones:

- `1ee7431 - Add fail-safe V3 telemetry recorder`
- `334150b - Add privacy-safe V3 telemetry foundation`
- `6f12d92 - Establish GovBA-GAR v3 architecture foundation`

V2 reference baseline:

- `cb1e550 - Add AI provider safety test`
- Tag: `v2.0.0`

## Runtime Environment

- Python: 3.12.1
- Streamlit: 1.60.0
- OpenAI Python SDK: 2.52.0
- PyMuPDF: 1.28.0
- python-docx: 1.2.0
- matplotlib: 3.11.1

## Legacy Functional Smoke Tests

The following seven existing GovBA capabilities passed:

1. Document reader
2. BRD generation and Word export
3. BRD review and Word export
4. Two-document comparison and Word export
5. Employee email summary
6. Source-grounded citizen Q&A
7. Visualization and PNG export

Result:

`7 passed | 0 failed`

## Existing AI Safety Tests

The following offline safety tests passed:

- `test_ai_provider.py`
- `test_ai_email_summary.py`

Both tests explicitly avoid external OpenAI requests.

## V3 Automated Tests

The following V3 suites passed:

- `test_v3_telemetry.py`
- `test_v3_telemetry_recorder.py`
- `test_v3_ai_provider_telemetry.py`

Result:

`27 tests passed`

The V3 tests cover:

- privacy-safe telemetry schema
- unique run identifiers
- JSON serialization
- latency measurement
- token-usage normalization
- cached-token extraction
- reasoning-token extraction
- disabled-by-default telemetry
- append-only JSONL recording
- recorder failure isolation
- provider-not-ready fallback
- mocked OpenAI success
- mocked OpenAI failure
- missing usage information
- empty responses
- incomplete responses
- structured JSON responses
- telemetry failure isolation
- provider validation compatibility

## Engineering Quality Gate

Whole-project compilation:

`PASS`

Command:

`python -m compileall -q .`

Exit code:

`0`

Git whitespace validation:

`PASS`

Command:

`git diff --check`

## Live Provider Test Policy

`test_ai_connection.py` is intentionally excluded from routine regression
testing because it can make a real OpenAI API request when the provider is
configured and enabled.

Live-provider testing must therefore be performed deliberately rather than as
part of the default offline regression suite.

## Privacy and Telemetry Position

GovBA-GAR telemetry is designed to avoid recording raw prompts, email bodies,
uploaded-document contents, API keys, access codes, or personal identifiers by
default.

Telemetry recording is opt-in and failures in telemetry recording must not
interrupt normal GovBA operation.

## Baseline Decision

The pre-RAG V3 architecture is considered technically stable for the next
development phase.

Future RAG, verification, temporal-validity, bilingual-retrieval, governance,
and agentic functionality should be regression-tested against this baseline.
