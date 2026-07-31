GovBA Assistant v1.0.0 — Internship Prototype

Overview

GovBA Assistant is a deployed Streamlit prototype supporting selected Business Analysis, employee-support, citizen-support, and visualization activities.

Live application:

https://govba-assistant.streamlit.app/

Included Functions

Generate a preliminary BRD draft.

Review a BRD or SRS.

Compare two requirements-related documents.

Summarize employee email content.

Answer citizen questions using supplied reference information.

Create bar, line, pie, and Gantt visualizations with PNG export.

Main Exports

DOCX reports

JSON structured outputs

TXT summaries

PNG visualizations

Testing Status

Automated smoke tests included.

Manual testing checklist included.

All six tasks tested in the deployed Streamlit application.

Download functions tested.

Safety Controls

Mandatory data-use confirmation.

Human-review warnings.

No Ministry-system integration.

No Gmail or Outlook integration.

No official government-database connection.

No API keys included.

Unsupported citizen answers are withheld.

Technology

Python

Streamlit

PyMuPDF

python-docx

Matplotlib

Important Limitations

The current processing engines are rule-based demonstrations.

Outputs are preliminary and require human verification.

Citizen responses are not official government responses.

The prototype is not approved for production or confidential government data.

Scanned image-only PDF files are not supported.

Direct AI API integration is reserved for a later development phase.

Recommended Next Phase

Phase 2 may add optional AI-model integration, stronger Arabic-language support, authentication, approved knowledge sources, audit logging, and enhanced document traceability.