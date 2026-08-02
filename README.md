GovBA Assistant

GovBA Assistant is a cloud-ready internship prototype that supports selected Business Analysis and government-service documentation activities.

The application is built with Python and Streamlit and currently uses controlled rule-based engines. It does not connect to Ministry systems, email accounts, official government databases, or paid AI APIs.

## Live Prototype

GovBA Assistant is deployed on Streamlit Community Cloud:

[Open GovBA Assistant](https://govba-assistant.streamlit.app/)

Prototype Version
0.12

Main Functions
1. Generate a BRD Draft
2. Converts service descriptions, meeting notes, or interview notes into a preliminary Business Requirements Document structure.
3. Exports results as Word and JSON.
4. Review a BRD or SRS
5. Checks expected sections, missing content, vague wording, and areas requiring Business Analyst confirmation.
6. Exports review results as Word and JSON.
7. Compare Two Documents
8. Compares two requirements-related documents.
9. Identifies matches, partial matches, missing items, additional items, and possible numerical conflicts.
10. Exports results as Word and JSON.
11. Summarize an Employee Email
12. Converts pasted or uploaded email content into short bullets.
13. Extracts action items, deadlines, decisions, and a priority indicator.
14. Exports results as TXT and JSON.
15. Answer a Citizen Question
16. Answers a question using only supplied public, fictional, anonymized, or approved governmental reference information.
17. Shows supporting passages and refuses unsupported answers.
18. Exports results as TXT and JSON.
19. Create a Visualization
20. Creates bar, line, pie, and Gantt charts from structured data.
21. Supports pasted data and CSV/TXT uploads.
22. Exports charts as PNG and metadata as JSON.

Supported Files
1. TXT
2. CSV for visualization
3. DOCX
4. Text-based PDF
5. Scanned image-only PDF files are not supported in this prototype.


Technology
Python
Streamlit
PyMuPDF
python-docx
Matplotlib


Project Structure

govba-assistant
├── .streamlit
│   └── config.toml
├── app.py
├── citizen_qa_engine.py
├── comparison_engine.py
├── comparison_word_exporter.py
├── demo_engine.py
├── document_reader.py
├── email_summary_engine.py
├── prompts.py
├── review_engine.py
├── review_word_exporter.py
├── smoke_tests.py
├── visualization_engine.py
├── word_exporter.py
├── FINAL_TESTING_CHECKLIST.md
├── USER_GUIDE.md
├── README.md
└── requirements.txt

Local or Codespaces Setup

Install the required packages:

python -m pip install -r requirements.txt

Check the Python files:

python -m py_compile app.py smoke_tests.py visualization_engine.py citizen_qa_engine.py email_summary_engine.py comparison_engine.py comparison_word_exporter.py document_reader.py demo_engine.py prompts.py review_engine.py review_word_exporter.py word_exporter.py

Run the automated tests:

python smoke_tests.py

Start the application:

python -m streamlit run app.py

Data Safety

Use only information that is:

Fictional

Public

Anonymized

Explicitly approved for prototype use

Do not enter:

Confidential Ministry data

Personal citizen records

Passwords or API keys

Real employee email content without approval

Restricted government documents

Important Limitations

Outputs are preliminary and require human review.

Completeness and confidence indicators are prototype indicators, not official scores.

Similarity does not prove legal, technical, or operational equivalence.

Citizen answers are not official government responses.

The application does not browse the internet.

The application does not connect to Gmail, Outlook, Ministry systems, or government databases.

The current engines are rule-based demonstrations and may miss context or produce imperfect classifications.

Human Review Requirement

Every generated output must be reviewed by an authorized employee or Business Analyst before it is used, shared, approved, or treated as official.

Testing

Automated smoke tests are included in:

smoke_tests.py

The manual testing checklist is included in:

FINAL_TESTING_CHECKLIST.md

Future Development

Possible future improvements include:

Optional AI API integration

Better Arabic-language processing

Secure authentication

Approved knowledge-base integration

Role-based access controls

Audit logs

Improved document traceability

Direct spreadsheet upload

More visualization types

Official deployment security review

Project Status

The application is ready for final documentation review and cloud deployment as an internship prototype
