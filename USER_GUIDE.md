GovBA Assistant — User Guide

1. Purpose

GovBA Assistant is a prototype that helps employees perform selected Business Analysis, documentation, communication, citizen-support, and visualization tasks.

All outputs are preliminary and require human verification.

2. Before Using the Application

Use only fictional, public, anonymized, or approved information.

Before running any task, select the confirmation box stating that the supplied information is appropriate for prototype use.

3. Generate a BRD Draft

Select Generate a BRD Draft.

Paste a service description, meeting notes, or interview notes, or upload a supported document.

Select the data-confirmation checkbox.

Click Generate BRD Draft.

Review the generated sections.

Download the Word or JSON output.

Check all requirements and missing-information notes before using the draft.

4. Review a BRD or SRS

Select Review a BRD or SRS.

Select BRD or SRS.

Paste or upload the document.

Confirm the data-use statement.

Click the review button.

Examine the completeness indicator, checklist, missing sections, wording issues, and recommendations.

Download the Word or JSON report.

The completeness indicator is not an official compliance score.

5. Compare Two Documents

Select Compare Two Documents.

Paste both documents or upload two supported files.

Name Document A and Document B.

Confirm the data-use statement.

Click Compare Documents.

Review matches, partial matches, missing items, additions, and possible conflicts.

Download the Word or JSON report.

A similarity result does not prove that two requirements are equivalent.

6. Summarize an Employee Email

Select Summarize an Employee Email.

Paste the email or upload a supported document.

Confirm the data-use statement.

Click Summarize Employee Email.

Review the short summary, action items, deadlines, decisions, and priority.

Download the TXT or JSON output.

Always review the original email before taking action.

7. Answer a Citizen Question

Select Answer a Citizen Question.

Paste or upload approved governmental reference information.

Enter the citizen question.

Confirm the data-use statement.

Click Answer Citizen Question.

Review the answer status, confidence indicator, supporting passage, missing information, and human-review notes.

Download the TXT or JSON output.

The application only uses the supplied information. It does not browse the internet or connect to an official database.

An authorized employee must approve the final answer before it is shared with a citizen.

8. Create a Visualization

Select Create a Visualization.

Choose Bar Chart, Line Chart, Pie Chart, or Gantt Chart.

Paste structured data or upload a CSV/TXT file.

Enter a chart title.

Add optional axis labels where available.

Confirm the data-use statement.

Click Create Visualization.

Review the chart.

Download the PNG image and optional JSON metadata.

Bar, Line, and Pie Format

Department,Requests
Licensing,120
Payments,85
Support,60

Gantt Format

Task,Start,End
Requirements,2026-08-01,2026-08-05
Development,2026-08-06,2026-08-15
Testing,2026-08-16,2026-08-20

Gantt dates must use YYYY-MM-DD.

9. Supported File Types

TXT

CSV for visualization

DOCX

Text-based PDF

Scanned image-only PDFs are not currently supported.

10. Common Problems

The task button is disabled

Select the data-confirmation checkbox.

A document cannot be read

Confirm that it is TXT, DOCX, or a text-based PDF.

A chart value is rejected

Check that the value column contains numbers only.

A Gantt chart date is rejected

Use YYYY-MM-DD and ensure the end date is not before the start date.

A citizen answer is reported as not found

The supplied source does not contain enough supporting information. Do not treat this as an error and do not invent an answer.

The application stops running in Codespaces

Run:

python -m streamlit run app.py

11. Privacy and Security

Do not enter confidential, personal, restricted, or classified data.

Do not enter passwords, private API keys, or real citizen records.

The prototype is not yet approved for production government use.

12. Final Responsibility

The employee or Business Analyst remains responsible for verifying every result before use.