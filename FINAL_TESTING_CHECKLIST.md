GovBA Assistant — Final Manual Testing Checklist

Complete this checklist using fictional, public, anonymized, or approved information only.

A. Application startup

python -m py_compile completes without an error.

python smoke_tests.py reports all tests as PASS.

python -m streamlit run app.py opens the application.

The interface shows prototype version 0.12.

All six tasks appear in the task selector.

The data-confirmation checkbox is visible.

The task button remains disabled until the checkbox is selected.

B. BRD generation

Pasted service information generates a preliminary BRD.

Service overview, stakeholders, requirements, process, and review tabs appear.

Word BRD download opens successfully.

JSON download contains structured data.

Missing information and human-review notes appear.

C. BRD/SRS review

BRD review displays a completeness indicator.

Section checklist appears.

Missing sections and wording issues appear.

SRS selection also runs without an error.

Word and JSON review downloads open successfully.

D. Two-document comparison

Pasted BRD and SRS texts can be compared.

Partial matches appear.

The 5-day versus 7-day difference appears as a possible conflict.

Missing and additional requirements appear.

Uploading two TXT files works.

Word and JSON comparison downloads open successfully.

E. Employee email summarization

Email headers are extracted.

Priority is displayed.

Summary bullets, actions, deadlines, and decisions appear.

TXT and JSON downloads open successfully.

A warning states that the original email must be reviewed.

F. Citizen question answering

A supported documents question returns a source-grounded answer.

Supporting information is displayed.

The home-delivery question is reported as not found.

The application does not invent unsupported information.

TXT and JSON downloads open successfully.

The non-official-response warning appears.

G. Visualization

Bar chart preview appears.

Line chart preview appears.

Pie chart preview appears.

Gantt chart preview appears.

CSV upload works.

PNG downloads open as valid images.

Invalid numeric data produces a readable error.

Invalid Gantt dates produce a readable error.

H. File support and safeguards

TXT upload works.

DOCX upload works.

A text-based PDF upload works.

CSV upload works for visualization.

An empty input is not processed.

Inputs above the character limit are blocked.

No confidential Ministry data is used.

No Gmail, Outlook, government database, or Ministry system is connected.

I. Regression check

After testing the newest task, quickly repeat one successful test for every older task.

BRD generation still works.

Document review still works.

Document comparison still works.

Email summarization still works.

Citizen Q&A still works.

Visualization still works.

Final result

All automated tests pass.

All essential manual tests pass.

Any known prototype limitations are documented.

The tested version is committed and pushed to GitHub.