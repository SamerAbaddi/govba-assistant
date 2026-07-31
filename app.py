import json

import streamlit as st

from comparison_engine import compare_requirements_documents
from comparison_word_exporter import create_comparison_word_report
from citizen_qa_engine import answer_citizen_question
from demo_engine import generate_demo_brd
from document_reader import read_uploaded_file
from email_summary_engine import summarize_employee_email
from review_engine import review_requirements_document
from review_word_exporter import create_review_word_report
from visualization_engine import (
    SUPPORTED_CHART_TYPES,
    create_visualization,
)
from word_exporter import create_brd_word_report


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="GovBA Assistant",
    page_icon="🏛️",
    layout="wide",
)

MAX_CHARACTERS_PER_DOCUMENT = 50_000
MAX_QUESTION_CHARACTERS = 1_000

st.markdown(
    """
    <style>
    .govba-banner {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 14px;
        margin-bottom: 1rem;
    }

    .govba-banner h1 {
        margin: 0;
        font-size: 2rem;
    }

    .govba-banner p {
        margin: 0.35rem 0 0 0;
        opacity: 0.82;
    }

    .govba-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0.55rem 0.35rem 0 0;
        border: 1px solid rgba(120, 120, 120, 0.35);
        border-radius: 999px;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_list(title: str, items: list[str]) -> None:
    """Display a heading followed by a simple bullet list."""

    st.markdown(f"### {title}")

    if not items:
        st.write("- Requires confirmation")
        return

    for item in items:
        st.markdown(f"- {item}")


def display_review_checklist(checklist: list[dict]) -> None:
    """Display the document-review checklist as a clear table."""

    rows = []

    for item in checklist:
        evidence = item.get("evidence", [])

        rows.append(
            {
                "Section": item.get("section", ""),
                "Status": item.get("status", ""),
                "Evidence": (
                    ", ".join(evidence)
                    if evidence
                    else "No clear evidence detected"
                ),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Application heading
# ---------------------------------------------------------
st.markdown(
    """
    <div class="govba-banner">
        <h1>🏛️ GovBA Assistant</h1>
        <p>
            AI-supported business analysis and government-service
            documentation prototype.
        </p>
        <span class="govba-badge">Cloud-based</span>
        <span class="govba-badge">Human-reviewed</span>
        <span class="govba-badge">No Ministry integration</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How this prototype works"):
    st.write(
        "1. Select a task. "
        "2. Paste text or upload supported documents. "
        "3. Run the rule-based demonstration. "
        "4. Review the findings. "
        "5. Download Word or JSON outputs."
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Prototype Status")

    st.success("Level 2 — Polished Multi-Task Demo")

    st.info(
        "The application currently uses rule-based demo engines. "
        "The AI connection will be added later."
    )

    st.markdown(
        "**Data rule:** Use only fictional, public, "
        "or approved non-confidential information."
    )

    st.divider()

    st.markdown("**Available tasks**")
    st.write("• Generate a BRD draft")
    st.write("• Review a BRD or SRS")
    st.write("• Compare two documents")
    st.write("• Summarize an employee email")
    st.write("• Answer a citizen question")
    st.write("• Create a visualization")

    st.divider()

    st.markdown("**Supported files**")
    st.write("• TXT")
    st.write("• CSV")
    st.write("• Word DOCX")
    st.write("• Text-based PDF")

    st.divider()

    st.caption("Prototype version: 0.12")

    st.warning(
        "All generated outputs require review "
        "and approval by a Business Analyst."
    )


# ---------------------------------------------------------
# Task selection
# ---------------------------------------------------------
st.subheader("1. Select the Task")

task = st.selectbox(
    "What would you like the assistant to do?",
    [
        "Generate a BRD Draft",
        "Review a BRD or SRS",
        "Compare Two Documents",
        "Summarize an Employee Email",
        "Answer a Citizen Question",
        "Create a Visualization",
    ],
)

document_type = "BRD"

if task == "Review a BRD or SRS":
    document_type = st.radio(
        "Select the document type:",
        ["BRD", "SRS"],
        horizontal=True,
    )

task_descriptions = {
    "Generate a BRD Draft": (
        "Convert meeting notes, interview notes, or a service "
        "description into a preliminary BRD structure."
    ),
    "Review a BRD or SRS": (
        "Check expected sections, unclear wording, and areas "
        "that require Business Analyst confirmation."
    ),
    "Compare Two Documents": (
        "Compare two requirements-related documents and identify "
        "matches, missing items, additions, and possible conflicts."
    ),
    "Summarize an Employee Email": (
        "Turn pasted or uploaded email content into short bullets, "
        "action items, deadlines, decisions, and a priority indicator."
    ),
    "Answer a Citizen Question": (
        "Answer a citizen question using only supplied public, "
        "approved, or fictional governmental reference information."
    ),
    "Create a Visualization": (
        "Turn structured data into a bar, line, pie, or Gantt chart "
        "and download the result as a PNG image."
    ),
}

st.info(task_descriptions[task])


# ---------------------------------------------------------
# Input method
# ---------------------------------------------------------
st.subheader("2. Provide the Source Information")

source_text = ""
source_name = "Manually entered text"

document_a_text = ""
document_b_text = ""
document_a_name = "Document A"
document_b_name = "Document B"
citizen_question = ""

chart_type = "Bar Chart"
chart_title = ""
x_axis_label = ""
y_axis_label = ""

if task == "Compare Two Documents":

    comparison_input_method = st.radio(
        "Choose how to provide both documents:",
        ["Paste both texts", "Upload two documents"],
        horizontal=True,
    )

    if comparison_input_method == "Paste both texts":

        comparison_column_1, comparison_column_2 = st.columns(2)

        with comparison_column_1:
            document_a_name = st.text_input(
                "Document A name:",
                value="BRD",
            )

            document_a_text = st.text_area(
                "Paste Document A:",
                height=280,
                placeholder=(
                    "Example: Applicants must provide an "
                    "identification number..."
                ),
            )

        with comparison_column_2:
            document_b_name = st.text_input(
                "Document B name:",
                value="SRS",
            )

            document_b_text = st.text_area(
                "Paste Document B:",
                height=280,
                placeholder=(
                    "Example: The applicant shall submit an "
                    "identification number..."
                ),
            )

    else:

        upload_column_1, upload_column_2 = st.columns(2)

        with upload_column_1:
            uploaded_file_a = st.file_uploader(
                "Upload Document A:",
                type=["txt", "docx", "pdf"],
                key="comparison_file_a",
            )

        with upload_column_2:
            uploaded_file_b = st.file_uploader(
                "Upload Document B:",
                type=["txt", "docx", "pdf"],
                key="comparison_file_b",
            )

        if uploaded_file_a is not None:

            try:
                document_a_text = read_uploaded_file(
                    uploaded_file_a
                )
                document_a_name = uploaded_file_a.name

                st.success(
                    f"Document A '{document_a_name}' "
                    "was read successfully."
                )

            except ValueError as error:
                st.error(f"Document A: {error}")

            except Exception as error:
                st.error(
                    "Document A could not be read."
                )
                st.caption(
                    f"Technical detail: {error}"
                )

        if uploaded_file_b is not None:

            try:
                document_b_text = read_uploaded_file(
                    uploaded_file_b
                )
                document_b_name = uploaded_file_b.name

                st.success(
                    f"Document B '{document_b_name}' "
                    "was read successfully."
                )

            except ValueError as error:
                st.error(f"Document B: {error}")

            except Exception as error:
                st.error(
                    "Document B could not be read."
                )
                st.caption(
                    f"Technical detail: {error}"
                )

        if document_a_text or document_b_text:

            with st.expander(
                "Preview extracted comparison texts"
            ):

                preview_column_1, preview_column_2 = st.columns(2)

                with preview_column_1:
                    st.markdown("**Document A preview**")
                    st.text(document_a_text[:2000])

                with preview_column_2:
                    st.markdown("**Document B preview**")
                    st.text(document_b_text[:2000])

elif task == "Answer a Citizen Question":

    st.info(
        "The answer will be based only on the reference information "
        "you supply below. The prototype does not browse the internet "
        "or connect to an official government database."
    )

    citizen_source_method = st.radio(
        "Choose how to provide the governmental reference information:",
        ["Paste reference information", "Upload a reference document"],
        horizontal=True,
    )

    if citizen_source_method == "Paste reference information":

        source_text = st.text_area(
            "Paste the approved, public, or fictional governmental information:",
            height=280,
            placeholder=(
                "Example: Licence renewal requires a valid national "
                "identification card and proof of payment. The service "
                "fee is 20 JOD. Processing takes three working days..."
            ),
        )

    else:

        uploaded_reference_file = st.file_uploader(
            "Upload a TXT, DOCX, or text-based PDF reference document:",
            type=["txt", "docx", "pdf"],
            key="citizen_reference_file",
            help=(
                "Use only public, fictional, anonymized, or approved "
                "non-confidential information."
            ),
        )

        if uploaded_reference_file is not None:

            try:
                source_text = read_uploaded_file(
                    uploaded_reference_file
                )
                source_name = uploaded_reference_file.name

                st.success(
                    f"Reference document '{source_name}' "
                    "was read successfully."
                )

                st.write(
                    f"Extracted characters: {len(source_text):,}"
                )

                with st.expander(
                    "Preview extracted governmental information"
                ):
                    preview_text = source_text[:3000]

                    if len(source_text) > 3000:
                        preview_text += (
                            "\n\n[Preview shortened. "
                            "The full text remains available.]"
                        )

                    st.text(preview_text)

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "The governmental reference document "
                    "could not be read."
                )
                st.caption(
                    f"Technical detail: {error}"
                )

    citizen_question = st.text_area(
        "Enter the citizen's question:",
        height=110,
        max_chars=MAX_QUESTION_CHARACTERS,
        placeholder=(
            "Example: What documents are required "
            "for licence renewal?"
        ),
    )

elif task == "Create a Visualization":

    st.info(
        "Provide structured comma-separated data. "
        "The application will create a chart directly from the "
        "supplied values without changing or estimating them."
    )

    chart_type = st.selectbox(
        "Select the visualization type:",
        SUPPORTED_CHART_TYPES,
    )

    if chart_type == "Gantt Chart":
        st.markdown("**Required Gantt format**")
        st.code(
            "Task,Start,End\n"
            "Requirements,2026-08-01,2026-08-05\n"
            "Development,2026-08-06,2026-08-15\n"
            "Testing,2026-08-16,2026-08-20",
            language="text",
        )
        st.caption(
            "Use YYYY-MM-DD for every start and end date."
        )
    else:
        st.markdown(
            f"**Required {chart_type.lower()} format**"
        )
        st.code(
            "Department,Requests\n"
            "Licensing,120\n"
            "Payments,85\n"
            "Support,60",
            language="text",
        )
        st.caption(
            "The first column contains categories and the "
            "second column contains numeric values."
        )

    visualization_input_method = st.radio(
        "Choose how to provide the structured data:",
        ["Paste structured data", "Upload CSV or TXT"],
        horizontal=True,
    )

    if visualization_input_method == "Paste structured data":

        source_text = st.text_area(
            "Paste the chart data:",
            height=240,
            placeholder=(
                "Department,Requests\n"
                "Licensing,120\n"
                "Payments,85\n"
                "Support,60"
            ),
        )

    else:

        uploaded_chart_file = st.file_uploader(
            "Upload a CSV or TXT data file:",
            type=["csv", "txt"],
            key="visualization_data_file",
        )

        if uploaded_chart_file is not None:

            try:
                source_text = uploaded_chart_file.getvalue().decode(
                    "utf-8-sig"
                )
                source_name = uploaded_chart_file.name

                st.success(
                    f"Data file '{source_name}' "
                    "was read successfully."
                )

                with st.expander("Preview uploaded data"):
                    st.text(source_text[:3000])

            except UnicodeDecodeError:
                st.error(
                    "The uploaded file could not be decoded. "
                    "Save it as a UTF-8 CSV or TXT file and try again."
                )

    chart_title = st.text_input(
        "Chart title:",
        placeholder=(
            "Example: Service Requests by Department"
            if chart_type != "Gantt Chart"
            else "Example: GovBA Implementation Plan"
        ),
    )

    if chart_type != "Gantt Chart":
        label_column_1, label_column_2 = st.columns(2)

        with label_column_1:
            x_axis_label = st.text_input(
                "Horizontal-axis label (optional):"
            )

        with label_column_2:
            y_axis_label = st.text_input(
                "Vertical-axis label (optional):"
            )

else:

    input_method = st.radio(
        "Choose how to provide the information:",
        ["Paste text", "Upload a document"],
        horizontal=True,
    )

    if input_method == "Paste text":

        if task == "Generate a BRD Draft":
            text_label = (
                "Paste meeting notes, interview notes, "
                "or a service description:"
            )
            placeholder_text = (
                "Example: A government entity wants to automate "
                "a licence-renewal service. Applicants provide "
                "their identification number, existing licence "
                "information, and supporting documents..."
            )

        elif task == "Review a BRD or SRS":
            text_label = "Paste the BRD or SRS content to review:"
            placeholder_text = (
                "Example: Service Scope: The system will automate "
                "licence renewal. Applicants submit an identification "
                "number and proof of payment..."
            )

        else:
            text_label = "Paste the employee email content:"
            placeholder_text = (
                "Example:\nFrom: Ahmad Saleh\n"
                "To: Project Team\n"
                "Subject: Urgent review of the service document\n"
                "Please review the requirements and send your "
                "comments by tomorrow..."
            )

        source_text = st.text_area(
            text_label,
            height=260,
            placeholder=placeholder_text,
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload a TXT, DOCX, or text-based PDF document:",
            type=["txt", "docx", "pdf"],
            help=(
                "Scanned image-only PDF files are not supported "
                "at this stage."
            ),
        )

        if uploaded_file is not None:

            try:
                source_text = read_uploaded_file(uploaded_file)
                source_name = uploaded_file.name

                st.success(
                    f"Document '{uploaded_file.name}' "
                    "was read successfully."
                )

                st.write(
                    f"Extracted characters: {len(source_text):,}"
                )

                with st.expander(
                    "Preview extracted document text"
                ):

                    preview_text = source_text[:3000]

                    if len(source_text) > 3000:
                        preview_text += (
                            "\n\n[Preview shortened. "
                            "The full text remains available.]"
                        )

                    st.text(preview_text)

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "The document could not be read. "
                    "Please check the file and try again."
                )

                st.caption(f"Technical detail: {error}")


# ---------------------------------------------------------
# Input validation and data confirmation
# ---------------------------------------------------------
if task == "Compare Two Documents":
    document_a_length = len(document_a_text)
    document_b_length = len(document_b_text)

    if document_a_text or document_b_text:
        count_column_1, count_column_2 = st.columns(2)

        with count_column_1:
            st.caption(
                f"Document A size: {document_a_length:,} characters"
            )

        with count_column_2:
            st.caption(
                f"Document B size: {document_b_length:,} characters"
            )

    input_too_long = (
        document_a_length > MAX_CHARACTERS_PER_DOCUMENT
        or document_b_length > MAX_CHARACTERS_PER_DOCUMENT
    )

elif task == "Answer a Citizen Question":
    source_length = len(source_text)
    question_length = len(citizen_question)

    if source_text or citizen_question:
        count_column_1, count_column_2 = st.columns(2)

        with count_column_1:
            st.caption(
                f"Reference information size: "
                f"{source_length:,} characters"
            )

        with count_column_2:
            st.caption(
                f"Question size: {question_length:,} characters"
            )

    input_too_long = (
        source_length > MAX_CHARACTERS_PER_DOCUMENT
        or question_length > MAX_QUESTION_CHARACTERS
    )

else:
    source_length = len(source_text)

    if source_text:
        st.caption(
            f"Source size: {source_length:,} characters"
        )

    input_too_long = (
        source_length > MAX_CHARACTERS_PER_DOCUMENT
    )

if input_too_long:
    st.error(
        "One or more inputs exceed the prototype limit of "
        f"{MAX_CHARACTERS_PER_DOCUMENT:,} characters. "
        "Please shorten the document before continuing."
    )

data_confirmation = st.checkbox(
    "I confirm that the supplied information is fictional, public, "
    "anonymized, or approved for use in this prototype."
)

if not data_confirmation:
    st.caption(
        "Confirm the data-use statement to activate the task button."
    )


# ---------------------------------------------------------
# Run selected task
# ---------------------------------------------------------
st.subheader("3. Run the Selected Task")

if task == "Generate a BRD Draft":
    button_label = "Generate BRD Draft"

elif task == "Review a BRD or SRS":
    button_label = f"Review {document_type} Document"

elif task == "Compare Two Documents":
    button_label = "Compare Documents"

elif task == "Summarize an Employee Email":
    button_label = "Summarize Employee Email"

elif task == "Answer a Citizen Question":
    button_label = "Answer Citizen Question"

else:
    button_label = "Create Visualization"

if st.button(
    button_label,
    type="primary",
    use_container_width=True,
    disabled=(
        not data_confirmation
        or input_too_long
    ),
):

    if (
        task == "Compare Two Documents"
        and (
            not document_a_text.strip()
            or not document_b_text.strip()
        )
    ):

        st.warning(
            "Please provide two readable documents "
            "before starting the comparison."
        )

    elif (
        task == "Answer a Citizen Question"
        and (
            not source_text.strip()
            or not citizen_question.strip()
        )
    ):

        st.warning(
            "Please provide readable governmental reference "
            "information and enter a citizen question."
        )

    elif (
        task != "Compare Two Documents"
        and not source_text.strip()
    ):

        st.warning(
            "Please paste information or upload "
            "a readable document first."
        )

    elif task == "Generate a BRD Draft":

        try:
            result = generate_demo_brd(source_text)

            st.success(
                "Preliminary BRD generated successfully."
            )

            st.caption(
                "This result was generated by the temporary "
                "rule-based demo engine."
            )

            st.markdown(
                "## Preliminary Business Requirements Document"
            )

            st.write(f"**Source:** {source_name}")

            with st.expander(
                "View source preview",
                expanded=False,
            ):

                preview = source_text[:1500]

                if len(source_text) > 1500:
                    preview += "..."

                st.write(preview)

            overview = result["service_overview"]

            st.markdown("## Service Overview")

            overview_column_1, overview_column_2 = st.columns(2)

            with overview_column_1:
                st.write(
                    f"**Service name:** "
                    f"{overview['service_name']}"
                )

                st.write(
                    f"**Service purpose:** "
                    f"{overview['service_purpose']}"
                )

            with overview_column_2:
                st.write(
                    f"**Service scope:** "
                    f"{overview['service_scope']}"
                )

                st.write(
                    "**Analysis mode:** Rule-based demo"
                )

            tab_1, tab_2, tab_3, tab_4 = st.tabs(
                [
                    "People",
                    "Requirements",
                    "Process and Data",
                    "Review",
                ]
            )

            with tab_1:

                people_column_1, people_column_2 = st.columns(2)

                with people_column_1:
                    display_list(
                        "Stakeholders",
                        result["stakeholders"],
                    )

                with people_column_2:
                    display_list(
                        "Service Recipients",
                        result["service_recipients"],
                    )

            with tab_2:

                requirements_column_1, requirements_column_2 = (
                    st.columns(2)
                )

                with requirements_column_1:
                    display_list(
                        "Functional Requirements",
                        result["functional_requirements"],
                    )

                    display_list(
                        "Non-Functional Requirements",
                        result[
                            "non_functional_requirements"
                        ],
                    )

                with requirements_column_2:
                    display_list(
                        "Business Rules",
                        result["business_rules"],
                    )

                    display_list(
                        "Integration Requirements",
                        result[
                            "integration_requirements"
                        ],
                    )

            with tab_3:

                display_list(
                    "Process Steps",
                    result["process_steps"],
                )

                display_list(
                    "Required Data and Documents",
                    result[
                        "required_data_and_documents"
                    ],
                )

            with tab_4:

                review_column_1, review_column_2 = st.columns(2)

                with review_column_1:
                    display_list(
                        "Missing Information",
                        result["missing_information"],
                    )

                with review_column_2:
                    display_list(
                        "Human Review Notes",
                        result["human_review_notes"],
                    )

            st.divider()

            result_json = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )

            word_report = create_brd_word_report(
                result,
                source_name,
            )

            download_column_1, download_column_2 = st.columns(2)

            with download_column_1:
                st.download_button(
                    label="Download Word BRD Report",
                    data=word_report,
                    file_name="GovBA_Preliminary_BRD.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                )

            with download_column_2:
                st.download_button(
                    label="Download Structured JSON",
                    data=result_json,
                    file_name="govba_brd_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The BRD could not be generated."
            )

            st.caption(f"Technical detail: {error}")

    elif task == "Review a BRD or SRS":

        try:
            review_result = review_requirements_document(
                source_text,
                document_type,
            )

            st.success(
                f"{document_type} review completed successfully."
            )

            st.caption(
                "This review was generated by the temporary "
                "rule-based review engine."
            )

            st.markdown(
                f"## Preliminary {document_type} Review Report"
            )

            st.write(f"**Source:** {source_name}")
            st.write(
                f"**Review mode:** "
                f"{review_result['review_mode']}"
            )

            completeness = review_result[
                "completeness_indicator"
            ]

            metric_column_1, metric_column_2 = st.columns(2)

            with metric_column_1:
                st.metric(
                    "Completeness Indicator",
                    f"{completeness}%",
                )

            with metric_column_2:
                st.metric(
                    "Missing or Unclear Sections",
                    len(review_result["missing_sections"]),
                )

            st.progress(completeness / 100)

            st.info(
                "This percentage is a prototype completeness "
                "indicator, not an official Ministry score."
            )

            review_tab_1, review_tab_2, review_tab_3, review_tab_4 = (
                st.tabs(
                    [
                        "Section Checklist",
                        "Detected and Missing",
                        "Issues and Recommendations",
                        "Human Review",
                    ]
                )
            )

            with review_tab_1:
                display_review_checklist(
                    review_result["section_checklist"]
                )

            with review_tab_2:

                detected_column, missing_column = st.columns(2)

                with detected_column:
                    display_list(
                        "Detected Sections",
                        review_result["detected_sections"],
                    )

                with missing_column:
                    display_list(
                        "Missing or Unclear Sections",
                        review_result["missing_sections"],
                    )

            with review_tab_3:

                issue_column, recommendation_column = st.columns(2)

                with issue_column:
                    display_list(
                        "Wording Issues",
                        review_result["wording_issues"],
                    )

                with recommendation_column:
                    display_list(
                        "Recommendations",
                        review_result["recommendations"],
                    )

            with review_tab_4:
                display_list(
                    "Human Review Notes",
                    review_result["human_review_notes"],
                )

            st.divider()

            review_json = json.dumps(
                review_result,
                indent=2,
                ensure_ascii=False,
            )

            review_word_report = create_review_word_report(
                review_result,
                source_name,
            )

            review_download_column_1, review_download_column_2 = (
                st.columns(2)
            )

            with review_download_column_1:
                st.download_button(
                    label="Download Word Review Report",
                    data=review_word_report,
                    file_name=(
                        f"GovBA_{document_type}_Review.docx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                )

            with review_download_column_2:
                st.download_button(
                    label="Download Review Result as JSON",
                    data=review_json,
                    file_name=(
                        f"GovBA_{document_type}_Review.json"
                    ),
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"The {document_type} review could not be completed."
            )

            st.caption(f"Technical detail: {error}")


    elif task == "Compare Two Documents":

        try:
            comparison_result = compare_requirements_documents(
                document_a_text,
                document_b_text,
                document_a_name,
                document_b_name,
            )

            st.success(
                "Document comparison completed successfully."
            )

            st.caption(
                "This result was generated by the temporary "
                "rule-based comparison engine."
            )

            st.markdown(
                "## Preliminary Document Comparison Report"
            )

            st.write(
                f"**Document A:** {document_a_name}"
            )
            st.write(
                f"**Document B:** {document_b_name}"
            )

            summary = comparison_result["summary"]
            coverage = comparison_result[
                "coverage_indicator"
            ]

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            with metric_1:
                st.metric(
                    "Coverage Indicator",
                    f"{coverage}%",
                )

            with metric_2:
                st.metric(
                    "Matches",
                    summary["matched_count"],
                )

            with metric_3:
                st.metric(
                    "Missing in B",
                    summary["missing_count"],
                )

            with metric_4:
                st.metric(
                    "Possible Conflicts",
                    summary["possible_conflict_count"],
                )

            st.progress(coverage / 100)

            st.info(
                "The coverage indicator is a prototype measure, "
                "not an official traceability or compliance score."
            )

            (
                comparison_tab_1,
                comparison_tab_2,
                comparison_tab_3,
                comparison_tab_4,
            ) = st.tabs(
                [
                    "Matches",
                    "Missing and Additional",
                    "Possible Conflicts",
                    "Human Review",
                ]
            )

            with comparison_tab_1:

                st.markdown("### Matched Items")

                if comparison_result["matched_items"]:
                    for index, item in enumerate(
                        comparison_result["matched_items"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Match {index} — "
                            f"{item['similarity_score']:.0%} similarity**"
                        )
                        st.write(
                            f"**Document A:** "
                            f"{item['document_a_item']}"
                        )
                        st.write(
                            f"**Document B:** "
                            f"{item['document_b_item']}"
                        )
                        st.divider()
                else:
                    st.write("No full matches detected.")

                st.markdown("### Partial Matches")

                if comparison_result["partial_matches"]:
                    for index, item in enumerate(
                        comparison_result["partial_matches"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Partial match {index} — "
                            f"{item['similarity_score']:.0%} similarity**"
                        )
                        st.write(
                            f"**Document A:** "
                            f"{item['document_a_item']}"
                        )
                        st.write(
                            f"**Document B:** "
                            f"{item['document_b_item']}"
                        )
                        st.caption(item["review_note"])
                        st.divider()
                else:
                    st.write("No partial matches detected.")

            with comparison_tab_2:

                missing_column, additional_column = st.columns(2)

                with missing_column:
                    display_list(
                        f"Missing from {document_b_name}",
                        comparison_result[
                            "missing_in_document_b"
                        ],
                    )

                with additional_column:
                    display_list(
                        f"Additional in {document_b_name}",
                        comparison_result[
                            "additional_in_document_b"
                        ],
                    )

            with comparison_tab_3:

                conflicts = comparison_result[
                    "possible_conflicts"
                ]

                if conflicts:
                    for index, conflict in enumerate(
                        conflicts,
                        start=1,
                    ):
                        st.warning(
                            f"Possible conflict {index}"
                        )
                        st.write(
                            f"**Document A:** "
                            f"{conflict['document_a_item']}"
                        )
                        st.write(
                            f"**Document B:** "
                            f"{conflict['document_b_item']}"
                        )
                        st.write(
                            f"**Reason:** {conflict['reason']}"
                        )
                        st.write(
                            f"**Similarity:** "
                            f"{conflict['similarity_score']:.0%}"
                        )
                        st.divider()
                else:
                    st.success(
                        "No numerical conflicts were detected."
                    )

            with comparison_tab_4:
                display_list(
                    "Human Review Notes",
                    comparison_result[
                        "human_review_notes"
                    ],
                )

            st.divider()

            comparison_json = json.dumps(
                comparison_result,
                indent=2,
                ensure_ascii=False,
            )

            comparison_word_report = (
                create_comparison_word_report(
                    comparison_result
                )
            )

            comparison_download_1, comparison_download_2 = (
                st.columns(2)
            )

            with comparison_download_1:
                st.download_button(
                    label="Download Word Comparison Report",
                    data=comparison_word_report,
                    file_name=(
                        "GovBA_Document_Comparison.docx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                )

            with comparison_download_2:
                st.download_button(
                    label="Download Comparison Result as JSON",
                    data=comparison_json,
                    file_name="GovBA_Document_Comparison.json",
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The document comparison could not be completed."
            )

            st.caption(f"Technical detail: {error}")


    elif task == "Summarize an Employee Email":

        try:
            email_result = summarize_employee_email(
                source_text
            )

            st.success(
                "Employee email summarized successfully."
            )

            st.caption(
                "This result was generated by the temporary "
                "rule-based email-summary engine."
            )

            st.markdown(
                "## Preliminary Employee Email Summary"
            )

            header_column_1, header_column_2 = st.columns(2)

            with header_column_1:
                st.write(
                    f"**From:** {email_result['sender']}"
                )
                st.write(
                    f"**To:** {email_result['recipient']}"
                )

            with header_column_2:
                st.write(
                    f"**Subject:** {email_result['subject']}"
                )
                st.write(
                    f"**Priority:** {email_result['priority']}"
                )

            summary_metric_1, summary_metric_2, summary_metric_3 = (
                st.columns(3)
            )

            with summary_metric_1:
                st.metric(
                    "Summary Bullets",
                    len(email_result["summary_bullets"]),
                )

            with summary_metric_2:
                st.metric(
                    "Action Items",
                    len(email_result["action_items"]),
                )

            with summary_metric_3:
                deadline_count = sum(
                    1
                    for item in email_result["deadlines"]
                    if item != "No clear deadline detected."
                )

                st.metric(
                    "Detected Deadlines",
                    deadline_count,
                )

            email_tab_1, email_tab_2, email_tab_3, email_tab_4 = (
                st.tabs(
                    [
                        "Short Summary",
                        "Actions and Deadlines",
                        "Decisions",
                        "Human Review",
                    ]
                )
            )

            with email_tab_1:
                display_list(
                    "Email Summary",
                    email_result["summary_bullets"],
                )

            with email_tab_2:
                action_column, deadline_column = st.columns(2)

                with action_column:
                    display_list(
                        "Action Items",
                        email_result["action_items"],
                    )

                with deadline_column:
                    display_list(
                        "Deadlines",
                        email_result["deadlines"],
                    )

            with email_tab_3:
                display_list(
                    "Detected Decisions",
                    email_result["decisions"],
                )

            with email_tab_4:
                display_list(
                    "Human Review Notes",
                    email_result["human_review_notes"],
                )

            st.warning(
                "Always check the original email before taking "
                "action. This prototype does not connect to Gmail, "
                "Outlook, or a Ministry email system."
            )

            st.divider()

            email_json = json.dumps(
                email_result,
                indent=2,
                ensure_ascii=False,
            )

            email_text_report_lines = [
                "GovBA Assistant — Employee Email Summary",
                "",
                f"From: {email_result['sender']}",
                f"To: {email_result['recipient']}",
                f"Subject: {email_result['subject']}",
                f"Priority: {email_result['priority']}",
                "",
                "SHORT SUMMARY",
            ]

            email_text_report_lines.extend(
                f"- {item}"
                for item in email_result["summary_bullets"]
            )

            email_text_report_lines.extend(
                [
                    "",
                    "ACTION ITEMS",
                ]
            )

            email_text_report_lines.extend(
                f"- {item}"
                for item in email_result["action_items"]
            )

            email_text_report_lines.extend(
                [
                    "",
                    "DEADLINES",
                ]
            )

            email_text_report_lines.extend(
                f"- {item}"
                for item in email_result["deadlines"]
            )

            email_text_report_lines.extend(
                [
                    "",
                    "DECISIONS",
                ]
            )

            email_text_report_lines.extend(
                f"- {item}"
                for item in email_result["decisions"]
            )

            email_text_report_lines.extend(
                [
                    "",
                    "HUMAN REVIEW NOTES",
                ]
            )

            email_text_report_lines.extend(
                f"- {item}"
                for item in email_result["human_review_notes"]
            )

            email_text_report = "\n".join(
                email_text_report_lines
            )

            email_download_1, email_download_2 = st.columns(2)

            with email_download_1:
                st.download_button(
                    label="Download Email Summary as TXT",
                    data=email_text_report,
                    file_name="GovBA_Email_Summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with email_download_2:
                st.download_button(
                    label="Download Email Summary as JSON",
                    data=email_json,
                    file_name="GovBA_Email_Summary.json",
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The employee email could not be summarized."
            )

            st.caption(f"Technical detail: {error}")


    elif task == "Answer a Citizen Question":

        try:
            citizen_result = answer_citizen_question(
                source_text,
                citizen_question,
            )

            answer_supported = (
                citizen_result["answer_status"]
                == "Supported by supplied information"
            )

            if answer_supported:
                st.success(
                    "A source-grounded answer was identified."
                )
            else:
                st.warning(
                    "A sufficiently supported answer was not found."
                )

            st.caption(
                "This response was generated by the temporary "
                "source-grounded rule-based Q&A engine."
            )

            st.markdown(
                "## Preliminary Citizen Question Response"
            )

            st.write(
                f"**Reference source:** {source_name}"
            )

            metric_column_1, metric_column_2 = st.columns(2)

            with metric_column_1:
                st.metric(
                    "Answer Status",
                    citizen_result["answer_status"],
                )

            with metric_column_2:
                st.metric(
                    "Confidence Indicator",
                    citizen_result["confidence_indicator"],
                )

            st.markdown("### Citizen Question")
            st.write(citizen_result["question"])

            st.markdown("### Source-Grounded Answer")

            if answer_supported:
                st.success(citizen_result["answer"])
            else:
                st.warning(citizen_result["answer"])

            (
                citizen_tab_1,
                citizen_tab_2,
                citizen_tab_3,
            ) = st.tabs(
                [
                    "Supporting Information",
                    "Missing Information",
                    "Human Review",
                ]
            )

            with citizen_tab_1:
                supporting_passages = citizen_result[
                    "supporting_passages"
                ]

                if supporting_passages:
                    for index, item in enumerate(
                        supporting_passages,
                        start=1,
                    ):
                        st.markdown(
                            f"**Supporting passage {index}**"
                        )
                        st.write(item["passage"])
                        st.caption(
                            f"Prototype relevance score: "
                            f"{item['relevance_score']:.0%}"
                        )

                        shared_terms = item.get(
                            "shared_terms",
                            [],
                        )

                        if shared_terms:
                            st.caption(
                                "Shared terms: "
                                + ", ".join(shared_terms)
                            )

                        st.divider()
                else:
                    st.write(
                        "No sufficiently relevant supporting "
                        "passage was identified."
                    )

            with citizen_tab_2:
                display_list(
                    "Information Requiring Confirmation",
                    citizen_result["missing_information"],
                )

            with citizen_tab_3:
                display_list(
                    "Human Review Notes",
                    citizen_result["human_review_notes"],
                )

            st.error(
                "This is not an official government response. "
                "An authorized employee must verify the source, "
                "its validity, and the final answer before it is "
                "shared with a citizen."
            )

            st.divider()

            citizen_json = json.dumps(
                citizen_result,
                indent=2,
                ensure_ascii=False,
            )

            citizen_text_lines = [
                "GovBA Assistant — Preliminary Citizen Response",
                "",
                f"Reference source: {source_name}",
                f"Question: {citizen_result['question']}",
                f"Status: {citizen_result['answer_status']}",
                (
                    "Confidence indicator: "
                    f"{citizen_result['confidence_indicator']}"
                ),
                "",
                "ANSWER",
                citizen_result["answer"],
                "",
                "SUPPORTING INFORMATION",
            ]

            if citizen_result["supporting_passages"]:
                citizen_text_lines.extend(
                    (
                        f"- {item['passage']} "
                        f"(score: {item['relevance_score']:.0%})"
                    )
                    for item in citizen_result[
                        "supporting_passages"
                    ]
                )
            else:
                citizen_text_lines.append(
                    "- No sufficiently relevant passage identified."
                )

            citizen_text_lines.extend(
                [
                    "",
                    "INFORMATION REQUIRING CONFIRMATION",
                ]
            )

            if citizen_result["missing_information"]:
                citizen_text_lines.extend(
                    f"- {item}"
                    for item in citizen_result[
                        "missing_information"
                    ]
                )
            else:
                citizen_text_lines.append(
                    "- No specific missing information detected."
                )

            citizen_text_lines.extend(
                [
                    "",
                    "HUMAN REVIEW NOTES",
                ]
            )

            citizen_text_lines.extend(
                f"- {item}"
                for item in citizen_result[
                    "human_review_notes"
                ]
            )

            citizen_text_report = "\n".join(
                citizen_text_lines
            )

            citizen_download_1, citizen_download_2 = st.columns(2)

            with citizen_download_1:
                st.download_button(
                    label="Download Citizen Response as TXT",
                    data=citizen_text_report,
                    file_name="GovBA_Citizen_Response.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with citizen_download_2:
                st.download_button(
                    label="Download Citizen Response as JSON",
                    data=citizen_json,
                    file_name="GovBA_Citizen_Response.json",
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The citizen question could not be answered."
            )
            st.caption(
                f"Technical detail: {error}"
            )


    else:

        try:
            visualization_result = create_visualization(
                source_text,
                chart_type,
                chart_title,
                x_axis_label,
                y_axis_label,
            )

            png_bytes = visualization_result[
                "png_bytes"
            ]
            metadata = visualization_result[
                "metadata"
            ]

            st.success(
                f"{chart_type} created successfully."
            )

            st.caption(
                "The image reflects only the supplied structured data."
            )

            st.markdown("## Visualization Preview")

            metric_column_1, metric_column_2 = st.columns(2)

            with metric_column_1:
                st.metric(
                    "Chart Type",
                    metadata["chart_type"],
                )

            with metric_column_2:
                st.metric(
                    "Data Rows",
                    metadata["row_count"],
                )

            st.image(
                png_bytes,
                caption=metadata["title"],
                use_container_width=True,
            )

            display_list(
                "Review Warnings",
                visualization_result["warnings"],
            )

            st.divider()

            metadata_json = json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            )

            safe_chart_name = (
                chart_type.lower()
                .replace(" ", "_")
            )

            visualization_download_1, visualization_download_2 = (
                st.columns(2)
            )

            with visualization_download_1:
                st.download_button(
                    label="Download Visualization as PNG",
                    data=png_bytes,
                    file_name=(
                        f"GovBA_{safe_chart_name}.png"
                    ),
                    mime="image/png",
                    use_container_width=True,
                )

            with visualization_download_2:
                st.download_button(
                    label="Download Chart Metadata as JSON",
                    data=metadata_json,
                    file_name=(
                        f"GovBA_{safe_chart_name}_metadata.json"
                    ),
                    mime="application/json",
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The visualization could not be created."
            )
            st.caption(
                f"Technical detail: {error}"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "GovBA Assistant v0.12 — Internship prototype. "
    "Human review is required before using any generated output."
)