import json

import streamlit as st

from comparison_engine import compare_requirements_documents
from comparison_word_exporter import create_comparison_word_report
from demo_engine import generate_demo_brd
from document_reader import read_uploaded_file
from review_engine import review_requirements_document
from review_word_exporter import create_review_word_report
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

    st.divider()

    st.markdown("**Supported files**")
    st.write("• TXT")
    st.write("• Word DOCX")
    st.write("• Text-based PDF")

    st.divider()

    st.caption("Prototype version: 0.9")

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
        else:
            text_label = "Paste the BRD or SRS content to review:"
            placeholder_text = (
                "Example: Service Scope: The system will automate "
                "licence renewal. Applicants submit an identification "
                "number and proof of payment..."
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

else:
    button_label = "Compare Documents"

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


    else:

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


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "GovBA Assistant v0.9 — Internship prototype. "
    "Human review is required before using any generated output."
)