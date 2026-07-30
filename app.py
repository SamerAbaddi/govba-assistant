import json

import streamlit as st

from demo_engine import generate_demo_brd
from document_reader import read_uploaded_file
from review_engine import review_requirements_document
from word_exporter import create_brd_word_report


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="GovBA Assistant",
    page_icon="🏛️",
    layout="wide",
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
st.title("🏛️ GovBA Assistant")

st.write(
    "AI Support Agent for Business Analysis "
    "and Government-Service Documentation"
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Prototype Status")

    st.success("Level 2 — Multi-Task Demo")

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

    st.divider()

    st.markdown("**Supported files**")
    st.write("• TXT")
    st.write("• Word DOCX")
    st.write("• Text-based PDF")

    st.divider()

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
    ],
)

document_type = "BRD"

if task == "Review a BRD or SRS":
    document_type = st.radio(
        "Select the document type:",
        ["BRD", "SRS"],
        horizontal=True,
    )


# ---------------------------------------------------------
# Input method
# ---------------------------------------------------------
st.subheader("2. Provide the Source Information")

input_method = st.radio(
    "Choose how to provide the information:",
    ["Paste text", "Upload a document"],
    horizontal=True,
)

source_text = ""
source_name = "Manually entered text"


# ---------------------------------------------------------
# Manual text input
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Document upload
# ---------------------------------------------------------
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
# Run selected task
# ---------------------------------------------------------
st.subheader("3. Run the Selected Task")

button_label = (
    "Generate BRD Draft"
    if task == "Generate a BRD Draft"
    else f"Review {document_type} Document"
)

if st.button(
    button_label,
    type="primary",
    use_container_width=True,
):

    if not source_text.strip():

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

    else:

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


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Internship prototype — human review is required "
    "before using any generated output."
)