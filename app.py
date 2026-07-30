import streamlit as st

from document_reader import read_uploaded_file


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="GovBA Assistant",
    page_icon="🏛️",
    layout="wide",
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

    st.success("Level 2 — Document Input")

    st.info(
        "Demo mode is active. "
        "The AI connection will be added later."
    )

    st.markdown(
        "**Data rule:** Use only fictional, public, "
        "or approved non-confidential information."
    )

    st.divider()

    st.markdown("**Supported files**")

    st.write("• TXT")
    st.write("• Word DOCX")
    st.write("• Text-based PDF")


# ---------------------------------------------------------
# Task selection
# ---------------------------------------------------------
st.subheader("1. Select the Task")

task = st.selectbox(
    "What would you like the assistant to do?",
    ["Generate a BRD Draft"],
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

    source_text = st.text_area(
        "Paste meeting notes, interview notes, "
        "or a service description:",
        height=260,
        placeholder=(
            "Example: A government entity wants to automate "
            "a licence-renewal service. Applicants provide "
            "their identification number, existing licence "
            "information, and supporting documents..."
        ),
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

            with st.expander("Preview extracted document text"):

                preview_text = source_text[:3000]

                if len(source_text) > 3000:
                    preview_text += (
                        "\n\n[Preview shortened. "
                        "The full text remains available.]"
                    )

                st.text(preview_text)

        except ValueError as error:
            st.error(str(error))

        except Exception:
            st.error(
                "The document could not be read. "
                "Please check the file and try again."
            )


# ---------------------------------------------------------
# Generate button
# ---------------------------------------------------------
st.subheader("3. Generate the Draft")

if st.button(
    "Generate BRD Draft",
    type="primary",
    use_container_width=True,
):

    if not source_text.strip():

        st.warning(
            "Please paste information or upload "
            "a readable document first."
        )

    else:

        st.success("Demo output generated successfully.")

        st.caption(
            "The document-input system is working. "
            "Real AI analysis will be activated later."
        )

        st.markdown(
            "## Preliminary Business Requirements Document"
        )

        st.write(f"**Source:** {source_name}")

        with st.expander("View source preview", expanded=True):

            preview = source_text[:1000]

            if len(source_text) > 1000:
                preview += "..."

            st.write(preview)

        left_column, right_column = st.columns(2)

        with left_column:

            st.markdown("### Service Overview")

            st.write(
                "**Service name:** Requires confirmation"
            )

            st.write(
                "**Service purpose:** Requires AI analysis"
            )

            st.write(
                "**Service scope:** Requires AI analysis"
            )

            st.markdown("### Stakeholders")

            st.write("- Requires AI analysis")

            st.markdown("### Functional Requirements")

            st.write("- Requires AI analysis")

        with right_column:

            st.markdown("### Business Rules")

            st.write("- Requires AI analysis")

            st.markdown(
                "### Required Data and Documents"
            )

            st.write("- Requires AI analysis")

            st.markdown(
                "### Integration Requirements"
            )

            st.write("- Requires AI analysis")

            st.markdown("### Missing Information")

            st.write(
                "- To be identified by the AI agent"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Internship prototype — human review is required "
    "before using any generated output."
)