import streamlit as st


# Basic webpage settings
st.set_page_config(
    page_title="GovBA Assistant",
    page_icon="🏛️",
    layout="wide",
)


# Application heading
st.title("🏛️ GovBA Assistant")
st.write(
    "AI Support Agent for Business Analysis "
    "and Government-Service Documentation"
)


# Sidebar information
with st.sidebar:
    st.header("Prototype Status")
    st.success("Level 1 — Interface Development")
    st.info(
        "Demo mode is active. "
        "The AI connection will be added later."
    )

    st.markdown(
        "**Data rule:** Use only fictional, public, "
        "or approved non-confidential information."
    )


# Task selection
st.subheader("1. Select the Task")

task = st.selectbox(
    "What would you like the assistant to do?",
    ["Generate a BRD Draft"],
)


# Text input
st.subheader("2. Enter the Source Information")

source_text = st.text_area(
    "Paste meeting notes, interview notes, or a service description:",
    height=260,
    placeholder=(
        "Example: A government entity wants to automate a "
        "licence-renewal service. Applicants provide their "
        "identification number, existing licence information, "
        "and required supporting documents..."
    ),
)


# Generate button
st.subheader("3. Generate the Draft")

if st.button(
    "Generate BRD Draft",
    type="primary",
    use_container_width=True,
):
    if not source_text.strip():
        st.warning(
            "Please enter sample information before generating the draft."
        )

    else:
        st.success("Demo output generated successfully.")

        st.caption(
            "The application structure is working. "
            "Real AI analysis will be activated later."
        )

        st.markdown("## Preliminary Business Requirements Document")

        st.markdown("### Source Preview")

        preview = source_text[:500]

        if len(source_text) > 500:
            preview += "..."

        st.write(preview)

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("### Service Overview")
            st.write("**Service name:** Requires confirmation")
            st.write("**Service purpose:** Requires AI analysis")
            st.write("**Service scope:** Requires AI analysis")

            st.markdown("### Stakeholders")
            st.write("- Requires AI analysis")

            st.markdown("### Functional Requirements")
            st.write("- Requires AI analysis")

        with right_column:
            st.markdown("### Business Rules")
            st.write("- Requires AI analysis")

            st.markdown("### Required Data and Documents")
            st.write("- Requires AI analysis")

            st.markdown("### Integration Requirements")
            st.write("- Requires AI analysis")

            st.markdown("### Missing Information")
            st.write("- To be identified by the AI agent")


# Footer
st.divider()

st.caption(
    "Internship prototype — human review is required "
    "before using any generated output."
)