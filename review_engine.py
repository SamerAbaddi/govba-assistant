import re


REQUIRES_CONFIRMATION = "Requires confirmation"


BRD_SECTION_RULES = {
    "Service Overview": [
        "service name",
        "service purpose",
        "service scope",
        "objective",
    ],
    "Stakeholders": [
        "stakeholder",
        "applicant",
        "employee",
        "entity",
        "department",
    ],
    "Service Recipients": [
        "service recipient",
        "applicant",
        "citizen",
        "business",
        "company",
        "government entity",
    ],
    "Functional Requirements": [
        "functional requirement",
        "shall",
        "must allow",
        "submit",
        "review",
        "approve",
        "reject",
        "notify",
    ],
    "Non-Functional Requirements": [
        "non-functional requirement",
        "security",
        "availability",
        "performance",
        "accessibility",
        "response time",
        "usability",
    ],
    "Business Rules": [
        "business rule",
        "eligibility",
        "condition",
        "approval",
        "rejection",
        "exception",
        "validation",
    ],
    "Process Steps": [
        "process step",
        "workflow",
        "procedure",
        "first",
        "then",
        "after",
    ],
    "Required Data and Documents": [
        "required data",
        "required document",
        "attachment",
        "identification number",
        "proof",
        "form",
    ],
    "Integration Requirements": [
        "integration",
        "api",
        "web service",
        "data source",
        "external system",
        "web form",
    ],
}


SRS_SECTION_RULES = {
    "System Purpose and Scope": [
        "system purpose",
        "system scope",
        "objective",
        "overview",
    ],
    "Users and Roles": [
        "user role",
        "user type",
        "administrator",
        "applicant",
        "employee",
    ],
    "Functional Requirements": [
        "functional requirement",
        "shall",
        "must",
        "system will",
    ],
    "Non-Functional Requirements": [
        "non-functional requirement",
        "security",
        "performance",
        "availability",
        "accessibility",
        "usability",
    ],
    "Data Requirements": [
        "data requirement",
        "data field",
        "database",
        "record",
        "information",
    ],
    "Integration Requirements": [
        "integration",
        "api",
        "web service",
        "external system",
        "interface",
    ],
    "Validation and Business Rules": [
        "validation",
        "business rule",
        "eligibility",
        "approval",
        "rejection",
        "exception",
    ],
    "Error Handling": [
        "error",
        "exception handling",
        "failure",
        "invalid",
        "error message",
    ],
    "Acceptance or Testing Criteria": [
        "acceptance criteria",
        "test case",
        "testing",
        "expected result",
    ],
}


VAGUE_PHRASES = [
    "etc.",
    "and so on",
    "as needed",
    "as required",
    "where appropriate",
    "if necessary",
    "user-friendly",
    "fast",
    "quickly",
    "soon",
    "efficient",
    "adequate",
]


def _normalize_text(source_text: str) -> str:
    """Normalize spaces and line breaks."""

    return re.sub(
        r"\s+",
        " ",
        source_text.strip(),
    )


def _find_present_sections(
    source_text: str,
    section_rules: dict,
) -> list[dict]:
    """Check whether expected document sections are represented."""

    lowered_text = source_text.lower()
    checklist = []

    for section_name, keywords in section_rules.items():
        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword in lowered_text
        ]

        status = (
            "Detected"
            if matched_keywords
            else "Missing or unclear"
        )

        checklist.append(
            {
                "section": section_name,
                "status": status,
                "evidence": matched_keywords[:4],
            }
        )

    return checklist


def _find_vague_language(source_text: str) -> list[str]:
    """Find wording that may make requirements unclear."""

    lowered_text = source_text.lower()
    issues = []

    for phrase in VAGUE_PHRASES:
        if phrase in lowered_text:
            issues.append(
                f"Vague wording detected: '{phrase}'. "
                "Replace it with a measurable or specific statement."
            )

    return issues


def _find_weak_requirements(source_text: str) -> list[str]:
    """Identify possible weaknesses in requirement wording."""

    issues = []
    sentences = re.split(
        r"(?<=[.!?])\s+",
        source_text,
    )

    requirement_sentences = [
        sentence.strip()
        for sentence in sentences
        if any(
            term in sentence.lower()
            for term in [
                "shall",
                "must",
                "should",
                "system will",
            ]
        )
    ]

    for sentence in requirement_sentences:
        lowered_sentence = sentence.lower()

        if "should" in lowered_sentence:
            issues.append(
                "A requirement uses 'should', which may be optional "
                f"or unclear: {sentence}"
            )

        if len(sentence.split()) < 5:
            issues.append(
                f"A requirement may be too short or incomplete: {sentence}"
            )

    return issues


def review_requirements_document(
    source_text: str,
    document_type: str = "BRD",
) -> dict:
    """
    Review a BRD or SRS using a rule-based demonstration engine.
    """

    cleaned_text = _normalize_text(source_text)

    if not cleaned_text:
        raise ValueError(
            "A document is required before starting the review."
        )

    normalized_type = document_type.strip().upper()

    if normalized_type == "BRD":
        section_rules = BRD_SECTION_RULES

    elif normalized_type == "SRS":
        section_rules = SRS_SECTION_RULES

    else:
        raise ValueError(
            "Document type must be either BRD or SRS."
        )

    checklist = _find_present_sections(
        cleaned_text,
        section_rules,
    )

    detected_sections = [
        item["section"]
        for item in checklist
        if item["status"] == "Detected"
    ]

    missing_sections = [
        item["section"]
        for item in checklist
        if item["status"] == "Missing or unclear"
    ]

    issues = (
        _find_vague_language(cleaned_text)
        + _find_weak_requirements(cleaned_text)
    )

    if not issues:
        issues = [
            "No basic wording issues were detected by the "
            "rule-based review. Human review is still required."
        ]

    total_sections = len(checklist)
    detected_count = len(detected_sections)

    completeness_percentage = round(
        detected_count / total_sections * 100
    )

    recommendations = [
        (
            f"Add or clarify the '{section}' section."
        )
        for section in missing_sections
    ]

    if not recommendations:
        recommendations = [
            "Review each section for accuracy, traceability, "
            "and formal approval."
        ]

    return {
        "document_type": normalized_type,
        "review_mode": "Rule-based demo",
        "completeness_indicator": completeness_percentage,
        "section_checklist": checklist,
        "detected_sections": detected_sections,
        "missing_sections": missing_sections,
        "wording_issues": issues,
        "recommendations": recommendations,
        "human_review_notes": [
            "The completeness percentage is an AI-assisted "
            "prototype indicator, not an official Ministry score.",
            "A Business Analyst must verify every finding.",
            "The engine checks visible wording and keywords only.",
        ],
    }