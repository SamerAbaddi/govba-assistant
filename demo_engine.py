import re


REQUIRES_CONFIRMATION = "Requires confirmation"


def _normalize_text(source_text: str) -> str:
    """Remove unnecessary line breaks and repeated spaces."""

    cleaned_text = source_text.strip()

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text,
    )

    return cleaned_text


def _split_sentences(source_text: str) -> list[str]:
    """Normalize the source and split it into complete sentences."""

    cleaned_text = _normalize_text(source_text)

    parts = re.split(
        r"(?<=[.!?])\s+",
        cleaned_text,
    )

    return [
        sentence.strip()
        for sentence in parts
        if sentence.strip()
    ]


def _remove_duplicates(items: list[str]) -> list[str]:
    """Remove duplicate values while preserving order."""

    unique_items = []

    for item in items:
        if item not in unique_items:
            unique_items.append(item)

    return unique_items


def _identify_service_name(source_text: str) -> str:
    """Attempt to identify the service name."""

    patterns = [
        r"automate (?:a|an|the) ([^.]*?service)",
        r"digitize (?:a|an|the) ([^.]*?service)",
        r"digitalize (?:a|an|the) ([^.]*?service)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            source_text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip().capitalize()

    return REQUIRES_CONFIRMATION


def _identify_stakeholders(source_text: str) -> list[str]:
    """Identify stakeholders explicitly mentioned."""

    stakeholder_keywords = {
        "applicant": "Applicant",
        "customer": "Customer",
        "citizen": "Citizen",
        "employee": "Authorized employee",
        "government entity": "Government entity",
        "business analyst": "Business Analyst",
        "vendor": "Vendor",
        "company": "Company",
        "department": "Department",
    }

    stakeholders = []
    lowered_text = source_text.lower()

    for keyword, label in stakeholder_keywords.items():
        if keyword in lowered_text:
            stakeholders.append(label)

    return stakeholders or [REQUIRES_CONFIRMATION]


def _identify_service_recipients(
    source_text: str,
) -> list[str]:
    """Identify possible service-recipient categories."""

    recipient_keywords = {
        "applicant": "Applicant",
        "citizen": "Individual or citizen",
        "business": "Business",
        "company": "Company",
        "government entity": "Government entity",
    }

    recipients = []
    lowered_text = source_text.lower()

    for keyword, label in recipient_keywords.items():
        if keyword in lowered_text:
            recipients.append(label)

    return recipients or [REQUIRES_CONFIRMATION]


def _extract_functional_requirements(
    sentences: list[str],
) -> list[str]:
    """Extract statements describing system functions."""

    functional_words = [
        "submit",
        "provide",
        "review",
        "approve",
        "reject",
        "return",
        "issue",
        "notify",
        "upload",
        "verify",
        "receive",
    ]

    requirements = []

    for sentence in sentences:
        lowered_sentence = sentence.lower()

        if any(
            word in lowered_sentence
            for word in functional_words
        ):
            requirements.append(sentence)

    return _remove_duplicates(requirements) or [
        REQUIRES_CONFIRMATION
    ]


def _extract_business_rules(
    sentences: list[str],
) -> list[str]:
    """Extract conditional and controlling statements."""

    rule_words = [
        "if ",
        "when ",
        "must ",
        "should ",
        "only ",
        "complete",
        "incomplete",
        "valid",
        "approved",
        "rejected",
    ]

    business_rules = []

    for sentence in sentences:
        lowered_sentence = sentence.lower()

        if any(
            word in lowered_sentence
            for word in rule_words
        ):
            business_rules.append(sentence)

    return _remove_duplicates(business_rules) or [
        REQUIRES_CONFIRMATION
    ]


def _extract_required_information(
    source_text: str,
) -> list[str]:
    """Identify data and documents mentioned in the source."""

    information_keywords = {
        "identification number": "Identification number",
        "national number": "National number",
        "licence number": "Existing licence number",
        "license number": "Existing license number",
        "proof of payment": "Proof of payment",
        "attachment": "Required attachment",
        "supporting document": "Supporting documents",
        "application form": "Application form",
    }

    identified_information = []
    lowered_text = source_text.lower()

    for keyword, label in information_keywords.items():
        if keyword in lowered_text:
            identified_information.append(label)

    return identified_information or [
        REQUIRES_CONFIRMATION
    ]


def generate_demo_brd(source_text: str) -> dict:
    """Generate a preliminary BRD without using an AI API."""

    cleaned_text = _normalize_text(source_text)

    if not cleaned_text:
        raise ValueError(
            "Source information is required."
        )

    sentences = _split_sentences(cleaned_text)
    service_name = _identify_service_name(cleaned_text)

    service_purpose = (
        sentences[0]
        if sentences
        else REQUIRES_CONFIRMATION
    )

    missing_information = [
        "Responsible government entity requires confirmation.",
        "Processing time requires confirmation.",
        "Approval authority requires confirmation.",
        "Integration method requires confirmation.",
        "Non-functional requirements require confirmation.",
    ]

    return {
        "service_overview": {
            "service_name": service_name,
            "service_purpose": service_purpose,
            "service_scope": REQUIRES_CONFIRMATION,
        },
        "stakeholders": _identify_stakeholders(
            cleaned_text
        ),
        "service_recipients": (
            _identify_service_recipients(cleaned_text)
        ),
        "functional_requirements": (
            _extract_functional_requirements(sentences)
        ),
        "non_functional_requirements": [
            REQUIRES_CONFIRMATION
        ],
        "business_rules": (
            _extract_business_rules(sentences)
        ),
        "process_steps": sentences,
        "required_data_and_documents": (
            _extract_required_information(cleaned_text)
        ),
        "integration_requirements": [
            REQUIRES_CONFIRMATION
        ],
        "missing_information": missing_information,
        "human_review_notes": [
            "This result was generated in demo mode.",
            "A Business Analyst must review and approve it.",
        ],
    }