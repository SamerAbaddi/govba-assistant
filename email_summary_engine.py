import re

REQUIRES_CONFIRMATION = "Requires confirmation"

ACTION_KEYWORDS = [
    "please",
    "need to",
    "must",
    "should",
    "kindly",
    "complete",
    "submit",
    "send",
    "review",
    "confirm",
    "prepare",
    "update",
    "follow up",
]

DECISION_KEYWORDS = [
    "decided",
    "approved",
    "agreed",
    "confirmed",
    "accepted",
    "rejected",
]

DEADLINE_PATTERNS = [
    r"\bby\s+(?:today|tomorrow|end of day|eod|end of week)\b",
    r"\b(?:today|tomorrow|this week|next week)\b",
    r"\bby\s+\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
]


def _normalize_text(text: str) -> str:
    """Normalize spaces while preserving line structure."""

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]

    return "\n".join(line for line in lines if line)


def _extract_header(text: str, label: str) -> str:
    """Extract a simple email header such as From, To, or Subject."""

    match = re.search(
        rf"(?im)^{re.escape(label)}\s*:\s*(.+)$",
        text,
    )

    return match.group(1).strip() if match else REQUIRES_CONFIRMATION


def _remove_headers_and_signature(text: str) -> str:
    """Remove common email headers and a simple signature block."""

    body_lines = []

    for line in text.splitlines():
        if re.match(
            r"(?i)^(from|to|cc|bcc|subject|date|sent)\s*:",
            line,
        ):
            continue

        if re.match(
            r"(?i)^(regards|best regards|kind regards|thanks|thank you)[,!]?$",
            line.strip(),
        ):
            break

        body_lines.append(line)

    return " ".join(body_lines).strip()


def _remove_duplicates(items: list[str]) -> list[str]:
    """Remove duplicate items while preserving order."""

    result = []
    seen = set()

    for item in items:
        key = item.lower()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _split_sentences(text: str) -> list[str]:
    """Split email content into readable sentences."""

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    items = []

    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" -•")

        if len(cleaned.split()) >= 3:
            items.append(cleaned)

    return _remove_duplicates(items)


def _shorten(text: str, limit: int = 180) -> str:
    """Shorten a long sentence without cutting the final word."""

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0] + "..."


def _detect_priority(text: str) -> str:
    """Estimate priority from explicit wording."""

    lowered = text.lower()

    if any(
        word in lowered
        for word in [
            "urgent",
            "asap",
            "immediately",
            "critical",
            "high priority",
        ]
    ):
        return "High"

    if any(
        word in lowered
        for word in [
            "please review",
            "follow up",
            "tomorrow",
            "deadline",
            "required",
        ]
    ):
        return "Medium"

    return "Normal"


def _find_items(
    sentences: list[str],
    keywords: list[str],
) -> list[str]:
    """Find sentences containing selected keywords."""

    items = []

    for sentence in sentences:
        lowered = sentence.lower()

        if any(keyword in lowered for keyword in keywords):
            items.append(_shorten(sentence))

    return _remove_duplicates(items)[:5]


def _find_deadlines(sentences: list[str]) -> list[str]:
    """Find sentences containing likely deadlines."""

    items = []

    for sentence in sentences:
        if any(
            re.search(pattern, sentence, flags=re.IGNORECASE)
            for pattern in DEADLINE_PATTERNS
        ):
            items.append(_shorten(sentence))

    return _remove_duplicates(items)[:5]


def _build_summary(sentences: list[str]) -> list[str]:
    """Build three or four concise summary bullets."""

    if not sentences:
        return [REQUIRES_CONFIRMATION]

    selected = []

    important_words = (
        ACTION_KEYWORDS
        + DECISION_KEYWORDS
        + [
            "meeting",
            "project",
            "request",
            "issue",
            "status",
        ]
    )

    for sentence in sentences:
        if any(word in sentence.lower() for word in important_words):
            selected.append(_shorten(sentence))

        if len(selected) == 4:
            break

    if len(selected) < 3:
        for sentence in sentences:
            shortened = _shorten(sentence)

            if shortened not in selected:
                selected.append(shortened)

            if len(selected) == 4:
                break

    return selected or [REQUIRES_CONFIRMATION]


def summarize_employee_email(email_text: str) -> dict:
    """
    Summarize pasted or uploaded email text.

    This prototype does not connect to Gmail, Outlook,
    or any Ministry email system.
    """

    cleaned = _normalize_text(email_text)

    if not cleaned:
        raise ValueError(
            "Email text is required before creating a summary."
        )

    body = _remove_headers_and_signature(cleaned)
    sentences = _split_sentences(body)

    return {
        "sender": _extract_header(cleaned, "From"),
        "recipient": _extract_header(cleaned, "To"),
        "subject": _extract_header(cleaned, "Subject"),
        "priority": _detect_priority(cleaned),
        "summary_bullets": _build_summary(sentences),
        "action_items": (
            _find_items(sentences, ACTION_KEYWORDS)
            or ["No clear action item detected."]
        ),
        "deadlines": (
            _find_deadlines(sentences)
            or ["No clear deadline detected."]
        ),
        "decisions": (
            _find_items(sentences, DECISION_KEYWORDS)
            or ["No clear decision detected."]
        ),
        "summary_mode": "Rule-based demo",
        "human_review_notes": [
            "Review the original email before acting.",
            "No email-account or Ministry-system connection is used.",
        ],
    }