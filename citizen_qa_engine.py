import re
from collections import Counter


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "do", "does", "for", "from", "how", "i", "in", "is", "it",
    "of", "on", "or", "the", "this", "to", "what", "when",
    "where", "which", "who", "with", "you", "your",
    "ما", "ماذا", "من", "في", "على", "إلى", "الى", "عن",
    "هل", "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "و", "أو",
    "او", "كيف", "متى", "أين", "اين", "التي", "الذي",
}

QUESTION_HINTS = {
    "documents": {
        "document", "documents", "attachment", "attachments",
        "required", "proof", "certificate", "identity", "id",
        "وثيقة", "وثائق", "مستند", "مستندات", "مرفق", "هوية",
    },
    "fees": {
        "fee", "fees", "cost", "price", "payment", "jod",
        "رسوم", "تكلفة", "سعر", "دفع", "دينار",
    },
    "time": {
        "time", "days", "hours", "duration", "processing",
        "deadline", "مدة", "يوم", "أيام", "ساعة", "وقت",
    },
    "eligibility": {
        "eligible", "eligibility", "condition", "conditions",
        "requirement", "requirements", "qualify",
        "مؤهل", "أهلية", "شرط", "شروط", "متطلبات",
    },
    "channel": {
        "apply", "application", "online", "website", "portal",
        "office", "branch", "submit", "تقديم", "طلب", "موقع",
        "بوابة", "مكتب", "فرع", "إلكتروني",
    },
}


def _normalize_text(text: str) -> str:
    """Normalize whitespace without changing the meaning."""

    return re.sub(r"\s+", " ", text.strip())


def _tokenize(text: str) -> list[str]:
    """Tokenize English and Arabic words and numbers."""

    tokens = re.findall(
        r"[A-Za-z0-9]+|[\u0600-\u06FF]+",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS
    ]


def _remove_duplicates(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""

    result = []
    seen = set()

    for item in items:
        key = item.lower()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _split_passages(source_text: str) -> list[str]:
    """Split supplied information into readable passages."""

    normalized = source_text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    raw_sections = re.split(
        r"\n{2,}|(?<=[.!?؟])\s+|^\s*[-•]\s+",
        normalized,
        flags=re.MULTILINE,
    )

    passages = []

    for section in raw_sections:
        cleaned = _normalize_text(section).strip(" -•")

        if len(_tokenize(cleaned)) >= 3:
            passages.append(cleaned)

    return _remove_duplicates(passages)


def _detect_question_hints(question_tokens: set[str]) -> set[str]:
    """Identify the likely information category requested."""

    detected = set()

    for category, hints in QUESTION_HINTS.items():
        if question_tokens.intersection(hints):
            detected.add(category)

    return detected


def _score_passage(
    passage: str,
    question_tokens: set[str],
    question_hints: set[str],
) -> tuple[float, list[str]]:
    """Score one source passage against the citizen question."""

    passage_tokens = _tokenize(passage)
    passage_token_set = set(passage_tokens)
    shared_tokens = question_tokens.intersection(
        passage_token_set
    )

    if not question_tokens:
        return 0.0, []

    direct_overlap = (
        len(shared_tokens) / len(question_tokens)
    )

    hint_bonus = 0.0

    for category in question_hints:
        if passage_token_set.intersection(
            QUESTION_HINTS[category]
        ):
            hint_bonus += 0.12

    frequency_bonus = 0.0
    token_counts = Counter(passage_tokens)

    for token in shared_tokens:
        frequency_bonus += min(
            token_counts[token] * 0.02,
            0.06,
        )

    score = min(
        direct_overlap + hint_bonus + frequency_bonus,
        1.0,
    )

    return round(score, 3), sorted(shared_tokens)


def _confidence_label(score: float) -> str:
    """Convert a relevance score into a readable label."""

    if score >= 0.65:
        return "High"

    if score >= 0.35:
        return "Medium"

    return "Low"


def answer_citizen_question(
    government_information: str,
    citizen_question: str,
) -> dict:
    """
    Answer a citizen question using only supplied information.

    The function does not browse the internet or connect to an
    official government database.
    """

    source_text = _normalize_text(
        government_information
    )
    question = _normalize_text(
        citizen_question
    )

    if not source_text:
        raise ValueError(
            "Governmental reference information is required."
        )

    if not question:
        raise ValueError(
            "A citizen question is required."
        )

    passages = _split_passages(
        government_information
    )

    if not passages:
        raise ValueError(
            "No readable information passages were detected."
        )

    question_tokens = set(
        _tokenize(question)
    )

    if not question_tokens:
        raise ValueError(
            "The question does not contain enough readable terms."
        )

    question_hints = _detect_question_hints(
        question_tokens
    )

    ranked_passages = []

    for passage in passages:
        score, shared_terms = _score_passage(
            passage,
            question_tokens,
            question_hints,
        )

        ranked_passages.append(
            {
                "passage": passage,
                "relevance_score": score,
                "shared_terms": shared_terms,
            }
        )

    ranked_passages.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    strongest = ranked_passages[0]
    strongest_score = strongest["relevance_score"]

    shared_term_count = len(
        strongest["shared_terms"]
    )

    supported = (
        strongest_score >= 0.30
        and shared_term_count >= 2
    )

    if supported:
        supporting_passages = [
            item
            for item in ranked_passages[:3]
            if item["relevance_score"] >= 0.16
        ]

        answer = (
            "Based on the supplied governmental information: "
            f"{strongest['passage']}"
        )

        answer_status = (
            "Supported by supplied information"
        )

        missing_information = []

    else:
        supporting_passages = []

        answer = (
            "The supplied governmental information does not "
            "contain a sufficiently clear answer to this question."
        )

        answer_status = (
            "Not found in supplied information"
        )

        missing_information = [
            "An authorized source or responsible government entity "
            "should confirm the requested information."
        ]

    return {
        "question": question,
        "answer": answer,
        "answer_status": answer_status,
        "confidence_indicator": _confidence_label(
            strongest_score
        ),
        "supporting_passages": supporting_passages,
        "missing_information": missing_information,
        "answer_mode": "Source-grounded rule-based demo",
        "human_review_notes": [
            "The response is based only on the information supplied "
            "inside this prototype.",
            "The response is not an official government decision, "
            "legal interpretation, or service commitment.",
            "Unclear, outdated, or incomplete source information "
            "must be verified by an authorized employee.",
        ],
    }