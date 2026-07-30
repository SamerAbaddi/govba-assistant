import re
from difflib import SequenceMatcher


MATCH_THRESHOLD = 0.72
PARTIAL_MATCH_THRESHOLD = 0.45


def _normalize_text(text: str) -> str:
    """Normalize whitespace for stable comparison."""

    return re.sub(r"\s+", " ", text.strip())


def _split_items(text: str) -> list[str]:
    """
    Split a document into meaningful requirement-like items.

    The function uses sentence endings, bullets, and numbered lines.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_parts = re.split(
        r"(?<=[.!?])\s+|\n+|^\s*[-•]\s+|^\s*\d+[.)]\s+",
        normalized,
        flags=re.MULTILINE,
    )

    items = []

    for part in raw_parts:
        cleaned = _normalize_text(part)

        if len(cleaned.split()) >= 3:
            items.append(cleaned)

    return _remove_duplicates(items)


def _remove_duplicates(items: list[str]) -> list[str]:
    """Remove duplicate items while preserving order."""

    unique_items = []
    seen = set()

    for item in items:
        key = item.lower()

        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def _similarity(item_a: str, item_b: str) -> float:
    """Calculate text similarity between two items."""

    return SequenceMatcher(
        None,
        item_a.lower(),
        item_b.lower(),
    ).ratio()


def _keywords(text: str) -> set[str]:
    """Return useful comparison keywords."""

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "be",
        "shall",
        "must",
        "should",
        "will",
        "system",
        "service",
        "application",
    }

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return {
        word
        for word in words
        if len(word) > 2 and word not in stop_words
    }


def _keyword_overlap(item_a: str, item_b: str) -> float:
    """Calculate shared-keyword coverage."""

    keywords_a = _keywords(item_a)
    keywords_b = _keywords(item_b)

    if not keywords_a or not keywords_b:
        return 0.0

    shared = keywords_a.intersection(keywords_b)
    smaller_set_size = min(
        len(keywords_a),
        len(keywords_b),
    )

    return len(shared) / smaller_set_size


def _numbers(text: str) -> set[str]:
    """Extract numbers that may signal conflicting requirements."""

    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?\b",
            text,
        )
    )


def _find_best_match(
    source_item: str,
    candidate_items: list[str],
) -> tuple[int | None, float]:
    """Find the strongest candidate match for one source item."""

    best_index = None
    best_score = 0.0

    for index, candidate in enumerate(candidate_items):
        text_score = _similarity(
            source_item,
            candidate,
        )

        keyword_score = _keyword_overlap(
            source_item,
            candidate,
        )

        combined_score = (
            text_score * 0.65
            + keyword_score * 0.35
        )

        if combined_score > best_score:
            best_index = index
            best_score = combined_score

    return best_index, round(best_score, 2)


def compare_requirements_documents(
    document_a_text: str,
    document_b_text: str,
    document_a_name: str = "Document A",
    document_b_name: str = "Document B",
) -> dict:
    """
    Compare two requirements-related documents.

    This is a rule-based prototype comparison and requires human review.
    """

    cleaned_a = _normalize_text(document_a_text)
    cleaned_b = _normalize_text(document_b_text)

    if not cleaned_a:
        raise ValueError(
            "Document A is required before comparison."
        )

    if not cleaned_b:
        raise ValueError(
            "Document B is required before comparison."
        )

    items_a = _split_items(document_a_text)
    items_b = _split_items(document_b_text)

    if not items_a or not items_b:
        raise ValueError(
            "Readable comparison items could not be identified "
            "in one or both documents."
        )

    matched = []
    partial_matches = []
    missing_in_b = []
    possible_conflicts = []
    matched_b_indexes = set()

    for item_a in items_a:
        best_index, score = _find_best_match(
            item_a,
            items_b,
        )

        if best_index is None:
            missing_in_b.append(item_a)
            continue

        item_b = items_b[best_index]

        numbers_a = _numbers(item_a)
        numbers_b = _numbers(item_b)

        same_topic = (
            _keyword_overlap(item_a, item_b) >= 0.5
        )

        numeric_conflict = (
            same_topic
            and numbers_a
            and numbers_b
            and numbers_a != numbers_b
        )

        if numeric_conflict:
            possible_conflicts.append(
                {
                    "document_a_item": item_a,
                    "document_b_item": item_b,
                    "reason": (
                        "Similar topic detected, but the numerical "
                        "values are different."
                    ),
                    "similarity_score": score,
                }
            )
            matched_b_indexes.add(best_index)

        elif score >= MATCH_THRESHOLD:
            matched.append(
                {
                    "document_a_item": item_a,
                    "document_b_item": item_b,
                    "similarity_score": score,
                }
            )
            matched_b_indexes.add(best_index)

        elif score >= PARTIAL_MATCH_THRESHOLD:
            partial_matches.append(
                {
                    "document_a_item": item_a,
                    "document_b_item": item_b,
                    "similarity_score": score,
                    "review_note": (
                        "The items appear related but may not be "
                        "fully equivalent."
                    ),
                }
            )
            matched_b_indexes.add(best_index)

        else:
            missing_in_b.append(item_a)

    additional_in_b = [
        item
        for index, item in enumerate(items_b)
        if index not in matched_b_indexes
    ]

    covered_items = (
        len(matched)
        + len(partial_matches)
        + len(possible_conflicts)
    )

    coverage_indicator = round(
        covered_items / len(items_a) * 100
    )

    return {
        "document_a_name": document_a_name,
        "document_b_name": document_b_name,
        "comparison_mode": "Rule-based demo",
        "document_a_item_count": len(items_a),
        "document_b_item_count": len(items_b),
        "coverage_indicator": coverage_indicator,
        "matched_items": matched,
        "partial_matches": partial_matches,
        "missing_in_document_b": missing_in_b,
        "additional_in_document_b": additional_in_b,
        "possible_conflicts": possible_conflicts,
        "summary": {
            "matched_count": len(matched),
            "partial_match_count": len(partial_matches),
            "missing_count": len(missing_in_b),
            "additional_count": len(additional_in_b),
            "possible_conflict_count": len(
                possible_conflicts
            ),
        },
        "human_review_notes": [
            "This is a prototype comparison, not an official "
            "traceability or compliance decision.",
            "Similarity does not prove that two requirements are "
            "legally, technically, or operationally equivalent.",
            "A Business Analyst must verify all matches, missing "
            "items, additions, and possible conflicts.",
        ],
    }