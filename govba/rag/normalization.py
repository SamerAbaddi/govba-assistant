"""Deterministic source-preserving text normalization for GovBA-GAR."""

from __future__ import annotations

import re
import unicodedata


TEXT_NORMALIZATION_VERSION = (
    "govba-text-normalization-v1"
)

_HORIZONTAL_WHITESPACE_RE = re.compile(
    r"[^\S\n]+"
)


def normalize_ingested_text(
    text: str,
) -> str:
    """Normalize extracted text without changing its linguistic meaning.

    Policy:
    - Unicode NFC normalization.
    - CRLF/CR and Unicode line separators become LF.
    - Non-breaking spaces become ordinary spaces.
    - Horizontal whitespace runs collapse to one space.
    - Leading/trailing whitespace on lines is removed.
    - Multiple blank lines collapse to one blank line.
    - Leading/trailing blank lines are removed.
    - Case, Arabic diacritics, punctuation, and language are preserved.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    value = unicodedata.normalize(
        "NFC",
        text,
    )

    value = (
        value
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
    )

    normalized_lines = []

    for line in value.split("\n"):
        line = _HORIZONTAL_WHITESPACE_RE.sub(
            " ",
            line,
        ).strip()

        normalized_lines.append(
            line
        )

    output_lines = []
    blank_pending = False

    for line in normalized_lines:
        if not line:
            if output_lines:
                blank_pending = True

            continue

        if blank_pending:
            output_lines.append("")
            blank_pending = False

        output_lines.append(
            line
        )

    return "\n".join(
        output_lines
    )


def is_normalized_ingested_text(
    text: str,
) -> bool:
    """Return whether text already satisfies the normalization policy."""

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    return (
        normalize_ingested_text(text)
        == text
    )
