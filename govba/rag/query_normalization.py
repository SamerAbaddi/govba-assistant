"""Deterministic Arabic/English query normalization for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from govba.rag.language import (
    BilingualLanguage,
    BilingualQueryRequest,
    DetectedLanguage,
)


QUERY_NORMALIZATION_VERSION = (
    "govba-query-normalization-v1"
)


_WHITESPACE_RE = re.compile(
    r"\s+",
    flags=re.UNICODE,
)


_ARABIC_DIACRITICS_RE = re.compile(
    "["
    "\u0610-\u061a"
    "\u064b-\u065f"
    "\u0670"
    "\u06d6-\u06ed"
    "]"
)


_ARABIC_NORMALIZATION_TABLE = (
    str.maketrans(
        {
            "\u0622": "\u0627",  # ALEF WITH MADDA
            "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE
            "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW
            "\u0671": "\u0627",  # ALEF WASLA
            "\u0649": "\u064a",  # ALEF MAKSURA -> YEH
            "\u0640": "",        # TATWEEL
        }
    )
)


_DIGIT_NORMALIZATION_TABLE = (
    str.maketrans(
        {
            # Arabic-Indic digits
            "\u0660": "0",
            "\u0661": "1",
            "\u0662": "2",
            "\u0663": "3",
            "\u0664": "4",
            "\u0665": "5",
            "\u0666": "6",
            "\u0667": "7",
            "\u0668": "8",
            "\u0669": "9",

            # Eastern Arabic/Persian digits
            "\u06f0": "0",
            "\u06f1": "1",
            "\u06f2": "2",
            "\u06f3": "3",
            "\u06f4": "4",
            "\u06f5": "5",
            "\u06f6": "6",
            "\u06f7": "7",
            "\u06f8": "8",
            "\u06f9": "9",
        }
    )
)


_SPACE_NORMALIZATION_TABLE = (
    str.maketrans(
        {
            "\u00a0": " ",
            "\u202f": " ",
            "\ufeff": "",
            "\u200b": "",
        }
    )
)


def _required_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _is_arabic_letter(
    character: str,
) -> bool:
    if not character.isalpha():
        return False

    codepoint = ord(
        character
    )

    ranges = (
        (
            0x0600,
            0x06FF,
        ),
        (
            0x0750,
            0x077F,
        ),
        (
            0x08A0,
            0x08FF,
        ),
        (
            0xFB50,
            0xFDFF,
        ),
        (
            0xFE70,
            0xFEFF,
        ),
        (
            0x1EE00,
            0x1EEFF,
        ),
    )

    return any(
        start <= codepoint <= end
        for start, end
        in ranges
    )


def _is_latin_letter(
    character: str,
) -> bool:
    if not character.isalpha():
        return False

    return (
        "LATIN"
        in unicodedata.name(
            character,
            "",
        )
    )


def count_query_scripts(
    text: str,
) -> tuple[
    int,
    int,
]:
    """Return Arabic-letter and Latin-letter counts."""

    text = _required_text(
        text,
        "text",
    )

    arabic_count = 0
    latin_count = 0

    for character in text:
        if _is_arabic_letter(
            character
        ):
            arabic_count += 1

        elif _is_latin_letter(
            character
        ):
            latin_count += 1

    return (
        arabic_count,
        latin_count,
    )


def detect_query_language(
    text: str,
) -> DetectedLanguage:
    """Detect Arabic/English script presence deterministically."""

    arabic_count, latin_count = (
        count_query_scripts(
            text
        )
    )

    if (
        arabic_count > 0
        and latin_count > 0
    ):
        return (
            DetectedLanguage.MIXED
        )

    if arabic_count > 0:
        return (
            DetectedLanguage.ARABIC
        )

    if latin_count > 0:
        return (
            DetectedLanguage.ENGLISH
        )

    return (
        DetectedLanguage.UNKNOWN
    )


def normalize_query_text(
    text: str,
) -> str:
    """Return deterministic retrieval-oriented query text.

    This representation is for retrieval only. It is not intended
    to replace the user's original display text.
    """

    text = _required_text(
        text,
        "text",
    )

    # Normalize compatibility forms, including Arabic presentation forms.
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove or normalize common invisible/non-standard spaces.
    normalized = normalized.translate(
        _SPACE_NORMALIZATION_TABLE
    )

    # Normalize Arabic/Persian digits to ASCII for consistent retrieval.
    normalized = normalized.translate(
        _DIGIT_NORMALIZATION_TABLE
    )

    # Conservative Arabic retrieval normalization.
    normalized = (
        _ARABIC_DIACRITICS_RE.sub(
            "",
            normalized,
        )
    )

    normalized = normalized.translate(
        _ARABIC_NORMALIZATION_TABLE
    )

    # Case-insensitive lexical matching for Latin text.
    normalized = normalized.casefold()

    # Collapse line breaks/tabs/repeated spaces.
    normalized = _WHITESPACE_RE.sub(
        " ",
        normalized,
    ).strip()

    if not normalized:
        raise ValueError(
            "Query normalization produced "
            "blank text."
        )

    return normalized


def _normalization_id(
    *,
    original_sha256: str,
    normalized_sha256: str,
    detected_language: DetectedLanguage,
) -> str:
    payload = "\x1f".join(
        (
            QUERY_NORMALIZATION_VERSION,
            original_sha256,
            normalized_sha256,
            detected_language.value,
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class QueryNormalizationResult:
    """Privacy-aware result of query normalization."""

    normalized_text: str

    detected_language: (
        DetectedLanguage
    )

    original_query_sha256: str

    arabic_letter_count: int

    latin_letter_count: int

    version: str = (
        QUERY_NORMALIZATION_VERSION
    )

    normalized_query_sha256: str = field(
        init=False
    )

    normalization_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        normalized_text = (
            _required_text(
                self.normalized_text,
                "normalized_text",
            )
        )

        if not isinstance(
            self.detected_language,
            DetectedLanguage,
        ):
            raise TypeError(
                "detected_language must be a "
                "DetectedLanguage."
            )

        original_hash = (
            _required_text(
                self.original_query_sha256,
                "original_query_sha256",
            )
        )

        if (
            len(original_hash) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in original_hash.lower()
            )
        ):
            raise ValueError(
                "original_query_sha256 must be "
                "a SHA-256 hexadecimal digest."
            )

        for (
            field_name,
            value,
        ) in (
            (
                "arabic_letter_count",
                self.arabic_letter_count,
            ),
            (
                "latin_letter_count",
                self.latin_letter_count,
            ),
        ):
            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a "
                    "non-negative integer."
                )

        expected_language = (
            self._language_from_counts(
                self.arabic_letter_count,
                self.latin_letter_count,
            )
        )

        if (
            expected_language
            is not self.detected_language
        ):
            raise ValueError(
                "detected_language does not "
                "match script counts."
            )

        normalized_hash = (
            _sha256_text(
                normalized_text
            )
        )

        object.__setattr__(
            self,
            "normalized_text",
            normalized_text,
        )

        object.__setattr__(
            self,
            "original_query_sha256",
            original_hash.lower(),
        )

        object.__setattr__(
            self,
            "normalized_query_sha256",
            normalized_hash,
        )

        object.__setattr__(
            self,
            "normalization_id",
            _normalization_id(
                original_sha256=(
                    original_hash.lower()
                ),
                normalized_sha256=(
                    normalized_hash
                ),
                detected_language=(
                    self.detected_language
                ),
            ),
        )

    @staticmethod
    def _language_from_counts(
        arabic_count: int,
        latin_count: int,
    ) -> DetectedLanguage:
        if (
            arabic_count > 0
            and latin_count > 0
        ):
            return (
                DetectedLanguage.MIXED
            )

        if arabic_count > 0:
            return (
                DetectedLanguage.ARABIC
            )

        if latin_count > 0:
            return (
                DetectedLanguage.ENGLISH
            )

        return (
            DetectedLanguage.UNKNOWN
        )

    @property
    def mixed_language(
        self,
    ) -> bool:
        return (
            self.detected_language
            is DetectedLanguage.MIXED
        )

    def to_bilingual_request(
        self,
        *,
        response_language: (
            BilingualLanguage | None
        ) = None,
        allow_cross_lingual: bool = True,
    ) -> BilingualQueryRequest:
        """Convert normalized text into the Stage 11.1 query contract."""

        return BilingualQueryRequest(
            query_text=(
                self.normalized_text
            ),
            detected_language=(
                self.detected_language
            ),
            response_language=(
                response_language
            ),
            allow_cross_lingual=(
                allow_cross_lingual
            ),
        )

    def to_dict(
        self,
        *,
        include_normalized_text: bool = False,
    ) -> dict[str, object]:
        """Serialize without query text by default."""

        if not isinstance(
            include_normalized_text,
            bool,
        ):
            raise TypeError(
                "include_normalized_text "
                "must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "normalization_id": (
                self.normalization_id
            ),
            "original_query_sha256": (
                self.original_query_sha256
            ),
            "normalized_query_sha256": (
                self.normalized_query_sha256
            ),
            "detected_language": (
                self.detected_language.value
            ),
            "arabic_letter_count": (
                self.arabic_letter_count
            ),
            "latin_letter_count": (
                self.latin_letter_count
            ),
            "mixed_language": (
                self.mixed_language
            ),
        }

        if include_normalized_text:
            data[
                "normalized_text"
            ] = self.normalized_text

        return data


def normalize_query(
    text: str,
) -> QueryNormalizationResult:
    """Normalize a query and detect its bilingual language state."""

    original = _required_text(
        text,
        "text",
    )

    normalized = normalize_query_text(
        original
    )

    arabic_count, latin_count = (
        count_query_scripts(
            normalized
        )
    )

    detected = (
        QueryNormalizationResult
        ._language_from_counts(
            arabic_count,
            latin_count,
        )
    )

    return QueryNormalizationResult(
        normalized_text=normalized,
        detected_language=detected,
        original_query_sha256=(
            _sha256_text(
                original
            )
        ),
        arabic_letter_count=(
            arabic_count
        ),
        latin_letter_count=(
            latin_count
        ),
    )
