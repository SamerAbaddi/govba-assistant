"""Deterministic PII detection and redaction for GovBA-GAR."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from govba.governance.security import (
    SecurityAssessment,
    SecurityDecision,
    SecurityFinding,
    SecurityIssueCode,
    SecuritySeverity,
    SecuritySurface,
    assess_security,
)


PII_REDACTION_VERSION = (
    "govba-pii-redaction-v1"
)


class PIIType(
    str,
    Enum,
):
    """PII classes supported by the deterministic detector."""

    EMAIL = "email"
    PHONE = "phone"
    IPV4_ADDRESS = "ipv4_address"
    IBAN = "iban"
    PAYMENT_CARD = "payment_card"


_SENSITIVE_TYPES = {
    PIIType.IBAN,
    PIIType.PAYMENT_CARD,
}


_REPLACEMENTS = {
    PIIType.EMAIL: "[REDACTED_EMAIL]",
    PIIType.PHONE: "[REDACTED_PHONE]",
    PIIType.IPV4_ADDRESS: "[REDACTED_IP]",
    PIIType.IBAN: "[REDACTED_IBAN]",
    PIIType.PAYMENT_CARD: "[REDACTED_PAYMENT_CARD]",
}


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])"
    r"[A-Z0-9._%+-]+"
    r"@[A-Z0-9.-]+"
    r"\.[A-Z]{2,63}"
    r"(?![\w-]|\.(?=\w))",
    flags=re.IGNORECASE,
)


_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"\+[1-9](?:[ -]?\d){7,14}"
    r"|"
    r"(?:00962|0)7[789](?:[ -]?\d){7}"
    r")"
    r"(?!\w)"
)


_IPV4_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d{1,3}\.){3}"
    r"\d{1,3}"
    r"(?!\d)"
)


_IBAN_RE = re.compile(
    r"(?<![A-Z0-9])"
    r"[A-Z]{2}\d{2}"
    r"(?:[ -]?[A-Z0-9]){11,30}"
    r"(?![A-Z0-9])",
    flags=re.IGNORECASE,
)


_PAYMENT_CARD_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d[ -]?){12,18}\d"
    r"(?!\d)"
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


def _normalize_digits(
    value: str,
) -> str:
    return "".join(
        character
        for character in value
        if character.isdigit()
    )


def _luhn_valid(
    value: str,
) -> bool:
    digits = _normalize_digits(
        value
    )

    if not (
        13 <= len(digits) <= 19
    ):
        return False

    if len(set(digits)) == 1:
        return False

    total = 0
    parity = len(digits) % 2

    for index, character in enumerate(
        digits
    ):
        number = int(
            character
        )

        if index % 2 == parity:
            number *= 2

            if number > 9:
                number -= 9

        total += number

    return (
        total % 10 == 0
    )


def _iban_valid(
    value: str,
) -> bool:
    normalized = "".join(
        character
        for character in value.upper()
        if character.isalnum()
    )

    if not (
        15 <= len(normalized) <= 34
    ):
        return False

    if (
        not normalized[:2].isalpha()
        or not normalized[2:4].isdigit()
    ):
        return False

    rearranged = (
        normalized[4:]
        + normalized[:4]
    )

    remainder = 0

    for character in rearranged:
        if character.isdigit():
            converted = character
        elif (
            "A"
            <= character
            <= "Z"
        ):
            converted = str(
                ord(character)
                - ord("A")
                + 10
            )
        else:
            return False

        for digit in converted:
            remainder = (
                remainder * 10
                + int(digit)
            ) % 97

    return (
        remainder == 1
    )


def _ipv4_valid(
    value: str,
) -> bool:
    try:
        address = ipaddress.ip_address(
            value
        )
    except ValueError:
        return False

    return (
        address.version == 4
    )


@dataclass(frozen=True)
class PIIMatch:
    """PII location metadata without storing the matched raw value."""

    pii_type: PIIType

    start: int
    end: int

    replacement: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.pii_type,
            PIIType,
        ):
            raise TypeError(
                "pii_type must be a PIIType."
            )

        for field_name in (
            "start",
            "end",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

        if self.start < 0:
            raise ValueError(
                "start cannot be negative."
            )

        if self.end <= self.start:
            raise ValueError(
                "end must be greater than start."
            )

        expected = _REPLACEMENTS[
            self.pii_type
        ]

        if (
            self.replacement
            != expected
        ):
            raise ValueError(
                "replacement does not match "
                "the PII type."
            )

    @property
    def sensitive(
        self,
    ) -> bool:
        return (
            self.pii_type
            in _SENSITIVE_TYPES
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "pii_type": (
                self.pii_type.value
            ),
            "start": (
                self.start
            ),
            "end": (
                self.end
            ),
            "replacement": (
                self.replacement
            ),
            "sensitive": (
                self.sensitive
            ),
        }


@dataclass(frozen=True)
class PIIRedactionResult:
    """Privacy-conscious result of deterministic PII redaction."""

    trace_id: str
    input_sha256: str

    redacted_text: str

    matches: tuple[
        PIIMatch,
        ...
    ]

    security_assessment: (
        SecurityAssessment
    )

    version: str = (
        PII_REDACTION_VERSION
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        input_hash = _required_text(
            self.input_sha256,
            "input_sha256",
        ).lower()

        if (
            len(input_hash) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in input_hash
            )
        ):
            raise ValueError(
                "input_sha256 must be a "
                "SHA-256 hexadecimal digest."
            )

        if not isinstance(
            self.redacted_text,
            str,
        ):
            raise TypeError(
                "redacted_text must be a string."
            )

        try:
            matches = tuple(
                self.matches
            )
        except TypeError as exc:
            raise TypeError(
                "matches must be iterable."
            ) from exc

        for match in matches:
            if not isinstance(
                match,
                PIIMatch,
            ):
                raise TypeError(
                    "matches must contain only "
                    "PIIMatch values."
                )

        previous_end = -1

        for match in matches:
            if match.start < previous_end:
                raise ValueError(
                    "PII matches must not overlap."
                )

            previous_end = match.end

        if not isinstance(
            self.security_assessment,
            SecurityAssessment,
        ):
            raise TypeError(
                "security_assessment must be "
                "a SecurityAssessment."
            )

        if (
            self.security_assessment.trace_id
            != trace_id
        ):
            raise ValueError(
                "Security assessment trace ID "
                "must match redaction trace ID."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "input_sha256",
            input_hash,
        )

        object.__setattr__(
            self,
            "matches",
            matches,
        )

    @property
    def pii_detected(
        self,
    ) -> bool:
        return bool(
            self.matches
        )

    @property
    def sensitive_pii_detected(
        self,
    ) -> bool:
        return any(
            match.sensitive
            for match
            in self.matches
        )

    @property
    def match_count(
        self,
    ) -> int:
        return len(
            self.matches
        )

    @property
    def pii_types(
        self,
    ) -> tuple[
        PIIType,
        ...
    ]:
        return tuple(
            sorted(
                {
                    match.pii_type
                    for match
                    in self.matches
                },
                key=lambda item: (
                    item.value
                ),
            )
        )

    def to_dict(
        self,
        *,
        include_redacted_text: bool = False,
    ) -> dict[str, object]:
        """Serialize safely; text is excluded by default."""

        if not isinstance(
            include_redacted_text,
            bool,
        ):
            raise TypeError(
                "include_redacted_text "
                "must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "trace_id": (
                self.trace_id
            ),
            "input_sha256": (
                self.input_sha256
            ),
            "pii_detected": (
                self.pii_detected
            ),
            "sensitive_pii_detected": (
                self.sensitive_pii_detected
            ),
            "match_count": (
                self.match_count
            ),
            "pii_types": [
                pii_type.value
                for pii_type
                in self.pii_types
            ],
            "matches": [
                match.to_dict()
                for match
                in self.matches
            ],
            "security_assessment": (
                self.security_assessment
                .to_dict()
            ),
        }

        if include_redacted_text:
            data[
                "redacted_text"
            ] = self.redacted_text

        return data


@dataclass(frozen=True)
class _Candidate:
    pii_type: PIIType
    start: int
    end: int


_TYPE_PRIORITY = {
    PIIType.PAYMENT_CARD: 0,
    PIIType.IBAN: 1,
    PIIType.EMAIL: 2,
    PIIType.PHONE: 3,
    PIIType.IPV4_ADDRESS: 4,
}


def _candidate_matches(
    text: str,
) -> tuple[
    _Candidate,
    ...
]:
    candidates = []

    for match in _EMAIL_RE.finditer(
        text
    ):
        candidates.append(
            _Candidate(
                pii_type=PIIType.EMAIL,
                start=match.start(),
                end=match.end(),
            )
        )

    for match in _IBAN_RE.finditer(
        text
    ):
        if _iban_valid(
            match.group(0)
        ):
            candidates.append(
                _Candidate(
                    pii_type=PIIType.IBAN,
                    start=match.start(),
                    end=match.end(),
                )
            )

    for match in (
        _PAYMENT_CARD_RE.finditer(
            text
        )
    ):
        if _luhn_valid(
            match.group(0)
        ):
            candidates.append(
                _Candidate(
                    pii_type=(
                        PIIType.PAYMENT_CARD
                    ),
                    start=match.start(),
                    end=match.end(),
                )
            )

    for match in _PHONE_RE.finditer(
        text
    ):
        candidates.append(
            _Candidate(
                pii_type=PIIType.PHONE,
                start=match.start(),
                end=match.end(),
            )
        )

    for match in _IPV4_RE.finditer(
        text
    ):
        if _ipv4_valid(
            match.group(0)
        ):
            candidates.append(
                _Candidate(
                    pii_type=(
                        PIIType.IPV4_ADDRESS
                    ),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return tuple(
        candidates
    )


def _select_non_overlapping(
    candidates: Iterable[
        _Candidate
    ],
) -> tuple[
    PIIMatch,
    ...
]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start,
            -(
                candidate.end
                - candidate.start
            ),
            _TYPE_PRIORITY[
                candidate.pii_type
            ],
            candidate.pii_type.value,
        ),
    )

    selected = []

    for candidate in ordered:
        overlaps = any(
            not (
                candidate.end
                <= existing.start
                or candidate.start
                >= existing.end
            )
            for existing
            in selected
        )

        if overlaps:
            continue

        selected.append(
            PIIMatch(
                pii_type=(
                    candidate.pii_type
                ),
                start=candidate.start,
                end=candidate.end,
                replacement=(
                    _REPLACEMENTS[
                        candidate.pii_type
                    ]
                ),
            )
        )

    return tuple(
        sorted(
            selected,
            key=lambda match: (
                match.start,
                match.end,
                match.pii_type.value,
            ),
        )
    )


def detect_pii(
    text: str,
) -> tuple[
    PIIMatch,
    ...
]:
    """Detect supported PII without returning matched raw values."""

    value = _required_text(
        text,
        "text",
    )

    return _select_non_overlapping(
        _candidate_matches(
            value
        )
    )


def _redact_text(
    text: str,
    matches: tuple[
        PIIMatch,
        ...
    ],
) -> str:
    if not matches:
        return text

    output = []
    cursor = 0

    for match in matches:
        output.append(
            text[
                cursor:match.start
            ]
        )

        output.append(
            match.replacement
        )

        cursor = match.end

    output.append(
        text[
            cursor:
        ]
    )

    return "".join(
        output
    )


def _security_findings(
    *,
    matches: tuple[
        PIIMatch,
        ...
    ],
    surface: SecuritySurface,
    input_sha256: str,
) -> tuple[
    SecurityFinding,
    ...
]:
    counts = Counter(
        match.pii_type
        for match
        in matches
    )

    findings = []

    for pii_type in sorted(
        counts,
        key=lambda item: (
            item.value
        ),
    ):
        sensitive = (
            pii_type
            in _SENSITIVE_TYPES
        )

        findings.append(
            SecurityFinding(
                code=(
                    SecurityIssueCode
                    .SENSITIVE_PII_DETECTED
                    if sensitive
                    else SecurityIssueCode
                    .PII_DETECTED
                ),
                surface=surface,
                severity=(
                    SecuritySeverity.HIGH
                    if sensitive
                    else SecuritySeverity.MEDIUM
                ),
                decision=(
                    SecurityDecision.REDACT
                ),
                occurrence_count=(
                    counts[
                        pii_type
                    ]
                ),
                reference_id=(
                    f"{input_sha256[:16]}:"
                    f"{pii_type.value}"
                ),
            )
        )

    return tuple(
        findings
    )


def redact_pii(
    text: str,
    *,
    trace_id: str,
    surface: SecuritySurface,
) -> PIIRedactionResult:
    """Detect and redact supported PII deterministically."""

    value = _required_text(
        text,
        "text",
    )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if not isinstance(
        surface,
        SecuritySurface,
    ):
        raise TypeError(
            "surface must be a "
            "SecuritySurface."
        )

    input_hash = _sha256_text(
        value
    )

    matches = detect_pii(
        value
    )

    redacted = _redact_text(
        value,
        matches,
    )

    findings = _security_findings(
        matches=matches,
        surface=surface,
        input_sha256=input_hash,
    )

    assessment = assess_security(
        trace_id=trace,
        findings=findings,
    )

    return PIIRedactionResult(
        trace_id=trace,
        input_sha256=input_hash,
        redacted_text=redacted,
        matches=matches,
        security_assessment=assessment,
    )
