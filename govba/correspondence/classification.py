"""Deterministic parsing and intent classification for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

from govba.correspondence.contract import (
    CorrespondenceClassification,
    CorrespondenceIntelligenceResult,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.rag.chunking import (
    normalize_ingested_text,
)


CORRESPONDENCE_CLASSIFICATION_VERSION = (
    "govba-correspondence-classification-v1"
)

CORRESPONDENCE_CLASSIFICATION_ALGORITHM = (
    "deterministic-correspondence-intent-v1"
)


class CorrespondenceIntentSignal(
    str,
    Enum,
):
    ESCALATION = "escalation_signal"
    COMPLAINT = "complaint_signal"
    DECISION = "decision_signal"
    APPROVAL = "approval_signal"
    ACTION_REQUIRED = "action_required_signal"
    REQUEST = "request_signal"
    NOTIFICATION = "notification_signal"
    INFORMATION = "information_signal"


_INTENT_PRECEDENCE = (
    CorrespondenceIntent.ESCALATION,
    CorrespondenceIntent.COMPLAINT,
    CorrespondenceIntent.DECISION,
    CorrespondenceIntent.APPROVAL,
    CorrespondenceIntent.ACTION_REQUIRED,
    CorrespondenceIntent.REQUEST,
    CorrespondenceIntent.NOTIFICATION,
    CorrespondenceIntent.INFORMATION,
)


_SIGNAL_TO_INTENT = {
    CorrespondenceIntentSignal.ESCALATION:
        CorrespondenceIntent.ESCALATION,

    CorrespondenceIntentSignal.COMPLAINT:
        CorrespondenceIntent.COMPLAINT,

    CorrespondenceIntentSignal.DECISION:
        CorrespondenceIntent.DECISION,

    CorrespondenceIntentSignal.APPROVAL:
        CorrespondenceIntent.APPROVAL,

    CorrespondenceIntentSignal.ACTION_REQUIRED:
        CorrespondenceIntent.ACTION_REQUIRED,

    CorrespondenceIntentSignal.REQUEST:
        CorrespondenceIntent.REQUEST,

    CorrespondenceIntentSignal.NOTIFICATION:
        CorrespondenceIntent.NOTIFICATION,

    CorrespondenceIntentSignal.INFORMATION:
        CorrespondenceIntent.INFORMATION,
}


_INTENT_PATTERNS = {
    CorrespondenceIntentSignal.ESCALATION: (
        re.compile(
            r"\bescalat(?:e|ed|ion|ing)\b"
            r"|\bimmediate\s+attention\b"
            r"|\bmanagement\s+attention\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تصعيد"
            r"|نرفع\s+إلى"
            r"|نرفع\s+الى"
            r"|للتصعيد"
            r"|عناية\s+فورية"
        ),
    ),

    CorrespondenceIntentSignal.COMPLAINT: (
        re.compile(
            r"\bcomplain(?:t|ts|ed|ing)?\b"
            r"|\bgrievance\b"
            r"|\bdissatisfied\b"
            r"|\bobjection\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"شكوى"
            r"|شكوى"
            r"|تظلم"
            r"|اعتراض"
            r"|غير\s+راض"
            r"|عدم\s+الرضا"
        ),
    ),

    CorrespondenceIntentSignal.DECISION: (
        re.compile(
            r"\bdecision\b"
            r"|\bdecided\b"
            r"|\bwe\s+have\s+decided\b"
            r"|\bhereby\s+decide\b"
            r"|\bresolution\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"قرار"
            r"|تقرر"
            r"|قررنا"
            r"|تم\s+اتخاذ\s+قرار"
        ),
    ),

    CorrespondenceIntentSignal.APPROVAL: (
        re.compile(
            r"\bapproval\b"
            r"|\bapprove(?:d|s|ing)?\b"
            r"|\bauthori[sz]ation\b"
            r"|\bauthori[sz]e(?:d)?\b"
            r"|\bconsent\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"موافق(?:ة|ات)"
            r"|الموافقة"
            r"|اعتماد"
            r"|تفويض"
            r"|إجازة"
        ),
    ),

    CorrespondenceIntentSignal.ACTION_REQUIRED: (
        re.compile(
            r"\bmust\b"
            r"|\bshall\b"
            r"|\brequired\s+to\b"
            r"|\bmandatory\b"
            r"|\baction\s+required\b"
            r"|\bneeds?\s+to\b"
            r"|\bhas\s+to\b"
            r"|\bhave\s+to\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يجب"
            r"|يتعين"
            r"|يلزم"
            r"|مطلوب\s+من"
            r"|إجراء\s+مطلوب"
            r"|اجراء\s+مطلوب"
            r"|إلزامي"
            r"|الزامي"
        ),
    ),

    CorrespondenceIntentSignal.REQUEST: (
        re.compile(
            r"\bplease\b"
            r"|\bkindly\b"
            r"|\bwe\s+request\b"
            r"|\bi\s+request\b"
            r"|\brequesting\b"
            r"|\brequest\s+for\b"
            r"|\bcould\s+you\b"
            r"|\bwould\s+you\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يرجى"
            r"|نرجو"
            r"|أرجو"
            r"|ارجو"
            r"|نطلب"
            r"|طلبنا"
            r"|نلتمس"
        ),
    ),

    CorrespondenceIntentSignal.NOTIFICATION: (
        re.compile(
            r"\bnotification\b"
            r"|\bnotice\b"
            r"|\bhereby\s+notify\b"
            r"|\bwe\s+notify\b"
            r"|\bto\s+notify\s+you\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"إشعار"
            r"|اشعار"
            r"|نحيطكم\s+علما"
            r"|نحيطكم\s+علمًا"
            r"|نبلغكم"
            r"|إبلاغ"
        ),
    ),

    CorrespondenceIntentSignal.INFORMATION: (
        re.compile(
            r"\bfor\s+your\s+information\b"
            r"|\bfor\s+information\b"
            r"|\bfyi\b"
            r"|\bfor\s+reference\b"
            r"|\binformational\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"للعلم"
            r"|للعلم\s+فقط"
            r"|للاطلاع"
            r"|للإحاطة"
            r"|للاحاطة"
            r"|للمعلومية"
        ),
    ),
}


_ACTION_INTENTS = {
    CorrespondenceIntent.ACTION_REQUIRED,
    CorrespondenceIntent.REQUEST,
    CorrespondenceIntent.APPROVAL,
    CorrespondenceIntent.ESCALATION,
    CorrespondenceIntent.COMPLAINT,
}


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _normalize_subject(
    value: str,
) -> str:
    if not value:
        return ""

    normalized = (
        normalize_ingested_text(
            value
        )
    )

    return " ".join(
        normalized.splitlines()
    ).strip()


@dataclass(frozen=True)
class ParsedCorrespondence:
    """Normalized correspondence structure before intelligence extraction."""

    request_id: str

    normalized_subject: str

    normalized_body: str

    paragraphs: tuple[
        str,
        ...
    ]

    version: str = (
        CORRESPONDENCE_CLASSIFICATION_VERSION
    )

    parser_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.request_id,
            str,
        ):
            raise TypeError(
                "request_id must be a string."
            )

        request_id = (
            self.request_id.strip()
        )

        if not request_id:
            raise ValueError(
                "request_id must not be blank."
            )

        if not isinstance(
            self.normalized_subject,
            str,
        ):
            raise TypeError(
                "normalized_subject must "
                "be a string."
            )

        if not isinstance(
            self.normalized_body,
            str,
        ):
            raise TypeError(
                "normalized_body must "
                "be a string."
            )

        body = (
            self.normalized_body.strip()
        )

        if not body:
            raise ValueError(
                "normalized_body must "
                "not be blank."
            )

        try:
            paragraphs = tuple(
                self.paragraphs
            )
        except TypeError as exc:
            raise TypeError(
                "paragraphs must be iterable."
            ) from exc

        if not paragraphs:
            raise ValueError(
                "At least one paragraph "
                "is required."
            )

        normalized_paragraphs = []

        for paragraph in paragraphs:
            if not isinstance(
                paragraph,
                str,
            ):
                raise TypeError(
                    "paragraphs must contain "
                    "strings."
                )

            value = paragraph.strip()

            if not value:
                raise ValueError(
                    "paragraphs cannot contain "
                    "blank values."
                )

            normalized_paragraphs.append(
                value
            )

        paragraphs = tuple(
            normalized_paragraphs
        )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                _sha256(
                    self.normalized_subject
                ),
                _sha256(
                    body
                ),
                ",".join(
                    _sha256(
                        paragraph
                    )
                    for paragraph
                    in paragraphs
                ),
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "normalized_body",
            body,
        )

        object.__setattr__(
            self,
            "paragraphs",
            paragraphs,
        )

        object.__setattr__(
            self,
            "parser_id",
            _sha256(
                payload
            ),
        )

    @property
    def paragraph_count(
        self,
    ) -> int:
        return len(
            self.paragraphs
        )

    @property
    def character_count(
        self,
    ) -> int:
        return len(
            self.normalized_body
        )

    def to_dict(
        self,
        *,
        include_text: bool = False,
    ) -> dict[str, object]:
        """Serialize without correspondence text by default."""

        if not isinstance(
            include_text,
            bool,
        ):
            raise TypeError(
                "include_text must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "parser_id": (
                self.parser_id
            ),
            "request_id": (
                self.request_id
            ),
            "subject_sha256": (
                _sha256(
                    self.normalized_subject
                )
                if self.normalized_subject
                else None
            ),
            "body_sha256": (
                _sha256(
                    self.normalized_body
                )
            ),
            "paragraph_count": (
                self.paragraph_count
            ),
            "character_count": (
                self.character_count
            ),
            "paragraph_hashes": [
                _sha256(
                    paragraph
                )
                for paragraph
                in self.paragraphs
            ],
        }

        if include_text:
            data[
                "normalized_subject"
            ] = self.normalized_subject

            data[
                "normalized_body"
            ] = self.normalized_body

            data[
                "paragraphs"
            ] = list(
                self.paragraphs
            )

        return data


def parse_correspondence(
    request: CorrespondenceRequest,
) -> ParsedCorrespondence:
    """Normalize correspondence into deterministic paragraphs."""

    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    normalized_body = (
        normalize_ingested_text(
            request.body_text
        )
    )

    if not normalized_body:
        raise ValueError(
            "Correspondence produced "
            "no normalized body."
        )

    paragraphs = tuple(
        part.strip()
        for part in re.split(
            r"\n\s*\n",
            normalized_body,
        )
        if part.strip()
    )

    if not paragraphs:
        paragraphs = (
            normalized_body,
        )

    return ParsedCorrespondence(
        request_id=(
            request.request_id
        ),
        normalized_subject=(
            _normalize_subject(
                request.subject_text
            )
        ),
        normalized_body=(
            normalized_body
        ),
        paragraphs=paragraphs,
    )


def detect_intent_signals(
    parsed: ParsedCorrespondence,
) -> tuple[
    CorrespondenceIntentSignal,
    ...
]:
    """Detect intent signals without retaining matched correspondence text."""

    if not isinstance(
        parsed,
        ParsedCorrespondence,
    ):
        raise TypeError(
            "parsed must be a "
            "ParsedCorrespondence."
        )

    searchable = "\n".join(
        value
        for value in (
            parsed.normalized_subject,
            parsed.normalized_body,
        )
        if value
    )

    signals = []

    for signal in (
        CorrespondenceIntentSignal
    ):
        patterns = (
            _INTENT_PATTERNS[
                signal
            ]
        )

        if any(
            pattern.search(
                searchable
            )
            for pattern
            in patterns
        ):
            signals.append(
                signal
            )

    return tuple(
        signals
    )


def _choose_intent(
    signals: tuple[
        CorrespondenceIntentSignal,
        ...
    ],
) -> CorrespondenceIntent:
    detected_intents = {
        _SIGNAL_TO_INTENT[
            signal
        ]
        for signal
        in signals
    }

    for intent in (
        _INTENT_PRECEDENCE
    ):
        if intent in detected_intents:
            return intent

    return CorrespondenceIntent.UNKNOWN


def _classification_confidence(
    intent: CorrespondenceIntent,
    signals: tuple[
        CorrespondenceIntentSignal,
        ...
    ],
) -> float:
    if (
        intent
        is CorrespondenceIntent.UNKNOWN
    ):
        return 0.25

    matching_signal_count = sum(
        1
        for signal in signals
        if (
            _SIGNAL_TO_INTENT[
                signal
            ]
            is intent
        )
    )

    competing_count = sum(
        1
        for signal in signals
        if (
            _SIGNAL_TO_INTENT[
                signal
            ]
            is not intent
        )
    )

    confidence = (
        0.72
        + 0.08
        * max(
            0,
            matching_signal_count - 1,
        )
        - 0.03
        * competing_count
    )

    return max(
        0.50,
        min(
            0.95,
            confidence,
        ),
    )


def classify_correspondence_intent(
    request: CorrespondenceRequest,
    *,
    parsed: ParsedCorrespondence | None = None,
) -> CorrespondenceClassification:
    """Classify correspondence intent deterministically."""

    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    if parsed is None:
        parsed = parse_correspondence(
            request
        )

    if not isinstance(
        parsed,
        ParsedCorrespondence,
    ):
        raise TypeError(
            "parsed must be a "
            "ParsedCorrespondence."
        )

    if (
        parsed.request_id
        != request.request_id
    ):
        raise ValueError(
            "Parsed correspondence request ID "
            "does not match request."
        )

    signals = detect_intent_signals(
        parsed
    )

    intent = _choose_intent(
        signals
    )

    requires_action = (
        intent
        in _ACTION_INTENTS
    )

    confidence = (
        _classification_confidence(
            intent,
            signals,
        )
    )

    reason_codes = tuple(
        signal.value
        for signal
        in signals
    )

    if not reason_codes:
        reason_codes = (
            "no_deterministic_intent_signal",
        )

    return CorrespondenceClassification(
        request_id=(
            request.request_id
        ),
        intent=intent,

        # Final priority intelligence is Stage 15.4.
        priority=(
            CorrespondencePriority.NORMAL
        ),

        requires_action=(
            requires_action
        ),
        confidence=confidence,
        reason_codes=reason_codes,
    )


def analyze_correspondence_intent(
    request: CorrespondenceRequest,
) -> CorrespondenceIntelligenceResult:
    """Run deterministic parsing and intent classification."""

    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    parsed = parse_correspondence(
        request
    )

    classification = (
        classify_correspondence_intent(
            request,
            parsed=parsed,
        )
    )

    return CorrespondenceIntelligenceResult(
        request_id=(
            request.request_id
        ),
        trace_id=(
            request.trace_id
        ),
        classification=(
            classification
        ),
    )
