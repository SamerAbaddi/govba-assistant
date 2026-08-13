"""Deterministic deadline and priority intelligence for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum

from govba.correspondence.classification import (
    ParsedCorrespondence,
    parse_correspondence,
)
from govba.correspondence.contract import (
    CorrespondenceClassification,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)


CORRESPONDENCE_PRIORITY_VERSION = (
    "govba-correspondence-priority-v1"
)

CORRESPONDENCE_PRIORITY_ALGORITHM = (
    "deterministic-priority-deadline-v1"
)


class CorrespondenceDeadlineKind(str, Enum):
    RELATIVE_HOURS = "relative_hours"
    RELATIVE_DAYS = "relative_days"
    ABSOLUTE_DATE = "absolute_date"
    TODAY = "today"
    TOMORROW = "tomorrow"


class CorrespondencePriorityReason(str, Enum):
    EXPLICIT_URGENT = "explicit_urgent"
    EXPLICIT_HIGH_PRIORITY = "explicit_high_priority"
    OVERDUE_DEADLINE = "overdue_deadline"
    DUE_WITHIN_24_HOURS = "due_within_24_hours"
    DUE_WITHIN_72_HOURS = "due_within_72_hours"
    ESCALATION = "escalation"
    ROUTINE_INFORMATION = "routine_information"
    STANDARD_PROCESSING = "standard_processing"


_ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


_RELATIVE_HOURS_PATTERNS = (
    re.compile(
        r"\bwithin\s+(\d{1,4})\s+hours?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"خلال\s+(\d{1,4})\s*"
        r"(?:ساعة|ساعات)"
    ),
)

_RELATIVE_DAYS_PATTERNS = (
    re.compile(
        r"\bwithin\s+(\d{1,4})\s+days?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"خلال\s+(\d{1,4})\s*"
        r"(?:يوم|أيام|ايام)"
    ),
)

_ISO_DATE_PATTERNS = (
    re.compile(
        r"\b(?:by|before|due(?:\s+on)?|"
        r"no\s+later\s+than)\s+"
        r"(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:قبل|بحلول|"
        r"في\s+موعد\s+أقصاه|"
        r"في\s+موعد\s+اقصاه)"
        r"\s*(20\d{2})-(\d{1,2})-(\d{1,2})"
    ),
)

_DMY_DATE_PATTERNS = (
    re.compile(
        r"\b(?:by|before|due(?:\s+on)?|"
        r"no\s+later\s+than)\s+"
        r"(\d{1,2})/(\d{1,2})/(20\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:قبل|بحلول|"
        r"في\s+موعد\s+أقصاه|"
        r"في\s+موعد\s+اقصاه)"
        r"\s*(\d{1,2})/(\d{1,2})/(20\d{2})"
    ),
)

_TODAY_PATTERNS = (
    re.compile(
        r"\b(?:by|before|due)\s+today\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:اليوم|قبل\s+نهاية\s+اليوم)"
    ),
)

_TOMORROW_PATTERNS = (
    re.compile(
        r"\b(?:by|before|due)\s+tomorrow\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:بحلول|قبل)\s+غد(?:اً|ًا|ا)?"
    ),
)

_URGENT_PATTERNS = (
    re.compile(
        r"\burgent\b"
        r"|\bimmediately\b"
        r"|\basap\b"
        r"|\bas\s+soon\s+as\s+possible\b"
        r"|\bwithout\s+delay\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"عاجل"
        r"|فوراً"
        r"|فورًا"
        r"|فورا"
        r"|بشكل\s+فوري"
        r"|على\s+الفور"
        r"|دون\s+تأخير"
    ),
)

_HIGH_PRIORITY_PATTERNS = (
    re.compile(
        r"\bhigh\s+priority\b"
        r"|\bpriority\s+matter\b"
        r"|\bimportant\s+matter\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"أولوية\s+عالية"
        r"|اولوية\s+عالية"
        r"|أهمية\s+عالية"
        r"|اهمية\s+عالية"
    ),
)


def _required_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _aware_datetime(
    value: datetime,
    field_name: str,
) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} must be a datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware."
        )

    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _normalized_digits(
    value: str,
) -> str:
    return value.translate(
        _ARABIC_DIGITS
    )


def _matches_any(
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...
    ],
) -> bool:
    return any(
        pattern.search(text)
        for pattern in patterns
    )


@dataclass(frozen=True)
class CorrespondenceDeadline:
    request_id: str
    sequence: int
    kind: CorrespondenceDeadlineKind
    deadline_at: datetime
    confidence: float
    source_text: str

    version: str = (
        CORRESPONDENCE_PRIORITY_VERSION
    )

    source_sha256: str = field(
        init=False
    )

    deadline_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        if (
            isinstance(self.sequence, bool)
            or not isinstance(
                self.sequence,
                int,
            )
            or self.sequence < 0
        ):
            raise ValueError(
                "sequence must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.kind,
            CorrespondenceDeadlineKind,
        ):
            raise TypeError(
                "kind must be a "
                "CorrespondenceDeadlineKind."
            )

        deadline_at = _aware_datetime(
            self.deadline_at,
            "deadline_at",
        )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(
                self.confidence,
                (int, float),
            )
            or not math.isfinite(
                self.confidence
            )
            or not (
                0.0
                <= self.confidence
                <= 1.0
            )
        ):
            raise ValueError(
                "confidence must be between "
                "0 and 1."
            )

        source_text = _required_text(
            self.source_text,
            "source_text",
        )

        source_hash = _sha256(
            source_text
        )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                str(self.sequence),
                self.kind.value,
                deadline_at.isoformat(),
                f"{float(self.confidence):.12f}",
                source_hash,
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "deadline_at",
            deadline_at,
        )

        object.__setattr__(
            self,
            "confidence",
            float(self.confidence),
        )

        object.__setattr__(
            self,
            "source_text",
            source_text,
        )

        object.__setattr__(
            self,
            "source_sha256",
            source_hash,
        )

        object.__setattr__(
            self,
            "deadline_id",
            _sha256(payload),
        )

    def to_dict(
        self,
        *,
        include_text: bool = False,
    ) -> dict[str, object]:
        if not isinstance(
            include_text,
            bool,
        ):
            raise TypeError(
                "include_text must be boolean."
            )

        data = {
            "version": self.version,
            "deadline_id": self.deadline_id,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "deadline_at": (
                self.deadline_at.isoformat()
            ),
            "confidence": self.confidence,
            "source_sha256": (
                self.source_sha256
            ),
        }

        if include_text:
            data["source_text"] = (
                self.source_text
            )

        return data


@dataclass(frozen=True)
class CorrespondenceDeadlineResult:
    request_id: str
    parser_id: str

    deadlines: tuple[
        CorrespondenceDeadline,
        ...
    ]

    version: str = (
        CORRESPONDENCE_PRIORITY_VERSION
    )

    extraction_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        parser_id = _required_text(
            self.parser_id,
            "parser_id",
        )

        try:
            deadlines = tuple(
                self.deadlines
            )
        except TypeError as exc:
            raise TypeError(
                "deadlines must be iterable."
            ) from exc

        for deadline in deadlines:
            if not isinstance(
                deadline,
                CorrespondenceDeadline,
            ):
                raise TypeError(
                    "deadlines must contain only "
                    "CorrespondenceDeadline values."
                )

            if (
                deadline.request_id
                != request_id
            ):
                raise ValueError(
                    "Deadline request ID mismatch."
                )

        sequences = tuple(
            value.sequence
            for value in deadlines
        )

        if sequences != tuple(
            range(len(deadlines))
        ):
            raise ValueError(
                "Deadline sequences must be "
                "contiguous from zero."
            )

        if (
            len(
                {
                    value.deadline_id
                    for value in deadlines
                }
            )
            != len(deadlines)
        ):
            raise ValueError(
                "Deadlines must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                parser_id,
                ",".join(
                    value.deadline_id
                    for value in deadlines
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
            "parser_id",
            parser_id,
        )

        object.__setattr__(
            self,
            "deadlines",
            deadlines,
        )

        object.__setattr__(
            self,
            "extraction_id",
            _sha256(payload),
        )

    @property
    def earliest_deadline(
        self,
    ) -> CorrespondenceDeadline | None:
        if not self.deadlines:
            return None

        return min(
            self.deadlines,
            key=lambda value: (
                value.deadline_at,
                value.sequence,
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "extraction_id": (
                self.extraction_id
            ),
            "request_id": self.request_id,
            "parser_id": self.parser_id,
            "deadline_count": len(
                self.deadlines
            ),
            "earliest_deadline_at": (
                self.earliest_deadline
                .deadline_at.isoformat()
                if self.earliest_deadline
                else None
            ),
            "deadlines": [
                value.to_dict()
                for value in self.deadlines
            ],
        }


def extract_correspondence_deadlines(
    request: CorrespondenceRequest,
    *,
    parsed: ParsedCorrespondence | None = None,
) -> CorrespondenceDeadlineResult:
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
            "Parsed correspondence does not "
            "match request."
        )

    searchable = "\n".join(
        value
        for value in (
            parsed.normalized_subject,
            parsed.normalized_body,
        )
        if value
    )

    searchable = _normalized_digits(
        searchable
    )

    timezone_value = (
        request.received_at.tzinfo
    )

    extracted = []

    for pattern in (
        _RELATIVE_HOURS_PATTERNS
    ):
        for match in pattern.finditer(
            searchable
        ):
            count = int(
                match.group(1)
            )

            if count <= 0:
                continue

            extracted.append(
                (
                    request.received_at
                    + timedelta(
                        hours=count
                    ),
                    CorrespondenceDeadlineKind
                    .RELATIVE_HOURS,
                    0.95,
                    match.group(0),
                )
            )

    for pattern in (
        _RELATIVE_DAYS_PATTERNS
    ):
        for match in pattern.finditer(
            searchable
        ):
            count = int(
                match.group(1)
            )

            if count <= 0:
                continue

            extracted.append(
                (
                    request.received_at
                    + timedelta(
                        days=count
                    ),
                    CorrespondenceDeadlineKind
                    .RELATIVE_DAYS,
                    0.95,
                    match.group(0),
                )
            )

    for pattern in _ISO_DATE_PATTERNS:
        for match in pattern.finditer(
            searchable
        ):
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))

            try:
                deadline = datetime.combine(
                    datetime(
                        year,
                        month,
                        day,
                    ).date(),
                    time(
                        23,
                        59,
                        59,
                    ),
                    tzinfo=timezone_value,
                )
            except ValueError:
                continue

            extracted.append(
                (
                    deadline,
                    CorrespondenceDeadlineKind
                    .ABSOLUTE_DATE,
                    0.98,
                    match.group(0),
                )
            )

    for pattern in _DMY_DATE_PATTERNS:
        for match in pattern.finditer(
            searchable
        ):
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))

            try:
                deadline = datetime.combine(
                    datetime(
                        year,
                        month,
                        day,
                    ).date(),
                    time(
                        23,
                        59,
                        59,
                    ),
                    tzinfo=timezone_value,
                )
            except ValueError:
                continue

            extracted.append(
                (
                    deadline,
                    CorrespondenceDeadlineKind
                    .ABSOLUTE_DATE,
                    0.96,
                    match.group(0),
                )
            )

    for pattern in _TODAY_PATTERNS:
        for match in pattern.finditer(
            searchable
        ):
            deadline = datetime.combine(
                request.received_at.date(),
                time(
                    23,
                    59,
                    59,
                ),
                tzinfo=timezone_value,
            )

            extracted.append(
                (
                    deadline,
                    CorrespondenceDeadlineKind
                    .TODAY,
                    0.95,
                    match.group(0),
                )
            )

    for pattern in _TOMORROW_PATTERNS:
        for match in pattern.finditer(
            searchable
        ):
            date_value = (
                request.received_at.date()
                + timedelta(days=1)
            )

            deadline = datetime.combine(
                date_value,
                time(
                    23,
                    59,
                    59,
                ),
                tzinfo=timezone_value,
            )

            extracted.append(
                (
                    deadline,
                    CorrespondenceDeadlineKind
                    .TOMORROW,
                    0.95,
                    match.group(0),
                )
            )

    unique = {}

    for (
        deadline,
        kind,
        confidence,
        source_text,
    ) in extracted:
        key = (
            deadline.isoformat(),
            kind.value,
            _sha256(source_text),
        )

        unique[key] = (
            deadline,
            kind,
            confidence,
            source_text,
        )

    ordered = sorted(
        unique.values(),
        key=lambda value: (
            value[0],
            value[1].value,
            _sha256(value[3]),
        ),
    )

    deadlines = tuple(
        CorrespondenceDeadline(
            request_id=request.request_id,
            sequence=index,
            kind=kind,
            deadline_at=deadline,
            confidence=confidence,
            source_text=source_text,
        )
        for index, (
            deadline,
            kind,
            confidence,
            source_text,
        )
        in enumerate(ordered)
    )

    return CorrespondenceDeadlineResult(
        request_id=request.request_id,
        parser_id=parsed.parser_id,
        deadlines=deadlines,
    )


@dataclass(frozen=True)
class CorrespondencePriorityAssessment:
    request_id: str
    classification_id: str
    deadline_extraction_id: str

    priority: CorrespondencePriority

    reasons: tuple[
        CorrespondencePriorityReason,
        ...
    ]

    assessed_at: datetime

    earliest_deadline_at: (
        datetime | None
    )

    overdue: bool

    seconds_to_deadline: (
        float | None
    )

    version: str = (
        CORRESPONDENCE_PRIORITY_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        classification_id = (
            _required_text(
                self.classification_id,
                "classification_id",
            )
        )

        deadline_extraction_id = (
            _required_text(
                self.deadline_extraction_id,
                "deadline_extraction_id",
            )
        )

        if not isinstance(
            self.priority,
            CorrespondencePriority,
        ):
            raise TypeError(
                "priority must be a "
                "CorrespondencePriority."
            )

        try:
            reasons = tuple(
                self.reasons
            )
        except TypeError as exc:
            raise TypeError(
                "reasons must be iterable."
            ) from exc

        if not reasons:
            raise ValueError(
                "At least one priority reason "
                "is required."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                CorrespondencePriorityReason,
            ):
                raise TypeError(
                    "reasons must contain only "
                    "CorrespondencePriorityReason values."
                )

        if len(set(reasons)) != len(reasons):
            raise ValueError(
                "Priority reasons must be unique."
            )

        assessed_at = _aware_datetime(
            self.assessed_at,
            "assessed_at",
        )

        earliest = (
            self.earliest_deadline_at
        )

        if earliest is not None:
            earliest = _aware_datetime(
                earliest,
                "earliest_deadline_at",
            )

        if not isinstance(
            self.overdue,
            bool,
        ):
            raise TypeError(
                "overdue must be boolean."
            )

        seconds = (
            self.seconds_to_deadline
        )

        if seconds is not None:
            if (
                isinstance(seconds, bool)
                or not isinstance(
                    seconds,
                    (int, float),
                )
                or not math.isfinite(
                    seconds
                )
            ):
                raise ValueError(
                    "seconds_to_deadline must "
                    "be finite or None."
                )

            seconds = float(seconds)

        if (
            earliest is None
            and (
                seconds is not None
                or self.overdue
            )
        ):
            raise ValueError(
                "Deadline timing fields are "
                "inconsistent."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                classification_id,
                deadline_extraction_id,
                self.priority.value,
                ",".join(
                    reason.value
                    for reason in reasons
                ),
                assessed_at.isoformat(),
                (
                    earliest.isoformat()
                    if earliest
                    else ""
                ),
                str(self.overdue).lower(),
                (
                    f"{seconds:.6f}"
                    if seconds is not None
                    else ""
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
            "classification_id",
            classification_id,
        )

        object.__setattr__(
            self,
            "deadline_extraction_id",
            deadline_extraction_id,
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "assessed_at",
            assessed_at,
        )

        object.__setattr__(
            self,
            "earliest_deadline_at",
            earliest,
        )

        object.__setattr__(
            self,
            "seconds_to_deadline",
            seconds,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _sha256(payload),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "assessment_id": (
                self.assessment_id
            ),
            "request_id": self.request_id,
            "classification_id": (
                self.classification_id
            ),
            "deadline_extraction_id": (
                self.deadline_extraction_id
            ),
            "priority": self.priority.value,
            "reasons": [
                reason.value
                for reason in self.reasons
            ],
            "assessed_at": (
                self.assessed_at.isoformat()
            ),
            "earliest_deadline_at": (
                self.earliest_deadline_at
                .isoformat()
                if self.earliest_deadline_at
                else None
            ),
            "overdue": self.overdue,
            "seconds_to_deadline": (
                self.seconds_to_deadline
            ),
        }


def assess_correspondence_priority(
    request: CorrespondenceRequest,
    classification: CorrespondenceClassification,
    *,
    assessed_at: datetime,
    parsed: ParsedCorrespondence | None = None,
) -> CorrespondencePriorityAssessment:
    if not isinstance(
        request,
        CorrespondenceRequest,
    ):
        raise TypeError(
            "request must be a "
            "CorrespondenceRequest."
        )

    if not isinstance(
        classification,
        CorrespondenceClassification,
    ):
        raise TypeError(
            "classification must be a "
            "CorrespondenceClassification."
        )

    if (
        classification.request_id
        != request.request_id
    ):
        raise ValueError(
            "Classification request ID "
            "does not match request."
        )

    assessed = _aware_datetime(
        assessed_at,
        "assessed_at",
    )

    if parsed is None:
        parsed = parse_correspondence(
            request
        )

    if (
        not isinstance(
            parsed,
            ParsedCorrespondence,
        )
        or parsed.request_id
        != request.request_id
    ):
        raise ValueError(
            "Parsed correspondence does not "
            "match request."
        )

    deadline_result = (
        extract_correspondence_deadlines(
            request,
            parsed=parsed,
        )
    )

    searchable = "\n".join(
        value
        for value in (
            parsed.normalized_subject,
            parsed.normalized_body,
        )
        if value
    )

    earliest = (
        deadline_result
        .earliest_deadline
    )

    earliest_at = (
        earliest.deadline_at
        if earliest
        else None
    )

    seconds = None
    overdue = False

    if earliest_at is not None:
        seconds = (
            earliest_at
            - assessed
        ).total_seconds()

        overdue = (
            seconds < 0
        )

    reasons = []

    if _matches_any(
        searchable,
        _URGENT_PATTERNS,
    ):
        priority = (
            CorrespondencePriority.URGENT
        )

        reasons.append(
            CorrespondencePriorityReason
            .EXPLICIT_URGENT
        )

    elif overdue:
        priority = (
            CorrespondencePriority.URGENT
        )

        reasons.append(
            CorrespondencePriorityReason
            .OVERDUE_DEADLINE
        )

    elif (
        seconds is not None
        and seconds <= 24 * 3600
    ):
        priority = (
            CorrespondencePriority.URGENT
        )

        reasons.append(
            CorrespondencePriorityReason
            .DUE_WITHIN_24_HOURS
        )

    elif _matches_any(
        searchable,
        _HIGH_PRIORITY_PATTERNS,
    ):
        priority = (
            CorrespondencePriority.HIGH
        )

        reasons.append(
            CorrespondencePriorityReason
            .EXPLICIT_HIGH_PRIORITY
        )

    elif (
        seconds is not None
        and seconds <= 72 * 3600
    ):
        priority = (
            CorrespondencePriority.HIGH
        )

        reasons.append(
            CorrespondencePriorityReason
            .DUE_WITHIN_72_HOURS
        )

    elif (
        classification.intent
        is CorrespondenceIntent.ESCALATION
    ):
        priority = (
            CorrespondencePriority.HIGH
        )

        reasons.append(
            CorrespondencePriorityReason
            .ESCALATION
        )

    elif (
        classification.intent
        in {
            CorrespondenceIntent.INFORMATION,
            CorrespondenceIntent.NOTIFICATION,
        }
        and not classification.requires_action
        and earliest is None
    ):
        priority = (
            CorrespondencePriority.ROUTINE
        )

        reasons.append(
            CorrespondencePriorityReason
            .ROUTINE_INFORMATION
        )

    else:
        priority = (
            CorrespondencePriority.NORMAL
        )

        reasons.append(
            CorrespondencePriorityReason
            .STANDARD_PROCESSING
        )

    return CorrespondencePriorityAssessment(
        request_id=request.request_id,
        classification_id=(
            classification.classification_id
        ),
        deadline_extraction_id=(
            deadline_result.extraction_id
        ),
        priority=priority,
        reasons=tuple(reasons),
        assessed_at=assessed,
        earliest_deadline_at=(
            earliest_at
        ),
        overdue=overdue,
        seconds_to_deadline=seconds,
    )
