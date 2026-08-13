"""Provider-independent correspondence intelligence contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from govba.rag.models import SourceLanguage


CORRESPONDENCE_CONTRACT_VERSION = (
    "govba-correspondence-contract-v1"
)


class CorrespondenceDirection(
    str,
    Enum,
):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class CorrespondenceChannel(
    str,
    Enum,
):
    EMAIL = "email"
    LETTER = "letter"
    MEMORANDUM = "memorandum"
    PORTAL = "portal"
    OTHER = "other"


class CorrespondenceIntent(
    str,
    Enum,
):
    INFORMATION = "information"
    ACTION_REQUIRED = "action_required"
    REQUEST = "request"
    APPROVAL = "approval"
    DECISION = "decision"
    ESCALATION = "escalation"
    COMPLAINT = "complaint"
    NOTIFICATION = "notification"
    UNKNOWN = "unknown"


class CorrespondencePriority(
    str,
    Enum,
):
    ROUTINE = "routine"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


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


def _optional_text(
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

    return value.strip()


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _aware_datetime(
    value: datetime,
    field_name: str,
) -> datetime:
    if not isinstance(
        value,
        datetime,
    ):
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


@dataclass(frozen=True)
class CorrespondenceRequest:
    """One raw correspondence item awaiting intelligence processing."""

    body_text: str

    language: SourceLanguage

    direction: CorrespondenceDirection

    channel: CorrespondenceChannel

    received_at: datetime

    trace_id: str

    subject_text: str = ""

    source_reference: str = ""

    version: str = (
        CORRESPONDENCE_CONTRACT_VERSION
    )

    body_sha256: str = field(
        init=False
    )

    subject_sha256: str = field(
        init=False
    )

    request_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        body = _required_text(
            self.body_text,
            "body_text",
        )

        subject = _optional_text(
            self.subject_text,
            "subject_text",
        )

        source_reference = (
            _optional_text(
                self.source_reference,
                "source_reference",
            )
        )

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.language,
            SourceLanguage,
        ):
            raise TypeError(
                "language must be a "
                "SourceLanguage."
            )

        if not isinstance(
            self.direction,
            CorrespondenceDirection,
        ):
            raise TypeError(
                "direction must be a "
                "CorrespondenceDirection."
            )

        if not isinstance(
            self.channel,
            CorrespondenceChannel,
        ):
            raise TypeError(
                "channel must be a "
                "CorrespondenceChannel."
            )

        received_at = _aware_datetime(
            self.received_at,
            "received_at",
        )

        body_hash = _sha256(
            body
        )

        subject_hash = (
            _sha256(
                subject
            )
            if subject
            else ""
        )

        identity_payload = "\x1f".join(
            (
                self.version,
                trace_id,
                body_hash,
                subject_hash,
                self.language.value,
                self.direction.value,
                self.channel.value,
                received_at.isoformat(),
                source_reference,
            )
        )

        request_id = _sha256(
            identity_payload
        )

        object.__setattr__(
            self,
            "body_text",
            body,
        )

        object.__setattr__(
            self,
            "subject_text",
            subject,
        )

        object.__setattr__(
            self,
            "source_reference",
            source_reference,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "received_at",
            received_at,
        )

        object.__setattr__(
            self,
            "body_sha256",
            body_hash,
        )

        object.__setattr__(
            self,
            "subject_sha256",
            subject_hash,
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Privacy-safe serialization by default."""

        if not isinstance(
            include_content,
            bool,
        ):
            raise TypeError(
                "include_content must be boolean."
            )

        data: dict[
            str,
            Any,
        ] = {
            "version": (
                self.version
            ),
            "request_id": (
                self.request_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "language": (
                self.language.value
            ),
            "direction": (
                self.direction.value
            ),
            "channel": (
                self.channel.value
            ),
            "received_at": (
                self.received_at
                .isoformat()
            ),
            "body_sha256": (
                self.body_sha256
            ),
            "subject_sha256": (
                self.subject_sha256
                or None
            ),
            "has_subject": bool(
                self.subject_text
            ),
            "has_source_reference": bool(
                self.source_reference
            ),
        }

        if include_content:
            data[
                "body_text"
            ] = self.body_text

            data[
                "subject_text"
            ] = self.subject_text

            data[
                "source_reference"
            ] = self.source_reference

        return data


@dataclass(frozen=True)
class CorrespondenceClassification:
    """Structured correspondence classification result."""

    request_id: str

    intent: CorrespondenceIntent

    priority: CorrespondencePriority

    requires_action: bool

    confidence: float

    reason_codes: tuple[
        str,
        ...
    ] = ()

    version: str = (
        CORRESPONDENCE_CONTRACT_VERSION
    )

    classification_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        if not isinstance(
            self.intent,
            CorrespondenceIntent,
        ):
            raise TypeError(
                "intent must be a "
                "CorrespondenceIntent."
            )

        if not isinstance(
            self.priority,
            CorrespondencePriority,
        ):
            raise TypeError(
                "priority must be a "
                "CorrespondencePriority."
            )

        if not isinstance(
            self.requires_action,
            bool,
        ):
            raise TypeError(
                "requires_action must be boolean."
            )

        if (
            isinstance(
                self.confidence,
                bool,
            )
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

        try:
            reason_codes = tuple(
                _required_text(
                    value,
                    "reason_code",
                )
                for value
                in self.reason_codes
            )
        except TypeError as exc:
            raise TypeError(
                "reason_codes must be iterable."
            ) from exc

        if (
            len(
                set(
                    reason_codes
                )
            )
            != len(
                reason_codes
            )
        ):
            raise ValueError(
                "reason_codes must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                self.intent.value,
                self.priority.value,
                str(
                    self.requires_action
                ).lower(),
                f"{float(self.confidence):.12f}",
                ",".join(
                    reason_codes
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
            "confidence",
            float(
                self.confidence
            ),
        )

        object.__setattr__(
            self,
            "reason_codes",
            reason_codes,
        )

        object.__setattr__(
            self,
            "classification_id",
            _sha256(
                payload
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "classification_id": (
                self.classification_id
            ),
            "request_id": (
                self.request_id
            ),
            "intent": (
                self.intent.value
            ),
            "priority": (
                self.priority.value
            ),
            "requires_action": (
                self.requires_action
            ),
            "confidence": (
                self.confidence
            ),
            "reason_codes": list(
                self.reason_codes
            ),
        }


@dataclass(frozen=True)
class CorrespondenceIntelligenceResult:
    """Top-level provider-independent correspondence result."""

    request_id: str

    trace_id: str

    classification: (
        CorrespondenceClassification
    )

    version: str = (
        CORRESPONDENCE_CONTRACT_VERSION
    )

    result_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.classification,
            CorrespondenceClassification,
        ):
            raise TypeError(
                "classification must be a "
                "CorrespondenceClassification."
            )

        if (
            self.classification.request_id
            != request_id
        ):
            raise ValueError(
                "Classification request ID "
                "must match result request ID."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                trace_id,
                self.classification
                .classification_id,
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "result_id",
            _sha256(
                payload
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "result_id": (
                self.result_id
            ),
            "request_id": (
                self.request_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "classification": (
                self.classification
                .to_dict()
            ),
        }
