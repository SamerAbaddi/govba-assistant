"""Temporal policy-state contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from govba.rag.models import (
    AuthoritativeSource,
)


TEMPORAL_POLICY_SCHEMA_VERSION = (
    "govba-temporal-policy-v1"
)


class TemporalPolicyState(
    str,
    Enum,
):
    """Governance-facing temporal state of authoritative evidence."""

    CURRENT = "current"
    SUPERSEDED = "superseded"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


class TemporalReasonCode(
    str,
    Enum,
):
    """Machine-readable explanation for temporal assessment."""

    WITHIN_EFFECTIVE_WINDOW = (
        "within_effective_window"
    )
    BEFORE_EFFECTIVE_DATE = (
        "before_effective_date"
    )
    AFTER_EFFECTIVE_DATE = (
        "after_effective_date"
    )
    EXPLICITLY_SUPERSEDED = (
        "explicitly_superseded"
    )
    SUPERSESSION_DATE_UNKNOWN = (
        "supersession_date_unknown"
    )
    CONFLICT_DETECTED = (
        "conflict_detected"
    )
    MISSING_TEMPORAL_METADATA = (
        "missing_temporal_metadata"
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


def _validate_date(
    value: date,
    field_name: str,
) -> date:
    if (
        isinstance(
            value,
            datetime,
        )
        or not isinstance(
            value,
            date,
        )
    ):
        raise TypeError(
            f"{field_name} must be a date."
        )

    return value


def _optional_date(
    value: date | None,
    field_name: str,
) -> date | None:
    if value is None:
        return None

    return _validate_date(
        value,
        field_name,
    )


def _normalize_ids(
    values,
    field_name: str,
) -> tuple[str, ...]:
    try:
        raw_values = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    normalized = tuple(
        _required_text(
            value,
            field_name,
        )
        for value in raw_values
    )

    if (
        len(set(normalized))
        != len(normalized)
    ):
        raise ValueError(
            f"{field_name} must not contain "
            "duplicate document IDs."
        )

    return normalized


def _assessment_id(
    *,
    document_id: str,
    as_of_date: date,
    state: TemporalPolicyState,
    reason_codes: tuple[
        TemporalReasonCode,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            TEMPORAL_POLICY_SCHEMA_VERSION,
            document_id,
            as_of_date.isoformat(),
            state.value,
            ",".join(
                reason.value
                for reason in reason_codes
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class TemporalPolicyAssessment:
    """Temporal governance assessment for one authoritative source."""

    document_id: str
    as_of_date: date
    state: TemporalPolicyState

    effective_from: date | None = None
    effective_until: date | None = None

    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()

    reason_codes: tuple[
        TemporalReasonCode,
        ...
    ] = ()

    schema_version: str = (
        TEMPORAL_POLICY_SCHEMA_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        document_id = _required_text(
            self.document_id,
            "document_id",
        )

        as_of_date = _validate_date(
            self.as_of_date,
            "as_of_date",
        )

        if not isinstance(
            self.state,
            TemporalPolicyState,
        ):
            raise TypeError(
                "state must be a "
                "TemporalPolicyState."
            )

        effective_from = _optional_date(
            self.effective_from,
            "effective_from",
        )

        effective_until = _optional_date(
            self.effective_until,
            "effective_until",
        )

        if (
            effective_from is not None
            and effective_until is not None
            and effective_until < effective_from
        ):
            raise ValueError(
                "effective_until must not "
                "precede effective_from."
            )

        supersedes = _normalize_ids(
            self.supersedes,
            "supersedes",
        )

        superseded_by = _normalize_ids(
            self.superseded_by,
            "superseded_by",
        )

        if document_id in supersedes:
            raise ValueError(
                "A document cannot supersede itself."
            )

        if document_id in superseded_by:
            raise ValueError(
                "A document cannot be superseded "
                "by itself."
            )

        try:
            reason_codes = tuple(
                self.reason_codes
            )
        except TypeError as exc:
            raise TypeError(
                "reason_codes must be iterable."
            ) from exc

        for reason in reason_codes:
            if not isinstance(
                reason,
                TemporalReasonCode,
            ):
                raise TypeError(
                    "reason_codes must contain only "
                    "TemporalReasonCode values."
                )

        if (
            len(set(reason_codes))
            != len(reason_codes)
        ):
            raise ValueError(
                "reason_codes must not "
                "contain duplicates."
            )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        object.__setattr__(
            self,
            "as_of_date",
            as_of_date,
        )

        object.__setattr__(
            self,
            "effective_from",
            effective_from,
        )

        object.__setattr__(
            self,
            "effective_until",
            effective_until,
        )

        object.__setattr__(
            self,
            "supersedes",
            supersedes,
        )

        object.__setattr__(
            self,
            "superseded_by",
            superseded_by,
        )

        object.__setattr__(
            self,
            "reason_codes",
            reason_codes,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _assessment_id(
                document_id=document_id,
                as_of_date=as_of_date,
                state=self.state,
                reason_codes=reason_codes,
            ),
        )

    @classmethod
    def from_source(
        cls,
        source: AuthoritativeSource,
        *,
        as_of_date: date,
        state: TemporalPolicyState,
        reason_codes: tuple[
            TemporalReasonCode,
            ...
        ] = (),
    ) -> "TemporalPolicyAssessment":
        """Build an assessment contract from source metadata."""

        if not isinstance(
            source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "source must be an "
                "AuthoritativeSource."
            )

        return cls(
            document_id=source.document_id,
            as_of_date=as_of_date,
            state=state,
            effective_from=(
                source.effective_from
            ),
            effective_until=(
                source.effective_until
            ),
            supersedes=(
                source.supersedes
            ),
            superseded_by=(
                source.superseded_by
            ),
            reason_codes=(
                reason_codes
            ),
        )

    @property
    def is_current(self) -> bool:
        return (
            self.state
            is TemporalPolicyState.CURRENT
        )

    @property
    def is_superseded(self) -> bool:
        return (
            self.state
            is TemporalPolicyState.SUPERSEDED
        )

    @property
    def should_abstain(self) -> bool:
        return self.state in {
            TemporalPolicyState.CONFLICTING,
            TemporalPolicyState.INSUFFICIENT,
        }

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "assessment_id": (
                self.assessment_id
            ),
            "document_id": (
                self.document_id
            ),
            "as_of_date": (
                self.as_of_date.isoformat()
            ),
            "state": (
                self.state.value
            ),
            "effective_from": (
                self.effective_from.isoformat()
                if self.effective_from
                else None
            ),
            "effective_until": (
                self.effective_until.isoformat()
                if self.effective_until
                else None
            ),
            "supersedes": list(
                self.supersedes
            ),
            "superseded_by": list(
                self.superseded_by
            ),
            "reason_codes": [
                reason.value
                for reason
                in self.reason_codes
            ],
            "is_current": (
                self.is_current
            ),
            "should_abstain": (
                self.should_abstain
            ),
        }
