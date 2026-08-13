"""Governance verification decision contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


GOVERNANCE_VERIFICATION_VERSION = (
    "govba-governance-verification-v1"
)


class GovernanceDecision(
    str,
    Enum,
):
    """Final governance disposition for an answer candidate."""

    ALLOW = "allow"
    ABSTAIN = "abstain"


class GovernanceReasonCode(
    str,
    Enum,
):
    """Machine-readable reason for a governance decision."""

    VERIFIED = "verified"

    GROUNDING_FAILURE = (
        "grounding_failure"
    )

    UNKNOWN_CITATION = (
        "unknown_citation"
    )

    REJECTED_EVIDENCE = (
        "rejected_evidence"
    )

    SUPERSEDED_EVIDENCE = (
        "superseded_evidence"
    )

    TEMPORAL_CONFLICT = (
        "temporal_conflict"
    )

    TEMPORAL_INSUFFICIENT = (
        "temporal_insufficient"
    )

    NO_ELIGIBLE_EVIDENCE = (
        "no_eligible_evidence"
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


def _normalize_reason_codes(
    values,
) -> tuple[
    GovernanceReasonCode,
    ...
]:
    try:
        reasons = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            "reason_codes must be iterable."
        ) from exc

    if not reasons:
        raise ValueError(
            "At least one governance "
            "reason code is required."
        )

    for reason in reasons:
        if not isinstance(
            reason,
            GovernanceReasonCode,
        ):
            raise TypeError(
                "reason_codes must contain only "
                "GovernanceReasonCode values."
            )

    if (
        len(set(reasons))
        != len(reasons)
    ):
        raise ValueError(
            "reason_codes must not "
            "contain duplicates."
        )

    return tuple(
        sorted(
            reasons,
            key=lambda item: item.value,
        )
    )


def _decision_id(
    *,
    trace_id: str,
    decision: GovernanceDecision,
    reason_codes: tuple[
        GovernanceReasonCode,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            GOVERNANCE_VERIFICATION_VERSION,
            trace_id,
            decision.value,
            ",".join(
                reason.value
                for reason
                in reason_codes
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class GovernanceVerificationResult:
    """Privacy-safe auditable governance decision."""

    trace_id: str
    decision: GovernanceDecision

    reason_codes: tuple[
        GovernanceReasonCode,
        ...
    ]

    evidence_count: int = 0
    eligible_evidence_count: int = 0

    version: str = (
        GOVERNANCE_VERIFICATION_VERSION
    )

    decision_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.decision,
            GovernanceDecision,
        ):
            raise TypeError(
                "decision must be a "
                "GovernanceDecision."
            )

        reasons = (
            _normalize_reason_codes(
                self.reason_codes
            )
        )

        for (
            field_name,
            value,
        ) in (
            (
                "evidence_count",
                self.evidence_count,
            ),
            (
                "eligible_evidence_count",
                self.eligible_evidence_count,
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
                    f"{field_name} must be "
                    "a non-negative integer."
                )

        if (
            self.eligible_evidence_count
            > self.evidence_count
        ):
            raise ValueError(
                "eligible_evidence_count cannot "
                "exceed evidence_count."
            )

        if (
            self.decision
            is GovernanceDecision.ALLOW
        ):
            if (
                reasons
                != (
                    GovernanceReasonCode
                    .VERIFIED,
                )
            ):
                raise ValueError(
                    "ALLOW requires VERIFIED as "
                    "the only reason code."
                )

            if (
                self.eligible_evidence_count
                < 1
            ):
                raise ValueError(
                    "ALLOW requires at least one "
                    "eligible evidence item."
                )

        if (
            self.decision
            is GovernanceDecision.ABSTAIN
            and GovernanceReasonCode.VERIFIED
            in reasons
        ):
            raise ValueError(
                "ABSTAIN cannot include VERIFIED."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "reason_codes",
            reasons,
        )

        object.__setattr__(
            self,
            "decision_id",
            _decision_id(
                trace_id=trace_id,
                decision=self.decision,
                reason_codes=reasons,
            ),
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            is GovernanceDecision.ALLOW
        )

    @property
    def should_abstain(
        self,
    ) -> bool:
        return not self.allowed

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without prompts, claims, or evidence text."""

        return {
            "version": (
                self.version
            ),
            "decision_id": (
                self.decision_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "decision": (
                self.decision.value
            ),
            "reason_codes": [
                reason.value
                for reason
                in self.reason_codes
            ],
            "evidence_count": (
                self.evidence_count
            ),
            "eligible_evidence_count": (
                self.eligible_evidence_count
            ),
            "allowed": (
                self.allowed
            ),
            "should_abstain": (
                self.should_abstain
            ),
        }
