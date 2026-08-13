"""Freshness and provenance controls for GovBA-GAR web evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import (
    datetime,
    timezone,
)
from enum import Enum

from govba.web.contract import (
    OfficialWebResult,
)
from govba.web.evidence_bridge import (
    WebEvidenceBridgeResult,
)


WEB_FRESHNESS_VERSION = (
    "govba-web-freshness-v1"
)


class WebFreshnessState(
    str,
    Enum,
):
    """Age state of controlled web evidence."""

    FRESH = "fresh"
    STALE = "stale"
    FUTURE = "future"


class WebProvenanceState(
    str,
    Enum,
):
    """Whether web-result provenance matches its evidence bridge."""

    VERIFIED = "verified"
    INVALID = "invalid"


class WebEvidenceControlDecision(
    str,
    Enum,
):
    """Disposition after freshness and provenance validation."""

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class WebEvidenceControlIssueCode(
    str,
    Enum,
):
    """Stable issue taxonomy."""

    SOURCE_URL_MISMATCH = (
        "source_url_mismatch"
    )

    CONTENT_HASH_MISMATCH = (
        "content_hash_mismatch"
    )

    RETRIEVAL_TIME_MISMATCH = (
        "retrieval_time_mismatch"
    )

    DOMAIN_MISMATCH = (
        "domain_mismatch"
    )

    RETRIEVAL_FROM_FUTURE = (
        "retrieval_from_future"
    )

    STALE_RETRIEVAL = (
        "stale_retrieval"
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


def _require_aware_datetime(
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


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class WebEvidenceControlPolicy:
    """Deployment policy for controlled-web freshness."""

    max_age_hours: int

    future_tolerance_minutes: int = 5

    version: str = (
        WEB_FRESHNESS_VERSION
    )

    policy_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.max_age_hours,
                bool,
            )
            or not isinstance(
                self.max_age_hours,
                int,
            )
            or self.max_age_hours < 1
        ):
            raise ValueError(
                "max_age_hours must be "
                "a positive integer."
            )

        if (
            isinstance(
                self.future_tolerance_minutes,
                bool,
            )
            or not isinstance(
                self.future_tolerance_minutes,
                int,
            )
            or self.future_tolerance_minutes < 0
        ):
            raise ValueError(
                "future_tolerance_minutes must be "
                "a non-negative integer."
            )

        payload = "\x1f".join(
            (
                self.version,
                str(
                    self.max_age_hours
                ),
                str(
                    self.future_tolerance_minutes
                ),
            )
        )

        object.__setattr__(
            self,
            "policy_id",
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
            "policy_id": (
                self.policy_id
            ),
            "max_age_hours": (
                self.max_age_hours
            ),
            "future_tolerance_minutes": (
                self.future_tolerance_minutes
            ),
        }


@dataclass(frozen=True)
class WebEvidenceControlIssue:
    """One privacy-safe freshness/provenance finding."""

    code: WebEvidenceControlIssueCode

    decision: WebEvidenceControlDecision

    def __post_init__(self) -> None:
        if not isinstance(
            self.code,
            WebEvidenceControlIssueCode,
        ):
            raise TypeError(
                "code must be a "
                "WebEvidenceControlIssueCode."
            )

        if not isinstance(
            self.decision,
            WebEvidenceControlDecision,
        ):
            raise TypeError(
                "decision must be a "
                "WebEvidenceControlDecision."
            )

        if (
            self.decision
            is WebEvidenceControlDecision.ALLOW
        ):
            raise ValueError(
                "Issues cannot carry "
                "an ALLOW decision."
            )

    def to_dict(
        self,
    ) -> dict[str, str]:
        return {
            "code": (
                self.code.value
            ),
            "decision": (
                self.decision.value
            ),
        }


_DECISION_PRIORITY = {
    WebEvidenceControlDecision.ALLOW: 0,
    WebEvidenceControlDecision.REVIEW: 1,
    WebEvidenceControlDecision.BLOCK: 2,
}


_PROVENANCE_CODES = {
    WebEvidenceControlIssueCode
    .SOURCE_URL_MISMATCH,

    WebEvidenceControlIssueCode
    .CONTENT_HASH_MISMATCH,

    WebEvidenceControlIssueCode
    .RETRIEVAL_TIME_MISMATCH,

    WebEvidenceControlIssueCode
    .DOMAIN_MISMATCH,
}


@dataclass(frozen=True)
class WebEvidenceControlAssessment:
    """Freshness and provenance assessment without web content."""

    trace_id: str

    result_id: str

    bridge_id: str

    document_id: str

    domain: str

    retrieved_at: datetime

    assessed_at: datetime

    age_seconds: float

    freshness_state: (
        WebFreshnessState
    )

    provenance_state: (
        WebProvenanceState
    )

    decision: (
        WebEvidenceControlDecision
    )

    issues: tuple[
        WebEvidenceControlIssue,
        ...
    ]

    policy_id: str

    version: str = (
        WEB_FRESHNESS_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        result_id = _required_text(
            self.result_id,
            "result_id",
        )

        bridge_id = _required_text(
            self.bridge_id,
            "bridge_id",
        )

        document_id = _required_text(
            self.document_id,
            "document_id",
        )

        domain = _required_text(
            self.domain,
            "domain",
        ).lower()

        retrieved_at = (
            _require_aware_datetime(
                self.retrieved_at,
                "retrieved_at",
            )
        )

        assessed_at = (
            _require_aware_datetime(
                self.assessed_at,
                "assessed_at",
            )
        )

        if (
            isinstance(
                self.age_seconds,
                bool,
            )
            or not isinstance(
                self.age_seconds,
                (int, float),
            )
            or self.age_seconds < 0
        ):
            raise ValueError(
                "age_seconds must be "
                "non-negative."
            )

        if not isinstance(
            self.freshness_state,
            WebFreshnessState,
        ):
            raise TypeError(
                "freshness_state must be "
                "a WebFreshnessState."
            )

        if not isinstance(
            self.provenance_state,
            WebProvenanceState,
        ):
            raise TypeError(
                "provenance_state must be "
                "a WebProvenanceState."
            )

        if not isinstance(
            self.decision,
            WebEvidenceControlDecision,
        ):
            raise TypeError(
                "decision must be a "
                "WebEvidenceControlDecision."
            )

        try:
            issues = tuple(
                self.issues
            )
        except TypeError as exc:
            raise TypeError(
                "issues must be iterable."
            ) from exc

        for issue in issues:
            if not isinstance(
                issue,
                WebEvidenceControlIssue,
            ):
                raise TypeError(
                    "issues must contain only "
                    "WebEvidenceControlIssue values."
                )

        issue_codes = tuple(
            issue.code
            for issue in issues
        )

        if (
            len(set(issue_codes))
            != len(issue_codes)
        ):
            raise ValueError(
                "Issue codes must be unique."
            )

        policy_id = _required_text(
            self.policy_id,
            "policy_id",
        ).lower()

        if (
            len(policy_id) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in policy_id
            )
        ):
            raise ValueError(
                "policy_id must be "
                "a SHA-256 digest."
            )

        payload = "\x1f".join(
            (
                self.version,
                trace_id,
                result_id,
                bridge_id,
                document_id,
                domain,
                retrieved_at.isoformat(),
                assessed_at.isoformat(),
                self.freshness_state.value,
                self.provenance_state.value,
                self.decision.value,
                ",".join(
                    sorted(
                        code.value
                        for code in issue_codes
                    )
                ),
                policy_id,
            )
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "result_id",
            result_id,
        )

        object.__setattr__(
            self,
            "bridge_id",
            bridge_id,
        )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        object.__setattr__(
            self,
            "domain",
            domain,
        )

        object.__setattr__(
            self,
            "retrieved_at",
            retrieved_at,
        )

        object.__setattr__(
            self,
            "assessed_at",
            assessed_at,
        )

        object.__setattr__(
            self,
            "age_seconds",
            float(
                self.age_seconds
            ),
        )

        object.__setattr__(
            self,
            "issues",
            issues,
        )

        object.__setattr__(
            self,
            "policy_id",
            policy_id,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _sha256(
                payload
            ),
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            is WebEvidenceControlDecision.ALLOW
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is WebEvidenceControlDecision.REVIEW
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return (
            self.decision
            is WebEvidenceControlDecision.BLOCK
        )

    @property
    def provenance_verified(
        self,
    ) -> bool:
        return (
            self.provenance_state
            is WebProvenanceState.VERIFIED
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without query or evidence text."""

        return {
            "version": (
                self.version
            ),
            "assessment_id": (
                self.assessment_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "result_id": (
                self.result_id
            ),
            "bridge_id": (
                self.bridge_id
            ),
            "document_id": (
                self.document_id
            ),
            "domain": (
                self.domain
            ),
            "retrieved_at": (
                self.retrieved_at
                .isoformat()
            ),
            "assessed_at": (
                self.assessed_at
                .isoformat()
            ),
            "age_seconds": (
                self.age_seconds
            ),
            "freshness_state": (
                self.freshness_state.value
            ),
            "provenance_state": (
                self.provenance_state.value
            ),
            "decision": (
                self.decision.value
            ),
            "allowed": (
                self.allowed
            ),
            "requires_review": (
                self.requires_review
            ),
            "should_block": (
                self.should_block
            ),
            "policy_id": (
                self.policy_id
            ),
            "issues": [
                issue.to_dict()
                for issue
                in self.issues
            ],
        }


def assess_web_evidence_controls(
    result: OfficialWebResult,
    *,
    bridge: WebEvidenceBridgeResult,
    policy: WebEvidenceControlPolicy,
    trace_id: str,
    assessed_at: datetime | None = None,
) -> WebEvidenceControlAssessment:
    """Verify provenance and freshness of controlled-web evidence."""

    if not isinstance(
        result,
        OfficialWebResult,
    ):
        raise TypeError(
            "result must be an "
            "OfficialWebResult."
        )

    if not isinstance(
        bridge,
        WebEvidenceBridgeResult,
    ):
        raise TypeError(
            "bridge must be a "
            "WebEvidenceBridgeResult."
        )

    if not isinstance(
        policy,
        WebEvidenceControlPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "WebEvidenceControlPolicy."
        )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if (
        bridge.trace_id
        != trace
    ):
        raise ValueError(
            "Bridge trace ID must match "
            "assessment trace ID."
        )

    if assessed_at is None:
        assessed_at = datetime.now(
            timezone.utc
        )

    assessed_at = (
        _require_aware_datetime(
            assessed_at,
            "assessed_at",
        )
    )

    issues = []

    if (
        bridge.source.official_source_url
        != result.url
    ):
        issues.append(
            WebEvidenceControlIssue(
                code=(
                    WebEvidenceControlIssueCode
                    .SOURCE_URL_MISMATCH
                ),
                decision=(
                    WebEvidenceControlDecision
                    .BLOCK
                ),
            )
        )

    if (
        bridge.source.content_hash
        != result.content_hash
    ):
        issues.append(
            WebEvidenceControlIssue(
                code=(
                    WebEvidenceControlIssueCode
                    .CONTENT_HASH_MISMATCH
                ),
                decision=(
                    WebEvidenceControlDecision
                    .BLOCK
                ),
            )
        )

    if (
        bridge.source.ingested_at
        != result.retrieved_at
    ):
        issues.append(
            WebEvidenceControlIssue(
                code=(
                    WebEvidenceControlIssueCode
                    .RETRIEVAL_TIME_MISMATCH
                ),
                decision=(
                    WebEvidenceControlDecision
                    .BLOCK
                ),
            )
        )

    if (
        bridge.source_assessment.domain
        != result.domain
    ):
        issues.append(
            WebEvidenceControlIssue(
                code=(
                    WebEvidenceControlIssueCode
                    .DOMAIN_MISMATCH
                ),
                decision=(
                    WebEvidenceControlDecision
                    .BLOCK
                ),
            )
        )

    future_seconds = (
        result.retrieved_at
        - assessed_at
    ).total_seconds()

    tolerance_seconds = (
        policy.future_tolerance_minutes
        * 60
    )

    if (
        future_seconds
        > tolerance_seconds
    ):
        freshness_state = (
            WebFreshnessState.FUTURE
        )

        age_seconds = 0.0

        issues.append(
            WebEvidenceControlIssue(
                code=(
                    WebEvidenceControlIssueCode
                    .RETRIEVAL_FROM_FUTURE
                ),
                decision=(
                    WebEvidenceControlDecision
                    .BLOCK
                ),
            )
        )

    else:
        age_seconds = max(
            0.0,
            (
                assessed_at
                - result.retrieved_at
            ).total_seconds(),
        )

        max_age_seconds = (
            policy.max_age_hours
            * 3600
        )

        if (
            age_seconds
            > max_age_seconds
        ):
            freshness_state = (
                WebFreshnessState.STALE
            )

            issues.append(
                WebEvidenceControlIssue(
                    code=(
                        WebEvidenceControlIssueCode
                        .STALE_RETRIEVAL
                    ),
                    decision=(
                        WebEvidenceControlDecision
                        .REVIEW
                    ),
                )
            )

        else:
            freshness_state = (
                WebFreshnessState.FRESH
            )

    provenance_state = (
        WebProvenanceState.INVALID
        if any(
            issue.code
            in _PROVENANCE_CODES
            for issue in issues
        )
        else WebProvenanceState.VERIFIED
    )

    if issues:
        decision = max(
            (
                issue.decision
                for issue in issues
            ),
            key=lambda value: (
                _DECISION_PRIORITY[
                    value
                ]
            ),
        )
    else:
        decision = (
            WebEvidenceControlDecision.ALLOW
        )

    return WebEvidenceControlAssessment(
        trace_id=trace,
        result_id=(
            result.result_id
        ),
        bridge_id=(
            bridge.bridge_id
        ),
        document_id=(
            bridge.source.document_id
        ),
        domain=(
            result.domain
        ),
        retrieved_at=(
            result.retrieved_at
        ),
        assessed_at=(
            assessed_at
        ),
        age_seconds=(
            age_seconds
        ),
        freshness_state=(
            freshness_state
        ),
        provenance_state=(
            provenance_state
        ),
        decision=decision,
        issues=tuple(
            issues
        ),
        policy_id=(
            policy.policy_id
        ),
    )
