"""Provider-independent security contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


SECURITY_CONTRACT_VERSION = (
    "govba-security-contract-v1"
)


class SecuritySurface(
    str,
    Enum,
):
    """Area of the GovBA pipeline where a security finding occurred."""

    USER_QUERY = "user_query"
    DOCUMENT = "document"
    RETRIEVAL = "retrieval"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    SOURCE = "source"


class SecuritySeverity(
    str,
    Enum,
):
    """Security finding severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityDecision(
    str,
    Enum,
):
    """Required security disposition."""

    ALLOW = "allow"
    REDACT = "redact"
    REVIEW = "review"
    BLOCK = "block"


class SecurityIssueCode(
    str,
    Enum,
):
    """Stable machine-readable security issue taxonomy."""

    PII_DETECTED = "pii_detected"

    SENSITIVE_PII_DETECTED = (
        "sensitive_pii_detected"
    )

    SECRET_DETECTED = (
        "secret_detected"
    )

    PROMPT_INJECTION_DETECTED = (
        "prompt_injection_detected"
    )

    UNTRUSTED_SOURCE = (
        "untrusted_source"
    )

    DISALLOWED_DOMAIN = (
        "disallowed_domain"
    )

    UNSAFE_OUTPUT = (
        "unsafe_output"
    )

    SECURITY_POLICY_VIOLATION = (
        "security_policy_violation"
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


def _finding_id(
    *,
    code: SecurityIssueCode,
    surface: SecuritySurface,
    severity: SecuritySeverity,
    decision: SecurityDecision,
    occurrence_count: int,
    reference_id: str,
) -> str:
    payload = "\x1f".join(
        (
            SECURITY_CONTRACT_VERSION,
            code.value,
            surface.value,
            severity.value,
            decision.value,
            str(
                occurrence_count
            ),
            reference_id,
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class SecurityFinding:
    """Privacy-safe description of one security finding.

    Raw query, document, model, or PII content must not be stored here.
    """

    code: SecurityIssueCode
    surface: SecuritySurface
    severity: SecuritySeverity
    decision: SecurityDecision

    occurrence_count: int = 1

    reference_id: str = ""

    version: str = (
        SECURITY_CONTRACT_VERSION
    )

    finding_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.code,
            SecurityIssueCode,
        ):
            raise TypeError(
                "code must be a "
                "SecurityIssueCode."
            )

        if not isinstance(
            self.surface,
            SecuritySurface,
        ):
            raise TypeError(
                "surface must be a "
                "SecuritySurface."
            )

        if not isinstance(
            self.severity,
            SecuritySeverity,
        ):
            raise TypeError(
                "severity must be a "
                "SecuritySeverity."
            )

        if not isinstance(
            self.decision,
            SecurityDecision,
        ):
            raise TypeError(
                "decision must be a "
                "SecurityDecision."
            )

        if (
            isinstance(
                self.occurrence_count,
                bool,
            )
            or not isinstance(
                self.occurrence_count,
                int,
            )
            or self.occurrence_count < 1
        ):
            raise ValueError(
                "occurrence_count must be "
                "a positive integer."
            )

        reference_id = _optional_text(
            self.reference_id,
            "reference_id",
        )

        object.__setattr__(
            self,
            "reference_id",
            reference_id,
        )

        object.__setattr__(
            self,
            "finding_id",
            _finding_id(
                code=self.code,
                surface=self.surface,
                severity=self.severity,
                decision=self.decision,
                occurrence_count=(
                    self.occurrence_count
                ),
                reference_id=(
                    reference_id
                ),
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize metadata without sensitive content."""

        return {
            "version": (
                self.version
            ),
            "finding_id": (
                self.finding_id
            ),
            "code": (
                self.code.value
            ),
            "surface": (
                self.surface.value
            ),
            "severity": (
                self.severity.value
            ),
            "decision": (
                self.decision.value
            ),
            "occurrence_count": (
                self.occurrence_count
            ),
            "reference_id": (
                self.reference_id
                or None
            ),
        }


_DECISION_PRIORITY = {
    SecurityDecision.ALLOW: 0,
    SecurityDecision.REDACT: 1,
    SecurityDecision.REVIEW: 2,
    SecurityDecision.BLOCK: 3,
}


def _resolve_decision(
    findings: tuple[
        SecurityFinding,
        ...
    ],
) -> SecurityDecision:
    if not findings:
        return (
            SecurityDecision.ALLOW
        )

    return max(
        (
            finding.decision
            for finding
            in findings
        ),
        key=lambda decision: (
            _DECISION_PRIORITY[
                decision
            ]
        ),
    )


def _assessment_id(
    *,
    trace_id: str,
    decision: SecurityDecision,
    finding_ids: tuple[
        str,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            SECURITY_CONTRACT_VERSION,
            trace_id,
            decision.value,
            ",".join(
                sorted(
                    finding_ids
                )
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class SecurityAssessment:
    """Aggregated fail-safe security assessment."""

    trace_id: str

    findings: tuple[
        SecurityFinding,
        ...
    ] = ()

    version: str = (
        SECURITY_CONTRACT_VERSION
    )

    decision: SecurityDecision = field(
        init=False
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        try:
            findings = tuple(
                self.findings
            )
        except TypeError as exc:
            raise TypeError(
                "findings must be iterable."
            ) from exc

        for finding in findings:
            if not isinstance(
                finding,
                SecurityFinding,
            ):
                raise TypeError(
                    "findings must contain only "
                    "SecurityFinding values."
                )

        finding_ids = tuple(
            finding.finding_id
            for finding
            in findings
        )

        if (
            len(set(finding_ids))
            != len(finding_ids)
        ):
            raise ValueError(
                "findings must not contain "
                "duplicates."
            )

        decision = _resolve_decision(
            findings
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "findings",
            findings,
        )

        object.__setattr__(
            self,
            "decision",
            decision,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _assessment_id(
                trace_id=trace_id,
                decision=decision,
                finding_ids=(
                    finding_ids
                ),
            ),
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.ALLOW
        )

    @property
    def requires_redaction(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.REDACT
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.REVIEW
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.BLOCK
        )

    @property
    def finding_count(
        self,
    ) -> int:
        return len(
            self.findings
        )

    @property
    def critical_finding_count(
        self,
    ) -> int:
        return sum(
            finding.severity
            is SecuritySeverity.CRITICAL
            for finding
            in self.findings
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize privacy-safe security metadata."""

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
            "decision": (
                self.decision.value
            ),
            "allowed": (
                self.allowed
            ),
            "requires_redaction": (
                self.requires_redaction
            ),
            "requires_review": (
                self.requires_review
            ),
            "should_block": (
                self.should_block
            ),
            "finding_count": (
                self.finding_count
            ),
            "critical_finding_count": (
                self.critical_finding_count
            ),
            "findings": [
                finding.to_dict()
                for finding
                in self.findings
            ],
        }


def assess_security(
    *,
    trace_id: str,
    findings: Iterable[
        SecurityFinding
    ] = (),
) -> SecurityAssessment:
    """Build a deterministic security assessment."""

    try:
        finding_values = tuple(
            findings
        )
    except TypeError as exc:
        raise TypeError(
            "findings must be iterable."
        ) from exc

    return SecurityAssessment(
        trace_id=trace_id,
        findings=finding_values,
    )
