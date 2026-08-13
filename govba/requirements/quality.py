"""Deterministic requirement quality and ambiguity intelligence for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

from govba.requirements.contract import (
    RequirementItem,
    RequirementType,
)
from govba.requirements.extraction import (
    RequirementExtractionResult,
)


REQUIREMENT_QUALITY_VERSION = (
    "govba-requirement-quality-v1"
)

REQUIREMENT_QUALITY_ALGORITHM = (
    "deterministic-bilingual-requirement-quality-v1"
)


class RequirementQualitySeverity(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RequirementQualityDecision(
    str,
    Enum,
):
    PASS = "pass"
    REVIEW = "review"
    REWRITE = "rewrite"


class RequirementQualityIssueCode(
    str,
    Enum,
):
    TBD_PLACEHOLDER = "tbd_placeholder"
    AMBIGUOUS_TERM = "ambiguous_term"
    SUBJECTIVE_TERM = "subjective_term"
    MULTIPLE_OBLIGATIONS = "multiple_obligations"
    IMPLICIT_ACTOR = "implicit_actor"
    UNQUANTIFIED_NON_FUNCTIONAL = (
        "unquantified_non_functional"
    )


_TBD_PATTERNS = (
    re.compile(
        r"\bTBD\b"
        r"|\bTBC\b"
        r"|\bto\s+be\s+determined\b"
        r"|\bto\s+be\s+confirmed\b"
        r"|\bto\s+be\s+defined\b"
        r"|\bto\s+be\s+agreed\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"يحدد\s+لاحق"
        r"|يُحدد\s+لاحق"
        r"|تحدد\s+لاحق"
        r"|يتم\s+تحديد(?:ه|ها)?\s+لاحق"
        r"|يتم\s+تعريف(?:ه|ها)?\s+لاحق"
        r"|يتفق\s+عليه\s+لاحق"
        r"|يتم\s+الاتفاق\s+عليه\s+لاحق"
    ),
)


_AMBIGUOUS_PATTERNS = (
    re.compile(
        r"\bas\s+soon\s+as\s+possible\b"
        r"|\bas\s+needed\b"
        r"|\bwhen\s+needed\b"
        r"|\bwhere\s+appropriate\b"
        r"|\bwhere\s+possible\b"
        r"|\bif\s+appropriate\b"
        r"|\bif\s+possible\b"
        r"|\bfrom\s+time\s+to\s+time\b"
        r"|\bappropriate\b"
        r"|\badequate\b"
        r"|\bsufficient\b"
        r"|\breasonable\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"حسب\s+الحاجة"
        r"|عند\s+الحاجة"
        r"|بأسرع\s+وقت"
        r"|باسرع\s+وقت"
        r"|في\s+أسرع\s+وقت"
        r"|في\s+اسرع\s+وقت"
        r"|عند\s+الإمكان"
        r"|عند\s+الامكان"
        r"|حيثما\s+أمكن"
        r"|حيثما\s+امكن"
        r"|مناسب"
        r"|ملائم"
        r"|كاف"
        r"|معقول"
    ),
)


_SUBJECTIVE_PATTERNS = (
    re.compile(
        r"\buser[- ]friendly\b"
        r"|\bintuitive\b"
        r"|\beasy\s+to\s+use\b"
        r"|\bsimple\b"
        r"|\bflexible\b"
        r"|\befficient\b"
        r"|\bfast\b"
        r"|\bquickly\b"
        r"|\bseamless\b"
        r"|\brobust\b"
        r"|\bmodern\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"سهل(?:ة)?\s+الاستخدام"
        r"|بديهي"
        r"|مرن"
        r"|فعال"
        r"|بكفاءة"
        r"|سريع"
        r"|بسرعة"
        r"|سلس"
        r"|حديث"
        r"|متين"
    ),
)


_OBLIGATION_PATTERNS = (
    re.compile(
        r"\bmust(?:\s+not)?\b"
        r"|\bshall(?:\s+not)?\b"
        r"|\bshould(?:\s+not)?\b"
        r"|\bcould\b"
        r"|\bmay\b"
        r"|\brequired\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"يجب"
        r"|يتعين"
        r"|ينبغي"
        r"|يمكن"
        r"|يلزم"
        r"|مطلوب\s+من"
    ),
)


_IMPLICIT_ACTOR_RE = re.compile(
    r"^\s*(?:"
    r"must"
    r"|shall"
    r"|should"
    r"|could"
    r"|may"
    r"|required\s+to"
    r")\b",
    re.IGNORECASE,
)


_MEASUREMENT_RE = re.compile(
    r"(?:"
    r"[0-9٠-٩۰-۹]"
    r"|%"
    r"|\bpercent\b"
    r"|\bpercentage\b"
    r"|\bmilliseconds?\b"
    r"|\bms\b"
    r"|\bseconds?\b"
    r"|\bminutes?\b"
    r"|\bhours?\b"
    r"|\bdays?\b"
    r"|\brequests?\s*/\s*(?:second|minute)\b"
    r"|\btransactions?\s*/\s*(?:second|minute)\b"
    r"|بالمئة"
    r"|بالمائة"
    r"|مللي\s*ثانية"
    r"|ملي\s*ثانية"
    r"|ثوان"
    r"|ثانية"
    r"|دقائق"
    r"|دقيقة"
    r"|ساعات"
    r"|ساعة"
    r")",
    re.IGNORECASE,
)


_SEVERITY_BY_ISSUE = {
    RequirementQualityIssueCode.TBD_PLACEHOLDER:
        RequirementQualitySeverity.HIGH,

    RequirementQualityIssueCode
    .UNQUANTIFIED_NON_FUNCTIONAL:
        RequirementQualitySeverity.HIGH,

    RequirementQualityIssueCode.AMBIGUOUS_TERM:
        RequirementQualitySeverity.MEDIUM,

    RequirementQualityIssueCode.SUBJECTIVE_TERM:
        RequirementQualitySeverity.MEDIUM,

    RequirementQualityIssueCode.MULTIPLE_OBLIGATIONS:
        RequirementQualitySeverity.MEDIUM,

    RequirementQualityIssueCode.IMPLICIT_ACTOR:
        RequirementQualitySeverity.LOW,
}


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


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _match_count(
    text: str,
    patterns: tuple[
        re.Pattern[str],
        ...
    ],
) -> int:
    return sum(
        len(
            tuple(
                pattern.finditer(text)
            )
        )
        for pattern
        in patterns
    )


@dataclass(frozen=True)
class RequirementQualityFinding:
    """Privacy-safe deterministic quality finding."""

    requirement_id: str

    statement_sha256: str

    issue_code: RequirementQualityIssueCode

    severity: RequirementQualitySeverity

    signal_count: int

    version: str = (
        REQUIREMENT_QUALITY_VERSION
    )

    finding_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        requirement_id = _required_text(
            self.requirement_id,
            "requirement_id",
        )

        statement_sha256 = _required_text(
            self.statement_sha256,
            "statement_sha256",
        )

        if (
            len(statement_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character
                in statement_sha256
            )
        ):
            raise ValueError(
                "statement_sha256 must be "
                "a SHA-256 hexadecimal digest."
            )

        statement_sha256 = (
            statement_sha256.lower()
        )

        if not isinstance(
            self.issue_code,
            RequirementQualityIssueCode,
        ):
            raise TypeError(
                "issue_code must be a "
                "RequirementQualityIssueCode."
            )

        if not isinstance(
            self.severity,
            RequirementQualitySeverity,
        ):
            raise TypeError(
                "severity must be a "
                "RequirementQualitySeverity."
            )

        if (
            isinstance(
                self.signal_count,
                bool,
            )
            or not isinstance(
                self.signal_count,
                int,
            )
            or self.signal_count < 1
        ):
            raise ValueError(
                "signal_count must be a "
                "positive integer."
            )

        expected_severity = (
            _SEVERITY_BY_ISSUE[
                self.issue_code
            ]
        )

        if (
            self.severity
            is not expected_severity
        ):
            raise ValueError(
                "severity does not match "
                "the deterministic issue policy."
            )

        payload = "\x1f".join(
            (
                self.version,
                requirement_id,
                statement_sha256,
                self.issue_code.value,
                self.severity.value,
                str(
                    self.signal_count
                ),
            )
        )

        object.__setattr__(
            self,
            "requirement_id",
            requirement_id,
        )

        object.__setattr__(
            self,
            "statement_sha256",
            statement_sha256,
        )

        object.__setattr__(
            self,
            "finding_id",
            _sha256(
                payload
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "finding_id": (
                self.finding_id
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "statement_sha256": (
                self.statement_sha256
            ),
            "issue_code": (
                self.issue_code.value
            ),
            "severity": (
                self.severity.value
            ),
            "signal_count": (
                self.signal_count
            ),
        }


@dataclass(frozen=True)
class RequirementQualityAssessment:
    """Quality assessment for one requirement."""

    request_id: str

    requirement_id: str

    statement_sha256: str

    decision: RequirementQualityDecision

    findings: tuple[
        RequirementQualityFinding,
        ...
    ]

    version: str = (
        REQUIREMENT_QUALITY_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        requirement_id = _required_text(
            self.requirement_id,
            "requirement_id",
        )

        statement_sha256 = _required_text(
            self.statement_sha256,
            "statement_sha256",
        ).lower()

        if (
            len(statement_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character
                in statement_sha256
            )
        ):
            raise ValueError(
                "statement_sha256 must be "
                "a SHA-256 hexadecimal digest."
            )

        if not isinstance(
            self.decision,
            RequirementQualityDecision,
        ):
            raise TypeError(
                "decision must be a "
                "RequirementQualityDecision."
            )

        try:
            findings = tuple(
                self.findings
            )
        except TypeError as exc:
            raise TypeError(
                "findings must be iterable."
            ) from exc

        issue_codes = []

        for finding in findings:
            if not isinstance(
                finding,
                RequirementQualityFinding,
            ):
                raise TypeError(
                    "findings must contain only "
                    "RequirementQualityFinding values."
                )

            if (
                finding.requirement_id
                != requirement_id
            ):
                raise ValueError(
                    "Finding requirement ID mismatch."
                )

            if (
                finding.statement_sha256
                != statement_sha256
            ):
                raise ValueError(
                    "Finding statement hash mismatch."
                )

            issue_codes.append(
                finding.issue_code
            )

        if (
            len(
                set(
                    issue_codes
                )
            )
            != len(
                issue_codes
            )
        ):
            raise ValueError(
                "Quality issue codes must be unique "
                "within one assessment."
            )

        expected_decision = (
            _decision_for_findings(
                findings
            )
        )

        if (
            self.decision
            is not expected_decision
        ):
            raise ValueError(
                "decision does not match "
                "the deterministic findings."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                requirement_id,
                statement_sha256,
                self.decision.value,
                ",".join(
                    finding.finding_id
                    for finding
                    in findings
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
            "requirement_id",
            requirement_id,
        )

        object.__setattr__(
            self,
            "statement_sha256",
            statement_sha256,
        )

        object.__setattr__(
            self,
            "findings",
            findings,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _sha256(
                payload
            ),
        )

    @property
    def issue_count(
        self,
    ) -> int:
        return len(
            self.findings
        )

    @property
    def has_high_severity_issue(
        self,
    ) -> bool:
        return any(
            finding.severity
            is RequirementQualitySeverity.HIGH
            for finding
            in self.findings
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "assessment_id": (
                self.assessment_id
            ),
            "request_id": (
                self.request_id
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "statement_sha256": (
                self.statement_sha256
            ),
            "decision": (
                self.decision.value
            ),
            "issue_count": (
                self.issue_count
            ),
            "has_high_severity_issue": (
                self.has_high_severity_issue
            ),
            "findings": [
                finding.to_dict()
                for finding
                in self.findings
            ],
        }


@dataclass(frozen=True)
class RequirementsQualityReport:
    """Aggregate deterministic quality report."""

    request_id: str

    extraction_id: str

    assessments: tuple[
        RequirementQualityAssessment,
        ...
    ]

    algorithm: str = (
        REQUIREMENT_QUALITY_ALGORITHM
    )

    version: str = (
        REQUIREMENT_QUALITY_VERSION
    )

    report_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        extraction_id = _required_text(
            self.extraction_id,
            "extraction_id",
        )

        algorithm = _required_text(
            self.algorithm,
            "algorithm",
        )

        try:
            assessments = tuple(
                self.assessments
            )
        except TypeError as exc:
            raise TypeError(
                "assessments must be iterable."
            ) from exc

        requirement_ids = []

        for assessment in assessments:
            if not isinstance(
                assessment,
                RequirementQualityAssessment,
            ):
                raise TypeError(
                    "assessments must contain only "
                    "RequirementQualityAssessment values."
                )

            if (
                assessment.request_id
                != request_id
            ):
                raise ValueError(
                    "Assessment request ID mismatch."
                )

            requirement_ids.append(
                assessment.requirement_id
            )

        if (
            len(
                set(
                    requirement_ids
                )
            )
            != len(
                requirement_ids
            )
        ):
            raise ValueError(
                "Requirement assessments "
                "must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                extraction_id,
                algorithm,
                ",".join(
                    assessment.assessment_id
                    for assessment
                    in assessments
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
            "extraction_id",
            extraction_id,
        )

        object.__setattr__(
            self,
            "assessments",
            assessments,
        )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "report_id",
            _sha256(
                payload
            ),
        )

    @property
    def pass_count(
        self,
    ) -> int:
        return sum(
            assessment.decision
            is RequirementQualityDecision.PASS
            for assessment
            in self.assessments
        )

    @property
    def review_count(
        self,
    ) -> int:
        return sum(
            assessment.decision
            is RequirementQualityDecision.REVIEW
            for assessment
            in self.assessments
        )

    @property
    def rewrite_count(
        self,
    ) -> int:
        return sum(
            assessment.decision
            is RequirementQualityDecision.REWRITE
            for assessment
            in self.assessments
        )

    @property
    def issue_count(
        self,
    ) -> int:
        return sum(
            assessment.issue_count
            for assessment
            in self.assessments
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "report_id": (
                self.report_id
            ),
            "request_id": (
                self.request_id
            ),
            "extraction_id": (
                self.extraction_id
            ),
            "algorithm": (
                self.algorithm
            ),
            "assessment_count": (
                len(
                    self.assessments
                )
            ),
            "issue_count": (
                self.issue_count
            ),
            "pass_count": (
                self.pass_count
            ),
            "review_count": (
                self.review_count
            ),
            "rewrite_count": (
                self.rewrite_count
            ),
            "assessments": [
                assessment.to_dict()
                for assessment
                in self.assessments
            ],
        }


def _decision_for_findings(
    findings: tuple[
        RequirementQualityFinding,
        ...
    ],
) -> RequirementQualityDecision:
    if any(
        finding.severity
        is RequirementQualitySeverity.HIGH
        for finding
        in findings
    ):
        return (
            RequirementQualityDecision.REWRITE
        )

    if findings:
        return (
            RequirementQualityDecision.REVIEW
        )

    return RequirementQualityDecision.PASS


def assess_requirement_quality(
    requirement: RequirementItem,
) -> RequirementQualityAssessment:
    """Assess one requirement using conservative deterministic rules."""

    if not isinstance(
        requirement,
        RequirementItem,
    ):
        raise TypeError(
            "requirement must be a "
            "RequirementItem."
        )

    text = requirement.statement

    detected: list[
        tuple[
            RequirementQualityIssueCode,
            int,
        ]
    ] = []

    tbd_count = _match_count(
        text,
        _TBD_PATTERNS,
    )

    if tbd_count:
        detected.append(
            (
                RequirementQualityIssueCode
                .TBD_PLACEHOLDER,
                tbd_count,
            )
        )

    ambiguous_count = _match_count(
        text,
        _AMBIGUOUS_PATTERNS,
    )

    if ambiguous_count:
        detected.append(
            (
                RequirementQualityIssueCode
                .AMBIGUOUS_TERM,
                ambiguous_count,
            )
        )

    subjective_count = _match_count(
        text,
        _SUBJECTIVE_PATTERNS,
    )

    if subjective_count:
        detected.append(
            (
                RequirementQualityIssueCode
                .SUBJECTIVE_TERM,
                subjective_count,
            )
        )

    obligation_count = _match_count(
        text,
        _OBLIGATION_PATTERNS,
    )

    if obligation_count > 1:
        detected.append(
            (
                RequirementQualityIssueCode
                .MULTIPLE_OBLIGATIONS,
                obligation_count,
            )
        )

    if _IMPLICIT_ACTOR_RE.search(
        text
    ):
        detected.append(
            (
                RequirementQualityIssueCode
                .IMPLICIT_ACTOR,
                1,
            )
        )

    if (
        requirement.requirement_type
        is RequirementType.NON_FUNCTIONAL
        and not _MEASUREMENT_RE.search(
            text
        )
    ):
        detected.append(
            (
                RequirementQualityIssueCode
                .UNQUANTIFIED_NON_FUNCTIONAL,
                1,
            )
        )

    findings = tuple(
        RequirementQualityFinding(
            requirement_id=(
                requirement.requirement_id
            ),
            statement_sha256=(
                requirement.statement_sha256
            ),
            issue_code=issue_code,
            severity=(
                _SEVERITY_BY_ISSUE[
                    issue_code
                ]
            ),
            signal_count=signal_count,
        )
        for issue_code, signal_count
        in detected
    )

    decision = _decision_for_findings(
        findings
    )

    return RequirementQualityAssessment(
        request_id=(
            requirement.request_id
        ),
        requirement_id=(
            requirement.requirement_id
        ),
        statement_sha256=(
            requirement.statement_sha256
        ),
        decision=decision,
        findings=findings,
    )


def assess_requirements_quality(
    extraction: RequirementExtractionResult,
) -> RequirementsQualityReport:
    """Assess every extracted requirement while preserving order."""

    if not isinstance(
        extraction,
        RequirementExtractionResult,
    ):
        raise TypeError(
            "extraction must be a "
            "RequirementExtractionResult."
        )

    assessments = tuple(
        assess_requirement_quality(
            requirement
        )
        for requirement
        in extraction.requirements
    )

    return RequirementsQualityReport(
        request_id=(
            extraction.request_id
        ),
        extraction_id=(
            extraction.extraction_id
        ),
        assessments=assessments,
    )
