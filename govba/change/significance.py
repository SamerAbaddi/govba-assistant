"""Deterministic policy-change significance classification for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from govba.change.clause_detection import (
    ClauseChangeDetectionResult,
)
from govba.change.contract import (
    PolicyChangeFinding,
    PolicyChangeImpact,
    PolicyChangeReport,
    PolicyChangeType,
    build_policy_change_report,
)
from govba.rag.evidence import EvidenceChunk


CHANGE_SIGNIFICANCE_VERSION = (
    "govba-change-significance-v1"
)

CHANGE_SIGNIFICANCE_ALGORITHM = (
    "deterministic-change-significance-v1"
)


class ChangeSignificanceSignal(
    str,
    Enum,
):
    PENALTY_SANCTION = "penalty_sanction"
    PROHIBITION = "prohibition"
    FINANCIAL = "financial"
    ELIGIBILITY_ENTITLEMENT = (
        "eligibility_entitlement"
    )
    DEADLINE = "deadline"
    APPROVAL_AUTHORIZATION = (
        "approval_authorization"
    )
    OBLIGATION = "obligation"
    PROCEDURAL = "procedural"


_IMPACT_PRIORITY = {
    PolicyChangeImpact.INFORMATIONAL: 0,
    PolicyChangeImpact.LOW: 1,
    PolicyChangeImpact.MEDIUM: 2,
    PolicyChangeImpact.HIGH: 3,
    PolicyChangeImpact.CRITICAL: 4,
}


_SIGNAL_PATTERNS = {
    ChangeSignificanceSignal.PENALTY_SANCTION: (
        re.compile(
            r"\bpenalt(?:y|ies)\b"
            r"|\bfines?\b"
            r"|\bsanctions?\b"
            r"|\bdisciplinary\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"غرام(?:ة|ات)"
            r"|عقوب(?:ة|ات)"
            r"|جزاء(?:ات)?"
            r"|مخالف(?:ة|ات)"
        ),
    ),

    ChangeSignificanceSignal.PROHIBITION: (
        re.compile(
            r"\bprohibited\b"
            r"|\bforbidden\b"
            r"|\bmust\s+not\b"
            r"|\bshall\s+not\b"
            r"|\bmay\s+not\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يحظر"
            r"|ممنوع"
            r"|لا\s+يجوز"
            r"|يمنع"
        ),
    ),

    ChangeSignificanceSignal.FINANCIAL: (
        re.compile(
            r"\bfees?\b"
            r"|\bpayments?\b"
            r"|\bsalar(?:y|ies)\b"
            r"|\ballowances?\b"
            r"|\btaxes?\b"
            r"|\btariffs?\b"
            r"|\bcharges?\b"
            r"|\bcosts?\b"
            r"|\bJOD\b"
            r"|\bdinars?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"رسوم"
            r"|دفع"
            r"|راتب"
            r"|رواتب"
            r"|بدل"
            r"|ضريب(?:ة|ات)"
            r"|تعرف(?:ة|ات)"
            r"|دينار"
            r"|تكلف(?:ة|ات)"
        ),
    ),

    ChangeSignificanceSignal.ELIGIBILITY_ENTITLEMENT: (
        re.compile(
            r"\beligib(?:le|ility)\b"
            r"|\bentitl(?:ed|ement)\b"
            r"|\bqualif(?:y|ies|ied|ication)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"الأهلية"
            r"|اهلية"
            r"|مؤهل"
            r"|يستحق"
            r"|استحقاق"
            r"|شروط\s+الاستحقاق"
        ),
    ),

    ChangeSignificanceSignal.DEADLINE: (
        re.compile(
            r"\bdeadline\b"
            r"|\bno\s+later\s+than\b"
            r"|\bwithin\s+\d+\s+days?\b"
            r"|\bwithin\s+\d+\s+hours?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"موعد\s+نهائي"
            r"|في\s+موعد\s+أقصاه"
            r"|مهلة"
            r"|خلال\s+\d+\s+"
            r"(?:يوم|أيام|ساعة|ساعات)"
        ),
    ),

    ChangeSignificanceSignal.APPROVAL_AUTHORIZATION: (
        re.compile(
            r"\bapproval\b"
            r"|\bauthori[sz](?:e|ed|ation)\b"
            r"|\bpermit\b"
            r"|\bconsent\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"موافق(?:ة|ات)"
            r"|اعتماد"
            r"|تفويض"
            r"|تصريح"
            r"|إذن"
        ),
    ),

    ChangeSignificanceSignal.OBLIGATION: (
        re.compile(
            r"\bmust\b"
            r"|\bshall\b"
            r"|\brequired\b"
            r"|\bmandatory\b"
            r"|\bobligation\b"
            r"|\bhas\s+to\b"
            r"|\bhave\s+to\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"يجب"
            r"|يتعين"
            r"|يلزم"
            r"|إلزامي"
            r"|الزامي"
            r"|ملزم"
        ),
    ),

    ChangeSignificanceSignal.PROCEDURAL: (
        re.compile(
            r"\bsubmit\b"
            r"|\bapplication\b"
            r"|\bforms?\b"
            r"|\bprocedure\b"
            r"|\bprocess\b"
            r"|\bnotify\b"
            r"|\bdocumentation\b"
            r"|\brecords?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"تقديم"
            r"|طلب"
            r"|نموذج"
            r"|إجراء"
            r"|إجراءات"
            r"|إخطار"
            r"|توثيق"
            r"|سجل"
            r"|سجلات"
        ),
    ),
}


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


def _normalize_text(
    value: str,
) -> str:
    return unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()


@dataclass(frozen=True)
class ChangeSignificancePolicy:
    """Configurable deterministic impact policy."""

    penalty_impact: PolicyChangeImpact = (
        PolicyChangeImpact.CRITICAL
    )

    prohibition_impact: PolicyChangeImpact = (
        PolicyChangeImpact.HIGH
    )

    financial_impact: PolicyChangeImpact = (
        PolicyChangeImpact.HIGH
    )

    eligibility_impact: PolicyChangeImpact = (
        PolicyChangeImpact.HIGH
    )

    deadline_impact: PolicyChangeImpact = (
        PolicyChangeImpact.MEDIUM
    )

    approval_impact: PolicyChangeImpact = (
        PolicyChangeImpact.MEDIUM
    )

    obligation_impact: PolicyChangeImpact = (
        PolicyChangeImpact.MEDIUM
    )

    procedural_impact: PolicyChangeImpact = (
        PolicyChangeImpact.LOW
    )

    default_changed_impact: PolicyChangeImpact = (
        PolicyChangeImpact.LOW
    )

    version: str = (
        CHANGE_SIGNIFICANCE_VERSION
    )

    policy_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        impact_fields = (
            "penalty_impact",
            "prohibition_impact",
            "financial_impact",
            "eligibility_impact",
            "deadline_impact",
            "approval_impact",
            "obligation_impact",
            "procedural_impact",
            "default_changed_impact",
        )

        for field_name in impact_fields:
            if not isinstance(
                getattr(self, field_name),
                PolicyChangeImpact,
            ):
                raise TypeError(
                    f"{field_name} must be a "
                    "PolicyChangeImpact."
                )

        payload = "\x1f".join(
            (
                self.version,
                *(
                    getattr(
                        self,
                        field_name,
                    ).value
                    for field_name
                    in impact_fields
                ),
            )
        )

        object.__setattr__(
            self,
            "policy_id",
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )

    def impact_for(
        self,
        *,
        change_type: PolicyChangeType,
        signals: tuple[
            ChangeSignificanceSignal,
            ...
        ],
    ) -> PolicyChangeImpact:
        if (
            change_type
            is PolicyChangeType.UNCHANGED
        ):
            return (
                PolicyChangeImpact
                .INFORMATIONAL
            )

        impact_map = {
            ChangeSignificanceSignal
            .PENALTY_SANCTION:
                self.penalty_impact,

            ChangeSignificanceSignal
            .PROHIBITION:
                self.prohibition_impact,

            ChangeSignificanceSignal
            .FINANCIAL:
                self.financial_impact,

            ChangeSignificanceSignal
            .ELIGIBILITY_ENTITLEMENT:
                self.eligibility_impact,

            ChangeSignificanceSignal
            .DEADLINE:
                self.deadline_impact,

            ChangeSignificanceSignal
            .APPROVAL_AUTHORIZATION:
                self.approval_impact,

            ChangeSignificanceSignal
            .OBLIGATION:
                self.obligation_impact,

            ChangeSignificanceSignal
            .PROCEDURAL:
                self.procedural_impact,
        }

        impacts = tuple(
            impact_map[
                signal
            ]
            for signal in signals
        )

        if not impacts:
            return (
                self.default_changed_impact
            )

        return max(
            impacts,
            key=lambda impact: (
                _IMPACT_PRIORITY[
                    impact
                ]
            ),
        )


def detect_significance_signals(
    text: str,
) -> tuple[
    ChangeSignificanceSignal,
    ...
]:
    """Detect semantic risk signals without returning matched text."""

    text = _required_text(
        text,
        "text",
    )

    normalized = _normalize_text(
        text
    )

    signals = []

    for signal in (
        ChangeSignificanceSignal
    ):
        patterns = (
            _SIGNAL_PATTERNS[
                signal
            ]
        )

        if any(
            pattern.search(
                normalized
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


@dataclass(frozen=True)
class ChangeSignificanceAssessment:
    """Privacy-safe significance classification for one change."""

    structural_finding_id: str

    classified_finding_id: str

    change_type: PolicyChangeType

    impact: PolicyChangeImpact

    signals: tuple[
        ChangeSignificanceSignal,
        ...
    ]

    version: str = (
        CHANGE_SIGNIFICANCE_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        structural_id = _required_text(
            self.structural_finding_id,
            "structural_finding_id",
        )

        classified_id = _required_text(
            self.classified_finding_id,
            "classified_finding_id",
        )

        if not isinstance(
            self.change_type,
            PolicyChangeType,
        ):
            raise TypeError(
                "change_type must be a "
                "PolicyChangeType."
            )

        if not isinstance(
            self.impact,
            PolicyChangeImpact,
        ):
            raise TypeError(
                "impact must be a "
                "PolicyChangeImpact."
            )

        try:
            signals = tuple(
                self.signals
            )
        except TypeError as exc:
            raise TypeError(
                "signals must be iterable."
            ) from exc

        for signal in signals:
            if not isinstance(
                signal,
                ChangeSignificanceSignal,
            ):
                raise TypeError(
                    "signals must contain "
                    "ChangeSignificanceSignal values."
                )

        if (
            len(set(signals))
            != len(signals)
        ):
            raise ValueError(
                "signals must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                structural_id,
                classified_id,
                self.change_type.value,
                self.impact.value,
                ",".join(
                    signal.value
                    for signal in signals
                ),
            )
        )

        object.__setattr__(
            self,
            "structural_finding_id",
            structural_id,
        )

        object.__setattr__(
            self,
            "classified_finding_id",
            classified_id,
        )

        object.__setattr__(
            self,
            "signals",
            signals,
        )

        object.__setattr__(
            self,
            "assessment_id",
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "assessment_id": (
                self.assessment_id
            ),
            "structural_finding_id": (
                self.structural_finding_id
            ),
            "classified_finding_id": (
                self.classified_finding_id
            ),
            "change_type": (
                self.change_type.value
            ),
            "impact": (
                self.impact.value
            ),
            "signals": [
                signal.value
                for signal
                in self.signals
            ],
        }


@dataclass(frozen=True)
class ChangeSignificanceResult:
    """Final deterministic impact-classification result."""

    detection_id: str

    report: PolicyChangeReport

    assessments: tuple[
        ChangeSignificanceAssessment,
        ...
    ]

    policy_id: str

    version: str = (
        CHANGE_SIGNIFICANCE_VERSION
    )

    classification_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        detection_id = _required_text(
            self.detection_id,
            "detection_id",
        )

        if not isinstance(
            self.report,
            PolicyChangeReport,
        ):
            raise TypeError(
                "report must be a "
                "PolicyChangeReport."
            )

        try:
            assessments = tuple(
                self.assessments
            )
        except TypeError as exc:
            raise TypeError(
                "assessments must be iterable."
            ) from exc

        for assessment in assessments:
            if not isinstance(
                assessment,
                ChangeSignificanceAssessment,
            ):
                raise TypeError(
                    "assessments must contain only "
                    "ChangeSignificanceAssessment values."
                )

        if (
            len(assessments)
            != len(
                self.report.findings
            )
        ):
            raise ValueError(
                "Assessment count must match "
                "classified finding count."
            )

        assessment_ids = tuple(
            value.assessment_id
            for value in assessments
        )

        if (
            len(set(assessment_ids))
            != len(assessment_ids)
        ):
            raise ValueError(
                "Significance assessments "
                "must be unique."
            )

        policy_id = _required_text(
            self.policy_id,
            "policy_id",
        )

        payload = "\x1f".join(
            (
                self.version,
                detection_id,
                self.report.report_id,
                policy_id,
                ",".join(
                    assessment_ids
                ),
            )
        )

        object.__setattr__(
            self,
            "detection_id",
            detection_id,
        )

        object.__setattr__(
            self,
            "assessments",
            assessments,
        )

        object.__setattr__(
            self,
            "policy_id",
            policy_id,
        )

        object.__setattr__(
            self,
            "classification_id",
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )

    @property
    def maximum_impact(
        self,
    ) -> PolicyChangeImpact:
        return (
            self.report.maximum_impact
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without policy clause text."""

        return {
            "version": self.version,
            "classification_id": (
                self.classification_id
            ),
            "detection_id": (
                self.detection_id
            ),
            "policy_id": (
                self.policy_id
            ),
            "maximum_impact": (
                self.maximum_impact.value
            ),
            "report": (
                self.report.to_dict()
            ),
            "assessments": [
                assessment.to_dict()
                for assessment
                in self.assessments
            ],
        }


def _chunk_map(
    chunks: Iterable[
        EvidenceChunk
    ],
    *,
    document_id: str,
    field_name: str,
) -> dict[
    str,
    EvidenceChunk,
]:
    if isinstance(
        chunks,
        (str, bytes),
    ):
        raise TypeError(
            f"{field_name} must contain "
            "EvidenceChunk values."
        )

    try:
        values = tuple(
            chunks
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    output = {}

    for chunk in values:
        if not isinstance(
            chunk,
            EvidenceChunk,
        ):
            raise TypeError(
                f"{field_name} must contain only "
                "EvidenceChunk values."
            )

        if (
            chunk.document_id
            != document_id
        ):
            raise ValueError(
                f"{field_name} contains a chunk "
                "from the wrong document."
            )

        if chunk.chunk_id in output:
            raise ValueError(
                f"{field_name} contains "
                "duplicate chunk IDs."
            )

        output[
            chunk.chunk_id
        ] = chunk

    return output


def classify_change_significance(
    detection: ClauseChangeDetectionResult,
    *,
    baseline_chunks: Iterable[
        EvidenceChunk
    ],
    candidate_chunks: Iterable[
        EvidenceChunk
    ],
    policy: (
        ChangeSignificancePolicy
        | None
    ) = None,
) -> ChangeSignificanceResult:
    """Assign deterministic triage significance to structural changes."""

    if not isinstance(
        detection,
        ClauseChangeDetectionResult,
    ):
        raise TypeError(
            "detection must be a "
            "ClauseChangeDetectionResult."
        )

    if policy is None:
        policy = (
            ChangeSignificancePolicy()
        )

    if not isinstance(
        policy,
        ChangeSignificancePolicy,
    ):
        raise TypeError(
            "policy must be a "
            "ChangeSignificancePolicy."
        )

    baseline_map = _chunk_map(
        baseline_chunks,
        document_id=(
            detection.report.request
            .baseline_source.document_id
        ),
        field_name="baseline_chunks",
    )

    candidate_map = _chunk_map(
        candidate_chunks,
        document_id=(
            detection.report.request
            .candidate_source.document_id
        ),
        field_name="candidate_chunks",
    )

    if (
        len(detection.matches)
        != len(
            detection.report.findings
        )
    ):
        raise ValueError(
            "Detection matches and findings "
            "must have equal length."
        )

    classified_findings = []
    provisional = []

    for match, structural_finding in zip(
        detection.matches,
        detection.report.findings,
        strict=True,
    ):
        if (
            match.change_type
            is not structural_finding.change_type
        ):
            raise ValueError(
                "Structural finding does not "
                "match clause-change result."
            )

        if (
            match.baseline_chunk_id
            != structural_finding
            .baseline_chunk_id
            or match.candidate_chunk_id
            != structural_finding
            .candidate_chunk_id
        ):
            raise ValueError(
                "Chunk provenance mismatch between "
                "detection and structural report."
            )

        texts = []

        if match.baseline_chunk_id:
            baseline_chunk = (
                baseline_map.get(
                    match.baseline_chunk_id
                )
            )

            if baseline_chunk is None:
                raise ValueError(
                    "Required baseline chunk "
                    "is missing."
                )

            texts.append(
                baseline_chunk.text
            )

        if match.candidate_chunk_id:
            candidate_chunk = (
                candidate_map.get(
                    match.candidate_chunk_id
                )
            )

            if candidate_chunk is None:
                raise ValueError(
                    "Required candidate chunk "
                    "is missing."
                )

            texts.append(
                candidate_chunk.text
            )

        if (
            match.change_type
            is PolicyChangeType.UNCHANGED
        ):
            signals = ()
        else:
            signal_set = set()

            for text in texts:
                signal_set.update(
                    detect_significance_signals(
                        text
                    )
                )

            signals = tuple(
                signal
                for signal
                in ChangeSignificanceSignal
                if signal in signal_set
            )

        impact = policy.impact_for(
            change_type=(
                match.change_type
            ),
            signals=signals,
        )

        classified = (
            PolicyChangeFinding(
                change_type=(
                    structural_finding
                    .change_type
                ),
                impact=impact,
                baseline_chunk_id=(
                    structural_finding
                    .baseline_chunk_id
                ),
                candidate_chunk_id=(
                    structural_finding
                    .candidate_chunk_id
                ),
                baseline_content_hash=(
                    structural_finding
                    .baseline_content_hash
                ),
                candidate_content_hash=(
                    structural_finding
                    .candidate_content_hash
                ),
                reason_code=(
                    structural_finding
                    .reason_code
                ),
            )
        )

        classified_findings.append(
            classified
        )

        provisional.append(
            (
                structural_finding,
                classified,
                signals,
            )
        )

    report = build_policy_change_report(
        detection.report.request,
        findings=tuple(
            classified_findings
        ),
        algorithm=(
            CHANGE_SIGNIFICANCE_ALGORITHM
        ),
    )

    assessments = tuple(
        ChangeSignificanceAssessment(
            structural_finding_id=(
                structural.finding_id
            ),
            classified_finding_id=(
                classified.finding_id
            ),
            change_type=(
                classified.change_type
            ),
            impact=(
                classified.impact
            ),
            signals=signals,
        )
        for (
            structural,
            classified,
            signals,
        )
        in provisional
    )

    return ChangeSignificanceResult(
        detection_id=(
            detection.detection_id
        ),
        report=report,
        assessments=assessments,
        policy_id=(
            policy.policy_id
        ),
    )
