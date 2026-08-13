"""Governed policy-change briefing for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from govba.change.contract import (
    PolicyChangeImpact,
    PolicyChangeType,
)
from govba.change.significance import (
    ChangeSignificanceAssessment,
    ChangeSignificanceResult,
    ChangeSignificanceSignal,
)
from govba.rag.evidence import EvidenceChunk


CHANGE_BRIEFING_VERSION = (
    "govba-change-briefing-v1"
)


class ChangeBriefingDecision(
    str,
    Enum,
):
    """Governance disposition for a change briefing."""

    READY = "ready"
    REVIEW = "review"
    ABSTAIN = "abstain"


class ChangeBriefingReason(
    str,
    Enum,
):
    """Stable governance reasons."""

    VERIFIED_CHANGES = (
        "verified_changes"
    )

    NO_CHANGES = (
        "no_changes"
    )

    CRITICAL_CHANGE = (
        "critical_change"
    )

    MISSING_EVIDENCE = (
        "missing_evidence"
    )

    EVIDENCE_HASH_MISMATCH = (
        "evidence_hash_mismatch"
    )

    ASSESSMENT_MISMATCH = (
        "assessment_mismatch"
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


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _prepare_chunk_map(
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
                f"{field_name} contains evidence "
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


@dataclass(frozen=True)
class ChangeEvidenceReference:
    """Privacy-safe reference to authoritative change evidence."""

    chunk_id: str

    document_id: str

    content_hash: str

    side: str

    version: str = (
        CHANGE_BRIEFING_VERSION
    )

    reference_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        chunk_id = _required_text(
            self.chunk_id,
            "chunk_id",
        )

        document_id = _required_text(
            self.document_id,
            "document_id",
        )

        content_hash = _required_text(
            self.content_hash,
            "content_hash",
        ).lower()

        if (
            len(content_hash) != 64
            or any(
                value
                not in "0123456789abcdef"
                for value in content_hash
            )
        ):
            raise ValueError(
                "content_hash must be a "
                "SHA-256 digest."
            )

        side = _required_text(
            self.side,
            "side",
        ).lower()

        if side not in {
            "baseline",
            "candidate",
        }:
            raise ValueError(
                "side must be baseline "
                "or candidate."
            )

        payload = "\x1f".join(
            (
                self.version,
                chunk_id,
                document_id,
                content_hash,
                side,
            )
        )

        object.__setattr__(
            self,
            "chunk_id",
            chunk_id,
        )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        object.__setattr__(
            self,
            "content_hash",
            content_hash,
        )

        object.__setattr__(
            self,
            "side",
            side,
        )

        object.__setattr__(
            self,
            "reference_id",
            _sha256(
                payload
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, str]:
        return {
            "version": (
                self.version
            ),
            "reference_id": (
                self.reference_id
            ),
            "chunk_id": (
                self.chunk_id
            ),
            "document_id": (
                self.document_id
            ),
            "content_hash": (
                self.content_hash
            ),
            "side": (
                self.side
            ),
        }


@dataclass(frozen=True)
class ChangeBriefingItem:
    """One governed briefing item linked to authoritative evidence."""

    finding_id: str

    change_type: (
        PolicyChangeType
    )

    impact: (
        PolicyChangeImpact
    )

    signals: tuple[
        ChangeSignificanceSignal,
        ...
    ]

    evidence: tuple[
        ChangeEvidenceReference,
        ...
    ]

    version: str = (
        CHANGE_BRIEFING_VERSION
    )

    item_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        finding_id = _required_text(
            self.finding_id,
            "finding_id",
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
                    "signals must contain only "
                    "ChangeSignificanceSignal values."
                )

        try:
            evidence = tuple(
                self.evidence
            )
        except TypeError as exc:
            raise TypeError(
                "evidence must be iterable."
            ) from exc

        if not evidence:
            raise ValueError(
                "A briefing item requires "
                "authoritative evidence."
            )

        for reference in evidence:
            if not isinstance(
                reference,
                ChangeEvidenceReference,
            ):
                raise TypeError(
                    "evidence must contain only "
                    "ChangeEvidenceReference values."
                )

        if (
            len(
                {
                    reference.reference_id
                    for reference in evidence
                }
            )
            != len(evidence)
        ):
            raise ValueError(
                "Evidence references must "
                "be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                finding_id,
                self.change_type.value,
                self.impact.value,
                ",".join(
                    signal.value
                    for signal in signals
                ),
                ",".join(
                    reference.reference_id
                    for reference in evidence
                ),
            )
        )

        object.__setattr__(
            self,
            "finding_id",
            finding_id,
        )

        object.__setattr__(
            self,
            "signals",
            signals,
        )

        object.__setattr__(
            self,
            "evidence",
            evidence,
        )

        object.__setattr__(
            self,
            "item_id",
            _sha256(
                payload
            ),
        )

    @property
    def headline(
        self,
    ) -> str:
        labels = {
            PolicyChangeType.ADDED:
                "Requirement added",

            PolicyChangeType.REMOVED:
                "Requirement removed",

            PolicyChangeType.MODIFIED:
                "Requirement modified",

            PolicyChangeType.UNCHANGED:
                "Requirement unchanged",
        }

        return labels[
            self.change_type
        ]

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "item_id": (
                self.item_id
            ),
            "finding_id": (
                self.finding_id
            ),
            "headline": (
                self.headline
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
            "evidence": [
                reference.to_dict()
                for reference
                in self.evidence
            ],
        }


@dataclass(frozen=True)
class GovernedChangeBriefing:
    """Governed evidence-linked policy-change briefing."""

    trace_id: str

    classification_id: str

    baseline_document_id: str

    candidate_document_id: str

    decision: (
        ChangeBriefingDecision
    )

    reasons: tuple[
        ChangeBriefingReason,
        ...
    ]

    items: tuple[
        ChangeBriefingItem,
        ...
    ]

    maximum_impact: (
        PolicyChangeImpact
    )

    version: str = (
        CHANGE_BRIEFING_VERSION
    )

    briefing_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        classification_id = (
            _required_text(
                self.classification_id,
                "classification_id",
            )
        )

        baseline_document_id = (
            _required_text(
                self.baseline_document_id,
                "baseline_document_id",
            )
        )

        candidate_document_id = (
            _required_text(
                self.candidate_document_id,
                "candidate_document_id",
            )
        )

        if not isinstance(
            self.decision,
            ChangeBriefingDecision,
        ):
            raise TypeError(
                "decision must be a "
                "ChangeBriefingDecision."
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
                "At least one briefing reason "
                "is required."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                ChangeBriefingReason,
            ):
                raise TypeError(
                    "reasons must contain only "
                    "ChangeBriefingReason values."
                )

        if len(
            set(
                reasons
            )
        ) != len(
            reasons
        ):
            raise ValueError(
                "Briefing reasons must be unique."
            )

        try:
            items = tuple(
                self.items
            )
        except TypeError as exc:
            raise TypeError(
                "items must be iterable."
            ) from exc

        for item in items:
            if not isinstance(
                item,
                ChangeBriefingItem,
            ):
                raise TypeError(
                    "items must contain only "
                    "ChangeBriefingItem values."
                )

        if (
            len(
                {
                    item.item_id
                    for item in items
                }
            )
            != len(items)
        ):
            raise ValueError(
                "Briefing items must be unique."
            )

        if not isinstance(
            self.maximum_impact,
            PolicyChangeImpact,
        ):
            raise TypeError(
                "maximum_impact must be a "
                "PolicyChangeImpact."
            )

        payload = "\x1f".join(
            (
                self.version,
                trace_id,
                classification_id,
                baseline_document_id,
                candidate_document_id,
                self.decision.value,
                self.maximum_impact.value,
                ",".join(
                    reason.value
                    for reason in reasons
                ),
                ",".join(
                    item.item_id
                    for item in items
                ),
            )
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "classification_id",
            classification_id,
        )

        object.__setattr__(
            self,
            "baseline_document_id",
            baseline_document_id,
        )

        object.__setattr__(
            self,
            "candidate_document_id",
            candidate_document_id,
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "items",
            items,
        )

        object.__setattr__(
            self,
            "briefing_id",
            _sha256(
                payload
            ),
        )

    @property
    def ready(
        self,
    ) -> bool:
        return (
            self.decision
            is ChangeBriefingDecision.READY
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is ChangeBriefingDecision.REVIEW
        )

    @property
    def abstained(
        self,
    ) -> bool:
        return (
            self.decision
            is ChangeBriefingDecision.ABSTAIN
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without raw policy text."""

        return {
            "version": (
                self.version
            ),
            "briefing_id": (
                self.briefing_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "classification_id": (
                self.classification_id
            ),
            "baseline_document_id": (
                self.baseline_document_id
            ),
            "candidate_document_id": (
                self.candidate_document_id
            ),
            "decision": (
                self.decision.value
            ),
            "ready": (
                self.ready
            ),
            "requires_review": (
                self.requires_review
            ),
            "abstained": (
                self.abstained
            ),
            "maximum_impact": (
                self.maximum_impact.value
            ),
            "reasons": [
                reason.value
                for reason in self.reasons
            ],
            "item_count": len(
                self.items
            ),
            "items": [
                item.to_dict()
                for item
                in self.items
            ],
        }


def _assessment_map(
    result: ChangeSignificanceResult,
) -> dict[
    str,
    ChangeSignificanceAssessment,
]:
    output = {}

    for assessment in (
        result.assessments
    ):
        if (
            assessment.classified_finding_id
            in output
        ):
            raise ValueError(
                "Duplicate classified finding "
                "assessment."
            )

        output[
            assessment.classified_finding_id
        ] = assessment

    return output


def build_governed_change_briefing(
    result: ChangeSignificanceResult,
    *,
    baseline_chunks: Iterable[
        EvidenceChunk
    ],
    candidate_chunks: Iterable[
        EvidenceChunk
    ],
    trace_id: str,
) -> GovernedChangeBriefing:
    """Build governed briefing from verified classified changes."""

    if not isinstance(
        result,
        ChangeSignificanceResult,
    ):
        raise TypeError(
            "result must be a "
            "ChangeSignificanceResult."
        )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    baseline_document_id = (
        result.report.request
        .baseline_source.document_id
    )

    candidate_document_id = (
        result.report.request
        .candidate_source.document_id
    )

    baseline_map = (
        _prepare_chunk_map(
            baseline_chunks,
            document_id=(
                baseline_document_id
            ),
            field_name="baseline_chunks",
        )
    )

    candidate_map = (
        _prepare_chunk_map(
            candidate_chunks,
            document_id=(
                candidate_document_id
            ),
            field_name="candidate_chunks",
        )
    )

    assessments = (
        _assessment_map(
            result
        )
    )

    items = []

    governance_failure = False

    for finding in (
        result.report.findings
    ):
        assessment = assessments.get(
            finding.finding_id
        )

        if assessment is None:
            governance_failure = True
            continue

        if (
            assessment.change_type
            is not finding.change_type
            or assessment.impact
            is not finding.impact
        ):
            governance_failure = True
            continue

        references = []

        if finding.baseline_chunk_id:
            chunk = baseline_map.get(
                finding.baseline_chunk_id
            )

            if (
                chunk is None
                or (
                    finding.baseline_content_hash
                    and chunk.content_hash
                    != finding.baseline_content_hash
                )
            ):
                governance_failure = True
                continue

            references.append(
                ChangeEvidenceReference(
                    chunk_id=chunk.chunk_id,
                    document_id=(
                        chunk.document_id
                    ),
                    content_hash=(
                        chunk.content_hash
                    ),
                    side="baseline",
                )
            )

        if finding.candidate_chunk_id:
            chunk = candidate_map.get(
                finding.candidate_chunk_id
            )

            if (
                chunk is None
                or (
                    finding.candidate_content_hash
                    and chunk.content_hash
                    != finding.candidate_content_hash
                )
            ):
                governance_failure = True
                continue

            references.append(
                ChangeEvidenceReference(
                    chunk_id=chunk.chunk_id,
                    document_id=(
                        chunk.document_id
                    ),
                    content_hash=(
                        chunk.content_hash
                    ),
                    side="candidate",
                )
            )

        if not references:
            governance_failure = True
            continue

        items.append(
            ChangeBriefingItem(
                finding_id=(
                    finding.finding_id
                ),
                change_type=(
                    finding.change_type
                ),
                impact=(
                    finding.impact
                ),
                signals=(
                    assessment.signals
                ),
                evidence=tuple(
                    references
                ),
            )
        )

    if governance_failure:
        decision = (
            ChangeBriefingDecision
            .ABSTAIN
        )

        reasons = (
            ChangeBriefingReason
            .MISSING_EVIDENCE,
        )

    elif (
        result.maximum_impact
        is PolicyChangeImpact.CRITICAL
    ):
        decision = (
            ChangeBriefingDecision
            .REVIEW
        )

        reasons = (
            ChangeBriefingReason
            .CRITICAL_CHANGE,
        )

    elif result.report.changed:
        decision = (
            ChangeBriefingDecision
            .READY
        )

        reasons = (
            ChangeBriefingReason
            .VERIFIED_CHANGES,
        )

    else:
        decision = (
            ChangeBriefingDecision
            .READY
        )

        reasons = (
            ChangeBriefingReason
            .NO_CHANGES,
        )

    return GovernedChangeBriefing(
        trace_id=trace,
        classification_id=(
            result.classification_id
        ),
        baseline_document_id=(
            baseline_document_id
        ),
        candidate_document_id=(
            candidate_document_id
        ),
        decision=decision,
        reasons=reasons,
        items=tuple(
            items
        ),
        maximum_impact=(
            result.maximum_impact
        ),
    )
