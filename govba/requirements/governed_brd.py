"""Governed BRD generation for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from govba.requirements.acceptance import (
    AcceptanceCriteriaResult,
    AcceptanceCriteriaStatus,
    enrich_requirements_with_acceptance_criteria,
)
from govba.requirements.contract import (
    BusinessRequirementsDocument,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    RequirementExtractionResult,
)
from govba.requirements.quality import (
    RequirementQualityDecision,
    RequirementsQualityReport,
)


GOVERNED_BRD_VERSION = (
    "govba-governed-brd-v1"
)

GOVERNED_BRD_ALGORITHM = (
    "deterministic-governed-brd-v1"
)


class GovernedBRDDecision(
    str,
    Enum,
):
    READY = "ready"
    REVIEW = "review"
    ABSTAIN = "abstain"


class GovernedBRDReason(
    str,
    Enum,
):
    NO_REQUIREMENTS = (
        "no_requirements"
    )

    QUALITY_REWRITE_REQUIRED = (
        "quality_rewrite_required"
    )

    QUALITY_REVIEW_REQUIRED = (
        "quality_review_required"
    )

    DRAFT_ACCEPTANCE_CRITERIA = (
        "draft_acceptance_criteria"
    )

    READY_FOR_HUMAN_REVIEW = (
        "ready_for_human_review"
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


@dataclass(frozen=True)
class GovernedBRDRequirementLink:
    """Traceability between source and generated BRD requirement."""

    sequence: int

    source_requirement_id: str

    brd_requirement_id: str

    acceptance_assessment_id: str

    quality_assessment_id: str

    acceptance_status: (
        AcceptanceCriteriaStatus
    )

    quality_decision: (
        RequirementQualityDecision
    )

    requires_human_review: bool

    version: str = (
        GOVERNED_BRD_VERSION
    )

    link_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.sequence,
                bool,
            )
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

        text_fields = (
            "source_requirement_id",
            "brd_requirement_id",
            "acceptance_assessment_id",
            "quality_assessment_id",
        )

        for field_name in text_fields:
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        if not isinstance(
            self.acceptance_status,
            AcceptanceCriteriaStatus,
        ):
            raise TypeError(
                "acceptance_status must be an "
                "AcceptanceCriteriaStatus."
            )

        if not isinstance(
            self.quality_decision,
            RequirementQualityDecision,
        ):
            raise TypeError(
                "quality_decision must be a "
                "RequirementQualityDecision."
            )

        if not isinstance(
            self.requires_human_review,
            bool,
        ):
            raise TypeError(
                "requires_human_review must "
                "be boolean."
            )

        expected_review = (
            self.acceptance_status
            is AcceptanceCriteriaStatus
            .DRAFT_REVIEW
            or self.quality_decision
            is not RequirementQualityDecision.PASS
        )

        if (
            self.requires_human_review
            != expected_review
        ):
            raise ValueError(
                "requires_human_review does not "
                "match governance state."
            )

        payload = "\x1f".join(
            (
                self.version,
                str(
                    self.sequence
                ),
                self.source_requirement_id,
                self.brd_requirement_id,
                self.acceptance_assessment_id,
                self.quality_assessment_id,
                self.acceptance_status.value,
                self.quality_decision.value,
                str(
                    self.requires_human_review
                ).lower(),
            )
        )

        object.__setattr__(
            self,
            "link_id",
            _sha256(
                payload
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "link_id": (
                self.link_id
            ),
            "sequence": (
                self.sequence
            ),
            "source_requirement_id": (
                self.source_requirement_id
            ),
            "brd_requirement_id": (
                self.brd_requirement_id
            ),
            "acceptance_assessment_id": (
                self.acceptance_assessment_id
            ),
            "quality_assessment_id": (
                self.quality_assessment_id
            ),
            "acceptance_status": (
                self.acceptance_status.value
            ),
            "quality_decision": (
                self.quality_decision.value
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
        }


def _reasons_for_links(
    links: tuple[
        GovernedBRDRequirementLink,
        ...
    ],
) -> tuple[
    GovernedBRDReason,
    ...
]:
    if not links:
        return (
            GovernedBRDReason
            .NO_REQUIREMENTS,
        )

    reasons = []

    if any(
        link.quality_decision
        is RequirementQualityDecision.REWRITE
        for link
        in links
    ):
        reasons.append(
            GovernedBRDReason
            .QUALITY_REWRITE_REQUIRED
        )

    if any(
        link.quality_decision
        is RequirementQualityDecision.REVIEW
        for link
        in links
    ):
        reasons.append(
            GovernedBRDReason
            .QUALITY_REVIEW_REQUIRED
        )

    if any(
        link.acceptance_status
        is AcceptanceCriteriaStatus.DRAFT_REVIEW
        for link
        in links
    ):
        reasons.append(
            GovernedBRDReason
            .DRAFT_ACCEPTANCE_CRITERIA
        )

    if not reasons:
        reasons.append(
            GovernedBRDReason
            .READY_FOR_HUMAN_REVIEW
        )

    return tuple(
        reasons
    )


def _decision_for_links(
    links: tuple[
        GovernedBRDRequirementLink,
        ...
    ],
) -> GovernedBRDDecision:
    if not links:
        return (
            GovernedBRDDecision.ABSTAIN
        )

    if any(
        (
            link.quality_decision
            is not RequirementQualityDecision.PASS
            or link.acceptance_status
            is AcceptanceCriteriaStatus.DRAFT_REVIEW
        )
        for link
        in links
    ):
        return (
            GovernedBRDDecision.REVIEW
        )

    return GovernedBRDDecision.READY


@dataclass(frozen=True)
class GovernedBRDResult:
    """Governed, traceable BRD generation result."""

    request_id: str

    project_id: str

    trace_id: str

    source_extraction_id: str

    enriched_extraction_id: str

    acceptance_result_id: str

    quality_report_id: str

    decision: GovernedBRDDecision

    reasons: tuple[
        GovernedBRDReason,
        ...
    ]

    brd: BusinessRequirementsDocument

    links: tuple[
        GovernedBRDRequirementLink,
        ...
    ]

    algorithm: str = (
        GOVERNED_BRD_ALGORITHM
    )

    version: str = (
        GOVERNED_BRD_VERSION
    )

    result_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        text_fields = (
            "request_id",
            "project_id",
            "trace_id",
            "source_extraction_id",
            "enriched_extraction_id",
            "acceptance_result_id",
            "quality_report_id",
            "algorithm",
        )

        for field_name in text_fields:
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        if not isinstance(
            self.decision,
            GovernedBRDDecision,
        ):
            raise TypeError(
                "decision must be a "
                "GovernedBRDDecision."
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
                "At least one governance reason "
                "is required."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                GovernedBRDReason,
            ):
                raise TypeError(
                    "reasons must contain only "
                    "GovernedBRDReason values."
                )

        if (
            len(
                set(
                    reasons
                )
            )
            != len(
                reasons
            )
        ):
            raise ValueError(
                "Governance reasons must be unique."
            )

        if not isinstance(
            self.brd,
            BusinessRequirementsDocument,
        ):
            raise TypeError(
                "brd must be a "
                "BusinessRequirementsDocument."
            )

        if (
            self.brd.request_id
            != self.request_id
        ):
            raise ValueError(
                "BRD request ID mismatch."
            )

        if (
            self.brd.project_id
            != self.project_id
        ):
            raise ValueError(
                "BRD project ID mismatch."
            )

        if (
            self.brd.trace_id
            != self.trace_id
        ):
            raise ValueError(
                "BRD trace ID mismatch."
            )

        try:
            links = tuple(
                self.links
            )
        except TypeError as exc:
            raise TypeError(
                "links must be iterable."
            ) from exc

        for link in links:
            if not isinstance(
                link,
                GovernedBRDRequirementLink,
            ):
                raise TypeError(
                    "links must contain only "
                    "GovernedBRDRequirementLink values."
                )

        sequences = tuple(
            link.sequence
            for link
            in links
        )

        if sequences != tuple(
            range(
                len(
                    links
                )
            )
        ):
            raise ValueError(
                "Link sequences must be "
                "contiguous from zero."
            )

        if (
            len(
                links
            )
            != self.brd.requirement_count
        ):
            raise ValueError(
                "Link count must match "
                "BRD requirement count."
            )

        for link, requirement in zip(
            links,
            self.brd.requirements,
            strict=True,
        ):
            if (
                link.brd_requirement_id
                != requirement.requirement_id
            ):
                raise ValueError(
                    "BRD requirement traceability "
                    "link mismatch."
                )

        if (
            len(
                {
                    link.link_id
                    for link
                    in links
                }
            )
            != len(
                links
            )
        ):
            raise ValueError(
                "BRD links must be unique."
            )

        expected_decision = (
            _decision_for_links(
                links
            )
        )

        expected_reasons = (
            _reasons_for_links(
                links
            )
        )

        if (
            self.decision
            is not expected_decision
        ):
            raise ValueError(
                "decision does not match "
                "governance state."
            )

        if (
            reasons
            != expected_reasons
        ):
            raise ValueError(
                "reasons do not match "
                "governance state."
            )

        payload = "\x1f".join(
            (
                self.version,
                self.request_id,
                self.project_id,
                self.trace_id,
                self.source_extraction_id,
                self.enriched_extraction_id,
                self.acceptance_result_id,
                self.quality_report_id,
                self.decision.value,
                ",".join(
                    reason.value
                    for reason
                    in reasons
                ),
                self.brd.brd_id,
                ",".join(
                    link.link_id
                    for link
                    in links
                ),
                self.algorithm,
            )
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "links",
            links,
        )

        object.__setattr__(
            self,
            "result_id",
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
            is GovernedBRDDecision.READY
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is GovernedBRDDecision.REVIEW
        )

    @property
    def abstained(
        self,
    ) -> bool:
        return (
            self.decision
            is GovernedBRDDecision.ABSTAIN
        )

    @property
    def requirement_count(
        self,
    ) -> int:
        return (
            self.brd.requirement_count
        )

    @property
    def human_review_count(
        self,
    ) -> int:
        return sum(
            link.requires_human_review
            for link
            in self.links
        )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, object]:
        if not isinstance(
            include_content,
            bool,
        ):
            raise TypeError(
                "include_content must be boolean."
            )

        return {
            "version": self.version,
            "result_id": (
                self.result_id
            ),
            "request_id": (
                self.request_id
            ),
            "project_id": (
                self.project_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "source_extraction_id": (
                self.source_extraction_id
            ),
            "enriched_extraction_id": (
                self.enriched_extraction_id
            ),
            "acceptance_result_id": (
                self.acceptance_result_id
            ),
            "quality_report_id": (
                self.quality_report_id
            ),
            "algorithm": (
                self.algorithm
            ),
            "decision": (
                self.decision.value
            ),
            "reasons": [
                reason.value
                for reason
                in self.reasons
            ],
            "ready": (
                self.ready
            ),
            "requires_review": (
                self.requires_review
            ),
            "abstained": (
                self.abstained
            ),
            "requirement_count": (
                self.requirement_count
            ),
            "human_review_count": (
                self.human_review_count
            ),
            "brd": (
                self.brd.to_dict(
                    include_content=(
                        include_content
                    )
                )
            ),
            "links": [
                link.to_dict()
                for link
                in self.links
            ],
        }


def build_governed_brd(
    request: RequirementsRequest,
    extraction: RequirementExtractionResult,
    acceptance: AcceptanceCriteriaResult,
    quality: RequirementsQualityReport,
    *,
    title: str | None = None,
) -> GovernedBRDResult:
    """Generate a governed BRD from validated Stage 16 components."""

    if not isinstance(
        request,
        RequirementsRequest,
    ):
        raise TypeError(
            "request must be a "
            "RequirementsRequest."
        )

    if not isinstance(
        extraction,
        RequirementExtractionResult,
    ):
        raise TypeError(
            "extraction must be a "
            "RequirementExtractionResult."
        )

    if not isinstance(
        acceptance,
        AcceptanceCriteriaResult,
    ):
        raise TypeError(
            "acceptance must be an "
            "AcceptanceCriteriaResult."
        )

    if not isinstance(
        quality,
        RequirementsQualityReport,
    ):
        raise TypeError(
            "quality must be a "
            "RequirementsQualityReport."
        )

    if (
        extraction.request_id
        != request.request_id
    ):
        raise ValueError(
            "Extraction request ID mismatch."
        )

    if (
        extraction.project_id
        != request.project_id
    ):
        raise ValueError(
            "Extraction project ID mismatch."
        )

    if (
        extraction.trace_id
        != request.trace_id
    ):
        raise ValueError(
            "Extraction trace ID mismatch."
        )

    if (
        acceptance.request_id
        != request.request_id
    ):
        raise ValueError(
            "Acceptance request ID mismatch."
        )

    if (
        acceptance.extraction_id
        != extraction.extraction_id
    ):
        raise ValueError(
            "Acceptance extraction ID mismatch."
        )

    if (
        quality.request_id
        != request.request_id
    ):
        raise ValueError(
            "Quality request ID mismatch."
        )

    if (
        quality.extraction_id
        != extraction.extraction_id
    ):
        raise ValueError(
            "Quality extraction ID mismatch."
        )

    requirement_count = len(
        extraction.requirements
    )

    if (
        len(
            acceptance.assessments
        )
        != requirement_count
    ):
        raise ValueError(
            "Acceptance assessment count "
            "must match requirement count."
        )

    if (
        len(
            quality.assessments
        )
        != requirement_count
    ):
        raise ValueError(
            "Quality assessment count "
            "must match requirement count."
        )

    enriched = (
        enrich_requirements_with_acceptance_criteria(
            extraction,
            acceptance,
        )
    )

    if title is None:
        title = (
            request.document_title
            or "Business Requirements Document"
        )

    title = _required_text(
        title,
        "title",
    )

    brd = BusinessRequirementsDocument(
        request_id=(
            request.request_id
        ),
        project_id=(
            request.project_id
        ),
        trace_id=(
            request.trace_id
        ),
        title=title,
        requirements=(
            enriched.requirements
        ),
    )

    links = []

    for (
        original_requirement,
        brd_requirement,
        acceptance_assessment,
        quality_assessment,
    ) in zip(
        extraction.requirements,
        enriched.requirements,
        acceptance.assessments,
        quality.assessments,
        strict=True,
    ):
        if (
            acceptance_assessment.requirement_id
            != original_requirement.requirement_id
        ):
            raise ValueError(
                "Acceptance requirement identity "
                "or ordering mismatch."
            )

        if (
            quality_assessment.requirement_id
            != original_requirement.requirement_id
        ):
            raise ValueError(
                "Quality requirement identity "
                "or ordering mismatch."
            )

        requires_review = (
            acceptance_assessment
            .requires_human_review
            or quality_assessment.decision
            is not RequirementQualityDecision.PASS
        )

        links.append(
            GovernedBRDRequirementLink(
                sequence=(
                    original_requirement.sequence
                ),
                source_requirement_id=(
                    original_requirement
                    .requirement_id
                ),
                brd_requirement_id=(
                    brd_requirement
                    .requirement_id
                ),
                acceptance_assessment_id=(
                    acceptance_assessment
                    .assessment_id
                ),
                quality_assessment_id=(
                    quality_assessment
                    .assessment_id
                ),
                acceptance_status=(
                    acceptance_assessment
                    .status
                ),
                quality_decision=(
                    quality_assessment
                    .decision
                ),
                requires_human_review=(
                    requires_review
                ),
            )
        )

    links_tuple = tuple(
        links
    )

    decision = _decision_for_links(
        links_tuple
    )

    reasons = _reasons_for_links(
        links_tuple
    )

    return GovernedBRDResult(
        request_id=(
            request.request_id
        ),
        project_id=(
            request.project_id
        ),
        trace_id=(
            request.trace_id
        ),
        source_extraction_id=(
            extraction.extraction_id
        ),
        enriched_extraction_id=(
            enriched.extraction_id
        ),
        acceptance_result_id=(
            acceptance.result_id
        ),
        quality_report_id=(
            quality.report_id
        ),
        decision=decision,
        reasons=reasons,
        brd=brd,
        links=links_tuple,
    )
