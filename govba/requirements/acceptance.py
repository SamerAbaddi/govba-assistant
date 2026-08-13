"""Governed acceptance-criteria intelligence for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from govba.rag.models import SourceLanguage
from govba.requirements.contract import (
    RequirementItem,
    RequirementPriority,
    RequirementType,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    RequirementExtractionResult,
)


ACCEPTANCE_CRITERIA_VERSION = (
    "govba-acceptance-criteria-v1"
)

ACCEPTANCE_CRITERIA_ALGORITHM = (
    "deterministic-governed-acceptance-criteria-v1"
)


class AcceptanceCriterionOrigin(
    str,
    Enum,
):
    EXPLICIT = "explicit"
    TEMPLATE_DERIVED = "template_derived"


class AcceptanceCriterionKind(
    str,
    Enum,
):
    FUNCTIONAL_BEHAVIOR = "functional_behavior"
    SECURITY_CONTROL = "security_control"
    PERFORMANCE_TARGET = "performance_target"
    DATA_HANDLING = "data_handling"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    BUSINESS_OUTCOME = "business_outcome"
    GENERIC_VALIDATION = "generic_validation"


class AcceptanceCriteriaStatus(
    str,
    Enum,
):
    EXPLICIT = "explicit"
    DRAFT_REVIEW = "draft_review"
    NOT_APPLICABLE = "not_applicable"


_TYPE_TO_KIND = {
    RequirementType.FUNCTIONAL:
        AcceptanceCriterionKind.FUNCTIONAL_BEHAVIOR,

    RequirementType.SECURITY:
        AcceptanceCriterionKind.SECURITY_CONTROL,

    RequirementType.NON_FUNCTIONAL:
        AcceptanceCriterionKind.PERFORMANCE_TARGET,

    RequirementType.DATA:
        AcceptanceCriterionKind.DATA_HANDLING,

    RequirementType.INTEGRATION:
        AcceptanceCriterionKind.INTEGRATION,

    RequirementType.REPORTING:
        AcceptanceCriterionKind.REPORTING,

    RequirementType.COMPLIANCE:
        AcceptanceCriterionKind.COMPLIANCE,

    RequirementType.BUSINESS:
        AcceptanceCriterionKind.BUSINESS_OUTCOME,

    RequirementType.OTHER:
        AcceptanceCriterionKind.GENERIC_VALIDATION,
}


_ENGLISH_TEMPLATES = {
    AcceptanceCriterionKind.FUNCTIONAL_BEHAVIOR:
        (
            "Verify that the behavior described in the "
            "requirement can be completed successfully "
            "under defined valid conditions."
        ),

    AcceptanceCriterionKind.SECURITY_CONTROL:
        (
            "Verify that the security control stated in "
            "the requirement is enforced under the "
            "defined test conditions."
        ),

    AcceptanceCriterionKind.PERFORMANCE_TARGET:
        (
            "Verify the requirement against its stated "
            "measurable performance or service target."
        ),

    AcceptanceCriterionKind.DATA_HANDLING:
        (
            "Verify that the specified data is handled "
            "in accordance with the behavior stated in "
            "the requirement."
        ),

    AcceptanceCriterionKind.INTEGRATION:
        (
            "Verify that the specified integration "
            "successfully exchanges the required "
            "information under valid conditions."
        ),

    AcceptanceCriterionKind.REPORTING:
        (
            "Verify that the specified report, dashboard, "
            "or analytical output can be produced as "
            "stated in the requirement."
        ),

    AcceptanceCriterionKind.COMPLIANCE:
        (
            "Verify documented evidence of conformance "
            "with the law, regulation, policy, or standard "
            "identified in the requirement."
        ),

    AcceptanceCriterionKind.BUSINESS_OUTCOME:
        (
            "Verify that an agreed measurable success "
            "measure exists for the business outcome "
            "stated in the requirement."
        ),

    AcceptanceCriterionKind.GENERIC_VALIDATION:
        (
            "Verify the requirement using an agreed "
            "objective and testable success condition."
        ),
}


_ARABIC_TEMPLATES = {
    AcceptanceCriterionKind.FUNCTIONAL_BEHAVIOR:
        (
            "التحقق من إمكانية تنفيذ السلوك الموضح في "
            "المتطلب بنجاح ضمن ظروف صحيحة ومحددة."
        ),

    AcceptanceCriterionKind.SECURITY_CONTROL:
        (
            "التحقق من تطبيق الضابط الأمني المحدد في "
            "المتطلب ضمن ظروف الاختبار المعتمدة."
        ),

    AcceptanceCriterionKind.PERFORMANCE_TARGET:
        (
            "التحقق من المتطلب مقابل هدف الأداء أو "
            "مستوى الخدمة القابل للقياس والمذكور فيه."
        ),

    AcceptanceCriterionKind.DATA_HANDLING:
        (
            "التحقق من التعامل مع البيانات المحددة وفق "
            "السلوك المذكور في المتطلب."
        ),

    AcceptanceCriterionKind.INTEGRATION:
        (
            "التحقق من أن التكامل المحدد يتبادل المعلومات "
            "المطلوبة بنجاح ضمن الظروف الصحيحة."
        ),

    AcceptanceCriterionKind.REPORTING:
        (
            "التحقق من إمكانية إنتاج التقرير أو لوحة "
            "المعلومات أو المخرج التحليلي كما هو محدد."
        ),

    AcceptanceCriterionKind.COMPLIANCE:
        (
            "التحقق من وجود دليل موثق على الامتثال "
            "للقانون أو التشريع أو السياسة أو المعيار "
            "المذكور في المتطلب."
        ),

    AcceptanceCriterionKind.BUSINESS_OUTCOME:
        (
            "التحقق من وجود مقياس نجاح متفق عليه وقابل "
            "للقياس للنتيجة التجارية المحددة في المتطلب."
        ),

    AcceptanceCriterionKind.GENERIC_VALIDATION:
        (
            "التحقق من المتطلب باستخدام شرط نجاح موضوعي "
            "وقابل للاختبار ومتفق عليه."
        ),
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
        value.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One explicit or conservatively derived acceptance criterion."""

    requirement_id: str

    sequence: int

    kind: AcceptanceCriterionKind

    origin: AcceptanceCriterionOrigin

    criterion_text: str

    requires_human_review: bool

    version: str = (
        ACCEPTANCE_CRITERIA_VERSION
    )

    criterion_sha256: str = field(
        init=False
    )

    criterion_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        requirement_id = _required_text(
            self.requirement_id,
            "requirement_id",
        )

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

        if not isinstance(
            self.kind,
            AcceptanceCriterionKind,
        ):
            raise TypeError(
                "kind must be an "
                "AcceptanceCriterionKind."
            )

        if not isinstance(
            self.origin,
            AcceptanceCriterionOrigin,
        ):
            raise TypeError(
                "origin must be an "
                "AcceptanceCriterionOrigin."
            )

        if not isinstance(
            self.requires_human_review,
            bool,
        ):
            raise TypeError(
                "requires_human_review must "
                "be boolean."
            )

        if (
            self.origin
            is AcceptanceCriterionOrigin.TEMPLATE_DERIVED
            and not self.requires_human_review
        ):
            raise ValueError(
                "Template-derived criteria must "
                "require human review."
            )

        criterion_text = _required_text(
            self.criterion_text,
            "criterion_text",
        )

        criterion_hash = _sha256(
            criterion_text
        )

        payload = "\x1f".join(
            (
                self.version,
                requirement_id,
                str(
                    self.sequence
                ),
                self.kind.value,
                self.origin.value,
                criterion_hash,
                str(
                    self.requires_human_review
                ).lower(),
            )
        )

        object.__setattr__(
            self,
            "requirement_id",
            requirement_id,
        )

        object.__setattr__(
            self,
            "criterion_text",
            criterion_text,
        )

        object.__setattr__(
            self,
            "criterion_sha256",
            criterion_hash,
        )

        object.__setattr__(
            self,
            "criterion_id",
            _sha256(
                payload
            ),
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

        data: dict[
            str,
            object,
        ] = {
            "version": self.version,
            "criterion_id": (
                self.criterion_id
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "sequence": (
                self.sequence
            ),
            "kind": (
                self.kind.value
            ),
            "origin": (
                self.origin.value
            ),
            "criterion_sha256": (
                self.criterion_sha256
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
        }

        if include_text:
            data[
                "criterion_text"
            ] = self.criterion_text

        return data


@dataclass(frozen=True)
class RequirementAcceptanceAssessment:
    """Acceptance-criteria assessment for one requirement."""

    requirement_id: str

    status: AcceptanceCriteriaStatus

    criteria: tuple[
        AcceptanceCriterion,
        ...
    ]

    version: str = (
        ACCEPTANCE_CRITERIA_VERSION
    )

    assessment_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        requirement_id = _required_text(
            self.requirement_id,
            "requirement_id",
        )

        if not isinstance(
            self.status,
            AcceptanceCriteriaStatus,
        ):
            raise TypeError(
                "status must be an "
                "AcceptanceCriteriaStatus."
            )

        try:
            criteria = tuple(
                self.criteria
            )
        except TypeError as exc:
            raise TypeError(
                "criteria must be iterable."
            ) from exc

        for criterion in criteria:
            if not isinstance(
                criterion,
                AcceptanceCriterion,
            ):
                raise TypeError(
                    "criteria must contain only "
                    "AcceptanceCriterion values."
                )

            if (
                criterion.requirement_id
                != requirement_id
            ):
                raise ValueError(
                    "Criterion requirement ID mismatch."
                )

        sequences = tuple(
            criterion.sequence
            for criterion in criteria
        )

        if sequences != tuple(
            range(
                len(criteria)
            )
        ):
            raise ValueError(
                "Criterion sequences must be "
                "contiguous from zero."
            )

        if (
            self.status
            is AcceptanceCriteriaStatus.NOT_APPLICABLE
            and criteria
        ):
            raise ValueError(
                "NOT_APPLICABLE assessment "
                "cannot contain criteria."
            )

        if (
            self.status
            is not AcceptanceCriteriaStatus.NOT_APPLICABLE
            and not criteria
        ):
            raise ValueError(
                "Applicable assessment requires "
                "at least one criterion."
            )

        payload = "\x1f".join(
            (
                self.version,
                requirement_id,
                self.status.value,
                ",".join(
                    criterion.criterion_id
                    for criterion
                    in criteria
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
            "criteria",
            criteria,
        )

        object.__setattr__(
            self,
            "assessment_id",
            _sha256(
                payload
            ),
        )

    @property
    def requires_human_review(
        self,
    ) -> bool:
        return any(
            criterion.requires_human_review
            for criterion
            in self.criteria
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "assessment_id": (
                self.assessment_id
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "status": (
                self.status.value
            ),
            "criterion_count": (
                len(
                    self.criteria
                )
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "criteria": [
                criterion.to_dict()
                for criterion
                in self.criteria
            ],
        }


@dataclass(frozen=True)
class AcceptanceCriteriaResult:
    """Governed acceptance-criteria intelligence result."""

    request_id: str

    extraction_id: str

    assessments: tuple[
        RequirementAcceptanceAssessment,
        ...
    ]

    algorithm: str = (
        ACCEPTANCE_CRITERIA_ALGORITHM
    )

    version: str = (
        ACCEPTANCE_CRITERIA_VERSION
    )

    result_id: str = field(
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

        for assessment in assessments:
            if not isinstance(
                assessment,
                RequirementAcceptanceAssessment,
            ):
                raise TypeError(
                    "assessments must contain only "
                    "RequirementAcceptanceAssessment values."
                )

        assessment_ids = tuple(
            assessment.assessment_id
            for assessment
            in assessments
        )

        if (
            len(
                set(
                    assessment_ids
                )
            )
            != len(
                assessment_ids
            )
        ):
            raise ValueError(
                "Assessments must be unique."
            )

        payload = "\x1f".join(
            (
                self.version,
                request_id,
                extraction_id,
                algorithm,
                ",".join(
                    assessment_ids
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
            "result_id",
            _sha256(
                payload
            ),
        )

    @property
    def criterion_count(
        self,
    ) -> int:
        return sum(
            len(
                assessment.criteria
            )
            for assessment
            in self.assessments
        )

    @property
    def review_count(
        self,
    ) -> int:
        return sum(
            assessment.requires_human_review
            for assessment
            in self.assessments
        )

    @property
    def not_applicable_count(
        self,
    ) -> int:
        return sum(
            assessment.status
            is AcceptanceCriteriaStatus.NOT_APPLICABLE
            for assessment
            in self.assessments
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "result_id": (
                self.result_id
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
            "criterion_count": (
                self.criterion_count
            ),
            "review_count": (
                self.review_count
            ),
            "not_applicable_count": (
                self.not_applicable_count
            ),
            "assessments": [
                assessment.to_dict()
                for assessment
                in self.assessments
            ],
        }


def _criterion_kind(
    requirement: RequirementItem,
) -> AcceptanceCriterionKind:
    return _TYPE_TO_KIND.get(
        requirement.requirement_type,
        AcceptanceCriterionKind.GENERIC_VALIDATION,
    )


def _template_for(
    requirement: RequirementItem,
    language: SourceLanguage,
) -> str:
    kind = _criterion_kind(
        requirement
    )

    if language is SourceLanguage.ARABIC:
        return _ARABIC_TEMPLATES[
            kind
        ]

    if language is SourceLanguage.BILINGUAL:
        return (
            _ENGLISH_TEMPLATES[kind]
            + " / "
            + _ARABIC_TEMPLATES[kind]
        )

    return _ENGLISH_TEMPLATES[
        kind
    ]


def generate_acceptance_criteria(
    request: RequirementsRequest,
    extraction: RequirementExtractionResult,
) -> AcceptanceCriteriaResult:
    """Preserve explicit criteria or create conservative reviewable drafts."""

    if not isinstance(
        request,
        RequirementsRequest,
    ):
        raise TypeError(
            "request must be a RequirementsRequest."
        )

    if not isinstance(
        extraction,
        RequirementExtractionResult,
    ):
        raise TypeError(
            "extraction must be a "
            "RequirementExtractionResult."
        )

    if (
        extraction.request_id
        != request.request_id
    ):
        raise ValueError(
            "Extraction request ID mismatch."
        )

    assessments = []

    for requirement in extraction.requirements:
        if (
            requirement.priority
            is RequirementPriority.WONT
        ):
            assessments.append(
                RequirementAcceptanceAssessment(
                    requirement_id=(
                        requirement.requirement_id
                    ),
                    status=(
                        AcceptanceCriteriaStatus
                        .NOT_APPLICABLE
                    ),
                    criteria=(),
                )
            )

            continue

        kind = _criterion_kind(
            requirement
        )

        if requirement.acceptance_criteria:
            criteria = tuple(
                AcceptanceCriterion(
                    requirement_id=(
                        requirement.requirement_id
                    ),
                    sequence=index,
                    kind=kind,
                    origin=(
                        AcceptanceCriterionOrigin
                        .EXPLICIT
                    ),
                    criterion_text=text,
                    requires_human_review=False,
                )
                for index, text
                in enumerate(
                    requirement.acceptance_criteria
                )
            )

            status = (
                AcceptanceCriteriaStatus.EXPLICIT
            )

        else:
            criteria = (
                AcceptanceCriterion(
                    requirement_id=(
                        requirement.requirement_id
                    ),
                    sequence=0,
                    kind=kind,
                    origin=(
                        AcceptanceCriterionOrigin
                        .TEMPLATE_DERIVED
                    ),
                    criterion_text=(
                        _template_for(
                            requirement,
                            request.language,
                        )
                    ),
                    requires_human_review=True,
                ),
            )

            status = (
                AcceptanceCriteriaStatus
                .DRAFT_REVIEW
            )

        assessments.append(
            RequirementAcceptanceAssessment(
                requirement_id=(
                    requirement.requirement_id
                ),
                status=status,
                criteria=criteria,
            )
        )

    return AcceptanceCriteriaResult(
        request_id=(
            request.request_id
        ),
        extraction_id=(
            extraction.extraction_id
        ),
        assessments=tuple(
            assessments
        ),
    )


def enrich_requirements_with_acceptance_criteria(
    extraction: RequirementExtractionResult,
    criteria_result: AcceptanceCriteriaResult,
) -> RequirementExtractionResult:
    """Return a new extraction containing the assessed criteria."""

    if not isinstance(
        extraction,
        RequirementExtractionResult,
    ):
        raise TypeError(
            "extraction must be a "
            "RequirementExtractionResult."
        )

    if not isinstance(
        criteria_result,
        AcceptanceCriteriaResult,
    ):
        raise TypeError(
            "criteria_result must be an "
            "AcceptanceCriteriaResult."
        )

    if (
        criteria_result.request_id
        != extraction.request_id
    ):
        raise ValueError(
            "Acceptance result request ID mismatch."
        )

    if (
        criteria_result.extraction_id
        != extraction.extraction_id
    ):
        raise ValueError(
            "Acceptance result extraction ID mismatch."
        )

    if (
        len(
            criteria_result.assessments
        )
        != len(
            extraction.requirements
        )
    ):
        raise ValueError(
            "Acceptance assessment count must "
            "match requirement count."
        )

    enriched = []

    for requirement, assessment in zip(
        extraction.requirements,
        criteria_result.assessments,
        strict=True,
    ):
        if (
            assessment.requirement_id
            != requirement.requirement_id
        ):
            raise ValueError(
                "Acceptance assessment order or "
                "requirement identity mismatch."
            )

        criterion_texts = tuple(
            criterion.criterion_text
            for criterion
            in assessment.criteria
        )

        enriched.append(
            RequirementItem(
                request_id=(
                    requirement.request_id
                ),
                sequence=(
                    requirement.sequence
                ),
                requirement_type=(
                    requirement.requirement_type
                ),
                priority=(
                    requirement.priority
                ),
                status=(
                    requirement.status
                ),
                statement=(
                    requirement.statement
                ),
                rationale=(
                    requirement.rationale
                ),
                acceptance_criteria=(
                    criterion_texts
                ),
                dependencies=(
                    requirement.dependencies
                ),
                source_references=(
                    requirement.source_references
                ),
            )
        )

    return RequirementExtractionResult(
        request_id=(
            extraction.request_id
        ),
        project_id=(
            extraction.project_id
        ),
        trace_id=(
            extraction.trace_id
        ),
        parser_id=(
            extraction.parser_id
        ),
        requirements=tuple(
            enriched
        ),
        algorithm=(
            extraction.algorithm
        ),
    )
