"""Offline end-to-end evaluation for GovBA-GAR requirements intelligence."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from statistics import mean

from govba.requirements.acceptance import (
    AcceptanceCriteriaStatus,
    generate_acceptance_criteria,
)
from govba.requirements.contract import (
    RequirementPriority,
    RequirementType,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    extract_requirements,
)
from govba.requirements.governed_brd import (
    GovernedBRDDecision,
    build_governed_brd,
)
from govba.requirements.quality import (
    RequirementQualityDecision,
    assess_requirements_quality,
)


REQUIREMENTS_EVALUATION_VERSION = (
    "govba-requirements-evaluation-v1"
)

REQUIREMENTS_EVALUATION_ALGORITHM = (
    "deterministic-requirements-benchmark-v1"
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


def _enum_tuple(
    values,
    enum_type,
    field_name: str,
) -> tuple:
    if isinstance(
        values,
        (str, bytes),
    ):
        raise TypeError(
            f"{field_name} must be iterable."
        )

    try:
        normalized = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    for value in normalized:
        if not isinstance(
            value,
            enum_type,
        ):
            raise TypeError(
                f"{field_name} must contain only "
                f"{enum_type.__name__} values."
            )

    return normalized


def _rate(
    value: float,
    field_name: str,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(
            value
        )
        or not (
            0.0
            <= value
            <= 1.0
        )
    ):
        raise ValueError(
            f"{field_name} must be between "
            "0 and 1."
        )

    return float(
        value
    )


@dataclass(frozen=True)
class RequirementsEvaluationGoldCase:
    """Gold labels for one complete requirements-intelligence case."""

    case_id: str

    request: RequirementsRequest

    expected_requirement_types: tuple[
        RequirementType,
        ...
    ]

    expected_priorities: tuple[
        RequirementPriority,
        ...
    ]

    expected_acceptance_statuses: tuple[
        AcceptanceCriteriaStatus,
        ...
    ]

    expected_quality_decisions: tuple[
        RequirementQualityDecision,
        ...
    ]

    expected_governed_decision: (
        GovernedBRDDecision
    )

    version: str = (
        REQUIREMENTS_EVALUATION_VERSION
    )

    case_hash: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        case_id = _required_text(
            self.case_id,
            "case_id",
        )

        if not isinstance(
            self.request,
            RequirementsRequest,
        ):
            raise TypeError(
                "request must be a "
                "RequirementsRequest."
            )

        requirement_types = _enum_tuple(
            self.expected_requirement_types,
            RequirementType,
            "expected_requirement_types",
        )

        priorities = _enum_tuple(
            self.expected_priorities,
            RequirementPriority,
            "expected_priorities",
        )

        acceptance_statuses = (
            _enum_tuple(
                self.expected_acceptance_statuses,
                AcceptanceCriteriaStatus,
                "expected_acceptance_statuses",
            )
        )

        quality_decisions = (
            _enum_tuple(
                self.expected_quality_decisions,
                RequirementQualityDecision,
                "expected_quality_decisions",
            )
        )

        expected_count = len(
            requirement_types
        )

        if (
            len(
                priorities
            )
            != expected_count
            or len(
                acceptance_statuses
            )
            != expected_count
            or len(
                quality_decisions
            )
            != expected_count
        ):
            raise ValueError(
                "All requirement-level gold label "
                "tuples must have equal length."
            )

        if not isinstance(
            self.expected_governed_decision,
            GovernedBRDDecision,
        ):
            raise TypeError(
                "expected_governed_decision must be "
                "a GovernedBRDDecision."
            )

        payload = "\x1f".join(
            (
                self.version,
                case_id,
                self.request.request_id,
                ",".join(
                    value.value
                    for value
                    in requirement_types
                ),
                ",".join(
                    value.value
                    for value
                    in priorities
                ),
                ",".join(
                    value.value
                    for value
                    in acceptance_statuses
                ),
                ",".join(
                    value.value
                    for value
                    in quality_decisions
                ),
                self.expected_governed_decision.value,
            )
        )

        object.__setattr__(
            self,
            "case_id",
            case_id,
        )

        object.__setattr__(
            self,
            "expected_requirement_types",
            requirement_types,
        )

        object.__setattr__(
            self,
            "expected_priorities",
            priorities,
        )

        object.__setattr__(
            self,
            "expected_acceptance_statuses",
            acceptance_statuses,
        )

        object.__setattr__(
            self,
            "expected_quality_decisions",
            quality_decisions,
        )

        object.__setattr__(
            self,
            "case_hash",
            _sha256(
                payload
            ),
        )

    @property
    def expected_requirement_count(
        self,
    ) -> int:
        return len(
            self.expected_requirement_types
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Privacy-safe gold serialization."""

        return {
            "version": (
                self.version
            ),
            "case_id": (
                self.case_id
            ),
            "case_hash": (
                self.case_hash
            ),
            "request_id": (
                self.request.request_id
            ),
            "expected_requirement_count": (
                self.expected_requirement_count
            ),
            "expected_requirement_types": [
                value.value
                for value
                in self.expected_requirement_types
            ],
            "expected_priorities": [
                value.value
                for value
                in self.expected_priorities
            ],
            "expected_acceptance_statuses": [
                value.value
                for value
                in self.expected_acceptance_statuses
            ],
            "expected_quality_decisions": [
                value.value
                for value
                in self.expected_quality_decisions
            ],
            "expected_governed_decision": (
                self.expected_governed_decision
                .value
            ),
        }


@dataclass(frozen=True)
class RequirementsEvaluationCaseResult:
    """Privacy-safe result for one requirements benchmark case."""

    case_id: str

    actual_requirement_types: tuple[
        RequirementType,
        ...
    ]

    actual_priorities: tuple[
        RequirementPriority,
        ...
    ]

    actual_acceptance_statuses: tuple[
        AcceptanceCriteriaStatus,
        ...
    ]

    actual_quality_decisions: tuple[
        RequirementQualityDecision,
        ...
    ]

    actual_governed_decision: (
        GovernedBRDDecision
    )

    requirement_count_correct: bool

    types_correct: bool

    priorities_correct: bool

    acceptance_correct: bool

    quality_correct: bool

    governed_decision_correct: bool

    case_success: bool

    latency_ms: float

    version: str = (
        REQUIREMENTS_EVALUATION_VERSION
    )

    result_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        case_id = _required_text(
            self.case_id,
            "case_id",
        )

        requirement_types = _enum_tuple(
            self.actual_requirement_types,
            RequirementType,
            "actual_requirement_types",
        )

        priorities = _enum_tuple(
            self.actual_priorities,
            RequirementPriority,
            "actual_priorities",
        )

        acceptance_statuses = (
            _enum_tuple(
                self.actual_acceptance_statuses,
                AcceptanceCriteriaStatus,
                "actual_acceptance_statuses",
            )
        )

        quality_decisions = (
            _enum_tuple(
                self.actual_quality_decisions,
                RequirementQualityDecision,
                "actual_quality_decisions",
            )
        )

        count = len(
            requirement_types
        )

        if (
            len(
                priorities
            )
            != count
            or len(
                acceptance_statuses
            )
            != count
            or len(
                quality_decisions
            )
            != count
        ):
            raise ValueError(
                "All actual requirement-level tuples "
                "must have equal length."
            )

        if not isinstance(
            self.actual_governed_decision,
            GovernedBRDDecision,
        ):
            raise TypeError(
                "actual_governed_decision must be "
                "a GovernedBRDDecision."
            )

        bool_fields = (
            "requirement_count_correct",
            "types_correct",
            "priorities_correct",
            "acceptance_correct",
            "quality_correct",
            "governed_decision_correct",
            "case_success",
        )

        for field_name in bool_fields:
            if not isinstance(
                getattr(
                    self,
                    field_name,
                ),
                bool,
            ):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        expected_success = all(
            getattr(
                self,
                field_name,
            )
            for field_name
            in bool_fields[:-1]
        )

        if (
            self.case_success
            != expected_success
        ):
            raise ValueError(
                "case_success does not match "
                "component correctness."
            )

        if (
            isinstance(
                self.latency_ms,
                bool,
            )
            or not isinstance(
                self.latency_ms,
                (int, float),
            )
            or not math.isfinite(
                self.latency_ms
            )
            or self.latency_ms < 0
        ):
            raise ValueError(
                "latency_ms must be non-negative."
            )

        payload = "\x1f".join(
            (
                self.version,
                case_id,
                ",".join(
                    value.value
                    for value
                    in requirement_types
                ),
                ",".join(
                    value.value
                    for value
                    in priorities
                ),
                ",".join(
                    value.value
                    for value
                    in acceptance_statuses
                ),
                ",".join(
                    value.value
                    for value
                    in quality_decisions
                ),
                self.actual_governed_decision.value,
                *(
                    str(
                        getattr(
                            self,
                            field_name,
                        )
                    ).lower()
                    for field_name
                    in bool_fields
                ),
            )
        )

        object.__setattr__(
            self,
            "case_id",
            case_id,
        )

        object.__setattr__(
            self,
            "actual_requirement_types",
            requirement_types,
        )

        object.__setattr__(
            self,
            "actual_priorities",
            priorities,
        )

        object.__setattr__(
            self,
            "actual_acceptance_statuses",
            acceptance_statuses,
        )

        object.__setattr__(
            self,
            "actual_quality_decisions",
            quality_decisions,
        )

        object.__setattr__(
            self,
            "latency_ms",
            float(
                self.latency_ms
            ),
        )

        object.__setattr__(
            self,
            "result_id",
            _sha256(
                payload
            ),
        )

    @property
    def actual_requirement_count(
        self,
    ) -> int:
        return len(
            self.actual_requirement_types
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
            "case_id": (
                self.case_id
            ),
            "actual_requirement_count": (
                self.actual_requirement_count
            ),
            "actual_requirement_types": [
                value.value
                for value
                in self.actual_requirement_types
            ],
            "actual_priorities": [
                value.value
                for value
                in self.actual_priorities
            ],
            "actual_acceptance_statuses": [
                value.value
                for value
                in self.actual_acceptance_statuses
            ],
            "actual_quality_decisions": [
                value.value
                for value
                in self.actual_quality_decisions
            ],
            "actual_governed_decision": (
                self.actual_governed_decision
                .value
            ),
            "requirement_count_correct": (
                self.requirement_count_correct
            ),
            "types_correct": (
                self.types_correct
            ),
            "priorities_correct": (
                self.priorities_correct
            ),
            "acceptance_correct": (
                self.acceptance_correct
            ),
            "quality_correct": (
                self.quality_correct
            ),
            "governed_decision_correct": (
                self.governed_decision_correct
            ),
            "case_success": (
                self.case_success
            ),
            "latency_ms": (
                self.latency_ms
            ),
        }


@dataclass(frozen=True)
class RequirementsEvaluationMetrics:
    """Aggregate end-to-end requirements benchmark metrics."""

    case_count: int

    requirement_count_accuracy: float

    type_sequence_accuracy: float

    priority_sequence_accuracy: float

    acceptance_sequence_accuracy: float

    quality_sequence_accuracy: float

    governed_decision_accuracy: float

    end_to_end_success_rate: float

    mean_latency_ms: float

    version: str = (
        REQUIREMENTS_EVALUATION_VERSION
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.case_count,
                bool,
            )
            or not isinstance(
                self.case_count,
                int,
            )
            or self.case_count < 1
        ):
            raise ValueError(
                "case_count must be positive."
            )

        rate_fields = (
            "requirement_count_accuracy",
            "type_sequence_accuracy",
            "priority_sequence_accuracy",
            "acceptance_sequence_accuracy",
            "quality_sequence_accuracy",
            "governed_decision_accuracy",
            "end_to_end_success_rate",
        )

        for field_name in rate_fields:
            object.__setattr__(
                self,
                field_name,
                _rate(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        if (
            isinstance(
                self.mean_latency_ms,
                bool,
            )
            or not isinstance(
                self.mean_latency_ms,
                (int, float),
            )
            or not math.isfinite(
                self.mean_latency_ms
            )
            or self.mean_latency_ms < 0
        ):
            raise ValueError(
                "mean_latency_ms must be "
                "non-negative."
            )

        object.__setattr__(
            self,
            "mean_latency_ms",
            float(
                self.mean_latency_ms
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "case_count": (
                self.case_count
            ),
            "requirement_count_accuracy": (
                self.requirement_count_accuracy
            ),
            "type_sequence_accuracy": (
                self.type_sequence_accuracy
            ),
            "priority_sequence_accuracy": (
                self.priority_sequence_accuracy
            ),
            "acceptance_sequence_accuracy": (
                self.acceptance_sequence_accuracy
            ),
            "quality_sequence_accuracy": (
                self.quality_sequence_accuracy
            ),
            "governed_decision_accuracy": (
                self.governed_decision_accuracy
            ),
            "end_to_end_success_rate": (
                self.end_to_end_success_rate
            ),
            "mean_latency_ms": (
                self.mean_latency_ms
            ),
        }


@dataclass(frozen=True)
class RequirementsBenchmarkResult:
    """Complete offline requirements-intelligence benchmark."""

    cases: tuple[
        RequirementsEvaluationCaseResult,
        ...
    ]

    metrics: RequirementsEvaluationMetrics

    algorithm: str = (
        REQUIREMENTS_EVALUATION_ALGORITHM
    )

    version: str = (
        REQUIREMENTS_EVALUATION_VERSION
    )

    benchmark_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        try:
            cases = tuple(
                self.cases
            )
        except TypeError as exc:
            raise TypeError(
                "cases must be iterable."
            ) from exc

        if not cases:
            raise ValueError(
                "At least one case result "
                "is required."
            )

        for case in cases:
            if not isinstance(
                case,
                RequirementsEvaluationCaseResult,
            ):
                raise TypeError(
                    "cases must contain only "
                    "RequirementsEvaluationCaseResult values."
                )

        case_ids = tuple(
            case.case_id
            for case
            in cases
        )

        if (
            len(
                set(
                    case_ids
                )
            )
            != len(
                case_ids
            )
        ):
            raise ValueError(
                "Case IDs must be unique."
            )

        if not isinstance(
            self.metrics,
            RequirementsEvaluationMetrics,
        ):
            raise TypeError(
                "metrics must be "
                "RequirementsEvaluationMetrics."
            )

        if (
            self.metrics.case_count
            != len(
                cases
            )
        ):
            raise ValueError(
                "Metric case_count must match "
                "case results."
            )

        algorithm = _required_text(
            self.algorithm,
            "algorithm",
        )

        payload = "\x1f".join(
            (
                self.version,
                algorithm,
                ",".join(
                    case.result_id
                    for case
                    in cases
                ),
            )
        )

        object.__setattr__(
            self,
            "cases",
            cases,
        )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "benchmark_id",
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
            "benchmark_id": (
                self.benchmark_id
            ),
            "algorithm": (
                self.algorithm
            ),
            "metrics": (
                self.metrics.to_dict()
            ),
            "cases": [
                case.to_dict()
                for case
                in self.cases
            ],
        }


def evaluate_requirements_intelligence(
    cases,
) -> RequirementsBenchmarkResult:
    """Evaluate the complete deterministic requirements pipeline."""

    if isinstance(
        cases,
        (str, bytes),
    ):
        raise TypeError(
            "cases must contain gold cases."
        )

    try:
        gold_cases = tuple(
            cases
        )
    except TypeError as exc:
        raise TypeError(
            "cases must be iterable."
        ) from exc

    if not gold_cases:
        raise ValueError(
            "At least one gold case is required."
        )

    for case in gold_cases:
        if not isinstance(
            case,
            RequirementsEvaluationGoldCase,
        ):
            raise TypeError(
                "cases must contain only "
                "RequirementsEvaluationGoldCase values."
            )

    case_ids = tuple(
        case.case_id
        for case
        in gold_cases
    )

    if (
        len(
            set(
                case_ids
            )
        )
        != len(
            case_ids
        )
    ):
        raise ValueError(
            "Gold case IDs must be unique."
        )

    outputs = []

    for case in gold_cases:
        started = time.perf_counter()

        extraction = (
            extract_requirements(
                case.request
            )
        )

        acceptance = (
            generate_acceptance_criteria(
                case.request,
                extraction,
            )
        )

        quality = (
            assess_requirements_quality(
                extraction
            )
        )

        governed = (
            build_governed_brd(
                case.request,
                extraction,
                acceptance,
                quality,
            )
        )

        actual_types = tuple(
            requirement.requirement_type
            for requirement
            in extraction.requirements
        )

        actual_priorities = tuple(
            requirement.priority
            for requirement
            in extraction.requirements
        )

        actual_acceptance_statuses = tuple(
            assessment.status
            for assessment
            in acceptance.assessments
        )

        actual_quality_decisions = tuple(
            assessment.decision
            for assessment
            in quality.assessments
        )

        requirement_count_correct = (
            len(
                extraction.requirements
            )
            == case.expected_requirement_count
        )

        types_correct = (
            actual_types
            == case.expected_requirement_types
        )

        priorities_correct = (
            actual_priorities
            == case.expected_priorities
        )

        acceptance_correct = (
            actual_acceptance_statuses
            == case.expected_acceptance_statuses
        )

        quality_correct = (
            actual_quality_decisions
            == case.expected_quality_decisions
        )

        governed_decision_correct = (
            governed.decision
            is case.expected_governed_decision
        )

        case_success = all(
            (
                requirement_count_correct,
                types_correct,
                priorities_correct,
                acceptance_correct,
                quality_correct,
                governed_decision_correct,
            )
        )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        outputs.append(
            RequirementsEvaluationCaseResult(
                case_id=(
                    case.case_id
                ),
                actual_requirement_types=(
                    actual_types
                ),
                actual_priorities=(
                    actual_priorities
                ),
                actual_acceptance_statuses=(
                    actual_acceptance_statuses
                ),
                actual_quality_decisions=(
                    actual_quality_decisions
                ),
                actual_governed_decision=(
                    governed.decision
                ),
                requirement_count_correct=(
                    requirement_count_correct
                ),
                types_correct=(
                    types_correct
                ),
                priorities_correct=(
                    priorities_correct
                ),
                acceptance_correct=(
                    acceptance_correct
                ),
                quality_correct=(
                    quality_correct
                ),
                governed_decision_correct=(
                    governed_decision_correct
                ),
                case_success=(
                    case_success
                ),
                latency_ms=(
                    latency_ms
                ),
            )
        )

    results = tuple(
        outputs
    )

    metrics = RequirementsEvaluationMetrics(
        case_count=len(
            results
        ),

        requirement_count_accuracy=mean(
            float(
                result.requirement_count_correct
            )
            for result
            in results
        ),

        type_sequence_accuracy=mean(
            float(
                result.types_correct
            )
            for result
            in results
        ),

        priority_sequence_accuracy=mean(
            float(
                result.priorities_correct
            )
            for result
            in results
        ),

        acceptance_sequence_accuracy=mean(
            float(
                result.acceptance_correct
            )
            for result
            in results
        ),

        quality_sequence_accuracy=mean(
            float(
                result.quality_correct
            )
            for result
            in results
        ),

        governed_decision_accuracy=mean(
            float(
                result.governed_decision_correct
            )
            for result
            in results
        ),

        end_to_end_success_rate=mean(
            float(
                result.case_success
            )
            for result
            in results
        ),

        mean_latency_ms=mean(
            result.latency_ms
            for result
            in results
        ),
    )

    return RequirementsBenchmarkResult(
        cases=results,
        metrics=metrics,
    )
