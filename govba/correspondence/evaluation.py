"""Offline end-to-end evaluation for GovBA-GAR correspondence intelligence."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean

from govba.correspondence.actions import (
    extract_correspondence_actions,
)
from govba.correspondence.briefing import (
    CorrespondenceBriefingDecision,
    CorrespondenceBriefingPolicy,
    build_governed_correspondence_briefing,
)
from govba.correspondence.classification import (
    classify_correspondence_intent,
    parse_correspondence,
)
from govba.correspondence.contract import (
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.correspondence.priority import (
    assess_correspondence_priority,
    extract_correspondence_deadlines,
)


CORRESPONDENCE_EVALUATION_VERSION = (
    "govba-correspondence-evaluation-v1"
)


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


def _nonnegative_integer(
    value: int,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a "
            "non-negative integer."
        )

    return value


def _rate(
    value: float,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"{field_name} must be "
            "between 0 and 1."
        )

    return float(value)


def _aware_datetime(
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
        value.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CorrespondenceEvaluationGoldCase:
    """Gold labels for one correspondence evaluation case."""

    case_id: str

    request: CorrespondenceRequest

    assessed_at: datetime

    expected_intent: CorrespondenceIntent

    expected_priority: CorrespondencePriority

    expected_action_count: int

    expected_commitment_count: int

    expected_deadline_count: int

    expected_briefing_decision: (
        CorrespondenceBriefingDecision
    )

    version: str = (
        CORRESPONDENCE_EVALUATION_VERSION
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
            CorrespondenceRequest,
        ):
            raise TypeError(
                "request must be a "
                "CorrespondenceRequest."
            )

        assessed_at = _aware_datetime(
            self.assessed_at,
            "assessed_at",
        )

        if not isinstance(
            self.expected_intent,
            CorrespondenceIntent,
        ):
            raise TypeError(
                "expected_intent must be a "
                "CorrespondenceIntent."
            )

        if not isinstance(
            self.expected_priority,
            CorrespondencePriority,
        ):
            raise TypeError(
                "expected_priority must be a "
                "CorrespondencePriority."
            )

        for field_name in (
            "expected_action_count",
            "expected_commitment_count",
            "expected_deadline_count",
        ):
            _nonnegative_integer(
                getattr(
                    self,
                    field_name,
                ),
                field_name,
            )

        if (
            self.expected_commitment_count
            > self.expected_action_count
        ):
            raise ValueError(
                "expected_commitment_count cannot "
                "exceed expected_action_count."
            )

        if not isinstance(
            self.expected_briefing_decision,
            CorrespondenceBriefingDecision,
        ):
            raise TypeError(
                "expected_briefing_decision must be "
                "a CorrespondenceBriefingDecision."
            )

        payload = "\x1f".join(
            (
                self.version,
                case_id,
                self.request.request_id,
                assessed_at.isoformat(),
                self.expected_intent.value,
                self.expected_priority.value,
                str(
                    self.expected_action_count
                ),
                str(
                    self.expected_commitment_count
                ),
                str(
                    self.expected_deadline_count
                ),
                self.expected_briefing_decision.value,
            )
        )

        object.__setattr__(
            self,
            "case_id",
            case_id,
        )

        object.__setattr__(
            self,
            "assessed_at",
            assessed_at,
        )

        object.__setattr__(
            self,
            "case_hash",
            _sha256(payload),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "case_id": self.case_id,
            "case_hash": self.case_hash,
            "request_id": (
                self.request.request_id
            ),
            "assessed_at": (
                self.assessed_at.isoformat()
            ),
            "expected_intent": (
                self.expected_intent.value
            ),
            "expected_priority": (
                self.expected_priority.value
            ),
            "expected_action_count": (
                self.expected_action_count
            ),
            "expected_commitment_count": (
                self.expected_commitment_count
            ),
            "expected_deadline_count": (
                self.expected_deadline_count
            ),
            "expected_briefing_decision": (
                self.expected_briefing_decision.value
            ),
        }


@dataclass(frozen=True)
class CorrespondenceEvaluationCaseResult:
    """Privacy-safe result for one correspondence case."""

    case_id: str

    actual_intent: CorrespondenceIntent

    actual_priority: CorrespondencePriority

    actual_action_count: int

    actual_commitment_count: int

    actual_deadline_count: int

    actual_briefing_decision: (
        CorrespondenceBriefingDecision
    )

    intent_correct: bool

    priority_correct: bool

    action_count_correct: bool

    commitment_count_correct: bool

    deadline_count_correct: bool

    briefing_correct: bool

    case_success: bool

    latency_ms: float

    version: str = (
        CORRESPONDENCE_EVALUATION_VERSION
    )

    result_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        case_id = _required_text(
            self.case_id,
            "case_id",
        )

        if not isinstance(
            self.actual_intent,
            CorrespondenceIntent,
        ):
            raise TypeError(
                "actual_intent must be a "
                "CorrespondenceIntent."
            )

        if not isinstance(
            self.actual_priority,
            CorrespondencePriority,
        ):
            raise TypeError(
                "actual_priority must be a "
                "CorrespondencePriority."
            )

        for field_name in (
            "actual_action_count",
            "actual_commitment_count",
            "actual_deadline_count",
        ):
            _nonnegative_integer(
                getattr(
                    self,
                    field_name,
                ),
                field_name,
            )

        if not isinstance(
            self.actual_briefing_decision,
            CorrespondenceBriefingDecision,
        ):
            raise TypeError(
                "actual_briefing_decision must be "
                "a CorrespondenceBriefingDecision."
            )

        bool_fields = (
            "intent_correct",
            "priority_correct",
            "action_count_correct",
            "commitment_count_correct",
            "deadline_count_correct",
            "briefing_correct",
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
                self.actual_intent.value,
                self.actual_priority.value,
                str(
                    self.actual_action_count
                ),
                str(
                    self.actual_commitment_count
                ),
                str(
                    self.actual_deadline_count
                ),
                self.actual_briefing_decision.value,
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
            "latency_ms",
            float(
                self.latency_ms
            ),
        )

        object.__setattr__(
            self,
            "result_id",
            _sha256(payload),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "result_id": self.result_id,
            "case_id": self.case_id,
            "actual_intent": (
                self.actual_intent.value
            ),
            "actual_priority": (
                self.actual_priority.value
            ),
            "actual_action_count": (
                self.actual_action_count
            ),
            "actual_commitment_count": (
                self.actual_commitment_count
            ),
            "actual_deadline_count": (
                self.actual_deadline_count
            ),
            "actual_briefing_decision": (
                self.actual_briefing_decision.value
            ),
            "intent_correct": (
                self.intent_correct
            ),
            "priority_correct": (
                self.priority_correct
            ),
            "action_count_correct": (
                self.action_count_correct
            ),
            "commitment_count_correct": (
                self.commitment_count_correct
            ),
            "deadline_count_correct": (
                self.deadline_count_correct
            ),
            "briefing_correct": (
                self.briefing_correct
            ),
            "case_success": (
                self.case_success
            ),
            "latency_ms": (
                self.latency_ms
            ),
        }


@dataclass(frozen=True)
class CorrespondenceEvaluationMetrics:
    """Aggregate correspondence benchmark metrics."""

    case_count: int

    intent_accuracy: float

    priority_accuracy: float

    action_count_accuracy: float

    commitment_count_accuracy: float

    deadline_count_accuracy: float

    briefing_accuracy: float

    end_to_end_success_rate: float

    mean_latency_ms: float

    version: str = (
        CORRESPONDENCE_EVALUATION_VERSION
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

        for field_name in (
            "intent_accuracy",
            "priority_accuracy",
            "action_count_accuracy",
            "commitment_count_accuracy",
            "deadline_count_accuracy",
            "briefing_accuracy",
            "end_to_end_success_rate",
        ):
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
            "version": self.version,
            "case_count": self.case_count,
            "intent_accuracy": (
                self.intent_accuracy
            ),
            "priority_accuracy": (
                self.priority_accuracy
            ),
            "action_count_accuracy": (
                self.action_count_accuracy
            ),
            "commitment_count_accuracy": (
                self.commitment_count_accuracy
            ),
            "deadline_count_accuracy": (
                self.deadline_count_accuracy
            ),
            "briefing_accuracy": (
                self.briefing_accuracy
            ),
            "end_to_end_success_rate": (
                self.end_to_end_success_rate
            ),
            "mean_latency_ms": (
                self.mean_latency_ms
            ),
        }


@dataclass(frozen=True)
class CorrespondenceBenchmarkResult:
    """Complete offline correspondence benchmark."""

    cases: tuple[
        CorrespondenceEvaluationCaseResult,
        ...
    ]

    metrics: CorrespondenceEvaluationMetrics

    version: str = (
        CORRESPONDENCE_EVALUATION_VERSION
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
                CorrespondenceEvaluationCaseResult,
            ):
                raise TypeError(
                    "cases must contain only "
                    "CorrespondenceEvaluationCaseResult values."
                )

        if (
            len(
                {
                    case.case_id
                    for case in cases
                }
            )
            != len(cases)
        ):
            raise ValueError(
                "Case IDs must be unique."
            )

        if not isinstance(
            self.metrics,
            CorrespondenceEvaluationMetrics,
        ):
            raise TypeError(
                "metrics must be "
                "CorrespondenceEvaluationMetrics."
            )

        if (
            self.metrics.case_count
            != len(cases)
        ):
            raise ValueError(
                "Metric case_count must match "
                "case results."
            )

        payload = "\x1f".join(
            (
                self.version,
                ",".join(
                    case.result_id
                    for case in cases
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
            "benchmark_id",
            _sha256(payload),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "benchmark_id": (
                self.benchmark_id
            ),
            "metrics": (
                self.metrics.to_dict()
            ),
            "cases": [
                case.to_dict()
                for case in self.cases
            ],
        }


def evaluate_correspondence_intelligence(
    cases: tuple[
        CorrespondenceEvaluationGoldCase,
        ...
    ],
    *,
    briefing_policy: (
        CorrespondenceBriefingPolicy
        | None
    ) = None,
) -> CorrespondenceBenchmarkResult:
    """Evaluate the complete deterministic correspondence pipeline."""

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
            CorrespondenceEvaluationGoldCase,
        ):
            raise TypeError(
                "cases must contain only "
                "CorrespondenceEvaluationGoldCase values."
            )

    if (
        len(
            {
                case.case_id
                for case in gold_cases
            }
        )
        != len(gold_cases)
    ):
        raise ValueError(
            "Gold case IDs must be unique."
        )

    if (
        briefing_policy is not None
        and not isinstance(
            briefing_policy,
            CorrespondenceBriefingPolicy,
        )
    ):
        raise TypeError(
            "briefing_policy must be a "
            "CorrespondenceBriefingPolicy or None."
        )

    outputs = []

    for case in gold_cases:
        started = time.perf_counter()

        parsed = parse_correspondence(
            case.request
        )

        classification = (
            classify_correspondence_intent(
                case.request,
                parsed=parsed,
            )
        )

        actions = (
            extract_correspondence_actions(
                case.request,
                parsed=parsed,
            )
        )

        deadlines = (
            extract_correspondence_deadlines(
                case.request,
                parsed=parsed,
            )
        )

        priority = (
            assess_correspondence_priority(
                case.request,
                classification,
                assessed_at=(
                    case.assessed_at
                ),
                parsed=parsed,
            )
        )

        briefing = (
            build_governed_correspondence_briefing(
                case.request,
                classification,
                actions,
                deadlines,
                priority,
                policy=briefing_policy,
            )
        )

        intent_correct = (
            classification.intent
            is case.expected_intent
        )

        priority_correct = (
            priority.priority
            is case.expected_priority
        )

        action_count_correct = (
            actions.action_count
            == case.expected_action_count
        )

        commitment_count_correct = (
            actions.commitment_count
            == case.expected_commitment_count
        )

        deadline_count_correct = (
            len(
                deadlines.deadlines
            )
            == case.expected_deadline_count
        )

        briefing_correct = (
            briefing.decision
            is case.expected_briefing_decision
        )

        case_success = all(
            (
                intent_correct,
                priority_correct,
                action_count_correct,
                commitment_count_correct,
                deadline_count_correct,
                briefing_correct,
            )
        )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        outputs.append(
            CorrespondenceEvaluationCaseResult(
                case_id=case.case_id,
                actual_intent=(
                    classification.intent
                ),
                actual_priority=(
                    priority.priority
                ),
                actual_action_count=(
                    actions.action_count
                ),
                actual_commitment_count=(
                    actions.commitment_count
                ),
                actual_deadline_count=len(
                    deadlines.deadlines
                ),
                actual_briefing_decision=(
                    briefing.decision
                ),
                intent_correct=(
                    intent_correct
                ),
                priority_correct=(
                    priority_correct
                ),
                action_count_correct=(
                    action_count_correct
                ),
                commitment_count_correct=(
                    commitment_count_correct
                ),
                deadline_count_correct=(
                    deadline_count_correct
                ),
                briefing_correct=(
                    briefing_correct
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

    metrics = CorrespondenceEvaluationMetrics(
        case_count=len(
            results
        ),
        intent_accuracy=mean(
            float(
                result.intent_correct
            )
            for result in results
        ),
        priority_accuracy=mean(
            float(
                result.priority_correct
            )
            for result in results
        ),
        action_count_accuracy=mean(
            float(
                result.action_count_correct
            )
            for result in results
        ),
        commitment_count_accuracy=mean(
            float(
                result.commitment_count_correct
            )
            for result in results
        ),
        deadline_count_accuracy=mean(
            float(
                result.deadline_count_correct
            )
            for result in results
        ),
        briefing_accuracy=mean(
            float(
                result.briefing_correct
            )
            for result in results
        ),
        end_to_end_success_rate=mean(
            float(
                result.case_success
            )
            for result in results
        ),
        mean_latency_ms=mean(
            result.latency_ms
            for result in results
        ),
    )

    return CorrespondenceBenchmarkResult(
        cases=results,
        metrics=metrics,
    )
