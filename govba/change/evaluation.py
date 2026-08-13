"""Offline evaluation framework for GovBA-GAR change intelligence."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from statistics import mean

from govba.change.briefing import (
    ChangeBriefingDecision,
    build_governed_change_briefing,
)
from govba.change.clause_detection import (
    DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD,
    detect_clause_changes,
)
from govba.change.contract import (
    PolicyChangeImpact,
)
from govba.change.significance import (
    classify_change_significance,
)
from govba.change.version_matching import (
    DocumentVersionMatchDecision,
    match_document_versions,
)
from govba.rag.evidence import EvidenceChunk
from govba.rag.models import AuthoritativeSource


CHANGE_INTELLIGENCE_EVALUATION_VERSION = (
    "govba-change-intelligence-evaluation-v1"
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


def _validate_rate(
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


def _prepare_chunks(
    chunks: tuple[
        EvidenceChunk,
        ...
    ],
    *,
    document_id: str,
    field_name: str,
) -> tuple[
    EvidenceChunk,
    ...
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
        values = tuple(chunks)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    for chunk in values:
        if not isinstance(
            chunk,
            EvidenceChunk,
        ):
            raise TypeError(
                f"{field_name} must contain only "
                "EvidenceChunk values."
            )

        if chunk.document_id != document_id:
            raise ValueError(
                f"{field_name} contains evidence "
                "from the wrong document."
            )

    return values


@dataclass(frozen=True)
class ChangeIntelligenceGoldCase:
    """One deterministic gold case for the full change pipeline."""

    case_id: str

    baseline_source: AuthoritativeSource

    candidate_source: AuthoritativeSource

    baseline_chunks: tuple[
        EvidenceChunk,
        ...
    ]

    candidate_chunks: tuple[
        EvidenceChunk,
        ...
    ]

    expected_match_decision: (
        DocumentVersionMatchDecision
    )

    expected_added: int | None = None

    expected_removed: int | None = None

    expected_modified: int | None = None

    expected_unchanged: int | None = None

    expected_maximum_impact: (
        PolicyChangeImpact | None
    ) = None

    expected_briefing_decision: (
        ChangeBriefingDecision | None
    ) = None

    similarity_threshold: float = (
        DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD
    )

    version: str = (
        CHANGE_INTELLIGENCE_EVALUATION_VERSION
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
            self.baseline_source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "baseline_source must be an "
                "AuthoritativeSource."
            )

        if not isinstance(
            self.candidate_source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "candidate_source must be an "
                "AuthoritativeSource."
            )

        baseline_chunks = _prepare_chunks(
            self.baseline_chunks,
            document_id=(
                self.baseline_source.document_id
            ),
            field_name="baseline_chunks",
        )

        candidate_chunks = _prepare_chunks(
            self.candidate_chunks,
            document_id=(
                self.candidate_source.document_id
            ),
            field_name="candidate_chunks",
        )

        if not isinstance(
            self.expected_match_decision,
            DocumentVersionMatchDecision,
        ):
            raise TypeError(
                "expected_match_decision must be a "
                "DocumentVersionMatchDecision."
            )

        if (
            isinstance(
                self.similarity_threshold,
                bool,
            )
            or not isinstance(
                self.similarity_threshold,
                (int, float),
            )
            or not math.isfinite(
                self.similarity_threshold
            )
            or not (
                0.0
                < self.similarity_threshold
                <= 1.0
            )
        ):
            raise ValueError(
                "similarity_threshold must be "
                "greater than 0 and at most 1."
            )

        downstream_values = (
            self.expected_added,
            self.expected_removed,
            self.expected_modified,
            self.expected_unchanged,
            self.expected_maximum_impact,
            self.expected_briefing_decision,
        )

        if (
            self.expected_match_decision
            is DocumentVersionMatchDecision.ALLOW
        ):
            count_fields = (
                "expected_added",
                "expected_removed",
                "expected_modified",
                "expected_unchanged",
            )

            for field_name in count_fields:
                value = getattr(
                    self,
                    field_name,
                )

                if value is None:
                    raise ValueError(
                        f"{field_name} is required "
                        "for ALLOW gold cases."
                    )

                _nonnegative_integer(
                    value,
                    field_name,
                )

            if not isinstance(
                self.expected_maximum_impact,
                PolicyChangeImpact,
            ):
                raise TypeError(
                    "expected_maximum_impact is "
                    "required for ALLOW cases."
                )

            if not isinstance(
                self.expected_briefing_decision,
                ChangeBriefingDecision,
            ):
                raise TypeError(
                    "expected_briefing_decision is "
                    "required for ALLOW cases."
                )

        elif any(
            value is not None
            for value in downstream_values
        ):
            raise ValueError(
                "Downstream expectations must be "
                "None for REVIEW or REJECT cases."
            )

        payload = "\x1f".join(
            (
                self.version,
                case_id,
                self.baseline_source.document_id,
                self.baseline_source.content_hash,
                self.candidate_source.document_id,
                self.candidate_source.content_hash,
                self.expected_match_decision.value,
                str(self.expected_added),
                str(self.expected_removed),
                str(self.expected_modified),
                str(self.expected_unchanged),
                (
                    self.expected_maximum_impact.value
                    if self.expected_maximum_impact
                    else ""
                ),
                (
                    self.expected_briefing_decision.value
                    if self.expected_briefing_decision
                    else ""
                ),
                f"{float(self.similarity_threshold):.12f}",
            )
        )

        object.__setattr__(
            self,
            "case_id",
            case_id,
        )

        object.__setattr__(
            self,
            "baseline_chunks",
            baseline_chunks,
        )

        object.__setattr__(
            self,
            "candidate_chunks",
            candidate_chunks,
        )

        object.__setattr__(
            self,
            "similarity_threshold",
            float(
                self.similarity_threshold
            ),
        )

        object.__setattr__(
            self,
            "case_hash",
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )

    @property
    def expects_downstream_analysis(
        self,
    ) -> bool:
        return (
            self.expected_match_decision
            is DocumentVersionMatchDecision.ALLOW
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize gold labels without clause text or URLs."""

        return {
            "version": self.version,
            "case_id": self.case_id,
            "case_hash": self.case_hash,
            "baseline_document_id": (
                self.baseline_source.document_id
            ),
            "candidate_document_id": (
                self.candidate_source.document_id
            ),
            "expected_match_decision": (
                self.expected_match_decision.value
            ),
            "expected_added": self.expected_added,
            "expected_removed": self.expected_removed,
            "expected_modified": (
                self.expected_modified
            ),
            "expected_unchanged": (
                self.expected_unchanged
            ),
            "expected_maximum_impact": (
                self.expected_maximum_impact.value
                if self.expected_maximum_impact
                else None
            ),
            "expected_briefing_decision": (
                self.expected_briefing_decision.value
                if self.expected_briefing_decision
                else None
            ),
            "similarity_threshold": (
                self.similarity_threshold
            ),
        }


@dataclass(frozen=True)
class ChangeIntelligenceCaseResult:
    """Privacy-safe result for one change benchmark case."""

    case_id: str

    expected_match_decision: (
        DocumentVersionMatchDecision
    )

    actual_match_decision: (
        DocumentVersionMatchDecision
    )

    match_correct: bool

    change_counts_correct: bool | None

    impact_correct: bool | None

    briefing_correct: bool | None

    actual_added: int | None

    actual_removed: int | None

    actual_modified: int | None

    actual_unchanged: int | None

    actual_maximum_impact: (
        PolicyChangeImpact | None
    )

    actual_briefing_decision: (
        ChangeBriefingDecision | None
    )

    case_success: bool

    latency_ms: float

    version: str = (
        CHANGE_INTELLIGENCE_EVALUATION_VERSION
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
            self.expected_match_decision,
            DocumentVersionMatchDecision,
        ):
            raise TypeError(
                "expected_match_decision must be a "
                "DocumentVersionMatchDecision."
            )

        if not isinstance(
            self.actual_match_decision,
            DocumentVersionMatchDecision,
        ):
            raise TypeError(
                "actual_match_decision must be a "
                "DocumentVersionMatchDecision."
            )

        if not isinstance(
            self.match_correct,
            bool,
        ):
            raise TypeError(
                "match_correct must be boolean."
            )

        if not isinstance(
            self.case_success,
            bool,
        ):
            raise TypeError(
                "case_success must be boolean."
            )

        for field_name in (
            "change_counts_correct",
            "impact_correct",
            "briefing_correct",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                value is not None
                and not isinstance(
                    value,
                    bool,
                )
            ):
                raise TypeError(
                    f"{field_name} must be "
                    "boolean or None."
                )

        for field_name in (
            "actual_added",
            "actual_removed",
            "actual_modified",
            "actual_unchanged",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is not None:
                _nonnegative_integer(
                    value,
                    field_name,
                )

        if (
            self.actual_maximum_impact
            is not None
            and not isinstance(
                self.actual_maximum_impact,
                PolicyChangeImpact,
            )
        ):
            raise TypeError(
                "actual_maximum_impact must be "
                "PolicyChangeImpact or None."
            )

        if (
            self.actual_briefing_decision
            is not None
            and not isinstance(
                self.actual_briefing_decision,
                ChangeBriefingDecision,
            )
        ):
            raise TypeError(
                "actual_briefing_decision must be "
                "ChangeBriefingDecision or None."
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
                self.expected_match_decision.value,
                self.actual_match_decision.value,
                str(self.match_correct),
                str(self.change_counts_correct),
                str(self.impact_correct),
                str(self.briefing_correct),
                str(self.actual_added),
                str(self.actual_removed),
                str(self.actual_modified),
                str(self.actual_unchanged),
                (
                    self.actual_maximum_impact.value
                    if self.actual_maximum_impact
                    else ""
                ),
                (
                    self.actual_briefing_decision.value
                    if self.actual_briefing_decision
                    else ""
                ),
                str(self.case_success),
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
            float(self.latency_ms),
        )

        object.__setattr__(
            self,
            "result_id",
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": self.version,
            "result_id": self.result_id,
            "case_id": self.case_id,
            "expected_match_decision": (
                self.expected_match_decision.value
            ),
            "actual_match_decision": (
                self.actual_match_decision.value
            ),
            "match_correct": self.match_correct,
            "change_counts_correct": (
                self.change_counts_correct
            ),
            "impact_correct": self.impact_correct,
            "briefing_correct": (
                self.briefing_correct
            ),
            "actual_added": self.actual_added,
            "actual_removed": self.actual_removed,
            "actual_modified": (
                self.actual_modified
            ),
            "actual_unchanged": (
                self.actual_unchanged
            ),
            "actual_maximum_impact": (
                self.actual_maximum_impact.value
                if self.actual_maximum_impact
                else None
            ),
            "actual_briefing_decision": (
                self.actual_briefing_decision.value
                if self.actual_briefing_decision
                else None
            ),
            "case_success": self.case_success,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class ChangeIntelligenceMetrics:
    """Aggregate Stage-14 evaluation metrics."""

    case_count: int

    downstream_case_count: int

    version_match_accuracy: float

    change_count_accuracy: float

    impact_accuracy: float

    briefing_accuracy: float

    end_to_end_success_rate: float

    mean_latency_ms: float

    version: str = (
        CHANGE_INTELLIGENCE_EVALUATION_VERSION
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.case_count, bool)
            or not isinstance(
                self.case_count,
                int,
            )
            or self.case_count < 1
        ):
            raise ValueError(
                "case_count must be positive."
            )

        _nonnegative_integer(
            self.downstream_case_count,
            "downstream_case_count",
        )

        if (
            self.downstream_case_count
            > self.case_count
        ):
            raise ValueError(
                "downstream_case_count cannot "
                "exceed case_count."
            )

        for field_name in (
            "version_match_accuracy",
            "change_count_accuracy",
            "impact_accuracy",
            "briefing_accuracy",
            "end_to_end_success_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_rate(
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
            "downstream_case_count": (
                self.downstream_case_count
            ),
            "version_match_accuracy": (
                self.version_match_accuracy
            ),
            "change_count_accuracy": (
                self.change_count_accuracy
            ),
            "impact_accuracy": (
                self.impact_accuracy
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
class ChangeIntelligenceBenchmarkResult:
    """Complete deterministic Stage-14 benchmark."""

    cases: tuple[
        ChangeIntelligenceCaseResult,
        ...
    ]

    metrics: ChangeIntelligenceMetrics

    version: str = (
        CHANGE_INTELLIGENCE_EVALUATION_VERSION
    )

    benchmark_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        try:
            cases = tuple(self.cases)
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
                ChangeIntelligenceCaseResult,
            ):
                raise TypeError(
                    "cases must contain only "
                    "ChangeIntelligenceCaseResult values."
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
            ChangeIntelligenceMetrics,
        ):
            raise TypeError(
                "metrics must be "
                "ChangeIntelligenceMetrics."
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
            hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest(),
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


def evaluate_change_intelligence(
    cases: tuple[
        ChangeIntelligenceGoldCase,
        ...
    ],
) -> ChangeIntelligenceBenchmarkResult:
    """Run the complete deterministic Stage-14 pipeline."""

    if isinstance(
        cases,
        (str, bytes),
    ):
        raise TypeError(
            "cases must contain gold cases."
        )

    try:
        gold_cases = tuple(cases)
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
            ChangeIntelligenceGoldCase,
        ):
            raise TypeError(
                "cases must contain only "
                "ChangeIntelligenceGoldCase values."
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

    outputs = []

    for case in gold_cases:
        started = time.perf_counter()

        match = match_document_versions(
            case.baseline_source,
            case.candidate_source,
        )

        match_correct = (
            match.decision
            is case.expected_match_decision
        )

        change_counts_correct = None
        impact_correct = None
        briefing_correct = None

        actual_added = None
        actual_removed = None
        actual_modified = None
        actual_unchanged = None
        actual_maximum_impact = None
        actual_briefing_decision = None

        if (
            case.expects_downstream_analysis
            and match.allowed
        ):
            trace_id = (
                "CHANGE-EVAL-"
                f"{case.case_hash[:24]}"
            )

            detection = detect_clause_changes(
                match,
                baseline_chunks=(
                    case.baseline_chunks
                ),
                candidate_chunks=(
                    case.candidate_chunks
                ),
                trace_id=trace_id,
                similarity_threshold=(
                    case.similarity_threshold
                ),
            )

            significance = (
                classify_change_significance(
                    detection,
                    baseline_chunks=(
                        case.baseline_chunks
                    ),
                    candidate_chunks=(
                        case.candidate_chunks
                    ),
                )
            )

            briefing = (
                build_governed_change_briefing(
                    significance,
                    baseline_chunks=(
                        case.baseline_chunks
                    ),
                    candidate_chunks=(
                        case.candidate_chunks
                    ),
                    trace_id=trace_id,
                )
            )

            actual_added = (
                significance.report.added_count
            )

            actual_removed = (
                significance.report.removed_count
            )

            actual_modified = (
                significance.report.modified_count
            )

            actual_unchanged = (
                significance.report.unchanged_count
            )

            actual_maximum_impact = (
                significance.maximum_impact
            )

            actual_briefing_decision = (
                briefing.decision
            )

            change_counts_correct = (
                actual_added
                == case.expected_added
                and actual_removed
                == case.expected_removed
                and actual_modified
                == case.expected_modified
                and actual_unchanged
                == case.expected_unchanged
            )

            impact_correct = (
                actual_maximum_impact
                is case.expected_maximum_impact
            )

            briefing_correct = (
                actual_briefing_decision
                is case.expected_briefing_decision
            )

        if case.expects_downstream_analysis:
            case_success = bool(
                match_correct
                and change_counts_correct
                and impact_correct
                and briefing_correct
            )
        else:
            case_success = match_correct

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        outputs.append(
            ChangeIntelligenceCaseResult(
                case_id=case.case_id,
                expected_match_decision=(
                    case.expected_match_decision
                ),
                actual_match_decision=(
                    match.decision
                ),
                match_correct=(
                    match_correct
                ),
                change_counts_correct=(
                    change_counts_correct
                ),
                impact_correct=(
                    impact_correct
                ),
                briefing_correct=(
                    briefing_correct
                ),
                actual_added=actual_added,
                actual_removed=actual_removed,
                actual_modified=actual_modified,
                actual_unchanged=actual_unchanged,
                actual_maximum_impact=(
                    actual_maximum_impact
                ),
                actual_briefing_decision=(
                    actual_briefing_decision
                ),
                case_success=case_success,
                latency_ms=latency_ms,
            )
        )

    results = tuple(outputs)

    downstream_results = tuple(
        result
        for result, case
        in zip(
            results,
            gold_cases,
            strict=True,
        )
        if case.expects_downstream_analysis
    )

    downstream_count = len(
        downstream_results
    )

    def downstream_rate(
        field_name: str,
    ) -> float:
        if not downstream_results:
            return 1.0

        return mean(
            float(
                bool(
                    getattr(
                        result,
                        field_name,
                    )
                )
            )
            for result
            in downstream_results
        )

    metrics = ChangeIntelligenceMetrics(
        case_count=len(results),
        downstream_case_count=(
            downstream_count
        ),
        version_match_accuracy=mean(
            float(
                result.match_correct
            )
            for result in results
        ),
        change_count_accuracy=(
            downstream_rate(
                "change_counts_correct"
            )
        ),
        impact_accuracy=(
            downstream_rate(
                "impact_correct"
            )
        ),
        briefing_accuracy=(
            downstream_rate(
                "briefing_correct"
            )
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

    return ChangeIntelligenceBenchmarkResult(
        cases=results,
        metrics=metrics,
    )
