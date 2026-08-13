"""Configurable bilingual retrieval parity safeguards for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from govba.rag.bilingual_evaluation import (
    BilingualBenchmarkResult,
)
from govba.rag.language import (
    BilingualLanguage,
)


LANGUAGE_PARITY_VERSION = (
    "govba-language-parity-v1"
)


class LanguageParityDecision(
    str,
    Enum,
):
    PASS = "pass"
    BLOCK = "block"


class LanguageParityIssueCode(
    str,
    Enum,
):
    ARABIC_HIT_RATE_LOW = (
        "arabic_hit_rate_low"
    )
    ENGLISH_HIT_RATE_LOW = (
        "english_hit_rate_low"
    )

    ARABIC_RECALL_LOW = (
        "arabic_recall_low"
    )
    ENGLISH_RECALL_LOW = (
        "english_recall_low"
    )

    ARABIC_MRR_LOW = (
        "arabic_mrr_low"
    )
    ENGLISH_MRR_LOW = (
        "english_mrr_low"
    )

    ARABIC_ZERO_RESULT_HIGH = (
        "arabic_zero_result_high"
    )
    ENGLISH_ZERO_RESULT_HIGH = (
        "english_zero_result_high"
    )

    HIT_RATE_GAP = (
        "hit_rate_gap"
    )
    RECALL_GAP = (
        "recall_gap"
    )
    MRR_GAP = (
        "mrr_gap"
    )
    ZERO_RESULT_GAP = (
        "zero_result_gap"
    )

    WRONG_RETRIEVAL_MODE = (
        "wrong_retrieval_mode"
    )


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
        or not isfinite(
            value
        )
        or not (
            0.0 <= value <= 1.0
        )
    ):
        raise ValueError(
            f"{field_name} must be "
            "between 0 and 1."
        )

    return float(
        value
    )


def _policy_id(
    policy: "LanguageParityPolicy",
) -> str:
    payload = "\x1f".join(
        (
            LANGUAGE_PARITY_VERSION,
            str(
                policy.min_hit_rate
            ),
            str(
                policy.min_mean_recall
            ),
            str(
                policy.min_mrr
            ),
            str(
                policy.max_zero_result_rate
            ),
            str(
                policy.max_hit_rate_gap
            ),
            str(
                policy.max_recall_gap
            ),
            str(
                policy.max_mrr_gap
            ),
            str(
                policy.max_zero_result_gap
            ),
            str(
                policy.require_cross_lingual
            ).lower(),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class LanguageParityPolicy:
    """Explicit deployment thresholds for bilingual retrieval."""

    min_hit_rate: float
    min_mean_recall: float
    min_mrr: float

    max_zero_result_rate: float

    max_hit_rate_gap: float
    max_recall_gap: float
    max_mrr_gap: float
    max_zero_result_gap: float

    require_cross_lingual: bool = True

    version: str = (
        LANGUAGE_PARITY_VERSION
    )

    policy_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        for field_name in (
            "min_hit_rate",
            "min_mean_recall",
            "min_mrr",
            "max_zero_result_rate",
            "max_hit_rate_gap",
            "max_recall_gap",
            "max_mrr_gap",
            "max_zero_result_gap",
        ):
            value = _rate(
                getattr(
                    self,
                    field_name,
                ),
                field_name,
            )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        if not isinstance(
            self.require_cross_lingual,
            bool,
        ):
            raise TypeError(
                "require_cross_lingual "
                "must be boolean."
            )

        object.__setattr__(
            self,
            "policy_id",
            _policy_id(
                self
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
            "min_hit_rate": (
                self.min_hit_rate
            ),
            "min_mean_recall": (
                self.min_mean_recall
            ),
            "min_mrr": (
                self.min_mrr
            ),
            "max_zero_result_rate": (
                self.max_zero_result_rate
            ),
            "max_hit_rate_gap": (
                self.max_hit_rate_gap
            ),
            "max_recall_gap": (
                self.max_recall_gap
            ),
            "max_mrr_gap": (
                self.max_mrr_gap
            ),
            "max_zero_result_gap": (
                self.max_zero_result_gap
            ),
            "require_cross_lingual": (
                self.require_cross_lingual
            ),
        }


@dataclass(frozen=True)
class LanguageParityIssue:
    """One failed bilingual quality/parity requirement."""

    code: LanguageParityIssueCode

    observed: float | None = None
    threshold: float | None = None

    language: (
        BilingualLanguage | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.code,
            LanguageParityIssueCode,
        ):
            raise TypeError(
                "code must be a "
                "LanguageParityIssueCode."
            )

        for field_name in (
            "observed",
            "threshold",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is None:
                continue

            _rate(
                value,
                field_name,
            )

        if (
            self.language is not None
            and not isinstance(
                self.language,
                BilingualLanguage,
            )
        ):
            raise TypeError(
                "language must be a "
                "BilingualLanguage or None."
            )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "code": (
                self.code.value
            ),
            "language": (
                self.language.value
                if self.language
                else None
            ),
            "observed": (
                self.observed
            ),
            "threshold": (
                self.threshold
            ),
        }


@dataclass(frozen=True)
class LanguageParityAssessment:
    """Fail-safe bilingual deployment assessment."""

    benchmark: (
        BilingualBenchmarkResult
    )

    policy: LanguageParityPolicy

    issues: tuple[
        LanguageParityIssue,
        ...
    ]

    version: str = (
        LANGUAGE_PARITY_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.benchmark,
            BilingualBenchmarkResult,
        ):
            raise TypeError(
                "benchmark must be a "
                "BilingualBenchmarkResult."
            )

        if not isinstance(
            self.policy,
            LanguageParityPolicy,
        ):
            raise TypeError(
                "policy must be a "
                "LanguageParityPolicy."
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
                LanguageParityIssue,
            ):
                raise TypeError(
                    "issues must contain only "
                    "LanguageParityIssue values."
                )

        object.__setattr__(
            self,
            "issues",
            issues,
        )

    @property
    def decision(
        self,
    ) -> LanguageParityDecision:
        if self.issues:
            return (
                LanguageParityDecision.BLOCK
            )

        return (
            LanguageParityDecision.PASS
        )

    @property
    def passed(
        self,
    ) -> bool:
        return (
            self.decision
            is LanguageParityDecision.PASS
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return not self.passed

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without query text."""

        return {
            "version": (
                self.version
            ),
            "decision": (
                self.decision.value
            ),
            "passed": (
                self.passed
            ),
            "should_block": (
                self.should_block
            ),
            "policy": (
                self.policy.to_dict()
            ),
            "benchmark_version": (
                self.benchmark.version
            ),
            "benchmark_cross_lingual": (
                self.benchmark.cross_lingual
            ),
            "issues": [
                issue.to_dict()
                for issue
                in self.issues
            ],
        }


def assess_language_parity(
    benchmark: BilingualBenchmarkResult,
    policy: LanguageParityPolicy,
) -> LanguageParityAssessment:
    """Apply explicit quality and parity thresholds."""

    if not isinstance(
        benchmark,
        BilingualBenchmarkResult,
    ):
        raise TypeError(
            "benchmark must be a "
            "BilingualBenchmarkResult."
        )

    if not isinstance(
        policy,
        LanguageParityPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "LanguageParityPolicy."
        )

    issues = []

    if (
        policy.require_cross_lingual
        and not benchmark.cross_lingual
    ):
        issues.append(
            LanguageParityIssue(
                code=(
                    LanguageParityIssueCode
                    .WRONG_RETRIEVAL_MODE
                ),
            )
        )

    metric_sets = (
        (
            BilingualLanguage.ARABIC,
            benchmark.arabic,
            LanguageParityIssueCode
            .ARABIC_HIT_RATE_LOW,
            LanguageParityIssueCode
            .ARABIC_RECALL_LOW,
            LanguageParityIssueCode
            .ARABIC_MRR_LOW,
            LanguageParityIssueCode
            .ARABIC_ZERO_RESULT_HIGH,
        ),
        (
            BilingualLanguage.ENGLISH,
            benchmark.english,
            LanguageParityIssueCode
            .ENGLISH_HIT_RATE_LOW,
            LanguageParityIssueCode
            .ENGLISH_RECALL_LOW,
            LanguageParityIssueCode
            .ENGLISH_MRR_LOW,
            LanguageParityIssueCode
            .ENGLISH_ZERO_RESULT_HIGH,
        ),
    )

    for (
        language,
        metrics,
        hit_code,
        recall_code,
        mrr_code,
        zero_code,
    ) in metric_sets:
        if (
            metrics.hit_rate
            < policy.min_hit_rate
        ):
            issues.append(
                LanguageParityIssue(
                    code=hit_code,
                    language=language,
                    observed=(
                        metrics.hit_rate
                    ),
                    threshold=(
                        policy.min_hit_rate
                    ),
                )
            )

        if (
            metrics.mean_recall
            < policy.min_mean_recall
        ):
            issues.append(
                LanguageParityIssue(
                    code=recall_code,
                    language=language,
                    observed=(
                        metrics.mean_recall
                    ),
                    threshold=(
                        policy.min_mean_recall
                    ),
                )
            )

        if (
            metrics.mrr
            < policy.min_mrr
        ):
            issues.append(
                LanguageParityIssue(
                    code=mrr_code,
                    language=language,
                    observed=(
                        metrics.mrr
                    ),
                    threshold=(
                        policy.min_mrr
                    ),
                )
            )

        if (
            metrics.zero_result_rate
            > policy.max_zero_result_rate
        ):
            issues.append(
                LanguageParityIssue(
                    code=zero_code,
                    language=language,
                    observed=(
                        metrics.zero_result_rate
                    ),
                    threshold=(
                        policy.max_zero_result_rate
                    ),
                )
            )

    gap_checks = (
        (
            benchmark.hit_rate_gap,
            policy.max_hit_rate_gap,
            LanguageParityIssueCode
            .HIT_RATE_GAP,
        ),
        (
            benchmark.recall_gap,
            policy.max_recall_gap,
            LanguageParityIssueCode
            .RECALL_GAP,
        ),
        (
            benchmark.mrr_gap,
            policy.max_mrr_gap,
            LanguageParityIssueCode
            .MRR_GAP,
        ),
        (
            benchmark.zero_result_gap,
            policy.max_zero_result_gap,
            LanguageParityIssueCode
            .ZERO_RESULT_GAP,
        ),
    )

    for (
        observed,
        threshold,
        code,
    ) in gap_checks:
        if observed > threshold:
            issues.append(
                LanguageParityIssue(
                    code=code,
                    observed=observed,
                    threshold=threshold,
                )
            )

    return LanguageParityAssessment(
        benchmark=benchmark,
        policy=policy,
        issues=tuple(
            issues
        ),
    )
