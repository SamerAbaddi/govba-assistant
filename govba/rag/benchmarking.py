"""Controlled retrieval benchmarking for GovBA-GAR.

This module compares multiple retrievers against the same gold-standard
cases and cutoffs. It is provider-independent and can benchmark lexical,
semantic, hybrid, or future governance-aware retrievers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from govba.rag.evaluation import (
    DEFAULT_RETRIEVAL_CUTOFFS,
    GoldRetrievalCase,
    RetrievalBenchmarkEvaluation,
    evaluate_benchmark,
    normalize_cutoffs,
)
from govba.rag.retrieval import Retriever


BENCHMARK_COMPARISON_SCHEMA_VERSION = (
    "govba-retrieval-comparison-v1"
)


def _normalize_retriever_name(
    value: str,
) -> str:
    """Validate and normalize a benchmark retriever name."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Retriever benchmark name is required."
        )

    return value.strip()


@dataclass(frozen=True)
class RetrieverBenchmarkRun:
    """Evaluation results for one named retriever."""

    name: str
    evaluation: RetrievalBenchmarkEvaluation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_retriever_name(
                self.name
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible benchmark-run data."""

        return {
            "name": self.name,
            "evaluation": (
                self.evaluation.to_dict()
            ),
        }


@dataclass(frozen=True)
class RetrievalBenchmarkComparison:
    """Controlled comparison of multiple retrieval engines."""

    case_count: int
    cutoffs: tuple[int, ...]
    runs: tuple[
        RetrieverBenchmarkRun,
        ...
    ]

    schema_version: str = (
        BENCHMARK_COMPARISON_SCHEMA_VERSION
    )

    def get(
        self,
        name: str,
    ) -> RetrieverBenchmarkRun:
        """Return one benchmark run by case-insensitive name."""

        target = _normalize_retriever_name(
            name
        ).casefold()

        for run in self.runs:
            if run.name.casefold() == target:
                return run

        raise KeyError(
            f"Unknown retriever benchmark: {name}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible comparison data."""

        return {
            "schema_version": self.schema_version,
            "case_count": self.case_count,
            "cutoffs": list(
                self.cutoffs
            ),
            "runs": [
                run.to_dict()
                for run in self.runs
            ],
        }

    def to_json(self) -> str:
        """Return stable UTF-8 JSON for experiment artifacts."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def compare_retrievers(
    *,
    retrievers: Mapping[str, Retriever],
    cases: Iterable[GoldRetrievalCase],
    cutoffs: Iterable[int] = DEFAULT_RETRIEVAL_CUTOFFS,
) -> RetrievalBenchmarkComparison:
    """Evaluate named retrievers against identical gold cases."""

    normalized_cutoffs = normalize_cutoffs(
        cutoffs
    )

    case_list = tuple(cases)

    if not case_list:
        raise ValueError(
            "At least one benchmark case is required."
        )

    if not retrievers:
        raise ValueError(
            "At least one retriever is required."
        )

    normalized_retrievers: list[
        tuple[str, Retriever]
    ] = []

    seen_names: set[str] = set()

    for raw_name, retriever in retrievers.items():
        name = _normalize_retriever_name(
            raw_name
        )

        name_key = name.casefold()

        if name_key in seen_names:
            raise ValueError(
                "Retriever benchmark names must be unique."
            )

        seen_names.add(
            name_key
        )

        if not isinstance(
            retriever,
            Retriever,
        ):
            raise TypeError(
                f"Retriever '{name}' does not satisfy "
                "the Retriever contract."
            )

        normalized_retrievers.append(
            (
                name,
                retriever,
            )
        )

    runs = tuple(
        RetrieverBenchmarkRun(
            name=name,
            evaluation=evaluate_benchmark(
                retriever=retriever,
                cases=case_list,
                cutoffs=normalized_cutoffs,
            ),
        )
        for name, retriever
        in normalized_retrievers
    )

    return RetrievalBenchmarkComparison(
        case_count=len(
            case_list
        ),
        cutoffs=normalized_cutoffs,
        runs=runs,
    )
