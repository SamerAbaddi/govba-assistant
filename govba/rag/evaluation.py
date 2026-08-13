"""Retrieval evaluation infrastructure for GovBA-GAR.

This module evaluates ranked evidence retrieval independently from answer
generation. It enables reproducible comparison of lexical, semantic, hybrid,
and governance-aware retrieval architectures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

from govba.rag.retrieval import (
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)


DEFAULT_RETRIEVAL_CUTOFFS = (1, 3, 5)
EVALUATION_SCHEMA_VERSION = "govba-retrieval-eval-v1"


def _require_text(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize required text."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")

    return value.strip()


def _normalize_chunk_ids(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Normalize and de-duplicate relevant chunk identifiers."""

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _require_text(
            value,
            "relevant_chunk_ids value",
        )

        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)

    if not result:
        raise ValueError(
            "At least one relevant chunk_id is required."
        )

    return tuple(result)


def normalize_cutoffs(
    cutoffs: Iterable[int],
) -> tuple[int, ...]:
    """Return validated, unique retrieval cutoffs in ascending order."""

    normalized: set[int] = set()

    for value in cutoffs:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(
                "Retrieval cutoffs must be positive integers."
            )

        normalized.add(value)

    if not normalized:
        raise ValueError(
            "At least one retrieval cutoff is required."
        )

    return tuple(sorted(normalized))


@dataclass(frozen=True)
class GoldRetrievalCase:
    """One gold-standard retrieval task."""

    case_id: str
    query: RetrievalQuery
    relevant_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate gold-standard identifiers."""

        object.__setattr__(
            self,
            "case_id",
            _require_text(
                self.case_id,
                "case_id",
            ),
        )

        object.__setattr__(
            self,
            "relevant_chunk_ids",
            _normalize_chunk_ids(
                self.relevant_chunk_ids
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible gold-case metadata."""

        return {
            "case_id": self.case_id,
            "query": self.query.text,
            "top_k": self.query.top_k,
            "relevant_chunk_ids": list(
                self.relevant_chunk_ids
            ),
        }


@dataclass(frozen=True)
class RetrievalCaseEvaluation:
    """Retrieval metrics for one gold-standard case."""

    case_id: str
    retrieved_count: int
    relevant_count: int

    first_relevant_rank: int | None
    reciprocal_rank: float

    latency_ms: float

    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]

    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible case metrics."""

        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "first_relevant_rank": self.first_relevant_rank,
            "reciprocal_rank": self.reciprocal_rank,
            "latency_ms": self.latency_ms,
            "hit_at_k": {
                str(k): value
                for k, value in self.hit_at_k.items()
            },
            "recall_at_k": {
                str(k): value
                for k, value in self.recall_at_k.items()
            },
        }


@dataclass(frozen=True)
class RetrievalBenchmarkEvaluation:
    """Aggregate retrieval metrics across a benchmark."""

    case_count: int
    cutoffs: tuple[int, ...]

    mean_reciprocal_rank: float
    mean_latency_ms: float
    zero_result_rate: float

    hit_rate_at_k: dict[int, float]
    mean_recall_at_k: dict[int, float]

    cases: tuple[
        RetrievalCaseEvaluation,
        ...
    ]

    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible benchmark metrics."""

        return {
            "schema_version": self.schema_version,
            "case_count": self.case_count,
            "cutoffs": list(self.cutoffs),
            "mean_reciprocal_rank": (
                self.mean_reciprocal_rank
            ),
            "mean_latency_ms": self.mean_latency_ms,
            "zero_result_rate": self.zero_result_rate,
            "hit_rate_at_k": {
                str(k): value
                for k, value in self.hit_rate_at_k.items()
            },
            "mean_recall_at_k": {
                str(k): value
                for k, value
                in self.mean_recall_at_k.items()
            },
            "cases": [
                case.to_dict()
                for case in self.cases
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


def evaluate_ranked_results(
    *,
    case: GoldRetrievalCase,
    results: Iterable[RetrievalResult],
    latency_ms: float = 0.0,
    cutoffs: Iterable[int] = DEFAULT_RETRIEVAL_CUTOFFS,
) -> RetrievalCaseEvaluation:
    """Evaluate already-ranked results against one gold case."""

    normalized_cutoffs = normalize_cutoffs(
        cutoffs
    )

    if (
        not isfinite(latency_ms)
        or latency_ms < 0
    ):
        raise ValueError(
            "latency_ms must be finite and non-negative."
        )

    ordered_results = tuple(
        sorted(
            results,
            key=lambda result: (
                result.rank,
                result.chunk.chunk_id,
            ),
        )
    )

    retrieved_ids = tuple(
        result.chunk.chunk_id
        for result in ordered_results
    )

    relevant_ids = set(
        case.relevant_chunk_ids
    )

    first_relevant_rank: int | None = None

    for position, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            first_relevant_rank = position
            break

    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    hit_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}

    for cutoff in normalized_cutoffs:
        retrieved_at_k = set(
            retrieved_ids[:cutoff]
        )

        relevant_retrieved = (
            relevant_ids
            & retrieved_at_k
        )

        hit_at_k[cutoff] = (
            1.0
            if relevant_retrieved
            else 0.0
        )

        recall_at_k[cutoff] = (
            len(relevant_retrieved)
            / len(relevant_ids)
        )

    return RetrievalCaseEvaluation(
        case_id=case.case_id,
        retrieved_count=len(
            ordered_results
        ),
        relevant_count=len(
            relevant_ids
        ),
        first_relevant_rank=(
            first_relevant_rank
        ),
        reciprocal_rank=(
            reciprocal_rank
        ),
        latency_ms=latency_ms,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
    )


def evaluate_retrieval_case(
    *,
    retriever: Retriever,
    case: GoldRetrievalCase,
    cutoffs: Iterable[int] = DEFAULT_RETRIEVAL_CUTOFFS,
) -> RetrievalCaseEvaluation:
    """Run and evaluate one retrieval case with latency measurement."""

    normalized_cutoffs = normalize_cutoffs(
        cutoffs
    )

    started = perf_counter()

    results = retriever.retrieve(
        case.query
    )

    latency_ms = (
        perf_counter() - started
    ) * 1000.0

    return evaluate_ranked_results(
        case=case,
        results=results,
        latency_ms=latency_ms,
        cutoffs=normalized_cutoffs,
    )


def evaluate_benchmark(
    *,
    retriever: Retriever,
    cases: Iterable[GoldRetrievalCase],
    cutoffs: Iterable[int] = DEFAULT_RETRIEVAL_CUTOFFS,
) -> RetrievalBenchmarkEvaluation:
    """Evaluate a retriever across multiple gold-standard cases."""

    normalized_cutoffs = normalize_cutoffs(
        cutoffs
    )

    case_list = tuple(cases)

    if not case_list:
        raise ValueError(
            "At least one benchmark case is required."
        )

    case_ids = [
        case.case_id
        for case in case_list
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Benchmark case_id values must be unique."
        )

    evaluations = tuple(
        evaluate_retrieval_case(
            retriever=retriever,
            case=case,
            cutoffs=normalized_cutoffs,
        )
        for case in case_list
    )

    case_count = len(evaluations)

    hit_rate_at_k = {
        cutoff: mean(
            evaluation.hit_at_k[cutoff]
            for evaluation in evaluations
        )
        for cutoff in normalized_cutoffs
    }

    mean_recall_at_k = {
        cutoff: mean(
            evaluation.recall_at_k[cutoff]
            for evaluation in evaluations
        )
        for cutoff in normalized_cutoffs
    }

    mean_reciprocal_rank = mean(
        evaluation.reciprocal_rank
        for evaluation in evaluations
    )

    mean_latency_ms = mean(
        evaluation.latency_ms
        for evaluation in evaluations
    )

    zero_result_rate = (
        sum(
            evaluation.retrieved_count == 0
            for evaluation in evaluations
        )
        / case_count
    )

    return RetrievalBenchmarkEvaluation(
        case_count=case_count,
        cutoffs=normalized_cutoffs,
        mean_reciprocal_rank=(
            mean_reciprocal_rank
        ),
        mean_latency_ms=(
            mean_latency_ms
        ),
        zero_result_rate=(
            zero_result_rate
        ),
        hit_rate_at_k=hit_rate_at_k,
        mean_recall_at_k=(
            mean_recall_at_k
        ),
        cases=evaluations,
    )
