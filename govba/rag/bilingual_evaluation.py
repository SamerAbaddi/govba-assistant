"""Bilingual retrieval benchmarking for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Iterable

from govba.rag.cross_lingual import (
    CrossLingualRetriever,
)
from govba.rag.language import (
    BilingualLanguage,
    BilingualQueryRequest,
    DetectedLanguage,
    build_language_route,
)
from govba.rag.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    Retriever,
)


BILINGUAL_EVALUATION_VERSION = (
    "govba-bilingual-evaluation-v1"
)


def _to_detected_language(
    language: BilingualLanguage,
) -> DetectedLanguage:
    """Map benchmark language to the query-detection contract."""

    if (
        language
        is BilingualLanguage.ARABIC
    ):
        return (
            DetectedLanguage.ARABIC
        )

    if (
        language
        is BilingualLanguage.ENGLISH
    ):
        return (
            DetectedLanguage.ENGLISH
        )

    raise ValueError(
        "Unsupported bilingual benchmark language."
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


def _normalize_ids(
    values,
    field_name: str,
) -> tuple[str, ...]:
    try:
        values = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    normalized = tuple(
        _required_text(
            value,
            field_name,
        )
        for value in values
    )

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    if (
        len(set(normalized))
        != len(normalized)
    ):
        raise ValueError(
            f"{field_name} must not "
            "contain duplicates."
        )

    return normalized


@dataclass(frozen=True)
class BilingualGoldCase:
    """One gold retrieval case in Arabic or English."""

    case_id: str
    pair_id: str

    query_text: str

    query_language: (
        BilingualLanguage
    )

    relevant_chunk_ids: tuple[
        str,
        ...
    ]

    top_k: int = 5

    filters: RetrievalFilters = (
        RetrievalFilters()
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _required_text(
                self.case_id,
                "case_id",
            ),
        )

        object.__setattr__(
            self,
            "pair_id",
            _required_text(
                self.pair_id,
                "pair_id",
            ),
        )

        object.__setattr__(
            self,
            "query_text",
            _required_text(
                self.query_text,
                "query_text",
            ),
        )

        if not isinstance(
            self.query_language,
            BilingualLanguage,
        ):
            raise TypeError(
                "query_language must be a "
                "BilingualLanguage."
            )

        object.__setattr__(
            self,
            "relevant_chunk_ids",
            _normalize_ids(
                self.relevant_chunk_ids,
                "relevant_chunk_ids",
            ),
        )

        if (
            isinstance(
                self.top_k,
                bool,
            )
            or not isinstance(
                self.top_k,
                int,
            )
            or not (
                1 <= self.top_k <= 100
            )
        ):
            raise ValueError(
                "top_k must be between "
                "1 and 100."
            )

        if not isinstance(
            self.filters,
            RetrievalFilters,
        ):
            raise TypeError(
                "filters must be "
                "RetrievalFilters."
            )


@dataclass(frozen=True)
class BilingualCaseResult:
    """Privacy-safe evaluation result for one gold case."""

    case_id: str
    pair_id: str

    query_language: (
        BilingualLanguage
    )

    top_k: int

    returned_chunk_ids: tuple[
        str,
        ...
    ]

    relevant_count: int
    retrieved_relevant_count: int

    first_relevant_rank: (
        int | None
    )

    latency_ms: float

    cross_lingual: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _required_text(
                self.case_id,
                "case_id",
            ),
        )

        object.__setattr__(
            self,
            "pair_id",
            _required_text(
                self.pair_id,
                "pair_id",
            ),
        )

        if not isinstance(
            self.query_language,
            BilingualLanguage,
        ):
            raise TypeError(
                "query_language must be "
                "BilingualLanguage."
            )

        returned = tuple(
            self.returned_chunk_ids
        )

        if (
            len(set(returned))
            != len(returned)
        ):
            raise ValueError(
                "returned_chunk_ids must "
                "be unique."
            )

        object.__setattr__(
            self,
            "returned_chunk_ids",
            returned,
        )

        for field_name in (
            "top_k",
            "relevant_count",
            "retrieved_relevant_count",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a "
                    "non-negative integer."
                )

        if (
            self.retrieved_relevant_count
            > self.relevant_count
        ):
            raise ValueError(
                "retrieved_relevant_count cannot "
                "exceed relevant_count."
            )

        if self.first_relevant_rank is not None:
            if (
                isinstance(
                    self.first_relevant_rank,
                    bool,
                )
                or not isinstance(
                    self.first_relevant_rank,
                    int,
                )
                or self.first_relevant_rank < 1
            ):
                raise ValueError(
                    "first_relevant_rank must be "
                    "a positive integer or None."
                )

        if (
            not isinstance(
                self.latency_ms,
                (int, float),
            )
            or isinstance(
                self.latency_ms,
                bool,
            )
            or not isfinite(
                self.latency_ms
            )
            or self.latency_ms < 0
        ):
            raise ValueError(
                "latency_ms must be a "
                "finite non-negative number."
            )

        if not isinstance(
            self.cross_lingual,
            bool,
        ):
            raise TypeError(
                "cross_lingual must be boolean."
            )

    @property
    def hit(
        self,
    ) -> bool:
        return (
            self.retrieved_relevant_count
            > 0
        )

    @property
    def recall(
        self,
    ) -> float:
        if self.relevant_count == 0:
            return 0.0

        return (
            self.retrieved_relevant_count
            / self.relevant_count
        )

    @property
    def reciprocal_rank(
        self,
    ) -> float:
        if (
            self.first_relevant_rank
            is None
        ):
            return 0.0

        return (
            1.0
            / self.first_relevant_rank
        )

    @property
    def zero_result(
        self,
    ) -> bool:
        return not bool(
            self.returned_chunk_ids
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "case_id": (
                self.case_id
            ),
            "pair_id": (
                self.pair_id
            ),
            "query_language": (
                self.query_language.value
            ),
            "top_k": (
                self.top_k
            ),
            "returned_chunk_ids": list(
                self.returned_chunk_ids
            ),
            "relevant_count": (
                self.relevant_count
            ),
            "retrieved_relevant_count": (
                self.retrieved_relevant_count
            ),
            "first_relevant_rank": (
                self.first_relevant_rank
            ),
            "hit": (
                self.hit
            ),
            "recall": (
                self.recall
            ),
            "reciprocal_rank": (
                self.reciprocal_rank
            ),
            "zero_result": (
                self.zero_result
            ),
            "latency_ms": (
                self.latency_ms
            ),
            "cross_lingual": (
                self.cross_lingual
            ),
        }


@dataclass(frozen=True)
class BilingualLanguageMetrics:
    """Aggregate retrieval metrics for one language."""

    language: BilingualLanguage

    case_count: int

    hit_rate: float
    mean_recall: float
    mrr: float
    zero_result_rate: float
    mean_latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.language,
            BilingualLanguage,
        ):
            raise TypeError(
                "language must be "
                "BilingualLanguage."
            )

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
                "case_count must be a "
                "positive integer."
            )

        for field_name in (
            "hit_rate",
            "mean_recall",
            "mrr",
            "zero_result_rate",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isfinite(
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

        if (
            not isfinite(
                self.mean_latency_ms
            )
            or self.mean_latency_ms < 0
        ):
            raise ValueError(
                "mean_latency_ms must be "
                "finite and non-negative."
            )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "language": (
                self.language.value
            ),
            "case_count": (
                self.case_count
            ),
            "hit_rate": (
                self.hit_rate
            ),
            "mean_recall": (
                self.mean_recall
            ),
            "mrr": (
                self.mrr
            ),
            "zero_result_rate": (
                self.zero_result_rate
            ),
            "mean_latency_ms": (
                self.mean_latency_ms
            ),
        }


def _aggregate_metrics(
    results: tuple[
        BilingualCaseResult,
        ...
    ],
    language: BilingualLanguage,
) -> BilingualLanguageMetrics:
    selected = tuple(
        result
        for result in results
        if result.query_language
        is language
    )

    if not selected:
        raise ValueError(
            f"No benchmark cases for "
            f"{language.value}."
        )

    count = len(
        selected
    )

    return BilingualLanguageMetrics(
        language=language,
        case_count=count,
        hit_rate=(
            sum(
                result.hit
                for result in selected
            )
            / count
        ),
        mean_recall=(
            sum(
                result.recall
                for result in selected
            )
            / count
        ),
        mrr=(
            sum(
                result.reciprocal_rank
                for result in selected
            )
            / count
        ),
        zero_result_rate=(
            sum(
                result.zero_result
                for result in selected
            )
            / count
        ),
        mean_latency_ms=(
            sum(
                result.latency_ms
                for result in selected
            )
            / count
        ),
    )


@dataclass(frozen=True)
class BilingualBenchmarkResult:
    """Aggregate bilingual retrieval benchmark."""

    cases: tuple[
        BilingualCaseResult,
        ...
    ]

    arabic: BilingualLanguageMetrics
    english: BilingualLanguageMetrics

    cross_lingual: bool

    version: str = (
        BILINGUAL_EVALUATION_VERSION
    )

    @property
    def case_count(
        self,
    ) -> int:
        return len(
            self.cases
        )

    @property
    def hit_rate_gap(
        self,
    ) -> float:
        return abs(
            self.arabic.hit_rate
            - self.english.hit_rate
        )

    @property
    def recall_gap(
        self,
    ) -> float:
        return abs(
            self.arabic.mean_recall
            - self.english.mean_recall
        )

    @property
    def mrr_gap(
        self,
    ) -> float:
        return abs(
            self.arabic.mrr
            - self.english.mrr
        )

    @property
    def zero_result_gap(
        self,
    ) -> float:
        return abs(
            self.arabic.zero_result_rate
            - self.english.zero_result_rate
        )

    @property
    def maximum_quality_gap(
        self,
    ) -> float:
        return max(
            self.hit_rate_gap,
            self.recall_gap,
            self.mrr_gap,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize benchmark results without query text."""

        return {
            "version": (
                self.version
            ),
            "cross_lingual": (
                self.cross_lingual
            ),
            "case_count": (
                self.case_count
            ),
            "arabic": (
                self.arabic.to_dict()
            ),
            "english": (
                self.english.to_dict()
            ),
            "parity": {
                "hit_rate_gap": (
                    self.hit_rate_gap
                ),
                "recall_gap": (
                    self.recall_gap
                ),
                "mrr_gap": (
                    self.mrr_gap
                ),
                "zero_result_gap": (
                    self.zero_result_gap
                ),
                "maximum_quality_gap": (
                    self.maximum_quality_gap
                ),
            },
            "cases": [
                case.to_dict()
                for case
                in self.cases
            ],
        }


@dataclass(frozen=True)
class BilingualModeComparison:
    """Compare monolingual and cross-lingual retrieval."""

    monolingual: (
        BilingualBenchmarkResult
    )

    cross_lingual: (
        BilingualBenchmarkResult
    )

    version: str = (
        BILINGUAL_EVALUATION_VERSION
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.monolingual,
                BilingualBenchmarkResult,
            )
            or not isinstance(
                self.cross_lingual,
                BilingualBenchmarkResult,
            )
        ):
            raise TypeError(
                "Both comparison values must be "
                "BilingualBenchmarkResult."
            )

        if self.monolingual.cross_lingual:
            raise ValueError(
                "monolingual result must use "
                "cross_lingual=False."
            )

        if not self.cross_lingual.cross_lingual:
            raise ValueError(
                "cross_lingual result must use "
                "cross_lingual=True."
            )

        if (
            tuple(
                (
                    case.case_id,
                    case.pair_id,
                    case.query_language,
                )
                for case
                in self.monolingual.cases
            )
            != tuple(
                (
                    case.case_id,
                    case.pair_id,
                    case.query_language,
                )
                for case
                in self.cross_lingual.cases
            )
        ):
            raise ValueError(
                "Comparison benchmarks must use "
                "the same ordered gold cases."
            )

    @property
    def arabic_mrr_delta(
        self,
    ) -> float:
        return (
            self.cross_lingual.arabic.mrr
            - self.monolingual.arabic.mrr
        )

    @property
    def english_mrr_delta(
        self,
    ) -> float:
        return (
            self.cross_lingual.english.mrr
            - self.monolingual.english.mrr
        )

    @property
    def parity_gap_delta(
        self,
    ) -> float:
        """Negative means cross-lingual retrieval improved parity."""

        return (
            self.cross_lingual
            .maximum_quality_gap
            - self.monolingual
            .maximum_quality_gap
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "arabic_mrr_delta": (
                self.arabic_mrr_delta
            ),
            "english_mrr_delta": (
                self.english_mrr_delta
            ),
            "parity_gap_delta": (
                self.parity_gap_delta
            ),
            "monolingual": (
                self.monolingual.to_dict()
            ),
            "cross_lingual": (
                self.cross_lingual.to_dict()
            ),
        }


def _validate_cases(
    cases: Iterable[
        BilingualGoldCase
    ],
) -> tuple[
    BilingualGoldCase,
    ...
]:
    try:
        values = tuple(
            cases
        )
    except TypeError as exc:
        raise TypeError(
            "cases must be iterable."
        ) from exc

    if not values:
        raise ValueError(
            "At least one benchmark case "
            "is required."
        )

    for case in values:
        if not isinstance(
            case,
            BilingualGoldCase,
        ):
            raise TypeError(
                "cases must contain only "
                "BilingualGoldCase values."
            )

    case_ids = tuple(
        case.case_id
        for case in values
    )

    if (
        len(set(case_ids))
        != len(case_ids)
    ):
        raise ValueError(
            "Benchmark case IDs must be unique."
        )

    languages = {
        case.query_language
        for case in values
    }

    if languages != {
        BilingualLanguage.ARABIC,
        BilingualLanguage.ENGLISH,
    }:
        raise ValueError(
            "Benchmark must contain both "
            "Arabic and English cases."
        )

    pair_languages: dict[
        str,
        set[
            BilingualLanguage
        ],
    ] = {}

    pair_counts: dict[
        str,
        int,
    ] = {}

    for case in values:
        pair_languages.setdefault(
            case.pair_id,
            set(),
        ).add(
            case.query_language
        )

        pair_counts[
            case.pair_id
        ] = (
            pair_counts.get(
                case.pair_id,
                0,
            )
            + 1
        )

    for pair_id in sorted(
        pair_languages
    ):
        if (
            pair_counts[
                pair_id
            ]
            != 2
            or pair_languages[
                pair_id
            ]
            != {
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            }
        ):
            raise ValueError(
                "Every pair_id must contain "
                "exactly one Arabic case and "
                "one English case."
            )

    return values


def evaluate_bilingual_retrieval(
    retriever: Retriever,
    cases: Iterable[
        BilingualGoldCase
    ],
    *,
    allow_cross_lingual: bool,
    candidate_pool_size: int = 100,
) -> BilingualBenchmarkResult:
    """Evaluate bilingual retrieval on paired gold cases."""

    if not isinstance(
        retriever,
        Retriever,
    ):
        raise TypeError(
            "retriever must implement "
            "the Retriever protocol."
        )

    if not isinstance(
        allow_cross_lingual,
        bool,
    ):
        raise TypeError(
            "allow_cross_lingual "
            "must be boolean."
        )

    case_values = _validate_cases(
        cases
    )

    results = []

    for case in case_values:
        request = BilingualQueryRequest(
            query_text=(
                case.query_text
            ),
            detected_language=(
                _to_detected_language(
                    case.query_language
                )
            ),
            response_language=(
                case.query_language
            ),
            allow_cross_lingual=(
                allow_cross_lingual
            ),
        )

        route = build_language_route(
            request
        )

        wrapper = CrossLingualRetriever(
            retriever=retriever,
            route=route,
            candidate_pool_size=(
                candidate_pool_size
            ),
        )

        query = RetrievalQuery(
            text=case.query_text,
            top_k=case.top_k,
            filters=case.filters,
        )

        started = perf_counter()

        retrieved = wrapper.retrieve(
            query
        )

        latency_ms = (
            perf_counter()
            - started
        ) * 1000.0

        returned_ids = tuple(
            result.chunk.chunk_id
            for result in retrieved
        )

        relevant_set = set(
            case.relevant_chunk_ids
        )

        retrieved_relevant = tuple(
            chunk_id
            for chunk_id in returned_ids
            if chunk_id in relevant_set
        )

        first_rank = None

        for rank, chunk_id in enumerate(
            returned_ids,
            start=1,
        ):
            if chunk_id in relevant_set:
                first_rank = rank
                break

        results.append(
            BilingualCaseResult(
                case_id=case.case_id,
                pair_id=case.pair_id,
                query_language=(
                    case.query_language
                ),
                top_k=case.top_k,
                returned_chunk_ids=(
                    returned_ids
                ),
                relevant_count=len(
                    relevant_set
                ),
                retrieved_relevant_count=len(
                    retrieved_relevant
                ),
                first_relevant_rank=(
                    first_rank
                ),
                latency_ms=(
                    latency_ms
                ),
                cross_lingual=(
                    allow_cross_lingual
                ),
            )
        )

    result_values = tuple(
        results
    )

    return BilingualBenchmarkResult(
        cases=result_values,
        arabic=_aggregate_metrics(
            result_values,
            BilingualLanguage.ARABIC,
        ),
        english=_aggregate_metrics(
            result_values,
            BilingualLanguage.ENGLISH,
        ),
        cross_lingual=(
            allow_cross_lingual
        ),
    )


def compare_bilingual_retrieval_modes(
    retriever: Retriever,
    cases: Iterable[
        BilingualGoldCase
    ],
    *,
    candidate_pool_size: int = 100,
) -> BilingualModeComparison:
    """Compare monolingual and cross-lingual retrieval on identical cases."""

    case_values = _validate_cases(
        cases
    )

    monolingual = (
        evaluate_bilingual_retrieval(
            retriever,
            case_values,
            allow_cross_lingual=False,
            candidate_pool_size=(
                candidate_pool_size
            ),
        )
    )

    cross_lingual = (
        evaluate_bilingual_retrieval(
            retriever,
            case_values,
            allow_cross_lingual=True,
            candidate_pool_size=(
                candidate_pool_size
            ),
        )
    )

    return BilingualModeComparison(
        monolingual=monolingual,
        cross_lingual=cross_lingual,
    )
