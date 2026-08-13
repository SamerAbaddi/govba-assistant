"""Offline evaluation framework for GovBA-GAR controlled web retrieval."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import mean
from urllib.parse import (
    urlsplit,
    urlunsplit,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
    assess_source_url,
)
from govba.rag.language import (
    BilingualLanguage,
)
from govba.web.contract import (
    OfficialWebRequest,
    OfficialWebResult,
    OfficialWebRetriever,
)


CONTROLLED_WEB_EVALUATION_VERSION = (
    "govba-controlled-web-evaluation-v1"
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


def _canonical_url(
    value: str,
) -> str:
    value = _required_text(
        value,
        "url",
    )

    parsed = urlsplit(
        value
    )

    if (
        parsed.scheme.lower()
        != "https"
    ):
        raise ValueError(
            "Evaluation URLs must use HTTPS."
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Evaluation URLs cannot contain "
            "embedded credentials."
        )

    hostname = parsed.hostname

    if not hostname:
        raise ValueError(
            "Evaluation URL requires a hostname."
        )

    try:
        hostname = (
            hostname
            .rstrip(".")
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError as exc:
        raise ValueError(
            "Invalid evaluation hostname."
        ) from exc

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "Invalid evaluation URL port."
        ) from exc

    if (
        port is None
        or port == 443
    ):
        netloc = hostname
    else:
        netloc = (
            f"{hostname}:{port}"
        )

    return urlunsplit(
        (
            "https",
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


@dataclass(frozen=True)
class ControlledWebGoldCase:
    """One gold-standard official-web retrieval case."""

    case_id: str

    query_text: str

    language: BilingualLanguage

    relevant_urls: tuple[
        str,
        ...
    ]

    top_k: int = 5

    version: str = (
        CONTROLLED_WEB_EVALUATION_VERSION
    )

    def __post_init__(self) -> None:
        case_id = _required_text(
            self.case_id,
            "case_id",
        )

        query_text = _required_text(
            self.query_text,
            "query_text",
        )

        if not isinstance(
            self.language,
            BilingualLanguage,
        ):
            raise TypeError(
                "language must be a "
                "BilingualLanguage."
            )

        if (
            isinstance(
                self.relevant_urls,
                (str, bytes),
            )
        ):
            raise TypeError(
                "relevant_urls must be "
                "a sequence of URLs."
            )

        try:
            relevant_urls = tuple(
                _canonical_url(
                    url
                )
                for url
                in self.relevant_urls
            )
        except TypeError as exc:
            raise TypeError(
                "relevant_urls must be iterable."
            ) from exc

        if not relevant_urls:
            raise ValueError(
                "At least one relevant URL "
                "is required."
            )

        if (
            len(set(relevant_urls))
            != len(relevant_urls)
        ):
            raise ValueError(
                "relevant_urls must be unique "
                "after canonicalization."
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
                1 <= self.top_k <= 20
            )
        ):
            raise ValueError(
                "top_k must be between "
                "1 and 20."
            )

        object.__setattr__(
            self,
            "case_id",
            case_id,
        )

        object.__setattr__(
            self,
            "query_text",
            query_text,
        )

        object.__setattr__(
            self,
            "relevant_urls",
            relevant_urls,
        )

    def to_dict(
        self,
        *,
        include_query_text: bool = False,
    ) -> dict[str, object]:
        if not isinstance(
            include_query_text,
            bool,
        ):
            raise TypeError(
                "include_query_text "
                "must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "case_id": (
                self.case_id
            ),
            "language": (
                self.language.value
            ),
            "relevant_urls": list(
                self.relevant_urls
            ),
            "top_k": (
                self.top_k
            ),
        }

        if include_query_text:
            data[
                "query_text"
            ] = self.query_text

        return data


@dataclass(frozen=True)
class ControlledWebCaseResult:
    """Privacy-safe evaluation result for one gold case."""

    case_id: str

    request_id: str

    language: BilingualLanguage

    top_k: int

    relevant_count: int

    returned_count: int

    relevant_retrieved_count: int

    trusted_result_count: int

    untrusted_result_count: int

    hit_at_1: bool

    hit_at_k: bool

    recall: float

    reciprocal_rank: float

    zero_result: bool

    latency_ms: float

    version: str = (
        CONTROLLED_WEB_EVALUATION_VERSION
    )

    def __post_init__(self) -> None:
        _required_text(
            self.case_id,
            "case_id",
        )

        _required_text(
            self.request_id,
            "request_id",
        )

        if not isinstance(
            self.language,
            BilingualLanguage,
        ):
            raise TypeError(
                "language must be a "
                "BilingualLanguage."
            )

        for field_name in (
            "top_k",
            "relevant_count",
            "returned_count",
            "relevant_retrieved_count",
            "trusted_result_count",
            "untrusted_result_count",
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
                    f"{field_name} must be "
                    "a non-negative integer."
                )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if self.relevant_count < 1:
            raise ValueError(
                "relevant_count must be positive."
            )

        if (
            self.relevant_retrieved_count
            > self.relevant_count
        ):
            raise ValueError(
                "relevant_retrieved_count cannot "
                "exceed relevant_count."
            )

        if (
            self.trusted_result_count
            + self.untrusted_result_count
            != self.returned_count
        ):
            raise ValueError(
                "Trusted and untrusted counts "
                "must equal returned_count."
            )

        for field_name in (
            "hit_at_1",
            "hit_at_k",
            "zero_result",
        ):
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

        object.__setattr__(
            self,
            "recall",
            _rate(
                self.recall,
                "recall",
            ),
        )

        object.__setattr__(
            self,
            "reciprocal_rank",
            _rate(
                self.reciprocal_rank,
                "reciprocal_rank",
            ),
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

        object.__setattr__(
            self,
            "latency_ms",
            float(
                self.latency_ms
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "case_id": (
                self.case_id
            ),
            "request_id": (
                self.request_id
            ),
            "language": (
                self.language.value
            ),
            "top_k": (
                self.top_k
            ),
            "relevant_count": (
                self.relevant_count
            ),
            "returned_count": (
                self.returned_count
            ),
            "relevant_retrieved_count": (
                self.relevant_retrieved_count
            ),
            "trusted_result_count": (
                self.trusted_result_count
            ),
            "untrusted_result_count": (
                self.untrusted_result_count
            ),
            "hit_at_1": (
                self.hit_at_1
            ),
            "hit_at_k": (
                self.hit_at_k
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
        }


@dataclass(frozen=True)
class ControlledWebMetrics:
    """Aggregate controlled-web benchmark metrics."""

    case_count: int

    hit_at_1_rate: float

    hit_at_k_rate: float

    mean_recall: float

    mrr: float

    zero_result_rate: float

    untrusted_result_rate: float

    mean_latency_ms: float

    version: str = (
        CONTROLLED_WEB_EVALUATION_VERSION
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
            "hit_at_1_rate",
            "hit_at_k_rate",
            "mean_recall",
            "mrr",
            "zero_result_rate",
            "untrusted_result_rate",
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
            "version": (
                self.version
            ),
            "case_count": (
                self.case_count
            ),
            "hit_at_1_rate": (
                self.hit_at_1_rate
            ),
            "hit_at_k_rate": (
                self.hit_at_k_rate
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
            "untrusted_result_rate": (
                self.untrusted_result_rate
            ),
            "mean_latency_ms": (
                self.mean_latency_ms
            ),
        }


@dataclass(frozen=True)
class ControlledWebBenchmarkResult:
    """Complete privacy-safe benchmark output."""

    cases: tuple[
        ControlledWebCaseResult,
        ...
    ]

    metrics: ControlledWebMetrics

    version: str = (
        CONTROLLED_WEB_EVALUATION_VERSION
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
                "Benchmark requires at least "
                "one case result."
            )

        for case in cases:
            if not isinstance(
                case,
                ControlledWebCaseResult,
            ):
                raise TypeError(
                    "cases must contain only "
                    "ControlledWebCaseResult values."
                )

        case_ids = tuple(
            case.case_id
            for case
            in cases
        )

        if (
            len(set(case_ids))
            != len(case_ids)
        ):
            raise ValueError(
                "Benchmark case IDs "
                "must be unique."
            )

        if not isinstance(
            self.metrics,
            ControlledWebMetrics,
        ):
            raise TypeError(
                "metrics must be "
                "ControlledWebMetrics."
            )

        if (
            self.metrics.case_count
            != len(cases)
        ):
            raise ValueError(
                "Metric case count must match "
                "case results."
            )

        object.__setattr__(
            self,
            "cases",
            cases,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
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


def _validate_results(
    results: tuple[
        OfficialWebResult,
        ...
    ],
    *,
    top_k: int,
) -> None:
    for result in results:
        if not isinstance(
            result,
            OfficialWebResult,
        ):
            raise TypeError(
                "Retriever must return only "
                "OfficialWebResult values."
            )

    if len(results) > top_k:
        raise ValueError(
            "Retriever returned more results "
            "than requested top_k."
        )

    ranks = tuple(
        result.rank
        for result
        in results
    )

    expected_ranks = tuple(
        range(
            1,
            len(results) + 1,
        )
    )

    if ranks != expected_ranks:
        raise ValueError(
            "Returned ranks must be contiguous "
            "and ordered from one."
        )

    urls = tuple(
        _canonical_url(
            result.url
        )
        for result
        in results
    )

    if (
        len(set(urls))
        != len(urls)
    ):
        raise ValueError(
            "Retriever returned duplicate URLs."
        )


def evaluate_controlled_web(
    retriever: OfficialWebRetriever,
    *,
    cases: tuple[
        ControlledWebGoldCase,
        ...
    ],
    policy: SourceAllowlistPolicy,
) -> ControlledWebBenchmarkResult:
    """Evaluate controlled official-web retrieval without live assumptions."""

    if not isinstance(
        retriever,
        OfficialWebRetriever,
    ):
        raise TypeError(
            "retriever must implement "
            "OfficialWebRetriever."
        )

    if not isinstance(
        policy,
        SourceAllowlistPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "SourceAllowlistPolicy."
        )

    if isinstance(
        cases,
        (str, bytes),
    ):
        raise TypeError(
            "cases must be a sequence "
            "of gold cases."
        )

    try:
        case_values = tuple(
            cases
        )
    except TypeError as exc:
        raise TypeError(
            "cases must be iterable."
        ) from exc

    if not case_values:
        raise ValueError(
            "At least one gold case "
            "is required."
        )

    for case in case_values:
        if not isinstance(
            case,
            ControlledWebGoldCase,
        ):
            raise TypeError(
                "cases must contain only "
                "ControlledWebGoldCase values."
            )

    case_ids = tuple(
        case.case_id
        for case
        in case_values
    )

    if (
        len(set(case_ids))
        != len(case_ids)
    ):
        raise ValueError(
            "Gold case IDs must be unique."
        )

    outputs = []

    total_returned = 0
    total_untrusted = 0

    for case in case_values:
        request = OfficialWebRequest(
            query_text=(
                case.query_text
            ),
            language=(
                case.language
            ),
            top_k=(
                case.top_k
            ),
        )

        started = time.perf_counter()

        raw_results = (
            retriever.search(
                request
            )
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        try:
            results = tuple(
                raw_results
            )
        except TypeError as exc:
            raise TypeError(
                "Retriever results "
                "must be iterable."
            ) from exc

        _validate_results(
            results,
            top_k=(
                case.top_k
            ),
        )

        returned_urls = tuple(
            _canonical_url(
                result.url
            )
            for result
            in results
        )

        relevant_set = set(
            case.relevant_urls
        )

        relevant_positions = tuple(
            index
            for index, url
            in enumerate(
                returned_urls,
                start=1,
            )
            if url in relevant_set
        )

        relevant_retrieved = len(
            set(
                returned_urls
            )
            & relevant_set
        )

        trusted_count = 0

        for result in results:
            source_assessment = (
                assess_source_url(
                    result.url,
                    document_id=(
                        "WEB-EVAL-"
                        f"{result.result_id[:16]}"
                    ),
                    trace_id=(
                        request.request_id
                    ),
                    policy=policy,
                )
            )

            if source_assessment.allowed:
                trusted_count += 1

        untrusted_count = (
            len(results)
            - trusted_count
        )

        total_returned += len(
            results
        )

        total_untrusted += (
            untrusted_count
        )

        hit_at_1 = bool(
            relevant_positions
            and relevant_positions[
                0
            ] == 1
        )

        hit_at_k = bool(
            relevant_positions
        )

        recall = (
            relevant_retrieved
            / len(
                relevant_set
            )
        )

        reciprocal_rank = (
            1.0
            / relevant_positions[
                0
            ]
            if relevant_positions
            else 0.0
        )

        outputs.append(
            ControlledWebCaseResult(
                case_id=(
                    case.case_id
                ),
                request_id=(
                    request.request_id
                ),
                language=(
                    case.language
                ),
                top_k=(
                    case.top_k
                ),
                relevant_count=len(
                    relevant_set
                ),
                returned_count=len(
                    results
                ),
                relevant_retrieved_count=(
                    relevant_retrieved
                ),
                trusted_result_count=(
                    trusted_count
                ),
                untrusted_result_count=(
                    untrusted_count
                ),
                hit_at_1=hit_at_1,
                hit_at_k=hit_at_k,
                recall=recall,
                reciprocal_rank=(
                    reciprocal_rank
                ),
                zero_result=(
                    not results
                ),
                latency_ms=(
                    elapsed_ms
                ),
            )
        )

    output_tuple = tuple(
        outputs
    )

    metrics = ControlledWebMetrics(
        case_count=len(
            output_tuple
        ),
        hit_at_1_rate=mean(
            float(
                case.hit_at_1
            )
            for case
            in output_tuple
        ),
        hit_at_k_rate=mean(
            float(
                case.hit_at_k
            )
            for case
            in output_tuple
        ),
        mean_recall=mean(
            case.recall
            for case
            in output_tuple
        ),
        mrr=mean(
            case.reciprocal_rank
            for case
            in output_tuple
        ),
        zero_result_rate=mean(
            float(
                case.zero_result
            )
            for case
            in output_tuple
        ),
        untrusted_result_rate=(
            total_untrusted
            / total_returned
            if total_returned
            else 0.0
        ),
        mean_latency_ms=mean(
            case.latency_ms
            for case
            in output_tuple
        ),
    )

    return ControlledWebBenchmarkResult(
        cases=output_tuple,
        metrics=metrics,
    )
