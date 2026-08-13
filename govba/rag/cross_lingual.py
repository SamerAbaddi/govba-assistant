"""Language-aware cross-lingual retrieval for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass, replace

from govba.rag.language import (
    BilingualLanguage,
    LanguageRoute,
)
from govba.rag.models import (
    SourceLanguage,
)
from govba.rag.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)


CROSS_LINGUAL_RETRIEVAL_VERSION = (
    "govba-cross-lingual-retrieval-v1"
)

CROSS_LINGUAL_RETRIEVAL_METHOD = (
    "cross-lingual-v1"
)

DEFAULT_CROSS_LINGUAL_CANDIDATE_POOL = 100


def _source_languages_for_lane(
    language: BilingualLanguage,
) -> tuple[
    SourceLanguage,
    ...
]:
    if (
        language
        is BilingualLanguage.ARABIC
    ):
        return (
            SourceLanguage.ARABIC,
            SourceLanguage.BILINGUAL,
        )

    if (
        language
        is BilingualLanguage.ENGLISH
    ):
        return (
            SourceLanguage.ENGLISH,
            SourceLanguage.BILINGUAL,
        )

    raise ValueError(
        "Unsupported bilingual retrieval language."
    )


def _chunk_matches_lane(
    language: SourceLanguage,
    lane: BilingualLanguage,
) -> bool:
    if (
        language
        is SourceLanguage.BILINGUAL
    ):
        return True

    if (
        lane
        is BilingualLanguage.ARABIC
    ):
        return (
            language
            is SourceLanguage.ARABIC
        )

    if (
        lane
        is BilingualLanguage.ENGLISH
    ):
        return (
            language
            is SourceLanguage.ENGLISH
        )

    return False


def _lane_filters(
    filters: RetrievalFilters,
    lane: BilingualLanguage,
) -> RetrievalFilters | None:
    if not isinstance(
        filters,
        RetrievalFilters,
    ):
        raise TypeError(
            "filters must be RetrievalFilters."
        )

    allowed = (
        _source_languages_for_lane(
            lane
        )
    )

    if filters.languages:
        existing = set(
            filters.languages
        )

        languages = tuple(
            language
            for language in allowed
            if language in existing
        )

        if not languages:
            return None

    else:
        languages = allowed

    return replace(
        filters,
        languages=languages,
    )


@dataclass(frozen=True)
class CrossLingualLaneSummary:
    """Privacy-safe summary of one retrieval-language lane."""

    language: BilingualLanguage

    source_languages: tuple[
        SourceLanguage,
        ...
    ]

    requested_top_k: int

    raw_result_count: int

    accepted_result_count: int

    skipped: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.language,
            BilingualLanguage,
        ):
            raise TypeError(
                "language must be a "
                "BilingualLanguage."
            )

        try:
            source_languages = tuple(
                self.source_languages
            )
        except TypeError as exc:
            raise TypeError(
                "source_languages must be iterable."
            ) from exc

        for language in source_languages:
            if not isinstance(
                language,
                SourceLanguage,
            ):
                raise TypeError(
                    "source_languages must contain "
                    "only SourceLanguage values."
                )

        if (
            len(set(source_languages))
            != len(source_languages)
        ):
            raise ValueError(
                "source_languages must not "
                "contain duplicates."
            )

        for field_name in (
            "requested_top_k",
            "raw_result_count",
            "accepted_result_count",
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
            self.accepted_result_count
            > self.raw_result_count
        ):
            raise ValueError(
                "accepted_result_count cannot "
                "exceed raw_result_count."
            )

        if not isinstance(
            self.skipped,
            bool,
        ):
            raise TypeError(
                "skipped must be boolean."
            )

        object.__setattr__(
            self,
            "source_languages",
            source_languages,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "language": (
                self.language.value
            ),
            "source_languages": [
                language.value
                for language
                in self.source_languages
            ],
            "requested_top_k": (
                self.requested_top_k
            ),
            "raw_result_count": (
                self.raw_result_count
            ),
            "accepted_result_count": (
                self.accepted_result_count
            ),
            "skipped": (
                self.skipped
            ),
        }


@dataclass(frozen=True)
class CrossLingualRetrievalReport:
    """Auditable result of one bilingual retrieval operation."""

    route: LanguageRoute

    results: tuple[
        RetrievalResult,
        ...
    ]

    lane_summaries: tuple[
        CrossLingualLaneSummary,
        ...
    ]

    version: str = (
        CROSS_LINGUAL_RETRIEVAL_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.route,
            LanguageRoute,
        ):
            raise TypeError(
                "route must be a LanguageRoute."
            )

        try:
            results = tuple(
                self.results
            )
            lane_summaries = tuple(
                self.lane_summaries
            )
        except TypeError as exc:
            raise TypeError(
                "results and lane_summaries "
                "must be iterable."
            ) from exc

        for result in results:
            if not isinstance(
                result,
                RetrievalResult,
            ):
                raise TypeError(
                    "results must contain only "
                    "RetrievalResult values."
                )

        for summary in lane_summaries:
            if not isinstance(
                summary,
                CrossLingualLaneSummary,
            ):
                raise TypeError(
                    "lane_summaries must contain only "
                    "CrossLingualLaneSummary values."
                )

        chunk_ids = tuple(
            result.chunk.chunk_id
            for result in results
        )

        if (
            len(set(chunk_ids))
            != len(chunk_ids)
        ):
            raise ValueError(
                "Cross-lingual results must "
                "have unique chunk IDs."
            )

        expected_ranks = tuple(
            range(
                1,
                len(results) + 1,
            )
        )

        actual_ranks = tuple(
            result.rank
            for result in results
        )

        if (
            actual_ranks
            != expected_ranks
        ):
            raise ValueError(
                "Cross-lingual result ranks "
                "must be contiguous."
            )

        object.__setattr__(
            self,
            "results",
            results,
        )

        object.__setattr__(
            self,
            "lane_summaries",
            lane_summaries,
        )

    @property
    def result_count(
        self,
    ) -> int:
        return len(
            self.results
        )

    @property
    def languages_searched(
        self,
    ) -> tuple[
        BilingualLanguage,
        ...
    ]:
        return tuple(
            summary.language
            for summary
            in self.lane_summaries
            if not summary.skipped
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize metadata without query or evidence text."""

        return {
            "version": (
                self.version
            ),
            "route_id": (
                self.route.route_id
            ),
            "result_count": (
                self.result_count
            ),
            "languages_searched": [
                language.value
                for language
                in self.languages_searched
            ],
            "lane_summaries": [
                summary.to_dict()
                for summary
                in self.lane_summaries
            ],
            "results": [
                {
                    "chunk_id": (
                        result.chunk.chunk_id
                    ),
                    "document_id": (
                        result.chunk.document_id
                    ),
                    "rank": (
                        result.rank
                    ),
                    "score": (
                        result.score
                    ),
                    "retrieval_method": (
                        result.retrieval_method
                    ),
                }
                for result
                in self.results
            ],
        }


class CrossLingualRetriever:
    """Language-aware wrapper around an existing GovBA retriever."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        route: LanguageRoute,
        candidate_pool_size: int = (
            DEFAULT_CROSS_LINGUAL_CANDIDATE_POOL
        ),
    ) -> None:
        if not isinstance(
            retriever,
            Retriever,
        ):
            raise TypeError(
                "retriever must implement "
                "the Retriever protocol."
            )

        if not isinstance(
            route,
            LanguageRoute,
        ):
            raise TypeError(
                "route must be a LanguageRoute."
            )

        if (
            isinstance(
                candidate_pool_size,
                bool,
            )
            or not isinstance(
                candidate_pool_size,
                int,
            )
            or not (
                1
                <= candidate_pool_size
                <= 100
            )
        ):
            raise ValueError(
                "candidate_pool_size must be "
                "between 1 and 100."
            )

        self._retriever = retriever
        self._route = route
        self._candidate_pool_size = (
            candidate_pool_size
        )

    @property
    def route(
        self,
    ) -> LanguageRoute:
        return self._route

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[
        RetrievalResult
    ]:
        """Return ordinary RetrievalResult values for pipeline compatibility."""

        return list(
            self.retrieve_with_report(
                query
            ).results
        )

    def retrieve_with_report(
        self,
        query: RetrievalQuery,
    ) -> CrossLingualRetrievalReport:
        """Retrieve each language lane and fuse deterministically."""

        if not isinstance(
            query,
            RetrievalQuery,
        ):
            raise TypeError(
                "query must be a RetrievalQuery."
            )

        if not isinstance(
            query.filters,
            RetrievalFilters,
        ):
            raise TypeError(
                "query.filters must be "
                "RetrievalFilters."
            )

        candidate_top_k = min(
            100,
            max(
                query.top_k,
                self._candidate_pool_size,
            ),
        )

        aggregates: dict[
            str,
            dict[str, object],
        ] = {}

        lane_summaries = []

        for lane_priority, lane in enumerate(
            self._route.retrieval_languages
        ):
            filters = _lane_filters(
                query.filters,
                lane,
            )

            if filters is None:
                lane_summaries.append(
                    CrossLingualLaneSummary(
                        language=lane,
                        source_languages=(),
                        requested_top_k=(
                            candidate_top_k
                        ),
                        raw_result_count=0,
                        accepted_result_count=0,
                        skipped=True,
                    )
                )

                continue

            lane_query = RetrievalQuery(
                text=query.text,
                top_k=candidate_top_k,
                filters=filters,
            )

            raw_results = tuple(
                self._retriever.retrieve(
                    lane_query
                )
            )

            for result in raw_results:
                if not isinstance(
                    result,
                    RetrievalResult,
                ):
                    raise TypeError(
                        "Underlying retriever must "
                        "return RetrievalResult values."
                    )

            accepted = tuple(
                sorted(
                    (
                        result
                        for result
                        in raw_results
                        if _chunk_matches_lane(
                            result.chunk.language,
                            lane,
                        )
                    ),
                    key=lambda result: (
                        -result.score,
                        result.rank,
                        result.chunk.document_id,
                        result.chunk.chunk_index,
                        result.chunk.chunk_id,
                    ),
                )
            )

            lane_summaries.append(
                CrossLingualLaneSummary(
                    language=lane,
                    source_languages=(
                        filters.languages
                    ),
                    requested_top_k=(
                        candidate_top_k
                    ),
                    raw_result_count=len(
                        raw_results
                    ),
                    accepted_result_count=len(
                        accepted
                    ),
                )
            )

            for lane_rank, result in enumerate(
                accepted,
                start=1,
            ):
                chunk_id = (
                    result.chunk.chunk_id
                )

                selection_key = (
                    -result.score,
                    lane_priority,
                    lane_rank,
                    result.rank,
                    result.chunk.document_id,
                    result.chunk.chunk_index,
                    result.chunk.chunk_id,
                )

                aggregate = (
                    aggregates.get(
                        chunk_id
                    )
                )

                if aggregate is None:
                    aggregates[
                        chunk_id
                    ] = {
                        "result": result,
                        "selection_key": (
                            selection_key
                        ),
                        "matched_terms": list(
                            result.matched_terms
                        ),
                        "matched_seen": set(
                            result.matched_terms
                        ),
                    }

                    continue

                matched_terms = aggregate[
                    "matched_terms"
                ]
                matched_seen = aggregate[
                    "matched_seen"
                ]

                for term in (
                    result.matched_terms
                ):
                    if term not in matched_seen:
                        matched_seen.add(
                            term
                        )
                        matched_terms.append(
                            term
                        )

                if (
                    selection_key
                    < aggregate[
                        "selection_key"
                    ]
                ):
                    aggregate[
                        "result"
                    ] = result

                    aggregate[
                        "selection_key"
                    ] = selection_key

        ranked = sorted(
            aggregates.values(),
            key=lambda item: (
                item[
                    "selection_key"
                ]
            ),
        )

        final_results = []

        for rank, aggregate in enumerate(
            ranked[
                : query.top_k
            ],
            start=1,
        ):
            result = aggregate[
                "result"
            ]

            final_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                    raw_score=(
                        result.raw_score
                    ),
                    rank=rank,
                    retrieval_method=(
                        f"{CROSS_LINGUAL_RETRIEVAL_METHOD}:"
                        f"{result.retrieval_method}"
                    ),
                    matched_terms=tuple(
                        aggregate[
                            "matched_terms"
                        ]
                    ),
                )
            )

        return CrossLingualRetrievalReport(
            route=self._route,
            results=tuple(
                final_results
            ),
            lane_summaries=tuple(
                lane_summaries
            ),
        )
