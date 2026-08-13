"""Deterministic hybrid rank fusion for GovBA-GAR.

This module combines multiple provider-independent retrievers using weighted
Reciprocal Rank Fusion (RRF). It intentionally fuses ranked evidence rather
than directly averaging retriever-specific scores whose scales may differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from govba.rag.retrieval import (
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)


HYBRID_RETRIEVAL_METHOD = "hybrid-rrf-v1"

DEFAULT_RRF_K = 60
DEFAULT_HYBRID_CANDIDATE_POOL = 20


def _normalize_component_name(
    value: str,
) -> str:
    """Validate and normalize a retriever component name."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Hybrid retriever component name is required."
        )

    return value.strip()


def _normalize_positive_weight(
    value: float,
    *,
    field_name: str,
) -> float:
    """Return a validated positive finite retriever weight."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            f"{field_name} must be a real number."
        )

    numeric = float(value)

    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            f"{field_name} must be finite and greater than zero."
        )

    return numeric


class HybridRetriever:
    """Fuse ranked evidence from multiple retrievers using weighted RRF."""

    def __init__(
        self,
        *,
        retrievers: Mapping[str, Retriever],
        weights: Mapping[str, float] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool_size: int = DEFAULT_HYBRID_CANDIDATE_POOL,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError(
                "HybridRetriever requires at least two retrievers."
            )

        if (
            not isinstance(rrf_k, int)
            or isinstance(rrf_k, bool)
            or rrf_k < 1
        ):
            raise ValueError(
                "rrf_k must be a positive integer."
            )

        if (
            not isinstance(candidate_pool_size, int)
            or isinstance(candidate_pool_size, bool)
            or not 1 <= candidate_pool_size <= 100
        ):
            raise ValueError(
                "candidate_pool_size must be between 1 and 100."
            )

        components: list[
            tuple[
                str,
                Retriever,
            ]
        ] = []

        seen_names: set[str] = set()

        for raw_name, retriever in retrievers.items():
            name = _normalize_component_name(
                raw_name
            )

            name_key = name.casefold()

            if name_key in seen_names:
                raise ValueError(
                    "Hybrid retriever component names must be unique."
                )

            if not isinstance(
                retriever,
                Retriever,
            ):
                raise TypeError(
                    f"Component '{name}' does not satisfy "
                    "the Retriever contract."
                )

            seen_names.add(
                name_key
            )

            components.append(
                (
                    name,
                    retriever,
                )
            )

        if weights is None:
            equal_weight = (
                1.0 / len(components)
            )

            normalized_weights = {
                name.casefold(): equal_weight
                for name, _ in components
            }

        else:
            supplied_weights: dict[
                str,
                float
            ] = {}

            for raw_name, raw_weight in weights.items():
                name = _normalize_component_name(
                    raw_name
                )

                key = name.casefold()

                if key in supplied_weights:
                    raise ValueError(
                        "Hybrid weight names must be unique."
                    )

                supplied_weights[key] = (
                    _normalize_positive_weight(
                        raw_weight,
                        field_name=(
                            f"weight[{name}]"
                        ),
                    )
                )

            component_keys = {
                name.casefold()
                for name, _ in components
            }

            if set(supplied_weights) != component_keys:
                raise ValueError(
                    "Hybrid weight names must exactly match "
                    "retriever component names."
                )

            total_weight = sum(
                supplied_weights.values()
            )

            normalized_weights = {
                key: value / total_weight
                for key, value
                in supplied_weights.items()
            }

        self._components = tuple(
            components
        )

        self._weights = {
            name: normalized_weights[
                name.casefold()
            ]
            for name, _ in components
        }

        self._rrf_k = rrf_k
        self._candidate_pool_size = (
            candidate_pool_size
        )

    @property
    def component_names(
        self,
    ) -> tuple[str, ...]:
        """Return component retriever names in deterministic order."""

        return tuple(
            name
            for name, _ in self._components
        )

    @property
    def weights(
        self,
    ) -> dict[str, float]:
        """Return normalized component weights."""

        return dict(
            self._weights
        )

    @property
    def rrf_k(self) -> int:
        """Return the Reciprocal Rank Fusion constant."""

        return self._rrf_k

    @property
    def candidate_pool_size(
        self,
    ) -> int:
        """Return candidate depth requested from each retriever."""

        return self._candidate_pool_size

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Return deterministically fused hybrid evidence."""

        candidate_top_k = max(
            query.top_k,
            self._candidate_pool_size,
        )

        candidate_top_k = min(
            candidate_top_k,
            100,
        )

        candidate_query = RetrievalQuery(
            text=query.text,
            top_k=candidate_top_k,
            filters=query.filters,
        )

        aggregates: dict[
            str,
            dict,
        ] = {}

        for component_name, retriever in self._components:
            weight = self._weights[
                component_name
            ]

            component_results = sorted(
                retriever.retrieve(
                    candidate_query
                ),
                key=lambda result: (
                    result.rank,
                    result.chunk.chunk_id,
                ),
            )[:candidate_top_k]

            seen_chunk_ids: set[str] = set()

            for result in component_results:
                chunk_id = (
                    result.chunk.chunk_id
                )

                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(
                    chunk_id
                )

                contribution = (
                    weight
                    / (
                        self._rrf_k
                        + result.rank
                    )
                )

                if chunk_id not in aggregates:
                    aggregates[
                        chunk_id
                    ] = {
                        "chunk": result.chunk,
                        "raw_score": 0.0,
                        "best_rank": result.rank,
                        "max_component_score": (
                            result.score
                        ),
                        "component_count": 0,
                        "matched_terms": [],
                        "matched_seen": set(),
                    }

                aggregate = aggregates[
                    chunk_id
                ]

                aggregate[
                    "raw_score"
                ] += contribution

                aggregate[
                    "best_rank"
                ] = min(
                    aggregate["best_rank"],
                    result.rank,
                )

                aggregate[
                    "max_component_score"
                ] = max(
                    aggregate[
                        "max_component_score"
                    ],
                    result.score,
                )

                aggregate[
                    "component_count"
                ] += 1

                for term in result.matched_terms:
                    if (
                        term
                        not in aggregate[
                            "matched_seen"
                        ]
                    ):
                        aggregate[
                            "matched_seen"
                        ].add(
                            term
                        )

                        aggregate[
                            "matched_terms"
                        ].append(
                            term
                        )

        if not aggregates:
            return []

        ranked = sorted(
            aggregates.values(),
            key=lambda item: (
                -item["raw_score"],
                -item["component_count"],
                item["best_rank"],
                -item["max_component_score"],
                item["chunk"].document_id,
                item["chunk"].chunk_index,
                item["chunk"].chunk_id,
            ),
        )

        maximum_rrf_score = (
            1.0
            / (
                self._rrf_k
                + 1
            )
        )

        results: list[
            RetrievalResult
        ] = []

        for rank, aggregate in enumerate(
            ranked[: query.top_k],
            start=1,
        ):
            normalized_score = (
                aggregate["raw_score"]
                / maximum_rrf_score
            )

            normalized_score = max(
                0.0,
                min(
                    1.0,
                    normalized_score,
                ),
            )

            results.append(
                RetrievalResult(
                    chunk=aggregate[
                        "chunk"
                    ],
                    score=normalized_score,
                    raw_score=aggregate[
                        "raw_score"
                    ],
                    rank=rank,
                    retrieval_method=(
                        HYBRID_RETRIEVAL_METHOD
                    ),
                    matched_terms=tuple(
                        aggregate[
                            "matched_terms"
                        ]
                    ),
                )
            )

        return results
