"""Deterministic lexical retrieval for GovBA-GAR.

This module provides a transparent offline retrieval baseline.
It performs no embedding generation, network operation, or AI request.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from govba.rag.evidence import EvidenceChunk
from govba.rag.models import AuthoritativeSource
from govba.rag.retrieval import RetrievalQuery, RetrievalResult


LEXICAL_RETRIEVAL_METHOD = "lexical-v1"

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def normalize_lexical_text(text: str) -> str:
    """Return normalized Unicode text for deterministic matching."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()

    return " ".join(normalized.split())


def tokenize_lexical(text: str) -> tuple[str, ...]:
    """Return deterministic Unicode-aware lexical tokens."""

    normalized = normalize_lexical_text(text)

    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(normalized)
        if token
    )


def _unique_in_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """De-duplicate strings while preserving first occurrence."""

    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return tuple(result)


class LexicalRetriever:
    """Deterministic governance-aware lexical evidence retriever."""

    def __init__(
        self,
        *,
        sources: Iterable[AuthoritativeSource],
        chunks: Iterable[EvidenceChunk],
    ) -> None:
        source_list = tuple(sources)
        chunk_list = tuple(chunks)

        self._sources: dict[str, AuthoritativeSource] = {}

        for source in source_list:
            if source.document_id in self._sources:
                raise ValueError(
                    "Duplicate authoritative source document_id: "
                    f"{source.document_id}"
                )

            self._sources[source.document_id] = source

        seen_chunk_ids: set[str] = set()

        for chunk in chunk_list:
            if chunk.document_id not in self._sources:
                raise ValueError(
                    "Evidence chunk references an unknown document_id: "
                    f"{chunk.document_id}"
                )

            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    "Duplicate evidence chunk_id: "
                    f"{chunk.chunk_id}"
                )

            seen_chunk_ids.add(chunk.chunk_id)

        self._chunks = chunk_list

    @property
    def source_count(self) -> int:
        """Return the number of registered authoritative sources."""

        return len(self._sources)

    @property
    def chunk_count(self) -> int:
        """Return the number of registered evidence chunks."""

        return len(self._chunks)

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Return deterministically ranked lexical evidence."""

        query_tokens = tokenize_lexical(query.text)

        unique_query_tokens = _unique_in_order(
            query_tokens
        )

        if not unique_query_tokens:
            return []

        normalized_query = normalize_lexical_text(
            query.text
        )

        candidates: list[
            tuple[
                float,
                float,
                EvidenceChunk,
                tuple[str, ...],
            ]
        ] = []

        for chunk in self._chunks:
            source = self._sources[
                chunk.document_id
            ]

            if not query.filters.allows(source):
                continue

            chunk_tokens = tokenize_lexical(
                chunk.text
            )

            if not chunk_tokens:
                continue

            frequencies = Counter(chunk_tokens)

            matched_terms = tuple(
                term
                for term in unique_query_tokens
                if term in frequencies
            )

            if not matched_terms:
                continue

            coverage = (
                len(matched_terms)
                / len(unique_query_tokens)
            )

            frequency_strength = (
                sum(
                    min(frequencies[term], 3) / 3
                    for term in matched_terms
                )
                / len(unique_query_tokens)
            )

            normalized_chunk = normalize_lexical_text(
                chunk.text
            )

            phrase_match = (
                1.0
                if normalized_query
                and normalized_query in normalized_chunk
                else 0.0
            )

            score = (
                0.75 * coverage
                + 0.15 * frequency_strength
                + 0.10 * phrase_match
            )

            score = max(
                0.0,
                min(1.0, score),
            )

            raw_score = float(
                len(matched_terms)
                + sum(
                    min(frequencies[term], 3) / 3
                    for term in matched_terms
                )
            )

            candidates.append(
                (
                    score,
                    raw_score,
                    chunk,
                    matched_terms,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2].document_id,
                item[2].chunk_index,
                item[2].chunk_id,
            )
        )

        results: list[RetrievalResult] = []

        for rank, candidate in enumerate(
            candidates[: query.top_k],
            start=1,
        ):
            (
                score,
                raw_score,
                chunk,
                matched_terms,
            ) = candidate

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    raw_score=raw_score,
                    rank=rank,
                    retrieval_method=LEXICAL_RETRIEVAL_METHOD,
                    matched_terms=matched_terms,
                )
            )

        return results
