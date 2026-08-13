"""Provider-independent semantic retrieval for GovBA-GAR.

The semantic retriever operates over precomputed evidence embeddings and a
query embedding supplied through the EmbeddingProvider contract.

This module contains no provider-specific API calls and performs no network
operations itself.
"""

from __future__ import annotations

from collections.abc import Iterable

from govba.rag.embeddings import (
    EmbeddingProvider,
    EmbeddingVector,
    cosine_similarity,
    cosine_to_unit_interval,
    embedding_norm,
    normalize_embedding_vector,
)
from govba.rag.evidence import EvidenceChunk
from govba.rag.models import AuthoritativeSource
from govba.rag.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


SEMANTIC_RETRIEVAL_METHOD = "semantic-v1"


class SemanticRetriever:
    """Deterministic provider-independent semantic evidence retriever."""

    def __init__(
        self,
        *,
        sources: Iterable[AuthoritativeSource],
        chunks: Iterable[EvidenceChunk],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        source_list = tuple(sources)
        chunk_list = tuple(chunks)

        if not isinstance(
            embedding_provider,
            EmbeddingProvider,
        ):
            raise TypeError(
                "embedding_provider must satisfy EmbeddingProvider."
            )

        self._embedding_provider = embedding_provider

        self._sources: dict[
            str,
            AuthoritativeSource,
        ] = {}

        for source in source_list:
            if source.document_id in self._sources:
                raise ValueError(
                    "Duplicate authoritative source document_id: "
                    f"{source.document_id}"
                )

            self._sources[
                source.document_id
            ] = source

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

            seen_chunk_ids.add(
                chunk.chunk_id
            )

        self._chunks = chunk_list

        self._chunk_vectors: tuple[
            EmbeddingVector,
            ...
        ] = ()

        self._embedding_dimension: int | None = None

        if not self._chunks:
            return

        raw_vectors = tuple(
            embedding_provider.embed_documents(
                tuple(
                    chunk.text
                    for chunk in self._chunks
                )
            )
        )

        if len(raw_vectors) != len(self._chunks):
            raise ValueError(
                "Embedding provider returned a document-vector count "
                "that does not match the number of evidence chunks."
            )

        normalized_vectors: list[
            EmbeddingVector
        ] = []

        expected_dimension: int | None = None

        for index, raw_vector in enumerate(
            raw_vectors
        ):
            vector = normalize_embedding_vector(
                raw_vector,
                field_name=(
                    f"document_embedding[{index}]"
                ),
            )

            if embedding_norm(vector) == 0.0:
                raise ValueError(
                    "Document embedding vectors cannot be zero vectors."
                )

            if expected_dimension is None:
                expected_dimension = len(vector)

            elif len(vector) != expected_dimension:
                raise ValueError(
                    "All document embedding vectors must have "
                    "the same dimension."
                )

            normalized_vectors.append(
                vector
            )

        self._chunk_vectors = tuple(
            normalized_vectors
        )

        self._embedding_dimension = (
            expected_dimension
        )

    @property
    def source_count(self) -> int:
        """Return the number of registered authoritative sources."""

        return len(self._sources)

    @property
    def chunk_count(self) -> int:
        """Return the number of registered evidence chunks."""

        return len(self._chunks)

    @property
    def embedding_dimension(
        self,
    ) -> int | None:
        """Return the semantic embedding dimension."""

        return self._embedding_dimension

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Return deterministically ranked semantic evidence."""

        if not self._chunks:
            return []

        query_vector = normalize_embedding_vector(
            self._embedding_provider.embed_query(
                query.text
            ),
            field_name="query_embedding",
        )

        if embedding_norm(query_vector) == 0.0:
            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        if (
            self._embedding_dimension is None
            or len(query_vector)
            != self._embedding_dimension
        ):
            raise ValueError(
                "Query embedding dimension does not match "
                "document embeddings."
            )

        candidates: list[
            tuple[
                float,
                float,
                EvidenceChunk,
            ]
        ] = []

        for chunk, chunk_vector in zip(
            self._chunks,
            self._chunk_vectors,
            strict=True,
        ):
            source = self._sources[
                chunk.document_id
            ]

            if not query.filters.allows(
                source
            ):
                continue

            similarity = cosine_similarity(
                query_vector,
                chunk_vector,
            )

            score = cosine_to_unit_interval(
                similarity
            )

            candidates.append(
                (
                    score,
                    similarity,
                    chunk,
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

        results: list[
            RetrievalResult
        ] = []

        for rank, candidate in enumerate(
            candidates[: query.top_k],
            start=1,
        ):
            (
                score,
                similarity,
                chunk,
            ) = candidate

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    raw_score=similarity,
                    rank=rank,
                    retrieval_method=(
                        SEMANTIC_RETRIEVAL_METHOD
                    ),
                    matched_terms=(),
                )
            )

        return results
