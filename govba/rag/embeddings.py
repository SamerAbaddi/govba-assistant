"""Provider-independent embedding infrastructure for GovBA-GAR.

This module defines the contract used by semantic retrieval without coupling
GovBA to OpenAI, a particular embedding model, or a vector database.

It also provides deterministic vector validation and cosine-similarity
utilities used by offline tests and future production retrievers.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt
from numbers import Real
from typing import Protocol, TypeAlias, runtime_checkable


EmbeddingVector: TypeAlias = tuple[float, ...]


def normalize_embedding_vector(
    vector: Sequence[float],
    *,
    field_name: str = "vector",
) -> EmbeddingVector:
    """Validate and convert an embedding vector to immutable floats."""

    if isinstance(vector, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a numeric sequence."
        )

    normalized: list[float] = []

    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{field_name}[{index}] must be a real number."
            )

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise ValueError(
                f"{field_name}[{index}] must be finite."
            )

        normalized.append(numeric_value)

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return tuple(normalized)


def embedding_norm(
    vector: Sequence[float],
) -> float:
    """Return the Euclidean norm of a validated embedding vector."""

    normalized = normalize_embedding_vector(
        vector
    )

    return sqrt(
        sum(
            value * value
            for value in normalized
        )
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return cosine similarity between two non-zero vectors.

    The result is constrained to the mathematical interval [-1, 1].
    """

    left_vector = normalize_embedding_vector(
        left,
        field_name="left",
    )

    right_vector = normalize_embedding_vector(
        right,
        field_name="right",
    )

    if len(left_vector) != len(right_vector):
        raise ValueError(
            "Embedding vectors must have the same dimension."
        )

    left_norm = sqrt(
        sum(
            value * value
            for value in left_vector
        )
    )

    right_norm = sqrt(
        sum(
            value * value
            for value in right_vector
        )
    )

    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError(
            "Cosine similarity is undefined for a zero vector."
        )

    dot_product = sum(
        left_value * right_value
        for left_value, right_value
        in zip(
            left_vector,
            right_vector,
            strict=True,
        )
    )

    similarity = (
        dot_product
        / (left_norm * right_norm)
    )

    # Floating-point arithmetic can produce tiny excursions
    # outside the mathematical range.
    return max(
        -1.0,
        min(1.0, similarity),
    )


def cosine_to_unit_interval(
    similarity: float,
) -> float:
    """Map cosine similarity from [-1, 1] into retrieval score [0, 1]."""

    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, Real)
    ):
        raise TypeError(
            "similarity must be a real number."
        )

    value = float(similarity)

    if not isfinite(value):
        raise ValueError(
            "similarity must be finite."
        )

    if not -1.0 <= value <= 1.0:
        raise ValueError(
            "similarity must be between -1 and 1."
        )

    return (value + 1.0) / 2.0


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract implemented by every GovBA embedding provider."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """Return one embedding vector for each supplied document text."""
        ...

    def embed_query(
        self,
        text: str,
    ) -> Sequence[float]:
        """Return one embedding vector for a retrieval query."""
        ...
