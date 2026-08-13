"""OpenAI embedding adapter for GovBA-GAR.

This module implements the provider-independent EmbeddingProvider contract
using the OpenAI Embeddings API.

The adapter contains provider-specific behavior while semantic retrieval
remains independent of OpenAI itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from govba.rag.embeddings import (
    EmbeddingVector,
    normalize_embedding_vector,
)


DEFAULT_OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE = 128


class OpenAIEmbeddingProviderError(
    RuntimeError
):
    """Safe GovBA error for embedding-provider failures."""


def _require_text(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize one required text value."""

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} is required."
        )

    return value.strip()


class OpenAIEmbeddingProvider:
    """OpenAI implementation of the GovBA EmbeddingProvider contract."""

    def __init__(
        self,
        *,
        model: str = (
            DEFAULT_OPENAI_EMBEDDING_MODEL
        ),
        dimensions: int | None = None,
        batch_size: int = (
            DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE
        ),
        client: Any | None = None,
        api_key: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        self._model = _require_text(
            model,
            "model",
        )

        if dimensions is not None:
            if (
                not isinstance(dimensions, int)
                or isinstance(dimensions, bool)
                or dimensions < 1
            ):
                raise ValueError(
                    "dimensions must be a positive integer "
                    "when provided."
                )

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 256
        ):
            raise ValueError(
                "batch_size must be between 1 and 256."
            )

        if (
            isinstance(timeout, bool)
            or not isinstance(
                timeout,
                (int, float),
            )
            or timeout <= 0
        ):
            raise ValueError(
                "timeout must be greater than zero."
            )

        self._dimensions = dimensions
        self._batch_size = batch_size

        if client is None:
            self._client = OpenAI(
                api_key=api_key,
                timeout=float(timeout),
                max_retries=1,
            )
        else:
            self._client = client

    @property
    def model(self) -> str:
        """Return the configured embedding model."""

        return self._model

    @property
    def dimensions(self) -> int | None:
        """Return the requested embedding dimensions."""

        return self._dimensions

    @property
    def batch_size(self) -> int:
        """Return the maximum GovBA request batch size."""

        return self._batch_size

    def _request_embeddings(
        self,
        texts: tuple[str, ...],
    ) -> tuple[
        EmbeddingVector,
        ...
    ]:
        """Request and validate one provider embedding batch."""

        request: dict[str, Any] = {
            "model": self._model,
            "input": list(texts),
            "encoding_format": "float",
        }

        if self._dimensions is not None:
            request[
                "dimensions"
            ] = self._dimensions

        try:
            response = (
                self._client
                .embeddings
                .create(
                    **request
                )
            )
        except Exception as exc:
            raise OpenAIEmbeddingProviderError(
                "OpenAI embedding request failed."
            ) from exc

        response_data = tuple(
            response.data
        )

        if len(response_data) != len(texts):
            raise OpenAIEmbeddingProviderError(
                "Embedding response count did not match "
                "the request count."
            )

        indexed_vectors: dict[
            int,
            EmbeddingVector,
        ] = {}

        for item in response_data:
            index = getattr(
                item,
                "index",
                None,
            )

            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(texts)
                or index in indexed_vectors
            ):
                raise OpenAIEmbeddingProviderError(
                    "Embedding response contained "
                    "invalid indices."
                )

            raw_embedding = getattr(
                item,
                "embedding",
                None,
            )

            try:
                vector = (
                    normalize_embedding_vector(
                        raw_embedding,
                        field_name=(
                            f"embedding[{index}]"
                        ),
                    )
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise OpenAIEmbeddingProviderError(
                    "Embedding response contained "
                    "an invalid vector."
                ) from exc

            indexed_vectors[
                index
            ] = vector

        expected_indices = set(
            range(
                len(texts)
            )
        )

        if (
            set(indexed_vectors)
            != expected_indices
        ):
            raise OpenAIEmbeddingProviderError(
                "Embedding response indices were incomplete."
            )

        return tuple(
            indexed_vectors[index]
            for index in range(
                len(texts)
            )
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[
        EmbeddingVector,
        ...
    ]:
        """Embed document texts while preserving input ordering."""

        if isinstance(
            texts,
            (
                str,
                bytes,
            ),
        ):
            raise TypeError(
                "texts must be a sequence of strings."
            )

        normalized_texts = tuple(
            _require_text(
                text,
                f"texts[{index}]",
            )
            for index, text
            in enumerate(texts)
        )

        if not normalized_texts:
            return ()

        vectors: list[
            EmbeddingVector
        ] = []

        for start in range(
            0,
            len(normalized_texts),
            self._batch_size,
        ):
            batch = normalized_texts[
                start:
                start + self._batch_size
            ]

            vectors.extend(
                self._request_embeddings(
                    batch
                )
            )

        return tuple(
            vectors
        )

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingVector:
        """Embed one retrieval query."""

        normalized_text = (
            _require_text(
                text,
                "text",
            )
        )

        return self._request_embeddings(
            (
                normalized_text,
            )
        )[0]
