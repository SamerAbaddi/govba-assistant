"""Provider-independent retrieval contracts for GovBA-GAR.

The application depends on these abstractions rather than on a specific
vector database, embedding model, search engine, or AI provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from govba.rag.evidence import EvidenceChunk
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def _normalize_text_tuple(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    """Normalize, validate, and de-duplicate textual filter values."""

    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} values must be non-empty strings."
            )

        cleaned = value.strip()
        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            normalized.append(cleaned)

    return tuple(normalized)


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional governance-aware constraints for retrieval."""

    document_ids: tuple[str, ...] = field(
        default_factory=tuple
    )

    document_types: tuple[DocumentType, ...] = field(
        default_factory=tuple
    )

    languages: tuple[SourceLanguage, ...] = field(
        default_factory=tuple
    )

    statuses: tuple[SourceStatus, ...] = field(
        default_factory=tuple
    )

    issuing_authorities: tuple[str, ...] = field(
        default_factory=tuple
    )

    jurisdiction: str = ""

    require_official_source: bool = False

    def __post_init__(self) -> None:
        """Normalize filter metadata."""

        object.__setattr__(
            self,
            "document_ids",
            _normalize_text_tuple(
                self.document_ids,
                "document_ids",
            ),
        )

        object.__setattr__(
            self,
            "issuing_authorities",
            _normalize_text_tuple(
                self.issuing_authorities,
                "issuing_authorities",
            ),
        )

        if self.jurisdiction:
            object.__setattr__(
                self,
                "jurisdiction",
                self.jurisdiction.strip(),
            )

    def allows(
        self,
        source: AuthoritativeSource,
    ) -> bool:
        """Return whether a source satisfies these filters."""

        if (
            self.document_ids
            and source.document_id not in self.document_ids
        ):
            return False

        if (
            self.document_types
            and source.document_type not in self.document_types
        ):
            return False

        if (
            self.languages
            and source.language not in self.languages
        ):
            return False

        if (
            self.statuses
            and source.status not in self.statuses
        ):
            return False

        if self.issuing_authorities:
            allowed_authorities = {
                value.casefold()
                for value in self.issuing_authorities
            }

            if (
                source.issuing_authority.casefold()
                not in allowed_authorities
            ):
                return False

        if (
            self.jurisdiction
            and source.jurisdiction.casefold()
            != self.jurisdiction.casefold()
        ):
            return False

        if (
            self.require_official_source
            and not source.official_source_url
        ):
            return False

        return True


@dataclass(frozen=True)
class RetrievalQuery:
    """One normalized request to a GovBA retrieval engine."""

    text: str

    top_k: int = 5

    filters: RetrievalFilters = field(
        default_factory=RetrievalFilters
    )

    def __post_init__(self) -> None:
        """Validate the retrieval request."""

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                "Retrieval query text is required."
            )

        object.__setattr__(
            self,
            "text",
            self.text.strip(),
        )

        if not 1 <= self.top_k <= 100:
            raise ValueError(
                "top_k must be between 1 and 100."
            )


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked evidence result returned by a retriever."""

    chunk: EvidenceChunk

    score: float
    rank: int

    retrieval_method: str

    raw_score: float | None = None

    matched_terms: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Validate normalized retrieval-result semantics."""

        if not isfinite(self.score):
            raise ValueError(
                "score must be finite."
            )

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "score must be between 0 and 1."
            )

        if self.rank < 1:
            raise ValueError(
                "rank must be at least 1."
            )

        if (
            not isinstance(
                self.retrieval_method,
                str,
            )
            or not self.retrieval_method.strip()
        ):
            raise ValueError(
                "retrieval_method is required."
            )

        object.__setattr__(
            self,
            "retrieval_method",
            self.retrieval_method.strip(),
        )

        normalized_terms = _normalize_text_tuple(
            self.matched_terms,
            "matched_terms",
        )

        object.__setattr__(
            self,
            "matched_terms",
            normalized_terms,
        )

        if (
            self.raw_score is not None
            and not isfinite(self.raw_score)
        ):
            raise ValueError(
                "raw_score must be finite when provided."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible retrieval metadata."""

        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "raw_score": self.raw_score,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "matched_terms": list(
                self.matched_terms
            ),
        }


@runtime_checkable
class Retriever(Protocol):
    """Contract implemented by every GovBA retrieval engine."""

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Return ranked evidence for one retrieval request."""
        ...
