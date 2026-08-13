"""Structured evidence-card contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from govba.rag.evidence import EvidenceChunk
from govba.rag.models import AuthoritativeSource


EVIDENCE_CARD_SCHEMA_VERSION = (
    "govba-evidence-card-v1"
)


class EvidenceVerificationState(
    str,
    Enum,
):
    """Verification state of retrieved evidence."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class EvidenceTemporalState(
    str,
    Enum,
):
    """Temporal assessment state of evidence."""

    UNASSESSED = "unassessed"
    CURRENT = "current"
    SUPERSEDED = "superseded"
    CONFLICTING = "conflicting"


def _optional_text(
    value: str | None,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be "
            "a string or None."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not "
            "be blank when supplied."
        )

    return value


def _date_text(
    value: date | None,
) -> str | None:
    if value is None:
        return None

    return value.isoformat()


def _build_card_id(
    source: AuthoritativeSource,
    chunk: EvidenceChunk,
) -> str:
    payload = (
        f"{EVIDENCE_CARD_SCHEMA_VERSION}"
        f"\x1f{source.document_id}"
        f"\x1f{chunk.chunk_id}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceCard:
    """Governance-aware representation of retrieved evidence."""

    source: AuthoritativeSource
    chunk: EvidenceChunk

    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    retrieval_method: str | None = None

    confidence: float | None = None

    verification_state: (
        EvidenceVerificationState
    ) = EvidenceVerificationState.UNVERIFIED

    temporal_state: (
        EvidenceTemporalState
    ) = EvidenceTemporalState.UNASSESSED

    trace_id: str | None = None

    schema_version: str = (
        EVIDENCE_CARD_SCHEMA_VERSION
    )

    card_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "source must be an "
                "AuthoritativeSource."
            )

        if not isinstance(
            self.chunk,
            EvidenceChunk,
        ):
            raise TypeError(
                "chunk must be an "
                "EvidenceChunk."
            )

        if (
            self.chunk.document_id
            != self.source.document_id
        ):
            raise ValueError(
                "Evidence chunk document_id "
                "does not match source."
            )

        if self.retrieval_rank is not None:
            if (
                isinstance(
                    self.retrieval_rank,
                    bool,
                )
                or not isinstance(
                    self.retrieval_rank,
                    int,
                )
                or self.retrieval_rank < 1
            ):
                raise ValueError(
                    "retrieval_rank must be "
                    "a positive integer or None."
                )

        if self.retrieval_score is not None:
            if (
                isinstance(
                    self.retrieval_score,
                    bool,
                )
                or not isinstance(
                    self.retrieval_score,
                    (int, float),
                )
                or not math.isfinite(
                    float(
                        self.retrieval_score
                    )
                )
            ):
                raise ValueError(
                    "retrieval_score must be "
                    "a finite number or None."
                )

            object.__setattr__(
                self,
                "retrieval_score",
                float(
                    self.retrieval_score
                ),
            )

        if self.confidence is not None:
            if (
                isinstance(
                    self.confidence,
                    bool,
                )
                or not isinstance(
                    self.confidence,
                    (int, float),
                )
                or not math.isfinite(
                    float(
                        self.confidence
                    )
                )
            ):
                raise ValueError(
                    "confidence must be "
                    "a finite number or None."
                )

            confidence = float(
                self.confidence
            )

            if not (
                0.0
                <= confidence
                <= 1.0
            ):
                raise ValueError(
                    "confidence must be "
                    "between 0 and 1."
                )

            object.__setattr__(
                self,
                "confidence",
                confidence,
            )

        retrieval_method = (
            _optional_text(
                self.retrieval_method,
                "retrieval_method",
            )
        )

        trace_id = _optional_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.verification_state,
            EvidenceVerificationState,
        ):
            raise TypeError(
                "verification_state must be an "
                "EvidenceVerificationState."
            )

        if not isinstance(
            self.temporal_state,
            EvidenceTemporalState,
        ):
            raise TypeError(
                "temporal_state must be an "
                "EvidenceTemporalState."
            )

        object.__setattr__(
            self,
            "retrieval_method",
            retrieval_method,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "card_id",
            _build_card_id(
                self.source,
                self.chunk,
            ),
        )

    @property
    def document_id(self) -> str:
        return self.source.document_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def title(self) -> str:
        return self.source.title

    @property
    def issuing_authority(self) -> str:
        return (
            self.source.issuing_authority
        )

    @property
    def official_source_url(
        self,
    ) -> str | None:
        return (
            self.source.official_source_url
        )

    @property
    def page_start(
        self,
    ) -> int | None:
        return self.chunk.page_start

    @property
    def page_end(
        self,
    ) -> int | None:
        return self.chunk.page_end

    @property
    def section_title(
        self,
    ) -> str | None:
        return self.chunk.section_title

    @property
    def section_path(
        self,
    ) -> tuple[str, ...]:
        return self.chunk.section_path

    @property
    def evidence_text(self) -> str:
        return self.chunk.text

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Return a serialization-friendly evidence-card representation."""

        return {
            "schema_version": (
                self.schema_version
            ),
            "card_id": self.card_id,
            "trace_id": self.trace_id,
            "document_id": (
                self.source.document_id
            ),
            "chunk_id": (
                self.chunk.chunk_id
            ),
            "title": (
                self.source.title
            ),
            "issuing_authority": (
                self.source.issuing_authority
            ),
            "document_type": (
                self.source.document_type.value
            ),
            "language": (
                self.chunk.language.value
            ),
            "jurisdiction": (
                self.source.jurisdiction
            ),
            "official_source_url": (
                self.source.official_source_url
            ),
            "version": (
                self.source.version
            ),
            "source_status": (
                self.source.status.value
            ),
            "publication_date": (
                _date_text(
                    self.source.publication_date
                )
            ),
            "effective_from": (
                _date_text(
                    self.source.effective_from
                )
            ),
            "effective_until": (
                _date_text(
                    self.source.effective_until
                )
            ),
            "supersedes": list(
                self.source.supersedes
            ),
            "superseded_by": list(
                self.source.superseded_by
            ),
            "page_start": (
                self.chunk.page_start
            ),
            "page_end": (
                self.chunk.page_end
            ),
            "section_title": (
                self.chunk.section_title
            ),
            "section_path": list(
                self.chunk.section_path
            ),
            "evidence_text": (
                self.chunk.text
            ),
            "retrieval_rank": (
                self.retrieval_rank
            ),
            "retrieval_score": (
                self.retrieval_score
            ),
            "retrieval_method": (
                self.retrieval_method
            ),
            "confidence": (
                self.confidence
            ),
            "verification_state": (
                self.verification_state.value
            ),
            "temporal_state": (
                self.temporal_state.value
            ),
        }
