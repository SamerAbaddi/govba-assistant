"""Evidence-card construction from retrieval results for GovBA-GAR."""

from __future__ import annotations

from collections.abc import Iterable

from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.models import (
    AuthoritativeSource,
)
from govba.rag.retrieval import (
    RetrievalResult,
)


EVIDENCE_CARD_BUILDER_VERSION = (
    "govba-evidence-card-builder-v1"
)


class EvidenceCardBuilder:
    """Build EvidenceCard objects from validated retrieval results."""

    def __init__(
        self,
        sources: Iterable[
            AuthoritativeSource
        ],
    ) -> None:
        try:
            source_values = tuple(
                sources
            )
        except TypeError as exc:
            raise TypeError(
                "sources must be an iterable of "
                "AuthoritativeSource values."
            ) from exc

        source_map: dict[
            str,
            AuthoritativeSource,
        ] = {}

        for source in source_values:
            if not isinstance(
                source,
                AuthoritativeSource,
            ):
                raise TypeError(
                    "sources must contain only "
                    "AuthoritativeSource values."
                )

            if (
                source.document_id
                in source_map
            ):
                raise ValueError(
                    "Authoritative source document IDs "
                    "must be unique."
                )

            source_map[
                source.document_id
            ] = source

        self._sources = source_map

    @property
    def source_count(
        self,
    ) -> int:
        return len(
            self._sources
        )

    @property
    def document_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._sources
            )
        )

    def build(
        self,
        result: RetrievalResult,
        *,
        confidence: float | None = None,
        verification_state: (
            EvidenceVerificationState
        ) = EvidenceVerificationState.UNVERIFIED,
        temporal_state: (
            EvidenceTemporalState
        ) = EvidenceTemporalState.UNASSESSED,
        trace_id: str | None = None,
    ) -> EvidenceCard:
        """Build one evidence card from one retrieval result."""

        if not isinstance(
            result,
            RetrievalResult,
        ):
            raise TypeError(
                "result must be a RetrievalResult."
            )

        document_id = (
            result.chunk.document_id
        )

        source = self._sources.get(
            document_id
        )

        if source is None:
            raise KeyError(
                "No authoritative source registered "
                f"for document_id {document_id!r}."
            )

        return EvidenceCard(
            source=source,
            chunk=result.chunk,
            retrieval_rank=result.rank,
            retrieval_score=result.score,
            retrieval_method=(
                result.retrieval_method
            ),
            confidence=confidence,
            verification_state=(
                verification_state
            ),
            temporal_state=(
                temporal_state
            ),
            trace_id=trace_id,
        )

    def build_many(
        self,
        results: Iterable[
            RetrievalResult
        ],
        *,
        trace_id: str | None = None,
    ) -> tuple[EvidenceCard, ...]:
        """Build deterministically ordered evidence cards."""

        try:
            result_values = tuple(
                results
            )
        except TypeError as exc:
            raise TypeError(
                "results must be an iterable of "
                "RetrievalResult values."
            ) from exc

        for result in result_values:
            if not isinstance(
                result,
                RetrievalResult,
            ):
                raise TypeError(
                    "results must contain only "
                    "RetrievalResult values."
                )

        ordered_results = sorted(
            result_values,
            key=lambda result: (
                result.rank,
                result.chunk.document_id,
                result.chunk.chunk_index,
                result.chunk.chunk_id,
            ),
        )

        return tuple(
            self.build(
                result,
                trace_id=trace_id,
            )
            for result in ordered_results
        )


def build_evidence_card(
    result: RetrievalResult,
    sources: Iterable[
        AuthoritativeSource
    ],
    *,
    confidence: float | None = None,
    verification_state: (
        EvidenceVerificationState
    ) = EvidenceVerificationState.UNVERIFIED,
    temporal_state: (
        EvidenceTemporalState
    ) = EvidenceTemporalState.UNASSESSED,
    trace_id: str | None = None,
) -> EvidenceCard:
    """Convenience function for building one evidence card."""

    return EvidenceCardBuilder(
        sources
    ).build(
        result,
        confidence=confidence,
        verification_state=(
            verification_state
        ),
        temporal_state=temporal_state,
        trace_id=trace_id,
    )


def build_evidence_cards(
    results: Iterable[
        RetrievalResult
    ],
    sources: Iterable[
        AuthoritativeSource
    ],
    *,
    trace_id: str | None = None,
) -> tuple[EvidenceCard, ...]:
    """Convenience function for building multiple evidence cards."""

    return EvidenceCardBuilder(
        sources
    ).build_many(
        results,
        trace_id=trace_id,
    )
