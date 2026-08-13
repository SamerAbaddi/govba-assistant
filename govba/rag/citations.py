"""Deterministic evidence citation and provenance formatting for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from govba.rag.evidence_card import EvidenceCard


CITATION_FORMATTER_VERSION = (
    "govba-citation-formatter-v1"
)


def _page_reference(
    card: EvidenceCard,
) -> str | None:
    start = card.page_start
    end = card.page_end

    if start is None:
        return None

    if (
        end is None
        or end == start
    ):
        return f"p. {start}"

    return f"pp. {start}-{end}"


def _section_reference(
    card: EvidenceCard,
) -> str | None:
    if card.section_path:
        return " > ".join(
            card.section_path
        )

    return card.section_title


def _effective_reference(
    card: EvidenceCard,
) -> str | None:
    source = card.source

    start = source.effective_from
    end = source.effective_until

    if (
        start is None
        and end is None
    ):
        return None

    if (
        start is not None
        and end is not None
    ):
        return (
            f"{start.isoformat()} "
            f"to {end.isoformat()}"
        )

    if start is not None:
        return (
            f"from {start.isoformat()}"
        )

    return (
        f"until {end.isoformat()}"
    )


def _publication_reference(
    value: date | None,
) -> str | None:
    if value is None:
        return None

    return value.isoformat()


@dataclass(frozen=True)
class EvidenceCitation:
    """Formatted citation linked to one EvidenceCard."""

    citation_id: str
    marker: str

    card_id: str
    document_id: str
    chunk_id: str

    short_citation: str
    provenance: str

    official_source_url: str | None

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "citation_id": (
                self.citation_id
            ),
            "marker": (
                self.marker
            ),
            "card_id": (
                self.card_id
            ),
            "document_id": (
                self.document_id
            ),
            "chunk_id": (
                self.chunk_id
            ),
            "short_citation": (
                self.short_citation
            ),
            "provenance": (
                self.provenance
            ),
            "official_source_url": (
                self.official_source_url
            ),
        }


def build_evidence_citation(
    card: EvidenceCard,
    ordinal: int,
) -> EvidenceCitation:
    """Build one deterministic citation from an evidence card."""

    if not isinstance(
        card,
        EvidenceCard,
    ):
        raise TypeError(
            "card must be an EvidenceCard."
        )

    if (
        isinstance(
            ordinal,
            bool,
        )
        or not isinstance(
            ordinal,
            int,
        )
        or ordinal < 1
    ):
        raise ValueError(
            "ordinal must be a positive integer."
        )

    citation_id = (
        f"E{ordinal}"
    )

    marker = (
        f"[{citation_id}]"
    )

    page = _page_reference(
        card
    )

    section = _section_reference(
        card
    )

    short_parts = [
        card.issuing_authority,
        card.title,
    ]

    if section:
        short_parts.append(
            section
        )

    if page:
        short_parts.append(
            page
        )

    short_citation = ", ".join(
        short_parts
    )

    source = card.source

    provenance_parts = [
        f"Authority: {source.issuing_authority}",
        f"Document: {source.title}",
        f"Document ID: {source.document_id}",
        f"Chunk ID: {card.chunk_id}",
        (
            "Status: "
            f"{source.status.value}"
        ),
    ]

    if source.version:
        provenance_parts.append(
            f"Version: {source.version}"
        )

    publication = (
        _publication_reference(
            source.publication_date
        )
    )

    if publication:
        provenance_parts.append(
            f"Published: {publication}"
        )

    effective = (
        _effective_reference(
            card
        )
    )

    if effective:
        provenance_parts.append(
            f"Effective: {effective}"
        )

    if section:
        provenance_parts.append(
            f"Section: {section}"
        )

    if page:
        provenance_parts.append(
            f"Location: {page}"
        )

    if card.retrieval_method:
        provenance_parts.append(
            "Retrieval: "
            f"{card.retrieval_method}"
        )

    if card.retrieval_rank is not None:
        provenance_parts.append(
            "Rank: "
            f"{card.retrieval_rank}"
        )

    if card.trace_id:
        provenance_parts.append(
            f"Trace ID: {card.trace_id}"
        )

    provenance = " | ".join(
        provenance_parts
    )

    return EvidenceCitation(
        citation_id=citation_id,
        marker=marker,
        card_id=card.card_id,
        document_id=card.document_id,
        chunk_id=card.chunk_id,
        short_citation=short_citation,
        provenance=provenance,
        official_source_url=(
            card.official_source_url
        ),
    )


def build_evidence_citations(
    cards: Iterable[
        EvidenceCard
    ],
) -> tuple[EvidenceCitation, ...]:
    """Build ordered citations for evidence cards."""

    try:
        card_values = tuple(
            cards
        )
    except TypeError as exc:
        raise TypeError(
            "cards must be an iterable "
            "of EvidenceCard values."
        ) from exc

    seen_card_ids = set()

    citations = []

    for ordinal, card in enumerate(
        card_values,
        start=1,
    ):
        if not isinstance(
            card,
            EvidenceCard,
        ):
            raise TypeError(
                "cards must contain only "
                "EvidenceCard values."
            )

        if card.card_id in seen_card_ids:
            raise ValueError(
                "Duplicate evidence cards "
                "cannot receive separate citations."
            )

        seen_card_ids.add(
            card.card_id
        )

        citations.append(
            build_evidence_citation(
                card,
                ordinal,
            )
        )

    return tuple(
        citations
    )


def citation_markers(
    citations: Iterable[
        EvidenceCitation
    ],
) -> str:
    """Return compact citation markers such as '[E1] [E2]'."""

    try:
        values = tuple(
            citations
        )
    except TypeError as exc:
        raise TypeError(
            "citations must be iterable."
        ) from exc

    for citation in values:
        if not isinstance(
            citation,
            EvidenceCitation,
        ):
            raise TypeError(
                "citations must contain only "
                "EvidenceCitation values."
            )

    return " ".join(
        citation.marker
        for citation in values
    )
