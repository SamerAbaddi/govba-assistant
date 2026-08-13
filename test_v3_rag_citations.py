"""Offline tests for GovBA-GAR citation formatting."""

from __future__ import annotations

import unittest
from datetime import date

from govba.rag.evidence import EvidenceChunk
from govba.rag.evidence_card import (
    EvidenceCard,
)
from govba.rag.citations import (
    CITATION_FORMATTER_VERSION,
    EvidenceCitation,
    build_evidence_citation,
    build_evidence_citations,
    citation_markers,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_source(
    document_id="DOC-CITE-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Government Leave Procedure",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=(
            DocumentType.PROCEDURE
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        jurisdiction="Jordan",
        publication_date=date(
            2026,
            1,
            1,
        ),
        effective_from=date(
            2026,
            2,
            1,
        ),
        version="2.0",
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/"
            "leave.pdf"
        ),
    )


def make_chunk(
    document_id="DOC-CITE-001",
    *,
    chunk_index=0,
    page_start=4,
    page_end=4,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=(
            "Annual leave requests require "
            "manager approval."
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        page_start=page_start,
        page_end=page_end,
        section_title="Annual Leave",
        section_path=(
            "Human Resources",
            "Annual Leave",
        ),
    )


def make_card(
    document_id="DOC-CITE-001",
    *,
    chunk_index=0,
    page_start=4,
    page_end=4,
):
    return EvidenceCard(
        source=make_source(
            document_id
        ),
        chunk=make_chunk(
            document_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
        ),
        retrieval_rank=1,
        retrieval_score=0.91,
        retrieval_method=(
            "hybrid-rrf-v1"
        ),
        trace_id="TRACE-CITE-001",
    )


class TestEvidenceCitations(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CITATION_FORMATTER_VERSION,
            "govba-citation-formatter-v1",
        )

    def test_single_citation_is_created(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIsInstance(
            citation,
            EvidenceCitation,
        )

        self.assertEqual(
            citation.citation_id,
            "E1",
        )

        self.assertEqual(
            citation.marker,
            "[E1]",
        )

    def test_card_identity_is_preserved(self):
        card = make_card()

        citation = (
            build_evidence_citation(
                card,
                1,
            )
        )

        self.assertEqual(
            citation.card_id,
            card.card_id,
        )

        self.assertEqual(
            citation.document_id,
            card.document_id,
        )

        self.assertEqual(
            citation.chunk_id,
            card.chunk_id,
        )

    def test_single_page_reference(self):
        citation = (
            build_evidence_citation(
                make_card(
                    page_start=4,
                    page_end=4,
                ),
                1,
            )
        )

        self.assertIn(
            "p. 4",
            citation.short_citation,
        )

    def test_page_range_reference(self):
        citation = (
            build_evidence_citation(
                make_card(
                    page_start=4,
                    page_end=6,
                ),
                1,
            )
        )

        self.assertIn(
            "pp. 4-6",
            citation.short_citation,
        )

    def test_section_path_is_formatted(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            (
                "Human Resources"
                " > Annual Leave"
            ),
            citation.short_citation,
        )

    def test_provenance_contains_authority(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            (
                "Authority: Synthetic "
                "Government Authority"
            ),
            citation.provenance,
        )

    def test_provenance_contains_status(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            "Status: current",
            citation.provenance,
        )

    def test_provenance_contains_dates(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            "Published: 2026-01-01",
            citation.provenance,
        )

        self.assertIn(
            "Effective: from 2026-02-01",
            citation.provenance,
        )

    def test_provenance_contains_retrieval_method(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            "Retrieval: hybrid-rrf-v1",
            citation.provenance,
        )

    def test_provenance_contains_trace_id(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        self.assertIn(
            "Trace ID: TRACE-CITE-001",
            citation.provenance,
        )

    def test_invalid_card_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            build_evidence_citation(
                object(),
                1,
            )

    def test_invalid_ordinal_is_rejected(self):
        for value in (
            0,
            -1,
            True,
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    build_evidence_citation(
                        make_card(),
                        value,
                    )

    def test_multiple_citations_use_stable_markers(self):
        cards = (
            make_card(
                "DOC-A",
                chunk_index=0,
            ),
            make_card(
                "DOC-B",
                chunk_index=0,
            ),
        )

        citations = (
            build_evidence_citations(
                cards
            )
        )

        self.assertEqual(
            tuple(
                citation.marker
                for citation in citations
            ),
            (
                "[E1]",
                "[E2]",
            ),
        )

    def test_input_order_is_preserved(self):
        cards = (
            make_card(
                "DOC-B"
            ),
            make_card(
                "DOC-A"
            ),
        )

        citations = (
            build_evidence_citations(
                cards
            )
        )

        self.assertEqual(
            tuple(
                citation.document_id
                for citation in citations
            ),
            (
                "DOC-B",
                "DOC-A",
            ),
        )

    def test_duplicate_cards_are_rejected(self):
        card = make_card()

        with self.assertRaises(
            ValueError
        ):
            build_evidence_citations(
                (
                    card,
                    card,
                )
            )

    def test_marker_string_is_compact(self):
        citations = (
            build_evidence_citations(
                (
                    make_card(
                        "DOC-A"
                    ),
                    make_card(
                        "DOC-B"
                    ),
                )
            )
        )

        self.assertEqual(
            citation_markers(
                citations
            ),
            "[E1] [E2]",
        )

    def test_to_dict_contains_linkage_fields(self):
        citation = (
            build_evidence_citation(
                make_card(),
                1,
            )
        )

        data = citation.to_dict()

        required = {
            "citation_id",
            "marker",
            "card_id",
            "document_id",
            "chunk_id",
            "short_citation",
            "provenance",
            "official_source_url",
        }

        self.assertTrue(
            required.issubset(
                data
            )
        )


if __name__ == "__main__":
    unittest.main()
