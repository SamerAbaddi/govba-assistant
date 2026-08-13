"""Offline tests for GovBA-GAR evidence-card construction."""

from __future__ import annotations

import unittest

from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.evidence_card_builder import (
    EVIDENCE_CARD_BUILDER_VERSION,
    EvidenceCardBuilder,
    build_evidence_card,
    build_evidence_cards,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.retrieval import (
    RetrievalResult,
)


def make_source(
    document_id="DOC-BUILD-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title=f"Policy {document_id}",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}.pdf"
        ),
    )


def make_chunk(
    document_id="DOC-BUILD-001",
    *,
    chunk_index=0,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=(
            f"Evidence for {document_id} "
            f"chunk {chunk_index}."
        ),
        language=SourceLanguage.ENGLISH,
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
    )


def make_result(
    document_id="DOC-BUILD-001",
    *,
    chunk_index=0,
    rank=1,
    score=0.9,
    retrieval_method="hybrid-rrf-v1",
):
    return RetrievalResult(
        chunk=make_chunk(
            document_id,
            chunk_index=chunk_index,
        ),
        score=score,
        rank=rank,
        retrieval_method=(
            retrieval_method
        ),
        matched_terms=(
            "policy",
            "evidence",
        ),
    )


class TestEvidenceCardBuilder(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            EVIDENCE_CARD_BUILDER_VERSION,
            "govba-evidence-card-builder-v1",
        )

    def test_source_registry_is_created(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-B"
                ),
                make_source(
                    "DOC-A"
                ),
            )
        )

        self.assertEqual(
            builder.source_count,
            2,
        )

        self.assertEqual(
            builder.document_ids,
            (
                "DOC-A",
                "DOC-B",
            ),
        )

    def test_invalid_source_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            EvidenceCardBuilder(
                (
                    object(),
                )
            )

    def test_duplicate_source_ids_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            EvidenceCardBuilder(
                (
                    make_source(),
                    make_source(),
                )
            )

    def test_build_returns_evidence_card(self):
        builder = EvidenceCardBuilder(
            (
                make_source(),
            )
        )

        card = builder.build(
            make_result()
        )

        self.assertIsInstance(
            card,
            EvidenceCard,
        )

    def test_retrieval_metadata_is_mapped(self):
        card = EvidenceCardBuilder(
            (
                make_source(),
            )
        ).build(
            make_result(
                rank=2,
                score=0.81,
                retrieval_method=(
                    "semantic-v1"
                ),
            )
        )

        self.assertEqual(
            card.retrieval_rank,
            2,
        )

        self.assertEqual(
            card.retrieval_score,
            0.81,
        )

        self.assertEqual(
            card.retrieval_method,
            "semantic-v1",
        )

    def test_source_is_resolved_by_document_id(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-A"
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        card = builder.build(
            make_result(
                "DOC-B"
            )
        )

        self.assertEqual(
            card.document_id,
            "DOC-B",
        )

        self.assertEqual(
            card.source.title,
            "Policy DOC-B",
        )

    def test_unknown_source_is_rejected(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-A"
                ),
            )
        )

        with self.assertRaises(
            KeyError
        ):
            builder.build(
                make_result(
                    "DOC-B"
                )
            )

    def test_governance_metadata_is_passed(self):
        card = EvidenceCardBuilder(
            (
                make_source(),
            )
        ).build(
            make_result(),
            confidence=0.88,
            verification_state=(
                EvidenceVerificationState.VERIFIED
            ),
            temporal_state=(
                EvidenceTemporalState.CURRENT
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            card.confidence,
            0.88,
        )

        self.assertEqual(
            card.verification_state,
            EvidenceVerificationState.VERIFIED,
        )

        self.assertEqual(
            card.temporal_state,
            EvidenceTemporalState.CURRENT,
        )

        self.assertEqual(
            card.trace_id,
            "TRACE-001",
        )

    def test_build_many_orders_by_rank(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-A"
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        cards = builder.build_many(
            (
                make_result(
                    "DOC-B",
                    rank=2,
                ),
                make_result(
                    "DOC-A",
                    rank=1,
                ),
            )
        )

        self.assertEqual(
            tuple(
                card.retrieval_rank
                for card in cards
            ),
            (
                1,
                2,
            ),
        )

        self.assertEqual(
            tuple(
                card.document_id
                for card in cards
            ),
            (
                "DOC-A",
                "DOC-B",
            ),
        )

    def test_build_many_is_deterministic(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-A"
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        results = (
            make_result(
                "DOC-B",
                rank=2,
            ),
            make_result(
                "DOC-A",
                rank=1,
            ),
        )

        first = builder.build_many(
            results
        )

        second = builder.build_many(
            reversed(results)
        )

        self.assertEqual(
            tuple(
                card.card_id
                for card in first
            ),
            tuple(
                card.card_id
                for card in second
            ),
        )

    def test_trace_id_is_applied_to_all_cards(self):
        builder = EvidenceCardBuilder(
            (
                make_source(
                    "DOC-A"
                ),
                make_source(
                    "DOC-B"
                ),
            )
        )

        cards = builder.build_many(
            (
                make_result(
                    "DOC-A",
                    rank=1,
                ),
                make_result(
                    "DOC-B",
                    rank=2,
                ),
            ),
            trace_id="TRACE-ALL",
        )

        self.assertTrue(
            all(
                card.trace_id
                == "TRACE-ALL"
                for card in cards
            )
        )

    def test_convenience_single_builder(self):
        card = build_evidence_card(
            make_result(),
            (
                make_source(),
            ),
        )

        self.assertIsInstance(
            card,
            EvidenceCard,
        )

    def test_convenience_multiple_builder(self):
        cards = build_evidence_cards(
            (
                make_result(
                    rank=1
                ),
            ),
            (
                make_source(),
            ),
        )

        self.assertEqual(
            len(cards),
            1,
        )

        self.assertIsInstance(
            cards[0],
            EvidenceCard,
        )


if __name__ == "__main__":
    unittest.main()
