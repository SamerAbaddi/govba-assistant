"""Offline tests for GovBA-GAR evidence cards."""

from __future__ import annotations

import unittest
from datetime import date

from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EVIDENCE_CARD_SCHEMA_VERSION,
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_source(
    document_id="DOC-EVIDENCE-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Government Procedure",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
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
        supersedes=(
            "DOC-EVIDENCE-000",
        ),
        official_source_url=(
            "https://example.gov.jo/"
            "procedure.pdf"
        ),
    )


def make_chunk(
    document_id="DOC-EVIDENCE-001",
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=0,
        text=(
            "Annual leave requests require "
            "manager approval."
        ),
        language=SourceLanguage.ENGLISH,
        page_start=4,
        page_end=4,
        section_title="Annual Leave",
        section_path=(
            "Human Resources",
            "Annual Leave",
        ),
    )


def make_card(
    **kwargs,
):
    values = {
        "source": make_source(),
        "chunk": make_chunk(),
    }

    values.update(
        kwargs
    )

    return EvidenceCard(
        **values
    )


class TestEvidenceCard(unittest.TestCase):
    def test_schema_version_is_stable(self):
        self.assertEqual(
            EVIDENCE_CARD_SCHEMA_VERSION,
            "govba-evidence-card-v1",
        )

    def test_basic_card_is_created(self):
        card = make_card()

        self.assertEqual(
            card.document_id,
            "DOC-EVIDENCE-001",
        )

        self.assertEqual(
            card.page_start,
            4,
        )

        self.assertEqual(
            card.section_title,
            "Annual Leave",
        )

    def test_card_id_is_sha256_hex(self):
        card = make_card()

        self.assertEqual(
            len(card.card_id),
            64,
        )

        int(
            card.card_id,
            16,
        )

    def test_card_id_is_deterministic(self):
        first = make_card()
        second = make_card()

        self.assertEqual(
            first.card_id,
            second.card_id,
        )

    def test_source_chunk_mismatch_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            EvidenceCard(
                source=make_source(
                    "DOC-A"
                ),
                chunk=make_chunk(
                    "DOC-B"
                ),
            )

    def test_invalid_source_type_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            EvidenceCard(
                source=object(),
                chunk=make_chunk(),
            )

    def test_invalid_chunk_type_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            EvidenceCard(
                source=make_source(),
                chunk=object(),
            )

    def test_retrieval_rank_validation(self):
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
                    make_card(
                        retrieval_rank=value
                    )

    def test_valid_retrieval_metadata(self):
        card = make_card(
            retrieval_rank=1,
            retrieval_score=0.91,
            retrieval_method="hybrid-rrf-v1",
        )

        self.assertEqual(
            card.retrieval_rank,
            1,
        )

        self.assertEqual(
            card.retrieval_score,
            0.91,
        )

        self.assertEqual(
            card.retrieval_method,
            "hybrid-rrf-v1",
        )

    def test_non_finite_score_is_rejected(self):
        for value in (
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    make_card(
                        retrieval_score=value
                    )

    def test_confidence_range_is_enforced(self):
        for value in (
            -0.01,
            1.01,
            float("nan"),
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    make_card(
                        confidence=value
                    )

    def test_valid_confidence_is_preserved(self):
        card = make_card(
            confidence=0.85
        )

        self.assertEqual(
            card.confidence,
            0.85,
        )

    def test_optional_text_is_normalized(self):
        card = make_card(
            retrieval_method=(
                "  semantic-v1  "
            ),
            trace_id="  TRACE-001  ",
        )

        self.assertEqual(
            card.retrieval_method,
            "semantic-v1",
        )

        self.assertEqual(
            card.trace_id,
            "TRACE-001",
        )

    def test_verification_and_temporal_states(self):
        card = make_card(
            verification_state=(
                EvidenceVerificationState.VERIFIED
            ),
            temporal_state=(
                EvidenceTemporalState.CURRENT
            ),
        )

        self.assertEqual(
            card.verification_state,
            EvidenceVerificationState.VERIFIED,
        )

        self.assertEqual(
            card.temporal_state,
            EvidenceTemporalState.CURRENT,
        )

    def test_page_and_section_provenance_is_preserved(self):
        card = make_card()

        self.assertEqual(
            card.page_start,
            4,
        )

        self.assertEqual(
            card.page_end,
            4,
        )

        self.assertEqual(
            card.section_path,
            (
                "Human Resources",
                "Annual Leave",
            ),
        )

    def test_source_temporal_metadata_serializes(self):
        data = make_card().to_dict()

        self.assertEqual(
            data["publication_date"],
            "2026-01-01",
        )

        self.assertEqual(
            data["effective_from"],
            "2026-02-01",
        )

        self.assertEqual(
            data["source_status"],
            SourceStatus.CURRENT.value,
        )

        self.assertEqual(
            data["supersedes"],
            [
                "DOC-EVIDENCE-000",
            ],
        )

    def test_to_dict_contains_governance_fields(self):
        card = make_card(
            retrieval_rank=1,
            retrieval_score=0.75,
            retrieval_method=(
                "hybrid-rrf-v1"
            ),
            confidence=0.8,
            trace_id="TRACE-001",
        )

        data = card.to_dict()

        required = {
            "card_id",
            "trace_id",
            "document_id",
            "chunk_id",
            "title",
            "issuing_authority",
            "official_source_url",
            "page_start",
            "page_end",
            "section_title",
            "section_path",
            "evidence_text",
            "source_status",
            "retrieval_rank",
            "retrieval_score",
            "retrieval_method",
            "confidence",
            "verification_state",
            "temporal_state",
        }

        self.assertTrue(
            required.issubset(
                data
            )
        )


if __name__ == "__main__":
    unittest.main()
