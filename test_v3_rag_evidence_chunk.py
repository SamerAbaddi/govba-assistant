"""Offline tests for GovBA-GAR evidence chunks and provenance."""

from __future__ import annotations

import json
import unittest

from govba.rag import (
    EvidenceChunk,
    SourceLanguage,
    compute_chunk_hash,
    compute_chunk_id,
)


class TestEvidenceChunk(unittest.TestCase):
    """Tests for evidence identity and provenance."""

    def test_valid_chunk_serializes(self):
        chunk = EvidenceChunk(
            document_id="DOC-001",
            chunk_index=0,
            text="Applicants must submit the required form.",
            language=SourceLanguage.ENGLISH,
            page_start=4,
            page_end=4,
            section_title="Application",
            section_path=(
                "Service Procedure",
                "Application",
            ),
        )

        data = chunk.to_dict()

        self.assertEqual(
            data["document_id"],
            "DOC-001",
        )
        self.assertTrue(data["chunk_id"])
        self.assertEqual(
            len(data["content_hash"]),
            64,
        )
        self.assertEqual(
            data["language"],
            "en",
        )
        self.assertEqual(
            data["page_start"],
            4,
        )

        json.dumps(data)

    def test_chunk_hash_is_deterministic(self):
        first = compute_chunk_hash(
            "same evidence"
        )
        second = compute_chunk_hash(
            "same evidence"
        )

        self.assertEqual(first, second)

    def test_changed_text_changes_hash(self):
        first = compute_chunk_hash(
            "evidence version one"
        )
        second = compute_chunk_hash(
            "evidence version two"
        )

        self.assertNotEqual(first, second)

    def test_chunk_id_is_deterministic(self):
        first = compute_chunk_id(
            document_id="DOC-002",
            chunk_index=2,
            text="Evidence text.",
            page_start=10,
            page_end=10,
            section_path=("Rules",),
        )

        second = compute_chunk_id(
            document_id="DOC-002",
            chunk_index=2,
            text="Evidence text.",
            page_start=10,
            page_end=10,
            section_path=("Rules",),
        )

        self.assertEqual(first, second)

    def test_changed_provenance_changes_chunk_id(self):
        first = compute_chunk_id(
            document_id="DOC-003",
            chunk_index=0,
            text="Same evidence.",
            page_start=1,
        )

        second = compute_chunk_id(
            document_id="DOC-003",
            chunk_index=0,
            text="Same evidence.",
            page_start=2,
        )

        self.assertNotEqual(first, second)

    def test_negative_chunk_index_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceChunk(
                document_id="DOC-004",
                chunk_index=-1,
                text="Evidence.",
                language=SourceLanguage.ENGLISH,
            )

    def test_invalid_page_range_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceChunk(
                document_id="DOC-005",
                chunk_index=0,
                text="Evidence.",
                language=SourceLanguage.ENGLISH,
                page_start=5,
                page_end=4,
            )

    def test_blank_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceChunk(
                document_id="DOC-006",
                chunk_index=0,
                text="   ",
                language=SourceLanguage.ENGLISH,
            )

    def test_incorrect_content_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceChunk(
                document_id="DOC-007",
                chunk_index=0,
                text="Evidence.",
                language=SourceLanguage.ENGLISH,
                content_hash="0" * 64,
            )

    def test_section_path_is_normalized(self):
        chunk = EvidenceChunk(
            document_id="DOC-008",
            chunk_index=0,
            text="Evidence.",
            language=SourceLanguage.ENGLISH,
            section_path=(
                " Chapter 1 ",
                "",
                " Eligibility ",
            ),
        )

        self.assertEqual(
            chunk.section_path,
            (
                "Chapter 1",
                "Eligibility",
            ),
        )


if __name__ == "__main__":
    unittest.main()
