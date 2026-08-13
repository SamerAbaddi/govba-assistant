"""Offline tests for the GovBA-GAR authoritative source model."""

from __future__ import annotations

import json
import unittest
from datetime import date

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
    compute_content_hash,
)


class TestAuthoritativeSource(unittest.TestCase):
    """Tests for authoritative-source metadata and invariants."""

    def test_valid_source_serializes(self):
        content_hash = compute_content_hash(
            "Official circular content."
        )

        source = AuthoritativeSource(
            document_id="MODEE-CIRC-2026-001",
            title="Digital Services Circular",
            issuing_authority=(
                "Ministry of Digital Economy and Entrepreneurship"
            ),
            document_type=DocumentType.CIRCULAR,
            language=SourceLanguage.BILINGUAL,
            publication_date=date(2026, 1, 10),
            effective_from=date(2026, 2, 1),
            version="1.0",
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://example.gov.jo/circular-001"
            ),
            content_hash=content_hash,
        )

        data = source.to_dict()

        self.assertEqual(
            data["document_id"],
            "MODEE-CIRC-2026-001",
        )
        self.assertEqual(
            data["document_type"],
            "circular",
        )
        self.assertEqual(
            data["language"],
            "ar-en",
        )
        self.assertEqual(
            data["status"],
            "current",
        )
        self.assertEqual(
            data["effective_from"],
            "2026-02-01",
        )

        json.dumps(data)

    def test_content_hash_is_deterministic(self):
        first = compute_content_hash(
            "same government document"
        )
        second = compute_content_hash(
            "same government document"
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_content_change_changes_hash(self):
        first = compute_content_hash(
            "version one"
        )
        second = compute_content_hash(
            "version two"
        )

        self.assertNotEqual(first, second)

    def test_required_metadata_cannot_be_blank(self):
        with self.assertRaises(ValueError):
            AuthoritativeSource(
                document_id="",
                title="Circular",
                issuing_authority="Authority",
                document_type=DocumentType.CIRCULAR,
                language=SourceLanguage.ARABIC,
            )

    def test_invalid_effective_date_range_is_rejected(self):
        with self.assertRaises(ValueError):
            AuthoritativeSource(
                document_id="TEST-001",
                title="Test policy",
                issuing_authority="Authority",
                document_type=DocumentType.POLICY,
                language=SourceLanguage.ENGLISH,
                effective_from=date(2026, 5, 1),
                effective_until=date(2026, 4, 1),
            )

    def test_invalid_source_url_is_rejected(self):
        with self.assertRaises(ValueError):
            AuthoritativeSource(
                document_id="TEST-002",
                title="Test regulation",
                issuing_authority="Authority",
                document_type=DocumentType.REGULATION,
                language=SourceLanguage.ENGLISH,
                official_source_url="not-a-valid-url",
            )

    def test_self_supersession_is_rejected(self):
        with self.assertRaises(ValueError):
            AuthoritativeSource(
                document_id="TEST-003",
                title="Test procedure",
                issuing_authority="Authority",
                document_type=DocumentType.PROCEDURE,
                language=SourceLanguage.ENGLISH,
                supersedes=("TEST-003",),
            )

    def test_invalid_content_hash_is_rejected(self):
        with self.assertRaises(ValueError):
            AuthoritativeSource(
                document_id="TEST-004",
                title="Test manual",
                issuing_authority="Authority",
                document_type=DocumentType.MANUAL,
                language=SourceLanguage.ENGLISH,
                content_hash="not-a-sha256-hash",
            )


if __name__ == "__main__":
    unittest.main()
