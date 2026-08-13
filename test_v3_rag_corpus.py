"""Offline tests for GovBA-GAR authoritative corpus registration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from govba.rag import (
    AuthoritativeSource,
    DocumentFormat,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.corpus import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusEntry,
    CorpusManifest,
    build_ingestion_request,
)


def make_source(
    document_id="DOC-CORPUS-001",
    *,
    official_url=True,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Government Procedure",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/"
            f"documents/{document_id}.pdf"
            if official_url
            else None
        ),
    )


def make_entry(
    document_id="DOC-CORPUS-001",
    path="procedures/service.pdf",
    approved=True,
):
    return CorpusEntry(
        source=make_source(
            document_id
        ),
        document_format=DocumentFormat.PDF,
        relative_path=path,
        approved_for_ingestion=approved,
    )


class TestCorpusRegistration(unittest.TestCase):
    def test_schema_version_is_stable(self):
        self.assertEqual(
            CORPUS_MANIFEST_SCHEMA_VERSION,
            "govba-corpus-manifest-v1",
        )

    def test_entry_normalizes_path(self):
        entry = make_entry()

        self.assertEqual(
            entry.relative_path,
            "procedures/service.pdf",
        )

    def test_absolute_path_is_rejected(self):
        with self.assertRaises(ValueError):
            make_entry(
                path="/tmp/service.pdf"
            )

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            make_entry(
                path="../service.pdf"
            )

    def test_backslash_path_is_rejected(self):
        with self.assertRaises(ValueError):
            make_entry(
                path="procedures\\service.pdf"
            )

    def test_extension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            CorpusEntry(
                source=make_source(),
                document_format=DocumentFormat.PDF,
                relative_path=(
                    "procedures/service.docx"
                ),
                approved_for_ingestion=True,
            )

    def test_approved_entry_requires_official_url(self):
        with self.assertRaises(ValueError):
            CorpusEntry(
                source=make_source(
                    official_url=False
                ),
                document_format=DocumentFormat.PDF,
                relative_path=(
                    "procedures/service.pdf"
                ),
                approved_for_ingestion=True,
            )

    def test_unapproved_entry_can_be_registered(self):
        entry = CorpusEntry(
            source=make_source(
                official_url=False
            ),
            document_format=DocumentFormat.PDF,
            relative_path="drafts/service.pdf",
            approved_for_ingestion=False,
        )

        self.assertFalse(
            entry.approved_for_ingestion
        )

    def test_manifest_counts_entries(self):
        manifest = CorpusManifest(
            manifest_id="JORDAN-GOV-TEST",
            name="Synthetic Government Corpus",
            entries=(
                make_entry(
                    "DOC-001",
                    "procedures/one.pdf",
                ),
                make_entry(
                    "DOC-002",
                    "procedures/two.pdf",
                    approved=False,
                ),
            ),
        )

        self.assertEqual(
            manifest.entry_count,
            2,
        )
        self.assertEqual(
            manifest.approved_count,
            1,
        )

    def test_duplicate_document_ids_are_rejected(self):
        first = make_entry(
            "DOC-001",
            "documents/one.pdf",
        )

        second = make_entry(
            "DOC-001",
            "documents/two.pdf",
        )

        with self.assertRaises(ValueError):
            CorpusManifest(
                manifest_id="TEST",
                name="Test Corpus",
                entries=(first, second),
            )

    def test_duplicate_paths_are_rejected_case_insensitively(self):
        first = make_entry(
            "DOC-001",
            "documents/Policy.pdf",
        )

        second = make_entry(
            "DOC-002",
            "documents/policy.pdf",
        )

        with self.assertRaises(ValueError):
            CorpusManifest(
                manifest_id="TEST",
                name="Test Corpus",
                entries=(first, second),
            )

    def test_get_returns_registered_entry(self):
        entry = make_entry()

        manifest = CorpusManifest(
            manifest_id="TEST",
            name="Test Corpus",
            entries=(entry,),
        )

        self.assertIs(
            manifest.get(
                "DOC-CORPUS-001"
            ),
            entry,
        )

    def test_unknown_document_id_raises_key_error(self):
        manifest = CorpusManifest(
            manifest_id="TEST",
            name="Test Corpus",
            entries=(),
        )

        with self.assertRaises(KeyError):
            manifest.get(
                "UNKNOWN"
            )

    def test_manifest_serializes_to_json(self):
        manifest = CorpusManifest(
            manifest_id="TEST",
            name="Test Corpus",
            entries=(make_entry(),),
        )

        data = json.loads(
            manifest.to_json()
        )

        self.assertEqual(
            data["schema_version"],
            CORPUS_MANIFEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            data["entry_count"],
            1,
        )
        self.assertEqual(
            data["approved_count"],
            1,
        )

    def test_unapproved_entry_cannot_build_request(self):
        entry = make_entry(
            approved=False
        )

        with self.assertRaises(
            PermissionError
        ):
            build_ingestion_request(
                entry,
                "/tmp/corpus",
            )

    def test_approved_entry_builds_safe_request(self):
        entry = make_entry(
            path="procedures/service.pdf"
        )

        with tempfile.TemporaryDirectory() as directory:
            request = build_ingestion_request(
                entry,
                directory,
            )

            expected = (
                Path(directory)
                / "procedures"
                / "service.pdf"
            ).resolve()

            self.assertEqual(
                request.file_path,
                expected,
            )
            self.assertEqual(
                request.source,
                entry.source,
            )
            self.assertEqual(
                request.document_format,
                DocumentFormat.PDF,
            )


if __name__ == "__main__":
    unittest.main()
