"""Offline tests for GovBA-GAR corpus validation."""

from __future__ import annotations

import hashlib
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
    CorpusEntry,
    CorpusManifest,
)
from govba.rag.corpus_validation import (
    CORPUS_VALIDATION_VERSION,
    CorpusEntryValidation,
    CorpusValidationReport,
    compute_file_sha256,
    validate_corpus,
)


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def make_source(
    *,
    document_id="DOC-VALIDATE-001",
    content_hash=None,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Government Policy",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/policy.pdf"
        ),
        content_hash=content_hash,
    )


def make_entry(
    *,
    document_id="DOC-VALIDATE-001",
    relative_path="policies/policy.pdf",
    approved=True,
    content_hash=None,
):
    return CorpusEntry(
        source=make_source(
            document_id=document_id,
            content_hash=content_hash,
        ),
        document_format=DocumentFormat.PDF,
        relative_path=relative_path,
        approved_for_ingestion=approved,
    )


def make_manifest(
    entry,
):
    return CorpusManifest(
        manifest_id="VALIDATION-TEST",
        name="Validation Test Corpus",
        entries=(entry,),
    )


class TestCorpusValidation(unittest.TestCase):
    def test_version_is_stable(self):
        self.assertEqual(
            CORPUS_VALIDATION_VERSION,
            "govba-corpus-validation-v1",
        )

    def test_file_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.pdf"

            path.write_bytes(
                b"synthetic document"
            )

            first = compute_file_sha256(path)
            second = compute_file_sha256(path)

            self.assertEqual(
                first,
                second,
            )

            self.assertEqual(
                len(first),
                64,
            )

    def test_missing_hash_file_is_rejected(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            compute_file_sha256(
                "/tmp/definitely-missing.pdf"
            )

    def test_valid_approved_entry_passes(self):
        data = b"synthetic policy"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            path = (
                root
                / "policies"
                / "policy.pdf"
            )

            path.parent.mkdir(
                parents=True
            )

            path.write_bytes(
                data
            )

            entry = make_entry(
                content_hash=(
                    sha256_bytes(data)
                )
            )

            report = validate_corpus(
                make_manifest(entry),
                root,
            )

            self.assertTrue(
                report.is_valid
            )

            self.assertEqual(
                report.invalid_count,
                0,
            )

            self.assertTrue(
                report.entries[0].hash_matches
            )

    def test_missing_approved_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_entry(
                content_hash=(
                    "a" * 64
                )
            )

            report = validate_corpus(
                make_manifest(entry),
                directory,
            )

            self.assertFalse(
                report.is_valid
            )

            self.assertFalse(
                report.entries[0].file_exists
            )

    def test_missing_content_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            path = (
                root
                / "policies"
                / "policy.pdf"
            )

            path.parent.mkdir(
                parents=True
            )

            path.write_bytes(
                b"policy"
            )

            report = validate_corpus(
                make_manifest(
                    make_entry(
                        content_hash=None
                    )
                ),
                root,
            )

            self.assertFalse(
                report.is_valid
            )

            self.assertIn(
                "Approved corpus entry has no content hash.",
                report.entries[0].errors,
            )

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            path = (
                root
                / "policies"
                / "policy.pdf"
            )

            path.parent.mkdir(
                parents=True
            )

            path.write_bytes(
                b"changed document"
            )

            report = validate_corpus(
                make_manifest(
                    make_entry(
                        content_hash=(
                            "a" * 64
                        )
                    )
                ),
                root,
            )

            self.assertFalse(
                report.is_valid
            )

            self.assertFalse(
                report.entries[0].hash_matches
            )

    def test_unapproved_entry_is_not_required_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_entry(
                approved=False
            )

            report = validate_corpus(
                make_manifest(entry),
                directory,
            )

            self.assertTrue(
                report.is_valid
            )

            self.assertFalse(
                report.entries[0].approved
            )

            self.assertIsNone(
                report.entries[0].hash_matches
            )

    def test_report_counts(self):
        entries = (
            CorpusEntryValidation(
                document_id="A",
                relative_path="a.pdf",
                approved=True,
                valid=True,
                file_exists=True,
                hash_required=True,
                hash_matches=True,
            ),
            CorpusEntryValidation(
                document_id="B",
                relative_path="b.pdf",
                approved=True,
                valid=False,
                file_exists=False,
                hash_required=True,
                hash_matches=None,
                errors=("missing",),
            ),
        )

        report = CorpusValidationReport(
            manifest_id="TEST",
            entries=entries,
        )

        self.assertEqual(
            report.entry_count,
            2,
        )
        self.assertEqual(
            report.approved_count,
            2,
        )
        self.assertEqual(
            report.valid_count,
            1,
        )
        self.assertEqual(
            report.invalid_count,
            1,
        )
        self.assertFalse(
            report.is_valid
        )

    def test_unknown_manifest_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                validate_corpus(
                    object(),
                    directory,
                )

    def test_missing_corpus_root_is_rejected(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            validate_corpus(
                CorpusManifest(
                    manifest_id="TEST",
                    name="Test",
                    entries=(),
                ),
                "/tmp/definitely-missing-corpus-root",
            )


if __name__ == "__main__":
    unittest.main()
