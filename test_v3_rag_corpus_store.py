"""Offline tests for GovBA-GAR corpus manifest persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import (
    date,
    datetime,
    timezone,
)
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
from govba.rag.corpus_store import (
    CORPUS_STORE_VERSION,
    CorpusManifestError,
    canonical_manifest_json,
    compute_manifest_hash,
    manifest_envelope,
    read_manifest,
    write_manifest,
)


def make_manifest():
    source = AuthoritativeSource(
        document_id="DOC-STORE-001",
        title="Synthetic Government Policy",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        jurisdiction="Jordan",
        publication_date=date(
            2026,
            1,
            5,
        ),
        effective_from=date(
            2026,
            2,
            1,
        ),
        version="2.0",
        status=SourceStatus.CURRENT,
        supersedes=(
            "DOC-STORE-000",
        ),
        official_source_url=(
            "https://example.gov.jo/"
            "policy.pdf"
        ),
        content_hash=(
            "a" * 64
        ),
        ingested_at=datetime(
            2026,
            8,
            13,
            8,
            0,
            tzinfo=timezone.utc,
        ),
    )

    entry = CorpusEntry(
        source=source,
        document_format=DocumentFormat.PDF,
        relative_path="policies/policy.pdf",
        approved_for_ingestion=True,
    )

    return CorpusManifest(
        manifest_id="JORDAN-GOV-TEST",
        name="Synthetic Government Corpus",
        entries=(entry,),
    )


class TestCorpusStore(unittest.TestCase):
    def test_store_version_is_stable(self):
        self.assertEqual(
            CORPUS_STORE_VERSION,
            "govba-corpus-store-v1",
        )

    def test_canonical_json_is_deterministic(self):
        manifest = make_manifest()

        self.assertEqual(
            canonical_manifest_json(
                manifest
            ),
            canonical_manifest_json(
                manifest
            ),
        )

    def test_hash_is_sha256_hex(self):
        digest = compute_manifest_hash(
            make_manifest()
        )

        self.assertEqual(
            len(digest),
            64,
        )

        int(
            digest,
            16,
        )

    def test_same_manifest_has_same_hash(self):
        self.assertEqual(
            compute_manifest_hash(
                make_manifest()
            ),
            compute_manifest_hash(
                make_manifest()
            ),
        )

    def test_envelope_contains_integrity_metadata(self):
        envelope = manifest_envelope(
            make_manifest()
        )

        self.assertEqual(
            envelope["store_version"],
            CORPUS_STORE_VERSION,
        )

        self.assertEqual(
            envelope[
                "integrity_algorithm"
            ],
            "sha256",
        )

    def test_write_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            original = make_manifest()

            write_manifest(
                original,
                path,
            )

            restored = read_manifest(
                path
            )

            self.assertEqual(
                restored,
                original,
            )

    def test_complete_source_metadata_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            original = make_manifest()

            write_manifest(
                original,
                path,
            )

            restored = read_manifest(
                path
            )

            self.assertEqual(
                restored.entries[0].source,
                original.entries[0].source,
            )

    def test_written_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            write_manifest(
                make_manifest(),
                path,
            )

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertIn(
                "integrity_hash",
                data,
            )

    def test_non_json_extension_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            write_manifest(
                make_manifest(),
                "/tmp/manifest.txt",
            )

    def test_missing_manifest_is_rejected(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            read_manifest(
                "/tmp/"
                "definitely-missing-manifest.json"
            )

    def test_invalid_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            path.write_text(
                "{not valid json",
                encoding="utf-8",
            )

            with self.assertRaises(
                CorpusManifestError
            ):
                read_manifest(
                    path
                )

    def test_tampered_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            write_manifest(
                make_manifest(),
                path,
            )

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            data["manifest"]["name"] = (
                "Tampered Corpus"
            )

            path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                CorpusManifestError
            ):
                read_manifest(
                    path
                )

    def test_tampered_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            write_manifest(
                make_manifest(),
                path,
            )

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            data["integrity_hash"] = (
                "0" * 64
            )

            path.write_text(
                json.dumps(
                    data
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                CorpusManifestError
            ):
                read_manifest(
                    path
                )

    def test_unknown_store_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "manifest.json"
            )

            write_manifest(
                make_manifest(),
                path,
            )

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            data["store_version"] = (
                "unknown"
            )

            path.write_text(
                json.dumps(
                    data
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                CorpusManifestError
            ):
                read_manifest(
                    path
                )


if __name__ == "__main__":
    unittest.main()
