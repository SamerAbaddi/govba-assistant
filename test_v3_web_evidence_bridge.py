"""Offline tests for GovBA-GAR official-web evidence bridge."""

from __future__ import annotations

import unittest
from datetime import (
    date,
    datetime,
    timezone,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
)
from govba.rag.chunking import (
    ChunkingConfig,
    split_ingested_text,
)
from govba.rag.models import (
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.web.contract import (
    OfficialWebResult,
)
from govba.web.evidence_bridge import (
    WEB_EVIDENCE_BRIDGE_VERSION,
    WebEvidenceMetadata,
    bridge_official_web_result,
    build_web_document_id,
)


def web_result(
    *,
    url="https://example.gov.jo/policy",
    text="Synthetic official policy content.",
    title="Synthetic Government Policy",
    rank=1,
    retrieval_method="controlled-web-test-v1",
):
    return OfficialWebResult(
        title=title,
        url=url,
        text=text,
        rank=rank,
        score=0.9,
        retrieval_method=(
            retrieval_method
        ),
        retrieved_at=datetime(
            2026,
            8,
            13,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )


def metadata(
    **overrides,
):
    values = {
        "issuing_authority": (
            "Synthetic Government Authority"
        ),
        "language": (
            SourceLanguage.ENGLISH
        ),
    }

    values.update(
        overrides
    )

    return WebEvidenceMetadata(
        **values
    )


def policy():
    return SourceAllowlistPolicy(
        allowed_domains=(
            "example.gov.jo",
        )
    )


class TestPublicTextSplitter(
    unittest.TestCase
):
    def test_text_splitter_normalizes_text(self):
        pieces = split_ingested_text(
            "Alpha    Beta\r\nGamma"
        )

        self.assertEqual(
            pieces,
            (
                "Alpha Beta\nGamma",
            ),
        )

    def test_text_splitter_rejects_blank_text(self):
        with self.assertRaises(
            ValueError
        ):
            split_ingested_text(
                "   "
            )

    def test_custom_chunking_config_is_used(self):
        pieces = split_ingested_text(
            "A" * 120,
            ChunkingConfig(
                target_characters=50,
                overlap_characters=10,
            ),
        )

        self.assertGreater(
            len(pieces),
            1,
        )


class TestWebEvidenceBridge(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            WEB_EVIDENCE_BRIDGE_VERSION,
            "govba-web-evidence-bridge-v1",
        )

    def test_document_id_is_deterministic(self):
        first = build_web_document_id(
            web_result()
        )

        second = build_web_document_id(
            web_result()
        )

        self.assertEqual(
            first,
            second,
        )

    def test_document_id_is_sha256_based(self):
        document_id = (
            build_web_document_id(
                web_result()
            )
        )

        self.assertTrue(
            document_id.startswith(
                "WEB-"
            )
        )

        digest = document_id[
            4:
        ]

        self.assertEqual(
            len(digest),
            64,
        )

        int(
            digest,
            16,
        )

    def test_content_change_changes_document_id(self):
        first = build_web_document_id(
            web_result(
                text="Version one"
            )
        )

        second = build_web_document_id(
            web_result(
                text="Version two"
            )
        )

        self.assertNotEqual(
            first,
            second,
        )

    def test_rank_does_not_change_document_identity(self):
        first = build_web_document_id(
            web_result(
                rank=1
            )
        )

        second = build_web_document_id(
            web_result(
                rank=5
            )
        )

        self.assertEqual(
            first,
            second,
        )

    def test_retrieval_method_does_not_change_document_identity(self):
        first = build_web_document_id(
            web_result(
                retrieval_method="backend-a"
            )
        )

        second = build_web_document_id(
            web_result(
                retrieval_method="backend-b"
            )
        )

        self.assertEqual(
            first,
            second,
        )

    def test_bridge_creates_authoritative_source(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.title,
            "Synthetic Government Policy",
        )

        self.assertEqual(
            result.source.issuing_authority,
            "Synthetic Government Authority",
        )

    def test_source_url_is_preserved(self):
        url = (
            "https://example.gov.jo/"
            "policy"
        )

        result = bridge_official_web_result(
            web_result(
                url=url
            ),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.official_source_url,
            url,
        )

    def test_source_content_hash_matches_web_result(self):
        raw = web_result()

        result = bridge_official_web_result(
            raw,
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.content_hash,
            raw.content_hash,
        )

    def test_retrieval_time_becomes_ingestion_time(self):
        raw = web_result()

        result = bridge_official_web_result(
            raw,
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.ingested_at,
            raw.retrieved_at,
        )

    def test_default_document_type_is_other(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.document_type,
            DocumentType.OTHER,
        )

    def test_default_status_is_unknown(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.status,
            SourceStatus.UNKNOWN,
        )

    def test_explicit_document_id_is_preserved(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(
                document_id=(
                    "POLICY-2026-001"
                )
            ),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.document_id,
            "POLICY-2026-001",
        )

    def test_temporal_metadata_is_preserved(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(
                document_type=(
                    DocumentType.POLICY
                ),
                publication_date=date(
                    2026,
                    7,
                    1,
                ),
                effective_from=date(
                    2026,
                    8,
                    1,
                ),
                effective_until=date(
                    2027,
                    7,
                    31,
                ),
                version="2.0",
                status=(
                    SourceStatus.CURRENT
                ),
            ),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.source.effective_from,
            date(
                2026,
                8,
                1,
            ),
        )

        self.assertEqual(
            result.source.status,
            SourceStatus.CURRENT,
        )

        self.assertEqual(
            result.source.version,
            "2.0",
        )

    def test_bridge_creates_evidence_chunks(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.chunk_count,
            1,
        )

        self.assertEqual(
            result.chunks[
                0
            ].document_id,
            result.source.document_id,
        )

    def test_long_web_text_creates_multiple_chunks(self):
        result = bridge_official_web_result(
            web_result(
                text=(
                    "Government policy text. "
                    * 100
                )
            ),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
            chunking_config=(
                ChunkingConfig(
                    target_characters=250,
                    overlap_characters=40,
                )
            ),
        )

        self.assertGreater(
            result.chunk_count,
            1,
        )

    def test_chunk_indexes_are_contiguous(self):
        result = bridge_official_web_result(
            web_result(
                text=(
                    "Policy sentence. "
                    * 100
                )
            ),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
            chunking_config=(
                ChunkingConfig(
                    target_characters=200,
                    overlap_characters=30,
                )
            ),
        )

        self.assertEqual(
            tuple(
                chunk.chunk_index
                for chunk
                in result.chunks
            ),
            tuple(
                range(
                    result.chunk_count
                )
            ),
        )

    def test_chunk_language_is_preserved(self):
        result = bridge_official_web_result(
            web_result(),
            metadata=metadata(
                language=(
                    SourceLanguage.ARABIC
                )
            ),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.chunks[
                0
            ].language,
            SourceLanguage.ARABIC,
        )

    def test_web_text_uses_ingestion_normalization(self):
        result = bridge_official_web_result(
            web_result(
                text=(
                    "Alpha    Beta\r\nGamma"
                )
            ),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.chunks[
                0
            ].text,
            "Alpha Beta\nGamma",
        )

    def test_untrusted_web_result_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            bridge_official_web_result(
                web_result(
                    url=(
                        "https://evil.example/"
                        "policy"
                    )
                ),
                metadata=metadata(),
                policy=policy(),
                trace_id="TRACE-001",
            )

    def test_invalid_metadata_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            WebEvidenceMetadata(
                issuing_authority=(
                    "Authority"
                ),
                language="en",
            )

    def test_blank_authority_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            WebEvidenceMetadata(
                issuing_authority=" ",
                language=(
                    SourceLanguage.ENGLISH
                ),
            )

    def test_invalid_effective_window_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            metadata(
                effective_from=date(
                    2026,
                    8,
                    2,
                ),
                effective_until=date(
                    2026,
                    8,
                    1,
                ),
            )

    def test_bridge_is_deterministic(self):
        raw = web_result()
        meta = metadata()

        first = bridge_official_web_result(
            raw,
            metadata=meta,
            policy=policy(),
            trace_id="TRACE-001",
        )

        second = bridge_official_web_result(
            raw,
            metadata=meta,
            policy=policy(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            first.bridge_id,
            second.bridge_id,
        )

        self.assertEqual(
            tuple(
                chunk.chunk_id
                for chunk
                in first.chunks
            ),
            tuple(
                chunk.chunk_id
                for chunk
                in second.chunks
            ),
        )

    def test_serialization_excludes_evidence_text(self):
        raw_text = (
            "Unique synthetic policy evidence."
        )

        result = bridge_official_web_result(
            web_result(
                text=raw_text
            ),
            metadata=metadata(),
            policy=policy(),
            trace_id="TRACE-001",
        )

        data = result.to_dict()

        self.assertNotIn(
            raw_text,
            repr(data),
        )

        self.assertEqual(
            data[
                "chunk_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
