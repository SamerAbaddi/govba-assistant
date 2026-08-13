"""Offline tests for GovBA-GAR deterministic chunking."""

from __future__ import annotations

import unittest

from govba.rag import (
    AuthoritativeSource,
    ChunkingConfig,
    DocumentFormat,
    DocumentType,
    EvidenceChunk,
    ExtractedBlock,
    ExtractedDocument,
    SourceLanguage,
    SourceStatus,
    chunk_extracted_document,
)
from govba.rag.chunking import (
    CHUNKING_VERSION,
    DEFAULT_OVERLAP_CHARACTERS,
    DEFAULT_TARGET_CHARACTERS,
)


def make_source():
    return AuthoritativeSource(
        document_id="DOC-CHUNK-001",
        title="Synthetic Chunking Procedure",
        issuing_authority=(
            "Synthetic GovBA Test Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
    )


def make_document(
    blocks,
):
    return ExtractedDocument(
        source=make_source(),
        document_format=DocumentFormat.PDF,
        source_file_name="synthetic.pdf",
        blocks=tuple(blocks),
    )


class TestDeterministicChunking(
    unittest.TestCase
):
    def test_chunking_version(self):
        self.assertEqual(
            CHUNKING_VERSION,
            "govba-chunking-v1",
        )

    def test_default_configuration(self):
        config = ChunkingConfig()

        self.assertEqual(
            config.target_characters,
            DEFAULT_TARGET_CHARACTERS,
        )
        self.assertEqual(
            config.overlap_characters,
            DEFAULT_OVERLAP_CHARACTERS,
        )

    def test_invalid_target_is_rejected(self):
        for value in (
            0,
            -1,
            True,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ChunkingConfig(
                        target_characters=value,
                        overlap_characters=0,
                    )

    def test_invalid_overlap_is_rejected(self):
        for value in (
            -1,
            True,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ChunkingConfig(
                        target_characters=100,
                        overlap_characters=value,
                    )

    def test_overlap_must_be_smaller_than_target(self):
        for overlap in (
            100,
            101,
        ):
            with self.subTest(
                overlap=overlap
            ):
                with self.assertRaises(
                    ValueError
                ):
                    ChunkingConfig(
                        target_characters=100,
                        overlap_characters=overlap,
                    )

    def test_short_block_produces_one_chunk(self):
        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text="Annual leave requires approval.",
            language=SourceLanguage.ENGLISH,
            page_number=4,
            section_title="Annual Leave",
            section_path=(
                "Human Resources",
                "Leave",
            ),
        )

        chunks = chunk_extracted_document(
            make_document((block,))
        )

        self.assertEqual(
            len(chunks),
            1,
        )
        self.assertIsInstance(
            chunks[0],
            EvidenceChunk,
        )
        self.assertEqual(
            chunks[0].chunk_index,
            0,
        )

    def test_page_metadata_is_preserved(self):
        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text="Page-aware evidence.",
            language=SourceLanguage.ENGLISH,
            page_number=7,
        )

        chunk = chunk_extracted_document(
            make_document((block,))
        )[0]

        self.assertEqual(
            chunk.page_start,
            7,
        )
        self.assertEqual(
            chunk.page_end,
            7,
        )

    def test_section_metadata_is_preserved(self):
        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text="Section-aware evidence.",
            language=SourceLanguage.ENGLISH,
            section_title="Eligibility",
            section_path=(
                "Services",
                "Eligibility",
            ),
        )

        chunk = chunk_extracted_document(
            make_document((block,))
        )[0]

        self.assertEqual(
            chunk.section_title,
            "Eligibility",
        )
        self.assertEqual(
            chunk.section_path,
            (
                "Services",
                "Eligibility",
            ),
        )

    def test_language_is_preserved(self):
        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text="يجب تقديم الطلب خلال خمسة أيام.",
            language=SourceLanguage.ARABIC,
            page_number=1,
        )

        chunk = chunk_extracted_document(
            make_document((block,))
        )[0]

        self.assertEqual(
            chunk.language,
            SourceLanguage.ARABIC,
        )
        self.assertIn(
            "خمسة",
            chunk.text,
        )

    def test_multiple_blocks_use_global_indexes(self):
        blocks = (
            ExtractedBlock(
                document_id="DOC-CHUNK-001",
                block_index=0,
                text="First block.",
                language=SourceLanguage.ENGLISH,
                page_number=1,
            ),
            ExtractedBlock(
                document_id="DOC-CHUNK-001",
                block_index=1,
                text="Second block.",
                language=SourceLanguage.ENGLISH,
                page_number=2,
            ),
        )

        chunks = chunk_extracted_document(
            make_document(blocks)
        )

        self.assertEqual(
            tuple(
                chunk.chunk_index
                for chunk in chunks
            ),
            (
                0,
                1,
            ),
        )

    def test_long_block_is_split(self):
        text = " ".join(
            f"word{index}"
            for index in range(80)
        )

        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text=text,
            language=SourceLanguage.ENGLISH,
            page_number=1,
        )

        chunks = chunk_extracted_document(
            make_document((block,)),
            ChunkingConfig(
                target_characters=100,
                overlap_characters=20,
            ),
        )

        self.assertGreater(
            len(chunks),
            1,
        )

        for chunk in chunks:
            self.assertLessEqual(
                len(chunk.text),
                100,
            )

    def test_overlap_preserves_shared_context(self):
        text = (
            "alpha beta gamma delta epsilon "
            "zeta eta theta iota kappa lambda "
            "mu nu xi omicron pi rho sigma tau "
            "upsilon phi chi psi omega"
        )

        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text=text,
            language=SourceLanguage.ENGLISH,
        )

        chunks = chunk_extracted_document(
            make_document((block,)),
            ChunkingConfig(
                target_characters=70,
                overlap_characters=20,
            ),
        )

        self.assertGreaterEqual(
            len(chunks),
            2,
        )

        first_words = set(
            chunks[0].text.split()
        )
        second_words = set(
            chunks[1].text.split()
        )

        self.assertTrue(
            first_words.intersection(
                second_words
            )
        )

    def test_text_is_normalized_before_chunking(self):
        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text=(
                "  Annual\t\tleave "
                "requires\u00a0approval.  "
            ),
            language=SourceLanguage.ENGLISH,
        )

        chunk = chunk_extracted_document(
            make_document((block,))
        )[0]

        self.assertEqual(
            chunk.text,
            "Annual leave requires approval.",
        )

    def test_chunks_do_not_cross_block_pages(self):
        blocks = (
            ExtractedBlock(
                document_id="DOC-CHUNK-001",
                block_index=0,
                text="Content on first page.",
                language=SourceLanguage.ENGLISH,
                page_number=1,
            ),
            ExtractedBlock(
                document_id="DOC-CHUNK-001",
                block_index=1,
                text="Content on second page.",
                language=SourceLanguage.ENGLISH,
                page_number=2,
            ),
        )

        chunks = chunk_extracted_document(
            make_document(blocks),
            ChunkingConfig(
                target_characters=500,
                overlap_characters=50,
            ),
        )

        self.assertEqual(
            len(chunks),
            2,
        )

        self.assertEqual(
            tuple(
                chunk.page_start
                for chunk in chunks
            ),
            (
                1,
                2,
            ),
        )

    def test_same_input_produces_same_chunk_ids(self):
        text = " ".join(
            f"policy{index}"
            for index in range(100)
        )

        block = ExtractedBlock(
            document_id="DOC-CHUNK-001",
            block_index=0,
            text=text,
            language=SourceLanguage.ENGLISH,
            page_number=3,
            section_title="Policy",
        )

        document = make_document(
            (block,)
        )

        config = ChunkingConfig(
            target_characters=120,
            overlap_characters=20,
        )

        first = chunk_extracted_document(
            document,
            config,
        )

        second = chunk_extracted_document(
            document,
            config,
        )

        self.assertEqual(
            tuple(
                chunk.chunk_id
                for chunk in first
            ),
            tuple(
                chunk.chunk_id
                for chunk in second
            ),
        )

        self.assertEqual(
            tuple(
                chunk.content_hash
                for chunk in first
            ),
            tuple(
                chunk.content_hash
                for chunk in second
            ),
        )

    def test_invalid_document_type_is_rejected(self):
        with self.assertRaises(TypeError):
            chunk_extracted_document(
                object()
            )


if __name__ == "__main__":
    unittest.main()
