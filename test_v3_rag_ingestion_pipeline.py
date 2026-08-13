"""Offline tests for the GovBA-GAR ingestion pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from govba.rag import (
    AuthoritativeSource,
    ChunkingConfig,
    DocumentFormat,
    DocumentType,
    ExtractedBlock,
    ExtractedDocument,
    IngestionRequest,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.pipeline import (
    INGESTION_PIPELINE_VERSION,
    IngestionPipeline,
    IngestionPipelineError,
    IngestionPipelineResult,
)


def make_source(
    document_id="DOC-PIPE-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Pipeline Procedure",
        issuing_authority=(
            "Synthetic GovBA Test Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
    )


def make_request(
    document_format=DocumentFormat.PDF,
):
    suffix = document_format.suffix

    return IngestionRequest(
        source=make_source(),
        file_path=Path(
            f"/tmp/synthetic{suffix}"
        ),
        document_format=document_format,
    )


class FakeExtractor:
    def extract(
        self,
        request,
    ):
        return ExtractedDocument(
            source=request.source,
            document_format=(
                request.document_format
            ),
            source_file_name=(
                request.file_path.name
            ),
            blocks=(
                ExtractedBlock(
                    document_id=(
                        request.source.document_id
                    ),
                    block_index=0,
                    text=(
                        "Annual leave requests "
                        "require approval."
                    ),
                    language=(
                        request.source.language
                    ),
                    page_number=3,
                    section_title=(
                        "Annual Leave"
                    ),
                    section_path=(
                        "Human Resources",
                        "Annual Leave",
                    ),
                ),
            ),
        )


class TestIngestionPipeline(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            INGESTION_PIPELINE_VERSION,
            "govba-ingestion-pipeline-v1",
        )

    def test_default_pipeline_supports_pdf_and_docx(self):
        pipeline = (
            IngestionPipeline()
        )

        self.assertEqual(
            set(
                pipeline.extractor_formats
            ),
            {
                DocumentFormat.PDF,
                DocumentFormat.DOCX,
            },
        )

    def test_custom_pipeline_ingests_document(self):
        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        )

        result = pipeline.ingest(
            make_request()
        )

        self.assertIsInstance(
            result,
            IngestionPipelineResult,
        )

        self.assertEqual(
            result.block_count,
            1,
        )

        self.assertEqual(
            result.chunk_count,
            1,
        )

    def test_source_provenance_is_preserved(self):
        request = make_request()

        result = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        ).ingest(
            request
        )

        self.assertEqual(
            result.extracted_document.source,
            request.source,
        )

        self.assertEqual(
            result.chunks[0].document_id,
            request.source.document_id,
        )

    def test_page_provenance_is_preserved(self):
        result = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        ).ingest(
            make_request()
        )

        self.assertEqual(
            result.page_numbers,
            (
                3,
            ),
        )

        self.assertEqual(
            result.chunks[0].page_start,
            3,
        )

        self.assertEqual(
            result.chunks[0].page_end,
            3,
        )

    def test_section_provenance_is_preserved(self):
        result = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        ).ingest(
            make_request()
        )

        chunk = result.chunks[0]

        self.assertEqual(
            chunk.section_title,
            "Annual Leave",
        )

        self.assertEqual(
            chunk.section_path,
            (
                "Human Resources",
                "Annual Leave",
            ),
        )

    def test_chunk_indexes_are_contiguous(self):
        result = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        ).ingest(
            make_request()
        )

        self.assertEqual(
            tuple(
                chunk.chunk_index
                for chunk in result.chunks
            ),
            tuple(
                range(
                    result.chunk_count
                )
            ),
        )

    def test_same_input_produces_same_chunk_ids(self):
        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        )

        request = make_request()

        first = pipeline.ingest(
            request
        )

        second = pipeline.ingest(
            request
        )

        self.assertEqual(
            tuple(
                chunk.chunk_id
                for chunk in first.chunks
            ),
            tuple(
                chunk.chunk_id
                for chunk in second.chunks
            ),
        )

        self.assertEqual(
            tuple(
                chunk.content_hash
                for chunk in first.chunks
            ),
            tuple(
                chunk.content_hash
                for chunk in second.chunks
            ),
        )

    def test_custom_chunking_config_is_used(self):
        class LongExtractor:
            def extract(
                self,
                request,
            ):
                text = " ".join(
                    f"policy{index}"
                    for index
                    in range(100)
                )

                return ExtractedDocument(
                    source=request.source,
                    document_format=(
                        request.document_format
                    ),
                    source_file_name=(
                        request.file_path.name
                    ),
                    blocks=(
                        ExtractedBlock(
                            document_id=(
                                request.source.document_id
                            ),
                            block_index=0,
                            text=text,
                            language=(
                                request.source.language
                            ),
                        ),
                    ),
                )

        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    LongExtractor()
                ),
            },
            chunking_config=(
                ChunkingConfig(
                    target_characters=100,
                    overlap_characters=20,
                )
            ),
        )

        result = pipeline.ingest(
            make_request()
        )

        self.assertGreater(
            result.chunk_count,
            1,
        )

    def test_unsupported_format_is_rejected(self):
        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        )

        with self.assertRaises(
            ValueError
        ):
            pipeline.ingest(
                make_request(
                    DocumentFormat.DOCX
                )
            )

    def test_non_request_is_rejected(self):
        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    FakeExtractor()
                ),
            }
        )

        with self.assertRaises(
            TypeError
        ):
            pipeline.ingest(
                object()
            )

    def test_empty_extractor_map_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            IngestionPipeline(
                extractors={}
            )

    def test_invalid_extractor_key_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            IngestionPipeline(
                extractors={
                    "pdf": FakeExtractor(),
                }
            )

    def test_invalid_extractor_value_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            IngestionPipeline(
                extractors={
                    DocumentFormat.PDF: (
                        object()
                    ),
                }
            )

    def test_wrong_source_from_extractor_is_rejected(self):
        class WrongSourceExtractor:
            def extract(
                self,
                request,
            ):
                wrong_source = make_source(
                    "DOC-WRONG"
                )

                return ExtractedDocument(
                    source=wrong_source,
                    document_format=(
                        request.document_format
                    ),
                    source_file_name=(
                        request.file_path.name
                    ),
                    blocks=(
                        ExtractedBlock(
                            document_id=(
                                "DOC-WRONG"
                            ),
                            block_index=0,
                            text="Wrong source.",
                            language=(
                                SourceLanguage.ENGLISH
                            ),
                        ),
                    ),
                )

        pipeline = IngestionPipeline(
            extractors={
                DocumentFormat.PDF: (
                    WrongSourceExtractor()
                ),
            }
        )

        with self.assertRaises(
            IngestionPipelineError
        ):
            pipeline.ingest(
                make_request()
            )


if __name__ == "__main__":
    unittest.main()
