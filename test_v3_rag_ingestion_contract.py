"""Offline tests for GovBA-GAR document ingestion contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

from govba.rag.ingestion import (
    DocumentExtractor,
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocument,
    IngestionRequest,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_source() -> AuthoritativeSource:
    return AuthoritativeSource(
        document_id="DOC-INGEST-001",
        title="Synthetic Service Procedure",
        issuing_authority=(
            "Synthetic GovBA Test Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
    )


def make_blocks():
    return (
        ExtractedBlock(
            document_id="DOC-INGEST-001",
            block_index=0,
            text="First extracted block.",
            language=SourceLanguage.ENGLISH,
            page_number=1,
            section_title="Introduction",
            section_path=(
                " Part One ",
                " Introduction ",
            ),
        ),
        ExtractedBlock(
            document_id="DOC-INGEST-001",
            block_index=1,
            text="Second extracted block.",
            language=SourceLanguage.ENGLISH,
            page_number=2,
        ),
    )


class FakeExtractor:
    def extract(
        self,
        request: IngestionRequest,
    ) -> ExtractedDocument:
        return ExtractedDocument(
            source=request.source,
            document_format=(
                request.document_format
            ),
            source_file_name=(
                request.file_path.name
            ),
            blocks=make_blocks(),
        )


class TestDocumentIngestionContract(
    unittest.TestCase
):
    def test_supported_document_formats(self):
        self.assertEqual(
            DocumentFormat.PDF.value,
            "pdf",
        )
        self.assertEqual(
            DocumentFormat.DOCX.value,
            "docx",
        )

    def test_pdf_ingestion_request(self):
        request = IngestionRequest(
            source=make_source(),
            file_path=Path(
                "/tmp/synthetic.pdf"
            ),
            document_format=(
                DocumentFormat.PDF
            ),
        )

        self.assertEqual(
            request.file_path.name,
            "synthetic.pdf",
        )

    def test_docx_ingestion_request(self):
        request = IngestionRequest(
            source=make_source(),
            file_path=Path(
                "/tmp/synthetic.docx"
            ),
            document_format=(
                DocumentFormat.DOCX
            ),
        )

        self.assertEqual(
            request.document_format,
            DocumentFormat.DOCX,
        )

    def test_request_does_not_require_file_io(self):
        request = IngestionRequest(
            source=make_source(),
            file_path=Path(
                "/definitely/not/"
                "present/document.pdf"
            ),
            document_format=(
                DocumentFormat.PDF
            ),
        )

        self.assertFalse(
            request.file_path.exists()
        )

    def test_extension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            IngestionRequest(
                source=make_source(),
                file_path=Path(
                    "/tmp/document.docx"
                ),
                document_format=(
                    DocumentFormat.PDF
                ),
            )

    def test_block_values_are_normalized(self):
        block = ExtractedBlock(
            document_id=(
                " DOC-INGEST-001 "
            ),
            block_index=0,
            text="  Extracted text.  ",
            language=(
                SourceLanguage.ENGLISH
            ),
            section_title="  Scope  ",
            section_path=(
                " Part One ",
                "",
                " Scope ",
            ),
        )

        self.assertEqual(
            block.document_id,
            "DOC-INGEST-001",
        )
        self.assertEqual(
            block.text,
            "Extracted text.",
        )
        self.assertEqual(
            block.section_title,
            "Scope",
        )
        self.assertEqual(
            block.section_path,
            (
                "Part One",
                "Scope",
            ),
        )

    def test_blank_block_text_is_rejected(self):
        with self.assertRaises(ValueError):
            ExtractedBlock(
                document_id=(
                    "DOC-INGEST-001"
                ),
                block_index=0,
                text="   ",
                language=(
                    SourceLanguage.ENGLISH
                ),
            )

    def test_invalid_page_number_is_rejected(self):
        for page_number in (
            0,
            -1,
            True,
        ):
            with self.subTest(
                page_number=page_number
            ):
                with self.assertRaises(
                    ValueError
                ):
                    ExtractedBlock(
                        document_id=(
                            "DOC-INGEST-001"
                        ),
                        block_index=0,
                        text="Text.",
                        language=(
                            SourceLanguage.ENGLISH
                        ),
                        page_number=(
                            page_number
                        ),
                    )

    def test_document_requires_blocks(self):
        with self.assertRaises(ValueError):
            ExtractedDocument(
                source=make_source(),
                document_format=(
                    DocumentFormat.PDF
                ),
                source_file_name=(
                    "document.pdf"
                ),
                blocks=(),
            )

    def test_block_source_mismatch_is_rejected(self):
        bad_block = ExtractedBlock(
            document_id="OTHER-DOC",
            block_index=0,
            text="Text.",
            language=(
                SourceLanguage.ENGLISH
            ),
        )

        with self.assertRaises(ValueError):
            ExtractedDocument(
                source=make_source(),
                document_format=(
                    DocumentFormat.PDF
                ),
                source_file_name=(
                    "document.pdf"
                ),
                blocks=(bad_block,),
            )

    def test_block_indexes_must_be_contiguous(self):
        blocks = (
            ExtractedBlock(
                document_id=(
                    "DOC-INGEST-001"
                ),
                block_index=0,
                text="First.",
                language=(
                    SourceLanguage.ENGLISH
                ),
            ),
            ExtractedBlock(
                document_id=(
                    "DOC-INGEST-001"
                ),
                block_index=2,
                text="Second.",
                language=(
                    SourceLanguage.ENGLISH
                ),
            ),
        )

        with self.assertRaises(ValueError):
            ExtractedDocument(
                source=make_source(),
                document_format=(
                    DocumentFormat.PDF
                ),
                source_file_name=(
                    "document.pdf"
                ),
                blocks=blocks,
            )

    def test_document_properties(self):
        document = ExtractedDocument(
            source=make_source(),
            document_format=(
                DocumentFormat.PDF
            ),
            source_file_name=(
                "document.pdf"
            ),
            blocks=make_blocks(),
        )

        self.assertEqual(
            document.block_count,
            2,
        )
        self.assertEqual(
            document.page_numbers,
            (
                1,
                2,
            ),
        )
        self.assertEqual(
            document.full_text,
            (
                "First extracted block."
                "\n\n"
                "Second extracted block."
            ),
        )
        self.assertEqual(
            document.character_count,
            (
                len(
                    "First extracted block."
                )
                + len(
                    "Second extracted block."
                )
            ),
        )

    def test_extractor_protocol(self):
        extractor = FakeExtractor()

        self.assertIsInstance(
            extractor,
            DocumentExtractor,
        )

        request = IngestionRequest(
            source=make_source(),
            file_path=Path(
                "/tmp/document.pdf"
            ),
            document_format=(
                DocumentFormat.PDF
            ),
        )

        result = extractor.extract(
            request
        )

        self.assertEqual(
            result.block_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
