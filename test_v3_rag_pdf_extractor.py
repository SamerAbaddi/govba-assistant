"""Offline tests for GovBA-GAR PDF extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from govba.rag import (
    AuthoritativeSource,
    DocumentExtractor,
    DocumentFormat,
    DocumentType,
    IngestionRequest,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.pdf_extractor import (
    PDF_EXTRACTION_VERSION,
    PDFExtractionError,
    PyMuPDFExtractor,
)


def make_source():
    return AuthoritativeSource(
        document_id="DOC-PDF-001",
        title="Synthetic PDF Procedure",
        issuing_authority=(
            "Synthetic GovBA Test Authority"
        ),
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
    )


def make_request(path):
    return IngestionRequest(
        source=make_source(),
        file_path=Path(path),
        document_format=DocumentFormat.PDF,
    )


def create_pdf(
    path,
    pages,
):
    document = pymupdf.open()

    try:
        for text in pages:
            page = document.new_page()

            if text is not None:
                page.insert_text(
                    (72, 72),
                    text,
                )

        document.save(path)

    finally:
        document.close()


class TestPDFExtractor(unittest.TestCase):
    def test_version_is_stable(self):
        self.assertEqual(
            PDF_EXTRACTION_VERSION,
            "govba-pdf-extraction-v1",
        )

    def test_satisfies_extractor_protocol(self):
        self.assertIsInstance(
            PyMuPDFExtractor(),
            DocumentExtractor,
        )

    def test_missing_file_is_rejected(self):
        extractor = PyMuPDFExtractor()

        with self.assertRaises(
            FileNotFoundError
        ):
            extractor.extract(
                make_request(
                    "/tmp/definitely-missing.pdf"
                )
            )

    def test_non_request_is_rejected(self):
        with self.assertRaises(TypeError):
            PyMuPDFExtractor().extract(
                object()
            )

    def test_non_pdf_request_is_rejected(self):
        request = IngestionRequest(
            source=make_source(),
            file_path=Path(
                "/tmp/document.docx"
            ),
            document_format=(
                DocumentFormat.DOCX
            ),
        )

        with self.assertRaises(ValueError):
            PyMuPDFExtractor().extract(
                request
            )

    def test_single_page_pdf_is_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "single.pdf"
            )

            create_pdf(
                path,
                (
                    "Annual leave requires approval.",
                ),
            )

            result = (
                PyMuPDFExtractor()
                .extract(
                    make_request(path)
                )
            )

            self.assertEqual(
                result.block_count,
                1,
            )

            self.assertEqual(
                result.blocks[0].page_number,
                1,
            )

            self.assertIn(
                "Annual leave",
                result.blocks[0].text,
            )

    def test_multiple_pages_preserve_page_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "multiple.pdf"
            )

            create_pdf(
                path,
                (
                    "First page content.",
                    "Second page content.",
                ),
            )

            result = (
                PyMuPDFExtractor()
                .extract(
                    make_request(path)
                )
            )

            self.assertEqual(
                result.page_numbers,
                (
                    1,
                    2,
                ),
            )

    def test_block_indexes_are_contiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "indexes.pdf"
            )

            create_pdf(
                path,
                (
                    "First page.",
                    "Second page.",
                ),
            )

            result = (
                PyMuPDFExtractor()
                .extract(
                    make_request(path)
                )
            )

            self.assertEqual(
                tuple(
                    block.block_index
                    for block
                    in result.blocks
                ),
                tuple(
                    range(
                        result.block_count
                    )
                ),
            )

    def test_source_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "metadata.pdf"
            )

            create_pdf(
                path,
                (
                    "Policy content.",
                ),
            )

            request = make_request(
                path
            )

            result = (
                PyMuPDFExtractor()
                .extract(request)
            )

            self.assertEqual(
                result.source,
                request.source,
            )

            self.assertEqual(
                result.source_file_name,
                "metadata.pdf",
            )

            self.assertEqual(
                result.document_format,
                DocumentFormat.PDF,
            )

    def test_language_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "language.pdf"
            )

            create_pdf(
                path,
                (
                    "Synthetic government text.",
                ),
            )

            result = (
                PyMuPDFExtractor()
                .extract(
                    make_request(path)
                )
            )

            self.assertEqual(
                result.blocks[0].language,
                SourceLanguage.ENGLISH,
            )

    def test_empty_pdf_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "empty.pdf"
            )

            create_pdf(
                path,
                (
                    None,
                ),
            )

            with self.assertRaises(
                PDFExtractionError
            ):
                PyMuPDFExtractor().extract(
                    make_request(path)
                )

    def test_invalid_pdf_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "invalid.pdf"
            )

            path.write_text(
                "This is not a PDF.",
                encoding="utf-8",
            )

            with self.assertRaises(
                PDFExtractionError
            ):
                PyMuPDFExtractor().extract(
                    make_request(path)
                )


if __name__ == "__main__":
    unittest.main()
