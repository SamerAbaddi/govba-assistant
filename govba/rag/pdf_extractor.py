"""Deterministic PDF extraction for GovBA-GAR."""

from __future__ import annotations

import pymupdf

from govba.rag.ingestion import (
    DocumentExtractor,
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocument,
    IngestionRequest,
)
from govba.rag.normalization import (
    normalize_ingested_text,
)


PDF_EXTRACTION_VERSION = (
    "govba-pdf-extraction-v1"
)


class PDFExtractionError(RuntimeError):
    """Safe failure raised when PDF extraction cannot complete."""


class PyMuPDFExtractor:
    """Extract ordered text blocks from PDF documents."""

    def extract(
        self,
        request: IngestionRequest,
    ) -> ExtractedDocument:
        if not isinstance(
            request,
            IngestionRequest,
        ):
            raise TypeError(
                "request must be an IngestionRequest."
            )

        if (
            request.document_format
            is not DocumentFormat.PDF
        ):
            raise ValueError(
                "PyMuPDFExtractor only accepts PDF requests."
            )

        path = request.file_path

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "PDF path must identify a file."
            )

        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise PDFExtractionError(
                "Unable to open PDF document."
            ) from exc

        try:
            if not document.is_pdf:
                raise PDFExtractionError(
                    "Input content is not a PDF document."
                )

            if document.needs_pass:
                raise PDFExtractionError(
                    "Password-protected PDF documents "
                    "are not supported."
                )

            if document.page_count < 1:
                raise PDFExtractionError(
                    "PDF document contains no pages."
                )

            blocks = []
            block_index = 0

            for page_index in range(
                document.page_count
            ):
                page = document[
                    page_index
                ]

                try:
                    page_blocks = page.get_text(
                        "blocks",
                        sort=True,
                    )
                except Exception as exc:
                    raise PDFExtractionError(
                        "PDF text extraction failed."
                    ) from exc

                for raw_block in page_blocks:
                    if len(raw_block) < 5:
                        continue

                    if (
                        len(raw_block) >= 7
                        and raw_block[6] != 0
                    ):
                        continue

                    text = normalize_ingested_text(
                        str(
                            raw_block[4]
                        )
                    )

                    if not text:
                        continue

                    blocks.append(
                        ExtractedBlock(
                            document_id=(
                                request.source.document_id
                            ),
                            block_index=block_index,
                            text=text,
                            language=(
                                request.source.language
                            ),
                            page_number=(
                                page_index + 1
                            ),
                        )
                    )

                    block_index += 1

            if not blocks:
                raise PDFExtractionError(
                    "PDF contains no extractable text. "
                    "OCR may be required."
                )

            return ExtractedDocument(
                source=request.source,
                document_format=(
                    DocumentFormat.PDF
                ),
                source_file_name=path.name,
                blocks=tuple(blocks),
            )

        finally:
            document.close()


PDFExtractor = PyMuPDFExtractor
