"""Deterministic DOCX extraction for GovBA-GAR."""

from __future__ import annotations

import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

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


DOCX_EXTRACTION_VERSION = (
    "govba-docx-extraction-v1"
)

_HEADING_STYLE_RE = re.compile(
    r"^Heading\s+([1-9])$",
    re.IGNORECASE,
)


class DOCXExtractionError(RuntimeError):
    """Safe failure raised when DOCX extraction cannot complete."""


def _heading_level(
    paragraph: Paragraph,
) -> int | None:
    try:
        style_name = (
            paragraph.style.name
            if paragraph.style is not None
            else ""
        )
    except Exception:
        return None

    match = _HEADING_STYLE_RE.match(
        style_name.strip()
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def _table_to_text(
    table: Table,
) -> str:
    rows = []

    for row in table.rows:
        cells = []
        previous = None

        for cell in row.cells:
            text = normalize_ingested_text(
                cell.text
            )

            if not text:
                continue

            # Avoid repeated text produced by merged cells.
            if text == previous:
                continue

            cells.append(text)
            previous = text

        if cells:
            rows.append(
                " | ".join(cells)
            )

    return normalize_ingested_text(
        "\n".join(rows)
    )


class PythonDocxExtractor:
    """Extract ordered content from DOCX documents."""

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
            is not DocumentFormat.DOCX
        ):
            raise ValueError(
                "PythonDocxExtractor only accepts "
                "DOCX requests."
            )

        path = request.file_path

        if not path.exists():
            raise FileNotFoundError(
                f"DOCX file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "DOCX path must identify a file."
            )

        try:
            document = Document(
                str(path)
            )
        except Exception as exc:
            raise DOCXExtractionError(
                "Unable to open DOCX document."
            ) from exc

        blocks = []
        block_index = 0
        section_path: list[str] = []

        def append_block(
            text: str,
        ) -> None:
            nonlocal block_index

            normalized = (
                normalize_ingested_text(
                    text
                )
            )

            if not normalized:
                return

            blocks.append(
                ExtractedBlock(
                    document_id=(
                        request.source.document_id
                    ),
                    block_index=block_index,
                    text=normalized,
                    language=(
                        request.source.language
                    ),
                    page_number=None,
                    section_title=(
                        section_path[-1]
                        if section_path
                        else None
                    ),
                    section_path=tuple(
                        section_path
                    ),
                )
            )

            block_index += 1

        try:
            for item in document.iter_inner_content():
                if isinstance(
                    item,
                    Paragraph,
                ):
                    text = (
                        normalize_ingested_text(
                            item.text
                        )
                    )

                    if not text:
                        continue

                    level = _heading_level(
                        item
                    )

                    if level is not None:
                        section_path = (
                            section_path[
                                : level - 1
                            ]
                        )

                        section_path.append(
                            text
                        )

                    append_block(
                        text
                    )

                elif isinstance(
                    item,
                    Table,
                ):
                    table_text = (
                        _table_to_text(
                            item
                        )
                    )

                    if table_text:
                        append_block(
                            table_text
                        )

        except DOCXExtractionError:
            raise

        except Exception as exc:
            raise DOCXExtractionError(
                "DOCX text extraction failed."
            ) from exc

        if not blocks:
            raise DOCXExtractionError(
                "DOCX contains no extractable text."
            )

        return ExtractedDocument(
            source=request.source,
            document_format=(
                DocumentFormat.DOCX
            ),
            source_file_name=path.name,
            blocks=tuple(blocks),
        )


DOCXExtractor = PythonDocxExtractor
