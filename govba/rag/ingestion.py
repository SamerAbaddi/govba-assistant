"""Provider-independent document ingestion contracts for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from govba.rag.models import (
    AuthoritativeSource,
    SourceLanguage,
)


class DocumentFormat(str, Enum):
    """Document formats supported by the ingestion pipeline."""

    PDF = "pdf"
    DOCX = "docx"

    @property
    def suffix(self) -> str:
        return f".{self.value}"


@dataclass(frozen=True)
class IngestionRequest:
    """Request to extract a single authoritative document."""

    source: AuthoritativeSource
    file_path: Path
    document_format: DocumentFormat

    def __post_init__(self) -> None:
        if not isinstance(
            self.source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "source must be an AuthoritativeSource."
            )

        path = Path(self.file_path)

        if str(path).strip() in {"", "."}:
            raise ValueError(
                "file_path must identify a document."
            )

        object.__setattr__(
            self,
            "file_path",
            path,
        )

        if not isinstance(
            self.document_format,
            DocumentFormat,
        ):
            raise TypeError(
                "document_format must be a DocumentFormat."
            )

        if (
            path.suffix.casefold()
            != self.document_format.suffix
        ):
            raise ValueError(
                "file extension does not match "
                "document_format."
            )


@dataclass(frozen=True)
class ExtractedBlock:
    """Ordered text block extracted from a source document."""

    document_id: str
    block_index: int
    text: str
    language: SourceLanguage
    page_number: int | None = None
    section_title: str | None = None
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        document_id = self.document_id.strip()

        if not document_id:
            raise ValueError(
                "document_id must not be blank."
            )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        if (
            isinstance(self.block_index, bool)
            or not isinstance(
                self.block_index,
                int,
            )
            or self.block_index < 0
        ):
            raise ValueError(
                "block_index must be a "
                "non-negative integer."
            )

        text = self.text.strip()

        if not text:
            raise ValueError(
                "text must not be blank."
            )

        object.__setattr__(
            self,
            "text",
            text,
        )

        if not isinstance(
            self.language,
            SourceLanguage,
        ):
            raise TypeError(
                "language must be a SourceLanguage."
            )

        if self.page_number is not None:
            if (
                isinstance(
                    self.page_number,
                    bool,
                )
                or not isinstance(
                    self.page_number,
                    int,
                )
                or self.page_number < 1
            ):
                raise ValueError(
                    "page_number must be a "
                    "positive integer or None."
                )

        if self.section_title is not None:
            normalized_title = (
                self.section_title.strip()
            )

            object.__setattr__(
                self,
                "section_title",
                normalized_title or None,
            )

        normalized_path = tuple(
            part.strip()
            for part in self.section_path
            if part.strip()
        )

        object.__setattr__(
            self,
            "section_path",
            normalized_path,
        )


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized extraction result before chunking."""

    source: AuthoritativeSource
    document_format: DocumentFormat
    source_file_name: str
    blocks: tuple[ExtractedBlock, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "source must be an AuthoritativeSource."
            )

        if not isinstance(
            self.document_format,
            DocumentFormat,
        ):
            raise TypeError(
                "document_format must be a DocumentFormat."
            )

        file_name = self.source_file_name.strip()

        if (
            not file_name
            or Path(file_name).name != file_name
            or "\\" in file_name
        ):
            raise ValueError(
                "source_file_name must be "
                "a file name without a path."
            )

        object.__setattr__(
            self,
            "source_file_name",
            file_name,
        )

        blocks = tuple(self.blocks)

        if not blocks:
            raise ValueError(
                "ExtractedDocument requires "
                "at least one block."
            )

        for block in blocks:
            if not isinstance(
                block,
                ExtractedBlock,
            ):
                raise TypeError(
                    "blocks must contain "
                    "ExtractedBlock values."
                )

            if (
                block.document_id
                != self.source.document_id
            ):
                raise ValueError(
                    "block document_id does not "
                    "match source document_id."
                )

        indexes = tuple(
            block.block_index
            for block in blocks
        )

        expected_indexes = tuple(
            range(len(blocks))
        )

        if indexes != expected_indexes:
            raise ValueError(
                "block indexes must be contiguous "
                "and ordered from zero."
            )

        object.__setattr__(
            self,
            "blocks",
            blocks,
        )

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def character_count(self) -> int:
        return sum(
            len(block.text)
            for block in self.blocks
        )

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            block.text
            for block in self.blocks
        )

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    block.page_number
                    for block in self.blocks
                    if block.page_number
                    is not None
                }
            )
        )


@runtime_checkable
class DocumentExtractor(Protocol):
    """Contract implemented by concrete document extractors."""

    def extract(
        self,
        request: IngestionRequest,
    ) -> ExtractedDocument:
        ...
