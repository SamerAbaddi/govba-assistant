"""Provenance-aware document ingestion pipeline for GovBA-GAR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from govba.rag.chunking import (
    ChunkingConfig,
    chunk_extracted_document,
)
from govba.rag.docx_extractor import (
    PythonDocxExtractor,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.ingestion import (
    DocumentExtractor,
    DocumentFormat,
    ExtractedDocument,
    IngestionRequest,
)
from govba.rag.pdf_extractor import (
    PyMuPDFExtractor,
)


INGESTION_PIPELINE_VERSION = (
    "govba-ingestion-pipeline-v1"
)


class IngestionPipelineError(RuntimeError):
    """Safe failure raised for invalid pipeline output."""


@dataclass(frozen=True)
class IngestionPipelineResult:
    """Validated result of one document-ingestion run."""

    request: IngestionRequest
    extracted_document: ExtractedDocument
    chunks: tuple[EvidenceChunk, ...]
    pipeline_version: str = (
        INGESTION_PIPELINE_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.request,
            IngestionRequest,
        ):
            raise TypeError(
                "request must be an IngestionRequest."
            )

        if not isinstance(
            self.extracted_document,
            ExtractedDocument,
        ):
            raise TypeError(
                "extracted_document must be "
                "an ExtractedDocument."
            )

        chunks = tuple(
            self.chunks
        )

        if not chunks:
            raise ValueError(
                "Pipeline result requires "
                "at least one evidence chunk."
            )

        for chunk in chunks:
            if not isinstance(
                chunk,
                EvidenceChunk,
            ):
                raise TypeError(
                    "chunks must contain "
                    "EvidenceChunk values."
                )

        source = self.request.source
        extracted = self.extracted_document

        if (
            extracted.source
            != source
        ):
            raise ValueError(
                "Extracted source does not match "
                "the ingestion request."
            )

        if (
            extracted.document_format
            != self.request.document_format
        ):
            raise ValueError(
                "Extracted document format does "
                "not match the ingestion request."
            )

        if (
            extracted.source_file_name
            != self.request.file_path.name
        ):
            raise ValueError(
                "Extracted source file name does "
                "not match the ingestion request."
            )

        expected_indexes = tuple(
            range(
                len(chunks)
            )
        )

        actual_indexes = tuple(
            chunk.chunk_index
            for chunk in chunks
        )

        if (
            actual_indexes
            != expected_indexes
        ):
            raise ValueError(
                "Evidence chunk indexes must be "
                "contiguous and ordered from zero."
            )

        for chunk in chunks:
            if (
                chunk.document_id
                != source.document_id
            ):
                raise ValueError(
                    "Evidence chunk document_id "
                    "does not match source."
                )

        object.__setattr__(
            self,
            "chunks",
            chunks,
        )

    @property
    def block_count(self) -> int:
        return (
            self.extracted_document.block_count
        )

    @property
    def chunk_count(self) -> int:
        return len(
            self.chunks
        )

    @property
    def page_numbers(
        self,
    ) -> tuple[int, ...]:
        return (
            self.extracted_document.page_numbers
        )


class IngestionPipeline:
    """Coordinate extraction and deterministic chunking."""

    def __init__(
        self,
        *,
        extractors: Mapping[
            DocumentFormat,
            DocumentExtractor,
        ]
        | None = None,
        chunking_config: (
            ChunkingConfig | None
        ) = None,
    ) -> None:
        if extractors is None:
            extractor_map = {
                DocumentFormat.PDF: (
                    PyMuPDFExtractor()
                ),
                DocumentFormat.DOCX: (
                    PythonDocxExtractor()
                ),
            }
        else:
            if not isinstance(
                extractors,
                Mapping,
            ):
                raise TypeError(
                    "extractors must be a mapping."
                )

            extractor_map = dict(
                extractors
            )

        if not extractor_map:
            raise ValueError(
                "At least one document extractor "
                "is required."
            )

        for (
            document_format,
            extractor,
        ) in extractor_map.items():
            if not isinstance(
                document_format,
                DocumentFormat,
            ):
                raise TypeError(
                    "Extractor keys must be "
                    "DocumentFormat values."
                )

            if not isinstance(
                extractor,
                DocumentExtractor,
            ):
                raise TypeError(
                    "Extractor values must satisfy "
                    "DocumentExtractor."
                )

        if chunking_config is None:
            chunking_config = (
                ChunkingConfig()
            )

        if not isinstance(
            chunking_config,
            ChunkingConfig,
        ):
            raise TypeError(
                "chunking_config must be "
                "a ChunkingConfig."
            )

        self._extractors = (
            extractor_map
        )
        self._chunking_config = (
            chunking_config
        )

    @property
    def extractor_formats(
        self,
    ) -> tuple[DocumentFormat, ...]:
        return tuple(
            sorted(
                self._extractors,
                key=lambda item: item.value,
            )
        )

    @property
    def chunking_config(
        self,
    ) -> ChunkingConfig:
        return self._chunking_config

    def ingest(
        self,
        request: IngestionRequest,
    ) -> IngestionPipelineResult:
        if not isinstance(
            request,
            IngestionRequest,
        ):
            raise TypeError(
                "request must be an IngestionRequest."
            )

        extractor = self._extractors.get(
            request.document_format
        )

        if extractor is None:
            raise ValueError(
                "No extractor is configured for "
                f"{request.document_format.value}."
            )

        extracted = extractor.extract(
            request
        )

        if not isinstance(
            extracted,
            ExtractedDocument,
        ):
            raise IngestionPipelineError(
                "Extractor returned an invalid result."
            )

        if (
            extracted.source
            != request.source
        ):
            raise IngestionPipelineError(
                "Extractor returned the wrong source."
            )

        if (
            extracted.document_format
            != request.document_format
        ):
            raise IngestionPipelineError(
                "Extractor returned the wrong "
                "document format."
            )

        if (
            extracted.source_file_name
            != request.file_path.name
        ):
            raise IngestionPipelineError(
                "Extractor returned the wrong "
                "source file name."
            )

        chunks = (
            chunk_extracted_document(
                extracted,
                self._chunking_config,
            )
        )

        return IngestionPipelineResult(
            request=request,
            extracted_document=extracted,
            chunks=chunks,
        )
