"""Deterministic page/section-aware chunking for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass

from govba.rag.evidence import EvidenceChunk
from govba.rag.ingestion import ExtractedDocument
from govba.rag.normalization import normalize_ingested_text


CHUNKING_VERSION = "govba-chunking-v1"

DEFAULT_TARGET_CHARACTERS = 1200
DEFAULT_OVERLAP_CHARACTERS = 150

_BOUNDARY_MARKERS = (
    "\n\n",
    "\n",
    ". ",
    "؟ ",
    "? ",
    "! ",
    "؛ ",
    "; ",
    "، ",
    ", ",
    " ",
)

_BOUNDARY_SEARCH_RATIO = 0.60


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for deterministic evidence chunking."""

    target_characters: int = (
        DEFAULT_TARGET_CHARACTERS
    )
    overlap_characters: int = (
        DEFAULT_OVERLAP_CHARACTERS
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.target_characters,
                bool,
            )
            or not isinstance(
                self.target_characters,
                int,
            )
            or self.target_characters < 1
        ):
            raise ValueError(
                "target_characters must be "
                "a positive integer."
            )

        if (
            isinstance(
                self.overlap_characters,
                bool,
            )
            or not isinstance(
                self.overlap_characters,
                int,
            )
            or self.overlap_characters < 0
        ):
            raise ValueError(
                "overlap_characters must be "
                "a non-negative integer."
            )

        if (
            self.overlap_characters
            >= self.target_characters
        ):
            raise ValueError(
                "overlap_characters must be "
                "smaller than target_characters."
            )


def _find_chunk_end(
    text: str,
    start: int,
    target_characters: int,
) -> int:
    desired_end = min(
        start + target_characters,
        len(text),
    )

    if desired_end >= len(text):
        return len(text)

    span = desired_end - start

    search_start = (
        start
        + int(
            span * _BOUNDARY_SEARCH_RATIO
        )
    )

    best_end = -1

    for marker in _BOUNDARY_MARKERS:
        index = text.rfind(
            marker,
            search_start,
            desired_end,
        )

        if index >= search_start:
            candidate_end = (
                index + len(marker)
            )

            if candidate_end > best_end:
                best_end = candidate_end

    if best_end > start:
        return best_end

    return desired_end


def _align_overlap_start(
    text: str,
    proposed_start: int,
    previous_end: int,
) -> int:
    if proposed_start <= 0:
        return 0

    if proposed_start >= previous_end:
        return previous_end

    if (
        text[proposed_start - 1].isspace()
        or text[proposed_start].isspace()
    ):
        while (
            proposed_start < previous_end
            and text[
                proposed_start
            ].isspace()
        ):
            proposed_start += 1

        return proposed_start

    next_space = proposed_start

    while (
        next_space < previous_end
        and not text[next_space].isspace()
    ):
        next_space += 1

    while (
        next_space < previous_end
        and text[next_space].isspace()
    ):
        next_space += 1

    if next_space < previous_end:
        return next_space

    return proposed_start


def _split_normalized_text(
    text: str,
    config: ChunkingConfig,
) -> tuple[str, ...]:
    if not text:
        return ()

    if (
        len(text)
        <= config.target_characters
    ):
        return (text,)

    chunks = []
    start = 0

    while start < len(text):
        end = _find_chunk_end(
            text,
            start,
            config.target_characters,
        )

        if end <= start:
            raise RuntimeError(
                "Chunking failed to make progress."
            )

        chunk_text = text[
            start:end
        ].strip()

        if chunk_text:
            chunks.append(
                chunk_text
            )

        if end >= len(text):
            break

        proposed_start = max(
            0,
            end
            - config.overlap_characters,
        )

        next_start = _align_overlap_start(
            text,
            proposed_start,
            end,
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return tuple(chunks)


def chunk_extracted_document(
    document: ExtractedDocument,
    config: ChunkingConfig | None = None,
) -> tuple[EvidenceChunk, ...]:
    """Convert extracted blocks into deterministic EvidenceChunk objects."""

    if not isinstance(
        document,
        ExtractedDocument,
    ):
        raise TypeError(
            "document must be an "
            "ExtractedDocument."
        )

    if config is None:
        config = ChunkingConfig()

    if not isinstance(
        config,
        ChunkingConfig,
    ):
        raise TypeError(
            "config must be a ChunkingConfig."
        )

    evidence_chunks = []
    chunk_index = 0

    for block in document.blocks:
        normalized_text = (
            normalize_ingested_text(
                block.text
            )
        )

        if not normalized_text:
            continue

        pieces = _split_normalized_text(
            normalized_text,
            config,
        )

        for piece in pieces:
            page = block.page_number

            evidence_chunks.append(
                EvidenceChunk(
                    document_id=(
                        document.source.document_id
                    ),
                    chunk_index=chunk_index,
                    text=piece,
                    language=block.language,
                    page_start=page,
                    page_end=page,
                    section_title=(
                        block.section_title
                    ),
                    section_path=(
                        block.section_path
                    ),
                )
            )

            chunk_index += 1

    if not evidence_chunks:
        raise ValueError(
            "Document produced no evidence chunks."
        )

    return tuple(
        evidence_chunks
    )
