"""Evidence and provenance models for GovBA-GAR retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from govba.rag.models import SourceLanguage


EVIDENCE_SCHEMA_VERSION = "govba-evidence-v1"


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> str:
    """Validate and normalize required text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")

    return value.strip()


def compute_chunk_hash(text: str) -> str:
    """Return a deterministic SHA-256 fingerprint of chunk text."""
    normalized = _require_text(text, "text")

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def compute_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    page_start: int | None = None,
    page_end: int | None = None,
    section_path: tuple[str, ...] = (),
) -> str:
    """Create a deterministic identifier for an evidence chunk."""

    document_id = _require_text(
        document_id,
        "document_id",
    )

    if chunk_index < 0:
        raise ValueError(
            "chunk_index cannot be negative."
        )

    normalized_text = _require_text(
        text,
        "text",
    )

    payload = {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": list(section_path),
        "text": normalized_text,
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return f"{document_id}:chunk:{digest[:24]}"


@dataclass(frozen=True)
class EvidenceChunk:
    """One retrievable unit of authoritative evidence."""

    document_id: str
    chunk_index: int
    text: str
    language: SourceLanguage

    page_start: int | None = None
    page_end: int | None = None

    section_title: str = ""
    section_path: tuple[str, ...] = field(
        default_factory=tuple
    )

    chunk_id: str = ""
    content_hash: str = ""

    created_at: datetime = field(
        default_factory=_utc_now
    )

    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate provenance and generate deterministic identifiers."""

        document_id = _require_text(
            self.document_id,
            "document_id",
        )
        text = _require_text(
            self.text,
            "text",
        )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )
        object.__setattr__(
            self,
            "text",
            text,
        )

        if self.chunk_index < 0:
            raise ValueError(
                "chunk_index cannot be negative."
            )

        if (
            self.page_start is not None
            and self.page_start < 1
        ):
            raise ValueError(
                "page_start must be at least 1."
            )

        if (
            self.page_end is not None
            and self.page_end < 1
        ):
            raise ValueError(
                "page_end must be at least 1."
            )

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end cannot be earlier than page_start."
            )

        if self.created_at.tzinfo is None:
            raise ValueError(
                "created_at must be timezone-aware."
            )

        normalized_path = tuple(
            item.strip()
            for item in self.section_path
            if isinstance(item, str)
            and item.strip()
        )

        object.__setattr__(
            self,
            "section_path",
            normalized_path,
        )

        calculated_hash = compute_chunk_hash(text)

        if self.content_hash:
            if self.content_hash.lower() != calculated_hash:
                raise ValueError(
                    "content_hash does not match chunk text."
                )
        else:
            object.__setattr__(
                self,
                "content_hash",
                calculated_hash,
            )

        calculated_id = compute_chunk_id(
            document_id=document_id,
            chunk_index=self.chunk_index,
            text=text,
            page_start=self.page_start,
            page_end=self.page_end,
            section_path=normalized_path,
        )

        if self.chunk_id:
            if self.chunk_id != calculated_id:
                raise ValueError(
                    "chunk_id does not match chunk provenance."
                )
        else:
            object.__setattr__(
                self,
                "chunk_id",
                calculated_id,
            )

    @property
    def character_count(self) -> int:
        """Return the number of characters in the evidence text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Return a provider-independent whitespace word count."""
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible evidence metadata."""

        data = asdict(self)

        data["language"] = self.language.value
        data["section_path"] = list(
            self.section_path
        )
        data["created_at"] = (
            self.created_at.isoformat()
        )
        data["character_count"] = (
            self.character_count
        )
        data["word_count"] = self.word_count

        return data
