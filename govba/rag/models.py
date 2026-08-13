"""Core authoritative-source models for GovBA-GAR retrieval.

These models describe government knowledge sources independently of any
specific vector database, embedding model, or AI provider.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse


SOURCE_SCHEMA_VERSION = "govba-source-v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DocumentType(str, Enum):
    """Supported classes of authoritative government documents."""

    LAW = "law"
    REGULATION = "regulation"
    POLICY = "policy"
    PROCEDURE = "procedure"
    CIRCULAR = "circular"
    DECISION = "decision"
    MANUAL = "manual"
    GUIDELINE = "guideline"
    MEMORANDUM = "memorandum"
    OTHER = "other"


class SourceLanguage(str, Enum):
    """Language coverage of an authoritative source."""

    ARABIC = "ar"
    ENGLISH = "en"
    BILINGUAL = "ar-en"
    OTHER = "other"


class SourceStatus(str, Enum):
    """Known temporal/governance status of a source."""

    CURRENT = "current"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    FUTURE = "future"
    UNKNOWN = "unknown"


def compute_content_hash(content: str | bytes) -> str:
    """Return a deterministic SHA-256 fingerprint for source content."""

    if isinstance(content, str):
        content = content.encode("utf-8")

    if not isinstance(content, bytes):
        raise TypeError("content must be str or bytes.")

    return hashlib.sha256(content).hexdigest()


def _utc_now() -> datetime:
    """Return a timezone-aware UTC ingestion timestamp."""

    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> str:
    """Validate and normalize a required text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")

    return value.strip()


@dataclass(frozen=True)
class AuthoritativeSource:
    """Metadata describing one authoritative government source."""

    document_id: str
    title: str
    issuing_authority: str

    document_type: DocumentType
    language: SourceLanguage

    jurisdiction: str = "Jordan"

    publication_date: date | None = None
    effective_from: date | None = None
    effective_until: date | None = None

    version: str = ""
    status: SourceStatus = SourceStatus.UNKNOWN

    supersedes: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: tuple[str, ...] = field(default_factory=tuple)

    official_source_url: str = ""
    content_hash: str = ""

    ingested_at: datetime = field(default_factory=_utc_now)

    schema_version: str = SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate metadata invariants after construction."""

        object.__setattr__(
            self,
            "document_id",
            _require_text(
                self.document_id,
                "document_id",
            ),
        )

        object.__setattr__(
            self,
            "title",
            _require_text(
                self.title,
                "title",
            ),
        )

        object.__setattr__(
            self,
            "issuing_authority",
            _require_text(
                self.issuing_authority,
                "issuing_authority",
            ),
        )

        object.__setattr__(
            self,
            "jurisdiction",
            _require_text(
                self.jurisdiction,
                "jurisdiction",
            ),
        )

        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_until < self.effective_from
        ):
            raise ValueError(
                "effective_until cannot be earlier than effective_from."
            )

        if self.document_id in self.supersedes:
            raise ValueError(
                "A document cannot supersede itself."
            )

        if self.document_id in self.superseded_by:
            raise ValueError(
                "A document cannot be superseded by itself."
            )

        if self.official_source_url:
            parsed = urlparse(self.official_source_url)

            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    "official_source_url must be a valid HTTP or HTTPS URL."
                )

        if (
            self.content_hash
            and not _SHA256_PATTERN.fullmatch(
                self.content_hash.lower()
            )
        ):
            raise ValueError(
                "content_hash must be a SHA-256 hexadecimal digest."
            )

        if self.ingested_at.tzinfo is None:
            raise ValueError(
                "ingested_at must be timezone-aware."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible source metadata."""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "title": self.title,
            "issuing_authority": self.issuing_authority,
            "document_type": self.document_type.value,
            "language": self.language.value,
            "jurisdiction": self.jurisdiction,
            "publication_date": (
                self.publication_date.isoformat()
                if self.publication_date
                else None
            ),
            "effective_from": (
                self.effective_from.isoformat()
                if self.effective_from
                else None
            ),
            "effective_until": (
                self.effective_until.isoformat()
                if self.effective_until
                else None
            ),
            "version": self.version,
            "status": self.status.value,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "official_source_url": self.official_source_url,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at.isoformat(),
        }
