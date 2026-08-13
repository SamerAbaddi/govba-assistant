"""Authoritative corpus registration for GovBA-GAR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from govba.rag.ingestion import (
    DocumentFormat,
    IngestionRequest,
)
from govba.rag.models import (
    AuthoritativeSource,
)


CORPUS_MANIFEST_SCHEMA_VERSION = (
    "govba-corpus-manifest-v1"
)


def _require_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _normalize_relative_path(
    value: str,
    document_format: DocumentFormat,
) -> str:
    value = _require_text(
        value,
        "relative_path",
    )

    if "\\" in value:
        raise ValueError(
            "relative_path must use forward slashes."
        )

    raw_parts = value.split("/")

    if any(
        part in {
            "",
            ".",
            "..",
        }
        for part in raw_parts
    ):
        raise ValueError(
            "relative_path contains an unsafe "
            "path component."
        )

    path = PurePosixPath(
        value
    )

    if path.is_absolute():
        raise ValueError(
            "relative_path must not be absolute."
        )

    normalized = str(path)

    if (
        PurePosixPath(normalized)
        .suffix
        .casefold()
        != document_format.suffix
    ):
        raise ValueError(
            "relative_path extension does not "
            "match document_format."
        )

    return normalized


@dataclass(frozen=True)
class CorpusEntry:
    """Registration of one authoritative corpus document."""

    source: AuthoritativeSource
    document_format: DocumentFormat
    relative_path: str
    approved_for_ingestion: bool = False

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

        if not isinstance(
            self.approved_for_ingestion,
            bool,
        ):
            raise TypeError(
                "approved_for_ingestion must be boolean."
            )

        normalized_path = (
            _normalize_relative_path(
                self.relative_path,
                self.document_format,
            )
        )

        object.__setattr__(
            self,
            "relative_path",
            normalized_path,
        )

        if (
            self.approved_for_ingestion
            and not self.source.official_source_url
        ):
            raise ValueError(
                "Approved corpus entries require "
                "an official_source_url."
            )

    @property
    def document_id(self) -> str:
        return self.source.document_id

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "document_id": (
                self.source.document_id
            ),
            "title": self.source.title,
            "issuing_authority": (
                self.source.issuing_authority
            ),
            "document_type": (
                self.source.document_type.value
            ),
            "language": (
                self.source.language.value
            ),
            "jurisdiction": (
                self.source.jurisdiction
            ),
            "status": (
                self.source.status.value
            ),
            "official_source_url": (
                self.source.official_source_url
            ),
            "document_format": (
                self.document_format.value
            ),
            "relative_path": (
                self.relative_path
            ),
            "approved_for_ingestion": (
                self.approved_for_ingestion
            ),
        }


@dataclass(frozen=True)
class CorpusManifest:
    """Versioned registration manifest for authoritative sources."""

    manifest_id: str
    name: str
    entries: tuple[CorpusEntry, ...]
    schema_version: str = (
        CORPUS_MANIFEST_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        manifest_id = _require_text(
            self.manifest_id,
            "manifest_id",
        )

        name = _require_text(
            self.name,
            "name",
        )

        object.__setattr__(
            self,
            "manifest_id",
            manifest_id,
        )

        object.__setattr__(
            self,
            "name",
            name,
        )

        entries = tuple(
            self.entries
        )

        for entry in entries:
            if not isinstance(
                entry,
                CorpusEntry,
            ):
                raise TypeError(
                    "entries must contain "
                    "CorpusEntry values."
                )

        document_ids = [
            entry.document_id
            for entry in entries
        ]

        if (
            len(set(document_ids))
            != len(document_ids)
        ):
            raise ValueError(
                "Corpus document IDs must be unique."
            )

        normalized_paths = [
            entry.relative_path.casefold()
            for entry in entries
        ]

        if (
            len(set(normalized_paths))
            != len(normalized_paths)
        ):
            raise ValueError(
                "Corpus relative paths must be unique."
            )

        object.__setattr__(
            self,
            "entries",
            entries,
        )

    @property
    def entry_count(self) -> int:
        return len(
            self.entries
        )

    @property
    def approved_entries(
        self,
    ) -> tuple[CorpusEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.approved_for_ingestion
        )

    @property
    def approved_count(self) -> int:
        return len(
            self.approved_entries
        )

    def get(
        self,
        document_id: str,
    ) -> CorpusEntry:
        document_id = _require_text(
            document_id,
            "document_id",
        )

        for entry in self.entries:
            if (
                entry.document_id
                == document_id
            ):
                return entry

        raise KeyError(
            document_id
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "manifest_id": (
                self.manifest_id
            ),
            "name": self.name,
            "entry_count": (
                self.entry_count
            ),
            "approved_count": (
                self.approved_count
            ),
            "entries": [
                entry.to_dict()
                for entry in self.entries
            ],
        }

    def to_json(
        self,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def build_ingestion_request(
    entry: CorpusEntry,
    corpus_root: str | Path,
) -> IngestionRequest:
    """Build a safe ingestion request for an approved corpus entry."""

    if not isinstance(
        entry,
        CorpusEntry,
    ):
        raise TypeError(
            "entry must be a CorpusEntry."
        )

    if not entry.approved_for_ingestion:
        raise PermissionError(
            "Corpus entry is not approved "
            "for ingestion."
        )

    root = Path(
        corpus_root
    ).resolve()

    relative = Path(
        *PurePosixPath(
            entry.relative_path
        ).parts
    )

    file_path = (
        root / relative
    ).resolve()

    try:
        file_path.relative_to(
            root
        )
    except ValueError as exc:
        raise ValueError(
            "Corpus path escapes the corpus root."
        ) from exc

    return IngestionRequest(
        source=entry.source,
        file_path=file_path,
        document_format=(
            entry.document_format
        ),
    )
