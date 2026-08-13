"""Controlled authoritative-corpus validation for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from govba.rag.corpus import (
    CorpusEntry,
    CorpusManifest,
    build_ingestion_request,
)


CORPUS_VALIDATION_VERSION = (
    "govba-corpus-validation-v1"
)


def compute_file_sha256(
    file_path: str | Path,
) -> str:
    """Compute SHA-256 for a corpus file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            "Corpus path must identify a file."
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusEntryValidation:
    """Validation result for one registered corpus entry."""

    document_id: str
    relative_path: str
    approved: bool
    valid: bool
    file_exists: bool
    hash_required: bool
    hash_matches: bool | None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "errors",
            tuple(self.errors),
        )


@dataclass(frozen=True)
class CorpusValidationReport:
    """Aggregate validation result for a corpus manifest."""

    manifest_id: str
    entries: tuple[CorpusEntryValidation, ...]
    validation_version: str = (
        CORPUS_VALIDATION_VERSION
    )

    def __post_init__(self) -> None:
        entries = tuple(
            self.entries
        )

        for entry in entries:
            if not isinstance(
                entry,
                CorpusEntryValidation,
            ):
                raise TypeError(
                    "entries must contain "
                    "CorpusEntryValidation values."
                )

        object.__setattr__(
            self,
            "entries",
            entries,
        )

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def approved_count(self) -> int:
        return sum(
            entry.approved
            for entry in self.entries
        )

    @property
    def valid_count(self) -> int:
        return sum(
            entry.valid
            for entry in self.entries
        )

    @property
    def invalid_count(self) -> int:
        return sum(
            not entry.valid
            for entry in self.entries
        )

    @property
    def is_valid(self) -> bool:
        return self.invalid_count == 0


def _validate_entry(
    entry: CorpusEntry,
    corpus_root: Path,
) -> CorpusEntryValidation:
    errors = []

    if not entry.approved_for_ingestion:
        return CorpusEntryValidation(
            document_id=entry.document_id,
            relative_path=entry.relative_path,
            approved=False,
            valid=True,
            file_exists=False,
            hash_required=False,
            hash_matches=None,
            errors=(),
        )

    try:
        request = build_ingestion_request(
            entry,
            corpus_root,
        )
    except Exception:
        return CorpusEntryValidation(
            document_id=entry.document_id,
            relative_path=entry.relative_path,
            approved=True,
            valid=False,
            file_exists=False,
            hash_required=True,
            hash_matches=None,
            errors=(
                "Unable to build safe ingestion request.",
            ),
        )

    path = request.file_path

    file_exists = (
        path.exists()
        and path.is_file()
    )

    if not file_exists:
        errors.append(
            "Approved corpus file is missing."
        )

    expected_hash = (
        entry.source.content_hash
    )

    hash_required = True
    hash_matches = None

    if not expected_hash:
        errors.append(
            "Approved corpus entry has no content hash."
        )

    elif file_exists:
        actual_hash = compute_file_sha256(
            path
        )

        hash_matches = (
            actual_hash
            == expected_hash
        )

        if not hash_matches:
            errors.append(
                "Corpus file content hash does not match manifest."
            )

    return CorpusEntryValidation(
        document_id=entry.document_id,
        relative_path=entry.relative_path,
        approved=True,
        valid=not errors,
        file_exists=file_exists,
        hash_required=hash_required,
        hash_matches=hash_matches,
        errors=tuple(errors),
    )


def validate_corpus(
    manifest: CorpusManifest,
    corpus_root: str | Path,
) -> CorpusValidationReport:
    """Validate registered corpus files before ingestion."""

    if not isinstance(
        manifest,
        CorpusManifest,
    ):
        raise TypeError(
            "manifest must be a CorpusManifest."
        )

    root = Path(
        corpus_root
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Corpus root not found: {root}"
        )

    if not root.is_dir():
        raise ValueError(
            "corpus_root must identify a directory."
        )

    results = tuple(
        _validate_entry(
            entry,
            root,
        )
        for entry in manifest.entries
    )

    return CorpusValidationReport(
        manifest_id=manifest.manifest_id,
        entries=results,
    )
