"""Persistent and integrity-checked corpus manifests for GovBA-GAR."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from govba.rag.corpus import (
    CORPUS_MANIFEST_SCHEMA_VERSION,
    CorpusEntry,
    CorpusManifest,
)
from govba.rag.ingestion import DocumentFormat
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


CORPUS_STORE_VERSION = "govba-corpus-store-v1"


class CorpusManifestError(RuntimeError):
    """Safe failure raised for invalid persisted manifests."""


def _date_to_text(
    value: date | None,
) -> str | None:
    if value is None:
        return None

    return value.isoformat()


def _datetime_to_text(
    value: datetime | None,
) -> str | None:
    if value is None:
        return None

    return value.isoformat()


def _parse_date(
    value,
    field_name: str,
) -> date | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise CorpusManifestError(
            f"{field_name} must be an ISO date or null."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CorpusManifestError(
            f"{field_name} contains an invalid ISO date."
        ) from exc


def _parse_datetime(
    value,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise CorpusManifestError(
            f"{field_name} must be an ISO datetime or null."
        )

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CorpusManifestError(
            f"{field_name} contains an invalid ISO datetime."
        ) from exc

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise CorpusManifestError(
            f"{field_name} must include timezone information."
        )

    return parsed


def _source_to_dict(
    source: AuthoritativeSource,
) -> dict[str, object]:
    return {
        "document_id": source.document_id,
        "title": source.title,
        "issuing_authority": source.issuing_authority,
        "document_type": source.document_type.value,
        "language": source.language.value,
        "jurisdiction": source.jurisdiction,
        "publication_date": _date_to_text(
            source.publication_date
        ),
        "effective_from": _date_to_text(
            source.effective_from
        ),
        "effective_until": _date_to_text(
            source.effective_until
        ),
        "version": source.version,
        "status": source.status.value,
        "supersedes": list(
            source.supersedes
        ),
        "superseded_by": list(
            source.superseded_by
        ),
        "official_source_url": (
            source.official_source_url
        ),
        "content_hash": source.content_hash,
        "ingested_at": _datetime_to_text(
            source.ingested_at
        ),
        "schema_version": (
            source.schema_version
        ),
    }


def _source_from_dict(
    data: dict[str, object],
) -> AuthoritativeSource:
    if not isinstance(data, dict):
        raise CorpusManifestError(
            "source must be an object."
        )

    try:
        return AuthoritativeSource(
            document_id=str(
                data["document_id"]
            ),
            title=str(
                data["title"]
            ),
            issuing_authority=str(
                data["issuing_authority"]
            ),
            document_type=DocumentType(
                data["document_type"]
            ),
            language=SourceLanguage(
                data["language"]
            ),
            jurisdiction=str(
                data["jurisdiction"]
            ),
            publication_date=_parse_date(
                data.get(
                    "publication_date"
                ),
                "publication_date",
            ),
            effective_from=_parse_date(
                data.get(
                    "effective_from"
                ),
                "effective_from",
            ),
            effective_until=_parse_date(
                data.get(
                    "effective_until"
                ),
                "effective_until",
            ),
            version=(
                data.get(
                    "version"
                )
            ),
            status=SourceStatus(
                data["status"]
            ),
            supersedes=tuple(
                data.get(
                    "supersedes",
                    (),
                )
            ),
            superseded_by=tuple(
                data.get(
                    "superseded_by",
                    (),
                )
            ),
            official_source_url=(
                data.get(
                    "official_source_url"
                )
            ),
            content_hash=(
                data.get(
                    "content_hash"
                )
            ),
            ingested_at=_parse_datetime(
                data.get(
                    "ingested_at"
                ),
                "ingested_at",
            ),
            schema_version=str(
                data.get(
                    "schema_version",
                    "govba-source-v1",
                )
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CorpusManifestError(
            "Invalid authoritative source metadata."
        ) from exc


def manifest_payload(
    manifest: CorpusManifest,
) -> dict[str, object]:
    """Return complete serializable manifest data."""

    if not isinstance(
        manifest,
        CorpusManifest,
    ):
        raise TypeError(
            "manifest must be a CorpusManifest."
        )

    return {
        "schema_version": (
            manifest.schema_version
        ),
        "manifest_id": (
            manifest.manifest_id
        ),
        "name": manifest.name,
        "entries": [
            {
                "source": _source_to_dict(
                    entry.source
                ),
                "document_format": (
                    entry.document_format.value
                ),
                "relative_path": (
                    entry.relative_path
                ),
                "approved_for_ingestion": (
                    entry.approved_for_ingestion
                ),
            }
            for entry in manifest.entries
        ],
    }


def canonical_manifest_json(
    manifest: CorpusManifest,
) -> str:
    """Serialize a manifest deterministically."""

    return json.dumps(
        manifest_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def compute_manifest_hash(
    manifest: CorpusManifest,
) -> str:
    """Compute deterministic SHA-256 manifest integrity hash."""

    payload = canonical_manifest_json(
        manifest
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def manifest_envelope(
    manifest: CorpusManifest,
) -> dict[str, object]:
    return {
        "store_version": (
            CORPUS_STORE_VERSION
        ),
        "integrity_algorithm": (
            "sha256"
        ),
        "integrity_hash": (
            compute_manifest_hash(
                manifest
            )
        ),
        "manifest": (
            manifest_payload(
                manifest
            )
        ),
    }


def write_manifest(
    manifest: CorpusManifest,
    file_path: str | Path,
) -> Path:
    """Atomically persist an integrity-protected manifest."""

    if not isinstance(
        manifest,
        CorpusManifest,
    ):
        raise TypeError(
            "manifest must be a CorpusManifest."
        )

    path = Path(
        file_path
    )

    if (
        path.suffix.casefold()
        != ".json"
    ):
        raise ValueError(
            "Manifest file must use a .json extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        manifest_envelope(
            manifest
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    temp_path = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                text
            )
            handle.write(
                "\n"
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )

            temp_path = Path(
                handle.name
            )

        temp_path.replace(
            path
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()

    return path


def _manifest_from_payload(
    payload: dict[str, object],
) -> CorpusManifest:
    if not isinstance(
        payload,
        dict,
    ):
        raise CorpusManifestError(
            "Manifest payload must be an object."
        )

    if (
        payload.get(
            "schema_version"
        )
        != CORPUS_MANIFEST_SCHEMA_VERSION
    ):
        raise CorpusManifestError(
            "Unsupported corpus manifest schema."
        )

    raw_entries = payload.get(
        "entries"
    )

    if not isinstance(
        raw_entries,
        list,
    ):
        raise CorpusManifestError(
            "Manifest entries must be a list."
        )

    entries = []

    for raw_entry in raw_entries:
        if not isinstance(
            raw_entry,
            dict,
        ):
            raise CorpusManifestError(
                "Corpus entry must be an object."
            )

        try:
            entry = CorpusEntry(
                source=_source_from_dict(
                    raw_entry[
                        "source"
                    ]
                ),
                document_format=DocumentFormat(
                    raw_entry[
                        "document_format"
                    ]
                ),
                relative_path=str(
                    raw_entry[
                        "relative_path"
                    ]
                ),
                approved_for_ingestion=(
                    raw_entry[
                        "approved_for_ingestion"
                    ]
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise CorpusManifestError(
                "Invalid corpus entry."
            ) from exc

        entries.append(
            entry
        )

    try:
        return CorpusManifest(
            manifest_id=str(
                payload[
                    "manifest_id"
                ]
            ),
            name=str(
                payload[
                    "name"
                ]
            ),
            entries=tuple(
                entries
            ),
            schema_version=str(
                payload[
                    "schema_version"
                ]
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CorpusManifestError(
            "Invalid corpus manifest."
        ) from exc


def read_manifest(
    file_path: str | Path,
) -> CorpusManifest:
    """Read and verify a persisted corpus manifest."""

    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            "Manifest path must identify a file."
        )

    try:
        envelope = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise CorpusManifestError(
            "Unable to read corpus manifest."
        ) from exc

    if not isinstance(
        envelope,
        dict,
    ):
        raise CorpusManifestError(
            "Manifest envelope must be an object."
        )

    if (
        envelope.get(
            "store_version"
        )
        != CORPUS_STORE_VERSION
    ):
        raise CorpusManifestError(
            "Unsupported corpus store version."
        )

    if (
        envelope.get(
            "integrity_algorithm"
        )
        != "sha256"
    ):
        raise CorpusManifestError(
            "Unsupported manifest integrity algorithm."
        )

    payload = envelope.get(
        "manifest"
    )

    manifest = _manifest_from_payload(
        payload
    )

    expected_hash = compute_manifest_hash(
        manifest
    )

    supplied_hash = envelope.get(
        "integrity_hash"
    )

    if (
        not isinstance(
            supplied_hash,
            str,
        )
        or supplied_hash
        != expected_hash
    ):
        raise CorpusManifestError(
            "Corpus manifest integrity check failed."
        )

    return manifest
