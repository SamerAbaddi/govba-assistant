"""Bridge trusted official-web content into GovBA authoritative evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import (
    urlsplit,
    urlunsplit,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
    SourceAllowlistResult,
    assess_source_url,
)
from govba.rag.chunking import (
    ChunkingConfig,
    split_ingested_text,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.web.contract import (
    OfficialWebResult,
)


WEB_EVIDENCE_BRIDGE_VERSION = (
    "govba-web-evidence-bridge-v1"
)


def _required_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} must not be blank."
        )

    return value


def _optional_text(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field_name} must be a string."
        )

    return value.strip()


def _validate_date(
    value: date | None,
    field_name: str,
) -> None:
    if value is None:
        return

    if (
        isinstance(
            value,
            datetime,
        )
        or not isinstance(
            value,
            date,
        )
    ):
        raise TypeError(
            f"{field_name} must be "
            "a date or None."
        )


def _canonical_url_identity(
    url: str,
) -> str:
    parsed = urlsplit(
        url
    )

    scheme = (
        parsed.scheme
        .lower()
    )

    hostname = (
        parsed.hostname
        or ""
    ).rstrip(
        "."
    ).lower()

    try:
        hostname = (
            hostname
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError:
        pass

    try:
        port = parsed.port
    except ValueError:
        port = None

    if (
        port is None
        or (
            scheme == "https"
            and port == 443
        )
    ):
        netloc = hostname
    else:
        netloc = (
            f"{hostname}:{port}"
        )

    return urlunsplit(
        (
            scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def build_web_document_id(
    result: OfficialWebResult,
) -> str:
    """Generate stable identity for one web-document content version."""

    if not isinstance(
        result,
        OfficialWebResult,
    ):
        raise TypeError(
            "result must be an "
            "OfficialWebResult."
        )

    payload = "\x1f".join(
        (
            WEB_EVIDENCE_BRIDGE_VERSION,
            _canonical_url_identity(
                result.url
            ),
            result.content_hash,
        )
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"WEB-{digest}"
    )


@dataclass(frozen=True)
class WebEvidenceMetadata:
    """Governance metadata supplied for one trusted web document."""

    issuing_authority: str

    language: SourceLanguage

    document_type: DocumentType = (
        DocumentType.OTHER
    )

    jurisdiction: str = "Jordan"

    publication_date: date | None = None

    effective_from: date | None = None

    effective_until: date | None = None

    version: str = ""

    status: SourceStatus = (
        SourceStatus.UNKNOWN
    )

    supersedes: tuple[
        str,
        ...
    ] = ()

    superseded_by: tuple[
        str,
        ...
    ] = ()

    document_id: str = ""

    def __post_init__(self) -> None:
        authority = _required_text(
            self.issuing_authority,
            "issuing_authority",
        )

        jurisdiction = _required_text(
            self.jurisdiction,
            "jurisdiction",
        )

        version = _optional_text(
            self.version,
            "version",
        )

        document_id = _optional_text(
            self.document_id,
            "document_id",
        )

        if not isinstance(
            self.language,
            SourceLanguage,
        ):
            raise TypeError(
                "language must be a "
                "SourceLanguage."
            )

        if not isinstance(
            self.document_type,
            DocumentType,
        ):
            raise TypeError(
                "document_type must be "
                "a DocumentType."
            )

        if not isinstance(
            self.status,
            SourceStatus,
        ):
            raise TypeError(
                "status must be a "
                "SourceStatus."
            )

        for field_name in (
            "publication_date",
            "effective_from",
            "effective_until",
        ):
            _validate_date(
                getattr(
                    self,
                    field_name,
                ),
                field_name,
            )

        if (
            self.effective_from
            is not None
            and self.effective_until
            is not None
            and self.effective_until
            < self.effective_from
        ):
            raise ValueError(
                "effective_until cannot be "
                "earlier than effective_from."
            )

        try:
            supersedes = tuple(
                self.supersedes
            )

            superseded_by = tuple(
                self.superseded_by
            )
        except TypeError as exc:
            raise TypeError(
                "supersession fields must "
                "be iterable."
            ) from exc

        for values, field_name in (
            (
                supersedes,
                "supersedes",
            ),
            (
                superseded_by,
                "superseded_by",
            ),
        ):
            for value in values:
                _required_text(
                    value,
                    field_name,
                )

        object.__setattr__(
            self,
            "issuing_authority",
            authority,
        )

        object.__setattr__(
            self,
            "jurisdiction",
            jurisdiction,
        )

        object.__setattr__(
            self,
            "version",
            version,
        )

        object.__setattr__(
            self,
            "document_id",
            document_id,
        )

        object.__setattr__(
            self,
            "supersedes",
            supersedes,
        )

        object.__setattr__(
            self,
            "superseded_by",
            superseded_by,
        )


@dataclass(frozen=True)
class WebEvidenceBridgeResult:
    """Authoritative source and chunks produced from trusted web evidence."""

    trace_id: str

    source: AuthoritativeSource

    chunks: tuple[
        EvidenceChunk,
        ...
    ]

    source_assessment: (
        SourceAllowlistResult
    )

    version: str = (
        WEB_EVIDENCE_BRIDGE_VERSION
    )

    bridge_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "source must be an "
                "AuthoritativeSource."
            )

        try:
            chunks = tuple(
                self.chunks
            )
        except TypeError as exc:
            raise TypeError(
                "chunks must be iterable."
            ) from exc

        if not chunks:
            raise ValueError(
                "At least one evidence chunk "
                "is required."
            )

        for chunk in chunks:
            if not isinstance(
                chunk,
                EvidenceChunk,
            ):
                raise TypeError(
                    "chunks must contain only "
                    "EvidenceChunk values."
                )

            if (
                chunk.document_id
                != self.source.document_id
            ):
                raise ValueError(
                    "Chunk document ID does not "
                    "match source document ID."
                )

        indexes = tuple(
            chunk.chunk_index
            for chunk in chunks
        )

        if indexes != tuple(
            range(
                len(chunks)
            )
        ):
            raise ValueError(
                "Chunk indexes must be "
                "contiguous from zero."
            )

        if not isinstance(
            self.source_assessment,
            SourceAllowlistResult,
        ):
            raise TypeError(
                "source_assessment must be a "
                "SourceAllowlistResult."
            )

        if not self.source_assessment.allowed:
            raise ValueError(
                "Web evidence source must pass "
                "the source allowlist."
            )

        if (
            self.source_assessment.trace_id
            != trace_id
        ):
            raise ValueError(
                "Source assessment trace ID "
                "must match bridge trace ID."
            )

        if (
            self.source_assessment.document_id
            != self.source.document_id
        ):
            raise ValueError(
                "Source assessment document ID "
                "must match authoritative source."
            )

        identity_payload = "\x1f".join(
            (
                self.version,
                self.source.document_id,
                self.source.content_hash,
                self.source.issuing_authority,
                self.source.document_type.value,
                self.source.language.value,
                ",".join(
                    chunk.chunk_id
                    for chunk in chunks
                ),
            )
        )

        bridge_id = hashlib.sha256(
            identity_payload.encode(
                "utf-8"
            )
        ).hexdigest()

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "chunks",
            chunks,
        )

        object.__setattr__(
            self,
            "bridge_id",
            bridge_id,
        )

    @property
    def chunk_count(
        self,
    ) -> int:
        return len(
            self.chunks
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize provenance without evidence text."""

        return {
            "version": (
                self.version
            ),
            "bridge_id": (
                self.bridge_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "source": (
                self.source.to_dict()
            ),
            "chunk_count": (
                self.chunk_count
            ),
            "chunks": [
                {
                    "chunk_id": (
                        chunk.chunk_id
                    ),
                    "content_hash": (
                        chunk.content_hash
                    ),
                    "chunk_index": (
                        chunk.chunk_index
                    ),
                    "language": (
                        chunk.language.value
                    ),
                    "character_count": (
                        chunk.character_count
                    ),
                    "word_count": (
                        chunk.word_count
                    ),
                }
                for chunk
                in self.chunks
            ],
            "source_assessment": (
                self.source_assessment
                .to_dict()
            ),
        }


def bridge_official_web_result(
    result: OfficialWebResult,
    *,
    metadata: WebEvidenceMetadata,
    policy: SourceAllowlistPolicy,
    trace_id: str,
    chunking_config: (
        ChunkingConfig | None
    ) = None,
) -> WebEvidenceBridgeResult:
    """Convert trusted controlled-web content into authoritative evidence."""

    if not isinstance(
        result,
        OfficialWebResult,
    ):
        raise TypeError(
            "result must be an "
            "OfficialWebResult."
        )

    if not isinstance(
        metadata,
        WebEvidenceMetadata,
    ):
        raise TypeError(
            "metadata must be "
            "WebEvidenceMetadata."
        )

    if not isinstance(
        policy,
        SourceAllowlistPolicy,
    ):
        raise TypeError(
            "policy must be a "
            "SourceAllowlistPolicy."
        )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if (
        chunking_config is not None
        and not isinstance(
            chunking_config,
            ChunkingConfig,
        )
    ):
        raise TypeError(
            "chunking_config must be "
            "a ChunkingConfig or None."
        )

    document_id = (
        metadata.document_id
        or build_web_document_id(
            result
        )
    )

    source_assessment = (
        assess_source_url(
            result.url,
            document_id=document_id,
            trace_id=trace,
            policy=policy,
        )
    )

    if not source_assessment.allowed:
        raise ValueError(
            "Official web result failed "
            "source allowlist validation."
        )

    pieces = split_ingested_text(
        result.text,
        chunking_config,
    )

    source = AuthoritativeSource(
        document_id=document_id,
        title=result.title,
        issuing_authority=(
            metadata.issuing_authority
        ),
        document_type=(
            metadata.document_type
        ),
        language=(
            metadata.language
        ),
        jurisdiction=(
            metadata.jurisdiction
        ),
        publication_date=(
            metadata.publication_date
        ),
        effective_from=(
            metadata.effective_from
        ),
        effective_until=(
            metadata.effective_until
        ),
        version=(
            metadata.version
        ),
        status=(
            metadata.status
        ),
        supersedes=(
            metadata.supersedes
        ),
        superseded_by=(
            metadata.superseded_by
        ),
        official_source_url=(
            result.url
        ),
        content_hash=(
            result.content_hash
        ),
        ingested_at=(
            result.retrieved_at
        ),
    )

    chunks = tuple(
        EvidenceChunk(
            document_id=document_id,
            chunk_index=index,
            text=piece,
            language=(
                metadata.language
            ),
        )
        for index, piece
        in enumerate(
            pieces
        )
    )

    return WebEvidenceBridgeResult(
        trace_id=trace,
        source=source,
        chunks=chunks,
        source_assessment=(
            source_assessment
        ),
    )
