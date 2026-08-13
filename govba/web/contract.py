"""Provider-independent controlled web retrieval contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from govba.rag.language import (
    BilingualLanguage,
)


OFFICIAL_WEB_RETRIEVAL_VERSION = (
    "govba-official-web-retrieval-v1"
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


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _normalize_domain_from_url(
    url: str,
) -> str:
    parsed = urlsplit(
        url
    )

    if (
        parsed.scheme.lower()
        != "https"
    ):
        raise ValueError(
            "Official web result URL "
            "must use HTTPS."
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "Official web result URL cannot "
            "contain embedded credentials."
        )

    hostname = parsed.hostname

    if not hostname:
        raise ValueError(
            "Official web result URL "
            "must contain a hostname."
        )

    try:
        domain = (
            hostname
            .rstrip(".")
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError as exc:
        raise ValueError(
            "Official web result hostname "
            "is invalid."
        ) from exc

    return domain


@dataclass(frozen=True)
class OfficialWebRequest:
    """One controlled official-web retrieval request."""

    query_text: str

    language: BilingualLanguage

    top_k: int = 5

    version: str = (
        OFFICIAL_WEB_RETRIEVAL_VERSION
    )

    query_sha256: str = field(
        init=False
    )

    request_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        query_text = _required_text(
            self.query_text,
            "query_text",
        )

        if not isinstance(
            self.language,
            BilingualLanguage,
        ):
            raise TypeError(
                "language must be a "
                "BilingualLanguage."
            )

        if (
            isinstance(
                self.top_k,
                bool,
            )
            or not isinstance(
                self.top_k,
                int,
            )
            or not (
                1 <= self.top_k <= 20
            )
        ):
            raise ValueError(
                "top_k must be between "
                "1 and 20."
            )

        query_hash = _sha256_text(
            query_text
        )

        request_payload = "\x1f".join(
            (
                self.version,
                query_hash,
                self.language.value,
                str(
                    self.top_k
                ),
            )
        )

        request_id = _sha256_text(
            request_payload
        )

        object.__setattr__(
            self,
            "query_text",
            query_text,
        )

        object.__setattr__(
            self,
            "query_sha256",
            query_hash,
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

    def to_dict(
        self,
        *,
        include_query_text: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(
            include_query_text,
            bool,
        ):
            raise TypeError(
                "include_query_text "
                "must be boolean."
            )

        data: dict[
            str,
            Any,
        ] = {
            "version": (
                self.version
            ),
            "request_id": (
                self.request_id
            ),
            "query_sha256": (
                self.query_sha256
            ),
            "language": (
                self.language.value
            ),
            "top_k": (
                self.top_k
            ),
        }

        if include_query_text:
            data[
                "query_text"
            ] = self.query_text

        return data


@dataclass(frozen=True)
class OfficialWebResult:
    """One controlled web result with provenance."""

    title: str

    url: str

    text: str

    rank: int

    score: float

    retrieval_method: str

    retrieved_at: datetime = field(
        default_factory=_utc_now
    )

    version: str = (
        OFFICIAL_WEB_RETRIEVAL_VERSION
    )

    domain: str = field(
        init=False
    )

    content_hash: str = field(
        init=False
    )

    result_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        title = _required_text(
            self.title,
            "title",
        )

        url = _required_text(
            self.url,
            "url",
        )

        text = _required_text(
            self.text,
            "text",
        )

        retrieval_method = (
            _required_text(
                self.retrieval_method,
                "retrieval_method",
            )
        )

        if (
            isinstance(
                self.rank,
                bool,
            )
            or not isinstance(
                self.rank,
                int,
            )
            or self.rank < 1
        ):
            raise ValueError(
                "rank must be at least 1."
            )

        if (
            isinstance(
                self.score,
                bool,
            )
            or not isinstance(
                self.score,
                (int, float),
            )
            or not isfinite(
                self.score
            )
            or not (
                0.0 <= self.score <= 1.0
            )
        ):
            raise ValueError(
                "score must be between "
                "0 and 1."
            )

        if not isinstance(
            self.retrieved_at,
            datetime,
        ):
            raise TypeError(
                "retrieved_at must be "
                "a datetime."
            )

        if (
            self.retrieved_at.tzinfo
            is None
            or self.retrieved_at.utcoffset()
            is None
        ):
            raise ValueError(
                "retrieved_at must be "
                "timezone-aware."
            )

        domain = (
            _normalize_domain_from_url(
                url
            )
        )

        content_hash = _sha256_text(
            text
        )

        identity_payload = "\x1f".join(
            (
                self.version,
                url,
                content_hash,
                str(
                    self.rank
                ),
                retrieval_method,
            )
        )

        result_id = _sha256_text(
            identity_payload
        )

        object.__setattr__(
            self,
            "title",
            title,
        )

        object.__setattr__(
            self,
            "url",
            url,
        )

        object.__setattr__(
            self,
            "text",
            text,
        )

        object.__setattr__(
            self,
            "retrieval_method",
            retrieval_method,
        )

        object.__setattr__(
            self,
            "score",
            float(
                self.score
            ),
        )

        object.__setattr__(
            self,
            "domain",
            domain,
        )

        object.__setattr__(
            self,
            "content_hash",
            content_hash,
        )

        object.__setattr__(
            self,
            "result_id",
            result_id,
        )

    def to_dict(
        self,
        *,
        include_text: bool = False,
        include_url: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(
            include_text,
            bool,
        ):
            raise TypeError(
                "include_text must be boolean."
            )

        if not isinstance(
            include_url,
            bool,
        ):
            raise TypeError(
                "include_url must be boolean."
            )

        data: dict[
            str,
            Any,
        ] = {
            "version": (
                self.version
            ),
            "result_id": (
                self.result_id
            ),
            "title": (
                self.title
            ),
            "domain": (
                self.domain
            ),
            "content_hash": (
                self.content_hash
            ),
            "rank": (
                self.rank
            ),
            "score": (
                self.score
            ),
            "retrieval_method": (
                self.retrieval_method
            ),
            "retrieved_at": (
                self.retrieved_at
                .isoformat()
            ),
        }

        if include_url:
            data[
                "url"
            ] = self.url

        if include_text:
            data[
                "text"
            ] = self.text

        return data


@runtime_checkable
class OfficialWebRetriever(
    Protocol
):
    """Contract implemented by controlled official-web adapters."""

    def search(
        self,
        request: OfficialWebRequest,
    ) -> list[
        OfficialWebResult
    ]:
        """Return ranked official-web results."""
        ...
