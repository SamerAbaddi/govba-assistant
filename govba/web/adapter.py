"""Controlled government-domain web adapter for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import (
    Iterable,
    Protocol,
    runtime_checkable,
)
from urllib.parse import (
    urlsplit,
    urlunsplit,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
    assess_source_url,
)
from govba.rag.language import (
    BilingualLanguage,
)
from govba.web.contract import (
    OfficialWebRequest,
    OfficialWebResult,
    OfficialWebRetriever,
)


GOVERNMENT_WEB_ADAPTER_VERSION = (
    "govba-government-web-adapter-v1"
)

DEFAULT_WEB_CANDIDATE_POOL = 20


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


def _canonical_url(
    value: str,
) -> str:
    """Return deterministic URL identity for deduplication."""

    value = _required_text(
        value,
        "url",
    )

    try:
        parsed = urlsplit(
            value
        )
    except ValueError:
        return value

    scheme = (
        parsed.scheme.lower()
    )

    hostname = (
        parsed.hostname
        or ""
    ).rstrip(".").lower()

    if not hostname:
        return value

    try:
        hostname = (
            hostname
            .encode("idna")
            .decode("ascii")
            .lower()
        )
    except UnicodeError:
        return value

    try:
        port = parsed.port
    except ValueError:
        return value

    if (
        port is None
        or (
            scheme == "https"
            and port == 443
        )
        or (
            scheme == "http"
            and port == 80
        )
    ):
        netloc = hostname
    else:
        netloc = (
            f"{hostname}:{port}"
        )

    path = (
        parsed.path
        or "/"
    )

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        )
    )


@dataclass(frozen=True)
class RawWebSearchResult:
    """Provider-neutral raw web-search result."""

    title: str
    url: str
    text: str

    score: float

    provider_rank: int

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

        if (
            isinstance(
                self.provider_rank,
                bool,
            )
            or not isinstance(
                self.provider_rank,
                int,
            )
            or self.provider_rank < 1
        ):
            raise ValueError(
                "provider_rank must be "
                "at least 1."
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
            "score",
            float(
                self.score
            ),
        )


@runtime_checkable
class WebSearchBackend(
    Protocol
):
    """Low-level search backend used by the controlled adapter."""

    def search(
        self,
        *,
        query_text: str,
        language: BilingualLanguage,
        limit: int,
        allowed_domains: tuple[
            str,
            ...
        ],
    ) -> Iterable[
        RawWebSearchResult
    ]:
        """Return raw provider results."""
        ...


class GovernmentWebAdapter:
    """Fail-closed official government web-search adapter."""

    def __init__(
        self,
        *,
        backend: WebSearchBackend,
        policy: SourceAllowlistPolicy,
        candidate_pool_size: int = (
            DEFAULT_WEB_CANDIDATE_POOL
        ),
        retrieval_method: str = (
            GOVERNMENT_WEB_ADAPTER_VERSION
        ),
    ) -> None:
        if not isinstance(
            backend,
            WebSearchBackend,
        ):
            raise TypeError(
                "backend must implement "
                "WebSearchBackend."
            )

        if not isinstance(
            policy,
            SourceAllowlistPolicy,
        ):
            raise TypeError(
                "policy must be a "
                "SourceAllowlistPolicy."
            )

        if (
            isinstance(
                candidate_pool_size,
                bool,
            )
            or not isinstance(
                candidate_pool_size,
                int,
            )
            or not (
                1
                <= candidate_pool_size
                <= 100
            )
        ):
            raise ValueError(
                "candidate_pool_size must be "
                "between 1 and 100."
            )

        retrieval_method = (
            _required_text(
                retrieval_method,
                "retrieval_method",
            )
        )

        self._backend = backend
        self._policy = policy
        self._candidate_pool_size = (
            candidate_pool_size
        )
        self._retrieval_method = (
            retrieval_method
        )

    @property
    def policy(
        self,
    ) -> SourceAllowlistPolicy:
        return self._policy

    @property
    def candidate_pool_size(
        self,
    ) -> int:
        return self._candidate_pool_size

    def search(
        self,
        request: OfficialWebRequest,
    ) -> list[
        OfficialWebResult
    ]:
        """Return trusted, deduplicated, deterministically ranked results."""

        if not isinstance(
            request,
            OfficialWebRequest,
        ):
            raise TypeError(
                "request must be an "
                "OfficialWebRequest."
            )

        candidate_limit = min(
            100,
            max(
                request.top_k,
                self._candidate_pool_size,
            ),
        )

        raw_values = tuple(
            self._backend.search(
                query_text=(
                    request.query_text
                ),
                language=(
                    request.language
                ),
                limit=(
                    candidate_limit
                ),
                allowed_domains=(
                    self._policy
                    .allowed_domains
                ),
            )
        )

        for value in raw_values:
            if not isinstance(
                value,
                RawWebSearchResult,
            ):
                raise TypeError(
                    "Backend must return only "
                    "RawWebSearchResult values."
                )

        ordered = sorted(
            raw_values,
            key=lambda value: (
                value.provider_rank,
                -value.score,
                _canonical_url(
                    value.url
                ),
                value.title.casefold(),
            ),
        )

        trusted = []
        seen_urls = set()

        for value in ordered:
            canonical_url = (
                _canonical_url(
                    value.url
                )
            )

            if canonical_url in seen_urls:
                continue

            source_assessment = (
                assess_source_url(
                    value.url,
                    document_id=(
                        f"WEB-{request.request_id[:16]}"
                    ),
                    trace_id=(
                        request.request_id
                    ),
                    policy=(
                        self._policy
                    ),
                )
            )

            if not source_assessment.allowed:
                continue

            seen_urls.add(
                canonical_url
            )

            trusted.append(
                value
            )

            if (
                len(trusted)
                >= request.top_k
            ):
                break

        return [
            OfficialWebResult(
                title=value.title,
                url=value.url,
                text=value.text,
                rank=rank,
                score=value.score,
                retrieval_method=(
                    self._retrieval_method
                ),
            )
            for rank, value
            in enumerate(
                trusted,
                start=1,
            )
        ]


def is_official_web_retriever(
    value: object,
) -> bool:
    """Return whether value satisfies the public retrieval protocol."""

    return isinstance(
        value,
        OfficialWebRetriever,
    )
