"""Bilingual language and retrieval-routing contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


BILINGUAL_LANGUAGE_VERSION = (
    "govba-bilingual-language-v1"
)


class BilingualLanguage(
    str,
    Enum,
):
    """Languages explicitly supported by the bilingual GovBA layer."""

    ARABIC = "ar"
    ENGLISH = "en"


class DetectedLanguage(
    str,
    Enum,
):
    """Language state assigned to incoming text."""

    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LanguageRoutingMode(
    str,
    Enum,
):
    """Retrieval-language routing strategy."""

    MONOLINGUAL = "monolingual"
    CROSS_LINGUAL = "cross_lingual"


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


def _query_id(
    *,
    query_sha256: str,
    detected_language: DetectedLanguage,
    response_language: BilingualLanguage | None,
    allow_cross_lingual: bool,
) -> str:
    payload = "\x1f".join(
        (
            BILINGUAL_LANGUAGE_VERSION,
            query_sha256,
            detected_language.value,
            (
                response_language.value
                if response_language
                else ""
            ),
            str(
                allow_cross_lingual
            ).lower(),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class BilingualQueryRequest:
    """Privacy-aware bilingual query contract."""

    query_text: str

    detected_language: DetectedLanguage = (
        DetectedLanguage.UNKNOWN
    )

    response_language: (
        BilingualLanguage | None
    ) = None

    allow_cross_lingual: bool = True

    version: str = (
        BILINGUAL_LANGUAGE_VERSION
    )

    query_sha256: str = field(
        init=False
    )

    query_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        query_text = _required_text(
            self.query_text,
            "query_text",
        )

        if not isinstance(
            self.detected_language,
            DetectedLanguage,
        ):
            raise TypeError(
                "detected_language must be a "
                "DetectedLanguage."
            )

        if (
            self.response_language is not None
            and not isinstance(
                self.response_language,
                BilingualLanguage,
            )
        ):
            raise TypeError(
                "response_language must be a "
                "BilingualLanguage or None."
            )

        if not isinstance(
            self.allow_cross_lingual,
            bool,
        ):
            raise TypeError(
                "allow_cross_lingual must be boolean."
            )

        query_hash = _sha256_text(
            query_text
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
            "query_id",
            _query_id(
                query_sha256=query_hash,
                detected_language=(
                    self.detected_language
                ),
                response_language=(
                    self.response_language
                ),
                allow_cross_lingual=(
                    self.allow_cross_lingual
                ),
            ),
        )

    @property
    def character_count(
        self,
    ) -> int:
        return len(
            self.query_text
        )

    @property
    def resolved_response_language(
        self,
    ) -> BilingualLanguage | None:
        if self.response_language is not None:
            return self.response_language

        if (
            self.detected_language
            is DetectedLanguage.ARABIC
        ):
            return BilingualLanguage.ARABIC

        if (
            self.detected_language
            is DetectedLanguage.ENGLISH
        ):
            return BilingualLanguage.ENGLISH

        return None

    @property
    def requires_response_language_resolution(
        self,
    ) -> bool:
        return (
            self.resolved_response_language
            is None
        )

    def to_dict(
        self,
        *,
        include_query_text: bool = False,
    ) -> dict[str, object]:
        """Serialize safely; raw query text is excluded by default."""

        if not isinstance(
            include_query_text,
            bool,
        ):
            raise TypeError(
                "include_query_text must be boolean."
            )

        resolved = (
            self.resolved_response_language
        )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "query_id": (
                self.query_id
            ),
            "query_sha256": (
                self.query_sha256
            ),
            "character_count": (
                self.character_count
            ),
            "detected_language": (
                self.detected_language.value
            ),
            "response_language": (
                self.response_language.value
                if self.response_language
                else None
            ),
            "resolved_response_language": (
                resolved.value
                if resolved
                else None
            ),
            "allow_cross_lingual": (
                self.allow_cross_lingual
            ),
            "requires_response_language_resolution": (
                self.requires_response_language_resolution
            ),
        }

        if include_query_text:
            data[
                "query_text"
            ] = self.query_text

        return data


def _route_id(
    *,
    query_id: str,
    response_language: BilingualLanguage,
    retrieval_languages: tuple[
        BilingualLanguage,
        ...
    ],
    mode: LanguageRoutingMode,
) -> str:
    payload = "\x1f".join(
        (
            BILINGUAL_LANGUAGE_VERSION,
            query_id,
            response_language.value,
            ",".join(
                language.value
                for language
                in retrieval_languages
            ),
            mode.value,
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class LanguageRoute:
    """Resolved bilingual routing instructions."""

    query_id: str

    query_language: DetectedLanguage

    response_language: BilingualLanguage

    retrieval_languages: tuple[
        BilingualLanguage,
        ...
    ]

    mode: LanguageRoutingMode

    version: str = (
        BILINGUAL_LANGUAGE_VERSION
    )

    route_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        query_id = _required_text(
            self.query_id,
            "query_id",
        )

        if not isinstance(
            self.query_language,
            DetectedLanguage,
        ):
            raise TypeError(
                "query_language must be a "
                "DetectedLanguage."
            )

        if not isinstance(
            self.response_language,
            BilingualLanguage,
        ):
            raise TypeError(
                "response_language must be a "
                "BilingualLanguage."
            )

        try:
            languages = tuple(
                self.retrieval_languages
            )
        except TypeError as exc:
            raise TypeError(
                "retrieval_languages must be iterable."
            ) from exc

        if not languages:
            raise ValueError(
                "retrieval_languages must not be empty."
            )

        for language in languages:
            if not isinstance(
                language,
                BilingualLanguage,
            ):
                raise TypeError(
                    "retrieval_languages must contain "
                    "only BilingualLanguage values."
                )

        if (
            len(set(languages))
            != len(languages)
        ):
            raise ValueError(
                "retrieval_languages must not "
                "contain duplicates."
            )

        if not isinstance(
            self.mode,
            LanguageRoutingMode,
        ):
            raise TypeError(
                "mode must be a "
                "LanguageRoutingMode."
            )

        if (
            self.mode
            is LanguageRoutingMode.MONOLINGUAL
            and len(languages) != 1
        ):
            raise ValueError(
                "MONOLINGUAL routing requires "
                "exactly one retrieval language."
            )

        if (
            self.mode
            is LanguageRoutingMode.CROSS_LINGUAL
            and set(languages)
            != {
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            }
        ):
            raise ValueError(
                "CROSS_LINGUAL routing requires "
                "Arabic and English."
            )

        object.__setattr__(
            self,
            "query_id",
            query_id,
        )

        object.__setattr__(
            self,
            "retrieval_languages",
            languages,
        )

        object.__setattr__(
            self,
            "route_id",
            _route_id(
                query_id=query_id,
                response_language=(
                    self.response_language
                ),
                retrieval_languages=(
                    languages
                ),
                mode=self.mode,
            ),
        )

    @property
    def cross_lingual(
        self,
    ) -> bool:
        return (
            self.mode
            is LanguageRoutingMode
            .CROSS_LINGUAL
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "route_id": (
                self.route_id
            ),
            "query_id": (
                self.query_id
            ),
            "query_language": (
                self.query_language.value
            ),
            "response_language": (
                self.response_language.value
            ),
            "retrieval_languages": [
                language.value
                for language
                in self.retrieval_languages
            ],
            "mode": (
                self.mode.value
            ),
            "cross_lingual": (
                self.cross_lingual
            ),
        }


def build_language_route(
    request: BilingualQueryRequest,
) -> LanguageRoute:
    """Resolve deterministic retrieval-language routing."""

    if not isinstance(
        request,
        BilingualQueryRequest,
    ):
        raise TypeError(
            "request must be a "
            "BilingualQueryRequest."
        )

    response_language = (
        request.resolved_response_language
    )

    if response_language is None:
        raise ValueError(
            "response_language must be specified "
            "when query language is mixed or unknown."
        )

    if request.allow_cross_lingual:
        if (
            request.detected_language
            is DetectedLanguage.ARABIC
        ):
            languages = (
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            )

        elif (
            request.detected_language
            is DetectedLanguage.ENGLISH
        ):
            languages = (
                BilingualLanguage.ENGLISH,
                BilingualLanguage.ARABIC,
            )

        elif (
            response_language
            is BilingualLanguage.ARABIC
        ):
            languages = (
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            )

        else:
            languages = (
                BilingualLanguage.ENGLISH,
                BilingualLanguage.ARABIC,
            )

        mode = (
            LanguageRoutingMode.CROSS_LINGUAL
        )

    else:
        if (
            request.detected_language
            is DetectedLanguage.ARABIC
        ):
            language = (
                BilingualLanguage.ARABIC
            )

        elif (
            request.detected_language
            is DetectedLanguage.ENGLISH
        ):
            language = (
                BilingualLanguage.ENGLISH
            )

        else:
            language = (
                response_language
            )

        languages = (
            language,
        )

        mode = (
            LanguageRoutingMode.MONOLINGUAL
        )

    return LanguageRoute(
        query_id=request.query_id,
        query_language=(
            request.detected_language
        ),
        response_language=(
            response_language
        ),
        retrieval_languages=(
            languages
        ),
        mode=mode,
    )
