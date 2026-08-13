"""Deterministic document-version matching for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from govba.change.contract import (
    PolicyChangeRequest,
)
from govba.rag.models import (
    AuthoritativeSource,
)


DOCUMENT_VERSION_MATCHING_VERSION = (
    "govba-document-version-matching-v1"
)


class DocumentVersionMatchDecision(
    str,
    Enum,
):
    """Disposition for whether two documents are safe to compare."""

    ALLOW = "allow"
    REVIEW = "review"
    REJECT = "reject"


class DocumentVersionMatchReason(
    str,
    Enum,
):
    """Stable deterministic version-matching reasons."""

    SAME_DOCUMENT_ID = (
        "same_document_id"
    )

    EXPLICIT_SUPERSESSION = (
        "explicit_supersession"
    )

    METADATA_MATCH = (
        "metadata_match"
    )

    SAME_CONTENT_DIFFERENT_ID = (
        "same_content_different_id"
    )

    TITLE_MISMATCH = (
        "title_mismatch"
    )

    AUTHORITY_MISMATCH = (
        "authority_mismatch"
    )

    DOCUMENT_TYPE_MISMATCH = (
        "document_type_mismatch"
    )

    JURISDICTION_MISMATCH = (
        "jurisdiction_mismatch"
    )


_WHITESPACE_RE = re.compile(
    r"\s+"
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


def _normalize_metadata_text(
    value: str,
) -> str:
    """Normalize metadata without changing semantic word content."""

    value = unicodedata.normalize(
        "NFKC",
        _required_text(
            value,
            "metadata value",
        ),
    ).casefold()

    characters = []

    for character in value:
        category = unicodedata.category(
            character
        )

        if category.startswith(
            ("P", "S")
        ):
            characters.append(
                " "
            )
        else:
            characters.append(
                character
            )

    return _WHITESPACE_RE.sub(
        " ",
        "".join(
            characters
        ),
    ).strip()


def _has_explicit_supersession(
    baseline: AuthoritativeSource,
    candidate: AuthoritativeSource,
) -> bool:
    """Detect a direct declared supersession link in either direction."""

    return any(
        (
            candidate.document_id
            in baseline.superseded_by,

            baseline.document_id
            in candidate.supersedes,

            candidate.document_id
            in baseline.supersedes,

            baseline.document_id
            in candidate.superseded_by,
        )
    )


def _match_id(
    *,
    baseline_document_id: str,
    candidate_document_id: str,
    decision: DocumentVersionMatchDecision,
    reasons: tuple[
        DocumentVersionMatchReason,
        ...
    ],
    flags: tuple[
        bool,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            DOCUMENT_VERSION_MATCHING_VERSION,
            baseline_document_id,
            candidate_document_id,
            decision.value,
            ",".join(
                reason.value
                for reason in reasons
            ),
            ",".join(
                str(flag).lower()
                for flag in flags
            ),
        )
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class DocumentVersionMatchResult:
    """Deterministic assessment of one possible document-version pair."""

    baseline_source: (
        AuthoritativeSource
    )

    candidate_source: (
        AuthoritativeSource
    )

    decision: (
        DocumentVersionMatchDecision
    )

    reasons: tuple[
        DocumentVersionMatchReason,
        ...
    ]

    same_document_id: bool

    explicit_supersession: bool

    same_title: bool

    same_authority: bool

    same_document_type: bool

    same_jurisdiction: bool

    same_content: bool

    version: str = (
        DOCUMENT_VERSION_MATCHING_VERSION
    )

    match_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.baseline_source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "baseline_source must be an "
                "AuthoritativeSource."
            )

        if not isinstance(
            self.candidate_source,
            AuthoritativeSource,
        ):
            raise TypeError(
                "candidate_source must be an "
                "AuthoritativeSource."
            )

        if not isinstance(
            self.decision,
            DocumentVersionMatchDecision,
        ):
            raise TypeError(
                "decision must be a "
                "DocumentVersionMatchDecision."
            )

        try:
            reasons = tuple(
                self.reasons
            )
        except TypeError as exc:
            raise TypeError(
                "reasons must be iterable."
            ) from exc

        if not reasons:
            raise ValueError(
                "At least one match reason "
                "is required."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                DocumentVersionMatchReason,
            ):
                raise TypeError(
                    "reasons must contain only "
                    "DocumentVersionMatchReason values."
                )

        if (
            len(set(reasons))
            != len(reasons)
        ):
            raise ValueError(
                "Match reasons must be unique."
            )

        boolean_fields = (
            "same_document_id",
            "explicit_supersession",
            "same_title",
            "same_authority",
            "same_document_type",
            "same_jurisdiction",
            "same_content",
        )

        for field_name in boolean_fields:
            if not isinstance(
                getattr(
                    self,
                    field_name,
                ),
                bool,
            ):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        match_id = _match_id(
            baseline_document_id=(
                self.baseline_source
                .document_id
            ),
            candidate_document_id=(
                self.candidate_source
                .document_id
            ),
            decision=(
                self.decision
            ),
            reasons=reasons,
            flags=tuple(
                getattr(
                    self,
                    field_name,
                )
                for field_name
                in boolean_fields
            ),
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "match_id",
            match_id,
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            is DocumentVersionMatchDecision.ALLOW
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is DocumentVersionMatchDecision.REVIEW
        )

    @property
    def rejected(
        self,
    ) -> bool:
        return (
            self.decision
            is DocumentVersionMatchDecision.REJECT
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize matching metadata without source URLs or text."""

        return {
            "version": (
                self.version
            ),
            "match_id": (
                self.match_id
            ),
            "baseline_document_id": (
                self.baseline_source
                .document_id
            ),
            "candidate_document_id": (
                self.candidate_source
                .document_id
            ),
            "decision": (
                self.decision.value
            ),
            "allowed": (
                self.allowed
            ),
            "requires_review": (
                self.requires_review
            ),
            "rejected": (
                self.rejected
            ),
            "reasons": [
                reason.value
                for reason
                in self.reasons
            ],
            "same_document_id": (
                self.same_document_id
            ),
            "explicit_supersession": (
                self.explicit_supersession
            ),
            "same_title": (
                self.same_title
            ),
            "same_authority": (
                self.same_authority
            ),
            "same_document_type": (
                self.same_document_type
            ),
            "same_jurisdiction": (
                self.same_jurisdiction
            ),
            "same_content": (
                self.same_content
            ),
        }


def match_document_versions(
    baseline_source: AuthoritativeSource,
    candidate_source: AuthoritativeSource,
) -> DocumentVersionMatchResult:
    """Determine whether two authoritative documents are safe to compare."""

    if not isinstance(
        baseline_source,
        AuthoritativeSource,
    ):
        raise TypeError(
            "baseline_source must be an "
            "AuthoritativeSource."
        )

    if not isinstance(
        candidate_source,
        AuthoritativeSource,
    ):
        raise TypeError(
            "candidate_source must be an "
            "AuthoritativeSource."
        )

    same_document_id = (
        baseline_source.document_id
        == candidate_source.document_id
    )

    explicit_supersession = (
        _has_explicit_supersession(
            baseline_source,
            candidate_source,
        )
    )

    same_title = (
        _normalize_metadata_text(
            baseline_source.title
        )
        == _normalize_metadata_text(
            candidate_source.title
        )
    )

    same_authority = (
        _normalize_metadata_text(
            baseline_source
            .issuing_authority
        )
        == _normalize_metadata_text(
            candidate_source
            .issuing_authority
        )
    )

    same_document_type = (
        baseline_source.document_type
        is candidate_source.document_type
    )

    same_jurisdiction = (
        _normalize_metadata_text(
            baseline_source.jurisdiction
        )
        == _normalize_metadata_text(
            candidate_source.jurisdiction
        )
    )

    baseline_hash = (
        baseline_source
        .content_hash
        .lower()
    )

    candidate_hash = (
        candidate_source
        .content_hash
        .lower()
    )

    same_content = bool(
        baseline_hash
        and candidate_hash
        and baseline_hash
        == candidate_hash
    )

    reasons = []

    if same_document_id:
        decision = (
            DocumentVersionMatchDecision
            .ALLOW
        )

        reasons.append(
            DocumentVersionMatchReason
            .SAME_DOCUMENT_ID
        )

    elif explicit_supersession:
        decision = (
            DocumentVersionMatchDecision
            .ALLOW
        )

        reasons.append(
            DocumentVersionMatchReason
            .EXPLICIT_SUPERSESSION
        )

    elif (
        same_title
        and same_authority
        and same_document_type
        and same_jurisdiction
    ):
        decision = (
            DocumentVersionMatchDecision
            .ALLOW
        )

        reasons.append(
            DocumentVersionMatchReason
            .METADATA_MATCH
        )

    elif not same_jurisdiction:
        decision = (
            DocumentVersionMatchDecision
            .REJECT
        )

        reasons.append(
            DocumentVersionMatchReason
            .JURISDICTION_MISMATCH
        )

    else:
        mismatches = []

        if not same_title:
            mismatches.append(
                DocumentVersionMatchReason
                .TITLE_MISMATCH
            )

        if not same_authority:
            mismatches.append(
                DocumentVersionMatchReason
                .AUTHORITY_MISMATCH
            )

        if not same_document_type:
            mismatches.append(
                DocumentVersionMatchReason
                .DOCUMENT_TYPE_MISMATCH
            )

        if len(
            mismatches
        ) == 1:
            decision = (
                DocumentVersionMatchDecision
                .REVIEW
            )
        else:
            decision = (
                DocumentVersionMatchDecision
                .REJECT
            )

        reasons.extend(
            mismatches
        )

    if (
        same_content
        and not same_document_id
    ):
        reasons.append(
            DocumentVersionMatchReason
            .SAME_CONTENT_DIFFERENT_ID
        )

    return DocumentVersionMatchResult(
        baseline_source=baseline_source,
        candidate_source=candidate_source,
        decision=decision,
        reasons=tuple(
            reasons
        ),
        same_document_id=(
            same_document_id
        ),
        explicit_supersession=(
            explicit_supersession
        ),
        same_title=(
            same_title
        ),
        same_authority=(
            same_authority
        ),
        same_document_type=(
            same_document_type
        ),
        same_jurisdiction=(
            same_jurisdiction
        ),
        same_content=(
            same_content
        ),
    )


def build_change_request_from_match(
    match: DocumentVersionMatchResult,
    *,
    trace_id: str,
) -> PolicyChangeRequest:
    """Build a change request only for an ALLOWed version pair."""

    if not isinstance(
        match,
        DocumentVersionMatchResult,
    ):
        raise TypeError(
            "match must be a "
            "DocumentVersionMatchResult."
        )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if not match.allowed:
        raise ValueError(
            "Document pair is not approved "
            "for automatic change analysis."
        )

    return PolicyChangeRequest(
        baseline_source=(
            match.baseline_source
        ),
        candidate_source=(
            match.candidate_source
        ),
        trace_id=trace,
    )
