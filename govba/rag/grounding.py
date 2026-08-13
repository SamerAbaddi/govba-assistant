"""Structural grounding validation and abstention gate for GovBA-GAR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from govba.rag.citations import EvidenceCitation
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceVerificationState,
)


GROUNDING_VALIDATOR_VERSION = (
    "govba-grounding-validator-v1"
)

_CITATION_MARKER_RE = re.compile(
    r"\[(E[1-9][0-9]*)\]"
)


class GroundingDecision(
    str,
    Enum,
):
    GROUNDED = "grounded"
    ABSTAIN = "abstain"


class GroundingIssueCode(
    str,
    Enum,
):
    MISSING_CITATION = "missing_citation"
    UNKNOWN_CITATION = "unknown_citation"
    UNKNOWN_CARD = "unknown_card"
    CITATION_CARD_MISMATCH = (
        "citation_card_mismatch"
    )
    CITATION_MARKER_MISMATCH = (
        "citation_marker_mismatch"
    )
    DUPLICATE_CITATION_ID = (
        "duplicate_citation_id"
    )
    DUPLICATE_CARD_ID = (
        "duplicate_card_id"
    )
    DUPLICATE_CLAIM_ID = (
        "duplicate_claim_id"
    )
    REJECTED_EVIDENCE = (
        "rejected_evidence"
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


@dataclass(frozen=True)
class GroundedClaim:
    """One answer claim with explicit evidence references."""

    claim_id: str
    text: str
    requires_evidence: bool = True
    citation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        claim_id = _required_text(
            self.claim_id,
            "claim_id",
        )

        text = _required_text(
            self.text,
            "text",
        )

        if not isinstance(
            self.requires_evidence,
            bool,
        ):
            raise TypeError(
                "requires_evidence must be boolean."
            )

        try:
            citation_ids = tuple(
                self.citation_ids
            )
        except TypeError as exc:
            raise TypeError(
                "citation_ids must be iterable."
            ) from exc

        normalized = []

        for citation_id in citation_ids:
            normalized.append(
                _required_text(
                    citation_id,
                    "citation_id",
                )
            )

        if (
            len(set(normalized))
            != len(normalized)
        ):
            raise ValueError(
                "citation_ids must be unique "
                "within a claim."
            )

        object.__setattr__(
            self,
            "claim_id",
            claim_id,
        )

        object.__setattr__(
            self,
            "text",
            text,
        )

        object.__setattr__(
            self,
            "citation_ids",
            tuple(normalized),
        )

    @property
    def inline_citation_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                _CITATION_MARKER_RE.findall(
                    self.text
                )
            )
        )

    @property
    def evidence_references(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.citation_ids,
                    *self.inline_citation_ids,
                )
            )
        )


@dataclass(frozen=True)
class GroundingIssue:
    """One grounding validation problem."""

    code: GroundingIssueCode
    message: str
    claim_id: str | None = None
    citation_id: str | None = None
    card_id: str | None = None


@dataclass(frozen=True)
class GroundingValidationReport:
    """Privacy-safe grounding validation summary."""

    claims: tuple[GroundedClaim, ...]
    citations: tuple[EvidenceCitation, ...]
    cards: tuple[EvidenceCard, ...]
    issues: tuple[GroundingIssue, ...]
    validator_version: str = (
        GROUNDING_VALIDATOR_VERSION
    )

    @property
    def claim_count(self) -> int:
        return len(
            self.claims
        )

    @property
    def citation_count(self) -> int:
        return len(
            self.citations
        )

    @property
    def card_count(self) -> int:
        return len(
            self.cards
        )

    @property
    def issue_count(self) -> int:
        return len(
            self.issues
        )

    @property
    def is_grounded(self) -> bool:
        return self.issue_count == 0

    @property
    def should_abstain(self) -> bool:
        return not self.is_grounded

    @property
    def decision(
        self,
    ) -> GroundingDecision:
        if self.is_grounded:
            return (
                GroundingDecision.GROUNDED
            )

        return GroundingDecision.ABSTAIN

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Return summary without raw claim or evidence text."""

        return {
            "validator_version": (
                self.validator_version
            ),
            "decision": (
                self.decision.value
            ),
            "is_grounded": (
                self.is_grounded
            ),
            "should_abstain": (
                self.should_abstain
            ),
            "claim_count": (
                self.claim_count
            ),
            "citation_count": (
                self.citation_count
            ),
            "card_count": (
                self.card_count
            ),
            "issue_count": (
                self.issue_count
            ),
            "issues": [
                {
                    "code": issue.code.value,
                    "claim_id": (
                        issue.claim_id
                    ),
                    "citation_id": (
                        issue.citation_id
                    ),
                    "card_id": (
                        issue.card_id
                    ),
                }
                for issue in self.issues
            ],
        }


class GroundingValidator:
    """Validate claims, citations, and evidence-card linkage."""

    def validate(
        self,
        claims: Iterable[
            GroundedClaim
        ],
        citations: Iterable[
            EvidenceCitation
        ],
        cards: Iterable[
            EvidenceCard
        ],
    ) -> GroundingValidationReport:
        try:
            claim_values = tuple(
                claims
            )
            citation_values = tuple(
                citations
            )
            card_values = tuple(
                cards
            )
        except TypeError as exc:
            raise TypeError(
                "claims, citations, and cards "
                "must be iterable."
            ) from exc

        for claim in claim_values:
            if not isinstance(
                claim,
                GroundedClaim,
            ):
                raise TypeError(
                    "claims must contain only "
                    "GroundedClaim values."
                )

        for citation in citation_values:
            if not isinstance(
                citation,
                EvidenceCitation,
            ):
                raise TypeError(
                    "citations must contain only "
                    "EvidenceCitation values."
                )

        for card in card_values:
            if not isinstance(
                card,
                EvidenceCard,
            ):
                raise TypeError(
                    "cards must contain only "
                    "EvidenceCard values."
                )

        issues = []

        card_map: dict[
            str,
            EvidenceCard,
        ] = {}

        for card in card_values:
            if card.card_id in card_map:
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .DUPLICATE_CARD_ID
                        ),
                        message=(
                            "Duplicate evidence card ID."
                        ),
                        card_id=card.card_id,
                    )
                )
            else:
                card_map[
                    card.card_id
                ] = card

        citation_map: dict[
            str,
            EvidenceCitation,
        ] = {}

        for citation in citation_values:
            if (
                citation.citation_id
                in citation_map
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .DUPLICATE_CITATION_ID
                        ),
                        message=(
                            "Duplicate citation ID."
                        ),
                        citation_id=(
                            citation.citation_id
                        ),
                    )
                )
                continue

            citation_map[
                citation.citation_id
            ] = citation

            expected_marker = (
                f"[{citation.citation_id}]"
            )

            if (
                citation.marker
                != expected_marker
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .CITATION_MARKER_MISMATCH
                        ),
                        message=(
                            "Citation marker does not "
                            "match citation ID."
                        ),
                        citation_id=(
                            citation.citation_id
                        ),
                    )
                )

            card = card_map.get(
                citation.card_id
            )

            if card is None:
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .UNKNOWN_CARD
                        ),
                        message=(
                            "Citation references an "
                            "unknown evidence card."
                        ),
                        citation_id=(
                            citation.citation_id
                        ),
                        card_id=(
                            citation.card_id
                        ),
                    )
                )
                continue

            if (
                citation.document_id
                != card.document_id
                or citation.chunk_id
                != card.chunk_id
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .CITATION_CARD_MISMATCH
                        ),
                        message=(
                            "Citation provenance does "
                            "not match evidence card."
                        ),
                        citation_id=(
                            citation.citation_id
                        ),
                        card_id=(
                            card.card_id
                        ),
                    )
                )

            if (
                card.verification_state
                is EvidenceVerificationState.REJECTED
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .REJECTED_EVIDENCE
                        ),
                        message=(
                            "Citation references rejected "
                            "evidence."
                        ),
                        citation_id=(
                            citation.citation_id
                        ),
                        card_id=(
                            card.card_id
                        ),
                    )
                )

        seen_claim_ids = set()

        for claim in claim_values:
            if (
                claim.claim_id
                in seen_claim_ids
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .DUPLICATE_CLAIM_ID
                        ),
                        message=(
                            "Duplicate claim ID."
                        ),
                        claim_id=(
                            claim.claim_id
                        ),
                    )
                )
            else:
                seen_claim_ids.add(
                    claim.claim_id
                )

            references = (
                claim.evidence_references
            )

            if (
                claim.requires_evidence
                and not references
            ):
                issues.append(
                    GroundingIssue(
                        code=(
                            GroundingIssueCode
                            .MISSING_CITATION
                        ),
                        message=(
                            "Evidence-dependent claim "
                            "has no citation."
                        ),
                        claim_id=(
                            claim.claim_id
                        ),
                    )
                )

            for citation_id in references:
                if (
                    citation_id
                    not in citation_map
                ):
                    issues.append(
                        GroundingIssue(
                            code=(
                                GroundingIssueCode
                                .UNKNOWN_CITATION
                            ),
                            message=(
                                "Claim references an "
                                "unknown citation."
                            ),
                            claim_id=(
                                claim.claim_id
                            ),
                            citation_id=(
                                citation_id
                            ),
                        )
                    )

        return GroundingValidationReport(
            claims=claim_values,
            citations=citation_values,
            cards=card_values,
            issues=tuple(
                issues
            ),
        )


def validate_grounding(
    claims: Iterable[
        GroundedClaim
    ],
    citations: Iterable[
        EvidenceCitation
    ],
    cards: Iterable[
        EvidenceCard
    ],
) -> GroundingValidationReport:
    """Convenience function for structural grounding validation."""

    return GroundingValidator().validate(
        claims,
        citations,
        cards,
    )
