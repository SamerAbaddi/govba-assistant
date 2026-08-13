"""Evidence eligibility gate for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)


EVIDENCE_ELIGIBILITY_VERSION = (
    "govba-evidence-eligibility-v1"
)


class EvidenceEligibilityReason(
    str,
    Enum,
):
    """Machine-readable eligibility outcome."""

    ELIGIBLE = "eligible"

    REJECTED_EVIDENCE = (
        "rejected_evidence"
    )

    TEMPORAL_UNASSESSED = (
        "temporal_unassessed"
    )

    SUPERSEDED_EVIDENCE = (
        "superseded_evidence"
    )

    TEMPORAL_CONFLICT = (
        "temporal_conflict"
    )

    TEMPORAL_INSUFFICIENT = (
        "temporal_insufficient"
    )


@dataclass(frozen=True)
class EvidenceEligibilityResult:
    """Eligibility decision for one Evidence Card."""

    card_id: str
    document_id: str
    eligible: bool

    reason_codes: tuple[
        EvidenceEligibilityReason,
        ...
    ]

    version: str = (
        EVIDENCE_ELIGIBILITY_VERSION
    )

    def __post_init__(self) -> None:
        for field_name in (
            "card_id",
            "document_id",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be "
                    "a non-blank string."
                )

            object.__setattr__(
                self,
                field_name,
                value.strip(),
            )

        if not isinstance(
            self.eligible,
            bool,
        ):
            raise TypeError(
                "eligible must be boolean."
            )

        try:
            reasons = tuple(
                self.reason_codes
            )
        except TypeError as exc:
            raise TypeError(
                "reason_codes must be iterable."
            ) from exc

        if not reasons:
            raise ValueError(
                "reason_codes must not be empty."
            )

        for reason in reasons:
            if not isinstance(
                reason,
                EvidenceEligibilityReason,
            ):
                raise TypeError(
                    "reason_codes must contain only "
                    "EvidenceEligibilityReason values."
                )

        if (
            len(set(reasons))
            != len(reasons)
        ):
            raise ValueError(
                "reason_codes must not contain "
                "duplicates."
            )

        reasons = tuple(
            sorted(
                reasons,
                key=lambda item: item.value,
            )
        )

        if self.eligible:
            if (
                reasons
                != (
                    EvidenceEligibilityReason
                    .ELIGIBLE,
                )
            ):
                raise ValueError(
                    "Eligible evidence must have "
                    "ELIGIBLE as its only reason."
                )

        else:
            if (
                EvidenceEligibilityReason
                .ELIGIBLE
                in reasons
            ):
                raise ValueError(
                    "Ineligible evidence cannot "
                    "include ELIGIBLE."
                )

        object.__setattr__(
            self,
            "reason_codes",
            reasons,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "card_id": (
                self.card_id
            ),
            "document_id": (
                self.document_id
            ),
            "eligible": (
                self.eligible
            ),
            "reason_codes": [
                reason.value
                for reason
                in self.reason_codes
            ],
        }


def evaluate_evidence_card(
    card: EvidenceCard,
) -> EvidenceEligibilityResult:
    """Evaluate one Evidence Card for governed-answer eligibility."""

    if not isinstance(
        card,
        EvidenceCard,
    ):
        raise TypeError(
            "card must be an EvidenceCard."
        )

    reasons = []

    if (
        card.verification_state
        is EvidenceVerificationState.REJECTED
    ):
        reasons.append(
            EvidenceEligibilityReason
            .REJECTED_EVIDENCE
        )

    if (
        card.temporal_state
        is EvidenceTemporalState.UNASSESSED
    ):
        reasons.append(
            EvidenceEligibilityReason
            .TEMPORAL_UNASSESSED
        )

    elif (
        card.temporal_state
        is EvidenceTemporalState.SUPERSEDED
    ):
        reasons.append(
            EvidenceEligibilityReason
            .SUPERSEDED_EVIDENCE
        )

    elif (
        card.temporal_state
        is EvidenceTemporalState.CONFLICTING
    ):
        reasons.append(
            EvidenceEligibilityReason
            .TEMPORAL_CONFLICT
        )

    elif (
        card.temporal_state
        is EvidenceTemporalState.INSUFFICIENT
    ):
        reasons.append(
            EvidenceEligibilityReason
            .TEMPORAL_INSUFFICIENT
        )

    if reasons:
        return EvidenceEligibilityResult(
            card_id=card.card_id,
            document_id=card.document_id,
            eligible=False,
            reason_codes=tuple(
                reasons
            ),
        )

    return EvidenceEligibilityResult(
        card_id=card.card_id,
        document_id=card.document_id,
        eligible=True,
        reason_codes=(
            EvidenceEligibilityReason.ELIGIBLE,
        ),
    )


@dataclass(frozen=True)
class EvidenceEligibilityReport:
    """Aggregate evidence-eligibility result."""

    cards: tuple[
        EvidenceCard,
        ...
    ]

    results: tuple[
        EvidenceEligibilityResult,
        ...
    ]

    version: str = (
        EVIDENCE_ELIGIBILITY_VERSION
    )

    def __post_init__(self) -> None:
        cards = tuple(
            self.cards
        )

        results = tuple(
            self.results
        )

        for card in cards:
            if not isinstance(
                card,
                EvidenceCard,
            ):
                raise TypeError(
                    "cards must contain only "
                    "EvidenceCard values."
                )

        for result in results:
            if not isinstance(
                result,
                EvidenceEligibilityResult,
            ):
                raise TypeError(
                    "results must contain only "
                    "EvidenceEligibilityResult values."
                )

        card_ids = tuple(
            card.card_id
            for card in cards
        )

        result_ids = tuple(
            result.card_id
            for result in results
        )

        if (
            len(set(card_ids))
            != len(card_ids)
        ):
            raise ValueError(
                "Evidence card IDs must be unique."
            )

        if card_ids != result_ids:
            raise ValueError(
                "Eligibility results must correspond "
                "to cards in the same order."
            )

        object.__setattr__(
            self,
            "cards",
            cards,
        )

        object.__setattr__(
            self,
            "results",
            results,
        )

    @property
    def evidence_count(
        self,
    ) -> int:
        return len(
            self.cards
        )

    @property
    def eligible_count(
        self,
    ) -> int:
        return sum(
            result.eligible
            for result in self.results
        )

    @property
    def ineligible_count(
        self,
    ) -> int:
        return (
            self.evidence_count
            - self.eligible_count
        )

    @property
    def eligible_cards(
        self,
    ) -> tuple[
        EvidenceCard,
        ...
    ]:
        return tuple(
            card
            for card, result
            in zip(
                self.cards,
                self.results,
            )
            if result.eligible
        )

    @property
    def ineligible_cards(
        self,
    ) -> tuple[
        EvidenceCard,
        ...
    ]:
        return tuple(
            card
            for card, result
            in zip(
                self.cards,
                self.results,
            )
            if not result.eligible
        )

    @property
    def has_eligible_evidence(
        self,
    ) -> bool:
        return (
            self.eligible_count > 0
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without evidence text."""

        return {
            "version": (
                self.version
            ),
            "evidence_count": (
                self.evidence_count
            ),
            "eligible_count": (
                self.eligible_count
            ),
            "ineligible_count": (
                self.ineligible_count
            ),
            "has_eligible_evidence": (
                self.has_eligible_evidence
            ),
            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


def evaluate_evidence_cards(
    cards: Iterable[
        EvidenceCard
    ],
) -> EvidenceEligibilityReport:
    """Evaluate multiple Evidence Cards deterministically."""

    try:
        card_values = tuple(
            cards
        )
    except TypeError as exc:
        raise TypeError(
            "cards must be iterable."
        ) from exc

    for card in card_values:
        if not isinstance(
            card,
            EvidenceCard,
        ):
            raise TypeError(
                "cards must contain only "
                "EvidenceCard values."
            )

    card_ids = tuple(
        card.card_id
        for card in card_values
    )

    if (
        len(set(card_ids))
        != len(card_ids)
    ):
        raise ValueError(
            "Evidence card IDs must be unique."
        )

    results = tuple(
        evaluate_evidence_card(
            card
        )
        for card in card_values
    )

    return EvidenceEligibilityReport(
        cards=card_values,
        results=results,
    )
