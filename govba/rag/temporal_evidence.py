"""Temporal integration for GovBA-GAR evidence cards."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Iterable

from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
)
from govba.rag.supersession import (
    SupersessionGraph,
)
from govba.rag.temporal import (
    TemporalPolicyState,
)
from govba.rag.temporal_conflicts import (
    TemporalConflictReport,
    detect_temporal_conflicts,
)


TEMPORAL_EVIDENCE_VERSION = (
    "govba-temporal-evidence-v1"
)


def _validate_date(
    value: date,
) -> date:
    if (
        isinstance(value, datetime)
        or not isinstance(value, date)
    ):
        raise TypeError(
            "as_of_date must be a date."
        )

    return value


def _map_temporal_state(
    state: TemporalPolicyState,
) -> EvidenceTemporalState:
    mapping = {
        TemporalPolicyState.CURRENT: (
            EvidenceTemporalState.CURRENT
        ),
        TemporalPolicyState.SUPERSEDED: (
            EvidenceTemporalState.SUPERSEDED
        ),
        TemporalPolicyState.CONFLICTING: (
            EvidenceTemporalState.CONFLICTING
        ),
        TemporalPolicyState.INSUFFICIENT: (
            EvidenceTemporalState.INSUFFICIENT
        ),
    }

    return mapping[state]


@dataclass(frozen=True)
class TemporalEvidenceIntegrationReport:
    """Auditable result of temporal evidence integration."""

    as_of_date: date

    cards: tuple[
        EvidenceCard,
        ...
    ]

    conflict_report: (
        TemporalConflictReport
    )

    integration_version: str = (
        TEMPORAL_EVIDENCE_VERSION
    )

    def __post_init__(self) -> None:
        _validate_date(
            self.as_of_date
        )

        cards = tuple(
            self.cards
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

        if not isinstance(
            self.conflict_report,
            TemporalConflictReport,
        ):
            raise TypeError(
                "conflict_report must be a "
                "TemporalConflictReport."
            )

        if (
            self.conflict_report.as_of_date
            != self.as_of_date
        ):
            raise ValueError(
                "Conflict-report date does not "
                "match integration date."
            )

        card_ids = tuple(
            card.card_id
            for card in cards
        )

        if (
            len(set(card_ids))
            != len(card_ids)
        ):
            raise ValueError(
                "Evidence card IDs must be unique."
            )

        object.__setattr__(
            self,
            "cards",
            cards,
        )

    @property
    def current_cards(
        self,
    ) -> tuple[
        EvidenceCard,
        ...
    ]:
        return tuple(
            card
            for card in self.cards
            if card.temporal_state
            is EvidenceTemporalState.CURRENT
        )

    @property
    def blocked_cards(
        self,
    ) -> tuple[
        EvidenceCard,
        ...
    ]:
        return tuple(
            card
            for card in self.cards
            if card.temporal_state
            in {
                EvidenceTemporalState.SUPERSEDED,
                EvidenceTemporalState.CONFLICTING,
                EvidenceTemporalState.INSUFFICIENT,
            }
        )

    @property
    def current_count(
        self,
    ) -> int:
        return len(
            self.current_cards
        )

    @property
    def blocked_count(
        self,
    ) -> int:
        return len(
            self.blocked_cards
        )

    @property
    def has_conflicting_evidence(
        self,
    ) -> bool:
        return any(
            card.temporal_state
            is EvidenceTemporalState.CONFLICTING
            for card in self.cards
        )

    @property
    def has_insufficient_evidence(
        self,
    ) -> bool:
        return any(
            card.temporal_state
            is EvidenceTemporalState.INSUFFICIENT
            for card in self.cards
        )

    @property
    def has_temporal_risk(
        self,
    ) -> bool:
        return bool(
            self.blocked_cards
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize metadata without evidence text."""

        return {
            "integration_version": (
                self.integration_version
            ),
            "as_of_date": (
                self.as_of_date.isoformat()
            ),
            "card_count": len(
                self.cards
            ),
            "current_count": (
                self.current_count
            ),
            "blocked_count": (
                self.blocked_count
            ),
            "has_temporal_risk": (
                self.has_temporal_risk
            ),
            "has_conflicting_evidence": (
                self.has_conflicting_evidence
            ),
            "has_insufficient_evidence": (
                self.has_insufficient_evidence
            ),
            "cards": [
                {
                    "card_id": (
                        card.card_id
                    ),
                    "document_id": (
                        card.document_id
                    ),
                    "temporal_state": (
                        card.temporal_state.value
                    ),
                }
                for card in self.cards
            ],
        }


class TemporalEvidenceIntegrator:
    """Apply temporal governance state to retrieved evidence cards."""

    def __init__(
        self,
        graph: SupersessionGraph,
    ) -> None:
        if not isinstance(
            graph,
            SupersessionGraph,
        ):
            raise TypeError(
                "graph must be a SupersessionGraph."
            )

        self._graph = graph

    @property
    def graph(
        self,
    ) -> SupersessionGraph:
        return self._graph

    def integrate(
        self,
        cards: Iterable[
            EvidenceCard
        ],
        *,
        as_of_date: date,
    ) -> TemporalEvidenceIntegrationReport:
        as_of_date = _validate_date(
            as_of_date
        )

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

        for card in card_values:
            self._graph.source(
                card.document_id
            )

        conflict_report = (
            detect_temporal_conflicts(
                self._graph,
                as_of_date=as_of_date,
            )
        )

        resolution_map = {
            resolution.document_id: (
                resolution
            )
            for resolution
            in conflict_report.resolutions
        }

        conflicting_document_ids = set()

        for conflict in (
            conflict_report.conflicts
        ):
            conflicting_document_ids.update(
                conflict.component_document_ids
            )

        integrated_cards = []

        for card in card_values:
            if (
                card.document_id
                in conflicting_document_ids
            ):
                state = (
                    EvidenceTemporalState
                    .CONFLICTING
                )

            else:
                resolution = resolution_map[
                    card.document_id
                ]

                state = _map_temporal_state(
                    resolution.state
                )

            integrated_cards.append(
                replace(
                    card,
                    temporal_state=state,
                )
            )

        return TemporalEvidenceIntegrationReport(
            as_of_date=as_of_date,
            cards=tuple(
                integrated_cards
            ),
            conflict_report=(
                conflict_report
            ),
        )


def integrate_temporal_evidence(
    cards: Iterable[
        EvidenceCard
    ],
    graph: SupersessionGraph,
    *,
    as_of_date: date,
) -> TemporalEvidenceIntegrationReport:
    """Convenience function for temporal evidence integration."""

    return TemporalEvidenceIntegrator(
        graph
    ).integrate(
        cards,
        as_of_date=as_of_date,
    )
