"""As-of-date temporal resolution for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from govba.rag.supersession import (
    SupersessionDeclaration,
    SupersessionGraph,
)
from govba.rag.temporal import (
    TemporalPolicyAssessment,
    TemporalPolicyState,
    TemporalReasonCode,
)


TEMPORAL_RESOLVER_VERSION = (
    "govba-temporal-resolver-v1"
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


@dataclass(frozen=True)
class TemporalResolution:
    """Auditable result of one as-of-date policy resolution."""

    assessment: TemporalPolicyAssessment

    active_superseding_document_ids: (
        tuple[str, ...]
    ) = ()

    future_superseding_document_ids: (
        tuple[str, ...]
    ) = ()

    unresolved_superseding_document_ids: (
        tuple[str, ...]
    ) = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.assessment,
            TemporalPolicyAssessment,
        ):
            raise TypeError(
                "assessment must be a "
                "TemporalPolicyAssessment."
            )

        for field_name in (
            "active_superseding_document_ids",
            "future_superseding_document_ids",
            "unresolved_superseding_document_ids",
        ):
            values = tuple(
                getattr(
                    self,
                    field_name,
                )
            )

            if (
                any(
                    not isinstance(
                        value,
                        str,
                    )
                    or not value.strip()
                    for value in values
                )
            ):
                raise ValueError(
                    f"{field_name} must contain "
                    "non-blank document IDs."
                )

            if len(set(values)) != len(values):
                raise ValueError(
                    f"{field_name} must not "
                    "contain duplicates."
                )

            object.__setattr__(
                self,
                field_name,
                tuple(
                    sorted(
                        value.strip()
                        for value in values
                    )
                ),
            )

    @property
    def document_id(self) -> str:
        return self.assessment.document_id

    @property
    def state(
        self,
    ) -> TemporalPolicyState:
        return self.assessment.state

    @property
    def should_abstain(self) -> bool:
        return (
            self.assessment.should_abstain
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "resolver_version": (
                TEMPORAL_RESOLVER_VERSION
            ),
            "assessment": (
                self.assessment.to_dict()
            ),
            "active_superseding_document_ids": list(
                self.active_superseding_document_ids
            ),
            "future_superseding_document_ids": list(
                self.future_superseding_document_ids
            ),
            "unresolved_superseding_document_ids": list(
                self.unresolved_superseding_document_ids
            ),
        }


class TemporalResolver:
    """Resolve authoritative policy state for a requested date."""

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

    def resolve(
        self,
        document_id: str,
        *,
        as_of_date: date,
    ) -> TemporalResolution:
        as_of_date = _validate_date(
            as_of_date
        )

        source = self._graph.source(
            document_id
        )

        if (
            source.document_id
            in self._graph.cycle_document_ids
        ):
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .CONFLICTING
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .CONFLICT_DETECTED,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
            )

        if (
            source.effective_from is not None
            and as_of_date
            < source.effective_from
        ):
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .INSUFFICIENT
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .BEFORE_EFFECTIVE_DATE,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
            )

        if (
            source.effective_until is not None
            and as_of_date
            > source.effective_until
        ):
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .SUPERSEDED
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .AFTER_EFFECTIVE_DATE,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
            )

        active = []
        future = []
        unknown_date = []

        for superseder_id in (
            self._graph
            .all_superseding_documents(
                source.document_id
            )
        ):
            superseder = (
                self._graph.source(
                    superseder_id
                )
            )

            if (
                superseder.effective_from
                is None
            ):
                unknown_date.append(
                    superseder_id
                )

            elif (
                superseder.effective_from
                <= as_of_date
            ):
                active.append(
                    superseder_id
                )

            else:
                future.append(
                    superseder_id
                )

        unresolved = []

        for reference in (
            self._graph
            .unresolved_references
        ):
            if (
                reference.owner_document_id
                == source.document_id
                and reference.declaration
                is SupersessionDeclaration
                .SUPERSEDED_BY
            ):
                unresolved.append(
                    reference.related_document_id
                )

        if active:
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .SUPERSEDED
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .EXPLICITLY_SUPERSEDED,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
                active_superseding_document_ids=(
                    tuple(active)
                ),
                future_superseding_document_ids=(
                    tuple(future)
                ),
                unresolved_superseding_document_ids=(
                    tuple(
                        sorted(
                            set(
                                unknown_date
                                + unresolved
                            )
                        )
                    )
                ),
            )

        unresolved_ids = tuple(
            sorted(
                set(
                    unknown_date
                    + unresolved
                )
            )
        )

        if unresolved_ids:
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .INSUFFICIENT
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .SUPERSESSION_DATE_UNKNOWN,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
                future_superseding_document_ids=(
                    tuple(future)
                ),
                unresolved_superseding_document_ids=(
                    unresolved_ids
                ),
            )

        if source.effective_from is None:
            assessment = (
                TemporalPolicyAssessment
                .from_source(
                    source,
                    as_of_date=as_of_date,
                    state=(
                        TemporalPolicyState
                        .INSUFFICIENT
                    ),
                    reason_codes=(
                        TemporalReasonCode
                        .MISSING_TEMPORAL_METADATA,
                    ),
                )
            )

            return TemporalResolution(
                assessment=assessment,
                future_superseding_document_ids=(
                    tuple(future)
                ),
            )

        assessment = (
            TemporalPolicyAssessment
            .from_source(
                source,
                as_of_date=as_of_date,
                state=(
                    TemporalPolicyState
                    .CURRENT
                ),
                reason_codes=(
                    TemporalReasonCode
                    .WITHIN_EFFECTIVE_WINDOW,
                ),
            )
        )

        return TemporalResolution(
            assessment=assessment,
            future_superseding_document_ids=(
                tuple(future)
            ),
        )

    def resolve_all(
        self,
        *,
        as_of_date: date,
    ) -> tuple[
        TemporalResolution,
        ...
    ]:
        """Resolve all controlled sources deterministically."""

        as_of_date = _validate_date(
            as_of_date
        )

        return tuple(
            self.resolve(
                document_id,
                as_of_date=as_of_date,
            )
            for document_id
            in self._graph.document_ids
        )
