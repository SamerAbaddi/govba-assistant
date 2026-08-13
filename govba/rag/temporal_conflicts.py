"""Deterministic temporal policy-conflict detection for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from govba.rag.supersession import (
    SupersessionGraph,
)
from govba.rag.temporal import (
    TemporalPolicyState,
)
from govba.rag.temporal_resolver import (
    TemporalResolution,
    TemporalResolver,
)


TEMPORAL_CONFLICT_VERSION = (
    "govba-temporal-conflict-v1"
)


class TemporalConflictCode(
    str,
    Enum,
):
    """Machine-readable temporal conflict type."""

    SUPERSESSION_CYCLE = (
        "supersession_cycle"
    )

    MULTIPLE_CURRENT_SUCCESSORS = (
        "multiple_current_successors"
    )


class TemporalConflictDecision(
    str,
    Enum,
):
    """Overall temporal conflict decision."""

    CLEAR = "clear"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


def _validate_date(
    value: date,
) -> date:
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
            "as_of_date must be a date."
        )

    return value


def _normalize_document_ids(
    values,
    field_name: str,
) -> tuple[str, ...]:
    try:
        raw_values = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    normalized = []

    for value in raw_values:
        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{field_name} must contain "
                "non-blank document IDs."
            )

        normalized.append(
            value.strip()
        )

    if (
        len(set(normalized))
        != len(normalized)
    ):
        raise ValueError(
            f"{field_name} must not "
            "contain duplicates."
        )

    return tuple(
        sorted(
            normalized
        )
    )


def _conflict_id(
    *,
    as_of_date: date,
    code: TemporalConflictCode,
    component_document_ids: tuple[
        str,
        ...
    ],
    current_document_ids: tuple[
        str,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            TEMPORAL_CONFLICT_VERSION,
            as_of_date.isoformat(),
            code.value,
            ",".join(
                component_document_ids
            ),
            ",".join(
                current_document_ids
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class TemporalConflict:
    """One structural conflict within a supersession family."""

    as_of_date: date
    code: TemporalConflictCode

    component_document_ids: tuple[
        str,
        ...
    ]

    current_document_ids: tuple[
        str,
        ...
    ] = ()

    conflict_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        as_of_date = _validate_date(
            self.as_of_date
        )

        if not isinstance(
            self.code,
            TemporalConflictCode,
        ):
            raise TypeError(
                "code must be a "
                "TemporalConflictCode."
            )

        component_ids = (
            _normalize_document_ids(
                self.component_document_ids,
                "component_document_ids",
            )
        )

        current_ids = (
            _normalize_document_ids(
                self.current_document_ids,
                "current_document_ids",
            )
        )

        if len(
            component_ids
        ) < 2:
            raise ValueError(
                "A temporal conflict must involve "
                "at least two documents."
            )

        if not set(
            current_ids
        ).issubset(
            component_ids
        ):
            raise ValueError(
                "current_document_ids must be "
                "within the conflict component."
            )

        if (
            self.code
            is TemporalConflictCode
            .MULTIPLE_CURRENT_SUCCESSORS
            and len(
                current_ids
            ) < 2
        ):
            raise ValueError(
                "Multiple-current conflict requires "
                "at least two current documents."
            )

        object.__setattr__(
            self,
            "as_of_date",
            as_of_date,
        )

        object.__setattr__(
            self,
            "component_document_ids",
            component_ids,
        )

        object.__setattr__(
            self,
            "current_document_ids",
            current_ids,
        )

        object.__setattr__(
            self,
            "conflict_id",
            _conflict_id(
                as_of_date=as_of_date,
                code=self.code,
                component_document_ids=(
                    component_ids
                ),
                current_document_ids=(
                    current_ids
                ),
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "conflict_id": (
                self.conflict_id
            ),
            "as_of_date": (
                self.as_of_date.isoformat()
            ),
            "code": (
                self.code.value
            ),
            "component_document_ids": list(
                self.component_document_ids
            ),
            "current_document_ids": list(
                self.current_document_ids
            ),
        }


@dataclass(frozen=True)
class TemporalConflictReport:
    """Auditable temporal conflict assessment."""

    as_of_date: date

    resolutions: tuple[
        TemporalResolution,
        ...
    ]

    conflicts: tuple[
        TemporalConflict,
        ...
    ]

    detector_version: str = (
        TEMPORAL_CONFLICT_VERSION
    )

    def __post_init__(self) -> None:
        _validate_date(
            self.as_of_date
        )

        resolutions = tuple(
            self.resolutions
        )

        conflicts = tuple(
            self.conflicts
        )

        for resolution in resolutions:
            if not isinstance(
                resolution,
                TemporalResolution,
            ):
                raise TypeError(
                    "resolutions must contain only "
                    "TemporalResolution values."
                )

        for conflict in conflicts:
            if not isinstance(
                conflict,
                TemporalConflict,
            ):
                raise TypeError(
                    "conflicts must contain only "
                    "TemporalConflict values."
                )

        object.__setattr__(
            self,
            "resolutions",
            resolutions,
        )

        object.__setattr__(
            self,
            "conflicts",
            conflicts,
        )

    @property
    def conflict_count(
        self,
    ) -> int:
        return len(
            self.conflicts
        )

    @property
    def has_conflict(
        self,
    ) -> bool:
        return bool(
            self.conflicts
        )

    @property
    def insufficient_document_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                resolution.document_id
                for resolution
                in self.resolutions
                if resolution.state
                is TemporalPolicyState
                .INSUFFICIENT
            )
        )

    @property
    def decision(
        self,
    ) -> TemporalConflictDecision:
        if self.has_conflict:
            return (
                TemporalConflictDecision
                .CONFLICTING
            )

        if self.insufficient_document_ids:
            return (
                TemporalConflictDecision
                .INSUFFICIENT
            )

        return (
            TemporalConflictDecision.CLEAR
        )

    @property
    def should_abstain(
        self,
    ) -> bool:
        return (
            self.decision
            is not TemporalConflictDecision.CLEAR
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without policy/evidence text."""

        return {
            "detector_version": (
                self.detector_version
            ),
            "as_of_date": (
                self.as_of_date.isoformat()
            ),
            "decision": (
                self.decision.value
            ),
            "should_abstain": (
                self.should_abstain
            ),
            "resolution_count": len(
                self.resolutions
            ),
            "conflict_count": (
                self.conflict_count
            ),
            "insufficient_document_ids": list(
                self.insufficient_document_ids
            ),
            "conflicts": [
                conflict.to_dict()
                for conflict
                in self.conflicts
            ],
        }


def _connected_components(
    graph: SupersessionGraph,
) -> tuple[
    tuple[str, ...],
    ...
]:
    """Return deterministic undirected supersession components."""

    remaining = set(
        graph.document_ids
    )

    components = []

    while remaining:
        start = min(
            remaining
        )

        pending = [
            start
        ]

        component = set()

        while pending:
            current = pending.pop()

            if current in component:
                continue

            component.add(
                current
            )

            neighbors = set(
                graph.supersedes(
                    current
                )
            )

            neighbors.update(
                graph.superseded_by(
                    current
                )
            )

            pending.extend(
                sorted(
                    neighbors
                    - component
                )
            )

        remaining.difference_update(
            component
        )

        components.append(
            tuple(
                sorted(
                    component
                )
            )
        )

    return tuple(
        sorted(
            components
        )
    )


class TemporalConflictDetector:
    """Detect structural temporal conflicts in supersession families."""

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
        self._resolver = (
            TemporalResolver(
                graph
            )
        )

    @property
    def graph(
        self,
    ) -> SupersessionGraph:
        return self._graph

    def detect(
        self,
        *,
        as_of_date: date,
    ) -> TemporalConflictReport:
        """Detect deterministic structural conflicts as of one date."""

        as_of_date = _validate_date(
            as_of_date
        )

        resolutions = (
            self._resolver.resolve_all(
                as_of_date=as_of_date
            )
        )

        resolution_map = {
            resolution.document_id: (
                resolution
            )
            for resolution
            in resolutions
        }

        cycle_ids = set(
            self._graph.cycle_document_ids
        )

        conflicts = []

        for component in (
            _connected_components(
                self._graph
            )
        ):
            if len(
                component
            ) < 2:
                continue

            cycle_members = tuple(
                document_id
                for document_id
                in component
                if document_id
                in cycle_ids
            )

            if cycle_members:
                conflicts.append(
                    TemporalConflict(
                        as_of_date=as_of_date,
                        code=(
                            TemporalConflictCode
                            .SUPERSESSION_CYCLE
                        ),
                        component_document_ids=(
                            component
                        ),
                    )
                )

                continue

            current_ids = tuple(
                document_id
                for document_id
                in component
                if (
                    resolution_map[
                        document_id
                    ].state
                    is TemporalPolicyState
                    .CURRENT
                )
            )

            if len(
                current_ids
            ) > 1:
                conflicts.append(
                    TemporalConflict(
                        as_of_date=as_of_date,
                        code=(
                            TemporalConflictCode
                            .MULTIPLE_CURRENT_SUCCESSORS
                        ),
                        component_document_ids=(
                            component
                        ),
                        current_document_ids=(
                            current_ids
                        ),
                    )
                )

        return TemporalConflictReport(
            as_of_date=as_of_date,
            resolutions=resolutions,
            conflicts=tuple(
                conflicts
            ),
        )


def detect_temporal_conflicts(
    graph: SupersessionGraph,
    *,
    as_of_date: date,
) -> TemporalConflictReport:
    """Convenience function for temporal conflict detection."""

    return TemporalConflictDetector(
        graph
    ).detect(
        as_of_date=as_of_date
    )
