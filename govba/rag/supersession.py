"""Deterministic policy supersession graph for GovBA-GAR."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from govba.rag.models import AuthoritativeSource


SUPERSESSION_GRAPH_VERSION = (
    "govba-supersession-graph-v1"
)


class SupersessionDeclaration(
    str,
    Enum,
):
    """Source metadata field declaring a supersession relationship."""

    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"


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


def _edge_id(
    newer_document_id: str,
    older_document_id: str,
    declarations: tuple[
        SupersessionDeclaration,
        ...
    ],
) -> str:
    payload = "\x1f".join(
        (
            SUPERSESSION_GRAPH_VERSION,
            newer_document_id,
            older_document_id,
            ",".join(
                declaration.value
                for declaration
                in declarations
            ),
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


@dataclass(frozen=True)
class SupersessionEdge:
    """One directed newer-document -> older-document relationship."""

    newer_document_id: str
    older_document_id: str
    declarations: tuple[
        SupersessionDeclaration,
        ...
    ]

    edge_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        newer = _required_text(
            self.newer_document_id,
            "newer_document_id",
        )

        older = _required_text(
            self.older_document_id,
            "older_document_id",
        )

        if newer == older:
            raise ValueError(
                "A document cannot supersede itself."
            )

        try:
            declarations = tuple(
                self.declarations
            )
        except TypeError as exc:
            raise TypeError(
                "declarations must be iterable."
            ) from exc

        if not declarations:
            raise ValueError(
                "At least one supersession "
                "declaration is required."
            )

        for declaration in declarations:
            if not isinstance(
                declaration,
                SupersessionDeclaration,
            ):
                raise TypeError(
                    "declarations must contain only "
                    "SupersessionDeclaration values."
                )

        if (
            len(set(declarations))
            != len(declarations)
        ):
            raise ValueError(
                "declarations must not contain duplicates."
            )

        declarations = tuple(
            sorted(
                declarations,
                key=lambda item: item.value,
            )
        )

        object.__setattr__(
            self,
            "newer_document_id",
            newer,
        )

        object.__setattr__(
            self,
            "older_document_id",
            older,
        )

        object.__setattr__(
            self,
            "declarations",
            declarations,
        )

        object.__setattr__(
            self,
            "edge_id",
            _edge_id(
                newer,
                older,
                declarations,
            ),
        )


@dataclass(frozen=True)
class UnresolvedSupersessionReference:
    """Relationship pointing to a document outside the controlled corpus."""

    owner_document_id: str
    related_document_id: str
    declaration: SupersessionDeclaration

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owner_document_id",
            _required_text(
                self.owner_document_id,
                "owner_document_id",
            ),
        )

        object.__setattr__(
            self,
            "related_document_id",
            _required_text(
                self.related_document_id,
                "related_document_id",
            ),
        )

        if not isinstance(
            self.declaration,
            SupersessionDeclaration,
        ):
            raise TypeError(
                "declaration must be a "
                "SupersessionDeclaration."
            )


def _find_cycle_nodes(
    adjacency: dict[
        str,
        set[str],
    ],
) -> tuple[str, ...]:
    state = {
        document_id: 0
        for document_id
        in adjacency
    }

    stack: list[str] = []
    positions: dict[
        str,
        int,
    ] = {}

    cycle_nodes = set()

    def visit(
        document_id: str,
    ) -> None:
        state[
            document_id
        ] = 1

        positions[
            document_id
        ] = len(
            stack
        )

        stack.append(
            document_id
        )

        for related_id in sorted(
            adjacency[
                document_id
            ]
        ):
            related_state = state[
                related_id
            ]

            if related_state == 0:
                visit(
                    related_id
                )

            elif related_state == 1:
                start = positions[
                    related_id
                ]

                cycle_nodes.update(
                    stack[
                        start:
                    ]
                )

        stack.pop()

        positions.pop(
            document_id,
            None,
        )

        state[
            document_id
        ] = 2

    for document_id in sorted(
        adjacency
    ):
        if (
            state[
                document_id
            ]
            == 0
        ):
            visit(
                document_id
            )

    return tuple(
        sorted(
            cycle_nodes
        )
    )


class SupersessionGraph:
    """Governance-aware directed graph of document supersession."""

    def __init__(
        self,
        sources: Iterable[
            AuthoritativeSource
        ],
    ) -> None:
        try:
            source_values = tuple(
                sources
            )
        except TypeError as exc:
            raise TypeError(
                "sources must be iterable."
            ) from exc

        source_map: dict[
            str,
            AuthoritativeSource,
        ] = {}

        for source in source_values:
            if not isinstance(
                source,
                AuthoritativeSource,
            ):
                raise TypeError(
                    "sources must contain only "
                    "AuthoritativeSource values."
                )

            if (
                source.document_id
                in source_map
            ):
                raise ValueError(
                    "Source document IDs must be unique."
                )

            source_map[
                source.document_id
            ] = source

        relationship_map: dict[
            tuple[str, str],
            set[
                SupersessionDeclaration
            ],
        ] = {}

        unresolved = []

        for source in source_values:
            for older_id in source.supersedes:
                if older_id not in source_map:
                    unresolved.append(
                        UnresolvedSupersessionReference(
                            owner_document_id=(
                                source.document_id
                            ),
                            related_document_id=(
                                older_id
                            ),
                            declaration=(
                                SupersessionDeclaration
                                .SUPERSEDES
                            ),
                        )
                    )

                    continue

                relationship_map.setdefault(
                    (
                        source.document_id,
                        older_id,
                    ),
                    set(),
                ).add(
                    SupersessionDeclaration
                    .SUPERSEDES
                )

            for newer_id in source.superseded_by:
                if newer_id not in source_map:
                    unresolved.append(
                        UnresolvedSupersessionReference(
                            owner_document_id=(
                                source.document_id
                            ),
                            related_document_id=(
                                newer_id
                            ),
                            declaration=(
                                SupersessionDeclaration
                                .SUPERSEDED_BY
                            ),
                        )
                    )

                    continue

                relationship_map.setdefault(
                    (
                        newer_id,
                        source.document_id,
                    ),
                    set(),
                ).add(
                    SupersessionDeclaration
                    .SUPERSEDED_BY
                )

        edges = tuple(
            SupersessionEdge(
                newer_document_id=newer,
                older_document_id=older,
                declarations=tuple(
                    declarations
                ),
            )
            for (
                newer,
                older,
            ),
            declarations
            in sorted(
                relationship_map.items()
            )
        )

        adjacency = {
            document_id: set()
            for document_id
            in source_map
        }

        reverse_adjacency = {
            document_id: set()
            for document_id
            in source_map
        }

        for edge in edges:
            adjacency[
                edge.newer_document_id
            ].add(
                edge.older_document_id
            )

            reverse_adjacency[
                edge.older_document_id
            ].add(
                edge.newer_document_id
            )

        self._sources = source_map
        self._edges = edges
        self._unresolved = tuple(
            sorted(
                unresolved,
                key=lambda item: (
                    item.owner_document_id,
                    item.related_document_id,
                    item.declaration.value,
                ),
            )
        )
        self._adjacency = adjacency
        self._reverse_adjacency = (
            reverse_adjacency
        )
        self._cycle_document_ids = (
            _find_cycle_nodes(
                adjacency
            )
        )

    @property
    def source_count(
        self,
    ) -> int:
        return len(
            self._sources
        )

    @property
    def edge_count(
        self,
    ) -> int:
        return len(
            self._edges
        )

    @property
    def unresolved_count(
        self,
    ) -> int:
        return len(
            self._unresolved
        )

    @property
    def document_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._sources
            )
        )

    @property
    def edges(
        self,
    ) -> tuple[
        SupersessionEdge,
        ...
    ]:
        return self._edges

    @property
    def unresolved_references(
        self,
    ) -> tuple[
        UnresolvedSupersessionReference,
        ...
    ]:
        return self._unresolved

    @property
    def cycle_document_ids(
        self,
    ) -> tuple[str, ...]:
        return self._cycle_document_ids

    @property
    def has_cycle(
        self,
    ) -> bool:
        return bool(
            self._cycle_document_ids
        )

    def source(
        self,
        document_id: str,
    ) -> AuthoritativeSource:
        document_id = _required_text(
            document_id,
            "document_id",
        )

        try:
            return self._sources[
                document_id
            ]
        except KeyError:
            raise KeyError(
                document_id
            ) from None

    def supersedes(
        self,
        document_id: str,
    ) -> tuple[str, ...]:
        self.source(
            document_id
        )

        return tuple(
            sorted(
                self._adjacency[
                    document_id
                ]
            )
        )

    def superseded_by(
        self,
        document_id: str,
    ) -> tuple[str, ...]:
        self.source(
            document_id
        )

        return tuple(
            sorted(
                self._reverse_adjacency[
                    document_id
                ]
            )
        )

    def all_superseded_documents(
        self,
        document_id: str,
    ) -> tuple[str, ...]:
        """Return all transitively superseded known documents."""

        self.source(
            document_id
        )

        visited = set()
        pending = list(
            self._adjacency[
                document_id
            ]
        )

        while pending:
            current = pending.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            pending.extend(
                self._adjacency[
                    current
                ]
            )

        visited.discard(
            document_id
        )

        return tuple(
            sorted(
                visited
            )
        )

    def all_superseding_documents(
        self,
        document_id: str,
    ) -> tuple[str, ...]:
        """Return all known documents that supersede this document."""

        self.source(
            document_id
        )

        visited = set()
        pending = list(
            self._reverse_adjacency[
                document_id
            ]
        )

        while pending:
            current = pending.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            pending.extend(
                self._reverse_adjacency[
                    current
                ]
            )

        visited.discard(
            document_id
        )

        return tuple(
            sorted(
                visited
            )
        )

    def has_direct_edge(
        self,
        newer_document_id: str,
        older_document_id: str,
    ) -> bool:
        self.source(
            newer_document_id
        )

        self.source(
            older_document_id
        )

        return (
            older_document_id
            in self._adjacency[
                newer_document_id
            ]
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "graph_version": (
                SUPERSESSION_GRAPH_VERSION
            ),
            "source_count": (
                self.source_count
            ),
            "edge_count": (
                self.edge_count
            ),
            "unresolved_count": (
                self.unresolved_count
            ),
            "has_cycle": (
                self.has_cycle
            ),
            "cycle_document_ids": list(
                self.cycle_document_ids
            ),
            "edges": [
                {
                    "edge_id": (
                        edge.edge_id
                    ),
                    "newer_document_id": (
                        edge.newer_document_id
                    ),
                    "older_document_id": (
                        edge.older_document_id
                    ),
                    "declarations": [
                        declaration.value
                        for declaration
                        in edge.declarations
                    ],
                }
                for edge in self.edges
            ],
            "unresolved_references": [
                {
                    "owner_document_id": (
                        reference.owner_document_id
                    ),
                    "related_document_id": (
                        reference.related_document_id
                    ),
                    "declaration": (
                        reference.declaration.value
                    ),
                }
                for reference
                in self.unresolved_references
            ],
        }
