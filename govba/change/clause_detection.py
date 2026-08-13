"""Deterministic clause/chunk change detection for GovBA-GAR."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Iterable

from govba.change.contract import (
    PolicyChangeFinding,
    PolicyChangeImpact,
    PolicyChangeReport,
    PolicyChangeType,
    build_policy_change_report,
)
from govba.change.version_matching import (
    DocumentVersionMatchResult,
    build_change_request_from_match,
)
from govba.rag.evidence import (
    EvidenceChunk,
)


CLAUSE_CHANGE_DETECTION_VERSION = (
    "govba-clause-change-detection-v1"
)

CLAUSE_CHANGE_DETECTION_ALGORITHM = (
    "deterministic-clause-diff-v1"
)

DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD = 0.55


class ClauseChangeReason(
    str,
    Enum,
):
    """Stable reason taxonomy for deterministic clause matching."""

    EXACT_CONTENT_MATCH = (
        "exact_content_match"
    )

    SIMILARITY_MATCH = (
        "similarity_match"
    )

    BASELINE_ONLY = (
        "baseline_only"
    )

    CANDIDATE_ONLY = (
        "candidate_only"
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


def _similarity_threshold(
    value: float,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(
            value
        )
        or not (
            0.0 < value <= 1.0
        )
    ):
        raise ValueError(
            "similarity_threshold must be "
            "greater than 0 and at most 1."
        )

    return float(
        value
    )


def _normalize_similarity_text(
    text: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    return _WHITESPACE_RE.sub(
        " ",
        value,
    ).strip()


def clause_similarity(
    baseline_text: str,
    candidate_text: str,
) -> float:
    """Return deterministic Unicode-aware structural similarity."""

    baseline = _normalize_similarity_text(
        _required_text(
            baseline_text,
            "baseline_text",
        )
    )

    candidate = _normalize_similarity_text(
        _required_text(
            candidate_text,
            "candidate_text",
        )
    )

    return float(
        SequenceMatcher(
            None,
            baseline,
            candidate,
            autojunk=False,
        ).ratio()
    )


def _prepare_chunks(
    values: Iterable[
        EvidenceChunk
    ],
    *,
    document_id: str,
    field_name: str,
) -> tuple[
    EvidenceChunk,
    ...
]:
    if isinstance(
        values,
        (str, bytes),
    ):
        raise TypeError(
            f"{field_name} must contain "
            "EvidenceChunk values."
        )

    try:
        chunks = tuple(
            values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    for chunk in chunks:
        if not isinstance(
            chunk,
            EvidenceChunk,
        ):
            raise TypeError(
                f"{field_name} must contain only "
                "EvidenceChunk values."
            )

        if (
            chunk.document_id
            != document_id
        ):
            raise ValueError(
                f"{field_name} contains a chunk "
                "from the wrong document."
            )

    chunk_ids = tuple(
        chunk.chunk_id
        for chunk in chunks
    )

    if (
        len(set(chunk_ids))
        != len(chunk_ids)
    ):
        raise ValueError(
            f"{field_name} contains "
            "duplicate chunk IDs."
        )

    ordered = tuple(
        sorted(
            chunks,
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.chunk_id,
            ),
        )
    )

    indexes = tuple(
        chunk.chunk_index
        for chunk in ordered
    )

    if indexes != tuple(
        range(
            len(ordered)
        )
    ):
        raise ValueError(
            f"{field_name} chunk indexes must "
            "be contiguous from zero."
        )

    return ordered


def _match_identity(
    *,
    change_type: PolicyChangeType,
    reason: ClauseChangeReason,
    baseline_chunk_id: str,
    candidate_chunk_id: str,
    similarity: float,
) -> str:
    payload = "\x1f".join(
        (
            CLAUSE_CHANGE_DETECTION_VERSION,
            change_type.value,
            reason.value,
            baseline_chunk_id,
            candidate_chunk_id,
            f"{similarity:.12f}",
        )
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class ClauseChangeMatch:
    """One privacy-safe chunk correspondence or structural change."""

    change_type: PolicyChangeType

    reason: ClauseChangeReason

    similarity: float

    baseline_chunk_id: str = ""

    candidate_chunk_id: str = ""

    version: str = (
        CLAUSE_CHANGE_DETECTION_VERSION
    )

    match_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.change_type,
            PolicyChangeType,
        ):
            raise TypeError(
                "change_type must be a "
                "PolicyChangeType."
            )

        if not isinstance(
            self.reason,
            ClauseChangeReason,
        ):
            raise TypeError(
                "reason must be a "
                "ClauseChangeReason."
            )

        if (
            isinstance(
                self.similarity,
                bool,
            )
            or not isinstance(
                self.similarity,
                (int, float),
            )
            or not math.isfinite(
                self.similarity
            )
            or not (
                0.0 <= self.similarity <= 1.0
            )
        ):
            raise ValueError(
                "similarity must be between "
                "0 and 1."
            )

        baseline_chunk_id = (
            self.baseline_chunk_id.strip()
        )

        candidate_chunk_id = (
            self.candidate_chunk_id.strip()
        )

        if (
            self.change_type
            is PolicyChangeType.ADDED
        ):
            if (
                baseline_chunk_id
                or not candidate_chunk_id
            ):
                raise ValueError(
                    "ADDED requires only a "
                    "candidate chunk."
                )

        elif (
            self.change_type
            is PolicyChangeType.REMOVED
        ):
            if (
                not baseline_chunk_id
                or candidate_chunk_id
            ):
                raise ValueError(
                    "REMOVED requires only a "
                    "baseline chunk."
                )

        else:
            if (
                not baseline_chunk_id
                or not candidate_chunk_id
            ):
                raise ValueError(
                    "MODIFIED and UNCHANGED "
                    "require both chunks."
                )

        if (
            self.change_type
            is PolicyChangeType.UNCHANGED
            and self.reason
            is not ClauseChangeReason.EXACT_CONTENT_MATCH
        ):
            raise ValueError(
                "UNCHANGED requires an "
                "exact-content reason."
            )

        if (
            self.change_type
            is PolicyChangeType.MODIFIED
            and self.reason
            is not ClauseChangeReason.SIMILARITY_MATCH
        ):
            raise ValueError(
                "MODIFIED requires a "
                "similarity-match reason."
            )

        if (
            self.change_type
            is PolicyChangeType.ADDED
            and self.reason
            is not ClauseChangeReason.CANDIDATE_ONLY
        ):
            raise ValueError(
                "ADDED requires candidate-only reason."
            )

        if (
            self.change_type
            is PolicyChangeType.REMOVED
            and self.reason
            is not ClauseChangeReason.BASELINE_ONLY
        ):
            raise ValueError(
                "REMOVED requires baseline-only reason."
            )

        similarity = float(
            self.similarity
        )

        match_id = _match_identity(
            change_type=(
                self.change_type
            ),
            reason=(
                self.reason
            ),
            baseline_chunk_id=(
                baseline_chunk_id
            ),
            candidate_chunk_id=(
                candidate_chunk_id
            ),
            similarity=(
                similarity
            ),
        )

        object.__setattr__(
            self,
            "baseline_chunk_id",
            baseline_chunk_id,
        )

        object.__setattr__(
            self,
            "candidate_chunk_id",
            candidate_chunk_id,
        )

        object.__setattr__(
            self,
            "similarity",
            similarity,
        )

        object.__setattr__(
            self,
            "match_id",
            match_id,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "match_id": (
                self.match_id
            ),
            "change_type": (
                self.change_type.value
            ),
            "reason": (
                self.reason.value
            ),
            "similarity": (
                self.similarity
            ),
            "baseline_chunk_id": (
                self.baseline_chunk_id
                or None
            ),
            "candidate_chunk_id": (
                self.candidate_chunk_id
                or None
            ),
        }


@dataclass(frozen=True)
class ClauseChangeDetectionResult:
    """Complete deterministic structural change-detection result."""

    version_match_id: str

    report: PolicyChangeReport

    matches: tuple[
        ClauseChangeMatch,
        ...
    ]

    similarity_threshold: float

    version: str = (
        CLAUSE_CHANGE_DETECTION_VERSION
    )

    detection_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        version_match_id = _required_text(
            self.version_match_id,
            "version_match_id",
        )

        if not isinstance(
            self.report,
            PolicyChangeReport,
        ):
            raise TypeError(
                "report must be a "
                "PolicyChangeReport."
            )

        try:
            matches = tuple(
                self.matches
            )
        except TypeError as exc:
            raise TypeError(
                "matches must be iterable."
            ) from exc

        for match in matches:
            if not isinstance(
                match,
                ClauseChangeMatch,
            ):
                raise TypeError(
                    "matches must contain only "
                    "ClauseChangeMatch values."
                )

        match_ids = tuple(
            match.match_id
            for match in matches
        )

        if (
            len(set(match_ids))
            != len(match_ids)
        ):
            raise ValueError(
                "Clause-change matches "
                "must be unique."
            )

        threshold = (
            _similarity_threshold(
                self.similarity_threshold
            )
        )

        payload = "\x1f".join(
            (
                self.version,
                version_match_id,
                self.report.report_id,
                f"{threshold:.12f}",
                ",".join(
                    sorted(
                        match_ids
                    )
                ),
            )
        )

        detection_id = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

        object.__setattr__(
            self,
            "version_match_id",
            version_match_id,
        )

        object.__setattr__(
            self,
            "matches",
            matches,
        )

        object.__setattr__(
            self,
            "similarity_threshold",
            threshold,
        )

        object.__setattr__(
            self,
            "detection_id",
            detection_id,
        )

    @property
    def changed(
        self,
    ) -> bool:
        return (
            self.report.changed
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without clause text."""

        return {
            "version": (
                self.version
            ),
            "detection_id": (
                self.detection_id
            ),
            "version_match_id": (
                self.version_match_id
            ),
            "similarity_threshold": (
                self.similarity_threshold
            ),
            "changed": (
                self.changed
            ),
            "report": (
                self.report.to_dict()
            ),
            "matches": [
                match.to_dict()
                for match
                in self.matches
            ],
        }


def _finding_from_match(
    match: ClauseChangeMatch,
    *,
    baseline_map: dict[
        str,
        EvidenceChunk,
    ],
    candidate_map: dict[
        str,
        EvidenceChunk,
    ],
) -> PolicyChangeFinding:
    baseline_chunk = (
        baseline_map.get(
            match.baseline_chunk_id
        )
    )

    candidate_chunk = (
        candidate_map.get(
            match.candidate_chunk_id
        )
    )

    return PolicyChangeFinding(
        change_type=(
            match.change_type
        ),
        impact=(
            PolicyChangeImpact.INFORMATIONAL
        ),
        baseline_chunk_id=(
            match.baseline_chunk_id
        ),
        candidate_chunk_id=(
            match.candidate_chunk_id
        ),
        baseline_content_hash=(
            baseline_chunk.content_hash
            if baseline_chunk
            else ""
        ),
        candidate_content_hash=(
            candidate_chunk.content_hash
            if candidate_chunk
            else ""
        ),
        reason_code=(
            match.reason.value
        ),
    )


def detect_clause_changes(
    version_match: DocumentVersionMatchResult,
    *,
    baseline_chunks: Iterable[
        EvidenceChunk
    ],
    candidate_chunks: Iterable[
        EvidenceChunk
    ],
    trace_id: str,
    similarity_threshold: float = (
        DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD
    ),
) -> ClauseChangeDetectionResult:
    """Compare matched authoritative evidence deterministically."""

    if not isinstance(
        version_match,
        DocumentVersionMatchResult,
    ):
        raise TypeError(
            "version_match must be a "
            "DocumentVersionMatchResult."
        )

    if not version_match.allowed:
        raise ValueError(
            "Document versions must be approved "
            "before automatic change detection."
        )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    threshold = _similarity_threshold(
        similarity_threshold
    )

    baseline = _prepare_chunks(
        baseline_chunks,
        document_id=(
            version_match
            .baseline_source
            .document_id
        ),
        field_name="baseline_chunks",
    )

    candidate = _prepare_chunks(
        candidate_chunks,
        document_id=(
            version_match
            .candidate_source
            .document_id
        ),
        field_name="candidate_chunks",
    )

    request = (
        build_change_request_from_match(
            version_match,
            trace_id=trace,
        )
    )

    if version_match.same_content:
        report = build_policy_change_report(
            request,
            findings=(),
            algorithm=(
                CLAUSE_CHANGE_DETECTION_ALGORITHM
                + ":same-content"
            ),
        )

        return ClauseChangeDetectionResult(
            version_match_id=(
                version_match.match_id
            ),
            report=report,
            matches=(),
            similarity_threshold=threshold,
        )

    baseline_map = {
        chunk.chunk_id: chunk
        for chunk in baseline
    }

    candidate_map = {
        chunk.chunk_id: chunk
        for chunk in candidate
    }

    unmatched_baseline = {
        chunk.chunk_id: chunk
        for chunk in baseline
    }

    unmatched_candidate = {
        chunk.chunk_id: chunk
        for chunk in candidate
    }

    matches = []

    candidate_by_hash: dict[
        str,
        list[EvidenceChunk],
    ] = {}

    for chunk in candidate:
        candidate_by_hash.setdefault(
            chunk.content_hash,
            [],
        ).append(
            chunk
        )

    for values in (
        candidate_by_hash.values()
    ):
        values.sort(
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.chunk_id,
            )
        )

    for baseline_chunk in baseline:
        candidates = candidate_by_hash.get(
            baseline_chunk.content_hash,
            [],
        )

        while (
            candidates
            and candidates[0].chunk_id
            not in unmatched_candidate
        ):
            candidates.pop(
                0
            )

        if not candidates:
            continue

        candidate_chunk = candidates.pop(
            0
        )

        unmatched_baseline.pop(
            baseline_chunk.chunk_id,
            None,
        )

        unmatched_candidate.pop(
            candidate_chunk.chunk_id,
            None,
        )

        matches.append(
            ClauseChangeMatch(
                change_type=(
                    PolicyChangeType.UNCHANGED
                ),
                reason=(
                    ClauseChangeReason
                    .EXACT_CONTENT_MATCH
                ),
                similarity=1.0,
                baseline_chunk_id=(
                    baseline_chunk.chunk_id
                ),
                candidate_chunk_id=(
                    candidate_chunk.chunk_id
                ),
            )
        )

    similarity_candidates = []

    for baseline_chunk in (
        unmatched_baseline.values()
    ):
        for candidate_chunk in (
            unmatched_candidate.values()
        ):
            similarity = clause_similarity(
                baseline_chunk.text,
                candidate_chunk.text,
            )

            if similarity < threshold:
                continue

            similarity_candidates.append(
                (
                    similarity,
                    abs(
                        baseline_chunk.chunk_index
                        - candidate_chunk.chunk_index
                    ),
                    baseline_chunk.chunk_index,
                    candidate_chunk.chunk_index,
                    baseline_chunk.chunk_id,
                    candidate_chunk.chunk_id,
                )
            )

    similarity_candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
            item[3],
            item[4],
            item[5],
        )
    )

    for (
        similarity,
        _distance,
        _baseline_index,
        _candidate_index,
        baseline_chunk_id,
        candidate_chunk_id,
    ) in similarity_candidates:
        if (
            baseline_chunk_id
            not in unmatched_baseline
            or candidate_chunk_id
            not in unmatched_candidate
        ):
            continue

        unmatched_baseline.pop(
            baseline_chunk_id
        )

        unmatched_candidate.pop(
            candidate_chunk_id
        )

        matches.append(
            ClauseChangeMatch(
                change_type=(
                    PolicyChangeType.MODIFIED
                ),
                reason=(
                    ClauseChangeReason
                    .SIMILARITY_MATCH
                ),
                similarity=(
                    similarity
                ),
                baseline_chunk_id=(
                    baseline_chunk_id
                ),
                candidate_chunk_id=(
                    candidate_chunk_id
                ),
            )
        )

    for chunk in sorted(
        unmatched_baseline.values(),
        key=lambda value: (
            value.chunk_index,
            value.chunk_id,
        ),
    ):
        matches.append(
            ClauseChangeMatch(
                change_type=(
                    PolicyChangeType.REMOVED
                ),
                reason=(
                    ClauseChangeReason
                    .BASELINE_ONLY
                ),
                similarity=0.0,
                baseline_chunk_id=(
                    chunk.chunk_id
                ),
            )
        )

    for chunk in sorted(
        unmatched_candidate.values(),
        key=lambda value: (
            value.chunk_index,
            value.chunk_id,
        ),
    ):
        matches.append(
            ClauseChangeMatch(
                change_type=(
                    PolicyChangeType.ADDED
                ),
                reason=(
                    ClauseChangeReason
                    .CANDIDATE_ONLY
                ),
                similarity=0.0,
                candidate_chunk_id=(
                    chunk.chunk_id
                ),
            )
        )

    order = {
        PolicyChangeType.UNCHANGED: 0,
        PolicyChangeType.MODIFIED: 1,
        PolicyChangeType.REMOVED: 2,
        PolicyChangeType.ADDED: 3,
    }

    matches = tuple(
        sorted(
            matches,
            key=lambda match: (
                order[
                    match.change_type
                ],
                (
                    baseline_map[
                        match.baseline_chunk_id
                    ].chunk_index
                    if match.baseline_chunk_id
                    else math.inf
                ),
                (
                    candidate_map[
                        match.candidate_chunk_id
                    ].chunk_index
                    if match.candidate_chunk_id
                    else math.inf
                ),
                match.match_id,
            ),
        )
    )

    findings = tuple(
        _finding_from_match(
            match,
            baseline_map=baseline_map,
            candidate_map=candidate_map,
        )
        for match in matches
    )

    report = build_policy_change_report(
        request,
        findings=findings,
        algorithm=(
            CLAUSE_CHANGE_DETECTION_ALGORITHM
        ),
    )

    return ClauseChangeDetectionResult(
        version_match_id=(
            version_match.match_id
        ),
        report=report,
        matches=matches,
        similarity_threshold=threshold,
    )
