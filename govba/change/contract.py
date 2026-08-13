"""Provider-independent policy change-intelligence contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from govba.rag.models import (
    AuthoritativeSource,
)


POLICY_CHANGE_CONTRACT_VERSION = (
    "govba-policy-change-contract-v1"
)


_SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


class PolicyChangeType(
    str,
    Enum,
):
    """Structural type of one detected policy change."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class PolicyChangeImpact(
    str,
    Enum,
):
    """Governance significance assigned to one change."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_IMPACT_PRIORITY = {
    PolicyChangeImpact.INFORMATIONAL: 0,
    PolicyChangeImpact.LOW: 1,
    PolicyChangeImpact.MEDIUM: 2,
    PolicyChangeImpact.HIGH: 3,
    PolicyChangeImpact.CRITICAL: 4,
}


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


def _optional_text(
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

    return value.strip()


def _optional_hash(
    value: str,
    field_name: str,
) -> str:
    value = _optional_text(
        value,
        field_name,
    ).lower()

    if (
        value
        and not _SHA256_RE.fullmatch(
            value
        )
    ):
        raise ValueError(
            f"{field_name} must be a "
            "SHA-256 hexadecimal digest."
        )

    return value


def _source_identity(
    source: AuthoritativeSource,
) -> str:
    payload = "\x1f".join(
        (
            source.document_id,
            source.content_hash,
            source.version,
            (
                source.effective_from.isoformat()
                if source.effective_from
                else ""
            ),
            (
                source.effective_until.isoformat()
                if source.effective_until
                else ""
            ),
            source.status.value,
        )
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class PolicyChangeRequest:
    """Request to compare two authoritative policy-document versions."""

    baseline_source: (
        AuthoritativeSource
    )

    candidate_source: (
        AuthoritativeSource
    )

    trace_id: str

    version: str = (
        POLICY_CHANGE_CONTRACT_VERSION
    )

    baseline_identity: str = field(
        init=False
    )

    candidate_identity: str = field(
        init=False
    )

    request_id: str = field(
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

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        baseline_identity = (
            _source_identity(
                self.baseline_source
            )
        )

        candidate_identity = (
            _source_identity(
                self.candidate_source
            )
        )

        payload = "\x1f".join(
            (
                self.version,
                trace_id,
                baseline_identity,
                candidate_identity,
            )
        )

        request_id = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "baseline_identity",
            baseline_identity,
        )

        object.__setattr__(
            self,
            "candidate_identity",
            candidate_identity,
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

    @property
    def same_content(
        self,
    ) -> bool:
        baseline_hash = (
            self.baseline_source
            .content_hash
        )

        candidate_hash = (
            self.candidate_source
            .content_hash
        )

        return bool(
            baseline_hash
            and candidate_hash
            and baseline_hash.lower()
            == candidate_hash.lower()
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize without document or web content."""

        return {
            "version": (
                self.version
            ),
            "request_id": (
                self.request_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "baseline": {
                "document_id": (
                    self.baseline_source
                    .document_id
                ),
                "identity": (
                    self.baseline_identity
                ),
                "content_hash": (
                    self.baseline_source
                    .content_hash
                    or None
                ),
                "version": (
                    self.baseline_source
                    .version
                    or None
                ),
                "status": (
                    self.baseline_source
                    .status.value
                ),
            },
            "candidate": {
                "document_id": (
                    self.candidate_source
                    .document_id
                ),
                "identity": (
                    self.candidate_identity
                ),
                "content_hash": (
                    self.candidate_source
                    .content_hash
                    or None
                ),
                "version": (
                    self.candidate_source
                    .version
                    or None
                ),
                "status": (
                    self.candidate_source
                    .status.value
                ),
            },
            "same_content": (
                self.same_content
            ),
        }


@dataclass(frozen=True)
class PolicyChangeFinding:
    """One deterministic structural change without storing raw clause text."""

    change_type: PolicyChangeType

    impact: PolicyChangeImpact

    baseline_chunk_id: str = ""

    candidate_chunk_id: str = ""

    baseline_content_hash: str = ""

    candidate_content_hash: str = ""

    reason_code: str = ""

    version: str = (
        POLICY_CHANGE_CONTRACT_VERSION
    )

    finding_id: str = field(
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
            self.impact,
            PolicyChangeImpact,
        ):
            raise TypeError(
                "impact must be a "
                "PolicyChangeImpact."
            )

        baseline_chunk_id = (
            _optional_text(
                self.baseline_chunk_id,
                "baseline_chunk_id",
            )
        )

        candidate_chunk_id = (
            _optional_text(
                self.candidate_chunk_id,
                "candidate_chunk_id",
            )
        )

        baseline_hash = (
            _optional_hash(
                self.baseline_content_hash,
                "baseline_content_hash",
            )
        )

        candidate_hash = (
            _optional_hash(
                self.candidate_content_hash,
                "candidate_content_hash",
            )
        )

        reason_code = (
            _optional_text(
                self.reason_code,
                "reason_code",
            )
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
                    "candidate_chunk_id."
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
                    "baseline_chunk_id."
                )

        elif self.change_type in {
            PolicyChangeType.MODIFIED,
            PolicyChangeType.UNCHANGED,
        }:
            if (
                not baseline_chunk_id
                or not candidate_chunk_id
            ):
                raise ValueError(
                    "MODIFIED and UNCHANGED "
                    "require both chunk IDs."
                )

        payload = "\x1f".join(
            (
                self.version,
                self.change_type.value,
                self.impact.value,
                baseline_chunk_id,
                candidate_chunk_id,
                baseline_hash,
                candidate_hash,
                reason_code,
            )
        )

        finding_id = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

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
            "baseline_content_hash",
            baseline_hash,
        )

        object.__setattr__(
            self,
            "candidate_content_hash",
            candidate_hash,
        )

        object.__setattr__(
            self,
            "reason_code",
            reason_code,
        )

        object.__setattr__(
            self,
            "finding_id",
            finding_id,
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "version": (
                self.version
            ),
            "finding_id": (
                self.finding_id
            ),
            "change_type": (
                self.change_type.value
            ),
            "impact": (
                self.impact.value
            ),
            "baseline_chunk_id": (
                self.baseline_chunk_id
                or None
            ),
            "candidate_chunk_id": (
                self.candidate_chunk_id
                or None
            ),
            "baseline_content_hash": (
                self.baseline_content_hash
                or None
            ),
            "candidate_content_hash": (
                self.candidate_content_hash
                or None
            ),
            "reason_code": (
                self.reason_code
                or None
            ),
        }


@dataclass(frozen=True)
class PolicyChangeReport:
    """Aggregate change result for one authoritative version pair."""

    request: PolicyChangeRequest

    findings: tuple[
        PolicyChangeFinding,
        ...
    ]

    algorithm: str

    version: str = (
        POLICY_CHANGE_CONTRACT_VERSION
    )

    report_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.request,
            PolicyChangeRequest,
        ):
            raise TypeError(
                "request must be a "
                "PolicyChangeRequest."
            )

        try:
            findings = tuple(
                self.findings
            )
        except TypeError as exc:
            raise TypeError(
                "findings must be iterable."
            ) from exc

        for finding in findings:
            if not isinstance(
                finding,
                PolicyChangeFinding,
            ):
                raise TypeError(
                    "findings must contain only "
                    "PolicyChangeFinding values."
                )

        finding_ids = tuple(
            finding.finding_id
            for finding
            in findings
        )

        if (
            len(set(finding_ids))
            != len(finding_ids)
        ):
            raise ValueError(
                "Duplicate policy-change "
                "findings are not allowed."
            )

        algorithm = _required_text(
            self.algorithm,
            "algorithm",
        )

        payload = "\x1f".join(
            (
                self.version,
                self.request.request_id,
                algorithm,
                ",".join(
                    sorted(
                        finding_ids
                    )
                ),
            )
        )

        report_id = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

        object.__setattr__(
            self,
            "findings",
            findings,
        )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "report_id",
            report_id,
        )

    @property
    def added_count(
        self,
    ) -> int:
        return self._count(
            PolicyChangeType.ADDED
        )

    @property
    def removed_count(
        self,
    ) -> int:
        return self._count(
            PolicyChangeType.REMOVED
        )

    @property
    def modified_count(
        self,
    ) -> int:
        return self._count(
            PolicyChangeType.MODIFIED
        )

    @property
    def unchanged_count(
        self,
    ) -> int:
        return self._count(
            PolicyChangeType.UNCHANGED
        )

    @property
    def changed(
        self,
    ) -> bool:
        return any(
            finding.change_type
            is not PolicyChangeType.UNCHANGED
            for finding
            in self.findings
        )

    @property
    def maximum_impact(
        self,
    ) -> PolicyChangeImpact:
        if not self.findings:
            return (
                PolicyChangeImpact
                .INFORMATIONAL
            )

        return max(
            (
                finding.impact
                for finding
                in self.findings
            ),
            key=lambda impact: (
                _IMPACT_PRIORITY[
                    impact
                ]
            ),
        )

    def _count(
        self,
        change_type: PolicyChangeType,
    ) -> int:
        return sum(
            finding.change_type
            is change_type
            for finding
            in self.findings
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Privacy-safe report serialization."""

        return {
            "version": (
                self.version
            ),
            "report_id": (
                self.report_id
            ),
            "request": (
                self.request.to_dict()
            ),
            "algorithm": (
                self.algorithm
            ),
            "changed": (
                self.changed
            ),
            "maximum_impact": (
                self.maximum_impact.value
            ),
            "counts": {
                "added": (
                    self.added_count
                ),
                "removed": (
                    self.removed_count
                ),
                "modified": (
                    self.modified_count
                ),
                "unchanged": (
                    self.unchanged_count
                ),
            },
            "findings": [
                finding.to_dict()
                for finding
                in self.findings
            ],
        }


def build_policy_change_report(
    request: PolicyChangeRequest,
    *,
    findings: Iterable[
        PolicyChangeFinding
    ],
    algorithm: str,
) -> PolicyChangeReport:
    """Create a deterministic policy-change report."""

    try:
        values = tuple(
            findings
        )
    except TypeError as exc:
        raise TypeError(
            "findings must be iterable."
        ) from exc

    return PolicyChangeReport(
        request=request,
        findings=values,
        algorithm=algorithm,
    )
