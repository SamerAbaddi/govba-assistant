"""Provider-independent requirements and BRD contract for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from govba.rag.models import SourceLanguage


REQUIREMENTS_CONTRACT_VERSION = (
    "govba-requirements-contract-v1"
)


class RequirementType(
    str,
    Enum,
):
    BUSINESS = "business"
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    DATA = "data"
    INTEGRATION = "integration"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    REPORTING = "reporting"
    OTHER = "other"


class RequirementPriority(
    str,
    Enum,
):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


class RequirementStatus(
    str,
    Enum,
):
    DRAFT = "draft"
    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class RequirementOrigin(
    str,
    Enum,
):
    USER_INPUT = "user_input"
    CORRESPONDENCE = "correspondence"
    POLICY = "policy"
    DOCUMENT = "document"
    WORKSHOP = "workshop"
    INTERVIEW = "interview"
    OTHER = "other"


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


def _sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _text_tuple(
    values: tuple[
        str,
        ...
    ],
    field_name: str,
) -> tuple[
    str,
    ...
]:
    if isinstance(
        values,
        (str, bytes),
    ):
        raise TypeError(
            f"{field_name} must be iterable."
        )

    try:
        normalized = tuple(
            _required_text(
                value,
                field_name,
            )
            for value in values
        )
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    if (
        len(
            set(
                normalized
            )
        )
        != len(
            normalized
        )
    ):
        raise ValueError(
            f"{field_name} values must be unique."
        )

    return normalized


@dataclass(frozen=True)
class RequirementsRequest:
    """Input contract for requirements and BRD intelligence."""

    source_text: str

    language: SourceLanguage

    project_id: str

    trace_id: str

    origin: RequirementOrigin = (
        RequirementOrigin.USER_INPUT
    )

    document_title: str = ""

    source_reference: str = ""

    version: str = (
        REQUIREMENTS_CONTRACT_VERSION
    )

    source_sha256: str = field(
        init=False
    )

    request_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        source_text = _required_text(
            self.source_text,
            "source_text",
        )

        project_id = _required_text(
            self.project_id,
            "project_id",
        )

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        document_title = _optional_text(
            self.document_title,
            "document_title",
        )

        source_reference = _optional_text(
            self.source_reference,
            "source_reference",
        )

        if not isinstance(
            self.language,
            SourceLanguage,
        ):
            raise TypeError(
                "language must be a "
                "SourceLanguage."
            )

        if not isinstance(
            self.origin,
            RequirementOrigin,
        ):
            raise TypeError(
                "origin must be a "
                "RequirementOrigin."
            )

        source_hash = _sha256(
            source_text
        )

        identity = "\x1f".join(
            (
                self.version,
                project_id,
                trace_id,
                source_hash,
                self.language.value,
                self.origin.value,
                document_title,
                source_reference,
            )
        )

        object.__setattr__(
            self,
            "source_text",
            source_text,
        )

        object.__setattr__(
            self,
            "project_id",
            project_id,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "document_title",
            document_title,
        )

        object.__setattr__(
            self,
            "source_reference",
            source_reference,
        )

        object.__setattr__(
            self,
            "source_sha256",
            source_hash,
        )

        object.__setattr__(
            self,
            "request_id",
            _sha256(
                identity
            ),
        )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Privacy-safe serialization by default."""

        if not isinstance(
            include_content,
            bool,
        ):
            raise TypeError(
                "include_content must be boolean."
            )

        data: dict[
            str,
            Any,
        ] = {
            "version": (
                self.version
            ),
            "request_id": (
                self.request_id
            ),
            "project_id": (
                self.project_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "language": (
                self.language.value
            ),
            "origin": (
                self.origin.value
            ),
            "source_sha256": (
                self.source_sha256
            ),
            "has_document_title": bool(
                self.document_title
            ),
            "has_source_reference": bool(
                self.source_reference
            ),
        }

        if include_content:
            data[
                "source_text"
            ] = self.source_text

            data[
                "document_title"
            ] = self.document_title

            data[
                "source_reference"
            ] = self.source_reference

        return data


@dataclass(frozen=True)
class RequirementItem:
    """One structured business or system requirement."""

    request_id: str

    sequence: int

    requirement_type: RequirementType

    priority: RequirementPriority

    status: RequirementStatus

    statement: str

    rationale: str = ""

    acceptance_criteria: tuple[
        str,
        ...
    ] = ()

    dependencies: tuple[
        str,
        ...
    ] = ()

    source_references: tuple[
        str,
        ...
    ] = ()

    version: str = (
        REQUIREMENTS_CONTRACT_VERSION
    )

    statement_sha256: str = field(
        init=False
    )

    requirement_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        if (
            isinstance(
                self.sequence,
                bool,
            )
            or not isinstance(
                self.sequence,
                int,
            )
            or self.sequence < 0
        ):
            raise ValueError(
                "sequence must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.requirement_type,
            RequirementType,
        ):
            raise TypeError(
                "requirement_type must be a "
                "RequirementType."
            )

        if not isinstance(
            self.priority,
            RequirementPriority,
        ):
            raise TypeError(
                "priority must be a "
                "RequirementPriority."
            )

        if not isinstance(
            self.status,
            RequirementStatus,
        ):
            raise TypeError(
                "status must be a "
                "RequirementStatus."
            )

        statement = _required_text(
            self.statement,
            "statement",
        )

        rationale = _optional_text(
            self.rationale,
            "rationale",
        )

        acceptance_criteria = (
            _text_tuple(
                self.acceptance_criteria,
                "acceptance_criteria",
            )
        )

        dependencies = _text_tuple(
            self.dependencies,
            "dependencies",
        )

        source_references = (
            _text_tuple(
                self.source_references,
                "source_references",
            )
        )

        statement_hash = _sha256(
            statement
        )

        acceptance_hashes = tuple(
            _sha256(
                value
            )
            for value in acceptance_criteria
        )

        rationale_hash = (
            _sha256(
                rationale
            )
            if rationale
            else ""
        )

        identity = "\x1f".join(
            (
                self.version,
                request_id,
                str(
                    self.sequence
                ),
                self.requirement_type.value,
                self.priority.value,
                self.status.value,
                statement_hash,
                rationale_hash,
                ",".join(
                    acceptance_hashes
                ),
                ",".join(
                    dependencies
                ),
                ",".join(
                    source_references
                ),
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "statement",
            statement,
        )

        object.__setattr__(
            self,
            "rationale",
            rationale,
        )

        object.__setattr__(
            self,
            "acceptance_criteria",
            acceptance_criteria,
        )

        object.__setattr__(
            self,
            "dependencies",
            dependencies,
        )

        object.__setattr__(
            self,
            "source_references",
            source_references,
        )

        object.__setattr__(
            self,
            "statement_sha256",
            statement_hash,
        )

        object.__setattr__(
            self,
            "requirement_id",
            _sha256(
                identity
            ),
        )

    @property
    def has_acceptance_criteria(
        self,
    ) -> bool:
        return bool(
            self.acceptance_criteria
        )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, object]:
        """Serialize requirement metadata without raw text by default."""

        if not isinstance(
            include_content,
            bool,
        ):
            raise TypeError(
                "include_content must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "requirement_id": (
                self.requirement_id
            ),
            "request_id": (
                self.request_id
            ),
            "sequence": (
                self.sequence
            ),
            "requirement_type": (
                self.requirement_type.value
            ),
            "priority": (
                self.priority.value
            ),
            "status": (
                self.status.value
            ),
            "statement_sha256": (
                self.statement_sha256
            ),
            "has_rationale": bool(
                self.rationale
            ),
            "acceptance_criteria_count": (
                len(
                    self.acceptance_criteria
                )
            ),
            "dependency_count": (
                len(
                    self.dependencies
                )
            ),
            "source_reference_count": (
                len(
                    self.source_references
                )
            ),
        }

        if include_content:
            data[
                "statement"
            ] = self.statement

            data[
                "rationale"
            ] = self.rationale

            data[
                "acceptance_criteria"
            ] = list(
                self.acceptance_criteria
            )

            data[
                "dependencies"
            ] = list(
                self.dependencies
            )

            data[
                "source_references"
            ] = list(
                self.source_references
            )

        return data


@dataclass(frozen=True)
class BusinessRequirementsDocument:
    """Structured provider-independent BRD representation."""

    request_id: str

    project_id: str

    trace_id: str

    title: str

    requirements: tuple[
        RequirementItem,
        ...
    ]

    version: str = (
        REQUIREMENTS_CONTRACT_VERSION
    )

    brd_id: str = field(
        init=False
    )

    def __post_init__(self) -> None:
        request_id = _required_text(
            self.request_id,
            "request_id",
        )

        project_id = _required_text(
            self.project_id,
            "project_id",
        )

        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        title = _required_text(
            self.title,
            "title",
        )

        try:
            requirements = tuple(
                self.requirements
            )
        except TypeError as exc:
            raise TypeError(
                "requirements must be iterable."
            ) from exc

        for requirement in requirements:
            if not isinstance(
                requirement,
                RequirementItem,
            ):
                raise TypeError(
                    "requirements must contain only "
                    "RequirementItem values."
                )

            if (
                requirement.request_id
                != request_id
            ):
                raise ValueError(
                    "Requirement request ID does not "
                    "match BRD request ID."
                )

        sequences = tuple(
            requirement.sequence
            for requirement
            in requirements
        )

        if sequences != tuple(
            range(
                len(
                    requirements
                )
            )
        ):
            raise ValueError(
                "Requirement sequences must be "
                "contiguous from zero."
            )

        requirement_ids = tuple(
            requirement.requirement_id
            for requirement
            in requirements
        )

        if (
            len(
                set(
                    requirement_ids
                )
            )
            != len(
                requirement_ids
            )
        ):
            raise ValueError(
                "Requirements must be unique."
            )

        identity = "\x1f".join(
            (
                self.version,
                request_id,
                project_id,
                trace_id,
                _sha256(
                    title
                ),
                ",".join(
                    requirement_ids
                ),
            )
        )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

        object.__setattr__(
            self,
            "project_id",
            project_id,
        )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "title",
            title,
        )

        object.__setattr__(
            self,
            "requirements",
            requirements,
        )

        object.__setattr__(
            self,
            "brd_id",
            _sha256(
                identity
            ),
        )

    @property
    def requirement_count(
        self,
    ) -> int:
        return len(
            self.requirements
        )

    @property
    def must_count(
        self,
    ) -> int:
        return sum(
            requirement.priority
            is RequirementPriority.MUST
            for requirement
            in self.requirements
        )

    @property
    def approved_count(
        self,
    ) -> int:
        return sum(
            requirement.status
            is RequirementStatus.APPROVED
            for requirement
            in self.requirements
        )

    def to_dict(
        self,
        *,
        include_content: bool = False,
    ) -> dict[str, object]:
        """Serialize BRD metadata privacy-safely by default."""

        if not isinstance(
            include_content,
            bool,
        ):
            raise TypeError(
                "include_content must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "brd_id": (
                self.brd_id
            ),
            "request_id": (
                self.request_id
            ),
            "project_id": (
                self.project_id
            ),
            "trace_id": (
                self.trace_id
            ),
            "title_sha256": (
                _sha256(
                    self.title
                )
            ),
            "requirement_count": (
                self.requirement_count
            ),
            "must_count": (
                self.must_count
            ),
            "approved_count": (
                self.approved_count
            ),
            "requirements": [
                requirement.to_dict(
                    include_content=(
                        include_content
                    )
                )
                for requirement
                in self.requirements
            ],
        }

        if include_content:
            data[
                "title"
            ] = self.title

        return data
