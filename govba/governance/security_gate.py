"""Fail-closed security gates for the GovBA-GAR pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from govba.governance.pii import (
    PIIRedactionResult,
    redact_pii,
)
from govba.governance.prompt_injection import (
    PromptInjectionAssessment,
    assess_prompt_injection,
)
from govba.governance.security import (
    SecurityAssessment,
    SecurityDecision,
    SecurityFinding,
    SecurityIssueCode,
    SecuritySeverity,
    SecuritySurface,
    assess_security,
)
from govba.governance.source_allowlist import (
    SourceAllowlistResult,
)
from govba.rag.retrieval import (
    RetrievalResult,
)


SECURITY_GATE_VERSION = (
    "govba-security-gate-v1"
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


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _merge_findings(
    *assessments: SecurityAssessment,
) -> tuple[
    SecurityFinding,
    ...
]:
    """Merge security findings without duplicating identical findings."""

    values: dict[
        str,
        SecurityFinding,
    ] = {}

    for assessment in assessments:
        if not isinstance(
            assessment,
            SecurityAssessment,
        ):
            raise TypeError(
                "All values must be "
                "SecurityAssessment objects."
            )

        for finding in (
            assessment.findings
        ):
            values[
                finding.finding_id
            ] = finding

    return tuple(
        values[
            finding_id
        ]
        for finding_id
        in sorted(
            values
        )
    )


@dataclass(frozen=True)
class SecureTextGateResult:
    """Security result for text entering or leaving the AI pipeline."""

    trace_id: str

    surface: SecuritySurface

    input_sha256: str

    sanitized_text: str

    pii_result: PIIRedactionResult

    injection_result: (
        PromptInjectionAssessment
    )

    security_assessment: (
        SecurityAssessment
    )

    version: str = (
        SECURITY_GATE_VERSION
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        if not isinstance(
            self.surface,
            SecuritySurface,
        ):
            raise TypeError(
                "surface must be a "
                "SecuritySurface."
            )

        input_hash = _required_text(
            self.input_sha256,
            "input_sha256",
        ).lower()

        if (
            len(input_hash) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character
                in input_hash
            )
        ):
            raise ValueError(
                "input_sha256 must be "
                "a SHA-256 hexadecimal digest."
            )

        if not isinstance(
            self.sanitized_text,
            str,
        ):
            raise TypeError(
                "sanitized_text must be a string."
            )

        if not isinstance(
            self.pii_result,
            PIIRedactionResult,
        ):
            raise TypeError(
                "pii_result must be a "
                "PIIRedactionResult."
            )

        if not isinstance(
            self.injection_result,
            PromptInjectionAssessment,
        ):
            raise TypeError(
                "injection_result must be a "
                "PromptInjectionAssessment."
            )

        if not isinstance(
            self.security_assessment,
            SecurityAssessment,
        ):
            raise TypeError(
                "security_assessment must be "
                "a SecurityAssessment."
            )

        for component_trace in (
            self.pii_result.trace_id,
            self.injection_result.trace_id,
            self.security_assessment.trace_id,
        ):
            if component_trace != trace_id:
                raise ValueError(
                    "All security components must "
                    "use the same trace ID."
                )

        if (
            self.pii_result.input_sha256
            != input_hash
        ):
            raise ValueError(
                "PII result does not match "
                "the inspected text."
            )

        if (
            self.injection_result.input_sha256
            != input_hash
        ):
            raise ValueError(
                "Prompt-injection result does not "
                "match the inspected text."
            )

        if (
            self.sanitized_text
            != self.pii_result.redacted_text
        ):
            raise ValueError(
                "sanitized_text must match "
                "the PII-redacted representation."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "input_sha256",
            input_hash,
        )

    @property
    def decision(
        self,
    ) -> SecurityDecision:
        return (
            self.security_assessment
            .decision
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            in {
                SecurityDecision.ALLOW,
                SecurityDecision.REDACT,
            }
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.BLOCK
        )

    @property
    def requires_review(
        self,
    ) -> bool:
        return (
            self.decision
            is SecurityDecision.REVIEW
        )

    @property
    def downstream_text(
        self,
    ) -> str | None:
        """Return the only text permitted to continue downstream."""

        if not self.allowed:
            return None

        return self.sanitized_text

    def to_dict(
        self,
        *,
        include_sanitized_text: bool = False,
    ) -> dict[str, object]:
        """Serialize without inspected text by default."""

        if not isinstance(
            include_sanitized_text,
            bool,
        ):
            raise TypeError(
                "include_sanitized_text "
                "must be boolean."
            )

        data: dict[
            str,
            object,
        ] = {
            "version": (
                self.version
            ),
            "trace_id": (
                self.trace_id
            ),
            "surface": (
                self.surface.value
            ),
            "input_sha256": (
                self.input_sha256
            ),
            "decision": (
                self.decision.value
            ),
            "allowed": (
                self.allowed
            ),
            "should_block": (
                self.should_block
            ),
            "requires_review": (
                self.requires_review
            ),
            "pii": (
                self.pii_result
                .to_dict()
            ),
            "prompt_injection": (
                self.injection_result
                .to_dict()
            ),
            "security_assessment": (
                self.security_assessment
                .to_dict()
            ),
        }

        if include_sanitized_text:
            data[
                "sanitized_text"
            ] = self.sanitized_text

        return data


def secure_text(
    text: str,
    *,
    trace_id: str,
    surface: SecuritySurface,
) -> SecureTextGateResult:
    """Run PII and injection controls over one text surface."""

    value = _required_text(
        text,
        "text",
    )

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    if not isinstance(
        surface,
        SecuritySurface,
    ):
        raise TypeError(
            "surface must be a "
            "SecuritySurface."
        )

    input_hash = _sha256_text(
        value
    )

    pii = redact_pii(
        value,
        trace_id=trace,
        surface=surface,
    )

    injection = assess_prompt_injection(
        value,
        trace_id=trace,
        surface=surface,
    )

    findings = _merge_findings(
        pii.security_assessment,
        injection.security_assessment,
    )

    security = assess_security(
        trace_id=trace,
        findings=findings,
    )

    return SecureTextGateResult(
        trace_id=trace,
        surface=surface,
        input_sha256=input_hash,
        sanitized_text=(
            pii.redacted_text
        ),
        pii_result=pii,
        injection_result=injection,
        security_assessment=security,
    )


def secure_user_query(
    text: str,
    *,
    trace_id: str,
) -> SecureTextGateResult:
    """Security gate for raw user queries."""

    return secure_text(
        text,
        trace_id=trace_id,
        surface=(
            SecuritySurface.USER_QUERY
        ),
    )


def secure_model_output(
    text: str,
    *,
    trace_id: str,
) -> SecureTextGateResult:
    """Security gate applied before model output reaches the user."""

    return secure_text(
        text,
        trace_id=trace_id,
        surface=(
            SecuritySurface.MODEL_OUTPUT
        ),
    )


@dataclass(frozen=True)
class SecureRetrievalGateResult:
    """Trusted retrieval candidates permitted into evidence processing."""

    trace_id: str

    results: tuple[
        RetrievalResult,
        ...
    ]

    original_result_count: int

    discarded_document_ids: tuple[
        str,
        ...
    ]

    security_assessment: (
        SecurityAssessment
    )

    version: str = (
        SECURITY_GATE_VERSION
    )

    def __post_init__(self) -> None:
        trace_id = _required_text(
            self.trace_id,
            "trace_id",
        )

        try:
            results = tuple(
                self.results
            )

            discarded = tuple(
                self.discarded_document_ids
            )

        except TypeError as exc:
            raise TypeError(
                "results and discarded_document_ids "
                "must be iterable."
            ) from exc

        for result in results:
            if not isinstance(
                result,
                RetrievalResult,
            ):
                raise TypeError(
                    "results must contain only "
                    "RetrievalResult values."
                )

        if (
            isinstance(
                self.original_result_count,
                bool,
            )
            or not isinstance(
                self.original_result_count,
                int,
            )
            or self.original_result_count < 0
        ):
            raise ValueError(
                "original_result_count must be "
                "a non-negative integer."
            )

        if (
            len(results)
            > self.original_result_count
        ):
            raise ValueError(
                "Secure retrieval cannot contain "
                "more results than the original retrieval."
            )

        if (
            len(set(discarded))
            != len(discarded)
        ):
            raise ValueError(
                "discarded_document_ids must "
                "be unique."
            )

        expected_ranks = tuple(
            range(
                1,
                len(results) + 1,
            )
        )

        actual_ranks = tuple(
            result.rank
            for result
            in results
        )

        if (
            actual_ranks
            != expected_ranks
        ):
            raise ValueError(
                "Secure retrieval ranks must "
                "be contiguous."
            )

        if not isinstance(
            self.security_assessment,
            SecurityAssessment,
        ):
            raise TypeError(
                "security_assessment must be "
                "a SecurityAssessment."
            )

        if (
            self.security_assessment.trace_id
            != trace_id
        ):
            raise ValueError(
                "Security assessment trace ID "
                "must match retrieval gate."
            )

        object.__setattr__(
            self,
            "trace_id",
            trace_id,
        )

        object.__setattr__(
            self,
            "results",
            results,
        )

        object.__setattr__(
            self,
            "discarded_document_ids",
            discarded,
        )

    @property
    def trusted_result_count(
        self,
    ) -> int:
        return len(
            self.results
        )

    @property
    def discarded_result_count(
        self,
    ) -> int:
        return (
            self.original_result_count
            - self.trusted_result_count
        )

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.security_assessment
            .decision
            is SecurityDecision.ALLOW
        )

    @property
    def should_block(
        self,
    ) -> bool:
        return (
            self.security_assessment
            .should_block
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Serialize retrieval security metadata without evidence text."""

        return {
            "version": (
                self.version
            ),
            "trace_id": (
                self.trace_id
            ),
            "original_result_count": (
                self.original_result_count
            ),
            "trusted_result_count": (
                self.trusted_result_count
            ),
            "discarded_result_count": (
                self.discarded_result_count
            ),
            "discarded_document_ids": list(
                self.discarded_document_ids
            ),
            "results": [
                {
                    "chunk_id": (
                        result.chunk.chunk_id
                    ),
                    "document_id": (
                        result.chunk.document_id
                    ),
                    "rank": (
                        result.rank
                    ),
                    "score": (
                        result.score
                    ),
                    "retrieval_method": (
                        result.retrieval_method
                    ),
                }
                for result
                in self.results
            ],
            "security_assessment": (
                self.security_assessment
                .to_dict()
            ),
        }


def secure_retrieval_results(
    results: Iterable[
        RetrievalResult
    ],
    *,
    source_assessments: Iterable[
        SourceAllowlistResult
    ],
    trace_id: str,
) -> SecureRetrievalGateResult:
    """Remove untrusted retrieval results and fail closed if none survive."""

    trace = _required_text(
        trace_id,
        "trace_id",
    )

    try:
        result_values = tuple(
            results
        )
    except TypeError as exc:
        raise TypeError(
            "results must be iterable."
        ) from exc

    for result in result_values:
        if not isinstance(
            result,
            RetrievalResult,
        ):
            raise TypeError(
                "results must contain only "
                "RetrievalResult values."
            )

    chunk_ids = tuple(
        result.chunk.chunk_id
        for result
        in result_values
    )

    if (
        len(set(chunk_ids))
        != len(chunk_ids)
    ):
        raise ValueError(
            "Retrieval results must not "
            "contain duplicate chunks."
        )

    try:
        assessment_values = tuple(
            source_assessments
        )
    except TypeError as exc:
        raise TypeError(
            "source_assessments must be iterable."
        ) from exc

    assessment_map: dict[
        str,
        SourceAllowlistResult,
    ] = {}

    for assessment in (
        assessment_values
    ):
        if not isinstance(
            assessment,
            SourceAllowlistResult,
        ):
            raise TypeError(
                "source_assessments must contain "
                "only SourceAllowlistResult values."
            )

        if assessment.trace_id != trace:
            raise ValueError(
                "Source assessment trace IDs "
                "must match retrieval trace ID."
            )

        if (
            assessment.document_id
            in assessment_map
        ):
            raise ValueError(
                "Duplicate source assessment "
                "for document ID."
            )

        assessment_map[
            assessment.document_id
        ] = assessment

    ordered_results = tuple(
        sorted(
            result_values,
            key=lambda result: (
                result.rank,
                result.chunk.document_id,
                result.chunk.chunk_index,
                result.chunk.chunk_id,
            ),
        )
    )

    trusted = []

    discarded_document_ids = set()

    for result in ordered_results:
        document_id = (
            result.chunk.document_id
        )

        assessment = (
            assessment_map.get(
                document_id
            )
        )

        if (
            assessment is None
            or not assessment.allowed
        ):
            discarded_document_ids.add(
                document_id
            )

            continue

        trusted.append(
            result
        )

    reranked = tuple(
        RetrievalResult(
            chunk=result.chunk,
            score=result.score,
            raw_score=result.raw_score,
            rank=rank,
            retrieval_method=(
                result.retrieval_method
            ),
            matched_terms=(
                result.matched_terms
            ),
        )
        for rank, result
        in enumerate(
            trusted,
            start=1,
        )
    )

    if (
        result_values
        and not reranked
    ):
        reference_payload = ",".join(
            sorted(
                discarded_document_ids
            )
        )

        reference_hash = (
            hashlib.sha256(
                reference_payload.encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        findings = (
            SecurityFinding(
                code=(
                    SecurityIssueCode
                    .UNTRUSTED_SOURCE
                ),
                surface=(
                    SecuritySurface.RETRIEVAL
                ),
                severity=(
                    SecuritySeverity.HIGH
                ),
                decision=(
                    SecurityDecision.BLOCK
                ),
                occurrence_count=max(
                    1,
                    len(
                        discarded_document_ids
                    ),
                ),
                reference_id=(
                    "retrieval:"
                    f"{reference_hash[:16]}"
                ),
            ),
        )

    else:
        findings = ()

    security = assess_security(
        trace_id=trace,
        findings=findings,
    )

    return SecureRetrievalGateResult(
        trace_id=trace,
        results=reranked,
        original_result_count=len(
            result_values
        ),
        discarded_document_ids=tuple(
            sorted(
                discarded_document_ids
            )
        ),
        security_assessment=security,
    )
