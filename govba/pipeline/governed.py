"""Minimal unified governed execution pipeline for GovBA-GAR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from govba.router import (
    Capability,
    CapabilityOrchestrator,
    RoutingDecision,
    RoutingRequest,
)


GOVERNED_PIPELINE_VERSION = "govba-governed-pipeline-v1"


class PipelineDecision(str, Enum):
    READY = "ready"
    REVIEW = "review"
    ABSTAIN = "abstain"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    value = value.strip()

    if not value:
        raise ValueError(f"{name} must not be blank.")

    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _text_tuple(
    values,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be iterable."
        )

    try:
        result = tuple(
            _text(value, name)
            for value in values
        )
    except TypeError as exc:
        raise TypeError(
            f"{name} must be iterable."
        ) from exc

    if len(set(result)) != len(result):
        raise ValueError(
            f"{name} values must be unique."
        )

    return result


@dataclass(frozen=True)
class GovernedCapabilityResult:
    """Common envelope returned by GovBA capability adapters."""

    capability: Capability
    decision: PipelineDecision
    payload: object | None
    result_reference: str = ""
    reason_codes: tuple[str, ...] = ()
    provenance_ids: tuple[str, ...] = ()
    version: str = GOVERNED_PIPELINE_VERSION
    governed_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.capability,
            Capability,
        ):
            raise TypeError(
                "capability must be a Capability."
            )

        if not isinstance(
            self.decision,
            PipelineDecision,
        ):
            raise TypeError(
                "decision must be a PipelineDecision."
            )

        if (
            self.decision
            is PipelineDecision.ABSTAIN
            and self.payload is not None
        ):
            raise ValueError(
                "ABSTAIN result cannot contain payload."
            )

        if (
            self.decision
            is not PipelineDecision.ABSTAIN
            and self.payload is None
        ):
            raise ValueError(
                "READY/REVIEW result requires payload."
            )

        result_reference = (
            self.result_reference.strip()
            if isinstance(
                self.result_reference,
                str,
            )
            else None
        )

        if result_reference is None:
            raise TypeError(
                "result_reference must be a string."
            )

        reasons = _text_tuple(
            self.reason_codes,
            "reason_codes",
        )

        provenance = _text_tuple(
            self.provenance_ids,
            "provenance_ids",
        )

        identity = "\x1f".join(
            (
                self.version,
                self.capability.value,
                self.decision.value,
                (
                    _sha256(result_reference)
                    if result_reference
                    else ""
                ),
                ",".join(
                    _sha256(value)
                    for value in reasons
                ),
                ",".join(
                    _sha256(value)
                    for value in provenance
                ),
                (
                    type(self.payload).__name__
                    if self.payload is not None
                    else ""
                ),
            )
        )

        object.__setattr__(
            self,
            "result_reference",
            result_reference,
        )

        object.__setattr__(
            self,
            "reason_codes",
            reasons,
        )

        object.__setattr__(
            self,
            "provenance_ids",
            provenance,
        )

        object.__setattr__(
            self,
            "governed_id",
            _sha256(identity),
        )

    def to_dict(self) -> dict[str, object]:
        """Privacy-safe serialization; payload content is excluded."""

        return {
            "version": self.version,
            "governed_id": self.governed_id,
            "capability": self.capability.value,
            "decision": self.decision.value,
            "result_reference_sha256": (
                _sha256(self.result_reference)
                if self.result_reference
                else ""
            ),
            "reason_codes": list(
                self.reason_codes
            ),
            "provenance_count": len(
                self.provenance_ids
            ),
            "provenance_hashes": [
                _sha256(value)
                for value
                in self.provenance_ids
            ],
            "payload_type": (
                type(self.payload).__name__
                if self.payload is not None
                else None
            ),
        }


@dataclass(frozen=True)
class UnifiedPipelineResult:
    """Final common result returned to the application integration layer."""

    route: RoutingDecision
    decision: PipelineDecision
    execution_success: bool
    fallback_used: bool
    payload: object | None
    governed_result: GovernedCapabilityResult | None
    error_type: str = ""
    version: str = GOVERNED_PIPELINE_VERSION
    pipeline_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.route,
            RoutingDecision,
        ):
            raise TypeError(
                "route must be a RoutingDecision."
            )

        if not isinstance(
            self.decision,
            PipelineDecision,
        ):
            raise TypeError(
                "decision must be a PipelineDecision."
            )

        if not isinstance(
            self.execution_success,
            bool,
        ):
            raise TypeError(
                "execution_success must be boolean."
            )

        if not isinstance(
            self.fallback_used,
            bool,
        ):
            raise TypeError(
                "fallback_used must be boolean."
            )

        if (
            self.governed_result is not None
            and not isinstance(
                self.governed_result,
                GovernedCapabilityResult,
            )
        ):
            raise TypeError(
                "governed_result must be "
                "GovernedCapabilityResult or None."
            )

        if not isinstance(
            self.error_type,
            str,
        ):
            raise TypeError(
                "error_type must be a string."
            )

        if (
            self.decision
            is PipelineDecision.ABSTAIN
            and self.payload is not None
        ):
            raise ValueError(
                "ABSTAIN pipeline result "
                "cannot contain payload."
            )

        identity = "\x1f".join(
            (
                self.version,
                self.route.route_id,
                self.decision.value,
                str(
                    self.execution_success
                ).lower(),
                str(
                    self.fallback_used
                ).lower(),
                (
                    self.governed_result.governed_id
                    if self.governed_result
                    else ""
                ),
                self.error_type,
            )
        )

        object.__setattr__(
            self,
            "pipeline_id",
            _sha256(identity),
        )

    @property
    def abstained(self) -> bool:
        return (
            self.decision
            is PipelineDecision.ABSTAIN
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pipeline_id": self.pipeline_id,
            "route": self.route.to_dict(),
            "decision": self.decision.value,
            "execution_success": (
                self.execution_success
            ),
            "fallback_used": (
                self.fallback_used
            ),
            "abstained": self.abstained,
            "payload_type": (
                type(self.payload).__name__
                if self.payload is not None
                else None
            ),
            "governed_result": (
                self.governed_result.to_dict()
                if self.governed_result
                else None
            ),
            "error_type": self.error_type,
        }


class UnifiedGovernedPipeline:
    """Thin governance layer over the Stage 17 orchestrator."""

    def __init__(
        self,
        orchestrator: CapabilityOrchestrator,
    ) -> None:
        if not isinstance(
            orchestrator,
            CapabilityOrchestrator,
        ):
            raise TypeError(
                "orchestrator must be "
                "a CapabilityOrchestrator."
            )

        self._orchestrator = orchestrator

    def execute(
        self,
        request: RoutingRequest,
    ) -> UnifiedPipelineResult:
        if not isinstance(
            request,
            RoutingRequest,
        ):
            raise TypeError(
                "request must be a RoutingRequest."
            )

        orchestration = (
            self._orchestrator.execute(
                request
            )
        )

        if not orchestration.success:
            return UnifiedPipelineResult(
                route=orchestration.decision,
                decision=PipelineDecision.ABSTAIN,
                execution_success=False,
                fallback_used=True,
                payload=None,
                governed_result=None,
                error_type=(
                    orchestration.error_type
                    or "execution_failed"
                ),
            )

        result = orchestration.result

        if not isinstance(
            result,
            GovernedCapabilityResult,
        ):
            return UnifiedPipelineResult(
                route=orchestration.decision,
                decision=PipelineDecision.ABSTAIN,
                execution_success=False,
                fallback_used=True,
                payload=None,
                governed_result=None,
                error_type=(
                    "invalid_governed_result"
                ),
            )

        if (
            result.capability
            is not orchestration.decision.capability
        ):
            return UnifiedPipelineResult(
                route=orchestration.decision,
                decision=PipelineDecision.ABSTAIN,
                execution_success=False,
                fallback_used=True,
                payload=None,
                governed_result=None,
                error_type=(
                    "capability_mismatch"
                ),
            )

        return UnifiedPipelineResult(
            route=orchestration.decision,
            decision=result.decision,
            execution_success=True,
            fallback_used=False,
            payload=result.payload,
            governed_result=result,
        )
