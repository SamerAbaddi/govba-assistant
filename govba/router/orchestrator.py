"""Minimal handler orchestrator for GovBA-GAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from govba.router.contract import (
    Capability,
    RoutingDecision,
    RoutingRequest,
)
from govba.router.router import (
    route_capability,
)


Handler = Callable[
    [RoutingRequest],
    object,
]


@dataclass(frozen=True)
class OrchestrationResult:
    decision: RoutingDecision
    success: bool
    result: object | None
    error_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.decision.to_dict(),
            "success": self.success,
            "result_type": (
                type(self.result).__name__
                if self.result is not None
                else None
            ),
            "error_type": self.error_type,
        }


class CapabilityOrchestrator:
    def __init__(
        self,
        handlers: Mapping[
            Capability,
            Handler,
        ],
    ) -> None:
        self._handlers = dict(
            handlers
        )

        for capability, handler in (
            self._handlers.items()
        ):
            if not isinstance(
                capability,
                Capability,
            ):
                raise TypeError(
                    "handler keys must be Capability values."
                )

            if not callable(
                handler
            ):
                raise TypeError(
                    "handlers must be callable."
                )

    def execute(
        self,
        request: RoutingRequest,
    ) -> OrchestrationResult:
        decision = route_capability(
            request
        )

        handler = self._handlers.get(
            decision.capability
        )

        if handler is None:
            return OrchestrationResult(
                decision=decision,
                success=False,
                result=None,
                error_type=(
                    "handler_unavailable"
                ),
            )

        try:
            result = handler(
                request
            )
        except Exception as exc:
            return OrchestrationResult(
                decision=decision,
                success=False,
                result=None,
                error_type=(
                    type(exc).__name__
                ),
            )

        return OrchestrationResult(
            decision=decision,
            success=True,
            result=result,
        )
