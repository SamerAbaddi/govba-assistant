"""Minimal capability routing for GovBA-GAR."""

from govba.router.contract import (
    ROUTER_CONTRACT_VERSION,
    Capability,
    RouteReason,
    RoutingDecision,
    RoutingRequest,
)
from govba.router.orchestrator import (
    CapabilityOrchestrator,
    OrchestrationResult,
)
from govba.router.router import (
    route_capability,
)


__all__ = [
    "ROUTER_CONTRACT_VERSION",
    "Capability",
    "RouteReason",
    "RoutingDecision",
    "RoutingRequest",
    "CapabilityOrchestrator",
    "OrchestrationResult",
    "route_capability",
]
