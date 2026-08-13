"""Tests for the minimal GovBA-GAR router."""

import unittest

from govba.rag.models import SourceLanguage
from govba.router import (
    ROUTER_CONTRACT_VERSION,
    Capability,
    CapabilityOrchestrator,
    RouteReason,
    RoutingRequest,
    route_capability,
)


def request(
    text,
    *,
    hint=None,
):
    return RoutingRequest(
        text=text,
        language=SourceLanguage.ENGLISH,
        trace_id="TRACE-ROUTER",
        capability_hint=hint,
    )


class TestRouter(unittest.TestCase):
    def test_version(self):
        self.assertEqual(
            ROUTER_CONTRACT_VERSION,
            "govba-router-contract-v1",
        )

    def test_requirements(self):
        result = route_capability(
            request(
                "Develop the business requirements BRD."
            )
        )

        self.assertEqual(
            result.capability,
            Capability.REQUIREMENTS,
        )

    def test_change(self):
        result = route_capability(
            request(
                "What changed between these policy versions?"
            )
        )

        self.assertEqual(
            result.capability,
            Capability.CHANGE,
        )

    def test_correspondence(self):
        result = route_capability(
            request(
                "Summarize this official email."
            )
        )

        self.assertEqual(
            result.capability,
            Capability.CORRESPONDENCE,
        )

    def test_web(self):
        result = route_capability(
            request(
                "Search the official website for the latest guidance."
            )
        )

        self.assertEqual(
            result.capability,
            Capability.WEB,
        )

    def test_rag(self):
        result = route_capability(
            request(
                "According to the policy, what is required?"
            )
        )

        self.assertEqual(
            result.capability,
            Capability.RAG,
        )

    def test_direct_fallback(self):
        result = route_capability(
            request(
                "Rewrite this sentence professionally."
            )
        )

        self.assertEqual(
            result.capability,
            Capability.DIRECT,
        )

        self.assertEqual(
            result.reason,
            RouteReason.DEFAULT_DIRECT,
        )

    def test_explicit_hint_wins(self):
        result = route_capability(
            request(
                "This mentions a policy.",
                hint=Capability.CORRESPONDENCE,
            )
        )

        self.assertEqual(
            result.capability,
            Capability.CORRESPONDENCE,
        )

        self.assertEqual(
            result.reason,
            RouteReason.EXPLICIT_HINT,
        )

    def test_arabic_requirements(self):
        value = RoutingRequest(
            text="استخرج متطلبات النظام.",
            language=SourceLanguage.ARABIC,
            trace_id="TRACE-AR",
        )

        self.assertEqual(
            route_capability(value).capability,
            Capability.REQUIREMENTS,
        )

    def test_request_serialization_is_private(self):
        raw = "Unique private government request."

        data = request(
            raw
        ).to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

    def test_deterministic_route_id(self):
        value = request(
            "According to the policy, what is required?"
        )

        first = route_capability(
            value
        )

        second = route_capability(
            value
        )

        self.assertEqual(
            first.route_id,
            second.route_id,
        )


class TestOrchestrator(unittest.TestCase):
    def test_handler_executes(self):
        orchestrator = CapabilityOrchestrator(
            {
                Capability.REQUIREMENTS:
                    lambda value: "handled",
            }
        )

        result = orchestrator.execute(
            request(
                "Create requirements for this BRD."
            )
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.result,
            "handled",
        )

    def test_missing_handler_fails_safely(self):
        orchestrator = (
            CapabilityOrchestrator(
                {}
            )
        )

        result = orchestrator.execute(
            request(
                "According to the policy..."
            )
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error_type,
            "handler_unavailable",
        )

    def test_handler_exception_fails_safely(self):
        def broken(_):
            raise RuntimeError(
                "private internal error"
            )

        orchestrator = CapabilityOrchestrator(
            {
                Capability.DIRECT:
                    broken,
            }
        )

        result = orchestrator.execute(
            request(
                "Rewrite this sentence."
            )
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.error_type,
            "RuntimeError",
        )

        self.assertNotIn(
            "private internal error",
            repr(
                result.to_dict()
            ),
        )


if __name__ == "__main__":
    unittest.main()
