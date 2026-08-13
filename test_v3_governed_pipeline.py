"""Tests for the minimal unified GovBA-GAR pipeline."""

import unittest

from govba.pipeline import (
    GOVERNED_PIPELINE_VERSION,
    GovernedCapabilityResult,
    PipelineDecision,
    UnifiedGovernedPipeline,
)
from govba.rag.models import SourceLanguage
from govba.router import (
    Capability,
    CapabilityOrchestrator,
    RoutingRequest,
)


def request(
    text="Create requirements for this BRD.",
):
    return RoutingRequest(
        text=text,
        language=SourceLanguage.ENGLISH,
        trace_id="TRACE-PIPELINE",
    )


class TestGovernedPipeline(unittest.TestCase):
    def test_version(self):
        self.assertEqual(
            GOVERNED_PIPELINE_VERSION,
            "govba-governed-pipeline-v1",
        )

    def test_ready_result(self):
        handler = lambda _: GovernedCapabilityResult(
            capability=Capability.REQUIREMENTS,
            decision=PipelineDecision.READY,
            payload={"result": "ok"},
            result_reference="RESULT-001",
            provenance_ids=("SRC-001",),
        )

        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        handler,
                }
            )
        )

        result = pipeline.execute(
            request()
        )

        self.assertTrue(
            result.execution_success
        )

        self.assertEqual(
            result.decision,
            PipelineDecision.READY,
        )

        self.assertFalse(
            result.fallback_used
        )

    def test_review_result(self):
        handler = lambda _: GovernedCapabilityResult(
            capability=Capability.REQUIREMENTS,
            decision=PipelineDecision.REVIEW,
            payload={"draft": True},
            result_reference="RESULT-002",
            reason_codes=(
                "human_review_required",
            ),
        )

        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        handler,
                }
            )
        )

        result = pipeline.execute(
            request()
        )

        self.assertEqual(
            result.decision,
            PipelineDecision.REVIEW,
        )

    def test_valid_abstention(self):
        handler = lambda _: GovernedCapabilityResult(
            capability=Capability.REQUIREMENTS,
            decision=PipelineDecision.ABSTAIN,
            payload=None,
            reason_codes=(
                "insufficient_evidence",
            ),
        )

        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        handler,
                }
            )
        )

        result = pipeline.execute(
            request()
        )

        self.assertTrue(
            result.execution_success
        )

        self.assertTrue(
            result.abstained
        )

    def test_missing_handler_abstains_safely(self):
        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator({})
        )

        result = pipeline.execute(
            request()
        )

        self.assertTrue(
            result.abstained
        )

        self.assertTrue(
            result.fallback_used
        )

        self.assertEqual(
            result.error_type,
            "handler_unavailable",
        )

    def test_invalid_handler_result_abstains(self):
        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        lambda _: "bad-result",
                }
            )
        )

        result = pipeline.execute(
            request()
        )

        self.assertTrue(
            result.abstained
        )

        self.assertEqual(
            result.error_type,
            "invalid_governed_result",
        )

    def test_capability_mismatch_abstains(self):
        handler = lambda _: GovernedCapabilityResult(
            capability=Capability.WEB,
            decision=PipelineDecision.READY,
            payload={"result": "wrong"},
        )

        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        handler,
                }
            )
        )

        result = pipeline.execute(
            request()
        )

        self.assertTrue(
            result.abstained
        )

        self.assertEqual(
            result.error_type,
            "capability_mismatch",
        )

    def test_payload_not_serialized(self):
        private = (
            "private government payload"
        )

        governed = GovernedCapabilityResult(
            capability=Capability.REQUIREMENTS,
            decision=PipelineDecision.READY,
            payload={"secret": private},
            result_reference="RESULT-003",
        )

        data = governed.to_dict()

        self.assertNotIn(
            private,
            repr(data),
        )

        self.assertNotIn(
            "payload",
            data,
        )

    def test_provenance_is_hashed(self):
        governed = GovernedCapabilityResult(
            capability=Capability.RAG,
            decision=PipelineDecision.READY,
            payload={"answer": "ok"},
            provenance_ids=(
                "DOCUMENT-PRIVATE-001",
            ),
        )

        data = governed.to_dict()

        self.assertEqual(
            data["provenance_count"],
            1,
        )

        self.assertNotIn(
            "DOCUMENT-PRIVATE-001",
            repr(data),
        )

    def test_duplicate_reasons_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GovernedCapabilityResult(
                capability=Capability.RAG,
                decision=PipelineDecision.REVIEW,
                payload={"answer": "draft"},
                reason_codes=(
                    "review",
                    "review",
                ),
            )

    def test_pipeline_identity_is_deterministic(self):
        def handler(_):
            return GovernedCapabilityResult(
                capability=Capability.REQUIREMENTS,
                decision=PipelineDecision.READY,
                payload={"result": "ok"},
                result_reference="RESULT-004",
            )

        pipeline = UnifiedGovernedPipeline(
            CapabilityOrchestrator(
                {
                    Capability.REQUIREMENTS:
                        handler,
                }
            )
        )

        value = request()

        first = pipeline.execute(
            value
        )

        second = pipeline.execute(
            value
        )

        self.assertEqual(
            first.pipeline_id,
            second.pipeline_id,
        )


if __name__ == "__main__":
    unittest.main()
