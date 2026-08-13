"""Offline tests for GovBA-GAR governance verification contract."""

from __future__ import annotations

import unittest

from govba.governance.verification import (
    GOVERNANCE_VERIFICATION_VERSION,
    GovernanceDecision,
    GovernanceReasonCode,
    GovernanceVerificationResult,
)


class TestGovernanceVerification(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GOVERNANCE_VERIFICATION_VERSION,
            "govba-governance-verification-v1",
        )

    def test_allow_decision(self):
        result = GovernanceVerificationResult(
            trace_id="TRACE-001",
            decision=(
                GovernanceDecision.ALLOW
            ),
            reason_codes=(
                GovernanceReasonCode.VERIFIED,
            ),
            evidence_count=2,
            eligible_evidence_count=2,
        )

        self.assertTrue(
            result.allowed
        )

        self.assertFalse(
            result.should_abstain
        )

    def test_abstain_decision(self):
        result = GovernanceVerificationResult(
            trace_id="TRACE-001",
            decision=(
                GovernanceDecision.ABSTAIN
            ),
            reason_codes=(
                GovernanceReasonCode
                .GROUNDING_FAILURE,
            ),
            evidence_count=1,
            eligible_evidence_count=0,
        )

        self.assertFalse(
            result.allowed
        )

        self.assertTrue(
            result.should_abstain
        )

    def test_blank_trace_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="  ",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .GROUNDING_FAILURE,
                ),
            )

    def test_reason_codes_are_required(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(),
            )

    def test_duplicate_reasons_are_rejected(self):
        reason = (
            GovernanceReasonCode
            .GROUNDING_FAILURE
        )

        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(
                    reason,
                    reason,
                ),
            )

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .GROUNDING_FAILURE,
                ),
                evidence_count=-1,
            )

    def test_eligible_count_cannot_exceed_total(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .GROUNDING_FAILURE,
                ),
                evidence_count=1,
                eligible_evidence_count=2,
            )

    def test_allow_requires_eligible_evidence(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ALLOW
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .VERIFIED,
                ),
                evidence_count=1,
                eligible_evidence_count=0,
            )

    def test_allow_rejects_failure_reason(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ALLOW
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .GROUNDING_FAILURE,
                ),
                evidence_count=1,
                eligible_evidence_count=1,
            )

    def test_abstain_rejects_verified_reason(self):
        with self.assertRaises(
            ValueError
        ):
            GovernanceVerificationResult(
                trace_id="TRACE-001",
                decision=(
                    GovernanceDecision.ABSTAIN
                ),
                reason_codes=(
                    GovernanceReasonCode
                    .VERIFIED,
                ),
                evidence_count=1,
                eligible_evidence_count=1,
            )

    def test_decision_id_is_sha256(self):
        result = GovernanceVerificationResult(
            trace_id="TRACE-001",
            decision=(
                GovernanceDecision.ABSTAIN
            ),
            reason_codes=(
                GovernanceReasonCode
                .TEMPORAL_CONFLICT,
            ),
        )

        self.assertEqual(
            len(
                result.decision_id
            ),
            64,
        )

        int(
            result.decision_id,
            16,
        )

    def test_decision_id_is_deterministic(self):
        kwargs = {
            "trace_id": "TRACE-001",
            "decision": (
                GovernanceDecision.ABSTAIN
            ),
            "reason_codes": (
                GovernanceReasonCode
                .TEMPORAL_CONFLICT,
                GovernanceReasonCode
                .GROUNDING_FAILURE,
            ),
        }

        first = GovernanceVerificationResult(
            **kwargs
        )

        second = GovernanceVerificationResult(
            **kwargs
        )

        self.assertEqual(
            first.decision_id,
            second.decision_id,
        )

    def test_serialization_is_privacy_safe(self):
        result = GovernanceVerificationResult(
            trace_id="TRACE-001",
            decision=(
                GovernanceDecision.ALLOW
            ),
            reason_codes=(
                GovernanceReasonCode.VERIFIED,
            ),
            evidence_count=3,
            eligible_evidence_count=2,
        )

        data = result.to_dict()

        self.assertEqual(
            data["decision"],
            "allow",
        )

        self.assertEqual(
            data["reason_codes"],
            [
                "verified",
            ],
        )

        self.assertNotIn(
            "evidence_text",
            data,
        )

        self.assertNotIn(
            "prompt",
            data,
        )


if __name__ == "__main__":
    unittest.main()
