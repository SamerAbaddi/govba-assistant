"""Offline tests for GovBA-GAR web freshness and provenance controls."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
)
from govba.rag.models import (
    SourceLanguage,
)
from govba.web.contract import (
    OfficialWebResult,
)
from govba.web.evidence_bridge import (
    WebEvidenceMetadata,
    bridge_official_web_result,
)
from govba.web.freshness import (
    WEB_FRESHNESS_VERSION,
    WebEvidenceControlDecision,
    WebEvidenceControlIssueCode,
    WebEvidenceControlPolicy,
    WebFreshnessState,
    WebProvenanceState,
    assess_web_evidence_controls,
)


BASE_TIME = datetime(
    2026,
    8,
    13,
    10,
    0,
    tzinfo=timezone.utc,
)


def source_policy():
    return SourceAllowlistPolicy(
        allowed_domains=(
            "example.gov.jo",
            "agency.gov.jo",
        )
    )


def web_result(
    *,
    url="https://example.gov.jo/policy",
    text="Synthetic official policy content.",
    retrieved_at=BASE_TIME,
):
    return OfficialWebResult(
        title="Synthetic Policy",
        url=url,
        text=text,
        rank=1,
        score=0.9,
        retrieval_method=(
            "controlled-web-test-v1"
        ),
        retrieved_at=retrieved_at,
    )


def metadata():
    return WebEvidenceMetadata(
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
    )


def bridge_for(
    result,
    *,
    trace_id="TRACE-001",
):
    return bridge_official_web_result(
        result,
        metadata=metadata(),
        policy=source_policy(),
        trace_id=trace_id,
    )


def freshness_policy(
    *,
    max_age_hours=24,
    future_tolerance_minutes=5,
):
    return WebEvidenceControlPolicy(
        max_age_hours=max_age_hours,
        future_tolerance_minutes=(
            future_tolerance_minutes
        ),
    )


class TestWebFreshnessPolicy(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            WEB_FRESHNESS_VERSION,
            "govba-web-freshness-v1",
        )

    def test_policy_id_is_deterministic(self):
        first = freshness_policy()
        second = freshness_policy()

        self.assertEqual(
            first.policy_id,
            second.policy_id,
        )

    def test_invalid_max_age_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            WebEvidenceControlPolicy(
                max_age_hours=0
            )

    def test_negative_future_tolerance_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            WebEvidenceControlPolicy(
                max_age_hours=24,
                future_tolerance_minutes=-1,
            )


class TestWebFreshnessControls(
    unittest.TestCase
):
    def test_fresh_verified_evidence_is_allowed(self):
        raw = web_result()

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=2
                    )
                ),
            )
        )

        self.assertEqual(
            assessment.decision,
            WebEvidenceControlDecision.ALLOW,
        )

        self.assertEqual(
            assessment.freshness_state,
            WebFreshnessState.FRESH,
        )

        self.assertEqual(
            assessment.provenance_state,
            WebProvenanceState.VERIFIED,
        )

    def test_stale_evidence_requires_review(self):
        raw = web_result()

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy(
                        max_age_hours=24
                    )
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=25
                    )
                ),
            )
        )

        self.assertEqual(
            assessment.freshness_state,
            WebFreshnessState.STALE,
        )

        self.assertEqual(
            assessment.decision,
            WebEvidenceControlDecision.REVIEW,
        )

        self.assertTrue(
            assessment.requires_review
        )

    def test_exact_max_age_is_still_fresh(self):
        raw = web_result()

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy(
                        max_age_hours=24
                    )
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=24
                    )
                ),
            )
        )

        self.assertEqual(
            assessment.freshness_state,
            WebFreshnessState.FRESH,
        )

    def test_future_retrieval_blocks(self):
        raw = web_result(
            retrieved_at=(
                BASE_TIME
                + timedelta(
                    minutes=10
                )
            )
        )

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy(
                        future_tolerance_minutes=5
                    )
                ),
                trace_id="TRACE-001",
                assessed_at=BASE_TIME,
            )
        )

        self.assertEqual(
            assessment.freshness_state,
            WebFreshnessState.FUTURE,
        )

        self.assertEqual(
            assessment.decision,
            WebEvidenceControlDecision.BLOCK,
        )

        self.assertTrue(
            assessment.should_block
        )

    def test_small_clock_skew_is_tolerated(self):
        raw = web_result(
            retrieved_at=(
                BASE_TIME
                + timedelta(
                    minutes=3
                )
            )
        )

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy(
                        future_tolerance_minutes=5
                    )
                ),
                trace_id="TRACE-001",
                assessed_at=BASE_TIME,
            )
        )

        self.assertEqual(
            assessment.freshness_state,
            WebFreshnessState.FRESH,
        )

        self.assertEqual(
            assessment.decision,
            WebEvidenceControlDecision.ALLOW,
        )

    def test_content_hash_mismatch_blocks(self):
        original = web_result(
            text="Version one"
        )

        bridge = bridge_for(
            original
        )

        changed = web_result(
            text="Version two"
        )

        assessment = (
            assess_web_evidence_controls(
                changed,
                bridge=bridge,
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=BASE_TIME,
            )
        )

        self.assertEqual(
            assessment.provenance_state,
            WebProvenanceState.INVALID,
        )

        self.assertTrue(
            assessment.should_block
        )

        self.assertIn(
            (
                WebEvidenceControlIssueCode
                .CONTENT_HASH_MISMATCH
            ),
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_url_mismatch_blocks(self):
        original = web_result()

        bridge = bridge_for(
            original
        )

        changed = web_result(
            url=(
                "https://agency.gov.jo/"
                "policy"
            )
        )

        assessment = (
            assess_web_evidence_controls(
                changed,
                bridge=bridge,
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=BASE_TIME,
            )
        )

        self.assertTrue(
            assessment.should_block
        )

        self.assertIn(
            (
                WebEvidenceControlIssueCode
                .SOURCE_URL_MISMATCH
            ),
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_retrieval_time_mismatch_blocks(self):
        original = web_result()

        bridge = bridge_for(
            original
        )

        changed = web_result(
            retrieved_at=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            )
        )

        assessment = (
            assess_web_evidence_controls(
                changed,
                bridge=bridge,
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=1
                    )
                ),
            )
        )

        self.assertIn(
            (
                WebEvidenceControlIssueCode
                .RETRIEVAL_TIME_MISMATCH
            ),
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

        self.assertTrue(
            assessment.should_block
        )

    def test_provenance_failure_overrides_stale_review(self):
        original = web_result(
            text="Original"
        )

        bridge = bridge_for(
            original
        )

        changed = web_result(
            text="Changed"
        )

        assessment = (
            assess_web_evidence_controls(
                changed,
                bridge=bridge,
                policy=(
                    freshness_policy(
                        max_age_hours=1
                    )
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=3
                    )
                ),
            )
        )

        self.assertEqual(
            assessment.decision,
            WebEvidenceControlDecision.BLOCK,
        )

    def test_age_seconds_is_deterministic(self):
        raw = web_result()

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=(
                    BASE_TIME
                    + timedelta(
                        hours=2
                    )
                ),
            )
        )

        self.assertEqual(
            assessment.age_seconds,
            7200.0,
        )

    def test_assessment_id_is_deterministic(self):
        raw = web_result()

        kwargs = {
            "bridge": bridge_for(
                raw
            ),
            "policy": (
                freshness_policy()
            ),
            "trace_id": "TRACE-001",
            "assessed_at": (
                BASE_TIME
                + timedelta(
                    hours=1
                )
            ),
        }

        first = (
            assess_web_evidence_controls(
                raw,
                **kwargs,
            )
        )

        second = (
            assess_web_evidence_controls(
                raw,
                **kwargs,
            )
        )

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

    def test_serialization_contains_no_evidence_text(self):
        raw_text = (
            "Unique synthetic official text."
        )

        raw = web_result(
            text=raw_text
        )

        assessment = (
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=BASE_TIME,
            )
        )

        data = assessment.to_dict()

        self.assertNotIn(
            raw_text,
            repr(data),
        )

        self.assertNotIn(
            raw.url,
            repr(data),
        )

    def test_trace_mismatch_is_rejected(self):
        raw = web_result()

        bridge = bridge_for(
            raw,
            trace_id="TRACE-A",
        )

        with self.assertRaises(
            ValueError
        ):
            assess_web_evidence_controls(
                raw,
                bridge=bridge,
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-B",
                assessed_at=BASE_TIME,
            )

    def test_naive_assessment_time_is_rejected(self):
        raw = web_result()

        with self.assertRaises(
            ValueError
        ):
            assess_web_evidence_controls(
                raw,
                bridge=bridge_for(
                    raw
                ),
                policy=(
                    freshness_policy()
                ),
                trace_id="TRACE-001",
                assessed_at=datetime(
                    2026,
                    8,
                    13,
                    12,
                    0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
