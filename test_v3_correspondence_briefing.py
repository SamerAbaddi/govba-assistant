"""Offline tests for GovBA-GAR governed correspondence briefing."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from govba.correspondence.actions import (
    extract_correspondence_actions,
)
from govba.correspondence.briefing import (
    CORRESPONDENCE_BRIEFING_VERSION,
    CorrespondenceBriefingDecision,
    CorrespondenceBriefingPolicy,
    CorrespondenceBriefingReason,
    build_governed_correspondence_briefing,
)
from govba.correspondence.classification import (
    classify_correspondence_intent,
)
from govba.correspondence.contract import (
    CorrespondenceChannel,
    CorrespondenceDirection,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.correspondence.priority import (
    assess_correspondence_priority,
    extract_correspondence_deadlines,
)
from govba.rag.models import (
    SourceLanguage,
)


UTC = timezone.utc

RECEIVED = datetime(
    2026,
    8,
    13,
    10,
    0,
    tzinfo=UTC,
)


def request(
    body,
    *,
    language=SourceLanguage.ENGLISH,
):
    return CorrespondenceRequest(
        body_text=body,
        language=language,
        direction=(
            CorrespondenceDirection.INBOUND
        ),
        channel=(
            CorrespondenceChannel.EMAIL
        ),
        received_at=RECEIVED,
        trace_id="TRACE-001",
    )


def build(
    value,
    *,
    assessed_at=RECEIVED,
    policy=None,
):
    classification = (
        classify_correspondence_intent(
            value
        )
    )

    actions = (
        extract_correspondence_actions(
            value
        )
    )

    deadlines = (
        extract_correspondence_deadlines(
            value
        )
    )

    priority = (
        assess_correspondence_priority(
            value,
            classification,
            assessed_at=assessed_at,
        )
    )

    return (
        build_governed_correspondence_briefing(
            value,
            classification,
            actions,
            deadlines,
            priority,
            policy=policy,
        )
    )


class TestCorrespondenceBriefing(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_BRIEFING_VERSION,
            "govba-correspondence-briefing-v1",
        )

    def test_standard_verified_request_is_ready(self):
        briefing = build(
            request(
                "Please review the report."
            )
        )

        self.assertEqual(
            briefing.decision,
            CorrespondenceBriefingDecision.READY,
        )

        self.assertIn(
            CorrespondenceBriefingReason.VERIFIED,
            briefing.reasons,
        )

    def test_action_is_linked(self):
        briefing = build(
            request(
                "Please submit the report."
            )
        )

        self.assertEqual(
            len(briefing.actions),
            1,
        )

        self.assertTrue(
            briefing.actions[0].action_id
        )

    def test_deadline_is_linked(self):
        briefing = build(
            request(
                "Please respond within 2 days."
            )
        )

        self.assertEqual(
            len(briefing.deadlines),
            1,
        )

    def test_urgent_requires_review(self):
        briefing = build(
            request(
                "Urgent: please review the report."
            )
        )

        self.assertTrue(
            briefing.requires_review
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .URGENT_CORRESPONDENCE,
            briefing.reasons,
        )

    def test_overdue_requires_review(self):
        value = request(
            "Please respond within 1 day."
        )

        briefing = build(
            value,
            assessed_at=(
                RECEIVED
                + timedelta(days=2)
            ),
        )

        self.assertTrue(
            briefing.requires_review
        )

        self.assertTrue(
            briefing.overdue
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .OVERDUE_DEADLINE,
            briefing.reasons,
        )

    def test_unknown_intent_abstains(self):
        briefing = build(
            request(
                "The office is on the second floor."
            )
        )

        self.assertTrue(
            briefing.abstained
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .UNKNOWN_INTENT,
            briefing.reasons,
        )

    def test_action_expected_but_missing_requires_review(self):
        briefing = build(
            request(
                "This is a formal complaint "
                "regarding the service."
            )
        )

        self.assertTrue(
            briefing.requires_review
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .ACTION_EXPECTED_BUT_NOT_EXTRACTED,
            briefing.reasons,
        )

    def test_information_without_action_is_ready(self):
        briefing = build(
            request(
                "For your information, "
                "the report has been issued."
            )
        )

        self.assertTrue(
            briefing.ready
        )

        self.assertFalse(
            briefing.requires_action
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .NO_ACTION_REQUIRED,
            briefing.reasons,
        )

    def test_custom_action_threshold_can_force_review(self):
        policy = (
            CorrespondenceBriefingPolicy(
                min_action_confidence=0.90
            )
        )

        briefing = build(
            request(
                "Please review the report."
            ),
            policy=policy,
        )

        self.assertTrue(
            briefing.requires_review
        )

        self.assertIn(
            CorrespondenceBriefingReason
            .LOW_ACTION_CONFIDENCE,
            briefing.reasons,
        )

    def test_high_priority_review_is_configurable(self):
        value = request(
            "Please respond within 2 days."
        )

        policy = (
            CorrespondenceBriefingPolicy(
                review_high_priority=True
            )
        )

        briefing = build(
            value,
            policy=policy,
        )

        self.assertEqual(
            briefing.priority,
            CorrespondencePriority.HIGH,
        )

        self.assertTrue(
            briefing.requires_review
        )

    def test_policy_id_is_deterministic(self):
        self.assertEqual(
            CorrespondenceBriefingPolicy().policy_id,
            CorrespondenceBriefingPolicy().policy_id,
        )

    def test_briefing_id_is_deterministic(self):
        value = request(
            "Please review the report."
        )

        first = build(value)
        second = build(value)

        self.assertEqual(
            first.briefing_id,
            second.briefing_id,
        )

    def test_arabic_request_is_ready(self):
        briefing = build(
            request(
                "يرجى تزويدنا بالتقرير.",
                language=(
                    SourceLanguage.ARABIC
                ),
            )
        )

        self.assertTrue(
            briefing.ready
        )

    def test_serialization_contains_no_raw_text(self):
        raw = (
            "Please submit the unique "
            "private government report."
        )

        briefing = build(
            request(raw)
        )

        data = briefing.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "action_text",
            repr(data),
        )

        self.assertNotIn(
            "source_text",
            repr(data),
        )

    def test_request_mismatch_is_rejected(self):
        first = request(
            "Please review the report."
        )

        second = request(
            "Please submit the report."
        )

        classification = (
            classify_correspondence_intent(
                first
            )
        )

        actions = (
            extract_correspondence_actions(
                second
            )
        )

        deadlines = (
            extract_correspondence_deadlines(
                first
            )
        )

        priority = (
            assess_correspondence_priority(
                first,
                classification,
                assessed_at=RECEIVED,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_correspondence_briefing(
                first,
                classification,
                actions,
                deadlines,
                priority,
            )

    def test_priority_deadline_provenance_mismatch_rejected(self):
        first = request(
            "Please respond within 2 days."
        )

        second = request(
            "Please respond within 3 days."
        )

        classification = (
            classify_correspondence_intent(
                first
            )
        )

        actions = (
            extract_correspondence_actions(
                first
            )
        )

        wrong_deadlines = (
            extract_correspondence_deadlines(
                second
            )
        )

        priority = (
            assess_correspondence_priority(
                first,
                classification,
                assessed_at=RECEIVED,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_correspondence_briefing(
                first,
                classification,
                actions,
                wrong_deadlines,
                priority,
            )


if __name__ == "__main__":
    unittest.main()
