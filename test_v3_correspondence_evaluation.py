"""Offline tests for GovBA-GAR correspondence evaluation."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from govba.correspondence.briefing import (
    CorrespondenceBriefingDecision,
)
from govba.correspondence.contract import (
    CorrespondenceChannel,
    CorrespondenceDirection,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.correspondence.evaluation import (
    CORRESPONDENCE_EVALUATION_VERSION,
    CorrespondenceEvaluationGoldCase,
    evaluate_correspondence_intelligence,
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
        trace_id="TRACE-EVAL",
    )


def gold(
    *,
    case_id="REQUEST",
    body="Please review the report.",
    language=SourceLanguage.ENGLISH,
    assessed_at=RECEIVED,
    intent=CorrespondenceIntent.REQUEST,
    priority=CorrespondencePriority.NORMAL,
    action_count=1,
    commitment_count=0,
    deadline_count=0,
    briefing=CorrespondenceBriefingDecision.READY,
):
    return CorrespondenceEvaluationGoldCase(
        case_id=case_id,
        request=request(
            body,
            language=language,
        ),
        assessed_at=assessed_at,
        expected_intent=intent,
        expected_priority=priority,
        expected_action_count=action_count,
        expected_commitment_count=(
            commitment_count
        ),
        expected_deadline_count=(
            deadline_count
        ),
        expected_briefing_decision=(
            briefing
        ),
    )


class TestCorrespondenceEvaluation(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_EVALUATION_VERSION,
            "govba-correspondence-evaluation-v1",
        )

    def test_standard_request_passes(self):
        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    gold(),
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_urgent_case(self):
        case = gold(
            case_id="URGENT",
            body=(
                "Urgent: please submit "
                "the report."
            ),
            priority=(
                CorrespondencePriority.URGENT
            ),
            briefing=(
                CorrespondenceBriefingDecision.REVIEW
            ),
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_information_case(self):
        case = gold(
            case_id="INFO",
            body=(
                "For your information, "
                "the report has been issued."
            ),
            intent=(
                CorrespondenceIntent.INFORMATION
            ),
            priority=(
                CorrespondencePriority.ROUTINE
            ),
            action_count=0,
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_deadline_case(self):
        case = gold(
            case_id="DEADLINE",
            body=(
                "Please respond within 2 days."
            ),
            priority=(
                CorrespondencePriority.HIGH
            ),
            deadline_count=1,
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_overdue_case(self):
        case = gold(
            case_id="OVERDUE",
            body=(
                "Please respond within 1 day."
            ),
            assessed_at=(
                RECEIVED
                + timedelta(days=2)
            ),
            priority=(
                CorrespondencePriority.URGENT
            ),
            deadline_count=1,
            briefing=(
                CorrespondenceBriefingDecision.REVIEW
            ),
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_unknown_case_abstains(self):
        case = gold(
            case_id="UNKNOWN",
            body=(
                "The office is on "
                "the second floor."
            ),
            intent=(
                CorrespondenceIntent.UNKNOWN
            ),
            priority=(
                CorrespondencePriority.NORMAL
            ),
            action_count=0,
            briefing=(
                CorrespondenceBriefingDecision.ABSTAIN
            ),
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_arabic_request(self):
        case = gold(
            case_id="AR",
            body=(
                "يرجى تزويدنا بالتقرير."
            ),
            language=(
                SourceLanguage.ARABIC
            ),
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertTrue(
            benchmark.cases[
                0
            ].case_success
        )

    def test_perfect_metrics(self):
        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    gold(
                        case_id="A"
                    ),
                    gold(
                        case_id="B",
                        body=(
                            "Urgent: please review "
                            "the report."
                        ),
                        priority=(
                            CorrespondencePriority.URGENT
                        ),
                        briefing=(
                            CorrespondenceBriefingDecision
                            .REVIEW
                        ),
                    ),
                )
            )
        )

        self.assertEqual(
            benchmark.metrics.intent_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics.priority_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics.action_count_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics.briefing_accuracy,
            1.0,
        )

        self.assertEqual(
            benchmark.metrics
            .end_to_end_success_rate,
            1.0,
        )

    def test_wrong_gold_label_fails_case(self):
        case = gold(
            case_id="WRONG",
            priority=(
                CorrespondencePriority.HIGH
            ),
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].priority_correct
        )

        self.assertFalse(
            benchmark.cases[
                0
            ].case_success
        )

    def test_duplicate_case_ids_rejected(self):
        case = gold(
            case_id="DUP"
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_correspondence_intelligence(
                (
                    case,
                    case,
                )
            )

    def test_empty_cases_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_correspondence_intelligence(
                ()
            )

    def test_naive_assessment_time_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                assessed_at=datetime(
                    2026,
                    8,
                    13,
                    10,
                    0,
                )
            )

    def test_invalid_commitment_count_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                action_count=0,
                commitment_count=1,
            )

    def test_benchmark_identity_is_deterministic(self):
        case = gold(
            case_id="DET"
        )

        first = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.benchmark_id,
            second.benchmark_id,
        )

    def test_latency_does_not_change_identity(self):
        case = gold(
            case_id="LATENCY"
        )

        first = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        second = (
            evaluate_correspondence_intelligence(
                (
                    case,
                )
            )
        )

        self.assertEqual(
            first.cases[
                0
            ].result_id,
            second.cases[
                0
            ].result_id,
        )

    def test_serialization_excludes_raw_text(self):
        raw = (
            "Please review unique "
            "private correspondence."
        )

        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    gold(
                        case_id="PRIVACY",
                        body=raw,
                    ),
                )
            )
        )

        data = benchmark.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "body_text",
            repr(data),
        )

    def test_latency_is_recorded(self):
        benchmark = (
            evaluate_correspondence_intelligence(
                (
                    gold(),
                )
            )
        )

        self.assertGreaterEqual(
            benchmark.cases[
                0
            ].latency_ms,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
