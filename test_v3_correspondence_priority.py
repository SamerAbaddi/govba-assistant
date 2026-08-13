"""Offline tests for GovBA-GAR priority and deadline intelligence."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
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
    CORRESPONDENCE_PRIORITY_VERSION,
    CorrespondenceDeadlineKind,
    CorrespondencePriorityReason,
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
    subject="",
    language=SourceLanguage.ENGLISH,
):
    return CorrespondenceRequest(
        body_text=body,
        subject_text=subject,
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


def assess(
    value,
    *,
    assessed_at=RECEIVED,
):
    classification = (
        classify_correspondence_intent(
            value
        )
    )

    return assess_correspondence_priority(
        value,
        classification,
        assessed_at=assessed_at,
    )


class TestDeadlineExtraction(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_PRIORITY_VERSION,
            "govba-correspondence-priority-v1",
        )

    def test_relative_hours(self):
        value = request(
            "Please respond within 12 hours."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0].kind,
            CorrespondenceDeadlineKind
            .RELATIVE_HOURS,
        )

        self.assertEqual(
            result.deadlines[0].deadline_at,
            RECEIVED
            + timedelta(hours=12),
        )

    def test_relative_days(self):
        value = request(
            "Please submit within 3 days."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0].deadline_at,
            RECEIVED
            + timedelta(days=3),
        )

    def test_arabic_relative_days(self):
        value = request(
            "يرجى تزويدنا بالتقرير خلال ٣ أيام.",
            language=SourceLanguage.ARABIC,
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0].deadline_at,
            RECEIVED
            + timedelta(days=3),
        )

    def test_iso_absolute_date(self):
        value = request(
            "Please respond by 2026-08-20."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        deadline = (
            result.deadlines[0]
            .deadline_at
        )

        self.assertEqual(
            deadline.date().isoformat(),
            "2026-08-20",
        )

    def test_dmy_absolute_date(self):
        value = request(
            "Please respond by 20/08/2026."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0]
            .deadline_at.date()
            .isoformat(),
            "2026-08-20",
        )

    def test_arabic_absolute_date(self):
        value = request(
            "يرجى الرد بحلول ٢٠٢٦-٠٨-٢٠.",
            language=SourceLanguage.ARABIC,
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0]
            .deadline_at.date()
            .isoformat(),
            "2026-08-20",
        )

    def test_tomorrow(self):
        value = request(
            "Please submit by tomorrow."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.deadlines[0].kind,
            CorrespondenceDeadlineKind
            .TOMORROW,
        )

        self.assertEqual(
            result.deadlines[0]
            .deadline_at.date(),
            (
                RECEIVED.date()
                + timedelta(days=1)
            ),
        )

    def test_earliest_deadline_selected(self):
        value = request(
            "Please review within 3 days. "
            "Please respond within 12 hours."
        )

        result = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            result.earliest_deadline
            .deadline_at,
            RECEIVED
            + timedelta(hours=12),
        )

    def test_no_deadline_returns_empty(self):
        result = (
            extract_correspondence_deadlines(
                request(
                    "Please review the file."
                )
            )
        )

        self.assertEqual(
            result.deadlines,
            (),
        )

        self.assertIsNone(
            result.earliest_deadline
        )

    def test_extraction_is_deterministic(self):
        value = request(
            "Please respond within 2 days."
        )

        first = (
            extract_correspondence_deadlines(
                value
            )
        )

        second = (
            extract_correspondence_deadlines(
                value
            )
        )

        self.assertEqual(
            first.extraction_id,
            second.extraction_id,
        )

    def test_serialization_excludes_source_text(self):
        raw = (
            "Please respond within 12 hours."
        )

        result = (
            extract_correspondence_deadlines(
                request(raw)
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "source_text",
            data["deadlines"][0],
        )


class TestPriorityAssessment(
    unittest.TestCase
):
    def test_explicit_urgent(self):
        result = assess(
            request(
                "Urgent: please review the file."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.URGENT,
        )

        self.assertIn(
            CorrespondencePriorityReason
            .EXPLICIT_URGENT,
            result.reasons,
        )

    def test_arabic_urgent(self):
        result = assess(
            request(
                "عاجل: يرجى تزويدنا بالتقرير.",
                language=SourceLanguage.ARABIC,
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.URGENT,
        )

    def test_due_within_24_hours_is_urgent(self):
        result = assess(
            request(
                "Please respond within 12 hours."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.URGENT,
        )

    def test_due_within_72_hours_is_high(self):
        result = assess(
            request(
                "Please respond within 2 days."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.HIGH,
        )

    def test_overdue_is_urgent(self):
        value = request(
            "Please respond within 2 days."
        )

        result = assess(
            value,
            assessed_at=(
                RECEIVED
                + timedelta(days=3)
            ),
        )

        self.assertTrue(
            result.overdue
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.URGENT,
        )

    def test_escalation_is_high(self):
        result = assess(
            request(
                "We are escalating this matter "
                "for management attention."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.HIGH,
        )

    def test_information_without_action_is_routine(self):
        result = assess(
            request(
                "For your information, "
                "the report has been issued."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.ROUTINE,
        )

    def test_standard_request_is_normal(self):
        result = assess(
            request(
                "Please review the report."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.NORMAL,
        )

    def test_high_priority_phrase(self):
        result = assess(
            request(
                "This is a high priority matter."
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.HIGH,
        )

    def test_assessment_is_deterministic(self):
        value = request(
            "Please respond within 2 days."
        )

        first = assess(value)
        second = assess(value)

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

    def test_naive_assessment_time_rejected(self):
        value = request(
            "Please review the report."
        )

        classification = (
            classify_correspondence_intent(
                value
            )
        )

        with self.assertRaises(
            ValueError
        ):
            assess_correspondence_priority(
                value,
                classification,
                assessed_at=datetime(
                    2026,
                    8,
                    13,
                    10,
                    0,
                ),
            )

    def test_mismatched_classification_rejected(self):
        first = request(
            "Please review the report."
        )

        second = request(
            "Please submit the report."
        )

        classification = (
            classify_correspondence_intent(
                second
            )
        )

        with self.assertRaises(
            ValueError
        ):
            assess_correspondence_priority(
                first,
                classification,
                assessed_at=RECEIVED,
            )

    def test_serialization_contains_no_raw_text(self):
        raw = (
            "Urgent: please submit "
            "the unique private report."
        )

        result = assess(
            request(raw)
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "text",
            repr(
                data.keys()
            ),
        )


if __name__ == "__main__":
    unittest.main()
