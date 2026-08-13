"""Offline tests for GovBA-GAR correspondence parsing and intent."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from govba.correspondence.classification import (
    CORRESPONDENCE_CLASSIFICATION_VERSION,
    CorrespondenceIntentSignal,
    analyze_correspondence_intent,
    classify_correspondence_intent,
    detect_intent_signals,
    parse_correspondence,
)
from govba.correspondence.contract import (
    CorrespondenceChannel,
    CorrespondenceDirection,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.rag.models import SourceLanguage


FIXED_TIME = datetime(
    2026,
    8,
    13,
    12,
    0,
    tzinfo=timezone.utc,
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
        received_at=FIXED_TIME,
        trace_id="TRACE-001",
    )


class TestCorrespondenceParsing(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_CLASSIFICATION_VERSION,
            "govba-correspondence-classification-v1",
        )

    def test_body_is_normalized(self):
        parsed = parse_correspondence(
            request(
                "Please    submit\r\n"
                "the report."
            )
        )

        self.assertEqual(
            parsed.normalized_body,
            "Please submit\n"
            "the report.",
        )

    def test_paragraphs_are_detected(self):
        parsed = parse_correspondence(
            request(
                "First paragraph.\n\n"
                "Second paragraph."
            )
        )

        self.assertEqual(
            parsed.paragraph_count,
            2,
        )

    def test_subject_is_normalized(self):
        parsed = parse_correspondence(
            request(
                "Body",
                subject=(
                    "  Monthly   Report  "
                ),
            )
        )

        self.assertEqual(
            parsed.normalized_subject,
            "Monthly Report",
        )

    def test_parser_id_is_deterministic(self):
        value = request(
            "Please submit the report."
        )

        self.assertEqual(
            parse_correspondence(
                value
            ).parser_id,
            parse_correspondence(
                value
            ).parser_id,
        )

    def test_default_serialization_excludes_text(self):
        raw = (
            "Unique private correspondence."
        )

        parsed = parse_correspondence(
            request(raw)
        )

        data = parsed.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "normalized_body",
            data,
        )

    def test_text_can_be_explicitly_serialized(self):
        parsed = parse_correspondence(
            request(
                "Synthetic body."
            )
        )

        data = parsed.to_dict(
            include_text=True
        )

        self.assertEqual(
            data[
                "normalized_body"
            ],
            "Synthetic body.",
        )


class TestIntentClassification(
    unittest.TestCase
):
    def test_request_intent(self):
        value = request(
            "Please provide the requested report."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.REQUEST,
        )

        self.assertTrue(
            result.requires_action
        )

    def test_action_required_intent(self):
        value = request(
            "Employees must submit "
            "the report."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent
            .ACTION_REQUIRED,
        )

    def test_approval_intent(self):
        value = request(
            "We request your approval "
            "for the attached proposal."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.APPROVAL,
        )

    def test_complaint_intent(self):
        value = request(
            "This is a formal complaint "
            "regarding the service."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.COMPLAINT,
        )

    def test_escalation_intent(self):
        value = request(
            "We are escalating this matter "
            "for immediate attention."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.ESCALATION,
        )

    def test_decision_intent(self):
        value = request(
            "We have decided to approve "
            "the revised structure."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.DECISION,
        )

    def test_notification_intent(self):
        value = request(
            "We hereby notify you that "
            "the office will relocate."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.NOTIFICATION,
        )

    def test_information_intent(self):
        value = request(
            "For your information, "
            "the report has been issued."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.INFORMATION,
        )

    def test_arabic_request_intent(self):
        value = request(
            "يرجى تزويدنا بالتقرير المطلوب.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.REQUEST,
        )

    def test_arabic_complaint_intent(self):
        value = request(
            "نتقدم بهذه الشكوى بخصوص الخدمة.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.COMPLAINT,
        )

    def test_subject_can_drive_intent(self):
        value = request(
            "Please see details below.",
            subject=(
                "Formal Complaint"
            ),
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.COMPLAINT,
        )

    def test_unknown_intent(self):
        value = request(
            "The meeting room is on "
            "the second floor."
        )

        result = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.intent,
            CorrespondenceIntent.UNKNOWN,
        )

        self.assertFalse(
            result.requires_action
        )

    def test_unknown_confidence_is_conservative(self):
        result = (
            classify_correspondence_intent(
                request(
                    "General correspondence."
                )
            )
        )

        self.assertEqual(
            result.confidence,
            0.25,
        )

    def test_priority_is_deferred_to_stage_15_4(self):
        result = (
            classify_correspondence_intent(
                request(
                    "This matter must be reviewed."
                )
            )
        )

        self.assertEqual(
            result.priority,
            CorrespondencePriority.NORMAL,
        )

    def test_signals_do_not_store_raw_text(self):
        parsed = parse_correspondence(
            request(
                "Please provide the report."
            )
        )

        signals = detect_intent_signals(
            parsed
        )

        self.assertIn(
            CorrespondenceIntentSignal.REQUEST,
            signals,
        )

        self.assertNotIn(
            "Please provide the report.",
            repr(signals),
        )

    def test_classification_is_deterministic(self):
        value = request(
            "Please provide the report."
        )

        first = (
            classify_correspondence_intent(
                value
            )
        )

        second = (
            classify_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            first.classification_id,
            second.classification_id,
        )


class TestCorrespondenceIntentAnalysis(
    unittest.TestCase
):
    def test_analysis_returns_contract_result(self):
        value = request(
            "Please provide the report."
        )

        result = (
            analyze_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            result.request_id,
            value.request_id,
        )

        self.assertEqual(
            result.trace_id,
            value.trace_id,
        )

        self.assertEqual(
            result.classification.intent,
            CorrespondenceIntent.REQUEST,
        )

    def test_analysis_id_is_deterministic(self):
        value = request(
            "Please provide the report."
        )

        first = (
            analyze_correspondence_intent(
                value
            )
        )

        second = (
            analyze_correspondence_intent(
                value
            )
        )

        self.assertEqual(
            first.result_id,
            second.result_id,
        )

    def test_result_serialization_contains_no_raw_text(self):
        raw = (
            "Please provide unique private material."
        )

        result = (
            analyze_correspondence_intent(
                request(raw)
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )


if __name__ == "__main__":
    unittest.main()
