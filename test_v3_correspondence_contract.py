"""Offline tests for GovBA-GAR correspondence contract."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from govba.correspondence.contract import (
    CORRESPONDENCE_CONTRACT_VERSION,
    CorrespondenceChannel,
    CorrespondenceClassification,
    CorrespondenceDirection,
    CorrespondenceIntelligenceResult,
    CorrespondenceIntent,
    CorrespondencePriority,
    CorrespondenceRequest,
)
from govba.rag.models import (
    SourceLanguage,
)


FIXED_TIME = datetime(
    2026,
    8,
    13,
    12,
    0,
    tzinfo=timezone.utc,
)


def request(
    *,
    body_text="Please submit the report.",
    subject_text="Monthly Report",
):
    return CorrespondenceRequest(
        body_text=body_text,
        subject_text=subject_text,
        language=(
            SourceLanguage.ENGLISH
        ),
        direction=(
            CorrespondenceDirection.INBOUND
        ),
        channel=(
            CorrespondenceChannel.EMAIL
        ),
        received_at=FIXED_TIME,
        trace_id="TRACE-001",
        source_reference="REF-001",
    )


def classification(
    request_id,
):
    return CorrespondenceClassification(
        request_id=request_id,
        intent=(
            CorrespondenceIntent
            .ACTION_REQUIRED
        ),
        priority=(
            CorrespondencePriority.NORMAL
        ),
        requires_action=True,
        confidence=0.9,
        reason_codes=(
            "explicit_request",
        ),
    )


class TestCorrespondenceRequest(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_CONTRACT_VERSION,
            "govba-correspondence-contract-v1",
        )

    def test_request_is_created(self):
        value = request()

        self.assertEqual(
            value.body_text,
            "Please submit the report.",
        )

    def test_body_hash_is_sha256(self):
        value = request()

        self.assertEqual(
            len(
                value.body_sha256
            ),
            64,
        )

        int(
            value.body_sha256,
            16,
        )

    def test_request_id_is_deterministic(self):
        self.assertEqual(
            request().request_id,
            request().request_id,
        )

    def test_body_change_changes_identity(self):
        first = request(
            body_text="Version one"
        )

        second = request(
            body_text="Version two"
        )

        self.assertNotEqual(
            first.request_id,
            second.request_id,
        )

    def test_blank_body_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            request(
                body_text=" "
            )

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            CorrespondenceRequest(
                body_text="Body",
                language="en",
                direction=(
                    CorrespondenceDirection
                    .INBOUND
                ),
                channel=(
                    CorrespondenceChannel.EMAIL
                ),
                received_at=FIXED_TIME,
                trace_id="TRACE",
            )

    def test_naive_datetime_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            CorrespondenceRequest(
                body_text="Body",
                language=(
                    SourceLanguage.ENGLISH
                ),
                direction=(
                    CorrespondenceDirection
                    .INBOUND
                ),
                channel=(
                    CorrespondenceChannel.EMAIL
                ),
                received_at=datetime(
                    2026,
                    8,
                    13,
                    12,
                    0,
                ),
                trace_id="TRACE",
            )

    def test_default_serialization_excludes_content(self):
        raw_body = (
            "Unique sensitive correspondence body."
        )

        raw_subject = (
            "Sensitive subject"
        )

        value = request(
            body_text=raw_body,
            subject_text=raw_subject,
        )

        data = value.to_dict()

        self.assertNotIn(
            raw_body,
            repr(data),
        )

        self.assertNotIn(
            raw_subject,
            repr(data),
        )

        self.assertNotIn(
            "body_text",
            data,
        )

    def test_content_can_be_explicitly_serialized(self):
        value = request()

        data = value.to_dict(
            include_content=True
        )

        self.assertEqual(
            data[
                "body_text"
            ],
            value.body_text,
        )


class TestCorrespondenceClassification(
    unittest.TestCase
):
    def test_classification_is_created(self):
        value = request()

        classified = classification(
            value.request_id
        )

        self.assertTrue(
            classified.requires_action
        )

    def test_classification_id_is_deterministic(self):
        value = request()

        first = classification(
            value.request_id
        )

        second = classification(
            value.request_id
        )

        self.assertEqual(
            first.classification_id,
            second.classification_id,
        )

    def test_invalid_confidence_is_rejected(self):
        value = request()

        with self.assertRaises(
            ValueError
        ):
            CorrespondenceClassification(
                request_id=(
                    value.request_id
                ),
                intent=(
                    CorrespondenceIntent.REQUEST
                ),
                priority=(
                    CorrespondencePriority.NORMAL
                ),
                requires_action=True,
                confidence=1.5,
            )

    def test_duplicate_reason_codes_are_rejected(self):
        value = request()

        with self.assertRaises(
            ValueError
        ):
            CorrespondenceClassification(
                request_id=(
                    value.request_id
                ),
                intent=(
                    CorrespondenceIntent.REQUEST
                ),
                priority=(
                    CorrespondencePriority.NORMAL
                ),
                requires_action=True,
                confidence=0.8,
                reason_codes=(
                    "request",
                    "request",
                ),
            )


class TestCorrespondenceResult(
    unittest.TestCase
):
    def test_result_is_created(self):
        value = request()

        classified = classification(
            value.request_id
        )

        result = (
            CorrespondenceIntelligenceResult(
                request_id=(
                    value.request_id
                ),
                trace_id=(
                    value.trace_id
                ),
                classification=classified,
            )
        )

        self.assertEqual(
            result.request_id,
            value.request_id,
        )

    def test_result_id_is_deterministic(self):
        value = request()

        classified = classification(
            value.request_id
        )

        first = (
            CorrespondenceIntelligenceResult(
                request_id=value.request_id,
                trace_id=value.trace_id,
                classification=classified,
            )
        )

        second = (
            CorrespondenceIntelligenceResult(
                request_id=value.request_id,
                trace_id=value.trace_id,
                classification=classified,
            )
        )

        self.assertEqual(
            first.result_id,
            second.result_id,
        )

    def test_mismatched_classification_is_rejected(self):
        value = request()

        classified = classification(
            "OTHER-REQUEST"
        )

        with self.assertRaises(
            ValueError
        ):
            CorrespondenceIntelligenceResult(
                request_id=value.request_id,
                trace_id=value.trace_id,
                classification=classified,
            )

    def test_result_serialization_has_no_raw_text(self):
        raw = (
            "Unique private correspondence content."
        )

        value = request(
            body_text=raw
        )

        result = (
            CorrespondenceIntelligenceResult(
                request_id=(
                    value.request_id
                ),
                trace_id=(
                    value.trace_id
                ),
                classification=(
                    classification(
                        value.request_id
                    )
                ),
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )


if __name__ == "__main__":
    unittest.main()
