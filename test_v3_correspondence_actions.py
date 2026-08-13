"""Offline tests for GovBA-GAR correspondence action extraction."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from govba.correspondence.actions import (
    CORRESPONDENCE_ACTION_VERSION,
    CorrespondenceActionKind,
    CorrespondenceActionNature,
    CorrespondenceActionOwner,
    extract_correspondence_actions,
)
from govba.correspondence.classification import (
    parse_correspondence,
)
from govba.correspondence.contract import (
    CorrespondenceChannel,
    CorrespondenceDirection,
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
        received_at=FIXED_TIME,
        trace_id="TRACE-001",
    )


class TestCorrespondenceActions(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CORRESPONDENCE_ACTION_VERSION,
            "govba-correspondence-action-v1",
        )

    def test_polite_request_is_extracted(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please submit the report."
                )
            )
        )

        self.assertEqual(
            result.action_count,
            1,
        )

        action = result.actions[0]

        self.assertEqual(
            action.kind,
            CorrespondenceActionKind.SUBMIT,
        )

        self.assertEqual(
            action.owner,
            CorrespondenceActionOwner.RECIPIENT,
        )

        self.assertEqual(
            action.nature,
            CorrespondenceActionNature.REQUESTED,
        )

    def test_required_action_is_extracted(self):
        result = (
            extract_correspondence_actions(
                request(
                    "You must provide the "
                    "supporting documents."
                )
            )
        )

        self.assertEqual(
            result.required_count,
            1,
        )

        self.assertEqual(
            result.actions[0].kind,
            CorrespondenceActionKind.PROVIDE,
        )

    def test_sender_commitment_is_extracted(self):
        result = (
            extract_correspondence_actions(
                request(
                    "We will review the proposal."
                )
            )
        )

        action = result.actions[0]

        self.assertTrue(
            action.is_commitment
        )

        self.assertEqual(
            action.owner,
            CorrespondenceActionOwner.SENDER,
        )

        self.assertEqual(
            action.kind,
            CorrespondenceActionKind.REVIEW,
        )

    def test_multiple_actions_in_sentence(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please review and approve "
                    "the application."
                )
            )
        )

        self.assertEqual(
            result.action_count,
            2,
        )

        self.assertEqual(
            tuple(
                action.kind
                for action
                in result.actions
            ),
            (
                CorrespondenceActionKind.REVIEW,
                CorrespondenceActionKind.APPROVE,
            ),
        )

    def test_multiple_sentences_are_extracted(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please review the file. "
                    "You must respond to the ministry."
                )
            )
        )

        self.assertEqual(
            result.action_count,
            2,
        )

        self.assertEqual(
            result.actions[1].kind,
            CorrespondenceActionKind.RESPOND,
        )

    def test_non_action_sentence_is_ignored(self):
        result = (
            extract_correspondence_actions(
                request(
                    "The meeting room is "
                    "on the second floor."
                )
            )
        )

        self.assertEqual(
            result.actions,
            (),
        )

    def test_unknown_requested_action_becomes_other(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please consider this matter."
                )
            )
        )

        self.assertEqual(
            result.actions[0].kind,
            CorrespondenceActionKind.OTHER,
        )

    def test_arabic_requested_action(self):
        result = (
            extract_correspondence_actions(
                request(
                    "يرجى تزويدنا بالتقرير.",
                    language=(
                        SourceLanguage.ARABIC
                    ),
                )
            )
        )

        action = result.actions[0]

        self.assertEqual(
            action.kind,
            CorrespondenceActionKind.PROVIDE,
        )

        self.assertEqual(
            action.nature,
            CorrespondenceActionNature.REQUESTED,
        )

    def test_arabic_required_action(self):
        result = (
            extract_correspondence_actions(
                request(
                    "يجب عليكم سداد الرسوم.",
                    language=(
                        SourceLanguage.ARABIC
                    ),
                )
            )
        )

        self.assertEqual(
            result.actions[0].kind,
            CorrespondenceActionKind.PAY,
        )

        self.assertEqual(
            result.actions[0].nature,
            CorrespondenceActionNature.REQUIRED,
        )

    def test_arabic_sender_commitment(self):
        result = (
            extract_correspondence_actions(
                request(
                    "سنقوم بإبلاغكم بالقرار.",
                    language=(
                        SourceLanguage.ARABIC
                    ),
                )
            )
        )

        self.assertTrue(
            result.actions[0].is_commitment
        )

        self.assertEqual(
            result.actions[0].kind,
            CorrespondenceActionKind.NOTIFY,
        )

    def test_action_sequences_are_contiguous(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please review and approve "
                    "the request."
                )
            )
        )

        self.assertEqual(
            tuple(
                action.sequence
                for action
                in result.actions
            ),
            tuple(
                range(
                    result.action_count
                )
            ),
        )

    def test_extraction_is_deterministic(self):
        value = request(
            "Please review the report."
        )

        first = (
            extract_correspondence_actions(
                value
            )
        )

        second = (
            extract_correspondence_actions(
                value
            )
        )

        self.assertEqual(
            first.extraction_id,
            second.extraction_id,
        )

        self.assertEqual(
            first.actions[0].action_id,
            second.actions[0].action_id,
        )

    def test_default_serialization_excludes_action_text(self):
        raw = (
            "Please submit the unique "
            "private report."
        )

        result = (
            extract_correspondence_actions(
                request(raw)
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "action_text",
            data["actions"][0],
        )

    def test_action_text_can_be_explicitly_serialized(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please review the report."
                )
            )
        )

        data = result.actions[
            0
        ].to_dict(
            include_text=True
        )

        self.assertEqual(
            data[
                "action_text"
            ],
            "Please review the report.",
        )

    def test_source_hash_is_sha256(self):
        result = (
            extract_correspondence_actions(
                request(
                    "Please review the report."
                )
            )
        )

        digest = (
            result.actions[
                0
            ].source_sha256
        )

        self.assertEqual(
            len(digest),
            64,
        )

        int(
            digest,
            16,
        )

    def test_preparsed_request_is_supported(self):
        value = request(
            "Please submit the report."
        )

        parsed = parse_correspondence(
            value
        )

        result = (
            extract_correspondence_actions(
                value,
                parsed=parsed,
            )
        )

        self.assertEqual(
            result.parser_id,
            parsed.parser_id,
        )

    def test_mismatched_parsed_request_is_rejected(self):
        first = request(
            "Please submit the report."
        )

        second = request(
            "Please review the report."
        )

        parsed = parse_correspondence(
            second
        )

        with self.assertRaises(
            ValueError
        ):
            extract_correspondence_actions(
                first,
                parsed=parsed,
            )

    def test_commitment_count(self):
        result = (
            extract_correspondence_actions(
                request(
                    "We will review the file. "
                    "We will notify the department."
                )
            )
        )

        self.assertEqual(
            result.commitment_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
