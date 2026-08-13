"""Offline tests for GovBA-GAR temporal policy-state contract."""

from __future__ import annotations

import unittest
from datetime import (
    date,
    datetime,
    timezone,
)

from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.temporal import (
    TEMPORAL_POLICY_SCHEMA_VERSION,
    TemporalPolicyAssessment,
    TemporalPolicyState,
    TemporalReasonCode,
)


def make_source():
    return AuthoritativeSource(
        document_id="DOC-TEMP-002",
        title="Government Leave Policy",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        effective_from=date(
            2026,
            1,
            1,
        ),
        effective_until=date(
            2026,
            12,
            31,
        ),
        supersedes=(
            "DOC-TEMP-001",
        ),
        official_source_url=(
            "https://example.gov.jo/"
            "leave-policy.pdf"
        ),
    )


class TestTemporalPolicyContract(
    unittest.TestCase
):
    def test_schema_version_is_stable(self):
        self.assertEqual(
            TEMPORAL_POLICY_SCHEMA_VERSION,
            "govba-temporal-policy-v1",
        )

    def test_expected_policy_states_exist(self):
        self.assertEqual(
            {
                state.value
                for state
                in TemporalPolicyState
            },
            {
                "current",
                "superseded",
                "conflicting",
                "insufficient",
            },
        )

    def test_basic_assessment_is_created(self):
        assessment = TemporalPolicyAssessment(
            document_id="DOC-001",
            as_of_date=date(
                2026,
                8,
                13,
            ),
            state=(
                TemporalPolicyState.CURRENT
            ),
        )

        self.assertTrue(
            assessment.is_current
        )

        self.assertFalse(
            assessment.should_abstain
        )

    def test_assessment_id_is_sha256(self):
        assessment = TemporalPolicyAssessment(
            document_id="DOC-001",
            as_of_date=date(
                2026,
                8,
                13,
            ),
            state=(
                TemporalPolicyState.CURRENT
            ),
        )

        self.assertEqual(
            len(
                assessment.assessment_id
            ),
            64,
        )

        int(
            assessment.assessment_id,
            16,
        )

    def test_assessment_id_is_deterministic(self):
        kwargs = {
            "document_id": "DOC-001",
            "as_of_date": date(
                2026,
                8,
                13,
            ),
            "state": (
                TemporalPolicyState.CURRENT
            ),
            "reason_codes": (
                TemporalReasonCode
                .WITHIN_EFFECTIVE_WINDOW,
            ),
        }

        first = TemporalPolicyAssessment(
            **kwargs
        )

        second = TemporalPolicyAssessment(
            **kwargs
        )

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

    def test_blank_document_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            TemporalPolicyAssessment(
                document_id="  ",
                as_of_date=date.today(),
                state=(
                    TemporalPolicyState.CURRENT
                ),
            )

    def test_datetime_as_of_date_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            TemporalPolicyAssessment(
                document_id="DOC-001",
                as_of_date=datetime.now(
                    timezone.utc
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
            )

    def test_invalid_effective_window_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            TemporalPolicyAssessment(
                document_id="DOC-001",
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
                effective_from=date(
                    2026,
                    12,
                    1,
                ),
                effective_until=date(
                    2026,
                    1,
                    1,
                ),
            )

    def test_self_supersession_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            TemporalPolicyAssessment(
                document_id="DOC-001",
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState
                    .SUPERSEDED
                ),
                supersedes=(
                    "DOC-001",
                ),
            )

    def test_duplicate_relationships_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            TemporalPolicyAssessment(
                document_id="DOC-001",
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
                supersedes=(
                    "DOC-000",
                    "DOC-000",
                ),
            )

    def test_duplicate_reason_codes_are_rejected(self):
        reason = (
            TemporalReasonCode
            .WITHIN_EFFECTIVE_WINDOW
        )

        with self.assertRaises(
            ValueError
        ):
            TemporalPolicyAssessment(
                document_id="DOC-001",
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
                reason_codes=(
                    reason,
                    reason,
                ),
            )

    def test_from_source_preserves_metadata(self):
        source = make_source()

        assessment = (
            TemporalPolicyAssessment
            .from_source(
                source,
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
                reason_codes=(
                    TemporalReasonCode
                    .WITHIN_EFFECTIVE_WINDOW,
                ),
            )
        )

        self.assertEqual(
            assessment.document_id,
            source.document_id,
        )

        self.assertEqual(
            assessment.effective_from,
            source.effective_from,
        )

        self.assertEqual(
            assessment.effective_until,
            source.effective_until,
        )

        self.assertEqual(
            assessment.supersedes,
            source.supersedes,
        )

    def test_conflicting_state_requires_abstention(self):
        assessment = TemporalPolicyAssessment(
            document_id="DOC-001",
            as_of_date=date(
                2026,
                8,
                13,
            ),
            state=(
                TemporalPolicyState.CONFLICTING
            ),
            reason_codes=(
                TemporalReasonCode
                .CONFLICT_DETECTED,
            ),
        )

        self.assertTrue(
            assessment.should_abstain
        )

    def test_insufficient_state_requires_abstention(self):
        assessment = TemporalPolicyAssessment(
            document_id="DOC-001",
            as_of_date=date(
                2026,
                8,
                13,
            ),
            state=(
                TemporalPolicyState.INSUFFICIENT
            ),
            reason_codes=(
                TemporalReasonCode
                .MISSING_TEMPORAL_METADATA,
            ),
        )

        self.assertTrue(
            assessment.should_abstain
        )

    def test_superseded_is_not_current(self):
        assessment = TemporalPolicyAssessment(
            document_id="DOC-001",
            as_of_date=date(
                2026,
                8,
                13,
            ),
            state=(
                TemporalPolicyState.SUPERSEDED
            ),
        )

        self.assertFalse(
            assessment.is_current
        )

        self.assertTrue(
            assessment.is_superseded
        )

    def test_serialization_contains_temporal_metadata(self):
        assessment = (
            TemporalPolicyAssessment
            .from_source(
                make_source(),
                as_of_date=date(
                    2026,
                    8,
                    13,
                ),
                state=(
                    TemporalPolicyState.CURRENT
                ),
                reason_codes=(
                    TemporalReasonCode
                    .WITHIN_EFFECTIVE_WINDOW,
                ),
            )
        )

        data = assessment.to_dict()

        self.assertEqual(
            data["state"],
            "current",
        )

        self.assertEqual(
            data["as_of_date"],
            "2026-08-13",
        )

        self.assertEqual(
            data["effective_from"],
            "2026-01-01",
        )

        self.assertEqual(
            data["effective_until"],
            "2026-12-31",
        )

        self.assertEqual(
            data["supersedes"],
            [
                "DOC-TEMP-001",
            ],
        )


if __name__ == "__main__":
    unittest.main()
