"""Offline tests for GovBA-GAR evidence eligibility."""

from __future__ import annotations

import unittest

from govba.governance.eligibility import (
    EVIDENCE_ELIGIBILITY_VERSION,
    EvidenceEligibilityReason,
    evaluate_evidence_card,
    evaluate_evidence_cards,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_source(
    document_id="DOC-ELIG-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title=f"Policy {document_id}",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=DocumentType.POLICY,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}.pdf"
        ),
    )


def make_card(
    document_id="DOC-ELIG-001",
    *,
    chunk_index=0,
    temporal_state=(
        EvidenceTemporalState.CURRENT
    ),
    verification_state=(
        EvidenceVerificationState.UNVERIFIED
    ),
):
    source = make_source(
        document_id
    )

    chunk = EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=(
            f"Evidence for {document_id}."
        ),
        language=SourceLanguage.ENGLISH,
        page_start=1,
        page_end=1,
    )

    return EvidenceCard(
        source=source,
        chunk=chunk,
        temporal_state=temporal_state,
        verification_state=(
            verification_state
        ),
    )


class TestEvidenceEligibility(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            EVIDENCE_ELIGIBILITY_VERSION,
            "govba-evidence-eligibility-v1",
        )

    def test_current_non_rejected_card_is_eligible(self):
        result = evaluate_evidence_card(
            make_card()
        )

        self.assertTrue(
            result.eligible
        )

        self.assertEqual(
            result.reason_codes,
            (
                EvidenceEligibilityReason
                .ELIGIBLE,
            ),
        )

    def test_unverified_current_card_can_be_eligible(self):
        result = evaluate_evidence_card(
            make_card(
                verification_state=(
                    EvidenceVerificationState
                    .UNVERIFIED
                )
            )
        )

        self.assertTrue(
            result.eligible
        )

    def test_verified_current_card_is_eligible(self):
        result = evaluate_evidence_card(
            make_card(
                verification_state=(
                    EvidenceVerificationState
                    .VERIFIED
                )
            )
        )

        self.assertTrue(
            result.eligible
        )

    def test_rejected_card_is_ineligible(self):
        result = evaluate_evidence_card(
            make_card(
                verification_state=(
                    EvidenceVerificationState
                    .REJECTED
                )
            )
        )

        self.assertFalse(
            result.eligible
        )

        self.assertIn(
            EvidenceEligibilityReason
            .REJECTED_EVIDENCE,
            result.reason_codes,
        )

    def test_unassessed_temporal_state_is_ineligible(self):
        result = evaluate_evidence_card(
            make_card(
                temporal_state=(
                    EvidenceTemporalState
                    .UNASSESSED
                )
            )
        )

        self.assertFalse(
            result.eligible
        )

        self.assertIn(
            EvidenceEligibilityReason
            .TEMPORAL_UNASSESSED,
            result.reason_codes,
        )

    def test_superseded_card_is_ineligible(self):
        result = evaluate_evidence_card(
            make_card(
                temporal_state=(
                    EvidenceTemporalState
                    .SUPERSEDED
                )
            )
        )

        self.assertIn(
            EvidenceEligibilityReason
            .SUPERSEDED_EVIDENCE,
            result.reason_codes,
        )

    def test_conflicting_card_is_ineligible(self):
        result = evaluate_evidence_card(
            make_card(
                temporal_state=(
                    EvidenceTemporalState
                    .CONFLICTING
                )
            )
        )

        self.assertIn(
            EvidenceEligibilityReason
            .TEMPORAL_CONFLICT,
            result.reason_codes,
        )

    def test_insufficient_card_is_ineligible(self):
        result = evaluate_evidence_card(
            make_card(
                temporal_state=(
                    EvidenceTemporalState
                    .INSUFFICIENT
                )
            )
        )

        self.assertIn(
            EvidenceEligibilityReason
            .TEMPORAL_INSUFFICIENT,
            result.reason_codes,
        )

    def test_multiple_failures_are_preserved(self):
        result = evaluate_evidence_card(
            make_card(
                temporal_state=(
                    EvidenceTemporalState
                    .SUPERSEDED
                ),
                verification_state=(
                    EvidenceVerificationState
                    .REJECTED
                ),
            )
        )

        self.assertEqual(
            set(
                result.reason_codes
            ),
            {
                EvidenceEligibilityReason
                .REJECTED_EVIDENCE,
                EvidenceEligibilityReason
                .SUPERSEDED_EVIDENCE,
            },
        )

    def test_multiple_cards_are_evaluated(self):
        cards = (
            make_card(
                "DOC-A",
                chunk_index=0,
            ),
            make_card(
                "DOC-B",
                chunk_index=1,
                temporal_state=(
                    EvidenceTemporalState
                    .SUPERSEDED
                ),
            ),
        )

        report = evaluate_evidence_cards(
            cards
        )

        self.assertEqual(
            report.evidence_count,
            2,
        )

        self.assertEqual(
            report.eligible_count,
            1,
        )

        self.assertEqual(
            report.ineligible_count,
            1,
        )

    def test_eligible_cards_are_filtered(self):
        cards = (
            make_card(
                "DOC-A"
            ),
            make_card(
                "DOC-B",
                chunk_index=1,
                temporal_state=(
                    EvidenceTemporalState
                    .CONFLICTING
                ),
            ),
        )

        report = evaluate_evidence_cards(
            cards
        )

        self.assertEqual(
            tuple(
                card.document_id
                for card
                in report.eligible_cards
            ),
            (
                "DOC-A",
            ),
        )

    def test_no_eligible_evidence_is_reported(self):
        report = evaluate_evidence_cards(
            (
                make_card(
                    temporal_state=(
                        EvidenceTemporalState
                        .INSUFFICIENT
                    )
                ),
            )
        )

        self.assertFalse(
            report.has_eligible_evidence
        )

    def test_input_order_is_preserved(self):
        cards = (
            make_card(
                "DOC-B"
            ),
            make_card(
                "DOC-A",
                chunk_index=1,
            ),
        )

        report = evaluate_evidence_cards(
            cards
        )

        self.assertEqual(
            tuple(
                result.document_id
                for result
                in report.results
            ),
            (
                "DOC-B",
                "DOC-A",
            ),
        )

    def test_duplicate_cards_are_rejected(self):
        card = make_card()

        with self.assertRaises(
            ValueError
        ):
            evaluate_evidence_cards(
                (
                    card,
                    card,
                )
            )

    def test_invalid_card_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            evaluate_evidence_card(
                object()
            )

    def test_report_serialization_excludes_evidence_text(self):
        card = make_card()

        report = evaluate_evidence_cards(
            (
                card,
            )
        )

        data = report.to_dict()

        self.assertEqual(
            data["eligible_count"],
            1,
        )

        self.assertNotIn(
            card.evidence_text,
            repr(
                data
            ),
        )


if __name__ == "__main__":
    unittest.main()
