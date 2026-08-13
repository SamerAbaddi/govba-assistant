"""Offline tests for GovBA-GAR unified governance evaluation."""

from __future__ import annotations

import unittest

from govba.governance.eligibility import (
    evaluate_evidence_cards,
)
from govba.governance.evaluator import (
    GOVERNANCE_EVALUATOR_VERSION,
    GovernanceAbstentionEvaluator,
    evaluate_governance,
)
from govba.governance.verification import (
    GovernanceDecision,
    GovernanceReasonCode,
)
from govba.rag.citations import (
    build_evidence_citations,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceTemporalState,
    EvidenceVerificationState,
)
from govba.rag.grounding import (
    GroundedClaim,
    validate_grounding,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_card(
    document_id="DOC-GOV-001",
    *,
    chunk_index=0,
    temporal_state=(
        EvidenceTemporalState.CURRENT
    ),
    verification_state=(
        EvidenceVerificationState.UNVERIFIED
    ),
):
    source = AuthoritativeSource(
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

    chunk = EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=f"Evidence for {document_id}.",
        language=SourceLanguage.ENGLISH,
        page_start=1,
        page_end=1,
    )

    return EvidenceCard(
        source=source,
        chunk=chunk,
        temporal_state=temporal_state,
        verification_state=verification_state,
    )


def make_grounded_bundle(
    cards,
    *,
    cited_indexes=(0,),
):
    cited_cards = tuple(
        cards[index]
        for index in cited_indexes
    )

    citations = build_evidence_citations(
        cited_cards
    )

    markers = " ".join(
        citation.marker
        for citation in citations
    )

    claim = GroundedClaim(
        claim_id="C1",
        text=f"Policy statement. {markers}",
        citation_ids=tuple(
            citation.citation_id
            for citation in citations
        ),
    )

    grounding = validate_grounding(
        (claim,),
        citations,
        cards,
    )

    eligibility = evaluate_evidence_cards(
        cards
    )

    return (
        grounding,
        eligibility,
    )


class TestGovernanceEvaluator(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GOVERNANCE_EVALUATOR_VERSION,
            "govba-governance-evaluator-v1",
        )

    def test_valid_grounded_current_evidence_allows(self):
        card = make_card()

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.decision,
            GovernanceDecision.ALLOW,
        )

        self.assertTrue(
            result.allowed
        )

        self.assertEqual(
            result.reason_codes,
            (
                GovernanceReasonCode
                .VERIFIED,
            ),
        )

    def test_missing_citation_abstains(self):
        card = make_card()

        claim = GroundedClaim(
            claim_id="C1",
            text="Policy statement.",
        )

        grounding = validate_grounding(
            (claim,),
            (),
            (card,),
        )

        eligibility = (
            evaluate_evidence_cards(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.decision,
            GovernanceDecision.ABSTAIN,
        )

        self.assertIn(
            GovernanceReasonCode
            .GROUNDING_FAILURE,
            result.reason_codes,
        )

    def test_unknown_citation_is_reported(self):
        card = make_card()

        claim = GroundedClaim(
            claim_id="C1",
            text="Policy statement. [E99]",
            citation_ids=("E99",),
        )

        grounding = validate_grounding(
            (claim,),
            (),
            (card,),
        )

        result = evaluate_governance(
            grounding,
            evaluate_evidence_cards(
                (card,)
            ),
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .UNKNOWN_CITATION,
            result.reason_codes,
        )

    def test_rejected_cited_evidence_abstains(self):
        card = make_card(
            verification_state=(
                EvidenceVerificationState
                .REJECTED
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .REJECTED_EVIDENCE,
            result.reason_codes,
        )

        self.assertTrue(
            result.should_abstain
        )

    def test_superseded_cited_evidence_abstains(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .SUPERSEDED
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .SUPERSEDED_EVIDENCE,
            result.reason_codes,
        )

    def test_conflicting_cited_evidence_abstains(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .CONFLICTING
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .TEMPORAL_CONFLICT,
            result.reason_codes,
        )

    def test_insufficient_cited_evidence_abstains(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .INSUFFICIENT
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .TEMPORAL_INSUFFICIENT,
            result.reason_codes,
        )

    def test_unassessed_cited_evidence_abstains(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .UNASSESSED
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .TEMPORAL_INSUFFICIENT,
            result.reason_codes,
        )

    def test_unreferenced_ineligible_card_does_not_block(self):
        eligible = make_card(
            "DOC-A",
            chunk_index=0,
        )

        unused_superseded = make_card(
            "DOC-B",
            chunk_index=1,
            temporal_state=(
                EvidenceTemporalState
                .SUPERSEDED
            ),
        )

        cards = (
            eligible,
            unused_superseded,
        )

        grounding, eligibility = (
            make_grounded_bundle(
                cards,
                cited_indexes=(0,),
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.decision,
            GovernanceDecision.ALLOW,
        )

        self.assertEqual(
            result.eligible_evidence_count,
            1,
        )

    def test_any_cited_ineligible_card_blocks_answer(self):
        eligible = make_card(
            "DOC-A",
            chunk_index=0,
        )

        superseded = make_card(
            "DOC-B",
            chunk_index=1,
            temporal_state=(
                EvidenceTemporalState
                .SUPERSEDED
            ),
        )

        cards = (
            eligible,
            superseded,
        )

        grounding, eligibility = (
            make_grounded_bundle(
                cards,
                cited_indexes=(
                    0,
                    1,
                ),
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.decision,
            GovernanceDecision.ABSTAIN,
        )

        self.assertIn(
            GovernanceReasonCode
            .SUPERSEDED_EVIDENCE,
            result.reason_codes,
        )

    def test_no_eligible_cited_evidence_is_reported(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .SUPERSEDED
            )
        )

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertIn(
            GovernanceReasonCode
            .NO_ELIGIBLE_EVIDENCE,
            result.reason_codes,
        )

    def test_card_report_mismatch_is_rejected(self):
        card_a = make_card(
            "DOC-A"
        )

        card_b = make_card(
            "DOC-B"
        )

        grounding, _ = (
            make_grounded_bundle(
                (card_a,)
            )
        )

        eligibility = (
            evaluate_evidence_cards(
                (card_b,)
            )
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_governance(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )

    def test_invalid_grounding_report_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            evaluate_governance(
                object(),
                evaluate_evidence_cards(
                    ()
                ),
                trace_id="TRACE-001",
            )

    def test_invalid_eligibility_report_is_rejected(self):
        card = make_card()

        grounding, _ = (
            make_grounded_bundle(
                (card,)
            )
        )

        with self.assertRaises(
            TypeError
        ):
            evaluate_governance(
                grounding,
                object(),
                trace_id="TRACE-001",
            )

    def test_trace_id_is_preserved(self):
        card = make_card()

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = (
            GovernanceAbstentionEvaluator()
            .evaluate(
                grounding,
                eligibility,
                trace_id=" TRACE-ABC ",
            )
        )

        self.assertEqual(
            result.trace_id,
            "TRACE-ABC",
        )

    def test_decision_is_deterministic(self):
        card = make_card()

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        first = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        second = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        self.assertEqual(
            first.decision_id,
            second.decision_id,
        )

    def test_result_is_privacy_safe(self):
        card = make_card()

        grounding, eligibility = (
            make_grounded_bundle(
                (card,)
            )
        )

        result = evaluate_governance(
            grounding,
            eligibility,
            trace_id="TRACE-001",
        )

        data = result.to_dict()

        self.assertNotIn(
            card.evidence_text,
            repr(data),
        )

        self.assertNotIn(
            "prompt",
            data,
        )


if __name__ == "__main__":
    unittest.main()
