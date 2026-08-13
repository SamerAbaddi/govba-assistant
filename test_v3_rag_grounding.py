"""Offline tests for GovBA-GAR grounding validation."""

from __future__ import annotations

import unittest
from dataclasses import replace

from govba.rag.citations import (
    build_evidence_citations,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.evidence_card import (
    EvidenceCard,
    EvidenceVerificationState,
)
from govba.rag.grounding import (
    GROUNDING_VALIDATOR_VERSION,
    GroundedClaim,
    GroundingDecision,
    GroundingIssueCode,
    GroundingValidator,
    validate_grounding,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def make_source(
    document_id="DOC-GROUND-001",
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Government Procedure",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=(
            DocumentType.PROCEDURE
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        status=SourceStatus.CURRENT,
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}.pdf"
        ),
    )


def make_card(
    document_id="DOC-GROUND-001",
    *,
    chunk_index=0,
    verification_state=(
        EvidenceVerificationState.UNVERIFIED
    ),
):
    chunk = EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=(
            "Annual leave requires "
            "manager approval."
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        page_start=4,
        page_end=4,
    )

    return EvidenceCard(
        source=make_source(
            document_id
        ),
        chunk=chunk,
        retrieval_rank=1,
        retrieval_score=0.9,
        retrieval_method=(
            "hybrid-rrf-v1"
        ),
        verification_state=(
            verification_state
        ),
    )


def make_valid_bundle():
    card = make_card()

    citation = (
        build_evidence_citations(
            (card,)
        )[0]
    )

    claim = GroundedClaim(
        claim_id="C1",
        text=(
            "Annual leave requires "
            "manager approval. [E1]"
        ),
        citation_ids=("E1",),
    )

    return (
        claim,
        citation,
        card,
    )


class TestGroundingValidator(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GROUNDING_VALIDATOR_VERSION,
            "govba-grounding-validator-v1",
        )

    def test_claim_values_are_normalized(self):
        claim = GroundedClaim(
            claim_id="  C1  ",
            text="  Policy applies.  ",
            citation_ids=(
                "  E1  ",
            ),
        )

        self.assertEqual(
            claim.claim_id,
            "C1",
        )

        self.assertEqual(
            claim.text,
            "Policy applies.",
        )

        self.assertEqual(
            claim.citation_ids,
            ("E1",),
        )

    def test_blank_claim_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GroundedClaim(
                claim_id="C1",
                text="   ",
            )

    def test_duplicate_claim_citations_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GroundedClaim(
                claim_id="C1",
                text="Policy.",
                citation_ids=(
                    "E1",
                    "E1",
                ),
            )

    def test_inline_markers_are_detected(self):
        claim = GroundedClaim(
            claim_id="C1",
            text=(
                "Policy applies [E1] "
                "and [E2]."
            ),
            citation_ids=(),
        )

        self.assertEqual(
            claim.inline_citation_ids,
            (
                "E1",
                "E2",
            ),
        )

    def test_valid_grounding_passes(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        report = validate_grounding(
            (claim,),
            (citation,),
            (card,),
        )

        self.assertTrue(
            report.is_grounded
        )

        self.assertFalse(
            report.should_abstain
        )

        self.assertEqual(
            report.decision,
            GroundingDecision.GROUNDED,
        )

    def test_missing_citation_for_required_claim_abstains(self):
        card = make_card()

        claim = GroundedClaim(
            claim_id="C1",
            text="Policy applies.",
        )

        report = validate_grounding(
            (claim,),
            (),
            (card,),
        )

        self.assertTrue(
            report.should_abstain
        )

        self.assertIn(
            GroundingIssueCode.MISSING_CITATION,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_unknown_declared_citation_abstains(self):
        claim = GroundedClaim(
            claim_id="C1",
            text="Policy applies.",
            citation_ids=(
                "E99",
            ),
        )

        report = validate_grounding(
            (claim,),
            (),
            (),
        )

        self.assertIn(
            GroundingIssueCode.UNKNOWN_CITATION,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

        self.assertTrue(
            report.should_abstain
        )

    def test_unknown_inline_marker_abstains(self):
        claim = GroundedClaim(
            claim_id="C1",
            text="Policy applies. [E99]",
        )

        report = validate_grounding(
            (claim,),
            (),
            (),
        )

        self.assertIn(
            GroundingIssueCode.UNKNOWN_CITATION,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_citation_links_to_known_card(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        report = (
            GroundingValidator()
            .validate(
                (claim,),
                (citation,),
                (card,),
            )
        )

        self.assertEqual(
            report.issue_count,
            0,
        )

    def test_unknown_card_is_rejected(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        wrong_card = make_card(
            "DOC-OTHER"
        )

        report = validate_grounding(
            (claim,),
            (citation,),
            (wrong_card,),
        )

        self.assertIn(
            GroundingIssueCode.UNKNOWN_CARD,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_document_mismatch_is_rejected(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        damaged = replace(
            citation,
            document_id="DOC-WRONG",
        )

        report = validate_grounding(
            (claim,),
            (damaged,),
            (card,),
        )

        self.assertIn(
            (
                GroundingIssueCode
                .CITATION_CARD_MISMATCH
            ),
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_chunk_mismatch_is_rejected(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        damaged = replace(
            citation,
            chunk_id="WRONG-CHUNK",
        )

        report = validate_grounding(
            (claim,),
            (damaged,),
            (card,),
        )

        self.assertIn(
            (
                GroundingIssueCode
                .CITATION_CARD_MISMATCH
            ),
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_marker_mismatch_is_rejected(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        damaged = replace(
            citation,
            marker="[E9]",
        )

        report = validate_grounding(
            (claim,),
            (damaged,),
            (card,),
        )

        self.assertIn(
            (
                GroundingIssueCode
                .CITATION_MARKER_MISMATCH
            ),
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_duplicate_citation_ids_are_rejected(self):
        card_a = make_card(
            "DOC-A"
        )

        card_b = make_card(
            "DOC-B"
        )

        citations = (
            build_evidence_citations(
                (
                    card_a,
                    card_b,
                )
            )
        )

        duplicate = replace(
            citations[1],
            citation_id="E1",
            marker="[E1]",
        )

        report = validate_grounding(
            (),
            (
                citations[0],
                duplicate,
            ),
            (
                card_a,
                card_b,
            ),
        )

        self.assertIn(
            (
                GroundingIssueCode
                .DUPLICATE_CITATION_ID
            ),
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_duplicate_card_ids_are_rejected(self):
        card = make_card()

        report = validate_grounding(
            (),
            (),
            (
                card,
                card,
            ),
        )

        self.assertIn(
            GroundingIssueCode.DUPLICATE_CARD_ID,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_duplicate_claim_ids_are_rejected(self):
        claims = (
            GroundedClaim(
                claim_id="C1",
                text="General statement.",
                requires_evidence=False,
            ),
            GroundedClaim(
                claim_id="C1",
                text="Another statement.",
                requires_evidence=False,
            ),
        )

        report = validate_grounding(
            claims,
            (),
            (),
        )

        self.assertIn(
            GroundingIssueCode.DUPLICATE_CLAIM_ID,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

    def test_non_evidence_claim_can_omit_citation(self):
        claim = GroundedClaim(
            claim_id="C1",
            text="Here is the requested summary.",
            requires_evidence=False,
        )

        report = validate_grounding(
            (claim,),
            (),
            (),
        )

        self.assertTrue(
            report.is_grounded
        )

    def test_rejected_evidence_forces_abstention(self):
        card = make_card(
            verification_state=(
                EvidenceVerificationState.REJECTED
            )
        )

        citation = (
            build_evidence_citations(
                (card,)
            )[0]
        )

        claim = GroundedClaim(
            claim_id="C1",
            text="Policy applies. [E1]",
            citation_ids=("E1",),
        )

        report = validate_grounding(
            (claim,),
            (citation,),
            (card,),
        )

        self.assertIn(
            GroundingIssueCode.REJECTED_EVIDENCE,
            tuple(
                issue.code
                for issue in report.issues
            ),
        )

        self.assertTrue(
            report.should_abstain
        )

    def test_report_serialization_excludes_raw_text(self):
        (
            claim,
            citation,
            card,
        ) = make_valid_bundle()

        report = validate_grounding(
            (claim,),
            (citation,),
            (card,),
        )

        data = report.to_dict()

        self.assertEqual(
            data["decision"],
            "grounded",
        )

        self.assertNotIn(
            claim.text,
            repr(data),
        )

        self.assertNotIn(
            card.evidence_text,
            repr(data),
        )

    def test_invalid_input_type_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            validate_grounding(
                (
                    object(),
                ),
                (),
                (),
            )


if __name__ == "__main__":
    unittest.main()
