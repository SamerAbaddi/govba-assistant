"""Offline tests for GovBA-GAR governed answer bundles."""

from __future__ import annotations

import unittest
from dataclasses import replace

from govba.governance.answer_bundle import (
    DEFAULT_ABSTENTION_MESSAGE,
    GOVERNED_ANSWER_BUNDLE_VERSION,
    GovernedAnswerBundle,
    build_governed_answer_bundle,
)
from govba.governance.eligibility import (
    evaluate_evidence_cards,
)
from govba.governance.evaluator import (
    evaluate_governance,
)
from govba.governance.verification import (
    GovernanceDecision,
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
    document_id="DOC-ANSWER-001",
    *,
    chunk_index=0,
    temporal_state=(
        EvidenceTemporalState.CURRENT
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
        text=(
            f"Evidence text for "
            f"{document_id}."
        ),
        language=SourceLanguage.ENGLISH,
        page_start=1,
        page_end=1,
    )

    return EvidenceCard(
        source=source,
        chunk=chunk,
        temporal_state=temporal_state,
    )


def make_reports(
    cards,
    *,
    inline_marker=True,
    include_intro=False,
):
    citation = (
        build_evidence_citations(
            (
                cards[0],
            )
        )[0]
    )

    claims = []

    if include_intro:
        claims.append(
            GroundedClaim(
                claim_id="C0",
                text=(
                    "Here is the verified answer."
                ),
                requires_evidence=False,
            )
        )

    evidence_text = (
        "The policy requires approval."
    )

    if inline_marker:
        evidence_text = (
            f"{evidence_text} "
            f"{citation.marker}"
        )

    claims.append(
        GroundedClaim(
            claim_id="C1",
            text=evidence_text,
            citation_ids=(
                citation.citation_id,
            ),
        )
    )

    grounding = validate_grounding(
        tuple(
            claims
        ),
        (
            citation,
        ),
        cards,
    )

    eligibility = (
        evaluate_evidence_cards(
            cards
        )
    )

    return (
        grounding,
        eligibility,
    )


class TestGovernedAnswerBundle(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GOVERNED_ANSWER_BUNDLE_VERSION,
            "govba-governed-answer-bundle-v1",
        )

    def test_allowed_bundle_is_created(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertTrue(
            bundle.allowed
        )

        self.assertEqual(
            bundle.verification.decision,
            GovernanceDecision.ALLOW,
        )

    def test_allowed_response_uses_validated_claim(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            bundle.response_text,
            (
                "The policy requires "
                "approval. [E1]"
            ),
        )

    def test_declared_citation_marker_is_appended(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                ),
                inline_marker=False,
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            bundle.response_text,
            (
                "The policy requires "
                "approval. [E1]"
            ),
        )

    def test_inline_marker_is_not_duplicated(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                ),
                inline_marker=True,
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            bundle.response_text.count(
                "[E1]"
            ),
            1,
        )

    def test_claim_order_is_preserved(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                ),
                include_intro=True,
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            bundle.response_text,
            (
                "Here is the verified answer."
                "\n\n"
                "The policy requires "
                "approval. [E1]"
            ),
        )

    def test_abstention_uses_controlled_message(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .SUPERSEDED
            )
        )

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertTrue(
            bundle.should_abstain
        )

        self.assertEqual(
            bundle.response_text,
            DEFAULT_ABSTENTION_MESSAGE,
        )

    def test_abstention_does_not_surface_claim_text(self):
        card = make_card(
            temporal_state=(
                EvidenceTemporalState
                .CONFLICTING
            )
        )

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertNotIn(
            "policy requires approval",
            bundle.response_text.lower(),
        )

    def test_trace_id_is_normalized(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id=" TRACE-ABC ",
            )
        )

        self.assertEqual(
            bundle.trace_id,
            "TRACE-ABC",
        )

    def test_bundle_exposes_evidence_contracts(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            len(
                bundle.claims
            ),
            1,
        )

        self.assertEqual(
            len(
                bundle.citations
            ),
            1,
        )

        self.assertEqual(
            len(
                bundle.evidence_cards
            ),
            1,
        )

    def test_cited_evidence_excludes_unused_card(self):
        cited = make_card(
            "DOC-A",
            chunk_index=0,
        )

        unused = make_card(
            "DOC-B",
            chunk_index=1,
        )

        grounding, eligibility = (
            make_reports(
                (
                    cited,
                    unused,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            bundle.cited_card_ids,
            (
                cited.card_id,
            ),
        )

        self.assertEqual(
            tuple(
                card.document_id
                for card
                in bundle
                .cited_evidence_cards
            ),
            (
                "DOC-A",
            ),
        )

    def test_response_hash_is_sha256(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            len(
                bundle.response_sha256
            ),
            64,
        )

        int(
            bundle.response_sha256,
            16,
        )

    def test_bundle_id_is_deterministic(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        first = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        second = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            first.bundle_id,
            second.bundle_id,
        )

    def test_metadata_serialization_excludes_response(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        data = bundle.to_dict()

        self.assertNotIn(
            "response_text",
            data,
        )

        self.assertNotIn(
            card.evidence_text,
            repr(
                data
            ),
        )

    def test_explicit_serialization_can_include_response(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        bundle = (
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        data = bundle.to_dict(
            include_response_text=True
        )

        self.assertEqual(
            data[
                "response_text"
            ],
            bundle.response_text,
        )

    def test_tampered_verification_is_rejected(self):
        card = make_card()

        grounding, eligibility = (
            make_reports(
                (
                    card,
                )
            )
        )

        verification = (
            evaluate_governance(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )
        )

        tampered = replace(
            verification,
            evidence_count=2,
        )

        with self.assertRaises(
            ValueError
        ):
            GovernedAnswerBundle(
                trace_id="TRACE-001",
                response_text=(
                    "The policy requires "
                    "approval. [E1]"
                ),
                grounding_report=grounding,
                eligibility_report=eligibility,
                verification=tampered,
            )

    def test_mismatched_reports_are_rejected(self):
        card_a = make_card(
            "DOC-A"
        )

        card_b = make_card(
            "DOC-B"
        )

        grounding, _ = (
            make_reports(
                (
                    card_a,
                )
            )
        )

        eligibility = (
            evaluate_evidence_cards(
                (
                    card_b,
                )
            )
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_answer_bundle(
                grounding,
                eligibility,
                trace_id="TRACE-001",
            )


if __name__ == "__main__":
    unittest.main()
