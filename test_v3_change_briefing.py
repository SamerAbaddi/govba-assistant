"""Offline tests for GovBA-GAR governed change briefing."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.briefing import (
    CHANGE_BRIEFING_VERSION,
    ChangeBriefingDecision,
    ChangeBriefingReason,
    build_governed_change_briefing,
)
from govba.change.clause_detection import (
    detect_clause_changes,
)
from govba.change.significance import (
    classify_change_significance,
)
from govba.change.version_matching import (
    match_document_versions,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def source(
    document_id,
    content_hash,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Policy",
        issuing_authority=(
            "Synthetic Ministry"
        ),
        document_type=(
            DocumentType.POLICY
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        jurisdiction="Jordan",
        effective_from=date(
            2026,
            1,
            1,
        ),
        status=(
            SourceStatus.CURRENT
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}"
        ),
        content_hash=content_hash,
    )


def chunk(
    document_id,
    index,
    text,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=index,
        text=text,
        language=(
            SourceLanguage.ENGLISH
        ),
    )


def classify(
    old_text,
    new_text,
):
    baseline_source = source(
        "OLD",
        "a" * 64,
    )

    candidate_source = source(
        "NEW",
        "b" * 64,
    )

    baseline_chunks = (
        chunk(
            "OLD",
            0,
            old_text,
        ),
    )

    candidate_chunks = (
        chunk(
            "NEW",
            0,
            new_text,
        ),
    )

    match = match_document_versions(
        baseline_source,
        candidate_source,
    )

    detection = detect_clause_changes(
        match,
        baseline_chunks=(
            baseline_chunks
        ),
        candidate_chunks=(
            candidate_chunks
        ),
        trace_id="TRACE-001",
    )

    result = (
        classify_change_significance(
            detection,
            baseline_chunks=(
                baseline_chunks
            ),
            candidate_chunks=(
                candidate_chunks
            ),
        )
    )

    return (
        result,
        baseline_chunks,
        candidate_chunks,
    )


class TestGovernedChangeBriefing(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CHANGE_BRIEFING_VERSION,
            "govba-change-briefing-v1",
        )

    def test_verified_change_is_ready(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Folders are blue.",
            "Folders are green.",
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            briefing.decision,
            ChangeBriefingDecision.READY,
        )

        self.assertIn(
            ChangeBriefingReason
            .VERIFIED_CHANGES,
            briefing.reasons,
        )

    def test_critical_change_requires_review(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "No sanction applies.",
            (
                "A penalty of 100 JOD "
                "shall apply."
            ),
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            briefing.decision,
            ChangeBriefingDecision.REVIEW,
        )

        self.assertTrue(
            briefing.requires_review
        )

    def test_items_are_evidence_linked(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            "Employees must submit.",
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertGreater(
            len(
                briefing.items
            ),
            0,
        )

        self.assertGreater(
            len(
                briefing.items[0].evidence
            ),
            0,
        )

    def test_modified_item_has_two_references(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Employees may submit a form.",
            "Employees must submit a form.",
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            len(
                briefing.items[0].evidence
            ),
            2,
        )

    def test_missing_evidence_causes_abstention(self):
        (
            result,
            _baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            "Employees must submit.",
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=(),
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertTrue(
            briefing.abstained
        )

        self.assertIn(
            ChangeBriefingReason
            .MISSING_EVIDENCE,
            briefing.reasons,
        )

    def test_wrong_document_evidence_is_rejected(self):
        (
            result,
            _baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            "Employees must submit.",
        )

        with self.assertRaises(
            ValueError
        ):
            build_governed_change_briefing(
                result,
                baseline_chunks=(
                    chunk(
                        "WRONG",
                        0,
                        "Employees may submit.",
                    ),
                ),
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )

    def test_briefing_id_is_deterministic(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            "Employees must submit.",
        )

        first = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        second = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            first.briefing_id,
            second.briefing_id,
        )

    def test_serialization_has_no_clause_text(self):
        raw = (
            "Employees must submit "
            "the confidential synthetic form."
        )

        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            raw,
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        data = briefing.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "text",
            data,
        )

    def test_headline_is_deterministic(self):
        (
            result,
            baseline,
            candidate,
        ) = classify(
            "Employees may submit.",
            "Employees must submit.",
        )

        briefing = (
            build_governed_change_briefing(
                result,
                baseline_chunks=baseline,
                candidate_chunks=candidate,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            briefing.items[0].headline,
            "Requirement modified",
        )


if __name__ == "__main__":
    unittest.main()
