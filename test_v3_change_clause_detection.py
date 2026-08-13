"""Offline tests for GovBA-GAR deterministic clause change detection."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.clause_detection import (
    CLAUSE_CHANGE_DETECTION_VERSION,
    DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD,
    ClauseChangeReason,
    clause_similarity,
    detect_clause_changes,
)
from govba.change.contract import (
    PolicyChangeImpact,
    PolicyChangeType,
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
    *,
    content_hash,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Annual Leave Policy",
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


def version_match(
    *,
    same_content=False,
):
    return match_document_versions(
        source(
            "OLD",
            content_hash=(
                "a" * 64
                if not same_content
                else "c" * 64
            ),
        ),
        source(
            "NEW",
            content_hash=(
                "b" * 64
                if not same_content
                else "c" * 64
            ),
        ),
    )


class TestClauseChangeDetection(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CLAUSE_CHANGE_DETECTION_VERSION,
            "govba-clause-change-detection-v1",
        )

    def test_default_threshold_is_stable(self):
        self.assertEqual(
            DEFAULT_MODIFICATION_SIMILARITY_THRESHOLD,
            0.55,
        )

    def test_identical_chunk_is_unchanged(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Employees receive 14 days leave.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Employees receive 14 days leave.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.unchanged_count,
            1,
        )

        self.assertFalse(
            result.changed
        )

        self.assertEqual(
            result.matches[0].reason,
            ClauseChangeReason.EXACT_CONTENT_MATCH,
        )

    def test_modified_chunk_is_detected(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Employees receive 14 days annual leave.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Employees receive 21 days annual leave.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.modified_count,
            1,
        )

        self.assertTrue(
            result.changed
        )

    def test_added_chunk_is_detected(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "A new reporting requirement applies.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.added_count,
            1,
        )

        self.assertEqual(
            result.matches[0].reason,
            ClauseChangeReason.CANDIDATE_ONLY,
        )

    def test_removed_chunk_is_detected(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Legacy reporting requirement.",
                ),
            ),
            candidate_chunks=(),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.removed_count,
            1,
        )

    def test_unrelated_chunks_become_removed_and_added(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Annual leave entitlement.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Cybersecurity incident procedure.",
                ),
            ),
            trace_id="TRACE-001",
            similarity_threshold=0.8,
        )

        self.assertEqual(
            result.report.removed_count,
            1,
        )

        self.assertEqual(
            result.report.added_count,
            1,
        )

    def test_exact_matches_are_paired_before_similarity(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Exact clause.",
                ),
                chunk(
                    "OLD",
                    1,
                    "Employees receive 14 days leave.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Exact clause.",
                ),
                chunk(
                    "NEW",
                    1,
                    "Employees receive 21 days leave.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.unchanged_count,
            1,
        )

        self.assertEqual(
            result.report.modified_count,
            1,
        )

    def test_similarity_is_deterministic(self):
        first = clause_similarity(
            "Employees receive annual leave.",
            "Employees receive paid annual leave.",
        )

        second = clause_similarity(
            "Employees receive annual leave.",
            "Employees receive paid annual leave.",
        )

        self.assertEqual(
            first,
            second,
        )

    def test_similarity_handles_case(self):
        value = clause_similarity(
            "ANNUAL LEAVE POLICY",
            "annual leave policy",
        )

        self.assertEqual(
            value,
            1.0,
        )

    def test_custom_threshold_controls_pairing(self):
        baseline = (
            chunk(
                "OLD",
                0,
                "Employees must submit the form.",
            ),
        )

        candidate = (
            chunk(
                "NEW",
                0,
                "Employees may submit a request online.",
            ),
        )

        strict = detect_clause_changes(
            version_match(),
            baseline_chunks=baseline,
            candidate_chunks=candidate,
            trace_id="TRACE-001",
            similarity_threshold=0.95,
        )

        self.assertEqual(
            strict.report.modified_count,
            0,
        )

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            detect_clause_changes(
                version_match(),
                baseline_chunks=(),
                candidate_chunks=(),
                trace_id="TRACE-001",
                similarity_threshold=0.0,
            )

    def test_wrong_baseline_document_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            detect_clause_changes(
                version_match(),
                baseline_chunks=(
                    chunk(
                        "WRONG",
                        0,
                        "Clause",
                    ),
                ),
                candidate_chunks=(),
                trace_id="TRACE-001",
            )

    def test_noncontiguous_indexes_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            detect_clause_changes(
                version_match(),
                baseline_chunks=(
                    chunk(
                        "OLD",
                        1,
                        "Clause",
                    ),
                ),
                candidate_chunks=(),
                trace_id="TRACE-001",
            )

    def test_review_match_cannot_be_analyzed(self):
        baseline_source = source(
            "OLD",
            content_hash="a" * 64,
        )

        candidate_source = AuthoritativeSource(
            document_id="NEW",
            title="Different Document",
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
                "https://example.gov.jo/NEW"
            ),
            content_hash="b" * 64,
        )

        match = match_document_versions(
            baseline_source,
            candidate_source,
        )

        with self.assertRaises(
            ValueError
        ):
            detect_clause_changes(
                match,
                baseline_chunks=(),
                candidate_chunks=(),
                trace_id="TRACE-001",
            )

    def test_same_document_content_short_circuits(self):
        result = detect_clause_changes(
            version_match(
                same_content=True
            ),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Same content.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Same content.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertFalse(
            result.changed
        )

        self.assertEqual(
            result.matches,
            (),
        )

    def test_structural_stage_uses_informational_impact(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "New requirement.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.findings[
                0
            ].impact,
            PolicyChangeImpact.INFORMATIONAL,
        )

    def test_reason_is_preserved_in_finding(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "New clause.",
                ),
            ),
            trace_id="TRACE-001",
        )

        self.assertEqual(
            result.report.findings[
                0
            ].reason_code,
            ClauseChangeReason
            .CANDIDATE_ONLY.value,
        )

    def test_detection_id_is_deterministic(self):
        kwargs = {
            "version_match": (
                version_match()
            ),
            "baseline_chunks": (
                chunk(
                    "OLD",
                    0,
                    "Old requirement.",
                ),
            ),
            "candidate_chunks": (
                chunk(
                    "NEW",
                    0,
                    "Updated requirement.",
                ),
            ),
            "trace_id": "TRACE-001",
        }

        first = detect_clause_changes(
            **kwargs
        )

        second = detect_clause_changes(
            **kwargs
        )

        self.assertEqual(
            first.detection_id,
            second.detection_id,
        )

    def test_serialization_contains_no_clause_text(self):
        secret_text = (
            "Unique synthetic confidential clause."
        )

        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    secret_text,
                ),
            ),
            trace_id="TRACE-001",
        )

        data = result.to_dict()

        self.assertNotIn(
            secret_text,
            repr(data),
        )

        self.assertNotIn(
            "text",
            data,
        )

    def test_mixed_change_types_are_counted(self):
        result = detect_clause_changes(
            version_match(),
            baseline_chunks=(
                chunk(
                    "OLD",
                    0,
                    "Unchanged clause.",
                ),
                chunk(
                    "OLD",
                    1,
                    "Employees receive 14 days leave.",
                ),
                chunk(
                    "OLD",
                    2,
                    "Old unique requirement xyz.",
                ),
            ),
            candidate_chunks=(
                chunk(
                    "NEW",
                    0,
                    "Unchanged clause.",
                ),
                chunk(
                    "NEW",
                    1,
                    "Employees receive 21 days leave.",
                ),
                chunk(
                    "NEW",
                    2,
                    "Completely new cybersecurity control.",
                ),
            ),
            trace_id="TRACE-001",
            similarity_threshold=0.75,
        )

        self.assertEqual(
            result.report.unchanged_count,
            1,
        )

        self.assertEqual(
            result.report.modified_count,
            1,
        )

        self.assertEqual(
            result.report.removed_count,
            1,
        )

        self.assertEqual(
            result.report.added_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
