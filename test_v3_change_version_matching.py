"""Offline tests for GovBA-GAR document-version matching."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.version_matching import (
    DOCUMENT_VERSION_MATCHING_VERSION,
    DocumentVersionMatchDecision,
    DocumentVersionMatchReason,
    build_change_request_from_match,
    match_document_versions,
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
    title="Annual Leave Policy",
    authority="Synthetic Ministry",
    document_type=DocumentType.POLICY,
    jurisdiction="Jordan",
    content_hash="a" * 64,
    version="1.0",
    supersedes=(),
    superseded_by=(),
):
    return AuthoritativeSource(
        document_id=document_id,
        title=title,
        issuing_authority=authority,
        document_type=document_type,
        language=(
            SourceLanguage.ENGLISH
        ),
        jurisdiction=jurisdiction,
        effective_from=date(
            2026,
            1,
            1,
        ),
        version=version,
        status=(
            SourceStatus.CURRENT
        ),
        supersedes=(
            supersedes
        ),
        superseded_by=(
            superseded_by
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}"
        ),
        content_hash=(
            content_hash
        ),
    )


class TestDocumentVersionMatching(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            DOCUMENT_VERSION_MATCHING_VERSION,
            "govba-document-version-matching-v1",
        )

    def test_same_document_id_is_allowed(self):
        baseline = source(
            "POLICY-001",
            content_hash="a" * 64,
        )

        candidate = source(
            "POLICY-001",
            content_hash="b" * 64,
            version="2.0",
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.allowed
        )

        self.assertIn(
            DocumentVersionMatchReason
            .SAME_DOCUMENT_ID,
            result.reasons,
        )

    def test_explicit_candidate_supersedes_baseline(self):
        baseline = source(
            "OLD"
        )

        candidate = source(
            "NEW",
            supersedes=(
                "OLD",
            ),
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.allowed
        )

        self.assertTrue(
            result.explicit_supersession
        )

    def test_baseline_superseded_by_candidate(self):
        baseline = source(
            "OLD",
            superseded_by=(
                "NEW",
            ),
        )

        candidate = source(
            "NEW"
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.allowed
        )

    def test_exact_metadata_match_is_allowed(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            content_hash="b" * 64,
            version="2.0",
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.ALLOW,
        )

        self.assertIn(
            DocumentVersionMatchReason
            .METADATA_MATCH,
            result.reasons,
        )

    def test_metadata_normalization_handles_case_and_punctuation(self):
        baseline = source(
            "A",
            title="Annual Leave Policy",
            authority="Ministry of Test",
        )

        candidate = source(
            "B",
            title="ANNUAL LEAVE - POLICY",
            authority="MINISTRY OF TEST",
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.allowed
        )

    def test_single_title_mismatch_requires_review(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            title="Revised Leave Framework",
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.REVIEW,
        )

        self.assertTrue(
            result.requires_review
        )

        self.assertIn(
            DocumentVersionMatchReason
            .TITLE_MISMATCH,
            result.reasons,
        )

    def test_single_authority_mismatch_requires_review(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            authority="Renamed Ministry",
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.REVIEW,
        )

    def test_single_type_mismatch_requires_review(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            document_type=(
                DocumentType.CIRCULAR
            ),
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.REVIEW,
        )

    def test_multiple_metadata_mismatches_reject(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            title="Different Document",
            authority="Different Authority",
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.REJECT,
        )

        self.assertTrue(
            result.rejected
        )

    def test_jurisdiction_mismatch_rejects(self):
        baseline = source(
            "A",
            jurisdiction="Jordan",
        )

        candidate = source(
            "B",
            jurisdiction="Other Jurisdiction",
            content_hash="b" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            result.decision,
            DocumentVersionMatchDecision.REJECT,
        )

        self.assertIn(
            DocumentVersionMatchReason
            .JURISDICTION_MISMATCH,
            result.reasons,
        )

    def test_same_content_is_detected(self):
        baseline = source(
            "A",
            content_hash="c" * 64,
        )

        candidate = source(
            "B",
            content_hash="c" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.same_content
        )

        self.assertIn(
            DocumentVersionMatchReason
            .SAME_CONTENT_DIFFERENT_ID,
            result.reasons,
        )

    def test_same_content_does_not_override_bad_metadata(self):
        baseline = source(
            "A",
            content_hash="c" * 64,
        )

        candidate = source(
            "B",
            title="Completely Different",
            authority="Different Authority",
            content_hash="c" * 64,
        )

        result = match_document_versions(
            baseline,
            candidate,
        )

        self.assertTrue(
            result.same_content
        )

        self.assertTrue(
            result.rejected
        )

    def test_match_id_is_deterministic(self):
        baseline = source(
            "A"
        )

        candidate = source(
            "B",
            content_hash="b" * 64,
        )

        first = match_document_versions(
            baseline,
            candidate,
        )

        second = match_document_versions(
            baseline,
            candidate,
        )

        self.assertEqual(
            first.match_id,
            second.match_id,
        )

    def test_match_id_is_sha256(self):
        result = match_document_versions(
            source("A"),
            source(
                "B",
                content_hash="b" * 64,
            ),
        )

        self.assertEqual(
            len(
                result.match_id
            ),
            64,
        )

        int(
            result.match_id,
            16,
        )

    def test_serialization_excludes_source_urls(self):
        result = match_document_versions(
            source("A"),
            source(
                "B",
                content_hash="b" * 64,
            ),
        )

        data = result.to_dict()

        self.assertNotIn(
            "official_source_url",
            repr(data),
        )

        self.assertNotIn(
            "https://",
            repr(data),
        )

    def test_allowed_match_builds_change_request(self):
        match = match_document_versions(
            source("A"),
            source(
                "B",
                content_hash="b" * 64,
            ),
        )

        request = (
            build_change_request_from_match(
                match,
                trace_id="TRACE-001",
            )
        )

        self.assertEqual(
            request.baseline_source
            .document_id,
            "A",
        )

        self.assertEqual(
            request.candidate_source
            .document_id,
            "B",
        )

    def test_review_match_cannot_start_automatic_analysis(self):
        match = match_document_versions(
            source("A"),
            source(
                "B",
                title="Different Title",
                content_hash="b" * 64,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_change_request_from_match(
                match,
                trace_id="TRACE-001",
            )

    def test_rejected_match_cannot_start_analysis(self):
        match = match_document_versions(
            source("A"),
            source(
                "B",
                title="Different",
                authority="Different Authority",
                content_hash="b" * 64,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_change_request_from_match(
                match,
                trace_id="TRACE-001",
            )

    def test_invalid_source_type_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            match_document_versions(
                object(),
                source("B"),
            )

    def test_blank_trace_id_is_rejected(self):
        match = match_document_versions(
            source("A"),
            source(
                "B",
                content_hash="b" * 64,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            build_change_request_from_match(
                match,
                trace_id=" ",
            )


if __name__ == "__main__":
    unittest.main()
