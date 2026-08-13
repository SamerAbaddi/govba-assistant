"""Offline tests for GovBA-GAR policy-change contract."""

from __future__ import annotations

import unittest
from datetime import date

from govba.change.contract import (
    POLICY_CHANGE_CONTRACT_VERSION,
    PolicyChangeFinding,
    PolicyChangeImpact,
    PolicyChangeRequest,
    PolicyChangeType,
    build_policy_change_report,
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
    version,
    effective_from,
):
    return AuthoritativeSource(
        document_id=document_id,
        title="Synthetic Policy",
        issuing_authority=(
            "Synthetic Government Authority"
        ),
        document_type=(
            DocumentType.POLICY
        ),
        language=(
            SourceLanguage.ENGLISH
        ),
        effective_from=(
            effective_from
        ),
        version=version,
        status=(
            SourceStatus.CURRENT
        ),
        official_source_url=(
            "https://example.gov.jo/"
            f"{document_id}"
        ),
        content_hash=(
            content_hash
        ),
    )


def request():
    return PolicyChangeRequest(
        baseline_source=source(
            "POLICY-V1",
            content_hash=(
                "a" * 64
            ),
            version="1.0",
            effective_from=date(
                2026,
                1,
                1,
            ),
        ),
        candidate_source=source(
            "POLICY-V2",
            content_hash=(
                "b" * 64
            ),
            version="2.0",
            effective_from=date(
                2026,
                8,
                1,
            ),
        ),
        trace_id="TRACE-001",
    )


class TestPolicyChangeContract(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            POLICY_CHANGE_CONTRACT_VERSION,
            "govba-policy-change-contract-v1",
        )

    def test_request_is_created(self):
        value = request()

        self.assertEqual(
            value.baseline_source.document_id,
            "POLICY-V1",
        )

        self.assertEqual(
            value.candidate_source.document_id,
            "POLICY-V2",
        )

    def test_request_id_is_deterministic(self):
        self.assertEqual(
            request().request_id,
            request().request_id,
        )

    def test_source_identities_are_sha256(self):
        value = request()

        self.assertEqual(
            len(
                value.baseline_identity
            ),
            64,
        )

        self.assertEqual(
            len(
                value.candidate_identity
            ),
            64,
        )

    def test_different_versions_are_not_same_content(self):
        self.assertFalse(
            request().same_content
        )

    def test_same_hash_is_same_content(self):
        baseline = source(
            "A",
            content_hash="c" * 64,
            version="1",
            effective_from=date(
                2026,
                1,
                1,
            ),
        )

        candidate = source(
            "B",
            content_hash="c" * 64,
            version="2",
            effective_from=date(
                2026,
                2,
                1,
            ),
        )

        value = PolicyChangeRequest(
            baseline_source=baseline,
            candidate_source=candidate,
            trace_id="TRACE-001",
        )

        self.assertTrue(
            value.same_content
        )

    def test_request_serialization_excludes_urls(self):
        data = request().to_dict()

        self.assertNotIn(
            "official_source_url",
            repr(data),
        )

        self.assertNotIn(
            "https://",
            repr(data),
        )

    def test_added_requires_candidate_chunk(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.ADDED
            ),
            impact=(
                PolicyChangeImpact.MEDIUM
            ),
            candidate_chunk_id="NEW-1",
        )

        self.assertEqual(
            finding.candidate_chunk_id,
            "NEW-1",
        )

    def test_invalid_added_finding_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            PolicyChangeFinding(
                change_type=(
                    PolicyChangeType.ADDED
                ),
                impact=(
                    PolicyChangeImpact.LOW
                ),
                baseline_chunk_id="OLD",
                candidate_chunk_id="NEW",
            )

    def test_removed_requires_baseline_chunk(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.REMOVED
            ),
            impact=(
                PolicyChangeImpact.HIGH
            ),
            baseline_chunk_id="OLD-1",
        )

        self.assertEqual(
            finding.baseline_chunk_id,
            "OLD-1",
        )

    def test_modified_requires_both_chunks(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.MODIFIED
            ),
            impact=(
                PolicyChangeImpact.HIGH
            ),
            baseline_chunk_id="OLD",
            candidate_chunk_id="NEW",
        )

        self.assertEqual(
            finding.change_type,
            PolicyChangeType.MODIFIED,
        )

    def test_unchanged_requires_both_chunks(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.UNCHANGED
            ),
            impact=(
                PolicyChangeImpact.INFORMATIONAL
            ),
            baseline_chunk_id="OLD",
            candidate_chunk_id="NEW",
        )

        self.assertEqual(
            finding.change_type,
            PolicyChangeType.UNCHANGED,
        )

    def test_invalid_content_hash_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            PolicyChangeFinding(
                change_type=(
                    PolicyChangeType.MODIFIED
                ),
                impact=(
                    PolicyChangeImpact.HIGH
                ),
                baseline_chunk_id="OLD",
                candidate_chunk_id="NEW",
                baseline_content_hash="bad",
            )

    def test_finding_id_is_deterministic(self):
        kwargs = {
            "change_type": (
                PolicyChangeType.MODIFIED
            ),
            "impact": (
                PolicyChangeImpact.HIGH
            ),
            "baseline_chunk_id": "OLD",
            "candidate_chunk_id": "NEW",
            "reason_code": (
                "content_changed"
            ),
        }

        first = PolicyChangeFinding(
            **kwargs
        )

        second = PolicyChangeFinding(
            **kwargs
        )

        self.assertEqual(
            first.finding_id,
            second.finding_id,
        )

    def test_report_counts_changes(self):
        report = build_policy_change_report(
            request(),
            findings=(
                PolicyChangeFinding(
                    change_type=(
                        PolicyChangeType.ADDED
                    ),
                    impact=(
                        PolicyChangeImpact.MEDIUM
                    ),
                    candidate_chunk_id="NEW",
                ),
                PolicyChangeFinding(
                    change_type=(
                        PolicyChangeType.REMOVED
                    ),
                    impact=(
                        PolicyChangeImpact.HIGH
                    ),
                    baseline_chunk_id="OLD",
                ),
            ),
            algorithm="synthetic-v1",
        )

        self.assertEqual(
            report.added_count,
            1,
        )

        self.assertEqual(
            report.removed_count,
            1,
        )

        self.assertTrue(
            report.changed
        )

    def test_unchanged_only_report_is_not_changed(self):
        report = build_policy_change_report(
            request(),
            findings=(
                PolicyChangeFinding(
                    change_type=(
                        PolicyChangeType.UNCHANGED
                    ),
                    impact=(
                        PolicyChangeImpact.INFORMATIONAL
                    ),
                    baseline_chunk_id="OLD",
                    candidate_chunk_id="NEW",
                ),
            ),
            algorithm="synthetic-v1",
        )

        self.assertFalse(
            report.changed
        )

    def test_maximum_impact_is_reported(self):
        report = build_policy_change_report(
            request(),
            findings=(
                PolicyChangeFinding(
                    change_type=(
                        PolicyChangeType.ADDED
                    ),
                    impact=(
                        PolicyChangeImpact.LOW
                    ),
                    candidate_chunk_id="NEW-A",
                ),
                PolicyChangeFinding(
                    change_type=(
                        PolicyChangeType.MODIFIED
                    ),
                    impact=(
                        PolicyChangeImpact.CRITICAL
                    ),
                    baseline_chunk_id="OLD-B",
                    candidate_chunk_id="NEW-B",
                ),
            ),
            algorithm="synthetic-v1",
        )

        self.assertEqual(
            report.maximum_impact,
            PolicyChangeImpact.CRITICAL,
        )

    def test_empty_report_is_valid(self):
        report = build_policy_change_report(
            request(),
            findings=(),
            algorithm="synthetic-v1",
        )

        self.assertFalse(
            report.changed
        )

        self.assertEqual(
            report.maximum_impact,
            PolicyChangeImpact.INFORMATIONAL,
        )

    def test_duplicate_findings_are_rejected(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.ADDED
            ),
            impact=(
                PolicyChangeImpact.MEDIUM
            ),
            candidate_chunk_id="NEW",
        )

        with self.assertRaises(
            ValueError
        ):
            build_policy_change_report(
                request(),
                findings=(
                    finding,
                    finding,
                ),
                algorithm="synthetic-v1",
            )

    def test_report_id_is_deterministic(self):
        finding = PolicyChangeFinding(
            change_type=(
                PolicyChangeType.ADDED
            ),
            impact=(
                PolicyChangeImpact.MEDIUM
            ),
            candidate_chunk_id="NEW",
        )

        first = build_policy_change_report(
            request(),
            findings=(
                finding,
            ),
            algorithm="synthetic-v1",
        )

        second = build_policy_change_report(
            request(),
            findings=(
                finding,
            ),
            algorithm="synthetic-v1",
        )

        self.assertEqual(
            first.report_id,
            second.report_id,
        )

    def test_serialization_contains_no_raw_document_text(self):
        report = build_policy_change_report(
            request(),
            findings=(),
            algorithm="synthetic-v1",
        )

        data = report.to_dict()

        self.assertNotIn(
            "text",
            data,
        )

        self.assertNotIn(
            "prompt",
            repr(data),
        )


if __name__ == "__main__":
    unittest.main()
