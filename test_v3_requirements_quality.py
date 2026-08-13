"""Offline tests for GovBA-GAR requirement quality intelligence."""

from __future__ import annotations

import unittest

from govba.rag.models import (
    SourceLanguage,
)
from govba.requirements.contract import (
    RequirementOrigin,
    RequirementsRequest,
)
from govba.requirements.extraction import (
    extract_requirements,
)
from govba.requirements.quality import (
    REQUIREMENT_QUALITY_VERSION,
    RequirementQualityDecision,
    RequirementQualityIssueCode,
    RequirementQualitySeverity,
    assess_requirement_quality,
    assess_requirements_quality,
)


def request(
    text,
    *,
    language=SourceLanguage.ENGLISH,
):
    return RequirementsRequest(
        source_text=text,
        language=language,
        project_id="PROJECT-001",
        trace_id="TRACE-001",
        origin=(
            RequirementOrigin.USER_INPUT
        ),
    )


def one_requirement(
    text,
    *,
    language=SourceLanguage.ENGLISH,
):
    extraction = extract_requirements(
        request(
            text,
            language=language,
        )
    )

    if (
        extraction.requirement_count
        != 1
    ):
        raise AssertionError(
            "Test helper expected exactly "
            "one extracted requirement."
        )

    return (
        extraction,
        extraction.requirements[0],
    )


class TestRequirementQuality(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            REQUIREMENT_QUALITY_VERSION,
            "govba-requirement-quality-v1",
        )

    def test_clear_functional_requirement_passes(self):
        _, requirement = one_requirement(
            "The system must allow users "
            "to submit requests."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.PASS,
        )

        self.assertEqual(
            result.findings,
            (),
        )

    def test_ambiguous_term_requires_review(self):
        _, requirement = one_requirement(
            "The system should respond "
            "as soon as possible."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REVIEW,
        )

        self.assertIn(
            RequirementQualityIssueCode
            .AMBIGUOUS_TERM,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

    def test_subjective_term_requires_review(self):
        _, requirement = one_requirement(
            "The system should provide "
            "a user-friendly interface."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertIn(
            RequirementQualityIssueCode
            .SUBJECTIVE_TERM,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

    def test_tbd_placeholder_requires_rewrite(self):
        _, requirement = one_requirement(
            "The system must use an authentication "
            "method to be determined."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REWRITE,
        )

        finding = next(
            finding
            for finding
            in result.findings
            if (
                finding.issue_code
                is RequirementQualityIssueCode
                .TBD_PLACEHOLDER
            )
        )

        self.assertEqual(
            finding.severity,
            RequirementQualitySeverity.HIGH,
        )

    def test_unquantified_nonfunctional_requires_rewrite(self):
        _, requirement = one_requirement(
            "The system must provide high availability."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REWRITE,
        )

        self.assertIn(
            RequirementQualityIssueCode
            .UNQUANTIFIED_NON_FUNCTIONAL,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

    def test_quantified_nonfunctional_can_pass(self):
        _, requirement = one_requirement(
            "The system must maintain "
            "99.9% availability."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.PASS,
        )

    def test_multiple_obligations_require_review(self):
        _, requirement = one_requirement(
            "The system must authenticate users "
            "and shall record access."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertIn(
            RequirementQualityIssueCode
            .MULTIPLE_OBLIGATIONS,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REVIEW,
        )

    def test_english_implicit_actor_requires_review(self):
        _, requirement = one_requirement(
            "Must allow users to submit requests."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertIn(
            RequirementQualityIssueCode
            .IMPLICIT_ACTOR,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

    def test_arabic_subjective_requirement_requires_review(self):
        _, requirement = one_requirement(
            "يجب أن تكون الواجهة سهلة الاستخدام.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REVIEW,
        )

        self.assertIn(
            RequirementQualityIssueCode
            .SUBJECTIVE_TERM,
            tuple(
                finding.issue_code
                for finding
                in result.findings
            ),
        )

    def test_arabic_tbd_requires_rewrite(self):
        _, requirement = one_requirement(
            "يجب أن يدعم النظام طريقة مصادقة "
            "يتم تحديدها لاحقاً.",
            language=(
                SourceLanguage.ARABIC
            ),
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        self.assertEqual(
            result.decision,
            RequirementQualityDecision.REWRITE,
        )

    def test_quality_assessment_is_deterministic(self):
        _, requirement = one_requirement(
            "The system should provide "
            "a user-friendly interface."
        )

        first = assess_requirement_quality(
            requirement
        )

        second = assess_requirement_quality(
            requirement
        )

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

        self.assertEqual(
            tuple(
                finding.finding_id
                for finding
                in first.findings
            ),
            tuple(
                finding.finding_id
                for finding
                in second.findings
            ),
        )

    def test_serialization_excludes_requirement_text(self):
        raw = (
            "The system should provide a unique "
            "private user-friendly workflow."
        )

        _, requirement = one_requirement(
            raw
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        self.assertNotIn(
            "statement",
            data,
        )

    def test_finding_contains_no_matched_raw_term(self):
        _, requirement = one_requirement(
            "The system should respond "
            "as soon as possible."
        )

        result = (
            assess_requirement_quality(
                requirement
            )
        )

        data = result.to_dict()

        self.assertNotIn(
            "as soon as possible",
            repr(data),
        )


class TestRequirementsQualityReport(
    unittest.TestCase
):
    def test_report_preserves_requirement_order(self):
        extraction = extract_requirements(
            request(
                "The system must allow login. "
                "The system should provide "
                "a user-friendly interface."
            )
        )

        report = (
            assess_requirements_quality(
                extraction
            )
        )

        self.assertEqual(
            tuple(
                assessment.requirement_id
                for assessment
                in report.assessments
            ),
            tuple(
                requirement.requirement_id
                for requirement
                in extraction.requirements
            ),
        )

    def test_report_counts_decisions(self):
        extraction = extract_requirements(
            request(
                "The system must allow login. "
                "The system should provide "
                "a user-friendly interface. "
                "The system must provide "
                "high availability."
            )
        )

        report = (
            assess_requirements_quality(
                extraction
            )
        )

        self.assertEqual(
            report.pass_count,
            1,
        )

        self.assertEqual(
            report.review_count,
            1,
        )

        self.assertEqual(
            report.rewrite_count,
            1,
        )

    def test_empty_extraction_is_supported(self):
        extraction = extract_requirements(
            request(
                "This paragraph only provides "
                "general background information."
            )
        )

        report = (
            assess_requirements_quality(
                extraction
            )
        )

        self.assertEqual(
            report.assessments,
            (),
        )

        self.assertEqual(
            report.issue_count,
            0,
        )

    def test_report_is_deterministic(self):
        extraction = extract_requirements(
            request(
                "The system should provide "
                "a user-friendly interface."
            )
        )

        first = assess_requirements_quality(
            extraction
        )

        second = assess_requirements_quality(
            extraction
        )

        self.assertEqual(
            first.report_id,
            second.report_id,
        )

    def test_report_serialization_is_privacy_safe(self):
        raw = (
            "The system should provide a unique "
            "private user-friendly interface."
        )

        extraction = extract_requirements(
            request(raw)
        )

        report = (
            assess_requirements_quality(
                extraction
            )
        )

        data = report.to_dict()

        self.assertNotIn(
            raw,
            repr(data),
        )

        # Hash metadata is intentionally retained, but
        # the raw requirement statement must never appear.
        for assessment in data["assessments"]:
            self.assertNotIn(
                "statement",
                assessment,
            )

            self.assertIn(
                "statement_sha256",
                assessment,
            )

            for finding in assessment["findings"]:
                self.assertNotIn(
                    "statement",
                    finding,
                )

                self.assertIn(
                    "statement_sha256",
                    finding,
                )

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            assess_requirements_quality(
                "not-an-extraction"
            )


if __name__ == "__main__":
    unittest.main()
