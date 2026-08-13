"""Offline tests for the GovBA-GAR security contract."""

from __future__ import annotations

import unittest

from govba.governance.security import (
    SECURITY_CONTRACT_VERSION,
    SecurityAssessment,
    SecurityDecision,
    SecurityFinding,
    SecurityIssueCode,
    SecuritySeverity,
    SecuritySurface,
    assess_security,
)


def finding(
    *,
    decision=SecurityDecision.REDACT,
    severity=SecuritySeverity.MEDIUM,
    code=SecurityIssueCode.PII_DETECTED,
    surface=SecuritySurface.USER_QUERY,
    reference_id="REF-001",
):
    return SecurityFinding(
        code=code,
        surface=surface,
        severity=severity,
        decision=decision,
        reference_id=reference_id,
    )


class TestSecurityContract(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            SECURITY_CONTRACT_VERSION,
            "govba-security-contract-v1",
        )

    def test_finding_is_created(self):
        value = finding()

        self.assertEqual(
            value.code,
            SecurityIssueCode.PII_DETECTED,
        )

        self.assertEqual(
            value.decision,
            SecurityDecision.REDACT,
        )

    def test_finding_id_is_sha256(self):
        value = finding()

        self.assertEqual(
            len(
                value.finding_id
            ),
            64,
        )

        int(
            value.finding_id,
            16,
        )

    def test_finding_id_is_deterministic(self):
        self.assertEqual(
            finding().finding_id,
            finding().finding_id,
        )

    def test_invalid_occurrence_count_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            SecurityFinding(
                code=(
                    SecurityIssueCode
                    .PII_DETECTED
                ),
                surface=(
                    SecuritySurface.USER_QUERY
                ),
                severity=(
                    SecuritySeverity.MEDIUM
                ),
                decision=(
                    SecurityDecision.REDACT
                ),
                occurrence_count=0,
            )

    def test_invalid_issue_code_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            SecurityFinding(
                code="pii_detected",
                surface=(
                    SecuritySurface.USER_QUERY
                ),
                severity=(
                    SecuritySeverity.MEDIUM
                ),
                decision=(
                    SecurityDecision.REDACT
                ),
            )

    def test_no_findings_allows(self):
        assessment = assess_security(
            trace_id="TRACE-001"
        )

        self.assertEqual(
            assessment.decision,
            SecurityDecision.ALLOW,
        )

        self.assertTrue(
            assessment.allowed
        )

    def test_redaction_is_required(self):
        assessment = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(
                    decision=(
                        SecurityDecision.REDACT
                    )
                ),
            ),
        )

        self.assertEqual(
            assessment.decision,
            SecurityDecision.REDACT,
        )

        self.assertTrue(
            assessment.requires_redaction
        )

    def test_review_overrides_redaction(self):
        assessment = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(
                    decision=(
                        SecurityDecision.REDACT
                    ),
                    reference_id="REF-A",
                ),
                finding(
                    decision=(
                        SecurityDecision.REVIEW
                    ),
                    code=(
                        SecurityIssueCode
                        .UNTRUSTED_SOURCE
                    ),
                    surface=(
                        SecuritySurface.SOURCE
                    ),
                    reference_id="REF-B",
                ),
            ),
        )

        self.assertEqual(
            assessment.decision,
            SecurityDecision.REVIEW,
        )

    def test_block_has_highest_priority(self):
        assessment = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(
                    decision=(
                        SecurityDecision.REDACT
                    ),
                    reference_id="A",
                ),
                finding(
                    decision=(
                        SecurityDecision.REVIEW
                    ),
                    reference_id="B",
                ),
                finding(
                    decision=(
                        SecurityDecision.BLOCK
                    ),
                    severity=(
                        SecuritySeverity.CRITICAL
                    ),
                    code=(
                        SecurityIssueCode
                        .PROMPT_INJECTION_DETECTED
                    ),
                    reference_id="C",
                ),
            ),
        )

        self.assertEqual(
            assessment.decision,
            SecurityDecision.BLOCK,
        )

        self.assertTrue(
            assessment.should_block
        )

    def test_blank_trace_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            assess_security(
                trace_id=" "
            )

    def test_duplicate_findings_are_rejected(self):
        value = finding()

        with self.assertRaises(
            ValueError
        ):
            SecurityAssessment(
                trace_id="TRACE-001",
                findings=(
                    value,
                    value,
                ),
            )

    def test_assessment_id_is_deterministic(self):
        first = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(),
            ),
        )

        second = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(),
            ),
        )

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

    def test_assessment_id_is_order_independent(self):
        first_finding = finding(
            reference_id="A"
        )

        second_finding = finding(
            code=(
                SecurityIssueCode
                .UNTRUSTED_SOURCE
            ),
            surface=(
                SecuritySurface.SOURCE
            ),
            decision=(
                SecurityDecision.REVIEW
            ),
            reference_id="B",
        )

        first = assess_security(
            trace_id="TRACE-001",
            findings=(
                first_finding,
                second_finding,
            ),
        )

        second = assess_security(
            trace_id="TRACE-001",
            findings=(
                second_finding,
                first_finding,
            ),
        )

        self.assertEqual(
            first.assessment_id,
            second.assessment_id,
        )

    def test_critical_findings_are_counted(self):
        assessment = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(
                    severity=(
                        SecuritySeverity.CRITICAL
                    ),
                    decision=(
                        SecurityDecision.BLOCK
                    ),
                ),
            ),
        )

        self.assertEqual(
            assessment.critical_finding_count,
            1,
        )

    def test_serialization_is_privacy_safe(self):
        assessment = assess_security(
            trace_id="TRACE-001",
            findings=(
                finding(),
            ),
        )

        data = assessment.to_dict()

        self.assertNotIn(
            "query_text",
            data,
        )

        self.assertNotIn(
            "document_text",
            data,
        )

        self.assertNotIn(
            "prompt",
            data,
        )

        self.assertNotIn(
            "email",
            data,
        )

        self.assertEqual(
            data["decision"],
            "redact",
        )


if __name__ == "__main__":
    unittest.main()
