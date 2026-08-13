"""Offline tests for GovBA-GAR deterministic PII redaction."""

from __future__ import annotations

import unittest

from govba.governance.pii import (
    PII_REDACTION_VERSION,
    PIIType,
    detect_pii,
    redact_pii,
)
from govba.governance.security import (
    SecurityDecision,
    SecurityIssueCode,
    SecuritySurface,
)


class TestPIIRedaction(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            PII_REDACTION_VERSION,
            "govba-pii-redaction-v1",
        )

    def test_email_is_detected(self):
        matches = detect_pii(
            "Contact synthetic.user@example.com today."
        )

        self.assertEqual(
            tuple(
                match.pii_type
                for match
                in matches
            ),
            (
                PIIType.EMAIL,
            ),
        )

    def test_email_is_redacted(self):
        original = (
            "Contact synthetic.user@example.com."
        )

        result = redact_pii(
            original,
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertNotIn(
            "synthetic.user@example.com",
            result.redacted_text,
        )

        self.assertIn(
            "[REDACTED_EMAIL]",
            result.redacted_text,
        )

    def test_international_phone_is_redacted(self):
        result = redact_pii(
            "Call +962 79 123 4567.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertIn(
            "[REDACTED_PHONE]",
            result.redacted_text,
        )

    def test_jordan_local_mobile_is_redacted(self):
        result = redact_pii(
            "Call 0791234567.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.DOCUMENT
            ),
        )

        self.assertIn(
            "[REDACTED_PHONE]",
            result.redacted_text,
        )

    def test_ipv4_address_is_redacted(self):
        result = redact_pii(
            "Internal host is 192.168.10.25.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.DOCUMENT
            ),
        )

        self.assertIn(
            "[REDACTED_IP]",
            result.redacted_text,
        )

    def test_invalid_ipv4_is_not_detected(self):
        matches = detect_pii(
            "Address 999.999.999.999 is invalid."
        )

        self.assertEqual(
            matches,
            (),
        )

    def test_valid_iban_is_redacted(self):
        result = redact_pii(
            "Test IBAN GB82 WEST 1234 5698 7654 32.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.MODEL_INPUT
            ),
        )

        self.assertIn(
            "[REDACTED_IBAN]",
            result.redacted_text,
        )

        self.assertTrue(
            result.sensitive_pii_detected
        )

    def test_invalid_iban_is_not_redacted(self):
        result = redact_pii(
            "Invalid IBAN GB00 WEST 1234 5698 7654 32.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.MODEL_INPUT
            ),
        )

        self.assertNotIn(
            "[REDACTED_IBAN]",
            result.redacted_text,
        )

    def test_valid_payment_card_is_redacted(self):
        result = redact_pii(
            "Synthetic test card 4111 1111 1111 1111.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.MODEL_INPUT
            ),
        )

        self.assertIn(
            "[REDACTED_PAYMENT_CARD]",
            result.redacted_text,
        )

        self.assertTrue(
            result.sensitive_pii_detected
        )

    def test_invalid_payment_card_is_not_redacted(self):
        result = redact_pii(
            "Number 4111 1111 1111 1112.",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.MODEL_INPUT
            ),
        )

        self.assertNotIn(
            "[REDACTED_PAYMENT_CARD]",
            result.redacted_text,
        )

    def test_multiple_types_are_redacted(self):
        result = redact_pii(
            (
                "Email synthetic.user@example.com "
                "and call +962791234567."
            ),
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertEqual(
            result.match_count,
            2,
        )

        self.assertEqual(
            set(
                result.pii_types
            ),
            {
                PIIType.EMAIL,
                PIIType.PHONE,
            },
        )

    def test_repeated_type_is_aggregated_in_security_finding(self):
        result = redact_pii(
            (
                "First a@example.com "
                "second b@example.com"
            ),
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.DOCUMENT
            ),
        )

        findings = (
            result.security_assessment
            .findings
        )

        self.assertEqual(
            len(
                findings
            ),
            1,
        )

        self.assertEqual(
            findings[
                0
            ].occurrence_count,
            2,
        )

    def test_ordinary_pii_maps_to_pii_security_code(self):
        result = redact_pii(
            "Email a@example.com",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertEqual(
            result.security_assessment
            .findings[
                0
            ].code,
            SecurityIssueCode.PII_DETECTED,
        )

    def test_sensitive_pii_maps_to_sensitive_code(self):
        result = redact_pii(
            "Card 4111 1111 1111 1111",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.MODEL_INPUT
            ),
        )

        self.assertEqual(
            result.security_assessment
            .findings[
                0
            ].code,
            (
                SecurityIssueCode
                .SENSITIVE_PII_DETECTED
            ),
        )

    def test_pii_requires_redaction(self):
        result = redact_pii(
            "Email a@example.com",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertEqual(
            result.security_assessment
            .decision,
            SecurityDecision.REDACT,
        )

        self.assertTrue(
            result.security_assessment
            .requires_redaction
        )

    def test_no_pii_allows(self):
        result = redact_pii(
            "What is the annual leave policy?",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertFalse(
            result.pii_detected
        )

        self.assertEqual(
            result.security_assessment
            .decision,
            SecurityDecision.ALLOW,
        )

        self.assertEqual(
            result.redacted_text,
            "What is the annual leave policy?",
        )

    def test_result_contains_sha256_not_original_text(self):
        result = redact_pii(
            "Email a@example.com",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertEqual(
            len(
                result.input_sha256
            ),
            64,
        )

        int(
            result.input_sha256,
            16,
        )

    def test_match_metadata_contains_no_raw_email(self):
        raw_value = "a@example.com"

        matches = detect_pii(
            f"Email {raw_value}"
        )

        self.assertNotIn(
            raw_value,
            repr(
                tuple(
                    match.to_dict()
                    for match
                    in matches
                )
            ),
        )

    def test_default_serialization_excludes_text(self):
        raw_value = (
            "synthetic.user@example.com"
        )

        result = redact_pii(
            f"Email {raw_value}",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        data = result.to_dict()

        self.assertNotIn(
            "redacted_text",
            data,
        )

        self.assertNotIn(
            raw_value,
            repr(data),
        )

    def test_redacted_text_can_be_explicitly_serialized(self):
        result = redact_pii(
            "Email a@example.com",
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        data = result.to_dict(
            include_redacted_text=True
        )

        self.assertEqual(
            data[
                "redacted_text"
            ],
            "Email [REDACTED_EMAIL]",
        )

    def test_redaction_is_deterministic(self):
        text = (
            "Email a@example.com "
            "and call +962791234567"
        )

        first = redact_pii(
            text,
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        second = redact_pii(
            text,
            trace_id="TRACE-001",
            surface=(
                SecuritySurface.USER_QUERY
            ),
        )

        self.assertEqual(
            first.redacted_text,
            second.redacted_text,
        )

        self.assertEqual(
            first.matches,
            second.matches,
        )

        self.assertEqual(
            first.security_assessment
            .assessment_id,
            second.security_assessment
            .assessment_id,
        )

    def test_invalid_surface_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            redact_pii(
                "Email a@example.com",
                trace_id="TRACE-001",
                surface="user_query",
            )

    def test_blank_text_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            redact_pii(
                " ",
                trace_id="TRACE-001",
                surface=(
                    SecuritySurface.USER_QUERY
                ),
            )

    def test_blank_trace_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            redact_pii(
                "Email a@example.com",
                trace_id=" ",
                surface=(
                    SecuritySurface.USER_QUERY
                ),
            )


if __name__ == "__main__":
    unittest.main()
