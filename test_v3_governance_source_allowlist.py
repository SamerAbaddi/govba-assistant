"""Offline tests for GovBA-GAR government-source allowlisting."""

from __future__ import annotations

import unittest

from govba.governance.security import (
    SecurityDecision,
    SecurityIssueCode,
)
from govba.governance.source_allowlist import (
    SOURCE_ALLOWLIST_VERSION,
    SourceAllowlistPolicy,
    SourceAllowlistReasonCode,
    assess_authoritative_source,
    assess_source_url,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)


def policy(
    *domains,
    allow_subdomains=True,
    require_https=True,
):
    return SourceAllowlistPolicy(
        allowed_domains=(
            domains
            or (
                "example.gov.jo",
            )
        ),
        allow_subdomains=(
            allow_subdomains
        ),
        require_https=(
            require_https
        ),
    )


def assess(
    url,
    *,
    active_policy=None,
):
    return assess_source_url(
        url,
        document_id="DOC-001",
        trace_id="TRACE-001",
        policy=(
            active_policy
            or policy()
        ),
    )


class TestGovernmentSourceAllowlist(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            SOURCE_ALLOWLIST_VERSION,
            "govba-source-allowlist-v1",
        )

    def test_policy_normalizes_domains(self):
        value = SourceAllowlistPolicy(
            allowed_domains=(
                "EXAMPLE.GOV.JO.",
            )
        )

        self.assertEqual(
            value.allowed_domains,
            (
                "example.gov.jo",
            ),
        )

    def test_empty_allowlist_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            SourceAllowlistPolicy(
                allowed_domains=()
            )

    def test_wildcard_domain_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            SourceAllowlistPolicy(
                allowed_domains=(
                    "*.gov.jo",
                )
            )

    def test_url_instead_of_domain_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            SourceAllowlistPolicy(
                allowed_domains=(
                    "https://example.gov.jo",
                )
            )

    def test_exact_domain_is_allowed(self):
        result = assess(
            "https://example.gov.jo/policy.pdf"
        )

        self.assertTrue(
            result.allowed
        )

        self.assertEqual(
            result.reason,
            SourceAllowlistReasonCode.ALLOWED,
        )

    def test_subdomain_is_allowed(self):
        result = assess(
            "https://docs.example.gov.jo/policy.pdf"
        )

        self.assertTrue(
            result.allowed
        )

    def test_subdomain_can_be_disabled(self):
        result = assess(
            "https://docs.example.gov.jo/policy.pdf",
            active_policy=policy(
                "example.gov.jo",
                allow_subdomains=False,
            ),
        )

        self.assertTrue(
            result.should_block
        )

    def test_lookalike_domain_is_rejected(self):
        result = assess(
            "https://example-gov.jo/policy.pdf"
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .DISALLOWED_DOMAIN
            ),
        )

    def test_suffix_attack_is_rejected(self):
        result = assess(
            "https://example.gov.jo.evil.example/policy"
        )

        self.assertFalse(
            result.allowed
        )

    def test_missing_url_blocks(self):
        result = assess(
            ""
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .MISSING_OFFICIAL_URL
            ),
        )

        self.assertTrue(
            result.should_block
        )

    def test_http_is_rejected_by_default(self):
        result = assess(
            "http://example.gov.jo/policy.pdf"
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .NON_HTTPS_URL
            ),
        )

    def test_http_can_be_explicitly_allowed(self):
        result = assess(
            "http://example.gov.jo/policy.pdf",
            active_policy=policy(
                "example.gov.jo",
                require_https=False,
            ),
        )

        self.assertTrue(
            result.allowed
        )

    def test_embedded_credentials_are_rejected(self):
        result = assess(
            "https://user:password@example.gov.jo/policy"
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .EMBEDDED_CREDENTIALS
            ),
        )

    def test_ip_literal_is_rejected(self):
        result = assess(
            "https://192.0.2.10/policy"
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .IP_LITERAL_HOST
            ),
        )

    def test_nonstandard_port_is_rejected(self):
        result = assess(
            "https://example.gov.jo:8443/policy"
        )

        self.assertEqual(
            result.reason,
            (
                SourceAllowlistReasonCode
                .DISALLOWED_PORT
            ),
        )

    def test_explicit_standard_https_port_is_allowed(self):
        result = assess(
            "https://example.gov.jo:443/policy"
        )

        self.assertTrue(
            result.allowed
        )

    def test_path_query_and_fragment_do_not_change_domain_trust(self):
        result = assess(
            (
                "https://example.gov.jo/"
                "policy.pdf?version=2#section"
            )
        )

        self.assertTrue(
            result.allowed
        )

        self.assertEqual(
            result.domain,
            "example.gov.jo",
        )

    def test_disallowed_domain_uses_security_code(self):
        result = assess(
            "https://evil.example/policy"
        )

        self.assertEqual(
            result.security_assessment
            .findings[0].code,
            (
                SecurityIssueCode
                .DISALLOWED_DOMAIN
            ),
        )

    def test_missing_url_uses_untrusted_source_code(self):
        result = assess(
            ""
        )

        self.assertEqual(
            result.security_assessment
            .findings[0].code,
            (
                SecurityIssueCode
                .UNTRUSTED_SOURCE
            ),
        )

    def test_disallowed_source_blocks(self):
        result = assess(
            "https://evil.example/policy"
        )

        self.assertEqual(
            result.security_assessment
            .decision,
            SecurityDecision.BLOCK,
        )

    def test_allowed_source_has_allow_security_decision(self):
        result = assess(
            "https://example.gov.jo/policy"
        )

        self.assertEqual(
            result.security_assessment
            .decision,
            SecurityDecision.ALLOW,
        )

    def test_policy_id_is_order_independent(self):
        first = policy(
            "example.gov.jo",
            "agency.gov.jo",
        )

        second = policy(
            "agency.gov.jo",
            "example.gov.jo",
        )

        self.assertEqual(
            first.policy_id,
            second.policy_id,
        )

    def test_serialization_does_not_expose_full_url(self):
        raw_url = (
            "https://example.gov.jo/"
            "private/path?token=synthetic-secret"
        )

        result = assess(
            raw_url
        )

        data = result.to_dict()

        self.assertNotIn(
            raw_url,
            repr(data),
        )

        self.assertNotIn(
            "synthetic-secret",
            repr(data),
        )

        self.assertEqual(
            data["domain"],
            "example.gov.jo",
        )

    def test_authoritative_source_integration(self):
        source = AuthoritativeSource(
            document_id="DOC-OFFICIAL",
            title="Synthetic Government Policy",
            issuing_authority=(
                "Synthetic Government Authority"
            ),
            document_type=(
                DocumentType.POLICY
            ),
            language=(
                SourceLanguage.ENGLISH
            ),
            status=(
                SourceStatus.CURRENT
            ),
            official_source_url=(
                "https://example.gov.jo/policy.pdf"
            ),
        )

        result = assess_authoritative_source(
            source,
            trace_id="TRACE-001",
            policy=policy(),
        )

        self.assertTrue(
            result.allowed
        )

        self.assertEqual(
            result.document_id,
            "DOC-OFFICIAL",
        )

    def test_invalid_source_type_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            assess_authoritative_source(
                object(),
                trace_id="TRACE-001",
                policy=policy(),
            )

    def test_blank_trace_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            assess_source_url(
                "https://example.gov.jo",
                document_id="DOC-001",
                trace_id=" ",
                policy=policy(),
            )


if __name__ == "__main__":
    unittest.main()
