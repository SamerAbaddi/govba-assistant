"""Offline tests for GovBA-GAR government web adapter."""

from __future__ import annotations

import unittest

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
)
from govba.rag.language import (
    BilingualLanguage,
)
from govba.web.adapter import (
    DEFAULT_WEB_CANDIDATE_POOL,
    GOVERNMENT_WEB_ADAPTER_VERSION,
    GovernmentWebAdapter,
    RawWebSearchResult,
    WebSearchBackend,
    is_official_web_retriever,
)
from govba.web.contract import (
    OfficialWebRequest,
)


def raw(
    *,
    title="Policy",
    url="https://example.gov.jo/policy",
    text="Synthetic policy content.",
    score=0.9,
    provider_rank=1,
):
    return RawWebSearchResult(
        title=title,
        url=url,
        text=text,
        score=score,
        provider_rank=provider_rank,
    )


class RecordingBackend:
    def __init__(
        self,
        results=(),
    ):
        self.results = tuple(
            results
        )
        self.calls = []

    def search(
        self,
        *,
        query_text,
        language,
        limit,
        allowed_domains,
    ):
        self.calls.append(
            {
                "query_text": (
                    query_text
                ),
                "language": (
                    language
                ),
                "limit": (
                    limit
                ),
                "allowed_domains": (
                    allowed_domains
                ),
            }
        )

        return list(
            self.results
        )


def policy():
    return SourceAllowlistPolicy(
        allowed_domains=(
            "example.gov.jo",
            "agency.gov.jo",
        )
    )


def request(
    *,
    top_k=5,
):
    return OfficialWebRequest(
        query_text=(
            "annual leave policy"
        ),
        language=(
            BilingualLanguage.ENGLISH
        ),
        top_k=top_k,
    )


class TestRawWebSearchResult(
    unittest.TestCase
):
    def test_valid_result_is_created(self):
        value = raw()

        self.assertEqual(
            value.provider_rank,
            1,
        )

    def test_blank_title_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            raw(
                title=" "
            )

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            raw(
                score=1.5
            )

    def test_invalid_rank_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            raw(
                provider_rank=0
            )


class TestGovernmentWebAdapter(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            GOVERNMENT_WEB_ADAPTER_VERSION,
            "govba-government-web-adapter-v1",
        )

    def test_default_candidate_pool_is_stable(self):
        self.assertEqual(
            DEFAULT_WEB_CANDIDATE_POOL,
            20,
        )

    def test_backend_protocol_is_structural(self):
        backend = RecordingBackend()

        self.assertIsInstance(
            backend,
            WebSearchBackend,
        )

    def test_adapter_satisfies_official_retriever_contract(self):
        adapter = GovernmentWebAdapter(
            backend=RecordingBackend(),
            policy=policy(),
        )

        self.assertTrue(
            is_official_web_retriever(
                adapter
            )
        )

    def test_request_is_forwarded_to_backend(self):
        backend = RecordingBackend()

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        adapter.search(
            request()
        )

        self.assertEqual(
            backend.calls[0][
                "query_text"
            ],
            "annual leave policy",
        )

        self.assertEqual(
            backend.calls[0][
                "language"
            ],
            BilingualLanguage.ENGLISH,
        )

    def test_allowlisted_domains_are_forwarded(self):
        backend = RecordingBackend()

        active_policy = policy()

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=active_policy,
        )

        adapter.search(
            request()
        )

        self.assertEqual(
            backend.calls[0][
                "allowed_domains"
            ],
            active_policy.allowed_domains,
        )

    def test_candidate_pool_overfetches(self):
        backend = RecordingBackend()

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        adapter.search(
            request(
                top_k=3
            )
        )

        self.assertEqual(
            backend.calls[0][
                "limit"
            ],
            20,
        )

    def test_top_k_larger_than_pool_is_used(self):
        backend = RecordingBackend()

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
            candidate_pool_size=5,
        )

        adapter.search(
            request(
                top_k=10
            )
        )

        self.assertEqual(
            backend.calls[0][
                "limit"
            ],
            10,
        )

    def test_trusted_result_is_returned(self):
        backend = RecordingBackend(
            (
                raw(),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].domain,
            "example.gov.jo",
        )

    def test_untrusted_domain_is_discarded(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "https://evil.example/"
                        "policy"
                    )
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            results,
            [],
        )

    def test_http_result_is_discarded(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "http://example.gov.jo/"
                        "policy"
                    )
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            results,
            [],
        )

    def test_lookalike_domain_is_discarded(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "https://example.gov.jo."
                        "evil.example/policy"
                    )
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        self.assertEqual(
            adapter.search(
                request()
            ),
            [],
        )

    def test_valid_subdomain_is_allowed(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "https://docs."
                        "example.gov.jo/policy"
                    )
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].domain,
            "docs.example.gov.jo",
        )

    def test_duplicate_urls_are_removed(self):
        backend = RecordingBackend(
            (
                raw(
                    provider_rank=1,
                    score=0.9,
                ),
                raw(
                    provider_rank=2,
                    score=0.8,
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            len(results),
            1,
        )

    def test_fragment_does_not_create_duplicate(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "https://example.gov.jo/"
                        "policy#one"
                    ),
                    provider_rank=1,
                ),
                raw(
                    url=(
                        "https://example.gov.jo/"
                        "policy#two"
                    ),
                    provider_rank=2,
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            len(results),
            1,
        )

    def test_provider_rank_controls_candidate_order(self):
        backend = RecordingBackend(
            (
                raw(
                    title="Second",
                    url=(
                        "https://example.gov.jo/"
                        "second"
                    ),
                    provider_rank=2,
                    score=0.99,
                ),
                raw(
                    title="First",
                    url=(
                        "https://example.gov.jo/"
                        "first"
                    ),
                    provider_rank=1,
                    score=0.80,
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request(
                top_k=2
            )
        )

        self.assertEqual(
            tuple(
                result.title
                for result in results
            ),
            (
                "First",
                "Second",
            ),
        )

    def test_final_ranks_are_contiguous(self):
        backend = RecordingBackend(
            (
                raw(
                    url=(
                        "https://evil.example/a"
                    ),
                    provider_rank=1,
                ),
                raw(
                    title="Trusted",
                    url=(
                        "https://example.gov.jo/b"
                    ),
                    provider_rank=2,
                ),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request()
        )

        self.assertEqual(
            results[0].rank,
            1,
        )

    def test_top_k_is_enforced_after_filtering(self):
        backend = RecordingBackend(
            tuple(
                raw(
                    title=f"Policy {index}",
                    url=(
                        "https://example.gov.jo/"
                        f"policy-{index}"
                    ),
                    score=(
                        1.0
                        - index * 0.05
                    ),
                    provider_rank=(
                        index + 1
                    ),
                )
                for index in range(
                    5
                )
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        results = adapter.search(
            request(
                top_k=2
            )
        )

        self.assertEqual(
            len(results),
            2,
        )

    def test_retrieval_method_is_controlled(self):
        backend = RecordingBackend(
            (
                raw(),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
            retrieval_method=(
                "synthetic-backend-v1"
            ),
        )

        result = adapter.search(
            request()
        )[0]

        self.assertEqual(
            result.retrieval_method,
            "synthetic-backend-v1",
        )

    def test_invalid_request_is_rejected(self):
        adapter = GovernmentWebAdapter(
            backend=RecordingBackend(),
            policy=policy(),
        )

        with self.assertRaises(
            TypeError
        ):
            adapter.search(
                object()
            )

    def test_invalid_candidate_pool_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GovernmentWebAdapter(
                backend=RecordingBackend(),
                policy=policy(),
                candidate_pool_size=101,
            )

    def test_invalid_backend_result_type_is_rejected(self):
        backend = RecordingBackend(
            (
                object(),
            )
        )

        adapter = GovernmentWebAdapter(
            backend=backend,
            policy=policy(),
        )

        with self.assertRaises(
            TypeError
        ):
            adapter.search(
                request()
            )


if __name__ == "__main__":
    unittest.main()
