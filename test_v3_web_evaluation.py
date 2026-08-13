"""Offline tests for GovBA-GAR controlled-web evaluation."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from govba.governance.source_allowlist import (
    SourceAllowlistPolicy,
)
from govba.rag.language import (
    BilingualLanguage,
)
from govba.web.contract import (
    OfficialWebResult,
)
from govba.web.evaluation import (
    CONTROLLED_WEB_EVALUATION_VERSION,
    ControlledWebGoldCase,
    evaluate_controlled_web,
)


FIXED_TIME = datetime(
    2026,
    8,
    13,
    10,
    0,
    tzinfo=timezone.utc,
)


def policy():
    return SourceAllowlistPolicy(
        allowed_domains=(
            "example.gov.jo",
        )
    )


def result(
    *,
    url,
    rank,
    text="Synthetic official evidence.",
):
    return OfficialWebResult(
        title=(
            f"Result {rank}"
        ),
        url=url,
        text=text,
        rank=rank,
        score=max(
            0.0,
            1.0
            - rank * 0.1,
        ),
        retrieval_method=(
            "evaluation-test-v1"
        ),
        retrieved_at=FIXED_TIME,
    )


def gold(
    *,
    case_id="CASE-001",
    query="annual leave policy",
    language=(
        BilingualLanguage.ENGLISH
    ),
    relevant_urls=(
        "https://example.gov.jo/policy",
    ),
    top_k=5,
):
    return ControlledWebGoldCase(
        case_id=case_id,
        query_text=query,
        language=language,
        relevant_urls=(
            relevant_urls
        ),
        top_k=top_k,
    )


class StaticRetriever:
    def __init__(
        self,
        results_by_query,
    ):
        self.results_by_query = (
            results_by_query
        )

    def search(
        self,
        request,
    ):
        return list(
            self.results_by_query.get(
                request.query_text,
                (),
            )
        )


class TestControlledWebGoldCase(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CONTROLLED_WEB_EVALUATION_VERSION,
            "govba-controlled-web-evaluation-v1",
        )

    def test_gold_case_is_created(self):
        case = gold()

        self.assertEqual(
            case.case_id,
            "CASE-001",
        )

    def test_query_is_trimmed(self):
        case = gold(
            query="  policy  "
        )

        self.assertEqual(
            case.query_text,
            "policy",
        )

    def test_blank_case_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                case_id=" "
            )

    def test_empty_relevant_urls_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                relevant_urls=()
            )

    def test_duplicate_canonical_urls_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                relevant_urls=(
                    (
                        "https://example.gov.jo/"
                        "policy#one"
                    ),
                    (
                        "https://example.gov.jo/"
                        "policy#two"
                    ),
                )
            )

    def test_http_gold_url_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            gold(
                relevant_urls=(
                    "http://example.gov.jo/policy",
                )
            )

    def test_default_serialization_excludes_query(self):
        raw = "sensitive evaluation query"

        case = gold(
            query=raw
        )

        data = case.to_dict()

        self.assertNotIn(
            "query_text",
            data,
        )

        self.assertNotIn(
            raw,
            repr(data),
        )


class TestControlledWebEvaluation(
    unittest.TestCase
):
    def test_hit_at_one(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy"
                        ),
                        rank=1,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        case = benchmark.cases[0]

        self.assertTrue(
            case.hit_at_1
        )

        self.assertTrue(
            case.hit_at_k
        )

        self.assertEqual(
            case.reciprocal_rank,
            1.0,
        )

    def test_relevant_result_at_rank_two(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "other"
                        ),
                        rank=1,
                    ),
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy"
                        ),
                        rank=2,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        case = benchmark.cases[0]

        self.assertFalse(
            case.hit_at_1
        )

        self.assertTrue(
            case.hit_at_k
        )

        self.assertEqual(
            case.reciprocal_rank,
            0.5,
        )

    def test_complete_miss(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "unrelated"
                        ),
                        rank=1,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        case = benchmark.cases[0]

        self.assertFalse(
            case.hit_at_k
        )

        self.assertEqual(
            case.recall,
            0.0,
        )

        self.assertEqual(
            case.reciprocal_rank,
            0.0,
        )

    def test_partial_recall(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/a"
                        ),
                        rank=1,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(
                    relevant_urls=(
                        (
                            "https://example.gov.jo/a"
                        ),
                        (
                            "https://example.gov.jo/b"
                        ),
                    )
                ),
            ),
            policy=policy(),
        )

        self.assertEqual(
            benchmark.cases[
                0
            ].recall,
            0.5,
        )

    def test_zero_result_is_measured(self):
        benchmark = evaluate_controlled_web(
            StaticRetriever({}),
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        case = benchmark.cases[0]

        self.assertTrue(
            case.zero_result
        )

        self.assertEqual(
            case.returned_count,
            0,
        )

        self.assertEqual(
            benchmark.metrics
            .zero_result_rate,
            1.0,
        )

    def test_untrusted_result_is_measured(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://evil.example/"
                            "policy"
                        ),
                        rank=1,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        case = benchmark.cases[0]

        self.assertEqual(
            case.untrusted_result_count,
            1,
        )

        self.assertEqual(
            benchmark.metrics
            .untrusted_result_rate,
            1.0,
        )

    def test_trusted_result_is_counted(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy"
                        ),
                        rank=1,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        self.assertEqual(
            benchmark.cases[
                0
            ].trusted_result_count,
            1,
        )

        self.assertEqual(
            benchmark.metrics
            .untrusted_result_rate,
            0.0,
        )

    def test_aggregate_metrics(self):
        retriever = StaticRetriever(
            {
                "query one": (
                    result(
                        url=(
                            "https://example.gov.jo/a"
                        ),
                        rank=1,
                    ),
                ),
                "query two": (),
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(
                    case_id="A",
                    query="query one",
                    relevant_urls=(
                        "https://example.gov.jo/a",
                    ),
                ),
                gold(
                    case_id="B",
                    query="query two",
                    relevant_urls=(
                        "https://example.gov.jo/b",
                    ),
                ),
            ),
            policy=policy(),
        )

        self.assertEqual(
            benchmark.metrics.case_count,
            2,
        )

        self.assertEqual(
            benchmark.metrics.hit_at_1_rate,
            0.5,
        )

        self.assertEqual(
            benchmark.metrics.hit_at_k_rate,
            0.5,
        )

        self.assertEqual(
            benchmark.metrics.mean_recall,
            0.5,
        )

        self.assertEqual(
            benchmark.metrics.mrr,
            0.5,
        )

        self.assertEqual(
            benchmark.metrics.zero_result_rate,
            0.5,
        )

    def test_arabic_language_is_preserved(self):
        benchmark = evaluate_controlled_web(
            StaticRetriever({}),
            cases=(
                gold(
                    language=(
                        BilingualLanguage.ARABIC
                    )
                ),
            ),
            policy=policy(),
        )

        self.assertEqual(
            benchmark.cases[
                0
            ].language,
            BilingualLanguage.ARABIC,
        )

    def test_latency_is_recorded(self):
        benchmark = evaluate_controlled_web(
            StaticRetriever({}),
            cases=(
                gold(),
            ),
            policy=policy(),
        )

        self.assertGreaterEqual(
            benchmark.cases[
                0
            ].latency_ms,
            0.0,
        )

    def test_duplicate_case_ids_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_controlled_web(
                StaticRetriever({}),
                cases=(
                    gold(
                        case_id="A"
                    ),
                    gold(
                        case_id="A"
                    ),
                ),
                policy=policy(),
            )

    def test_empty_cases_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_controlled_web(
                StaticRetriever({}),
                cases=(),
                policy=policy(),
            )

    def test_wrong_result_type_is_rejected(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    object(),
                )
            }
        )

        with self.assertRaises(
            TypeError
        ):
            evaluate_controlled_web(
                retriever,
                cases=(
                    gold(),
                ),
                policy=policy(),
            )

    def test_noncontiguous_ranks_are_rejected(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy"
                        ),
                        rank=2,
                    ),
                )
            }
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_controlled_web(
                retriever,
                cases=(
                    gold(),
                ),
                policy=policy(),
            )

    def test_duplicate_result_urls_are_rejected(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy#one"
                        ),
                        rank=1,
                    ),
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy#two"
                        ),
                        rank=2,
                    ),
                )
            }
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_controlled_web(
                retriever,
                cases=(
                    gold(),
                ),
                policy=policy(),
            )

    def test_more_than_top_k_is_rejected(self):
        retriever = StaticRetriever(
            {
                "annual leave policy": (
                    result(
                        url=(
                            "https://example.gov.jo/a"
                        ),
                        rank=1,
                    ),
                    result(
                        url=(
                            "https://example.gov.jo/b"
                        ),
                        rank=2,
                    ),
                )
            }
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_controlled_web(
                retriever,
                cases=(
                    gold(
                        top_k=1
                    ),
                ),
                policy=policy(),
            )

    def test_serialization_contains_no_query_or_evidence_text(self):
        raw_query = (
            "annual leave policy"
        )

        raw_text = (
            "Unique synthetic evidence."
        )

        retriever = StaticRetriever(
            {
                raw_query: (
                    result(
                        url=(
                            "https://example.gov.jo/"
                            "policy"
                        ),
                        rank=1,
                        text=raw_text,
                    ),
                )
            }
        )

        benchmark = evaluate_controlled_web(
            retriever,
            cases=(
                gold(
                    query=raw_query
                ),
            ),
            policy=policy(),
        )

        data = benchmark.to_dict()

        self.assertNotIn(
            raw_query,
            repr(data),
        )

        self.assertNotIn(
            raw_text,
            repr(data),
        )


if __name__ == "__main__":
    unittest.main()
