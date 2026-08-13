"""Offline tests for GovBA-GAR bilingual parity safeguards."""

from __future__ import annotations

import unittest

from govba.rag.bilingual_evaluation import (
    BilingualBenchmarkResult,
    BilingualLanguageMetrics,
)
from govba.rag.language import (
    BilingualLanguage,
)
from govba.rag.language_parity import (
    LANGUAGE_PARITY_VERSION,
    LanguageParityDecision,
    LanguageParityIssueCode,
    LanguageParityPolicy,
    assess_language_parity,
)


def metrics(
    language,
    *,
    hit=0.90,
    recall=0.90,
    mrr=0.90,
    zero=0.05,
):
    return BilingualLanguageMetrics(
        language=language,
        case_count=10,
        hit_rate=hit,
        mean_recall=recall,
        mrr=mrr,
        zero_result_rate=zero,
        mean_latency_ms=1.0,
    )


def benchmark(
    *,
    ar=None,
    en=None,
    cross_lingual=True,
):
    return BilingualBenchmarkResult(
        cases=(),
        arabic=(
            ar
            or metrics(
                BilingualLanguage.ARABIC
            )
        ),
        english=(
            en
            or metrics(
                BilingualLanguage.ENGLISH
            )
        ),
        cross_lingual=cross_lingual,
    )


def policy():
    # Synthetic test thresholds only.
    # Production values will come from benchmark evidence.
    return LanguageParityPolicy(
        min_hit_rate=0.80,
        min_mean_recall=0.80,
        min_mrr=0.80,
        max_zero_result_rate=0.20,
        max_hit_rate_gap=0.10,
        max_recall_gap=0.10,
        max_mrr_gap=0.10,
        max_zero_result_gap=0.10,
        require_cross_lingual=True,
    )


class TestLanguageParitySafeguard(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            LANGUAGE_PARITY_VERSION,
            "govba-language-parity-v1",
        )

    def test_balanced_good_benchmark_passes(self):
        assessment = (
            assess_language_parity(
                benchmark(),
                policy(),
            )
        )

        self.assertEqual(
            assessment.decision,
            LanguageParityDecision.PASS,
        )

        self.assertTrue(
            assessment.passed
        )

    def test_low_arabic_hit_rate_blocks(self):
        assessment = (
            assess_language_parity(
                benchmark(
                    ar=metrics(
                        BilingualLanguage.ARABIC,
                        hit=0.70,
                    )
                ),
                policy(),
            )
        )

        self.assertIn(
            LanguageParityIssueCode
            .ARABIC_HIT_RATE_LOW,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_low_english_recall_blocks(self):
        assessment = (
            assess_language_parity(
                benchmark(
                    en=metrics(
                        BilingualLanguage.ENGLISH,
                        recall=0.70,
                    )
                ),
                policy(),
            )
        )

        self.assertIn(
            LanguageParityIssueCode
            .ENGLISH_RECALL_LOW,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_low_arabic_mrr_blocks(self):
        assessment = (
            assess_language_parity(
                benchmark(
                    ar=metrics(
                        BilingualLanguage.ARABIC,
                        mrr=0.70,
                    )
                ),
                policy(),
            )
        )

        self.assertIn(
            LanguageParityIssueCode
            .ARABIC_MRR_LOW,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_high_zero_result_rate_blocks(self):
        assessment = (
            assess_language_parity(
                benchmark(
                    en=metrics(
                        BilingualLanguage.ENGLISH,
                        zero=0.30,
                    )
                ),
                policy(),
            )
        )

        self.assertIn(
            LanguageParityIssueCode
            .ENGLISH_ZERO_RESULT_HIGH,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_hit_rate_gap_blocks(self):
        assessment = (
            assess_language_parity(
                benchmark(
                    ar=metrics(
                        BilingualLanguage.ARABIC,
                        hit=0.90,
                    ),
                    en=metrics(
                        BilingualLanguage.ENGLISH,
                        hit=0.70,
                    ),
                ),
                policy(),
            )
        )

        self.assertIn(
            LanguageParityIssueCode
            .HIT_RATE_GAP,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_recall_gap_blocks(self):
        assessment = assess_language_parity(
            benchmark(
                ar=metrics(
                    BilingualLanguage.ARABIC,
                    recall=0.95,
                ),
                en=metrics(
                    BilingualLanguage.ENGLISH,
                    recall=0.80,
                ),
            ),
            policy(),
        )

        self.assertIn(
            LanguageParityIssueCode
            .RECALL_GAP,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_mrr_gap_blocks(self):
        assessment = assess_language_parity(
            benchmark(
                ar=metrics(
                    BilingualLanguage.ARABIC,
                    mrr=0.95,
                ),
                en=metrics(
                    BilingualLanguage.ENGLISH,
                    mrr=0.80,
                ),
            ),
            policy(),
        )

        self.assertIn(
            LanguageParityIssueCode
            .MRR_GAP,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_zero_result_gap_blocks(self):
        assessment = assess_language_parity(
            benchmark(
                ar=metrics(
                    BilingualLanguage.ARABIC,
                    zero=0.05,
                ),
                en=metrics(
                    BilingualLanguage.ENGLISH,
                    zero=0.20,
                ),
            ),
            policy(),
        )

        self.assertIn(
            LanguageParityIssueCode
            .ZERO_RESULT_GAP,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_wrong_mode_blocks_when_required(self):
        assessment = assess_language_parity(
            benchmark(
                cross_lingual=False
            ),
            policy(),
        )

        self.assertIn(
            LanguageParityIssueCode
            .WRONG_RETRIEVAL_MODE,
            tuple(
                issue.code
                for issue
                in assessment.issues
            ),
        )

    def test_wrong_mode_can_be_allowed_by_policy(self):
        custom = LanguageParityPolicy(
            min_hit_rate=0.80,
            min_mean_recall=0.80,
            min_mrr=0.80,
            max_zero_result_rate=0.20,
            max_hit_rate_gap=0.10,
            max_recall_gap=0.10,
            max_mrr_gap=0.10,
            max_zero_result_gap=0.10,
            require_cross_lingual=False,
        )

        assessment = assess_language_parity(
            benchmark(
                cross_lingual=False
            ),
            custom,
        )

        self.assertTrue(
            assessment.passed
        )

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            LanguageParityPolicy(
                min_hit_rate=1.1,
                min_mean_recall=0.8,
                min_mrr=0.8,
                max_zero_result_rate=0.2,
                max_hit_rate_gap=0.1,
                max_recall_gap=0.1,
                max_mrr_gap=0.1,
                max_zero_result_gap=0.1,
            )

    def test_policy_id_is_deterministic(self):
        self.assertEqual(
            policy().policy_id,
            policy().policy_id,
        )

        self.assertEqual(
            len(
                policy().policy_id
            ),
            64,
        )

    def test_block_decision_is_fail_safe(self):
        assessment = assess_language_parity(
            benchmark(
                ar=metrics(
                    BilingualLanguage.ARABIC,
                    hit=0.50,
                )
            ),
            policy(),
        )

        self.assertEqual(
            assessment.decision,
            LanguageParityDecision.BLOCK,
        )

        self.assertTrue(
            assessment.should_block
        )

    def test_serialization_contains_no_queries(self):
        assessment = assess_language_parity(
            benchmark(),
            policy(),
        )

        data = assessment.to_dict()

        self.assertNotIn(
            "query_text",
            repr(
                data
            ),
        )

        self.assertEqual(
            data[
                "decision"
            ],
            "pass",
        )


if __name__ == "__main__":
    unittest.main()
