"""Offline tests for GovBA-GAR bilingual retrieval evaluation."""

from __future__ import annotations

import unittest

from govba.rag.bilingual_evaluation import (
    BILINGUAL_EVALUATION_VERSION,
    BilingualGoldCase,
    compare_bilingual_retrieval_modes,
    evaluate_bilingual_retrieval,
)
from govba.rag.evidence import EvidenceChunk
from govba.rag.language import (
    BilingualLanguage,
)
from govba.rag.models import (
    SourceLanguage,
)
from govba.rag.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


def make_chunk(
    document_id,
    language,
    *,
    index,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=index,
        text=f"Evidence {document_id}",
        language=language,
    )


AR_CHUNK = make_chunk(
    "DOC-AR",
    SourceLanguage.ARABIC,
    index=0,
)

EN_CHUNK = make_chunk(
    "DOC-EN",
    SourceLanguage.ENGLISH,
    index=1,
)


class LanguageAwareStubRetriever:
    """Small deterministic filter-aware retriever."""

    def retrieve(
        self,
        query: RetrievalQuery,
    ):
        results = []

        languages = set(
            query.filters.languages
        )

        if (
            SourceLanguage.ARABIC
            in languages
            and "arabic-target"
            in query.text
        ):
            results.append(
                RetrievalResult(
                    chunk=AR_CHUNK,
                    score=0.95,
                    raw_score=0.95,
                    rank=1,
                    retrieval_method="stub-v1",
                )
            )

        if (
            SourceLanguage.ENGLISH
            in languages
            and (
                "english-target"
                in query.text
                or "cross-target"
                in query.text
            )
        ):
            results.append(
                RetrievalResult(
                    chunk=EN_CHUNK,
                    score=0.90,
                    raw_score=0.90,
                    rank=1,
                    retrieval_method="stub-v1",
                )
            )

        return results[
            : query.top_k
        ]


def paired_cases():
    return (
        BilingualGoldCase(
            case_id="PAIR-1-AR",
            pair_id="PAIR-1",
            query_text="arabic-target",
            query_language=(
                BilingualLanguage.ARABIC
            ),
            relevant_chunk_ids=(
                AR_CHUNK.chunk_id,
            ),
            top_k=3,
        ),
        BilingualGoldCase(
            case_id="PAIR-1-EN",
            pair_id="PAIR-1",
            query_text="english-target",
            query_language=(
                BilingualLanguage.ENGLISH
            ),
            relevant_chunk_ids=(
                EN_CHUNK.chunk_id,
            ),
            top_k=3,
        ),
    )


class TestBilingualEvaluation(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            BILINGUAL_EVALUATION_VERSION,
            "govba-bilingual-evaluation-v1",
        )

    def test_gold_case_is_created(self):
        case = paired_cases()[0]

        self.assertEqual(
            case.pair_id,
            "PAIR-1",
        )

        self.assertEqual(
            case.query_language,
            BilingualLanguage.ARABIC,
        )

    def test_blank_case_id_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            BilingualGoldCase(
                case_id=" ",
                pair_id="PAIR",
                query_text="query",
                query_language=(
                    BilingualLanguage.ARABIC
                ),
                relevant_chunk_ids=(
                    "chunk",
                ),
            )

    def test_relevant_ids_are_required(self):
        with self.assertRaises(
            ValueError
        ):
            BilingualGoldCase(
                case_id="C1",
                pair_id="PAIR",
                query_text="query",
                query_language=(
                    BilingualLanguage.ARABIC
                ),
                relevant_chunk_ids=(),
            )

    def test_duplicate_relevant_ids_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            BilingualGoldCase(
                case_id="C1",
                pair_id="PAIR",
                query_text="query",
                query_language=(
                    BilingualLanguage.ARABIC
                ),
                relevant_chunk_ids=(
                    "A",
                    "A",
                ),
            )

    def test_benchmark_requires_both_languages(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_bilingual_retrieval(
                LanguageAwareStubRetriever(),
                (
                    paired_cases()[0],
                ),
                allow_cross_lingual=False,
            )

    def test_pair_requires_arabic_and_english(self):
        cases = (
            BilingualGoldCase(
                case_id="A1",
                pair_id="PAIR",
                query_text="arabic-target",
                query_language=(
                    BilingualLanguage.ARABIC
                ),
                relevant_chunk_ids=(
                    AR_CHUNK.chunk_id,
                ),
            ),
            BilingualGoldCase(
                case_id="A2",
                pair_id="PAIR",
                query_text="arabic-target",
                query_language=(
                    BilingualLanguage.ARABIC
                ),
                relevant_chunk_ids=(
                    AR_CHUNK.chunk_id,
                ),
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_bilingual_retrieval(
                LanguageAwareStubRetriever(),
                cases,
                allow_cross_lingual=False,
            )

    def test_monolingual_benchmark_hits_both_cases(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.arabic.hit_rate,
            1.0,
        )

        self.assertEqual(
            result.english.hit_rate,
            1.0,
        )

    def test_recall_is_computed(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.arabic.mean_recall,
            1.0,
        )

        self.assertEqual(
            result.english.mean_recall,
            1.0,
        )

    def test_mrr_is_computed(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.arabic.mrr,
            1.0,
        )

        self.assertEqual(
            result.english.mrr,
            1.0,
        )

    def test_zero_result_rate_is_computed(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.arabic.zero_result_rate,
            0.0,
        )

    def test_perfect_languages_have_zero_parity_gap(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.hit_rate_gap,
            0.0,
        )

        self.assertEqual(
            result.recall_gap,
            0.0,
        )

        self.assertEqual(
            result.mrr_gap,
            0.0,
        )

    def test_case_results_do_not_store_query_text(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        data = result.to_dict()

        self.assertNotIn(
            "arabic-target",
            repr(data),
        )

        self.assertNotIn(
            "english-target",
            repr(data),
        )

    def test_cross_lingual_flag_is_recorded(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=True,
        )

        self.assertTrue(
            result.cross_lingual
        )

        self.assertTrue(
            all(
                case.cross_lingual
                for case
                in result.cases
            )
        )

    def test_mode_comparison_uses_same_cases(self):
        comparison = (
            compare_bilingual_retrieval_modes(
                LanguageAwareStubRetriever(),
                paired_cases(),
            )
        )

        self.assertFalse(
            comparison
            .monolingual
            .cross_lingual
        )

        self.assertTrue(
            comparison
            .cross_lingual
            .cross_lingual
        )

    def test_mode_comparison_serializes(self):
        comparison = (
            compare_bilingual_retrieval_modes(
                LanguageAwareStubRetriever(),
                paired_cases(),
            )
        )

        data = comparison.to_dict()

        self.assertIn(
            "monolingual",
            data,
        )

        self.assertIn(
            "cross_lingual",
            data,
        )

        self.assertIn(
            "parity_gap_delta",
            data,
        )

    def test_latency_is_non_negative(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertTrue(
            all(
                case.latency_ms >= 0
                for case
                in result.cases
            )
        )

    def test_case_order_is_preserved(self):
        cases = paired_cases()

        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            cases,
            allow_cross_lingual=False,
        )

        self.assertEqual(
            tuple(
                case.case_id
                for case
                in result.cases
            ),
            tuple(
                case.case_id
                for case
                in cases
            ),
        )

    def test_duplicate_case_ids_are_rejected(self):
        first, second = paired_cases()

        duplicate = BilingualGoldCase(
            case_id=first.case_id,
            pair_id=second.pair_id,
            query_text=second.query_text,
            query_language=(
                second.query_language
            ),
            relevant_chunk_ids=(
                second.relevant_chunk_ids
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_bilingual_retrieval(
                LanguageAwareStubRetriever(),
                (
                    first,
                    duplicate,
                ),
                allow_cross_lingual=False,
            )

    def test_invalid_cross_lingual_flag_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            evaluate_bilingual_retrieval(
                LanguageAwareStubRetriever(),
                paired_cases(),
                allow_cross_lingual=1,
            )

    def test_maximum_quality_gap_uses_largest_metric_gap(self):
        result = evaluate_bilingual_retrieval(
            LanguageAwareStubRetriever(),
            paired_cases(),
            allow_cross_lingual=False,
        )

        self.assertEqual(
            result.maximum_quality_gap,
            max(
                result.hit_rate_gap,
                result.recall_gap,
                result.mrr_gap,
            ),
        )


if __name__ == "__main__":
    unittest.main()
