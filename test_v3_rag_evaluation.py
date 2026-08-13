"""Offline tests for GovBA-GAR retrieval evaluation."""

from __future__ import annotations

import json
import unittest

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    GoldRetrievalCase,
    LexicalRetriever,
    RetrievalQuery,
    RetrievalResult,
    SourceLanguage,
    SourceStatus,
    evaluate_benchmark,
    evaluate_ranked_results,
    normalize_cutoffs,
)


def make_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
) -> EvidenceChunk:
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        language=SourceLanguage.ENGLISH,
    )


CHUNK_A = make_chunk(
    "DOC-A",
    0,
    "Alpha evidence.",
)

CHUNK_B = make_chunk(
    "DOC-B",
    0,
    "Beta evidence.",
)

CHUNK_C = make_chunk(
    "DOC-C",
    0,
    "Gamma evidence.",
)


def make_result(
    chunk: EvidenceChunk,
    rank: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=max(
            0.1,
            1.0 - (rank - 1) * 0.1,
        ),
        rank=rank,
        retrieval_method="test",
    )


class TestGoldRetrievalCase(unittest.TestCase):

    def test_gold_case_normalizes_and_deduplicates_ids(self):
        case = GoldRetrievalCase(
            case_id="  CASE-001  ",
            query=RetrievalQuery(
                text="test"
            ),
            relevant_chunk_ids=(
                CHUNK_A.chunk_id,
                CHUNK_A.chunk_id,
                CHUNK_B.chunk_id,
            ),
        )

        self.assertEqual(
            case.case_id,
            "CASE-001",
        )

        self.assertEqual(
            case.relevant_chunk_ids,
            (
                CHUNK_A.chunk_id,
                CHUNK_B.chunk_id,
            ),
        )

    def test_blank_case_id_is_rejected(self):
        with self.assertRaises(ValueError):
            GoldRetrievalCase(
                case_id=" ",
                query=RetrievalQuery(
                    text="test"
                ),
                relevant_chunk_ids=(
                    CHUNK_A.chunk_id,
                ),
            )

    def test_empty_relevance_is_rejected(self):
        with self.assertRaises(ValueError):
            GoldRetrievalCase(
                case_id="CASE-001",
                query=RetrievalQuery(
                    text="test"
                ),
                relevant_chunk_ids=(),
            )

    def test_invalid_cutoff_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_cutoffs(
                (1, 0, 5)
            )


class TestCaseEvaluation(unittest.TestCase):

    def test_metrics_when_relevant_result_is_rank_two(self):
        case = GoldRetrievalCase(
            case_id="CASE-RANK-2",
            query=RetrievalQuery(
                text="test"
            ),
            relevant_chunk_ids=(
                CHUNK_B.chunk_id,
            ),
        )

        evaluation = evaluate_ranked_results(
            case=case,
            results=(
                make_result(
                    CHUNK_A,
                    1,
                ),
                make_result(
                    CHUNK_B,
                    2,
                ),
                make_result(
                    CHUNK_C,
                    3,
                ),
            ),
            latency_ms=4.0,
            cutoffs=(1, 3),
        )

        self.assertEqual(
            evaluation.first_relevant_rank,
            2,
        )

        self.assertEqual(
            evaluation.reciprocal_rank,
            0.5,
        )

        self.assertEqual(
            evaluation.hit_at_k[1],
            0.0,
        )

        self.assertEqual(
            evaluation.hit_at_k[3],
            1.0,
        )

        self.assertEqual(
            evaluation.recall_at_k[1],
            0.0,
        )

        self.assertEqual(
            evaluation.recall_at_k[3],
            1.0,
        )

    def test_no_relevant_result_returns_zero_rr(self):
        case = GoldRetrievalCase(
            case_id="CASE-MISS",
            query=RetrievalQuery(
                text="test"
            ),
            relevant_chunk_ids=(
                CHUNK_C.chunk_id,
            ),
        )

        evaluation = evaluate_ranked_results(
            case=case,
            results=(
                make_result(
                    CHUNK_A,
                    1,
                ),
                make_result(
                    CHUNK_B,
                    2,
                ),
            ),
        )

        self.assertIsNone(
            evaluation.first_relevant_rank
        )

        self.assertEqual(
            evaluation.reciprocal_rank,
            0.0,
        )

    def test_multiple_relevant_chunks_compute_recall(self):
        case = GoldRetrievalCase(
            case_id="CASE-MULTI",
            query=RetrievalQuery(
                text="test"
            ),
            relevant_chunk_ids=(
                CHUNK_A.chunk_id,
                CHUNK_B.chunk_id,
            ),
        )

        evaluation = evaluate_ranked_results(
            case=case,
            results=(
                make_result(
                    CHUNK_A,
                    1,
                ),
            ),
            cutoffs=(1,),
        )

        self.assertEqual(
            evaluation.hit_at_k[1],
            1.0,
        )

        self.assertEqual(
            evaluation.recall_at_k[1],
            0.5,
        )

    def test_empty_results_are_supported(self):
        case = GoldRetrievalCase(
            case_id="CASE-EMPTY",
            query=RetrievalQuery(
                text="test"
            ),
            relevant_chunk_ids=(
                CHUNK_A.chunk_id,
            ),
        )

        evaluation = evaluate_ranked_results(
            case=case,
            results=(),
            cutoffs=(1, 5),
        )

        self.assertEqual(
            evaluation.retrieved_count,
            0,
        )

        self.assertEqual(
            evaluation.hit_at_k[1],
            0.0,
        )

        self.assertEqual(
            evaluation.recall_at_k[5],
            0.0,
        )


class TestBenchmarkEvaluation(unittest.TestCase):

    def test_benchmark_computes_mrr_and_hit_rate(self):
        class FakeRetriever:
            def retrieve(self, query):
                if query.text == "alpha":
                    return [
                        make_result(
                            CHUNK_A,
                            1,
                        )
                    ]

                return []

        cases = (
            GoldRetrievalCase(
                case_id="CASE-A",
                query=RetrievalQuery(
                    text="alpha"
                ),
                relevant_chunk_ids=(
                    CHUNK_A.chunk_id,
                ),
            ),
            GoldRetrievalCase(
                case_id="CASE-B",
                query=RetrievalQuery(
                    text="beta"
                ),
                relevant_chunk_ids=(
                    CHUNK_B.chunk_id,
                ),
            ),
        )

        benchmark = evaluate_benchmark(
            retriever=FakeRetriever(),
            cases=cases,
            cutoffs=(1, 3),
        )

        self.assertEqual(
            benchmark.case_count,
            2,
        )

        self.assertEqual(
            benchmark.mean_reciprocal_rank,
            0.5,
        )

        self.assertEqual(
            benchmark.hit_rate_at_k[1],
            0.5,
        )

    def test_benchmark_computes_zero_result_rate(self):
        class FakeRetriever:
            def retrieve(self, query):
                if query.text == "found":
                    return [
                        make_result(
                            CHUNK_A,
                            1,
                        )
                    ]

                return []

        cases = (
            GoldRetrievalCase(
                case_id="FOUND",
                query=RetrievalQuery(
                    text="found"
                ),
                relevant_chunk_ids=(
                    CHUNK_A.chunk_id,
                ),
            ),
            GoldRetrievalCase(
                case_id="EMPTY",
                query=RetrievalQuery(
                    text="missing"
                ),
                relevant_chunk_ids=(
                    CHUNK_B.chunk_id,
                ),
            ),
        )

        benchmark = evaluate_benchmark(
            retriever=FakeRetriever(),
            cases=cases,
        )

        self.assertEqual(
            benchmark.zero_result_rate,
            0.5,
        )

        self.assertGreaterEqual(
            benchmark.mean_latency_ms,
            0.0,
        )

    def test_empty_benchmark_is_rejected(self):
        class FakeRetriever:
            def retrieve(self, query):
                return []

        with self.assertRaises(ValueError):
            evaluate_benchmark(
                retriever=FakeRetriever(),
                cases=(),
            )

    def test_duplicate_case_ids_are_rejected(self):
        class FakeRetriever:
            def retrieve(self, query):
                return []

        cases = (
            GoldRetrievalCase(
                case_id="DUPLICATE",
                query=RetrievalQuery(
                    text="one"
                ),
                relevant_chunk_ids=(
                    CHUNK_A.chunk_id,
                ),
            ),
            GoldRetrievalCase(
                case_id="DUPLICATE",
                query=RetrievalQuery(
                    text="two"
                ),
                relevant_chunk_ids=(
                    CHUNK_B.chunk_id,
                ),
            ),
        )

        with self.assertRaises(ValueError):
            evaluate_benchmark(
                retriever=FakeRetriever(),
                cases=cases,
            )

    def test_benchmark_serializes_to_json(self):
        class FakeRetriever:
            def retrieve(self, query):
                return [
                    make_result(
                        CHUNK_A,
                        1,
                    )
                ]

        case = GoldRetrievalCase(
            case_id="JSON",
            query=RetrievalQuery(
                text="alpha"
            ),
            relevant_chunk_ids=(
                CHUNK_A.chunk_id,
            ),
        )

        benchmark = evaluate_benchmark(
            retriever=FakeRetriever(),
            cases=(case,),
        )

        parsed = json.loads(
            benchmark.to_json()
        )

        self.assertEqual(
            parsed["case_count"],
            1,
        )

        self.assertEqual(
            parsed["mean_reciprocal_rank"],
            1.0,
        )

    def test_lexical_retriever_integrates_with_evaluation(self):
        source = AuthoritativeSource(
            document_id="DOC-LIVE",
            title="Application Procedure",
            issuing_authority=(
                "Digital Government Authority"
            ),
            document_type=(
                DocumentType.PROCEDURE
            ),
            language=(
                SourceLanguage.ENGLISH
            ),
            status=SourceStatus.CURRENT,
        )

        chunk = EvidenceChunk(
            document_id="DOC-LIVE",
            chunk_index=0,
            text=(
                "Applicants must submit the "
                "required application form."
            ),
            language=(
                SourceLanguage.ENGLISH
            ),
        )

        retriever = LexicalRetriever(
            sources=(source,),
            chunks=(chunk,),
        )

        case = GoldRetrievalCase(
            case_id="LEXICAL-INTEGRATION",
            query=RetrievalQuery(
                text="required application form",
                top_k=5,
            ),
            relevant_chunk_ids=(
                chunk.chunk_id,
            ),
        )

        benchmark = evaluate_benchmark(
            retriever=retriever,
            cases=(case,),
        )

        self.assertEqual(
            benchmark.mean_reciprocal_rank,
            1.0,
        )

        self.assertEqual(
            benchmark.hit_rate_at_k[1],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
