"""Offline tests for GovBA-GAR hybrid Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import unittest

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    HybridRetriever,
    LexicalRetriever,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    SemanticRetriever,
    SourceLanguage,
    SourceStatus,
)


def make_chunk(
    index: int,
    text: str,
) -> EvidenceChunk:
    return EvidenceChunk(
        document_id="DOC-001",
        chunk_index=index,
        text=text,
        language=SourceLanguage.ENGLISH,
    )


CHUNK_A = make_chunk(
    0,
    "Applicants submit the application form.",
)

CHUNK_B = make_chunk(
    1,
    "Service fees are paid electronically.",
)

CHUNK_C = make_chunk(
    2,
    "Annual statistical reports are published.",
)


def make_result(
    chunk: EvidenceChunk,
    rank: int,
    method: str,
    matched_terms=(),
) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=max(
            0.1,
            1.0 - (rank - 1) * 0.1,
        ),
        rank=rank,
        retrieval_method=method,
        matched_terms=matched_terms,
    )


class FixedRetriever:
    """Deterministic retriever used to isolate fusion behavior."""

    def __init__(self, results):
        self.results = tuple(results)
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)

        return list(
            self.results[
                : query.top_k
            ]
        )


def make_hybrid():
    lexical = FixedRetriever(
        (
            make_result(
                CHUNK_A,
                1,
                "lexical",
                (
                    "application",
                    "form",
                ),
            ),
            make_result(
                CHUNK_B,
                2,
                "lexical",
            ),
        )
    )

    semantic = FixedRetriever(
        (
            make_result(
                CHUNK_C,
                1,
                "semantic",
            ),
            make_result(
                CHUNK_A,
                2,
                "semantic",
            ),
            make_result(
                CHUNK_B,
                3,
                "semantic",
            ),
        )
    )

    hybrid = HybridRetriever(
        retrievers={
            "Lexical": lexical,
            "Semantic": semantic,
        },
        candidate_pool_size=3,
    )

    return hybrid, lexical, semantic


class TestHybridRetriever(unittest.TestCase):

    def test_hybrid_satisfies_retriever_contract(self):
        hybrid, _, _ = make_hybrid()

        self.assertIsInstance(
            hybrid,
            Retriever,
        )

    def test_shared_evidence_ranks_first(self):
        hybrid, _, _ = make_hybrid()

        results = hybrid.retrieve(
            RetrievalQuery(
                text="application requirements",
                top_k=3,
            )
        )

        self.assertEqual(
            results[0].chunk.chunk_id,
            CHUNK_A.chunk_id,
        )

    def test_method_and_scores_are_exposed(self):
        hybrid, _, _ = make_hybrid()

        result = hybrid.retrieve(
            RetrievalQuery(
                text="application",
                top_k=1,
            )
        )[0]

        self.assertEqual(
            result.retrieval_method,
            "hybrid-rrf-v1",
        )

        self.assertGreater(
            result.raw_score,
            0.0,
        )

        self.assertGreaterEqual(
            result.score,
            0.0,
        )

        self.assertLessEqual(
            result.score,
            1.0,
        )

    def test_top_k_is_enforced(self):
        hybrid, _, _ = make_hybrid()

        results = hybrid.retrieve(
            RetrievalQuery(
                text="test",
                top_k=2,
            )
        )

        self.assertEqual(
            len(results),
            2,
        )

    def test_candidate_pool_expands_component_query(self):
        hybrid, lexical, semantic = (
            make_hybrid()
        )

        hybrid.retrieve(
            RetrievalQuery(
                text="test",
                top_k=1,
            )
        )

        self.assertEqual(
            lexical.queries[-1].top_k,
            3,
        )

        self.assertEqual(
            semantic.queries[-1].top_k,
            3,
        )

    def test_filters_are_propagated(self):
        hybrid, lexical, semantic = (
            make_hybrid()
        )

        filters = RetrievalFilters(
            statuses=(
                SourceStatus.CURRENT,
            )
        )

        hybrid.retrieve(
            RetrievalQuery(
                text="test",
                top_k=1,
                filters=filters,
            )
        )

        self.assertEqual(
            lexical.queries[-1].filters,
            filters,
        )

        self.assertEqual(
            semantic.queries[-1].filters,
            filters,
        )

    def test_matched_terms_are_preserved(self):
        hybrid, _, _ = make_hybrid()

        results = hybrid.retrieve(
            RetrievalQuery(
                text="application form",
                top_k=3,
            )
        )

        application_result = next(
            item
            for item in results
            if item.chunk.chunk_id
            == CHUNK_A.chunk_id
        )

        self.assertIn(
            "application",
            application_result.matched_terms,
        )

        self.assertIn(
            "form",
            application_result.matched_terms,
        )

    def test_results_are_deterministic(self):
        hybrid, _, _ = make_hybrid()

        query = RetrievalQuery(
            text="application",
            top_k=3,
        )

        first = hybrid.retrieve(query)
        second = hybrid.retrieve(query)

        self.assertEqual(
            [
                item.chunk.chunk_id
                for item in first
            ],
            [
                item.chunk.chunk_id
                for item in second
            ],
        )

        self.assertEqual(
            [
                item.score
                for item in first
            ],
            [
                item.score
                for item in second
            ],
        )

    def test_custom_weights_can_change_ranking(self):
        lexical = FixedRetriever(
            (
                make_result(
                    CHUNK_A,
                    1,
                    "lexical",
                ),
                make_result(
                    CHUNK_B,
                    2,
                    "lexical",
                ),
            )
        )

        semantic = FixedRetriever(
            (
                make_result(
                    CHUNK_B,
                    1,
                    "semantic",
                ),
                make_result(
                    CHUNK_A,
                    2,
                    "semantic",
                ),
            )
        )

        hybrid = HybridRetriever(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            weights={
                "Lexical": 0.1,
                "Semantic": 0.9,
            },
            candidate_pool_size=2,
        )

        results = hybrid.retrieve(
            RetrievalQuery(
                text="test",
                top_k=2,
            )
        )

        self.assertEqual(
            results[0].chunk.chunk_id,
            CHUNK_B.chunk_id,
        )

    def test_requires_at_least_two_retrievers(self):
        lexical = FixedRetriever(())

        with self.assertRaises(ValueError):
            HybridRetriever(
                retrievers={
                    "Lexical": lexical,
                }
            )

    def test_invalid_retriever_is_rejected(self):
        lexical = FixedRetriever(())

        with self.assertRaises(TypeError):
            HybridRetriever(
                retrievers={
                    "Lexical": lexical,
                    "Invalid": object(),
                }
            )

    def test_invalid_weights_are_rejected(self):
        first = FixedRetriever(())
        second = FixedRetriever(())

        invalid_weight_sets = (
            {
                "First": 1.0,
            },
            {
                "First": 0.0,
                "Second": 1.0,
            },
            {
                "First": math.inf,
                "Second": 1.0,
            },
        )

        for weights in invalid_weight_sets:
            with self.subTest(
                weights=weights
            ):
                with self.assertRaises(
                    (
                        ValueError,
                        TypeError,
                    )
                ):
                    HybridRetriever(
                        retrievers={
                            "First": first,
                            "Second": second,
                        },
                        weights=weights,
                    )

    def test_invalid_rrf_k_is_rejected(self):
        first = FixedRetriever(())
        second = FixedRetriever(())

        with self.assertRaises(ValueError):
            HybridRetriever(
                retrievers={
                    "First": first,
                    "Second": second,
                },
                rrf_k=0,
            )

    def test_invalid_candidate_pool_is_rejected(self):
        first = FixedRetriever(())
        second = FixedRetriever(())

        with self.assertRaises(ValueError):
            HybridRetriever(
                retrievers={
                    "First": first,
                    "Second": second,
                },
                candidate_pool_size=101,
            )

    def test_real_lexical_and_semantic_retrievers_integrate(self):
        source = AuthoritativeSource(
            document_id="DOC-REAL",
            title="Service Procedure",
            issuing_authority=(
                "Digital Government Authority"
            ),
            document_type=(
                DocumentType.PROCEDURE
            ),
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.CURRENT,
        )

        chunks = (
            EvidenceChunk(
                document_id="DOC-REAL",
                chunk_index=0,
                text=(
                    "Applicants submit the "
                    "application form."
                ),
                language=(
                    SourceLanguage.ENGLISH
                ),
            ),
            EvidenceChunk(
                document_id="DOC-REAL",
                chunk_index=1,
                text=(
                    "Service fees are paid "
                    "electronically."
                ),
                language=(
                    SourceLanguage.ENGLISH
                ),
            ),
        )

        class FakeEmbeddingProvider:
            def embed_documents(
                self,
                texts,
            ):
                return (
                    (
                        1.0,
                        0.0,
                    ),
                    (
                        0.0,
                        1.0,
                    ),
                )

            def embed_query(
                self,
                text,
            ):
                return (
                    1.0,
                    0.0,
                )

        lexical = LexicalRetriever(
            sources=(source,),
            chunks=chunks,
        )

        semantic = SemanticRetriever(
            sources=(source,),
            chunks=chunks,
            embedding_provider=(
                FakeEmbeddingProvider()
            ),
        )

        hybrid = HybridRetriever(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            candidate_pool_size=2,
        )

        results = hybrid.retrieve(
            RetrievalQuery(
                text="request paperwork",
                top_k=2,
            )
        )

        self.assertTrue(results)

        self.assertEqual(
            results[0].chunk.chunk_id,
            chunks[0].chunk_id,
        )


if __name__ == "__main__":
    unittest.main()
