"""Offline tests for controlled GovBA-GAR retriever comparison."""

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
    SemanticRetriever,
    SourceLanguage,
    SourceStatus,
    compare_retrievers,
)


APPLICATION_TEXT = (
    "Applicants must submit the required application form."
)

PAYMENT_TEXT = (
    "Service fees are paid electronically."
)

REPORT_TEXT = (
    "The authority publishes annual statistical reports."
)


class FakeEmbeddingProvider:
    """Deterministic semantic vectors for comparison tests."""

    def __init__(self):
        self.document_vectors = {
            APPLICATION_TEXT: (
                1.0,
                0.0,
                0.0,
            ),
            PAYMENT_TEXT: (
                0.0,
                1.0,
                0.0,
            ),
            REPORT_TEXT: (
                0.0,
                0.0,
                1.0,
            ),
        }

        self.query_vectors = {
            "application form": (
                1.0,
                0.0,
                0.0,
            ),
            "request paperwork": (
                1.0,
                0.0,
                0.0,
            ),
            "service fees": (
                0.0,
                1.0,
                0.0,
            ),
        }

    def embed_documents(self, texts):
        return [
            self.document_vectors[text]
            for text in texts
        ]

    def embed_query(self, text):
        return self.query_vectors[text]


def make_comparison_fixture():
    source = AuthoritativeSource(
        document_id="DOC-SERVICE",
        title="Government Service Procedure",
        issuing_authority="Digital Government Authority",
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        status=SourceStatus.CURRENT,
    )

    chunks = (
        EvidenceChunk(
            document_id="DOC-SERVICE",
            chunk_index=0,
            text=APPLICATION_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
        EvidenceChunk(
            document_id="DOC-SERVICE",
            chunk_index=1,
            text=PAYMENT_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
        EvidenceChunk(
            document_id="DOC-SERVICE",
            chunk_index=2,
            text=REPORT_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
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

    cases = (
        GoldRetrievalCase(
            case_id="EXACT-APPLICATION",
            query=RetrievalQuery(
                text="application form",
                top_k=3,
            ),
            relevant_chunk_ids=(
                chunks[0].chunk_id,
            ),
        ),
        GoldRetrievalCase(
            case_id="SEMANTIC-SYNONYM",
            query=RetrievalQuery(
                text="request paperwork",
                top_k=3,
            ),
            relevant_chunk_ids=(
                chunks[0].chunk_id,
            ),
        ),
        GoldRetrievalCase(
            case_id="PAYMENT",
            query=RetrievalQuery(
                text="service fees",
                top_k=3,
            ),
            relevant_chunk_ids=(
                chunks[1].chunk_id,
            ),
        ),
    )

    return lexical, semantic, cases


class TestRetrieverBenchmarkComparison(unittest.TestCase):

    def test_comparison_contains_named_runs(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            cases=cases,
            cutoffs=(1, 3),
        )

        self.assertEqual(
            len(comparison.runs),
            2,
        )

        self.assertEqual(
            comparison.runs[0].name,
            "Lexical",
        )

        self.assertEqual(
            comparison.runs[1].name,
            "Semantic",
        )

    def test_semantic_outperforms_lexical_on_fixture_mrr(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            cases=cases,
            cutoffs=(1, 3),
        )

        lexical_mrr = (
            comparison
            .get("Lexical")
            .evaluation
            .mean_reciprocal_rank
        )

        semantic_mrr = (
            comparison
            .get("Semantic")
            .evaluation
            .mean_reciprocal_rank
        )

        self.assertAlmostEqual(
            lexical_mrr,
            2.0 / 3.0,
        )

        self.assertEqual(
            semantic_mrr,
            1.0,
        )

        self.assertGreater(
            semantic_mrr,
            lexical_mrr,
        )

    def test_zero_result_rate_is_compared(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            cases=cases,
        )

        lexical_rate = (
            comparison
            .get("Lexical")
            .evaluation
            .zero_result_rate
        )

        semantic_rate = (
            comparison
            .get("Semantic")
            .evaluation
            .zero_result_rate
        )

        self.assertAlmostEqual(
            lexical_rate,
            1.0 / 3.0,
        )

        self.assertEqual(
            semantic_rate,
            0.0,
        )

    def test_case_count_and_cutoffs_are_shared(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            cases=cases,
            cutoffs=(3, 1, 3),
        )

        self.assertEqual(
            comparison.case_count,
            3,
        )

        self.assertEqual(
            comparison.cutoffs,
            (
                1,
                3,
            ),
        )

    def test_get_is_case_insensitive(self):
        lexical, _, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
            },
            cases=cases,
        )

        self.assertEqual(
            comparison.get(
                " lexical "
            ).name,
            "Lexical",
        )

    def test_comparison_serializes_to_json(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        comparison = compare_retrievers(
            retrievers={
                "Lexical": lexical,
                "Semantic": semantic,
            },
            cases=cases,
        )

        data = json.loads(
            comparison.to_json()
        )

        self.assertEqual(
            data["case_count"],
            3,
        )

        self.assertEqual(
            len(data["runs"]),
            2,
        )

    def test_empty_retriever_set_is_rejected(self):
        _, _, cases = (
            make_comparison_fixture()
        )

        with self.assertRaises(ValueError):
            compare_retrievers(
                retrievers={},
                cases=cases,
            )

    def test_empty_case_set_is_rejected(self):
        lexical, _, _ = (
            make_comparison_fixture()
        )

        with self.assertRaises(ValueError):
            compare_retrievers(
                retrievers={
                    "Lexical": lexical,
                },
                cases=(),
            )

    def test_blank_retriever_name_is_rejected(self):
        lexical, _, cases = (
            make_comparison_fixture()
        )

        with self.assertRaises(ValueError):
            compare_retrievers(
                retrievers={
                    "   ": lexical,
                },
                cases=cases,
            )

    def test_duplicate_normalized_names_are_rejected(self):
        lexical, semantic, cases = (
            make_comparison_fixture()
        )

        with self.assertRaises(ValueError):
            compare_retrievers(
                retrievers={
                    "Lexical": lexical,
                    " lexical ": semantic,
                },
                cases=cases,
            )

    def test_invalid_retriever_is_rejected(self):
        _, _, cases = (
            make_comparison_fixture()
        )

        with self.assertRaises(TypeError):
            compare_retrievers(
                retrievers={
                    "Invalid": object(),
                },
                cases=cases,
            )


if __name__ == "__main__":
    unittest.main()
