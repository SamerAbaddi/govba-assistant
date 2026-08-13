"""Offline tests for GovBA-GAR semantic retrieval."""

from __future__ import annotations

import unittest

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    RetrievalFilters,
    RetrievalQuery,
    Retriever,
    SemanticRetriever,
    SourceLanguage,
    SourceStatus,
)


APPLICATION_TEXT = (
    "Applicants must submit the required application form."
)

PAYMENT_TEXT = (
    "Service fees are paid through the electronic payment gateway."
)

OLD_APPLICATION_TEXT = (
    "Applicants must submit the previous application form."
)

ARABIC_TEXT = (
    "يجب تقديم طلب الخدمة والوثائق المطلوبة."
)


class FakeEmbeddingProvider:
    """Deterministic embedding provider for offline testing."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

        self.vectors = {
            APPLICATION_TEXT: (
                1.0,
                0.0,
            ),
            PAYMENT_TEXT: (
                0.7,
                0.7,
            ),
            OLD_APPLICATION_TEXT: (
                0.95,
                0.05,
            ),
            ARABIC_TEXT: (
                0.0,
                1.0,
            ),
        }

    def embed_documents(self, texts):
        self.document_calls += 1

        return [
            self.vectors[text]
            for text in texts
        ]

    def embed_query(self, text):
        self.query_calls += 1

        if text == "arabic query":
            return (
                0.0,
                1.0,
            )

        return (
            1.0,
            0.0,
        )


def make_sources():
    return (
        AuthoritativeSource(
            document_id="DOC-CURRENT",
            title="Current Procedure",
            issuing_authority="Digital Government Authority",
            document_type=DocumentType.PROCEDURE,
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.CURRENT,
        ),
        AuthoritativeSource(
            document_id="DOC-OLD",
            title="Old Procedure",
            issuing_authority="Digital Government Authority",
            document_type=DocumentType.PROCEDURE,
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.SUPERSEDED,
        ),
        AuthoritativeSource(
            document_id="DOC-AR",
            title="تعليمات الخدمة",
            issuing_authority="وزارة رقمية تجريبية",
            document_type=DocumentType.GUIDELINE,
            language=SourceLanguage.ARABIC,
            status=SourceStatus.CURRENT,
        ),
    )


def make_chunks():
    return (
        EvidenceChunk(
            document_id="DOC-CURRENT",
            chunk_index=0,
            text=APPLICATION_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
        EvidenceChunk(
            document_id="DOC-CURRENT",
            chunk_index=1,
            text=PAYMENT_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
        EvidenceChunk(
            document_id="DOC-OLD",
            chunk_index=0,
            text=OLD_APPLICATION_TEXT,
            language=SourceLanguage.ENGLISH,
        ),
        EvidenceChunk(
            document_id="DOC-AR",
            chunk_index=0,
            text=ARABIC_TEXT,
            language=SourceLanguage.ARABIC,
        ),
    )


class TestSemanticRetriever(unittest.TestCase):

    def setUp(self):
        self.provider = FakeEmbeddingProvider()

        self.retriever = SemanticRetriever(
            sources=make_sources(),
            chunks=make_chunks(),
            embedding_provider=self.provider,
        )

    def test_retriever_satisfies_contract(self):
        self.assertIsInstance(
            self.retriever,
            Retriever,
        )

    def test_counts_and_dimension_are_available(self):
        self.assertEqual(
            self.retriever.source_count,
            3,
        )

        self.assertEqual(
            self.retriever.chunk_count,
            4,
        )

        self.assertEqual(
            self.retriever.embedding_dimension,
            2,
        )

    def test_document_embeddings_are_created_once(self):
        self.assertEqual(
            self.provider.document_calls,
            1,
        )

        self.retriever.retrieve(
            RetrievalQuery(
                text="application requirements"
            )
        )

        self.retriever.retrieve(
            RetrievalQuery(
                text="application procedure"
            )
        )

        self.assertEqual(
            self.provider.document_calls,
            1,
        )

        self.assertEqual(
            self.provider.query_calls,
            2,
        )

    def test_semantically_closest_chunk_ranks_first(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application requirements",
                top_k=4,
            )
        )

        self.assertEqual(
            results[0].chunk.document_id,
            "DOC-CURRENT",
        )

        self.assertEqual(
            results[0].chunk.chunk_index,
            0,
        )

        self.assertEqual(
            results[0].raw_score,
            1.0,
        )

        self.assertEqual(
            results[0].score,
            1.0,
        )

        self.assertEqual(
            results[0].retrieval_method,
            "semantic-v1",
        )

    def test_top_k_is_enforced(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application",
                top_k=1,
            )
        )

        self.assertEqual(
            len(results),
            1,
        )

    def test_status_filter_excludes_superseded_source(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application",
                top_k=5,
                filters=RetrievalFilters(
                    statuses=(
                        SourceStatus.CURRENT,
                    )
                ),
            )
        )

        document_ids = {
            result.chunk.document_id
            for result in results
        }

        self.assertNotIn(
            "DOC-OLD",
            document_ids,
        )

    def test_document_filter_is_enforced(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application",
                top_k=5,
                filters=RetrievalFilters(
                    document_ids=(
                        "DOC-OLD",
                    )
                ),
            )
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].chunk.document_id,
            "DOC-OLD",
        )

    def test_semantic_retrieval_can_rank_arabic_vector(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="arabic query",
                top_k=1,
            )
        )

        self.assertEqual(
            results[0].chunk.document_id,
            "DOC-AR",
        )

    def test_results_are_deterministic(self):
        query = RetrievalQuery(
            text="application requirements",
            top_k=4,
        )

        first = self.retriever.retrieve(
            query
        )

        second = self.retriever.retrieve(
            query
        )

        self.assertEqual(
            [
                result.chunk.chunk_id
                for result in first
            ],
            [
                result.chunk.chunk_id
                for result in second
            ],
        )

        self.assertEqual(
            [
                result.score
                for result in first
            ],
            [
                result.score
                for result in second
            ],
        )

    def test_duplicate_source_id_is_rejected(self):
        source = make_sources()[0]

        with self.assertRaises(ValueError):
            SemanticRetriever(
                sources=(
                    source,
                    source,
                ),
                chunks=(),
                embedding_provider=(
                    FakeEmbeddingProvider()
                ),
            )

    def test_unknown_chunk_source_is_rejected(self):
        unknown_chunk = EvidenceChunk(
            document_id="UNKNOWN",
            chunk_index=0,
            text="Unknown evidence.",
            language=SourceLanguage.ENGLISH,
        )

        with self.assertRaises(ValueError):
            SemanticRetriever(
                sources=make_sources(),
                chunks=(
                    unknown_chunk,
                ),
                embedding_provider=(
                    FakeEmbeddingProvider()
                ),
            )

    def test_wrong_document_vector_count_is_rejected(self):
        class BadProvider(FakeEmbeddingProvider):
            def embed_documents(self, texts):
                return [
                    (
                        1.0,
                        0.0,
                    )
                ]

        with self.assertRaises(ValueError):
            SemanticRetriever(
                sources=make_sources(),
                chunks=make_chunks(),
                embedding_provider=BadProvider(),
            )

    def test_mixed_document_dimensions_are_rejected(self):
        class BadProvider(FakeEmbeddingProvider):
            def embed_documents(self, texts):
                vectors = list(
                    super().embed_documents(
                        texts
                    )
                )

                vectors[1] = (
                    1.0,
                    0.0,
                    0.0,
                )

                return vectors

        with self.assertRaises(ValueError):
            SemanticRetriever(
                sources=make_sources(),
                chunks=make_chunks(),
                embedding_provider=BadProvider(),
            )

    def test_zero_document_vector_is_rejected(self):
        class BadProvider(FakeEmbeddingProvider):
            def embed_documents(self, texts):
                vectors = list(
                    super().embed_documents(
                        texts
                    )
                )

                vectors[0] = (
                    0.0,
                    0.0,
                )

                return vectors

        with self.assertRaises(ValueError):
            SemanticRetriever(
                sources=make_sources(),
                chunks=make_chunks(),
                embedding_provider=BadProvider(),
            )

    def test_wrong_query_dimension_is_rejected(self):
        class BadProvider(FakeEmbeddingProvider):
            def embed_query(self, text):
                return (
                    1.0,
                    0.0,
                    0.0,
                )

        retriever = SemanticRetriever(
            sources=make_sources(),
            chunks=make_chunks(),
            embedding_provider=BadProvider(),
        )

        with self.assertRaises(ValueError):
            retriever.retrieve(
                RetrievalQuery(
                    text="application"
                )
            )

    def test_zero_query_vector_is_rejected(self):
        class BadProvider(FakeEmbeddingProvider):
            def embed_query(self, text):
                return (
                    0.0,
                    0.0,
                )

        retriever = SemanticRetriever(
            sources=make_sources(),
            chunks=make_chunks(),
            embedding_provider=BadProvider(),
        )

        with self.assertRaises(ValueError):
            retriever.retrieve(
                RetrievalQuery(
                    text="application"
                )
            )


if __name__ == "__main__":
    unittest.main()
