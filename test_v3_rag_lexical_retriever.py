"""Offline tests for GovBA-GAR deterministic lexical retrieval."""

from __future__ import annotations

import unittest

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    LexicalRetriever,
    RetrievalFilters,
    RetrievalQuery,
    Retriever,
    SourceLanguage,
    SourceStatus,
    tokenize_lexical,
)


def make_sources():
    """Return deterministic authoritative sources for retrieval tests."""

    return [
        AuthoritativeSource(
            document_id="DOC-CURRENT",
            title="Current Service Procedure",
            issuing_authority="Digital Government Authority",
            document_type=DocumentType.PROCEDURE,
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://example.gov.jo/current"
            ),
        ),
        AuthoritativeSource(
            document_id="DOC-SUPERSEDED",
            title="Old Service Procedure",
            issuing_authority="Digital Government Authority",
            document_type=DocumentType.PROCEDURE,
            language=SourceLanguage.ENGLISH,
            status=SourceStatus.SUPERSEDED,
            official_source_url=(
                "https://example.gov.jo/old"
            ),
        ),
        AuthoritativeSource(
            document_id="DOC-AR",
            title="تعليمات الخدمة الرقمية",
            issuing_authority="وزارة رقمية تجريبية",
            document_type=DocumentType.GUIDELINE,
            language=SourceLanguage.ARABIC,
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://example.gov.jo/ar"
            ),
        ),
    ]


def make_chunks():
    """Return deterministic evidence chunks for retrieval tests."""

    return [
        EvidenceChunk(
            document_id="DOC-CURRENT",
            chunk_index=0,
            text=(
                "Applicants must submit the required "
                "application form and identification documents."
            ),
            language=SourceLanguage.ENGLISH,
            page_start=3,
        ),
        EvidenceChunk(
            document_id="DOC-CURRENT",
            chunk_index=1,
            text=(
                "The application fee is paid through the "
                "government electronic payment gateway."
            ),
            language=SourceLanguage.ENGLISH,
            page_start=4,
        ),
        EvidenceChunk(
            document_id="DOC-SUPERSEDED",
            chunk_index=0,
            text=(
                "Applicants must submit an old application "
                "form and identification documents."
            ),
            language=SourceLanguage.ENGLISH,
            page_start=2,
        ),
        EvidenceChunk(
            document_id="DOC-AR",
            chunk_index=0,
            text=(
                "يجب على المتقدم تقديم طلب الخدمة "
                "والوثائق المطلوبة."
            ),
            language=SourceLanguage.ARABIC,
            page_start=5,
        ),
    ]


class TestLexicalRetriever(unittest.TestCase):
    """Tests for deterministic lexical retrieval behavior."""

    def setUp(self):
        self.retriever = LexicalRetriever(
            sources=make_sources(),
            chunks=make_chunks(),
        )

    def test_retriever_satisfies_protocol(self):
        self.assertIsInstance(
            self.retriever,
            Retriever,
        )

    def test_counts_are_available(self):
        self.assertEqual(
            self.retriever.source_count,
            3,
        )
        self.assertEqual(
            self.retriever.chunk_count,
            4,
        )

    def test_relevant_chunk_ranks_first(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="required application documents",
                top_k=3,
            )
        )

        self.assertGreater(
            len(results),
            0,
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
            results[0].rank,
            1,
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
                text="application identification documents",
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
            "DOC-SUPERSEDED",
            document_ids,
        )

    def test_document_filter_is_enforced(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application",
                top_k=5,
                filters=RetrievalFilters(
                    document_ids=(
                        "DOC-SUPERSEDED",
                    )
                ),
            )
        )

        self.assertTrue(results)

        self.assertTrue(
            all(
                result.chunk.document_id
                == "DOC-SUPERSEDED"
                for result in results
            )
        )

    def test_no_overlap_returns_no_results(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="completely unrelated astronomy",
                top_k=5,
            )
        )

        self.assertEqual(
            results,
            [],
        )

    def test_results_are_deterministic(self):
        query = RetrievalQuery(
            text="application documents",
            top_k=5,
        )

        first = self.retriever.retrieve(query)
        second = self.retriever.retrieve(query)

        first_ids = [
            item.chunk.chunk_id
            for item in first
        ]
        second_ids = [
            item.chunk.chunk_id
            for item in second
        ]

        first_scores = [
            item.score
            for item in first
        ]
        second_scores = [
            item.score
            for item in second
        ]

        self.assertEqual(
            first_ids,
            second_ids,
        )
        self.assertEqual(
            first_scores,
            second_scores,
        )

    def test_arabic_unicode_retrieval(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="الوثائق المطلوبة",
                top_k=3,
            )
        )

        self.assertTrue(results)

        self.assertEqual(
            results[0].chunk.document_id,
            "DOC-AR",
        )

    def test_matched_terms_are_exposed(self):
        results = self.retriever.retrieve(
            RetrievalQuery(
                text="application documents",
                top_k=1,
            )
        )

        self.assertIn(
            "application",
            results[0].matched_terms,
        )

        self.assertIn(
            "documents",
            results[0].matched_terms,
        )

    def test_duplicate_source_id_is_rejected(self):
        source = make_sources()[0]

        with self.assertRaises(ValueError):
            LexicalRetriever(
                sources=(
                    source,
                    source,
                ),
                chunks=(),
            )

    def test_unknown_chunk_source_is_rejected(self):
        unknown_chunk = EvidenceChunk(
            document_id="UNKNOWN-DOC",
            chunk_index=0,
            text="Evidence.",
            language=SourceLanguage.ENGLISH,
        )

        with self.assertRaises(ValueError):
            LexicalRetriever(
                sources=make_sources(),
                chunks=(
                    unknown_chunk,
                ),
            )

    def test_unicode_tokenization_supports_arabic(self):
        tokens = tokenize_lexical(
            "طلب الخدمة والوثائق المطلوبة"
        )

        self.assertIn(
            "الخدمة",
            tokens,
        )

        self.assertIn(
            "المطلوبة",
            tokens,
        )


if __name__ == "__main__":
    unittest.main()
