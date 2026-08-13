"""Offline tests for the GovBA-GAR retrieval contract."""

from __future__ import annotations

import json
import unittest

from govba.rag import (
    AuthoritativeSource,
    DocumentType,
    EvidenceChunk,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    SourceLanguage,
    SourceStatus,
)


def make_source(
    *,
    status: SourceStatus = SourceStatus.CURRENT,
    official_source_url: str = "https://example.gov.jo/doc",
) -> AuthoritativeSource:
    """Create a deterministic source for retrieval tests."""

    return AuthoritativeSource(
        document_id="DOC-001",
        title="Government Service Procedure",
        issuing_authority="Digital Government Authority",
        document_type=DocumentType.PROCEDURE,
        language=SourceLanguage.ENGLISH,
        jurisdiction="Jordan",
        status=status,
        official_source_url=official_source_url,
    )


def make_chunk() -> EvidenceChunk:
    """Create a deterministic evidence chunk."""

    return EvidenceChunk(
        document_id="DOC-001",
        chunk_index=0,
        text="Applicants must submit the required form.",
        language=SourceLanguage.ENGLISH,
        page_start=4,
        page_end=4,
        section_title="Application",
        section_path=(
            "Service Procedure",
            "Application",
        ),
    )


class TestRetrievalFilters(unittest.TestCase):

    def test_default_filters_allow_source(self):
        filters = RetrievalFilters()

        self.assertTrue(
            filters.allows(make_source())
        )

    def test_matching_filters_allow_source(self):
        filters = RetrievalFilters(
            document_ids=("DOC-001",),
            document_types=(
                DocumentType.PROCEDURE,
            ),
            languages=(
                SourceLanguage.ENGLISH,
            ),
            statuses=(
                SourceStatus.CURRENT,
            ),
            issuing_authorities=(
                "Digital Government Authority",
            ),
            jurisdiction="Jordan",
            require_official_source=True,
        )

        self.assertTrue(
            filters.allows(make_source())
        )

    def test_mismatched_status_rejects_source(self):
        filters = RetrievalFilters(
            statuses=(
                SourceStatus.CURRENT,
            )
        )

        self.assertFalse(
            filters.allows(
                make_source(
                    status=SourceStatus.SUPERSEDED
                )
            )
        )

    def test_authority_filter_is_case_insensitive(self):
        filters = RetrievalFilters(
            issuing_authorities=(
                "digital government authority",
            )
        )

        self.assertTrue(
            filters.allows(make_source())
        )

    def test_official_source_requirement_is_enforced(self):
        filters = RetrievalFilters(
            require_official_source=True,
        )

        self.assertFalse(
            filters.allows(
                make_source(
                    official_source_url=""
                )
            )
        )


class TestRetrievalQuery(unittest.TestCase):

    def test_query_text_is_normalized(self):
        query = RetrievalQuery(
            text="  service requirements  ",
            top_k=3,
        )

        self.assertEqual(
            query.text,
            "service requirements",
        )
        self.assertEqual(query.top_k, 3)

    def test_blank_query_is_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalQuery(
                text="   "
            )

    def test_invalid_top_k_is_rejected(self):
        for value in (0, 101):
            with self.subTest(top_k=value):
                with self.assertRaises(ValueError):
                    RetrievalQuery(
                        text="test",
                        top_k=value,
                    )


class TestRetrievalResult(unittest.TestCase):

    def test_result_serializes(self):
        result = RetrievalResult(
            chunk=make_chunk(),
            score=0.91,
            raw_score=8.25,
            rank=1,
            retrieval_method="lexical",
            matched_terms=(
                " applicants ",
                "form",
                "form",
            ),
        )

        data = result.to_dict()

        self.assertEqual(
            data["score"],
            0.91,
        )
        self.assertEqual(
            data["rank"],
            1,
        )
        self.assertEqual(
            data["retrieval_method"],
            "lexical",
        )
        self.assertEqual(
            data["matched_terms"],
            [
                "applicants",
                "form",
            ],
        )

        json.dumps(data)

    def test_invalid_normalized_score_is_rejected(self):
        for score in (-0.1, 1.1):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    RetrievalResult(
                        chunk=make_chunk(),
                        score=score,
                        rank=1,
                        retrieval_method="test",
                    )

    def test_invalid_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalResult(
                chunk=make_chunk(),
                score=0.5,
                rank=0,
                retrieval_method="test",
            )


class TestRetrieverProtocol(unittest.TestCase):

    def test_structural_retriever_contract(self):
        class FakeRetriever:
            def retrieve(
                self,
                query: RetrievalQuery,
            ) -> list[RetrievalResult]:
                return []

        retriever = FakeRetriever()

        self.assertIsInstance(
            retriever,
            Retriever,
        )

        self.assertEqual(
            retriever.retrieve(
                RetrievalQuery(
                    text="test"
                )
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
