"""Offline tests for GovBA-GAR cross-lingual retrieval."""

from __future__ import annotations

import unittest

from govba.rag.cross_lingual import (
    CROSS_LINGUAL_RETRIEVAL_METHOD,
    CROSS_LINGUAL_RETRIEVAL_VERSION,
    CrossLingualRetriever,
)
from govba.rag.evidence import (
    EvidenceChunk,
)
from govba.rag.language import (
    BilingualLanguage,
    BilingualQueryRequest,
    DetectedLanguage,
    build_language_route,
)
from govba.rag.models import (
    AuthoritativeSource,
    DocumentType,
    SourceLanguage,
    SourceStatus,
)
from govba.rag.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
)


def make_chunk(
    document_id,
    language,
    *,
    index=0,
):
    return EvidenceChunk(
        document_id=document_id,
        chunk_index=index,
        text=f"Evidence {document_id}",
        language=language,
    )


def make_result(
    document_id,
    language,
    *,
    score,
    rank=1,
    index=0,
):
    return RetrievalResult(
        chunk=make_chunk(
            document_id,
            language,
            index=index,
        ),
        score=score,
        raw_score=score,
        rank=rank,
        retrieval_method="stub-v1",
    )


def make_route(
    detected,
    *,
    cross_lingual=True,
):
    response = (
        BilingualLanguage.ARABIC
        if detected
        is DetectedLanguage.ARABIC
        else BilingualLanguage.ENGLISH
    )

    request = BilingualQueryRequest(
        query_text="policy",
        detected_language=detected,
        response_language=response,
        allow_cross_lingual=(
            cross_lingual
        ),
    )

    return build_language_route(
        request
    )


class RecordingRetriever:
    def __init__(
        self,
        results,
    ):
        self.results = tuple(
            results
        )
        self.calls = []

    def retrieve(
        self,
        query,
    ):
        self.calls.append(
            query
        )

        return list(
            self.results
        )


class TestCrossLingualRetrieval(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            CROSS_LINGUAL_RETRIEVAL_VERSION,
            "govba-cross-lingual-retrieval-v1",
        )

    def test_method_name_is_stable(self):
        self.assertEqual(
            CROSS_LINGUAL_RETRIEVAL_METHOD,
            "cross-lingual-v1",
        )

    def test_arabic_route_runs_two_language_lanes(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        retriever.retrieve(
            RetrievalQuery(
                text="سياسة"
            )
        )

        self.assertEqual(
            len(
                underlying.calls
            ),
            2,
        )

        self.assertEqual(
            underlying.calls[
                0
            ].filters.languages,
            (
                SourceLanguage.ARABIC,
                SourceLanguage.BILINGUAL,
            ),
        )

        self.assertEqual(
            underlying.calls[
                1
            ].filters.languages,
            (
                SourceLanguage.ENGLISH,
                SourceLanguage.BILINGUAL,
            ),
        )

    def test_english_route_prioritizes_english_lane(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH
            ),
        )

        retriever.retrieve(
            RetrievalQuery(
                text="policy"
            )
        )

        self.assertEqual(
            underlying.calls[
                0
            ].filters.languages,
            (
                SourceLanguage.ENGLISH,
                SourceLanguage.BILINGUAL,
            ),
        )

    def test_monolingual_route_runs_once(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC,
                cross_lingual=False,
            ),
        )

        retriever.retrieve(
            RetrievalQuery(
                text="سياسة"
            )
        )

        self.assertEqual(
            len(
                underlying.calls
            ),
            1,
        )

    def test_existing_filters_are_preserved(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        query = RetrievalQuery(
            text="policy",
            filters=RetrievalFilters(
                document_ids=(
                    "DOC-A",
                ),
                statuses=(
                    SourceStatus.CURRENT,
                ),
                require_official_source=True,
            ),
        )

        retriever.retrieve(
            query
        )

        for call in underlying.calls:
            self.assertEqual(
                call.filters.document_ids,
                (
                    "DOC-A",
                ),
            )

            self.assertEqual(
                call.filters.statuses,
                (
                    SourceStatus.CURRENT,
                ),
            )

            self.assertTrue(
                call.filters
                .require_official_source
            )

    def test_existing_language_filter_is_not_widened(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        retriever.retrieve(
            RetrievalQuery(
                text="policy",
                filters=RetrievalFilters(
                    languages=(
                        SourceLanguage.ARABIC,
                    )
                ),
            )
        )

        self.assertEqual(
            len(
                underlying.calls
            ),
            1,
        )

        self.assertEqual(
            underlying.calls[
                0
            ].filters.languages,
            (
                SourceLanguage.ARABIC,
            ),
        )

    def test_wrong_language_chunk_is_removed_from_lane(self):
        underlying = RecordingRetriever(
            (
                make_result(
                    "DOC-EN",
                    SourceLanguage.ENGLISH,
                    score=0.9,
                ),
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC,
                cross_lingual=False,
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="سياسة"
            )
        )

        self.assertEqual(
            results,
            [],
        )

    def test_bilingual_chunk_is_allowed_in_arabic_lane(self):
        result = make_result(
            "DOC-BI",
            SourceLanguage.BILINGUAL,
            score=0.9,
        )

        underlying = RecordingRetriever(
            (
                result,
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC,
                cross_lingual=False,
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="سياسة"
            )
        )

        self.assertEqual(
            len(
                results
            ),
            1,
        )

    def test_duplicate_bilingual_chunk_is_deduplicated(self):
        result = make_result(
            "DOC-BI",
            SourceLanguage.BILINGUAL,
            score=0.9,
        )

        underlying = RecordingRetriever(
            (
                result,
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="سياسة"
            )
        )

        self.assertEqual(
            len(
                results
            ),
            1,
        )

    def test_higher_score_wins_across_languages(self):
        arabic = make_result(
            "DOC-AR",
            SourceLanguage.ARABIC,
            score=0.70,
        )

        english = make_result(
            "DOC-EN",
            SourceLanguage.ENGLISH,
            score=0.95,
            index=1,
        )

        underlying = RecordingRetriever(
            (
                english,
                arabic,
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="سياسة",
                top_k=2,
            )
        )

        self.assertEqual(
            tuple(
                result.chunk.document_id
                for result in results
            ),
            (
                "DOC-EN",
                "DOC-AR",
            ),
        )

    def test_primary_language_breaks_equal_score_tie(self):
        arabic = make_result(
            "DOC-AR",
            SourceLanguage.ARABIC,
            score=0.8,
        )

        english = make_result(
            "DOC-EN",
            SourceLanguage.ENGLISH,
            score=0.8,
            index=1,
        )

        underlying = RecordingRetriever(
            (
                english,
                arabic,
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ARABIC
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="سياسة",
                top_k=2,
            )
        )

        self.assertEqual(
            results[
                0
            ].chunk.document_id,
            "DOC-AR",
        )

    def test_final_ranks_are_contiguous(self):
        underlying = RecordingRetriever(
            (
                make_result(
                    "DOC-AR",
                    SourceLanguage.ARABIC,
                    score=0.8,
                ),
                make_result(
                    "DOC-EN",
                    SourceLanguage.ENGLISH,
                    score=0.9,
                    index=1,
                ),
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="policy",
                top_k=2,
            )
        )

        self.assertEqual(
            tuple(
                result.rank
                for result in results
            ),
            (
                1,
                2,
            ),
        )

    def test_final_method_preserves_underlying_method(self):
        underlying = RecordingRetriever(
            (
                make_result(
                    "DOC-EN",
                    SourceLanguage.ENGLISH,
                    score=0.9,
                ),
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH,
                cross_lingual=False,
            ),
        )

        result = retriever.retrieve(
            RetrievalQuery(
                text="policy"
            )
        )[0]

        self.assertEqual(
            result.retrieval_method,
            "cross-lingual-v1:stub-v1",
        )

    def test_top_k_is_respected(self):
        results = tuple(
            make_result(
                f"DOC-{index}",
                SourceLanguage.ENGLISH,
                score=(
                    1.0
                    - index * 0.05
                ),
                index=index,
            )
            for index in range(
                5
            )
        )

        underlying = RecordingRetriever(
            results
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH,
                cross_lingual=False,
            ),
        )

        output = retriever.retrieve(
            RetrievalQuery(
                text="policy",
                top_k=2,
            )
        )

        self.assertEqual(
            len(
                output
            ),
            2,
        )

    def test_candidate_pool_defaults_to_contract_maximum(self):
        underlying = RecordingRetriever(
            ()
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH,
                cross_lingual=False,
            ),
        )

        retriever.retrieve(
            RetrievalQuery(
                text="policy",
                top_k=5,
            )
        )

        self.assertEqual(
            underlying.calls[
                0
            ].top_k,
            100,
        )

    def test_report_is_privacy_safe(self):
        underlying = RecordingRetriever(
            (
                make_result(
                    "DOC-EN",
                    SourceLanguage.ENGLISH,
                    score=0.9,
                ),
            )
        )

        retriever = CrossLingualRetriever(
            retriever=underlying,
            route=make_route(
                DetectedLanguage.ENGLISH,
                cross_lingual=False,
            ),
        )

        report = (
            retriever
            .retrieve_with_report(
                RetrievalQuery(
                    text="Sensitive query"
                )
            )
        )

        data = report.to_dict()

        self.assertNotIn(
            "Sensitive query",
            repr(
                data
            ),
        )

        self.assertNotIn(
            "Evidence DOC-EN",
            repr(
                data
            ),
        )

    def test_invalid_candidate_pool_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            CrossLingualRetriever(
                retriever=RecordingRetriever(
                    ()
                ),
                route=make_route(
                    DetectedLanguage.ENGLISH
                ),
                candidate_pool_size=101,
            )

    def test_invalid_query_is_rejected(self):
        retriever = CrossLingualRetriever(
            retriever=RecordingRetriever(
                ()
            ),
            route=make_route(
                DetectedLanguage.ENGLISH
            ),
        )

        with self.assertRaises(
            TypeError
        ):
            retriever.retrieve(
                object()
            )

    def test_real_lexical_retriever_is_compatible(self):
        from govba.rag.lexical import (
            LexicalRetriever,
        )

        source = AuthoritativeSource(
            document_id="DOC-REAL",
            title="Leave Policy",
            issuing_authority=(
                "Synthetic Government Authority"
            ),
            document_type=(
                DocumentType.POLICY
            ),
            language=(
                SourceLanguage.ENGLISH
            ),
            status=SourceStatus.CURRENT,
            official_source_url=(
                "https://example.gov.jo/"
                "leave.pdf"
            ),
        )

        chunk = EvidenceChunk(
            document_id="DOC-REAL",
            chunk_index=0,
            text="annual leave policy",
            language=(
                SourceLanguage.ENGLISH
            ),
        )

        base = LexicalRetriever(
            sources=(
                source,
            ),
            chunks=(
                chunk,
            ),
        )

        retriever = CrossLingualRetriever(
            retriever=base,
            route=make_route(
                DetectedLanguage.ENGLISH
            ),
        )

        results = retriever.retrieve(
            RetrievalQuery(
                text="annual leave",
                top_k=5,
            )
        )

        self.assertEqual(
            len(
                results
            ),
            1,
        )

        self.assertEqual(
            results[
                0
            ].chunk.document_id,
            "DOC-REAL",
        )


if __name__ == "__main__":
    unittest.main()
