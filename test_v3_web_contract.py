"""Offline tests for GovBA-GAR official-web retrieval contract."""

from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from govba.rag.language import (
    BilingualLanguage,
)
from govba.web.contract import (
    OFFICIAL_WEB_RETRIEVAL_VERSION,
    OfficialWebRequest,
    OfficialWebResult,
    OfficialWebRetriever,
)


class FakeOfficialWebRetriever:
    def search(
        self,
        request,
    ):
        return []


class TestOfficialWebRequest(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            OFFICIAL_WEB_RETRIEVAL_VERSION,
            "govba-official-web-retrieval-v1",
        )

    def test_request_is_created(self):
        request = OfficialWebRequest(
            query_text="annual leave policy",
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            request.query_text,
            "annual leave policy",
        )

    def test_query_is_trimmed(self):
        request = OfficialWebRequest(
            query_text="  policy  ",
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            request.query_text,
            "policy",
        )

    def test_blank_query_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            OfficialWebRequest(
                query_text=" ",
                language=(
                    BilingualLanguage.ENGLISH
                ),
            )

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            OfficialWebRequest(
                query_text="policy",
                language="en",
            )

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            OfficialWebRequest(
                query_text="policy",
                language=(
                    BilingualLanguage.ENGLISH
                ),
                top_k=21,
            )

    def test_query_hash_is_sha256(self):
        request = OfficialWebRequest(
            query_text="policy",
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            len(
                request.query_sha256
            ),
            64,
        )

        int(
            request.query_sha256,
            16,
        )

    def test_request_id_is_deterministic(self):
        first = OfficialWebRequest(
            query_text="policy",
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        second = OfficialWebRequest(
            query_text="policy",
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            first.request_id,
            second.request_id,
        )

    def test_default_serialization_excludes_query(self):
        raw = "sensitive search terms"

        request = OfficialWebRequest(
            query_text=raw,
            language=(
                BilingualLanguage.ENGLISH
            ),
        )

        data = request.to_dict()

        self.assertNotIn(
            "query_text",
            data,
        )

        self.assertNotIn(
            raw,
            repr(data),
        )


class TestOfficialWebResult(
    unittest.TestCase
):
    def make_result(
        self,
        **overrides,
    ):
        values = {
            "title": "Synthetic Government Policy",
            "url": (
                "https://example.gov.jo/"
                "policy.pdf"
            ),
            "text": (
                "Synthetic public policy content."
            ),
            "rank": 1,
            "score": 0.9,
            "retrieval_method": (
                "controlled-web-test-v1"
            ),
            "retrieved_at": datetime(
                2026,
                8,
                13,
                tzinfo=timezone.utc,
            ),
        }

        values.update(
            overrides
        )

        return OfficialWebResult(
            **values
        )

    def test_result_is_created(self):
        result = self.make_result()

        self.assertEqual(
            result.domain,
            "example.gov.jo",
        )

    def test_http_url_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.make_result(
                url=(
                    "http://example.gov.jo/"
                    "policy"
                )
            )

    def test_embedded_credentials_are_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.make_result(
                url=(
                    "https://user:password@"
                    "example.gov.jo/policy"
                )
            )

    def test_invalid_rank_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.make_result(
                rank=0
            )

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.make_result(
                score=1.5
            )

    def test_naive_datetime_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.make_result(
                retrieved_at=datetime(
                    2026,
                    8,
                    13,
                )
            )

    def test_content_hash_is_sha256(self):
        result = self.make_result()

        self.assertEqual(
            len(
                result.content_hash
            ),
            64,
        )

    def test_result_id_is_deterministic(self):
        first = self.make_result()
        second = self.make_result()

        self.assertEqual(
            first.result_id,
            second.result_id,
        )

    def test_default_serialization_excludes_text(self):
        raw = (
            "Synthetic public policy content."
        )

        result = self.make_result(
            text=raw
        )

        data = result.to_dict()

        self.assertNotIn(
            "text",
            data,
        )

        self.assertNotIn(
            raw,
            repr(data),
        )

    def test_text_can_be_explicitly_serialized(self):
        result = self.make_result()

        data = result.to_dict(
            include_text=True
        )

        self.assertEqual(
            data["text"],
            result.text,
        )

    def test_url_can_be_excluded(self):
        result = self.make_result()

        data = result.to_dict(
            include_url=False
        )

        self.assertNotIn(
            "url",
            data,
        )

    def test_retriever_protocol_is_structural(self):
        retriever = (
            FakeOfficialWebRetriever()
        )

        self.assertIsInstance(
            retriever,
            OfficialWebRetriever,
        )


if __name__ == "__main__":
    unittest.main()
