"""Offline tests for GovBA-GAR bilingual language contract."""

from __future__ import annotations

import unittest

from govba.rag.language import (
    BILINGUAL_LANGUAGE_VERSION,
    BilingualLanguage,
    BilingualQueryRequest,
    DetectedLanguage,
    LanguageRoute,
    LanguageRoutingMode,
    build_language_route,
)


class TestBilingualLanguageContract(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            BILINGUAL_LANGUAGE_VERSION,
            "govba-bilingual-language-v1",
        )

    def test_supported_languages_are_arabic_and_english(self):
        self.assertEqual(
            {
                language.value
                for language
                in BilingualLanguage
            },
            {
                "ar",
                "en",
            },
        )

    def test_detected_language_states_are_stable(self):
        self.assertEqual(
            {
                language.value
                for language
                in DetectedLanguage
            },
            {
                "ar",
                "en",
                "mixed",
                "unknown",
            },
        )

    def test_query_text_is_trimmed(self):
        request = BilingualQueryRequest(
            query_text="  Annual leave policy  ",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            request.query_text,
            "Annual leave policy",
        )

    def test_blank_query_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            BilingualQueryRequest(
                query_text="   "
            )

    def test_invalid_detected_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            BilingualQueryRequest(
                query_text="Policy",
                detected_language="en",
            )

    def test_invalid_response_language_is_rejected(self):
        with self.assertRaises(
            TypeError
        ):
            BilingualQueryRequest(
                query_text="Policy",
                response_language="en",
            )

    def test_cross_lingual_flag_must_be_boolean(self):
        with self.assertRaises(
            TypeError
        ):
            BilingualQueryRequest(
                query_text="Policy",
                allow_cross_lingual=1,
            )

    def test_query_hash_is_sha256(self):
        request = BilingualQueryRequest(
            query_text="Policy"
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

    def test_query_id_is_deterministic(self):
        first = BilingualQueryRequest(
            query_text="Annual leave",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        second = BilingualQueryRequest(
            query_text="Annual leave",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            first.query_id,
            second.query_id,
        )

    def test_arabic_query_defaults_response_to_arabic(self):
        request = BilingualQueryRequest(
            query_text="سياسة الإجازات",
            detected_language=(
                DetectedLanguage.ARABIC
            ),
        )

        self.assertEqual(
            request.resolved_response_language,
            BilingualLanguage.ARABIC,
        )

    def test_english_query_defaults_response_to_english(self):
        request = BilingualQueryRequest(
            query_text="Leave policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        self.assertEqual(
            request.resolved_response_language,
            BilingualLanguage.ENGLISH,
        )

    def test_mixed_query_requires_response_resolution(self):
        request = BilingualQueryRequest(
            query_text="سياسة leave",
            detected_language=(
                DetectedLanguage.MIXED
            ),
        )

        self.assertTrue(
            request
            .requires_response_language_resolution
        )

        with self.assertRaises(
            ValueError
        ):
            build_language_route(
                request
            )

    def test_arabic_cross_lingual_route_prioritizes_arabic(self):
        request = BilingualQueryRequest(
            query_text="سياسة الإجازات",
            detected_language=(
                DetectedLanguage.ARABIC
            ),
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.retrieval_languages,
            (
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            ),
        )

        self.assertEqual(
            route.mode,
            LanguageRoutingMode.CROSS_LINGUAL,
        )

    def test_english_cross_lingual_route_prioritizes_english(self):
        request = BilingualQueryRequest(
            query_text="Leave policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.retrieval_languages,
            (
                BilingualLanguage.ENGLISH,
                BilingualLanguage.ARABIC,
            ),
        )

    def test_monolingual_arabic_route(self):
        request = BilingualQueryRequest(
            query_text="سياسة الإجازات",
            detected_language=(
                DetectedLanguage.ARABIC
            ),
            allow_cross_lingual=False,
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.retrieval_languages,
            (
                BilingualLanguage.ARABIC,
            ),
        )

        self.assertFalse(
            route.cross_lingual
        )

    def test_mixed_query_with_explicit_arabic_response_routes_bilingually(self):
        request = BilingualQueryRequest(
            query_text="سياسة leave",
            detected_language=(
                DetectedLanguage.MIXED
            ),
            response_language=(
                BilingualLanguage.ARABIC
            ),
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.response_language,
            BilingualLanguage.ARABIC,
        )

        self.assertEqual(
            route.retrieval_languages,
            (
                BilingualLanguage.ARABIC,
                BilingualLanguage.ENGLISH,
            ),
        )

    def test_mixed_monolingual_route_uses_explicit_response_language(self):
        request = BilingualQueryRequest(
            query_text="سياسة leave",
            detected_language=(
                DetectedLanguage.MIXED
            ),
            response_language=(
                BilingualLanguage.ENGLISH
            ),
            allow_cross_lingual=False,
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.retrieval_languages,
            (
                BilingualLanguage.ENGLISH,
            ),
        )

    def test_route_id_is_deterministic(self):
        request = BilingualQueryRequest(
            query_text="Leave policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        first = build_language_route(
            request
        )

        second = build_language_route(
            request
        )

        self.assertEqual(
            first.route_id,
            second.route_id,
        )

    def test_request_serialization_excludes_raw_query_by_default(self):
        request = BilingualQueryRequest(
            query_text="Sensitive query text",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        data = request.to_dict()

        self.assertNotIn(
            "query_text",
            data,
        )

        self.assertNotIn(
            request.query_text,
            repr(data),
        )

    def test_query_text_can_be_explicitly_serialized(self):
        request = BilingualQueryRequest(
            query_text="Leave policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        data = request.to_dict(
            include_query_text=True
        )

        self.assertEqual(
            data["query_text"],
            "Leave policy",
        )

    def test_route_serialization_contains_no_query_text(self):
        request = BilingualQueryRequest(
            query_text="Leave policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        route = build_language_route(
            request
        )

        data = route.to_dict()

        self.assertEqual(
            data["mode"],
            "cross_lingual",
        )

        self.assertNotIn(
            "query_text",
            data,
        )

    def test_invalid_monolingual_route_is_rejected(self):
        request = BilingualQueryRequest(
            query_text="Policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            LanguageRoute(
                query_id=request.query_id,
                query_language=(
                    DetectedLanguage.ENGLISH
                ),
                response_language=(
                    BilingualLanguage.ENGLISH
                ),
                retrieval_languages=(
                    BilingualLanguage.ENGLISH,
                    BilingualLanguage.ARABIC,
                ),
                mode=(
                    LanguageRoutingMode.MONOLINGUAL
                ),
            )

    def test_invalid_cross_lingual_route_is_rejected(self):
        request = BilingualQueryRequest(
            query_text="Policy",
            detected_language=(
                DetectedLanguage.ENGLISH
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            LanguageRoute(
                query_id=request.query_id,
                query_language=(
                    DetectedLanguage.ENGLISH
                ),
                response_language=(
                    BilingualLanguage.ENGLISH
                ),
                retrieval_languages=(
                    BilingualLanguage.ENGLISH,
                ),
                mode=(
                    LanguageRoutingMode.CROSS_LINGUAL
                ),
            )


if __name__ == "__main__":
    unittest.main()
