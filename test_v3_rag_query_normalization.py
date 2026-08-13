"""Offline tests for GovBA-GAR bilingual query normalization."""

from __future__ import annotations

import unittest

from govba.rag.language import (
    BilingualLanguage,
    DetectedLanguage,
    build_language_route,
)
from govba.rag.query_normalization import (
    QUERY_NORMALIZATION_VERSION,
    QueryNormalizationResult,
    count_query_scripts,
    detect_query_language,
    normalize_query,
    normalize_query_text,
)


class TestQueryNormalization(
    unittest.TestCase
):
    def test_version_is_stable(self):
        self.assertEqual(
            QUERY_NORMALIZATION_VERSION,
            "govba-query-normalization-v1",
        )

    def test_english_is_lowercased(self):
        self.assertEqual(
            normalize_query_text(
                "Annual LEAVE Policy"
            ),
            "annual leave policy",
        )

    def test_whitespace_is_collapsed(self):
        self.assertEqual(
            normalize_query_text(
                "annual   leave\n\tpolicy"
            ),
            "annual leave policy",
        )

    def test_nonbreaking_spaces_are_normalized(self):
        self.assertEqual(
            normalize_query_text(
                "annual\u00a0leave\u202fpolicy"
            ),
            "annual leave policy",
        )

    def test_arabic_diacritics_are_removed(self):
        self.assertEqual(
            normalize_query_text(
                "الإِجَازَات"
            ),
            "الاجازات",
        )

    def test_arabic_tatweel_is_removed(self):
        self.assertEqual(
            normalize_query_text(
                "الإجــازات"
            ),
            "الاجازات",
        )

    def test_arabic_alef_variants_are_normalized(self):
        self.assertEqual(
            normalize_query_text(
                "إجازة أحمد"
            ),
            "اجازة احمد",
        )

    def test_alef_maksura_is_normalized(self):
        self.assertEqual(
            normalize_query_text(
                "إلى"
            ),
            "الي",
        )

    def test_arabic_indic_digits_are_normalized(self):
        self.assertEqual(
            normalize_query_text(
                "المادة ١٢٣"
            ),
            "المادة 123",
        )

    def test_eastern_arabic_digits_are_normalized(self):
        self.assertEqual(
            normalize_query_text(
                "المادة ۱۲۳"
            ),
            "المادة 123",
        )

    def test_arabic_language_is_detected(self):
        self.assertEqual(
            detect_query_language(
                "سياسة الإجازات"
            ),
            DetectedLanguage.ARABIC,
        )

    def test_english_language_is_detected(self):
        self.assertEqual(
            detect_query_language(
                "Annual leave policy"
            ),
            DetectedLanguage.ENGLISH,
        )

    def test_mixed_language_is_detected(self):
        self.assertEqual(
            detect_query_language(
                "سياسة leave"
            ),
            DetectedLanguage.MIXED,
        )

    def test_numbers_only_are_unknown(self):
        self.assertEqual(
            detect_query_language(
                "12345"
            ),
            DetectedLanguage.UNKNOWN,
        )

    def test_script_counts_are_correct(self):
        arabic, latin = (
            count_query_scripts(
                "abc اب"
            )
        )

        self.assertEqual(
            arabic,
            2,
        )

        self.assertEqual(
            latin,
            3,
        )

    def test_normalize_query_returns_result(self):
        result = normalize_query(
            " Annual Leave "
        )

        self.assertIsInstance(
            result,
            QueryNormalizationResult,
        )

        self.assertEqual(
            result.normalized_text,
            "annual leave",
        )

        self.assertEqual(
            result.detected_language,
            DetectedLanguage.ENGLISH,
        )

    def test_normalization_id_is_sha256(self):
        result = normalize_query(
            "Annual leave"
        )

        self.assertEqual(
            len(
                result.normalization_id
            ),
            64,
        )

        int(
            result.normalization_id,
            16,
        )

    def test_normalization_is_deterministic(self):
        first = normalize_query(
            "إجازة سنوية"
        )

        second = normalize_query(
            "إجازة سنوية"
        )

        self.assertEqual(
            first.normalization_id,
            second.normalization_id,
        )

        self.assertEqual(
            first.normalized_text,
            second.normalized_text,
        )

    def test_original_and_normalized_hashes_are_present(self):
        result = normalize_query(
            " Annual Leave "
        )

        self.assertEqual(
            len(
                result.original_query_sha256
            ),
            64,
        )

        self.assertEqual(
            len(
                result.normalized_query_sha256
            ),
            64,
        )

    def test_default_serialization_excludes_query_text(self):
        result = normalize_query(
            "Sensitive leave request"
        )

        data = result.to_dict()

        self.assertNotIn(
            "normalized_text",
            data,
        )

        self.assertNotIn(
            result.normalized_text,
            repr(data),
        )

    def test_normalized_text_can_be_explicitly_serialized(self):
        result = normalize_query(
            "Annual Leave"
        )

        data = result.to_dict(
            include_normalized_text=True
        )

        self.assertEqual(
            data["normalized_text"],
            "annual leave",
        )

    def test_arabic_result_converts_to_bilingual_request(self):
        result = normalize_query(
            "سياسة الإجازات"
        )

        request = (
            result.to_bilingual_request()
        )

        self.assertEqual(
            request.detected_language,
            DetectedLanguage.ARABIC,
        )

        self.assertEqual(
            request.resolved_response_language,
            BilingualLanguage.ARABIC,
        )

    def test_english_result_builds_cross_lingual_route(self):
        result = normalize_query(
            "Annual leave policy"
        )

        request = (
            result.to_bilingual_request()
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

    def test_mixed_query_still_requires_explicit_response_language(self):
        result = normalize_query(
            "سياسة leave"
        )

        request = (
            result.to_bilingual_request()
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

    def test_mixed_query_can_be_resolved_explicitly(self):
        result = normalize_query(
            "سياسة leave"
        )

        request = (
            result.to_bilingual_request(
                response_language=(
                    BilingualLanguage.ARABIC
                )
            )
        )

        route = build_language_route(
            request
        )

        self.assertEqual(
            route.response_language,
            BilingualLanguage.ARABIC,
        )

    def test_blank_query_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            normalize_query(
                "   "
            )


if __name__ == "__main__":
    unittest.main()
