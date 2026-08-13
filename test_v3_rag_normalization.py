"""Offline tests for GovBA-GAR ingestion text normalization."""

from __future__ import annotations

import unittest
import unicodedata

from govba.rag.normalization import (
    TEXT_NORMALIZATION_VERSION,
    is_normalized_ingested_text,
    normalize_ingested_text,
)


class TestIngestionTextNormalization(
    unittest.TestCase
):
    def test_normalization_version_is_stable(self):
        self.assertEqual(
            TEXT_NORMALIZATION_VERSION,
            "govba-text-normalization-v1",
        )

    def test_non_string_is_rejected(self):
        for value in (
            None,
            123,
            b"text",
            [],
        ):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    normalize_ingested_text(value)

    def test_windows_and_legacy_line_endings(self):
        text = "First\r\nSecond\rThird"

        self.assertEqual(
            normalize_ingested_text(text),
            "First\nSecond\nThird",
        )

    def test_unicode_line_separators(self):
        text = "First\u2028Second\u2029Third"

        self.assertEqual(
            normalize_ingested_text(text),
            "First\nSecond\nThird",
        )

    def test_non_breaking_spaces_are_normalized(self):
        text = (
            "Public\u00a0service "
            "digital\u202fportal"
        )

        self.assertEqual(
            normalize_ingested_text(text),
            "Public service digital portal",
        )

    def test_horizontal_whitespace_is_collapsed(self):
        text = (
            "  Annual\t\tleave    "
            "request.  "
        )

        self.assertEqual(
            normalize_ingested_text(text),
            "Annual leave request.",
        )

    def test_excess_blank_lines_are_collapsed(self):
        text = (
            "\n\n"
            "First paragraph."
            "\n\n\n\n"
            "Second paragraph."
            "\n\n\n"
        )

        self.assertEqual(
            normalize_ingested_text(text),
            (
                "First paragraph."
                "\n\n"
                "Second paragraph."
            ),
        )

    def test_single_line_break_is_preserved(self):
        text = (
            "First line\n"
            "Second line"
        )

        self.assertEqual(
            normalize_ingested_text(text),
            text,
        )

    def test_arabic_text_is_preserved(self):
        text = (
            "يجب تقديم الطلب خلال "
            "خمسة أيام عمل."
        )

        self.assertEqual(
            normalize_ingested_text(text),
            text,
        )

    def test_arabic_diacritics_are_preserved(self):
        text = "الخِدْمَةُ الحُكُومِيَّةُ"

        result = normalize_ingested_text(text)

        self.assertEqual(
            result,
            unicodedata.normalize(
                "NFC",
                text,
            ),
        )

        self.assertIn(
            "\u0650",
            result,
        )

    def test_case_and_punctuation_are_preserved(self):
        text = (
            "GovBA Policy: "
            "CURRENT / SUPERSEDED."
        )

        self.assertEqual(
            normalize_ingested_text(text),
            text,
        )

    def test_unicode_is_normalized_to_nfc(self):
        decomposed = "Cafe\u0301"

        result = normalize_ingested_text(
            decomposed
        )

        self.assertEqual(
            result,
            unicodedata.normalize(
                "NFC",
                decomposed,
            ),
        )

        self.assertTrue(
            unicodedata.is_normalized(
                "NFC",
                result,
            )
        )

    def test_whitespace_only_becomes_empty(self):
        self.assertEqual(
            normalize_ingested_text(
                " \t\r\n\u00a0 "
            ),
            "",
        )

    def test_normalization_is_idempotent(self):
        text = (
            "  Annual\t leave\r\n"
            "\r\n\r\n"
            "يجب\u00a0تقديم الطلب.  "
        )

        once = normalize_ingested_text(
            text
        )

        twice = normalize_ingested_text(
            once
        )

        self.assertEqual(
            once,
            twice,
        )

    def test_normalized_text_detection(self):
        normalized = (
            "First paragraph."
            "\n\n"
            "Second paragraph."
        )

        self.assertTrue(
            is_normalized_ingested_text(
                normalized
            )
        )

        self.assertFalse(
            is_normalized_ingested_text(
                " First\tparagraph. "
            )
        )


if __name__ == "__main__":
    unittest.main()
