"""Offline tests for GovBA-GAR shared configuration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import ai_provider

from govba.core.settings import (
    read_boolean_setting,
    read_setting,
)


TEST_NAME = "GOVBA_TEST_SETTING"


class TestSharedSettings(unittest.TestCase):

    def tearDown(self):
        os.environ.pop(
            TEST_NAME,
            None,
        )

    def test_environment_value_has_priority(self):
        os.environ[
            TEST_NAME
        ] = " environment-value "

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value="streamlit-value",
        ):
            value = read_setting(
                TEST_NAME
            )

        self.assertEqual(
            value,
            "environment-value",
        )

    def test_blank_environment_falls_back_to_streamlit(self):
        os.environ[
            TEST_NAME
        ] = "   "

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value="streamlit-value",
        ):
            value = read_setting(
                TEST_NAME
            )

        self.assertEqual(
            value,
            "streamlit-value",
        )

    def test_streamlit_is_used_when_environment_missing(self):
        os.environ.pop(
            TEST_NAME,
            None,
        )

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value="streamlit-value",
        ):
            value = read_setting(
                TEST_NAME
            )

        self.assertEqual(
            value,
            "streamlit-value",
        )

    def test_default_is_used_when_setting_missing(self):
        os.environ.pop(
            TEST_NAME,
            None,
        )

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value=None,
        ):
            value = read_setting(
                TEST_NAME,
                "default-value",
            )

        self.assertEqual(
            value,
            "default-value",
        )

    def test_blank_streamlit_value_falls_back_to_default(self):
        os.environ.pop(
            TEST_NAME,
            None,
        )

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value=None,
        ):
            value = read_setting(
                TEST_NAME,
                "default-value",
            )

        self.assertEqual(
            value,
            "default-value",
        )

    def test_supported_boolean_true_values(self):
        true_values = (
            "1",
            "true",
            "TRUE",
            "yes",
            "on",
            "enabled",
        )

        for value in true_values:
            with self.subTest(
                value=value
            ):
                os.environ[
                    TEST_NAME
                ] = value

                self.assertTrue(
                    read_boolean_setting(
                        TEST_NAME
                    )
                )

    def test_other_boolean_values_are_false(self):
        false_values = (
            "0",
            "false",
            "no",
            "off",
            "disabled",
            "unexpected",
        )

        for value in false_values:
            with self.subTest(
                value=value
            ):
                os.environ[
                    TEST_NAME
                ] = value

                self.assertFalse(
                    read_boolean_setting(
                        TEST_NAME
                    )
                )

    def test_missing_boolean_uses_default(self):
        os.environ.pop(
            TEST_NAME,
            None,
        )

        with patch(
            "govba.core.settings._read_streamlit_secret",
            return_value=None,
        ):
            self.assertTrue(
                read_boolean_setting(
                    TEST_NAME,
                    default=True,
                )
            )

    def test_ai_provider_preserves_legacy_setting_aliases(self):
        self.assertIs(
            ai_provider._read_setting,
            read_setting,
        )

        self.assertIs(
            ai_provider._read_boolean_setting,
            read_boolean_setting,
        )


if __name__ == "__main__":
    unittest.main()
